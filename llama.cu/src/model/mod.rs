mod chat_template;
mod gguf;

use crate::utils::{Blob, Data, meta};
use ggus::{GGufFileName, GGufMetaMapExt};
use memmap2::Mmap;
use nn::{
    Activation, Attention, Embedding, LLaMA, Linear, Merger, Mlp, NormType, Normalization,
    OutputHead, PatchEmbd, Qwen2VLmmproj, RoPE, Table, Tensor, TransformerBlk, digit_layout::types,
};
use std::{fmt::Debug, fs::File, path::Path};

pub(crate) use chat_template::ChatTemplate;
pub(crate) use gguf::GGufModel;

pub use chat_template::Message;

/// 从指定文件的路径出发，映射所有分片文件。
pub(crate) fn map_files(path: impl AsRef<Path>) -> Box<[Mmap]> {
    fn throw(path: &Path, e: impl Debug) -> ! {
        let path = path.display();
        panic!(
            "\
Error occurred at path: {path}
  error: {e:?}"
        )
    }

    #[inline]
    fn map_file(path: &Path) -> Mmap {
        let file = File::open(path).unwrap_or_else(|e| throw(path, e));
        unsafe { Mmap::map(&file) }.unwrap()
    }

    let path = path.as_ref();
    let name = GGufFileName::try_from(path).unwrap_or_else(|e| throw(path, e));

    if name.shard_count() == 1 {
        Box::new([map_file(path)])
    } else {
        let dir = path.parent().unwrap();
        name.iter_all()
            .map(|name| map_file(&dir.join(name.to_string())))
            .collect()
    }
}

impl GGufModel<'_> {
    pub fn insert_sin_cos(&mut self) {
        let nctx = meta![self => llm_context_length];
        let d = meta![self => llm_embedding_length];
        let nh = meta![self => llm_attention_head_count];
        let dh = meta![self => llm_rope_dimension_count; d / nh];
        let theta = meta![self => llm_rope_freq_base; 1e4];

        let [sin, cos] = build_sin_cos(nctx, dh, theta);
        self.tensors.insert("sin_table", sin);
        self.tensors.insert("cos_table", cos);
    }

    pub fn llama(&self) -> nn::LLaMA<Tensor<&[u8], 2>> {
        let arch = meta![self => general_architecture];
        let dt_bias = match arch {
            "llama" => None,
            "qwen2" => Some(self.tensors["blk.0.attn_qkv.bias"].dt()),
            arch => panic!("unsupported arch {arch}"),
        };

        let nvoc = meta![self => tokenizer_ggml_tokens].len();
        let nctx = meta![self => llm_context_length];
        let nblk = meta![self => llm_block_count];
        let d = meta![self => llm_embedding_length];
        let nh = meta![self => llm_attention_head_count];
        let nkvh = meta![self => llm_attention_head_count_kv; nh];
        let dh = meta![self => llm_rope_dimension_count; d / nh];
        let di = meta![self => llm_feed_forward_length];
        let epsilon = meta![self => llm_attention_layer_norm_rms_epsilon; 1e-5];
        let dt_linear = self.tensors["blk.0.attn_qkv.weight"].dt();

        let get = |name: &str| self.tensors[name].as_deref();

        let token_embd = get("token_embd.weight");
        let out_norm = get("output_norm.weight");
        let out_linear = if self.tensors.contains_key("output.weight") {
            get("output.weight")
        } else {
            token_embd.clone()
        };

        LLaMA {
            embedding: Embedding {
                dt: token_embd.dt(),
                d,
                wte: Table {
                    row: nvoc,
                    weight: token_embd,
                },
                wpe: None,
            },
            blks: (0..nblk)
                .map(|iblk| {
                    TransformerBlk::new(
                        Normalization {
                            d,
                            epsilon: epsilon as _,
                            items: NormType::RmsNorm {
                                dt: out_norm.dt(),
                                scale: get(&format!("blk.{iblk}.attn_norm.weight")),
                            },
                        },
                        Attention {
                            nh,
                            nkvh,
                            qkv: Linear::new(
                                dt_linear,
                                [(nh + nkvh + nkvh) * dh, d],
                                get(&format!("blk.{iblk}.attn_qkv.weight")),
                                dt_bias.map(|dt| (dt, get(&format!("blk.{iblk}.attn_qkv.bias")))),
                            ),
                            rope: Some(RoPE {
                                multimodal: false,
                                nctx,
                                sin: get("sin_table"),
                                cos: get("cos_table"),
                            }),
                            output: Linear::new(
                                dt_linear,
                                [d, nh * dh],
                                get(&format!("blk.{iblk}.attn_output.weight")),
                                None,
                            ),
                        },
                        Normalization {
                            d,
                            epsilon: epsilon as _,
                            items: NormType::RmsNorm {
                                dt: out_norm.dt(),
                                scale: get(&format!("blk.{iblk}.ffn_norm.weight")),
                            },
                        },
                        Mlp {
                            up: Linear::new(
                                dt_linear,
                                [di * 2, d],
                                get(&format!("blk.{iblk}.ffn_gate_up.weight")),
                                None,
                            ),
                            act: Activation::SwiGLU,
                            down: Linear::new(
                                dt_linear,
                                [d, di],
                                get(&format!("blk.{iblk}.ffn_down.weight")),
                                None,
                            ),
                        },
                    )
                })
                .collect(),
            output_head: Some(OutputHead {
                out_norm: Normalization {
                    d,
                    epsilon: epsilon as _,
                    items: NormType::RmsNorm {
                        dt: out_norm.dt(),
                        scale: out_norm,
                    },
                },
                lm_head: Linear::new(out_linear.dt(), [nvoc, d], out_linear, None),
            }),
        }
    }

    /// 构造 kv cache 张量
    pub fn kv_cache<const N: usize>(&self) -> Tensor<usize, N> {
        let dt = self.tensors["token_embd.weight"].dt();
        let nblk = meta![self => llm_block_count];
        let nctx = meta![self => llm_context_length];
        let d = meta![self => llm_embedding_length];
        let nh = meta![self => llm_attention_head_count];
        let nkvh = meta![self => llm_attention_head_count_kv; nh];
        let dh = meta![self => llm_rope_dimension_count; d / nh];
        Tensor::from_dim_slice(dt, [nctx, nblk, 2, nkvh, dh])
    }

    #[allow(dead_code)]
    pub fn qw2vl_mmproj(&self) -> nn::Qwen2VLmmproj<Tensor<&[u8], 2>> {
        let nblk = meta![self => llm_block_count];
        let d = meta![self => llm_embedding_length];
        let nh = meta![self => llm_attention_head_count];
        let nkvh = meta![self => llm_attention_head_count_kv; nh];
        let dh = meta![self => llm_rope_dimension_count; d / nh];
        let _di = meta![self => llm_feed_forward_length];
        let epsilon = meta![self => llm_attention_layer_norm_epsilon; 1e-6];
        let d_patch = 14; // ggus todo
        let d_proj = 1536;
        let dt = self.tensors["v.blk.0.attn_qkv.weight"].dt();
        let dt_norm = self.tensors["v.blk.0.ln1.weight"].dt();

        let get = |name: &str| self.tensors[name].as_deref();

        Qwen2VLmmproj {
            patch_embd: PatchEmbd {
                dt,
                shape: [d, 3, d_patch, d_patch],
                patch_embd: get(&format!("v.patch_embd.weight")),
                patch_embd1: get(&format!("v.patch_embd.weight.1")),
            },
            vision_blks: (0..nblk)
                .map(|iblk| {
                    TransformerBlk::new(
                        Normalization {
                            d,
                            epsilon: epsilon as _,
                            items: NormType::LayerNorm {
                                dt_scale: dt_norm,
                                scale: get(&format!("v.blk.{iblk}.ln1.weight")),
                                dt_bias: dt_norm,
                                bias: get(&format!("v.blk.{iblk}.ln1.bias")),
                            },
                        },
                        Attention {
                            nh,
                            nkvh,
                            qkv: Linear::new(
                                dt,
                                [(nh + nkvh + nkvh) * dh, d],
                                get(&format!("v.blk.{iblk}.attn_qkv.weight")),
                                Some((dt_norm, get(&format!("v.blk.{iblk}.attn_qkv.bias")))),
                            ),
                            rope: Some(RoPE {
                                nctx: 34, // mrope todo
                                sin: get("sin_table"),
                                cos: get("cos_table"),
                            }),
                            output: Linear::new(
                                dt,
                                [d, nh * dh],
                                get(&format!("v.blk.{iblk}.attn_out.weight")),
                                Some((dt_norm, get(&format!("v.blk.{iblk}.attn_out.bias")))),
                            ),
                        },
                        Normalization {
                            d,
                            epsilon: epsilon as _,
                            items: NormType::LayerNorm {
                                dt_scale: dt_norm,
                                scale: get(&format!("v.blk.{iblk}.ln2.weight")),
                                dt_bias: dt_norm,
                                bias: get(&format!("v.blk.{iblk}.ln2.bias")),
                            },
                        },
                        Mlp {
                            up: Linear::new(
                                dt,
                                [d * 4, d],
                                get(&format!("v.blk.{iblk}.ffn_up.weight")),
                                Some((dt_norm, get(&format!("v.blk.{iblk}.ffn_up.bias")))),
                            ),
                            act: Activation::GeLU,
                            down: Linear::new(
                                dt,
                                [d, d * 4],
                                get(&format!("v.blk.{iblk}.ffn_down.weight")),
                                Some((dt_norm, get(&format!("v.blk.{iblk}.ffn_down.bias")))),
                            ),
                        },
                    )
                })
                .collect(),
            merger: Merger {
                post_norm: Normalization {
                    d,
                    epsilon: epsilon as _,
                    items: NormType::LayerNorm {
                        dt_scale: dt_norm,
                        scale: get(&format!("v.post_ln.weight")),
                        dt_bias: dt_norm,
                        bias: get(&format!("v.post_ln.bias")),
                    },
                },
                mlp: Mlp {
                    up: Linear::new(
                        dt,
                        [d * 4, d * 4],
                        get(&format!("mm.0.weight")),
                        Some((dt_norm, get(&format!("mm.0.bias")))),
                    ),
                    act: Activation::GeLU,
                    down: Linear::new(
                        dt,
                        [d_proj, d * 4],
                        get(&format!("mm.2.weight")),
                        Some((dt_norm, get(&format!("mm.2.bias")))),
                    ),
                },
            },
        }
    }
}

/// 构造 sin cos 表张量
fn build_sin_cos<'a, const N: usize>(
    nctx: usize,
    dh: usize,
    theta: f32,
) -> [Tensor<Data<'a>, N>; 2] {
    let ty = types::F32;
    let mut sin = Blob::new(nctx * dh / 2 * ty.nbytes());
    let mut cos = Blob::new(nctx * dh / 2 * ty.nbytes());

    {
        let ([], sin, []) = (unsafe { sin.align_to_mut::<f32>() }) else {
            unreachable!()
        };
        let ([], cos, []) = (unsafe { cos.align_to_mut::<f32>() }) else {
            unreachable!()
        };
        for pos in 0..nctx {
            for i in 0..dh / 2 {
                let theta = theta.powf(-((2 * i) as f32 / dh as f32));
                let freq = pos as f32 * theta;
                let (sin_, cos_) = freq.sin_cos();
                sin[pos * dh / 2 + i] = sin_;
                cos[pos * dh / 2 + i] = cos_;
            }
        }
    }

    let tensor = |data: Blob| {
        Tensor::from_dim_slice(ty, [nctx, dh / 2]).map(|len| {
            assert_eq!(len, data.len());
            data.into()
        })
    };
    [tensor(sin), tensor(cos)]
}
