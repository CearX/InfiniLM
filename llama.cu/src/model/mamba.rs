use super::GGufModel;
use crate::utils::meta;
use ggus::GGufMetaMapExt;
use nn::Tensor;

impl GGufModel<'_> {
    /// 构造 mamba 模型
    pub fn mamba(&self) -> nn::Mamba<Tensor<&[u8], 2>> {
        let nvoc = meta![self => tokenizer_ggml_tokens].len();
        let nblk = meta![self => llm_block_count];
        let d = meta![self => llm_embedding_length];
        let epsilon = meta![self => llm_attention_layer_norm_rms_epsilon; 1e-5];
        let dt_embd = self.tensors["token_embd.weight"].dt();
        let dt_norm = self.tensors["output_norm.weight"].dt();
        let dt_linear = self.tensors["blk.0.ssm_in.weight"].dt();

        let d_kernel = 4;
        let d_inner = 5120;
        let d_state = 16;
        let dt_rank = 160; // ggus todo: mamba.ssm.

        let get = |name: &str| self.tensors[name].as_deref();

        ::nn::Mamba {
            embedding: ::nn::Embedding {
                dt: dt_embd,
                d,
                wte: ::nn::Table {
                    row: nvoc,
                    weight: get("token_embd.weight"),
                },
                wpe: None,
            },
            blks: (0..nblk)
                .map(|iblk| ::nn::MambaBlock {
                    mamba_norm: nn::Normalization {
                        d,
                        epsilon: epsilon as _,
                        items: ::nn::NormType::RmsNorm {
                            dt: dt_norm,
                            scale: get(&format!("blk.{iblk}.attn_norm.weight")),
                        },
                    },
                    mamba_mixer: nn::MambaMixer {
                        d_inner,
                        in_proj: nn::Linear {
                            dt: dt_linear,
                            shape: [d_inner * 2, d],
                            weight: get(&format!("blk.{iblk}.ssm_in.weight")),
                            bias: None,
                            allow_residual: false,
                        },
                        causal_conv1d: nn::CausalConv1d::new(
                            dt_norm,
                            get(&format!("blk.{iblk}.ssm_conv1d.weight")),
                            get(&format!("blk.{iblk}.ssm_conv1d.bias")),
                            d_kernel,
                            d_inner,
                        ),
                        act: nn::Activation::SiLU,
                        selective_ssm: nn::SelectiveSSM {
                            dt: dt_norm,
                            d_state,
                            dt_rank,
                            dt_proj: nn::Linear {
                                dt: dt_linear,
                                shape: [d_inner, dt_rank],
                                weight: get(&format!("blk.{iblk}.ssm_dt.weight")),
                                bias: Some(dt_linear)
                                    .map(|dt| (dt, get(&format!("blk.{iblk}.ssm_dt.bias")))),
                                allow_residual: false,
                            },
                            x_proj: nn::Linear {
                                dt: dt_linear,
                                shape: [dt_rank + d_state * 2, d_inner],
                                weight: get(&format!("blk.{iblk}.ssm_x.weight")),
                                bias: None,
                                allow_residual: false,
                            },
                            a: get(&format!("blk.{iblk}.ssm_a")),
                            d: get(&format!("blk.{iblk}.ssm_d")),
                        },
                        out_proj: nn::Linear {
                            dt: dt_linear,
                            shape: [d, d_inner],
                            weight: get(&format!("blk.{iblk}.ssm_out.weight")),
                            bias: None,
                            allow_residual: true,
                        },
                    },
                })
                .collect(),
            output_head: Some(::nn::OutputHead {
                out_norm: ::nn::Normalization {
                    d,
                    epsilon: epsilon as _,
                    items: ::nn::NormType::RmsNorm {
                        dt: dt_norm,
                        scale: get("output_norm.weight"),
                    },
                },
                lm_head: {
                    let out_linear = if self.tensors.contains_key("output.weight") {
                        get("output.weight")
                    } else {
                        get("token_embd.weight")
                    };
                    ::nn::Linear::new(dt_embd, [nvoc, d], out_linear, None)
                },
            }),
        }
    }
}
