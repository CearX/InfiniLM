use super::{GGufModel, build_sin_cos};
use crate::utils::meta;
use ggus::GGufMetaMapExt;
use nn::{
    Activation, Attention, Linear, Merger, Mlp, NormType, Normalization, PatchEmbd, Qwen2VLmmproj,
    RoPE, Tensor, TransformerBlk,
};
use std::cmp::max;

impl GGufModel<'_> {
    /// 构造 qw2vl_mmproj 模型
    #[allow(dead_code)]
    pub fn qw2vl_mmproj(&self, nctx: usize) -> nn::Qwen2VLmmproj<Tensor<&[u8], 2>> {
        let nblk = meta![self => llm_block_count; 32];
        let d = meta![self => llm_embedding_length;1280];
        let nh = meta![self => llm_attention_head_count;16];
        let nkvh = meta![self => llm_attention_head_count_kv; nh];
        let dh = meta![self => llm_rope_dimension_count; d / nh];
        let _di = meta![self => llm_feed_forward_length; 5120];
        let epsilon = meta![self => llm_attention_layer_norm_epsilon; 1e-6];
        let d_patch = 14; // todo: ggus
        let d_proj = 1536;
        let dt = self.tensors["v.blk.0.attn_qkv.weight"].dt();
        let dt_norm = self.tensors["v.blk.0.ln1.weight"].dt();

        let get = |name: &str| self.tensors[name].as_deref();

        Qwen2VLmmproj {
            patch_embd: PatchEmbd {
                dt,
                shape: [d, 3, d_patch, d_patch],
                patch_embd: get("v.patch_embd.weight"),
                patch_embd1: get("v.patch_embd.weight.1"),
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
                                Some((dt, get(&format!("v.blk.{iblk}.attn_qkv.bias")))),
                            ),
                            q_norm: None,
                            k_norm: None,
                            rope: Some(RoPE {
                                multimodal: nn::MRoPE::MRoPE2D,
                                nctx,
                                sin: get("sin_table"),
                                cos: get("cos_table"),
                            }),
                            output: Linear::new(
                                dt,
                                [d, nh * dh],
                                get(&format!("v.blk.{iblk}.attn_out.weight")),
                                Some((dt, get(&format!("v.blk.{iblk}.attn_out.bias")))),
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
                                Some((dt, get(&format!("v.blk.{iblk}.ffn_up.bias")))),
                            ),
                            act: Activation::GeLU,
                            down: Linear::new(
                                dt,
                                [d, d * 4],
                                get(&format!("v.blk.{iblk}.ffn_down.weight")),
                                Some((dt, get(&format!("v.blk.{iblk}.ffn_down.bias")))),
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
                        scale: get("v.post_ln.weight"),
                        dt_bias: dt_norm,
                        bias: get("v.post_ln.bias"),
                    },
                },
                mlp: Mlp {
                    up: Linear::new(
                        dt,
                        [d * 4, d * 4],
                        get("mm.0.weight"),
                        Some((dt, get("mm.0.bias"))),
                    ),
                    act: Activation::GeLU,
                    down: Linear::new(
                        dt,
                        [d_proj, d * 4],
                        get("mm.2.weight"),
                        Some((dt, get("mm.2.bias"))),
                    ),
                },
            },
        }
    }

    /// 插入用于 MRoPE 的 sin cos 表张量
    #[allow(dead_code)]
    pub(crate) fn insert_sin_cos_qw2vl_mmproj(&mut self, nctx: usize) {
        let d = meta![self => llm_embedding_length; 1280];
        let nh = meta![self => llm_attention_head_count; 16];
        let dh = meta![self => llm_rope_dimension_count; d / nh]; // todo: ggus
        let dh_div_2 = dh / 2; // h, w 维度均分 dh_div_2
        let theta = meta![self => llm_rope_freq_base; 1e4];
        let [sin, cos] = build_sin_cos(nctx, dh_div_2, theta, |pos, _| pos as _);
        self.tensors.insert("sin_table", sin);
        self.tensors.insert("cos_table", cos);
    }
}

/// 构造 pos_ids 表
#[allow(dead_code)]
pub(crate) fn build_pos_ids(h: usize, w: usize, d_patch: usize) -> Vec<u32> {
    let hp = h / d_patch;
    let wp = w / d_patch;
    let mut pos = vec![0; hp * wp * 2];

    let mut ptr = 0;
    for y in (0..hp).step_by(2) {
        for x in (0..wp).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    pos[ptr * 2] = (y + dy) as u32;
                    pos[ptr * 2 + 1] = (x + dx) as u32;
                    ptr += 1;
                }
            }
        }
    }

    pos
}

/// # 构造 3D RoPE 的 pos_ids 表
///
/// 对于视觉和文本嵌入序列，我们为视觉部分计算 3D 旋转位置嵌入，为文本部分计算 1D 旋转位置嵌入。
///
/// ## 示例
///
/// 假设我们有一个视频输入，包含 3 个时序 patch，2 个高度 patch 和 2 个宽度 patch。
///
/// **输入序列：**
/// - `input_ids` = `[T T T T V V V V V V V V V V V V T T T T T]`
/// - 其中 `V` 代表视觉 patch，`T` 代表文本 patch
///
/// **文本部分 position_ids（图像前）：**
/// - `text temporal position_ids`: `[0, 1, 2, 3]`
/// - `text height position_ids`: `[0, 1, 2, 3]`
/// - `text width position_ids`: `[0, 1, 2, 3]`
///
/// **视觉部分 position_ids：**
/// - `vision temporal position_ids`: `[4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6]`
/// - `vision height position_ids`: `[4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5]`
/// - `vision width position_ids`: `[4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5]`
///
/// **文本部分 position_ids（图像后）：**
/// - `text temporal position_ids`: `[7, 8, 9, 10, 11]`
/// - `text height position_ids`: `[7, 8, 9, 10, 11]`
/// - `text width position_ids`: `[7, 8, 9, 10, 11]`
///
/// **计算规则：**
/// - 图像起始 `position_ids` 计算为图像前文本最大 `position_ids` 加 1
/// - 图像后文本起始 `position_ids` 计算为最大视觉 `position_ids` 加 1
#[allow(dead_code)]
pub fn build_3d_pos_ids(
    t: usize,
    h: usize,
    w: usize,
    d_patch: usize,
    pre_text_len: usize,
    post_text_len: usize,
) -> Vec<u32> {
    let spatial_merge_size = 2;
    let t_len = t;
    let h_len = h / d_patch / spatial_merge_size;
    let w_len = w / d_patch / spatial_merge_size;
    let vision_len = t_len * h_len * w_len;
    let total_len = pre_text_len + vision_len + post_text_len;

    let mut pos = vec![0u32; total_len * 3];
    let mut idx = 0;

    // 图像前文本
    for i in 0..pre_text_len as u32 {
        pos[idx * 3] = i;
        pos[idx * 3 + 1] = i;
        pos[idx * 3 + 2] = i;
        idx += 1;
    }

    // 图像
    let img_start_pos = pre_text_len as u32;
    for t in 0..t_len as u32 {
        for h in 0..h_len as u32 {
            for w in 0..w_len as u32 {
                let t_pos = img_start_pos + t;
                let h_pos = img_start_pos + h;
                let w_pos = img_start_pos + w;
                pos[idx * 3] = t_pos;
                pos[idx * 3 + 1] = h_pos;
                pos[idx * 3 + 2] = w_pos;
                idx += 1;
            }
        }
    }

    // 图像后文本
    let t_max_pos = img_start_pos + t_len as u32 - 1;
    let h_max_pos = img_start_pos + h_len as u32 - 1;
    let w_max_pos = img_start_pos + w_len as u32 - 1;
    let image_max_pos = max(t_max_pos, max(h_max_pos, w_max_pos));
    let text_start_pos = image_max_pos + 1;
    for i in 0..post_text_len as u32 {
        let pos_val = text_start_pos + i;
        pos[idx * 3] = pos_val;
        pos[idx * 3 + 1] = pos_val;
        pos[idx * 3 + 2] = pos_val;
        idx += 1;
    }

    assert_eq!(idx, total_len);
    pos
}
