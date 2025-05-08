use gguf::{ggml_quants::digit_layout::DigitLayout, tensor, GGufMetaMapExt, GGufModel};
use tensor::Tensor;

#[derive(Clone, Debug)]
pub struct Meta {
    pub d: usize,     // 1280
    pub d_img: usize, // 1536
}

impl Meta {
    pub fn from_gguf(gguf: &GGufModel) -> Self {
        Self {
            d: gguf::meta![gguf => (usize) "clip.vision.embedding_length"],
            d_img: gguf::meta![gguf => (usize) "clip.vision.projection_dim"],
        }
    }

    #[inline]
    pub fn img_embd(&self, dt: DigitLayout, batch: usize) -> Tensor<usize> {
        Tensor::new(dt, &[batch, 816, self.d_img]) // 不需要, img_embd = [np, d_img]
    }
}

#[derive(Clone)]
pub struct Storage<T> {
    pub mm_0: [T; 2],
    pub mm_2: [T; 2],
}

impl<'a> Storage<&'a [u8]> {
    #[rustfmt::skip]
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        Self {
            mm_0   : [tensor![gguf => "mm.0.weight"    ].data ,
                      tensor![gguf => "mm.0.bias"      ].data],
            mm_2  : [tensor![gguf  => "mm.2.weight"   ].data ,
                      tensor![gguf => "mm.2.bias"     ].data],
        }
    }
}
