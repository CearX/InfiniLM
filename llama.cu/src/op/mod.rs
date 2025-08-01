mod add;
mod add4d;
#[cfg(nccl)]
mod all_reduce;
mod embedding;
mod embedding_qw2vl;
mod fast_embedding;
mod gelu;
mod layer_norm;
mod linear;
mod mrope;
mod mrope_3d;
mod rms_norm;
mod rope;
mod silu;
mod swiglu;

use crate::handle::Handle;
use nn::{
    Tensor,
    digit_layout::{DigitLayout, types},
};
use operators::cuda::{Stream, VirByte};

pub mod random_sample;

pub use add::Add;
pub use add4d::Add4d;
#[cfg(nccl)]
pub use all_reduce::AllReduce;
pub use embedding::Embedding;
pub use embedding_qw2vl::EmbeddingQw2vl;
pub use fast_embedding::FastEmbedding;
pub use gelu::Gelu;
pub use layer_norm::LayerNorm;
pub use linear::Linear;
pub use mrope::MRope;
pub use mrope_3d::MRope3d;
pub use rms_norm::RmsNorm;
pub use rope::Rope;
pub use silu::Silu;
pub use swiglu::Swiglu;

pub trait Operator {
    fn launch<const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    );
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModuleKey {
    Text(&'static str),
    Type(DigitLayout),
    Size(usize),
}

fn cuda_type(ty: DigitLayout) -> &'static str {
    match ty {
        types::U32 => "unsigned int",
        types::F32 => "float",
        types::F16 => "half",
        _ => todo!(),
    }
}

fn move_type(unit: usize) -> &'static str {
    match unit {
        1 => "char",
        2 => "short",
        4 => "float",
        8 => "float2",
        16 => "float4",
        32 => "double4",
        _ => todo!(),
    }
}

// 最大公约数
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}
