mod args;
mod compute;
mod image;
mod storage;

use gguf::ggml_quants::digit_layout::DigitLayout;

pub use args::Args as ClipArgs;
pub use compute::{BlkWeight, ClipWorker, Operators, WeightLoader};
pub use image::{Image, ImageGrid};
pub use storage::{BlkStorage as ClipBlkStorage, Storage as ClipStorage};
pub use tensor::Tensor;
pub mod ext {
    pub use gguf::{
        ext::{utok, Mmap},
        ggml_quants,
    };
}

#[derive(Clone, Debug)]
pub struct ClipMeta {
    pub projector: ProjectorType,
    pub minicpmv_version: u8,

    pub dt: DigitLayout,

    pub d_patch: usize,
    pub d_image: usize,

    pub nblk: usize,
    pub nh: usize,
    pub nkvh: usize,
    pub d: usize,
    pub dh: usize,
    pub di: usize,

    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub epsilon: f32,
}

pub const D_POS_EMBD: usize = 70;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ProjectorType {
    Mlp,
    MlpNorm,
    Ldp,
    LdpV2,
    Resampler,
    Unknown,
}

impl ClipMeta {
    pub fn n_patch(&self) -> usize {
        let &Self {
            d_image, d_patch, ..
        } = self;
        let n_patch = (d_image / d_patch).pow(2);
        match self.projector {
            ProjectorType::Resampler => match self.minicpmv_version {
                2 => 96,
                3 => 64,
                _ => n_patch,
            },
            ProjectorType::Ldp | ProjectorType::LdpV2 => n_patch / 4,
            _ => n_patch,
        }
    }

    pub fn n_mmproj_embd(&self) -> usize {
        match self.projector {
            ProjectorType::Resampler => match self.minicpmv_version {
                2 => 4096,
                3 => 3584,
                _ => unreachable!(),
            },
            _ => todo!(),
        }
    }

    pub fn embd(&self, np: usize) -> Tensor<usize> {
        let &Self { dt, d, .. } = self;
        Tensor::new(dt, &[np, d])
    }

    pub fn pos_embd(&self) -> Tensor<usize> {
        let &Self { dt, d, .. } = self;
        Tensor::new(dt, &[D_POS_EMBD.pow(2), d])
    }

    pub fn patch_embd_w(&self) -> Tensor<usize> {
        let &Self { d, d_patch, .. } = self;
        Tensor::new(self.dt, &[d, 3, d_patch, d_patch])
    }

    pub fn patch_embd_b(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        Tensor::new(self.dt, &[d])
    }

    pub fn norm(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        Tensor::new(self.dt, &[d])
    }

    pub fn attn_qkv_w(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(3 * d, d)
    }

    pub fn attn_qkv_b(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(3 * d, 1)
    }

    pub fn attn_o_w(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(d, d)
    }

    pub fn attn_o_b(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(d, 1)
    }

    pub fn ffn_up_w(&self) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(di, d)
    }

    pub fn ffn_up_b(&self) -> Tensor<usize> {
        let &Self { di, .. } = self;
        self.mat(di, 1)
    }

    pub fn ffn_down_w(&self) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(d, di)
    }

    pub fn ffn_down_b(&self) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(d, 1)
    }

    fn mat(&self, row: usize, col: usize) -> Tensor<usize> {
        assert_eq!(self.dt.group_size(), 1);
        Tensor::new(self.dt, &[row, col]).transpose(&[1, 0])
    }

    // resampler
    pub fn r_q(&self) -> Tensor<usize> {
        self.mat(3584, 64)
    }

    pub fn r_norm(&self) -> Tensor<usize> {
        Tensor::new(self.dt, &[3584])
    }

    pub fn r_kv_w(&self) -> Tensor<usize> {
        Tensor::new(self.dt, &[self.d, 3584])
    }

    pub fn r_pos_k(&self) -> Tensor<usize> {
        self.mat(3584, 4900)
    }

    pub fn r_attn_qkv_w(&self) -> Tensor<usize> {
        Tensor::new(self.dt, &[3584, 10752])
    }

    pub fn r_attn_qkv_b(&self) -> Tensor<usize> {
        self.mat(3584, 3)
    }

    pub fn r_attn_o_w(&self) -> Tensor<usize> {
        Tensor::new(self.dt, &[3584, 3584])
    }

    pub fn r_attn_o_b(&self) -> Tensor<usize> {
        Tensor::new(self.dt, &[3584])
    }

    pub fn r_proj_w(&self) -> Tensor<usize> {
        Tensor::new(self.dt, &[3584, 3584])
    }
}
