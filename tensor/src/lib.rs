mod fmt;
mod split;

use ggus::ggml_quants::{digit_layout::DigitLayout, DataBlock};
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use operators::{random_sample::KVPair, TensorLayout};
use std::{
    alloc::Layout,
    ops::{Deref, DerefMut, Range},
    slice::from_raw_parts,
};

pub use split::{LocalSplitable, Splitable};

#[derive(Clone)]
pub struct Tensor<T> {
    element: DigitLayout,
    layout: ArrayLayout<5>,
    physical: T,
}

impl Tensor<usize> {
    pub fn new(element: DigitLayout, shape: &[usize]) -> Self {
        assert!(shape.len() <= 5);

        let mut buf = [0; 5];
        buf[..shape.len()].copy_from_slice(shape);
        buf[shape.len() - 1] /= block_size(element);
        let shape = &buf[..shape.len()];

        let ele = dt_size(element);
        Self {
            element,
            layout: ArrayLayout::new_contiguous(shape, BigEndian, ele),
            physical: shape.iter().product::<usize>() * ele,
        }
    }
}

/// access
impl<T> Tensor<T> {
    #[inline]
    pub const fn dt(&self) -> DigitLayout {
        self.element
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.layout.offset()
    }

    #[inline]
    pub fn take(self) -> T {
        self.physical
    }

    #[inline]
    pub fn get(&self) -> &T {
        &self.physical
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.physical
    }

    #[inline]
    pub fn as_ref(&self) -> Tensor<&T> {
        Tensor {
            element: self.element,
            layout: self.layout.clone(),
            physical: &self.physical,
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> Tensor<&mut T> {
        Tensor {
            element: self.element,
            layout: self.layout.clone(),
            physical: &mut self.physical,
        }
    }

    #[inline]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Tensor<U> {
        Tensor {
            element: self.element,
            layout: self.layout,
            physical: f(self.physical),
        }
    }
}

impl<T, B> Tensor<T>
where
    T: Deref<Target = [B]>,
{
    /// # Safety
    ///
    /// 这个函数将在移除生命周期约束的情况下引用原始数据，对这块存储空间进行读写的安全性由开发者保证。
    #[inline]
    pub unsafe fn map_slice_static(&self) -> Tensor<&'static [B]> {
        self.as_ref()
            .map(|x| unsafe { from_raw_parts(x.as_ptr(), x.len()) })
    }

    #[inline]
    pub fn map_slice(&self) -> Tensor<&[B]> {
        self.as_ref().map(|x| &x[..])
    }

    #[inline]
    pub fn base(&self) -> *const B {
        unsafe { self.physical.as_ptr().byte_add(self.layout.offset()) }
    }
}

impl<T, B> Tensor<T>
where
    T: DerefMut<Target = [B]>,
{
    #[inline]
    pub fn map_slice_mut(&mut self) -> Tensor<&mut [B]> {
        self.as_mut().map(|x| &mut x[..])
    }

    #[inline]
    pub fn base_mut(&mut self) -> *mut B {
        unsafe { self.physical.as_mut_ptr().byte_add(self.layout.offset()) }
    }
}

/// operators
impl<T> Tensor<T> {
    pub fn kv_pair_vec(n: usize, f: impl FnOnce(usize) -> T) -> Self {
        Self {
            element: KVPair::<()>::LAYOUT,
            layout: ArrayLayout::new_contiguous(&[n], BigEndian, size_of::<KVPair>()),
            physical: f(Layout::array::<KVPair>(n).unwrap().size()),
        }
    }

    #[inline]
    pub fn layout(&self) -> TensorLayout {
        TensorLayout::new(self.element, self.layout.shape(), self.layout.strides())
    }
}

/// transform
impl<T> Tensor<T> {
    #[inline]
    pub fn transpose(self, perm: &[usize]) -> Self {
        Self {
            layout: self.layout.transpose(perm),
            ..self
        }
    }

    #[inline]
    pub fn index(self, axis: usize, index: usize) -> Self {
        Self {
            layout: self.layout.index(axis, index),
            ..self
        }
    }

    #[inline]
    pub fn slice(self, axis: usize, start: usize, step: isize, len: usize) -> Self {
        Self {
            layout: self.layout.slice(axis, start, step, len),
            ..self
        }
    }

    #[inline]
    pub fn tile(self, axis: usize, tiles: &[usize]) -> Self {
        Self {
            layout: self.layout.tile_be(axis, tiles),
            ..self
        }
    }

    #[inline]
    pub fn merge(self, range: Range<usize>) -> Option<Self> {
        self.layout
            .merge(range)
            .map(|layout| Self { layout, ..self })
    }
}

impl<T: Splitable> Tensor<T> {
    pub fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> impl Iterator<Item = Self> + 'a {
        self.layout.split(axis, parts).map(|layout| Self {
            element: self.element,
            layout,
            physical: self.physical.split(),
        })
    }
}

pub const fn dt_size(dt: DigitLayout) -> usize {
    use ggus::ggml_quants::{
        bf16, digit_layout::types as primitive, f16, types as quantized, IQ1M, IQ1S, IQ2S, IQ2XS,
        IQ2XXS, IQ3S, IQ3XXS, IQ4NL, IQ4XS, Q2K, Q3K, Q4K, Q4_0, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8,
        Q4_1, Q5K, Q5_0, Q5_1, Q6K, Q8K, Q8_0, Q8_1,
    };
    #[rustfmt::skip]
    let ans = match dt {
        primitive::U8       => size_of::<u8      >(),
        primitive::I8       => size_of::<i8      >(),
        primitive::U16      => size_of::<u16     >(),
        primitive::I16      => size_of::<i16     >(),
        primitive::F16      => size_of::<f16     >(),
        primitive::BF16     => size_of::<bf16    >(),
        primitive::U32      => size_of::<u32     >(),
        primitive::I32      => size_of::<i32     >(),
        primitive::F32      => size_of::<f32     >(),
        primitive::U64      => size_of::<u64     >(),
        primitive::I64      => size_of::<i64     >(),
        primitive::F64      => size_of::<f64     >(),
        quantized::IQ1M     => size_of::<IQ1M    >(),
        quantized::IQ1S     => size_of::<IQ1S    >(),
        quantized::IQ2S     => size_of::<IQ2S    >(),
        quantized::IQ2XS    => size_of::<IQ2XS   >(),
        quantized::IQ2XXS   => size_of::<IQ2XXS  >(),
        quantized::IQ3S     => size_of::<IQ3S    >(),
        quantized::IQ3XXS   => size_of::<IQ3XXS  >(),
        quantized::IQ4NL    => size_of::<IQ4NL   >(),
        quantized::IQ4XS    => size_of::<IQ4XS   >(),
        quantized::Q2K      => size_of::<Q2K     >(),
        quantized::Q3K      => size_of::<Q3K     >(),
        quantized::Q4_0     => size_of::<Q4_0    >(),
        quantized::Q4_0_4_4 => size_of::<Q4_0_4_4>(),
        quantized::Q4_0_4_8 => size_of::<Q4_0_4_8>(),
        quantized::Q4_0_8_8 => size_of::<Q4_0_8_8>(),
        quantized::Q4_1     => size_of::<Q4_1    >(),
        quantized::Q4K      => size_of::<Q4K     >(),
        quantized::Q5_0     => size_of::<Q5_0    >(),
        quantized::Q5_1     => size_of::<Q5_1    >(),
        quantized::Q5K      => size_of::<Q5K     >(),
        quantized::Q6K      => size_of::<Q6K     >(),
        quantized::Q8_0     => size_of::<Q8_0    >(),
        quantized::Q8_1     => size_of::<Q8_1    >(),
        quantized::Q8K      => size_of::<Q8K     >(),
        _                   =>               todo!(),
    };
    ans
}

pub const fn block_size(dt: DigitLayout) -> usize {
    use ggus::ggml_quants::{
        digit_layout::types as primitive, types as quantized, IQ1M, IQ1S, IQ2S, IQ2XS, IQ2XXS,
        IQ3S, IQ3XXS, IQ4NL, IQ4XS, Q2K, Q3K, Q4K, Q4_0, Q4_1, Q5K, Q5_0, Q5_1, Q6K, Q8K, Q8_0,
        Q8_1,
    };
    #[rustfmt::skip]
    let ans = match dt {
        primitive::U8   |
        primitive::I8   |
        primitive::U16  |
        primitive::I16  |
        primitive::F16  |
        primitive::BF16 |
        primitive::U32  |
        primitive::I32  |
        primitive::F32  |
        primitive::U64  |
        primitive::I64  |
        primitive::F64    => 1,
        quantized::IQ1M   => IQ1M  ::COUNT,
        quantized::IQ1S   => IQ1S  ::COUNT,
        quantized::IQ2S   => IQ2S  ::COUNT,
        quantized::IQ2XS  => IQ2XS ::COUNT,
        quantized::IQ2XXS => IQ2XXS::COUNT,
        quantized::IQ3S   => IQ3S  ::COUNT,
        quantized::IQ3XXS => IQ3XXS::COUNT,
        quantized::IQ4NL  => IQ4NL ::COUNT,
        quantized::IQ4XS  => IQ4XS ::COUNT,
        quantized::Q2K    => Q2K   ::COUNT,
        quantized::Q3K    => Q3K   ::COUNT,
        quantized::Q4_0   => Q4_0  ::COUNT,
        quantized::Q4_1   => Q4_1  ::COUNT,
        quantized::Q4K    => Q4K   ::COUNT,
        quantized::Q5_0   => Q5_0  ::COUNT,
        quantized::Q5_1   => Q5_1  ::COUNT,
        quantized::Q5K    => Q5K   ::COUNT,
        quantized::Q6K    => Q6K   ::COUNT,
        quantized::Q8_0   => Q8_0  ::COUNT,
        quantized::Q8_1   => Q8_1  ::COUNT,
        quantized::Q8K    => Q8K   ::COUNT,
        _                 => todo!(),
    };
    ans
}
