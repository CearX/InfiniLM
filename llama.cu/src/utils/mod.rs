mod blob;
mod fmt;
mod macros;

use nn::Tensor;

pub(crate) use blob::{Blob, Data};
pub(crate) use fmt::fmt;
pub(crate) use macros::*;

#[inline(always)]
pub(crate) fn offset_ptr<T, const N: usize>(t: &Tensor<*const T, N>) -> *const T {
    unsafe { t.get().byte_offset(t.layout().offset()) }
}

pub(crate) fn distinct<T: Eq + Copy>(val: &[T]) -> Option<T> {
    let [ans, tail @ ..] = val else {
        return None;
    };
    if tail.iter().all(|x| x == ans) {
        Some(*ans)
    } else {
        None
    }
}
