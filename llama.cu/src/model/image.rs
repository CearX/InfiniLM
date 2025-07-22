use super::preprocess::{PreprocessConfig, bicubic_resize, normalize};
use clip::{Image, qwen2vl_image_preprocess};
use half::f16;
use image::{DynamicImage, GenericImageView};
use mem_rearrange::Rearranging;
use ndarray::{Array4, Axis};
use ndarray_layout::{ArrayLayout, Endian};
use nn::{Tensor, digit_layout::types};
use std::{env::var_os, path::PathBuf};

#[allow(dead_code)]
pub fn image_path() -> Option<PathBuf> {
    var_os("TEST_IMAGE").map(PathBuf::from)
}

/// 合成图片预处理，返回标准化后的 Array4<f16>
#[allow(dead_code)]
fn preprocess_image_for_qw2vl(img: &DynamicImage, image_size: u32) -> Array4<f16> {
    let (in_w, in_h) = img.dimensions();
    let patch_size = 14;
    let factor = patch_size * 2;
    let out_w = ((in_w + factor - 1) / factor) * factor;
    let out_h = ((in_h + factor - 1) / factor) * factor;
    let rgb = img.to_rgb8();
    let resized = bicubic_resize(&rgb, out_w, out_h);
    let config = PreprocessConfig {
        image_size,
        ..Default::default()
    };
    let arr = normalize(&resized, config.mean, config.std); // (3, H, W)
    let arr_f16 = arr.mapv(|x| f16::from_f32(x));
    arr_f16.insert_axis(Axis(0)) // (1, 3, H, W)
}

#[allow(dead_code)]
pub fn qw2vl_image_preprocess_2() -> Tensor<Vec<u8>, 2> {
    use std::time::Instant;
    let time = Instant::now();
    // image preprocess
    let Some(picture) = image_path() else {
        panic!("No test image found");
    };
    let buf = std::fs::read(&picture).expect("Failed to read image file");
    let img = image::load_from_memory(&buf).expect("Failed to load image");
    println!("load image {:?}", time.elapsed());
    let arr = preprocess_image_for_qw2vl(&img, 0);
    println!("image preprocess {:?}", time.elapsed());
    let shape = arr.shape().to_vec();
    let strides = arr.strides().to_vec();
    let offset = 0 as isize;
    let arr = unsafe {
        std::slice::from_raw_parts(arr.as_ptr() as *const u8, arr.len() * size_of::<f16>())
    };
    // rearrange
    let shape = <[usize; 4]>::try_from(shape).unwrap();
    let src_strides = <[isize; 4]>::try_from(strides)
        .unwrap()
        .map(|x| x * size_of::<f16>() as isize);
    let src_layout = ArrayLayout::<2>::new(&shape, &src_strides, offset);
    let dst_layout = ArrayLayout::<2>::new_contiguous(&shape, Endian::BigEndian, 2);
    let scheme = Rearranging::new(&dst_layout, &src_layout, 2).unwrap();
    let binding = vec![0u8; arr.len()];
    let image = binding.as_slice();
    let dst = image.as_ptr() as *mut u8;
    let src = arr.as_ptr() as *const u8;
    unsafe { scheme.launch(dst, src) };
    // return tensor
    Tensor::from_raw_parts(types::F16, dst_layout, image.to_vec())
}

#[test]
fn test_qwen2vl_image_preprocess_2() {
    use half::f16;
    let image = qw2vl_image_preprocess_2();
    let shape = image.shape().to_vec();
    let strides = image.strides().to_vec();
    let offset = image.offset();
    let image = unsafe {
        std::slice::from_raw_parts(
            image.get().as_ptr() as *const f16,
            image.get().len() / size_of::<f16>(),
        )
    };
    println!("image shape: {shape:?}");
    println!("image strides: {strides:?}");
    println!("image offset: {offset:?}");
    println!("image tensor: {image:?}");
}

pub fn qw2vl_image_preprocess() -> Tensor<Vec<u8>, 2> {
    use std::time::Instant;
    let time = Instant::now();
    // image preprocess
    let Some(picture) = test_utils::image() else {
        panic!();
    };
    let image = Image::load(&picture);
    println!("load image {:?}", time.elapsed());
    let image_mean: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    let image_std: [f32; 3] = [0.26862954, 0.26130258, 0.27577711]; // todo: from model
    let whole = qwen2vl_image_preprocess(&image, image_mean, image_std);
    let raw = whole.to_nchw();
    let shape = raw.shape().to_vec();
    let strides = raw.strides().to_vec();
    let offset = raw.offset() as isize;
    // rearrange
    let shape = <[usize; 4]>::try_from(shape).unwrap();
    let src_strides = <[isize; 4]>::try_from(strides).unwrap();
    let src_layout = ArrayLayout::<2>::new(&shape, &src_strides, offset);
    let dst_layout = ArrayLayout::<2>::new_contiguous(&shape, Endian::BigEndian, 2);
    let scheme = Rearranging::new(&dst_layout, &src_layout, 2).unwrap();
    let binding = vec![0u8; raw.get().len()];
    let image = binding.as_slice();
    let dst = image.as_ptr() as *mut u8;
    let src = raw.get().as_ptr() as *const u8;
    unsafe { scheme.launch(dst, src) };
    // return tensor
    Tensor::from_raw_parts(types::F16, dst_layout, image.to_vec())
}

#[test]
fn test_qwen2vl_image_preprocess() {
    use half::f16;
    let image = qw2vl_image_preprocess();
    let shape = image.shape().to_vec();
    let strides = image.strides().to_vec();
    let offset = image.offset();
    let image = unsafe {
        std::slice::from_raw_parts(
            image.get().as_ptr() as *const f16,
            image.get().len() / size_of::<f16>(),
        )
    };
    println!("image shape: {shape:?}");
    println!("image strides: {strides:?}");
    println!("image offset: {offset:?}");
    println!("image tensor: {image:?}");
}
