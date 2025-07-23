use half::f16;
use image::{DynamicImage, GenericImageView, RgbImage};
use mem_rearrange::Rearranging;
use ndarray::{Array3, Array4, Axis};
use ndarray_layout::{ArrayLayout, Endian};
use nn::{Tensor, digit_layout::types};
use std::{env::var_os, path::PathBuf};

fn cubic_kernel(x: f32) -> f32 {
    let abs_x = x.abs();
    if abs_x < 1.0 {
        (1.5 * abs_x.powi(3)) - (2.5 * abs_x.powi(2)) + 1.0
    } else if abs_x < 2.0 {
        (-0.5 * abs_x.powi(3)) + (2.5 * abs_x.powi(2)) - (4.0 * abs_x) + 2.0
    } else {
        0.0
    }
}

fn bicubic_resize(input: &RgbImage, out_w: u32, out_h: u32) -> RgbImage {
    let (in_w, in_h) = input.dimensions();
    let mut out = RgbImage::new(out_w, out_h);
    for y in 0..out_h {
        let fy = (y as f32 + 0.5) * (in_h as f32 / out_h as f32) - 0.5;
        let y_int = fy.floor() as i32;
        let y_frac = fy - y_int as f32;
        for x in 0..out_w {
            let fx = (x as f32 + 0.5) * (in_w as f32 / out_w as f32) - 0.5;
            let x_int = fx.floor() as i32;
            let x_frac = fx - x_int as f32;
            let mut rgb = [0.0f32; 3];
            for m in -1..=2 {
                let wy = cubic_kernel(m as f32 - y_frac);
                let sy = y_int + m;
                if sy < 0 || sy >= in_h as i32 {
                    continue;
                }
                for n in -1..=2 {
                    let wx = cubic_kernel(x_frac - n as f32);
                    let sx = x_int + n;
                    if sx < 0 || sx >= in_w as i32 {
                        continue;
                    }
                    let pixel = input.get_pixel(sx as u32, sy as u32);
                    for c in 0..3 {
                        rgb[c] += pixel[c] as f32 * wx * wy;
                    }
                }
            }
            let pixel = image::Rgb([
                rgb[0].clamp(0.0, 255.0) as u8,
                rgb[1].clamp(0.0, 255.0) as u8,
                rgb[2].clamp(0.0, 255.0) as u8,
            ]);
            out.put_pixel(x, y, pixel);
        }
    }
    out
}

fn normalize(img: &RgbImage, mean: [f32; 3], std: [f32; 3]) -> Array3<f32> {
    let (w, h) = img.dimensions();
    let mut arr = Array3::<f32>::zeros((3, h as usize, w as usize));
    for (x, y, pixel) in img.enumerate_pixels() {
        for c in 0..3 {
            let val = pixel[c] as f32 / 255.0;
            arr[[c, y as usize, x as usize]] = (val - mean[c]) / std[c];
        }
    }
    arr
}

fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> (u32, u32) {
    let height = height as f32;
    let width = width as f32;
    let factor = factor as f32;
    let min_pixels = min_pixels as f32;
    let max_pixels = max_pixels as f32;

    if height < factor || width < factor {
        panic!("height:{height} or width:{width} must be larger than factor:{factor}");
    } else if (height.max(width) / height.min(width)) > 200.0 {
        panic!(
            "absolute aspect ratio must be smaller than 200, got {}",
            height.max(width) / height.min(width)
        );
    }

    let mut h_bar = (height / factor).round() * factor;
    let mut w_bar = (width / factor).round() * factor;
    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) / max_pixels).sqrt();
        h_bar = ((height / beta) / factor).floor() * factor;
        w_bar = ((width / beta) / factor).floor() * factor;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels / (height * width)).sqrt();
        h_bar = ((height * beta) / factor).ceil() * factor;
        w_bar = ((width * beta) / factor).ceil() * factor;
    }

    (h_bar as u32, w_bar as u32)
}

fn preprocess_image_for_qw2vl(
    img: &DynamicImage,
    image_mean: [f32; 3],
    image_std: [f32; 3],
) -> Array4<f16> {
    let (in_w, in_h) = img.dimensions();
    let patch_size = 14;
    let factor = patch_size * 2;
    let min_pixels = 56 * 56;
    let max_pixels = 14 * 14 * 4 * 1280;
    let (out_h, out_w) = smart_resize(in_h, in_w, factor, min_pixels, max_pixels);
    let rgb = img.to_rgb8();
    let resized = bicubic_resize(&rgb, out_w, out_h);
    let arr = normalize(&resized, image_mean, image_std); // (3, H, W)
    let arr_f16 = arr.mapv(f16::from_f32);
    arr_f16.insert_axis(Axis(0)) // (1, 3, H, W)
}

#[allow(dead_code)]
pub(crate) fn image_from_env() -> PathBuf {
    let Some(img) = var_os("TEST_IMAGE").map(PathBuf::from) else {
        panic!("TEST_IMAGE not set");
    };
    img
}

pub(crate) fn qw2vl_image_preprocess(
    image: PathBuf,
    image_mean: [f32; 3],
    image_std: [f32; 3],
) -> Tensor<Vec<u8>, 2> {
    use std::time::Instant;
    let time = Instant::now();
    // image preprocess
    let buf = std::fs::read(&image).unwrap();
    let img = image::load_from_memory(&buf).unwrap();
    println!("load image {:?}", time.elapsed());
    let arr = preprocess_image_for_qw2vl(&img, image_mean, image_std);
    println!("image preprocess {:?}", time.elapsed());
    let shape = arr.shape().to_vec();
    let strides = arr.strides().to_vec();
    let offset = 0_isize;
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
    let src = arr.as_ptr();
    unsafe { scheme.launch(dst, src) };
    // return tensor
    Tensor::from_raw_parts(types::F16, dst_layout, image.to_vec())
}

// #[test]
fn _test_qw2vl_image_preprocess() {
    let image = image_from_env();
    let image_mean: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
    let image_std: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
    let image = qw2vl_image_preprocess(image, image_mean, image_std);
    let img = unsafe {
        std::slice::from_raw_parts(
            image.get().as_ptr() as *const f16,
            image.get().len() / size_of::<f16>(),
        )
    };
    println!("image shape: {:?}", image.shape());
    println!("image strides: {:?}", image.strides());
    println!("image offset: {:?}", image.offset());
    println!("image tensor: {img:?}");
}
