use half::f16;
use image::{DynamicImage, RgbImage};
use ndarray::Array3;

const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub struct PreprocessConfig {
    pub image_size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            mean: DEFAULT_MEAN,
            std: DEFAULT_STD,
        }
    }
}

pub fn preprocess(img: &DynamicImage, config: &PreprocessConfig) -> Array3<f32> {
    let resized = bicubic_resize(&img.to_rgb8(), config.image_size, config.image_size);
    normalize(&resized, config.mean, config.std)
}

pub fn preprocess_f16(img: &DynamicImage, config: &PreprocessConfig) -> Array3<f16> {
    let arr = preprocess(img, config);
    arr.mapv(|x| f16::from_f32(x))
}

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

pub fn bicubic_resize(input: &RgbImage, out_w: u32, out_h: u32) -> RgbImage {
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

pub fn normalize(img: &RgbImage, mean: [f32; 3], std: [f32; 3]) -> Array3<f32> {
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
