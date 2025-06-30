use clip::{Image, qwen2vl_image_preprocess};
use ndarray_layout::ArrayLayout;
use nn::{Tensor, digit_layout::types};

pub fn qw2vl_image_preprocess() -> Tensor<Vec<u8>, 4> {
    use std::time::Instant;
    let time = Instant::now();
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
    let offset = raw.offset();
    assert_eq!(shape.len(), 4);
    assert_eq!(strides.len(), 4);
    let layout = ArrayLayout::<4>::new(
        &[shape[0], shape[1], shape[2], shape[3]],
        &[strides[0], strides[1], strides[2], strides[3]],
        offset as isize,
    );
    Tensor::from_raw_parts(types::F16, layout, raw.take().to_vec())
}

#[test]
fn test_qwen2vl_image_preprocess() {
    use half::f16;
    let image = qw2vl_image_preprocess();
    let shape = image.shape().to_vec();
    let strides = image.strides().to_vec();
    let offset = image.offset();
    let len = shape.clone().into_iter().product();
    let image = image.take();
    let image = unsafe { std::slice::from_raw_parts(image.as_ptr() as *const f16, len) };
    println!("image shape: {len:?}");
    println!("image shape: {shape:?}");
    println!("image strides: {strides:?}");
    println!("image offset: {offset:?}");
    println!("image tensor: {image:?}"); // <(1*336*476) * 3>
}
