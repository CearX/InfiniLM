use crate::utils::Blob;
use crate::{
    exec::{
        engine::BufN,
        group::{ModelGroupConfig, ModelGroupQw2vl},
        upos,
    },
    handle::Handle,
    model::{image::qw2vl_image_preprocess, qw2vl_mmproj::build_pos_ids},
};
use nn::{Distribution, Tensor};
use operators::cuda::VirByte;
use operators::{
    Operator as _,
    attention::common_cpu::Operator as AttnCpu,
    // attention::cuda::Operator as Attn,
    common_cpu::Cpu,
    conv::cuda::ConvIm2Col,
    cuda::{DevByte, Device, Gpu, memcpy_d2h},
    rearrange::cuda::Operator as Rearrange,
};
use std::{env::var_os, path::PathBuf, time::Instant};

#[allow(dead_code)]
pub fn model_from_env() -> PathBuf {
    let Some(model) = var_os("TEST_MODEL").map(PathBuf::from) else {
        panic!("TEST_MODEL not set");
    };
    model
}

#[allow(dead_code)]
pub fn qw2vl_infer(
    model_path: PathBuf,
    image: PathBuf,
    use_cuda_graph: bool,
) -> (Vec<u8>, [usize; 5]) {
    use crate::model::{GGufModel, map_files};
    use operators::cuda;
    // 初始化 CUDA
    assert!(cuda::init().is_ok());
    // 加载 model 和 image
    let maps = map_files(model_path);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    let d_patch = 14;
    let image_mean: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
    let image_std: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1]; // todo: ggus
    let image = qw2vl_image_preprocess(image, image_mean, image_std);
    let image_shape = <[usize; 4]>::try_from(image.shape().to_vec()).unwrap();
    let [n, _c, h, w] = image_shape;
    let patches = (h / d_patch) * (w / d_patch);
    let nctx = (h / d_patch).max(w / d_patch);
    gguf.insert_sin_cos_qw2vl_mmproj(nctx);
    let model = gguf.qw2vl_mmproj(nctx);
    // 初始化算子
    let device = Device::new(0);
    let gpu = Gpu::new(device.retain_primary(), Default::default());
    // let attn = Attn::new(&gpu);
    let attn = AttnCpu::new(&Cpu);
    let conv = ConvIm2Col::new(&gpu);
    let rearrange = Rearrange::new(&gpu);

    gpu.apply(|ctx| {
        let mut handle = Handle::new(ctx);
        let dist = Distribution {
            start: 0,
            len: 1,
            total: 1,
        };
        let mut models = ModelGroupQw2vl::new(
            model,
            image_shape,
            d_patch,
            dist,
            None,
            ModelGroupConfig {
                static_model_keys: [patches],
                dyn_cache_size: 1,
                use_cuda_graph,
            },
            attn,
            conv,
            rearrange,
            &mut handle,
            None,
        );

        // 保存 pos_ids 和 image
        let pos_ids = build_pos_ids(h, w, d_patch);
        let pos_len = pos_ids.len();
        let image_data = image.take();
        let image_len = image_data.len();
        const BUF_LEVEL: usize = 3;
        let mut image_buf = BufN::<u8>::new(image_len, BUF_LEVEL, ctx);
        let mut pos_buf = BufN::<upos>::new(pos_len, BUF_LEVEL, ctx);
        image_buf.save(image_data.as_slice());
        pos_buf.save(&pos_ids);

        // 加载到设备
        let stream = ctx.stream();
        let (key, _tok_buf) = models.load_inputs_qw2vl_mmproj(
            &mut handle,
            image_len,
            pos_len,
            &image_buf,
            &pos_buf,
            image_shape,
            d_patch,
            &stream,
        );

        // 推理
        let time = Instant::now();
        let reqs = vec![]; // QW2VLMMProj 不需要 cache
        let x = models.launch(key, &reqs, &mut handle, &stream);
        // utils::fmt(&_x, stream.ctx());
        let img_token_len = x.shape()[0];
        let d2h = |tensor: &Tensor<*const VirByte, 2>| {
            let mem_range = tensor.layout().data_range();
            let ptr = tensor.get().cast::<DevByte>();
            let len = *mem_range.end() as usize + tensor.dt().nbytes();
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut host = Blob::new(len);
            memcpy_d2h(&mut host, slice);
            tensor.as_ref().map(|_| host)
        };
        let x = d2h(&x);
        let x = x.as_deref().map(|t| t.to_vec()).take();
        println!("encode {n} x {h} x {w} image in {:?}", time.elapsed());
        (x, [n, h, w, d_patch, img_token_len])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    fn _test_qw2vl_infer() {
        use crate::model::image::image_from_env;
        let model = model_from_env();
        let image = image_from_env();
        let _x = qw2vl_infer(model, image, false);
    }
}
