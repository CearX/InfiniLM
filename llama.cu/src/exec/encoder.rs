use crate::{
    exec::{
        Model,
        engine::BufN,
        group::{ModelGroup, ModelGroupConfig},
        model::AttnType,
        upos,
    },
    handle::Handle,
    model::build_pos_ids,
};
use nn::{Distribution, Qwen2VLmmproj, Tensor};
use operators::{
    Operator,
    attention::cuda::Operator as Attn,
    conv::cuda::ConvIm2Col,
    cuda::{Device, Gpu},
    rearrange::cuda::Operator as Rearr,
};

pub fn qw2vl_infer(
    model: Qwen2VLmmproj<Tensor<&[u8], 2>>,
    device: Device,
    use_cuda_graph: bool,
    image: Tensor<Vec<u8>, 2>,
) {
    let gpu = Gpu::new(device.retain_primary(), Default::default());
    let attn = Attn::new(&gpu);
    let conv = ConvIm2Col::new(&gpu);
    let rearr = Rearr::new(&gpu);

    gpu.apply(|ctx| {
        let mut handle = Handle::new(ctx);
        let dist = Distribution {
            start: 0,
            len: 1,
            total: 1,
        };
        let mut models = ModelGroup::new(
            Model::QWEN2VLMMPROJ(model),
            dist,
            None,
            ModelGroupConfig {
                static_model_keys: [1],
                dyn_cache_size: 1,
                use_cuda_graph,
            },
            AttnType::ATTN(attn),
            Some(&conv),
            Some(&rearr),
            &mut handle,
            None,
        );

        let stream = ctx.stream();
        let shape = image.shape().to_vec();
        let h = shape[2];
        let w = shape[3];
        let pos_ids = build_pos_ids(h, w, 14);

        // 图像处理和缓冲区
        let image_len = 1 * 3 * 336 * 476 * 2;
        let pos_len = 24 * 34 * 2 * 4;
        const BUF_LEVEL: usize = 3;
        let mut image_buf = BufN::<u8>::new(image_len, BUF_LEVEL, ctx);
        let mut pos_buf = BufN::<upos>::new(pos_len, BUF_LEVEL, ctx);

        // 获取和保存图像数据
        let image_data = image.take();
        image_buf.save(image_data.as_slice());
        pos_buf.save(&pos_ids);

        // 然后传 buffer 给 load_inputs_qw2vl_mmproj
        let (key, _tok_buf) = models.load_inputs_qw2vl_mmproj(
            &mut handle,
            image_data.len(),
            pos_ids.len(),
            &image_buf,
            &pos_buf,
            &stream,
        );

        // 推理
        let reqs = vec![]; // QW2VL 不需要 cache
        let _x = models.launch(key, &reqs, &mut handle, &stream);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GGufModel, map_files, qw2vl_image_preprocess};

    #[test]
    fn test_qw2vl_infer() {
        let model = "/home/cearx/qy/model/Qwen2VLnn-mmproj-2B-Instruct-v2.0-F16.gguf";
        let maps = map_files(model);
        let gguf = GGufModel::read(maps.iter().map(|x| &**x));
        let qw2 = gguf.qw2vl_mmproj();

        let image = qw2vl_image_preprocess();

        let device = Device::new(0);
        qw2vl_infer(qw2, device, true, image);
    }
}
