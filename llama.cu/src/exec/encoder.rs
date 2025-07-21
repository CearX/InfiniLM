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
    utils::meta,
};
use ggus::GGufMetaMapExt;
use nn::{Distribution, Tensor};
use operators::{
    Operator,
    attention::common_cpu::Operator as AttnCpu,
    attention::cuda::Operator as Attn,
    common_cpu::Cpu,
    conv::cuda::ConvIm2Col,
    cuda::{Device, Gpu},
    rearrange::cuda::Operator as Rearr,
};
use std::time::Instant;

pub fn qw2vl_infer(model_path: &str, use_cuda_graph: bool, image: Tensor<Vec<u8>, 2>) {
    use crate::model::{GGufModel, map_files};
    use operators::cuda;
    // 初始化 CUDA
    assert!(cuda::init().is_ok());
    // 加载模型
    let maps = map_files(model_path);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    gguf.insert_sin_cos_qw2vl();
    let d_patch = meta![&gguf => llm_attention_head_count_kv; 14]; // todo: ggus
    let model = gguf.qw2vl_mmproj();
    // 初始化算子
    let device = Device::new(0);
    let gpu = Gpu::new(device.retain_primary(), Default::default());
    // let attn = Attn::new(&gpu);
    let attn = AttnCpu::new(&Cpu);
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
            AttnType::AttnCpu(attn),
            Some(&conv),
            Some(&rearr),
            &mut handle,
            None,
        );

        // 保存 pos_ids 和 image
        let [n, _, h, w] = <[usize; 4]>::try_from(image.shape().to_vec()).unwrap();
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
            &stream,
        );

        // 推理
        let time = Instant::now();
        let reqs = vec![]; // QW2VL 不需要 cache
        let x = models.launch(key, &reqs, &mut handle, &stream);
        // utils::fmt(&x, stream.ctx());
        println!("encode {n} x {h} x {w} image in {:?}", time.elapsed());
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qw2vl_infer() {
        use crate::model::qw2vl_image_preprocess;
        let model = "/home/cearx/qy/model/Qwen2VLnn-mmproj-2B-Instruct-v2.0-F16.gguf";
        let image = qw2vl_image_preprocess();
        qw2vl_infer(model, false, image);
    }
}
