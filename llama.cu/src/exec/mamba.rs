use crate::exec::group::{ModelGroupConfig, ModelGroupMamba};
use crate::exec::mamba_cache::MambaCache;
use crate::exec::output_head::OutputHead;
use crate::exec::sample_manager::SampleManager;
use crate::memory::MemPages;
use crate::op::random_sample::{KVPair, SampleArgs};
use crate::utils::meta;
use crate::{handle::Handle, model::map_files};
use cuda::Device;
use ggus::GGufMetaMapExt;
use nn::Distribution;
use std::env;
use std::time::Instant;

#[allow(dead_code)]
pub fn mamba_infer(
    model_path: std::path::PathBuf,
    text: &str,
    use_cuda_graph: bool,
) -> (String, usize, f64) {
    use crate::model::GGufModel;
    // 初始化 CUDA
    assert!(cuda::init().is_ok());

    // 加载模型
    let maps = map_files(model_path);
    let gguf = GGufModel::read(maps.iter().map(|x| &**x));
    // let tokenizer = Bpe::from_gguf(&gguf);

    use tokenizers::tokenizer::Tokenizer;
    let tokenizer =
        Tokenizer::from_file("/home/shared/models/mamba-2.8b-hf/tokenizer.json").unwrap();
    let encoding = tokenizer.encode(text, false).unwrap();
    let mut tokens = encoding.get_ids().to_vec();

    // let mut tokens = tokenizer.encode(text);

    let n_tok = tokens.len();

    // 取出输出头用于 logits 计算
    let mut mamba = gguf.mamba();
    let output_head_nn = mamba
        .output_head
        .take()
        .expect("mamba model missing output_head");

    let n_layer: usize = meta![gguf => llm_block_count];
    let d_inner: usize = 5120; // TODO: ggus
    let d_conv: usize = 4; // kernel size
    let d_state: usize = 16; // ssm state size

    // 单卡
    let device = Device::new(0);
    device.retain_primary().apply(|ctx| {
        let mut handle = Handle::new(ctx);
        let dist = Distribution {
            start: 0,
            len: 1,
            total: 1,
        };
        let mut models = ModelGroupMamba::new(
            mamba,
            dist,
            None,
            ModelGroupConfig {
                static_model_keys: [n_tok],
                dyn_cache_size: 1,
                use_cuda_graph,
            },
            &mut handle,
            None,
        );

        // 组件：输出头与采样器
        let stream = ctx.stream();
        let mut output_head = OutputHead::new(output_head_nn, ctx);

        // 读取词表大小构建采样器与 eos
        let eos: tokeneer::utok = meta![gguf => tokenizer_ggml_eos_token_id];
        let nvoc = output_head.nvoc();
        let mut sample_manager = SampleManager::new(nvoc, eos, ctx);

        // 初始化 MambaCache
        let mut pages = MemPages::new(device);
        let mut mcache = MambaCache::new(n_layer, d_inner, d_conv, d_state, &mut pages);

        let start = Instant::now();
        // Prefill
        let (key, _tok_buf) =
            models.load_inputs_mamba_prefill(&mut handle, tokens.len(), &tokens, &stream);

        let mut x = models.launch_mamba(key, &mut mcache, &mut handle, &stream);

        let last_idx: [tokeneer::utok; 1] = [(tokens.len() - 1) as tokeneer::utok];
        let logits_prefill_last = output_head.launch(x.clone(), &last_idx, &mut handle, &stream);

        let prefill_time = start.elapsed();
        println!("prefill time = {:.2}", prefill_time.as_secs_f32());

        // let logits_prefill_last_vir = logits_prefill_last
        //     .as_ref()
        //     .map(|mem| mem.as_ptr().cast::<VirByte>());
        // utils::fmt(&logits_prefill_last_vir, stream.ctx());
        // check prefill logits

        let mut next_id: tokeneer::utok;
        {
            let mut input = stream.malloc::<tokeneer::utok>(tokens.len());
            stream.memcpy_h2d(&mut input, &tokens);
            let cfg0 = vec![(
                crate::batch::SessionId(0),
                crate::batch::SampleInfo {
                    args: SampleArgs::new(0.8, 0.95, 50, 1.3).unwrap(),
                    input_idx: tokens.len(),
                    decode_len: tokens.len(),
                },
            )];
            let kv_pairs0 = sample_manager.sample(logits_prefill_last, &input, &cfg0, &stream);
            stream.free(input);
            let mut host_kv0 = vec![KVPair::ZERO; 1];
            stream.memcpy_d2h(&mut host_kv0, &kv_pairs0).free(kv_pairs0);
            next_id = host_kv0[0].idx as tokeneer::utok;
        }
        let mut generated: Vec<tokeneer::utok> = Vec::new();
        if next_id != eos {
            tokens.push(next_id);
            generated.push(next_id);
            let (key, _tok_buf) = models.load_input_mamba_decode(&mut handle, next_id, &stream);
            x = models.launch_mamba(key, &mut mcache, &mut handle, &stream);
        }

        let max_decode_steps: usize = env::var("MAMBA_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);
        println!("max steps    = {}", max_decode_steps);
        for _step in 1..max_decode_steps {
            let out_idx: [tokeneer::utok; 1] = [0];

            let logits = output_head.launch(x.clone(), &out_idx, &mut handle, &stream);

            let mut input = stream.malloc::<tokeneer::utok>(tokens.len());
            stream.memcpy_h2d(&mut input, &tokens);
            let cfg = vec![(
                crate::batch::SessionId(0),
                crate::batch::SampleInfo {
                    args: SampleArgs::new(0.8, 0.95, 50, 1.3).unwrap(),
                    input_idx: tokens.len(),
                    decode_len: tokens.len(),
                },
            )];
            let kv_pairs = sample_manager.sample(logits, &input, &cfg, &stream);
            stream.free(input);
            let mut host_kv = vec![KVPair::ZERO; 1];
            stream.memcpy_d2h(&mut host_kv, &kv_pairs).free(kv_pairs);
            next_id = host_kv[0].idx as tokeneer::utok;

            if next_id == eos {
                break;
            }

            tokens.push(next_id);
            generated.push(next_id);
            let (key, _tok_buf) = models.load_input_mamba_decode(&mut handle, next_id, &stream);
            x = models.launch_mamba(key, &mut mcache, &mut handle, &stream);
        }

        let decode_time = start.elapsed() - prefill_time;
        println!("decode time  = {:.2}", decode_time.as_secs_f64());
        // println!("tokens = {:?}", tokens);
        // let mut text_buf = tokeneer::TextBuf::new();
        // let s = tokenizer.decode(&generated, &mut text_buf);
        // let text = String::from_utf8_lossy(&s.into_bytes()).to_string();
        let text = tokenizer.decode(&generated, false).unwrap();

        let total_infer_time = (prefill_time + decode_time).as_secs_f64();

        (text, tokens.len(), total_infer_time)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::PathBuf, time::Instant};

    #[test]
    fn test_mamba_infer_decode() {
        let start = Instant::now();
        let model = PathBuf::from("/home/cearx/Mamba-2.8B-hf-v1.0-F16.gguf");
        let prompt = "Once upon a time,";
        let (text, len, infer_time) = mamba_infer(model, prompt, false);
        let end = Instant::now();
        let tokens_per_second = len as f64 / infer_time;
        let total_time = end - start;
        println!("total time   = {:.2} s", total_time.as_secs_f64());
        println!("infer time   = {:.2} s", infer_time);
        println!("tokens/s     = {:.2}", tokens_per_second);
        println!("prompt       = {}", prompt);
        println!("output text  = {}", text);
    }
}
