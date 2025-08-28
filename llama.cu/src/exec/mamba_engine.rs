use super::{
    Command, Output, engine_manager::EngineManager, group::ModelGroupMamba, output_head::OutputHead,
};
use crate::{
    CacheParts,
    batch::{Req, Round},
    exec::{group::ModelGroupConfig, mamba_cache::MambaCache, sample_manager::SampleManager, upos},
    handle::Handle,
    memory::MemPages,
    op::{FastEmbedding, random_sample::KVPair},
    utils::{Blob, meta},
};
use cuda::{ContextResource, CurrentCtx, Device, Event, HostMem};
use ggus::GGufMetaMapExt;
use nn::{Distribution, Mamba, Tensor};
use std::sync::{Mutex, OnceLock};
use std::{
    ffi::c_int,
    iter::zip,
    marker::PhantomData,
    num::NonZeroUsize,
    ops::Deref,
    sync::{
        Arc, Barrier, RwLock,
        mpsc::{Receiver, Sender},
    },
};
use tokeneer::utok;

// #[cfg(nccl)]
// use nccl::{Communicator, CommunicatorGroup};

// 全局存储用于PPL请求的logprobs
static LOGPROBS_STORAGE: OnceLock<Mutex<Option<Vec<f32>>>> = OnceLock::new();

// 全局PPL模式标记：指示当前是否有PPL请求正在处理
static PPL_MODE_ACTIVE: OnceLock<Mutex<bool>> = OnceLock::new();

/// Set PPL mode for current request
pub fn set_ppl_mode(active: bool) {
    let storage = PPL_MODE_ACTIVE.get_or_init(|| Mutex::new(false));
    if let Ok(mut guard) = storage.lock() {
        *guard = active;
    }
}

/// Check if PPL mode is currently active
fn is_ppl_mode_active() -> bool {
    let storage = PPL_MODE_ACTIVE.get_or_init(|| Mutex::new(false));
    if let Ok(guard) = storage.lock() {
        *guard
    } else {
        false
    }
}

/// Check if we should compute logprobs based on the requests
fn should_compute_logprobs(reqs: &[Req<CacheParts>]) -> bool {
    // 检查是否有PPL请求正在处理
    let should_compute = is_ppl_mode_active() && !reqs.is_empty();
    if should_compute {
        println!(
            "DEBUG: should_compute_logprobs = true (PPL mode), reqs.len() = {}",
            reqs.len()
        );
    }
    should_compute
}

/// Compute log_softmax on GPU from logits tensor
/// For PPL calculation, we only need logprobs for specific target tokens
fn compute_log_softmax_on_gpu(
    logits: &nn::Tensor<cuda::DevMem, 2>,
    stream: &cuda::Stream,
    target_tokens: Option<&[utok]>,
) -> Vec<f32> {
    // 获取 logits 的维度信息
    let shape = logits.shape();
    let seq_len = shape[0];
    let vocab_size = shape[1];
    let total_elements = seq_len * vocab_size;
    // println!("DEBUG: logits shape: [{}, {}]", seq_len, vocab_size);

    // 将 logits 从 GPU 复制到 CPU 进行计算
    let d2h = |tensor: &Tensor<cuda::DevMem, 2>| {
        let mem_range = tensor.layout().data_range();
        let ptr = tensor.get().as_ptr().cast::<cuda::DevByte>();
        let len = *mem_range.end() as usize + tensor.dt().nbytes();

        // // 调试：检查内存复制参数
        // println!(
        //     "DEBUG: d2h params - mem_range: {:?}, len: {}, dtype_nbytes: {}",
        //     mem_range,
        //     len,
        //     tensor.dt().nbytes()
        // );
        // println!(
        //     "DEBUG: expected_elements: {}, calculated_bytes: {}",
        //     total_elements,
        //     total_elements * 4
        // ); // f32 = 4 bytes

        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        let mut host = Blob::new(len);
        stream.memcpy_d2h(&mut host, slice);
        tensor.as_ref().map(|_| host)
    };

    let host_data = d2h(logits);
    let host_logits_raw = host_data.as_deref();

    // 检查数据类型：如果是 f16，需要转换为 f32
    let dtype_size = logits.dt().nbytes();
    println!("DEBUG: dtype size: {} bytes per element", dtype_size);

    let host_logits: Vec<f32> = if dtype_size == 2 {
        // f16 格式，需要转换
        println!("DEBUG: Converting f16 to f32");
        let host_logits_f16 = unsafe {
            std::slice::from_raw_parts(
                host_logits_raw.get().as_ptr().cast::<half::f16>(),
                total_elements,
            )
        };

        // // 调试：验证 f16 原始数据
        // println!("DEBUG: f16 raw data - first 10 values:");
        // for i in 0..10.min(total_elements) {
        //     println!(
        //         "  f16[{}] = {:?} -> f32 = {:.6}",
        //         i,
        //         host_logits_f16[i],
        //         host_logits_f16[i].to_f32()
        //     );
        // }

        // 转换为 f32
        host_logits_f16.iter().map(|&x| x.to_f32()).collect()
    } else {
        // f32 格式，直接使用
        println!("DEBUG: Using f32 directly");
        let host_logits_f32 = unsafe {
            std::slice::from_raw_parts(host_logits_raw.get().as_ptr().cast::<f32>(), total_elements)
        };
        println!("DEBUG: f32 raw data - first 10 values:");
        for i in 0..10.min(total_elements) {
            println!("  f32[{}] = {:?}", i, host_logits_f32[i]);
        }
        host_logits_f32.to_vec()
    };

    // 验证转换后的数据范围
    let min_val = host_logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = host_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let avg_val = host_logits.iter().sum::<f32>() / total_elements as f32;
    println!(
        "DEBUG: converted f32 range - min: {:.6}, max: {:.6}, avg: {:.6}",
        min_val, max_val, avg_val
    );

    // 检查是否有异常值
    let nan_count = host_logits.iter().filter(|&&x| x.is_nan()).count();
    let inf_count = host_logits.iter().filter(|&&x| x.is_infinite()).count();
    println!(
        "DEBUG: anomaly count - NaN: {}, Inf: {}",
        nan_count, inf_count
    );

    // 在 CPU 上计算 log_softmax
    if let Some(targets) = target_tokens {
        // PPL 模式：只计算目标 token 的 logprobs
        let mut logprobs = Vec::with_capacity(seq_len);
        println!(
            "DEBUG: PPL mode - computing logprobs for {} target tokens",
            targets.len()
        );

        for batch_idx in 0..seq_len {
            if batch_idx >= targets.len() {
                break; // 防止越界
            }

            let start_idx = batch_idx * vocab_size;
            let batch_logits = &host_logits[start_idx..start_idx + vocab_size];
            let target_token = targets[batch_idx] as usize;

            // 计算这个位置的 log_softmax，但只返回目标 token 的值
            // 1. 找到最大值以提高数值稳定性
            let max_logit = batch_logits
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // 2. 计算 sum(exp(x_j - max))
            let sum_exp: f32 = batch_logits.iter().map(|&x| (x - max_logit).exp()).sum();

            // 3. 计算 log_sum_exp = max + log(sum_exp)
            let log_sum_exp = max_logit + sum_exp.ln();

            // 4. 只计算目标 token 的 log_softmax
            if target_token < vocab_size {
                let target_logprob = batch_logits[target_token] - log_sum_exp;
                logprobs.push(target_logprob);

                // // 调试：打印一些样本值
                // if batch_idx < 5 {
                //     println!(
                //         "DEBUG: pos={}, target_token={}, target_logit={:.4}, log_sum_exp={:.4}, logprob={:.4}",
                //         batch_idx,
                //         target_token,
                //         batch_logits[target_token],
                //         log_sum_exp,
                //         target_logprob
                //     );
                // }
            } else {
                println!(
                    "WARNING: target_token {} >= vocab_size {}",
                    target_token, vocab_size
                );
                logprobs.push(f32::NEG_INFINITY); // 无效 token
            }
        }

        // 统计 logprobs 的范围
        if !logprobs.is_empty() {
            let min_logprob = logprobs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_logprob = logprobs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let avg_logprob = logprobs.iter().sum::<f32>() / logprobs.len() as f32;
            println!(
                "DEBUG: Logprobs stats - count: {}, min: {:.4}, max: {:.4}, avg: {:.4}",
                logprobs.len(),
                min_logprob,
                max_logprob,
                avg_logprob
            );
        }

        println!("DEBUG: Computed {} target token logprobs", logprobs.len());
        logprobs
    } else {
        // 原始模式：计算所有词汇的 logprobs（兼容性）
        let mut logprobs = Vec::with_capacity(total_elements);

        for batch_idx in 0..seq_len {
            let start_idx = batch_idx * vocab_size;
            let end_idx = start_idx + vocab_size;
            let batch_logits = &host_logits[start_idx..end_idx];

            // 计算 log_softmax
            // log_softmax(x_i) = log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))

            // 1. 找到最大值以提高数值稳定性
            let max_logit = batch_logits
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // 2. 计算 sum(exp(x_j - max))
            let sum_exp: f32 = batch_logits.iter().map(|&x| (x - max_logit).exp()).sum();

            // 3. 计算 log_sum_exp = max + log(sum_exp)
            let log_sum_exp = max_logit + sum_exp.ln();

            // 4. 计算每个位置的 log_softmax
            for &logit in batch_logits {
                logprobs.push(logit - log_sum_exp);
            }
        }

        println!("DEBUG: Computed {} total logprobs", logprobs.len());
        logprobs
    }
}

/// Store logprobs in global storage for API access
fn store_logprobs(logprobs: Vec<f32>) {
    let storage = LOGPROBS_STORAGE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = storage.lock() {
        *guard = Some(logprobs);
    }
}

/// Retrieve and clear stored logprobs
pub fn take_stored_logprobs() -> Option<Vec<f32>> {
    let storage = LOGPROBS_STORAGE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = storage.lock() {
        guard.take()
    } else {
        None
    }
}

// Mamba 专用的推理引擎参数
const MAMBA_NTOKS: [usize; 7] = [1, 8, 32, 64, 128, 256, 1024];
const MAMBA_CHUNKED_PREFILL_LEN: Option<usize> = None; // 关闭 chunked prefill 以支持 PPL
const MAMBA_MAX_TOKS: usize = 1024;

pub(crate) fn mamba_engine(
    mamba: Mamba<Tensor<&[u8], 2>>,
    gguf: &impl GGufMetaMapExt,
    eos: utok,
    workers: &[(c_int, Option<Arc<super::Progress>>)],
    commands: Receiver<Command>,
    outputs: Sender<Output>,
    use_cuda_graph: bool,
) {
    if let &[(gpu, progress)] = &workers {
        return mamba_mono(
            mamba,
            gguf,
            eos,
            Device::new(*gpu),
            progress.clone(),
            commands,
            outputs,
            use_cuda_graph,
        );
    }

    #[cfg(not(nccl))]
    unreachable!();

    // #[cfg(nccl)]
    // {
    //     use std::collections::HashMap;

    //     let devlist = workers.iter().map(|(gpu, _)| *gpu).collect::<Vec<_>>();
    //     let mut workers = workers.iter().cloned().collect::<HashMap<_, _>>();

    //     let mut comms = CommunicatorGroup::new(&devlist).into_vec().into_iter();
    //     let first = comms.next().unwrap();

    //     let mut mamba = mamba;
    //     let output_head = mamba.output_head.take().unwrap();
    //     let worker = MambaWorker {
    //         dev: first.device(),
    //         dist: Distribution {
    //             start: 0,
    //             len: 1,
    //             total: devlist.len(),
    //         },
    //         progress: workers.remove(&first.device().index()).unwrap(),
    //         config: ModelGroupConfig {
    //             static_model_keys: MAMBA_NTOKS,
    //             dyn_cache_size: 1,
    //             use_cuda_graph,
    //         },
    //         max_toks: MAMBA_MAX_TOKS,
    //         barrier: Some(Arc::new(Barrier::new(devlist.len()))),
    //         task_box: Default::default(),
    //         chunked_prefill_len: MAMBA_CHUNKED_PREFILL_LEN,
    //     };
    //     std::thread::scope(|s| {
    //         let _threads = comms
    //             .map(|comm| {
    //                 let dev = comm.device();
    //                 let dist = Distribution::new(comm.rank(), 1, devlist.len());
    //                 let worker = MambaWorker {
    //                     dev,
    //                     dist,
    //                     progress: workers.remove(&dev.index()).unwrap(),
    //                     ..worker.clone()
    //                 };
    //                 let mamba = mamba.clone();
    //                 s.spawn(move || worker.work(mamba, comm))
    //             })
    //             .collect::<Vec<_>>();

    //         worker.lead(mamba, gguf, eos, output_head, commands, outputs, |ctx| {
    //             Handle::with_comm(ctx, first)
    //         })
    //     })
    // }
}

fn mamba_mono(
    mut mamba: Mamba<Tensor<&[u8], 2>>,
    gguf: &impl GGufMetaMapExt,
    eos: utok,
    dev: Device,
    progress: Option<Arc<super::Progress>>,
    commands: Receiver<Command>,
    outputs: Sender<Output>,
    use_cuda_graph: bool,
) {
    let output_head = mamba.output_head.take().unwrap();
    MambaWorker {
        dev,
        dist: Distribution {
            start: 0,
            len: 1,
            total: 1,
        },
        progress,
        config: ModelGroupConfig {
            static_model_keys: MAMBA_NTOKS,
            dyn_cache_size: 1,
            use_cuda_graph,
        },
        max_toks: MAMBA_MAX_TOKS,
        barrier: None,
        task_box: Default::default(),
        chunked_prefill_len: MAMBA_CHUNKED_PREFILL_LEN,
    }
    .lead(mamba, gguf, eos, output_head, commands, outputs, |ctx| {
        Handle::new(ctx)
    })
}

#[derive(Clone)]
struct MambaWorker<T> {
    dev: Device,
    dist: Distribution,
    progress: Option<Arc<super::Progress>>,
    config: ModelGroupConfig<T>,
    max_toks: usize,
    barrier: Option<Arc<Barrier>>,
    task_box: MambaTaskBox,
    chunked_prefill_len: Option<usize>,
}

type MambaTaskBox = Arc<RwLock<Option<MambaTask>>>;

#[cfg_attr(not(nccl), allow(dead_code))]
#[allow(unused)]
struct MambaTask {
    key: NonZeroUsize,
    reqs: Vec<Req<CacheParts>>,
}

impl<T: IntoIterator<Item = usize>> MambaWorker<T> {
    fn lead(
        self,
        mamba: Mamba<Tensor<&[u8], 2>>,
        gguf: &impl GGufMetaMapExt,
        eos: utok,
        output_head: nn::OutputHead<Tensor<&[u8], 2>>,
        commands: Receiver<Command>,
        outputs: Sender<Output>,
        handle: impl FnOnce(&CurrentCtx) -> Handle,
    ) {
        let Self {
            dev,
            dist,
            progress,
            config,
            max_toks,
            barrier,
            task_box,
            chunked_prefill_len,
        } = self;

        dev.set_mempool_threshold(u64::MAX);
        dev.retain_primary().apply(|ctx| {
            let mut handle = handle(ctx);

            // 初始化 Mamba 模型组
            let mut models = ModelGroupMamba::new(
                mamba,
                dist,
                progress.clone(),
                config,
                &mut handle,
                barrier.as_deref(),
            );

            let mut manager = EngineManager::new(chunked_prefill_len, max_toks);
            let mut output_head = OutputHead::new(output_head, ctx);
            let mut sample_manager = SampleManager::new(output_head.nvoc(), eos, ctx);

            // 初始化 Mamba 缓存
            let n_layer: usize = meta![gguf => llm_block_count];
            let d_inner: usize = 5120; // TODO: 从权重/元数据推导
            let d_conv: usize = 4; // kernel size
            let d_state: usize = 16; // ssm state size

            let mut pages = MemPages::new(dev);
            let mut mamba_cache = MambaCache::new(n_layer, d_inner, d_conv, d_state, &mut pages);

            let max_tok = max_toks;
            let mut fast_embd = FastEmbedding::new(max_tok, ctx);
            let mut pre_kv_pairs = ctx.malloc::<KVPair>(max_tok);

            let stream = ctx.stream();
            let len = max_toks;
            const BUF_LEVEL: usize = 3;
            let mut events: [Event; BUF_LEVEL] = std::array::from_fn(|_| stream.record());
            let mut tok_buf = BufN::<utok>::new(len, BUF_LEVEL, ctx);
            let mut pos_buf = BufN::<upos>::new(len, BUF_LEVEL, ctx);
            let mut out_idx_buf = BufN::<utok>::new(len, BUF_LEVEL, ctx);
            let mut fast_embd_buf = BufN::<(utok, utok)>::new(len, BUF_LEVEL, ctx);

            if outputs.send(Output::Ready).is_ok() {
                while let Ok(removed) = manager.receive(&commands, &outputs) {
                    // 处理已移除会话
                    sample_manager.remove(removed);
                    // 组织请求
                    let Round {
                        overflow,
                        tokens,
                        reqs,
                        sample,
                        output,
                        fast_map,
                        finished,
                    } = manager.prepare();
                    // 处理缓存溢出
                    sample_manager.remove(overflow.iter().map(|s| s.id));
                    if !overflow.is_empty()
                        && outputs.send(Output::Overflow(overflow.into())).is_err()
                    {
                        break;
                    }
                    // 如果不需要推理
                    if tokens.is_empty() {
                        assert!(
                            reqs.is_empty()
                                && sample.is_empty()
                                && output.is_empty()
                                && fast_map.is_empty()
                                && finished.is_empty()
                        );
                        continue;
                    }

                    // println!("DEBUG: originaltokens= {:?}", tokens);
                    // println!("DEBUG: tokens.len()= {:?}", tokens.len());

                    // 更新 host 多级缓存
                    let out_idx = out_idx(&reqs, output.iter().map(|(_, len)| *len));
                    events[out_idx_buf.index()].synchronize();
                    tok_buf.save(&tokens);
                    pos_buf.save(&pos(&reqs));
                    out_idx_buf.save(&out_idx);
                    fast_embd_buf.save(&fast_map);
                    events[out_idx_buf.index()] = stream.record();

                    // 加载输入 - 区分 prefill 和 decode
                    let (key, tok) = if reqs.len() == 1 && reqs[0].seq == tokens.len() {
                        // Prefill 阶段
                        models.load_inputs_mamba_prefill(
                            &mut handle,
                            tokens.len(),
                            &tokens,
                            &stream,
                        )
                    } else if tokens.len() == 1 {
                        // Decode 阶段 (单个 token)
                        models.load_input_mamba_decode(&mut handle, tokens[0], &stream)
                    } else {
                        // 多个 token 但不是完整 prefill，使用 prefill 方法
                        models.load_inputs_mamba_prefill(
                            &mut handle,
                            tokens.len(),
                            &tokens,
                            &stream,
                        )
                    };

                    // 快速启动路径
                    fast_embd.launch(
                        tok,
                        &pre_kv_pairs,
                        &fast_embd_buf[..fast_map.len()],
                        &mut handle,
                        &stream,
                    );
                    let mut input = stream.malloc::<utok>(tok.len() / size_of::<utok>());
                    stream.memcpy_d2d(&mut input, tok);

                    // // 通知协处理单元
                    // #[cfg(nccl)]
                    // if let Some(barrier) = &barrier {
                    //     *task_box.write().unwrap() = Some(MambaTask {
                    //         key,
                    //         reqs: reqs.clone(),
                    //     });
                    //     barrier.wait();
                    //     models.share_inputs(key, &mut handle, &stream);
                    // }

                    // Mamba 推理
                    let x = models.launch_mamba(key, &mut mamba_cache, &mut handle, &stream);

                    // println!("DEBUG: x shape: {:?}", x.shape());

                    // 对于 PPL 计算，我们需要所有位置的 logits（除了最后一个位置）
                    let need_logprobs = should_compute_logprobs(&reqs);
                    let effective_out_idx = if need_logprobs {
                        // PPL 需要计算位置 0..seq_len-1 的 logits 来预测位置 1..seq_len 的 token
                        let seq_len = tokens.len();
                        if seq_len > 1 {
                            (0..seq_len - 1).map(|i| i as utok).collect()
                        } else {
                            out_idx.clone()
                        }
                    } else {
                        out_idx.clone()
                    };

                    // // 如果没有需要计算的位置，则跳过
                    // if effective_out_idx.is_empty() {
                    //     continue;
                    // }

                    // println!("DEBUG: original tokens: {:?}", tokens);
                    // println!("DEBUG: token len: {:?}", tokens.len());
                    // println!("DEBUG: out_idx len: {:?}", out_idx.len());
                    // println!(
                    //     "DEBUG: effective_out_idx len: {:?}",
                    //     effective_out_idx.len()
                    // );
                    // println!("DEBUG: out_idx: {:?}", out_idx);
                    // println!("DEBUG: effective_out_idx: {:?}", effective_out_idx);

                    // let logits_prefill_last = output_head.launch(
                    //     x.clone(),
                    //     // &out_idx_buf[..out_idx.len()],
                    //     &[0], // 打印logits first token
                    //     &mut handle,
                    //     &stream,
                    // );
                    // let logits_prefill_last_vir =
                    //     logits_prefill_last.as_ref().map(|mem| mem.as_ptr().cast());
                    // utils::fmt(&logits_prefill_last_vir, stream.ctx()); // 打印logits_prefill_last

                    // 计算输出头 - 这里可以计算 logprobs
                    let logits = output_head.launch(x, &effective_out_idx, &mut handle, &stream);

                    // 计算真实的 logprobs (log_softmax)
                    if need_logprobs {
                        println!("DEBUG: Computing logprobs...");

                        // 对于 PPL 计算，我们需要目标 token（即输入序列的下一个位置）
                        let target_tokens: Vec<utok> = if tokens.len() > 1 {
                            // 目标 token 是位置 1..seq_len 的 token（用于计算位置 0..seq_len-1 的概率）
                            let targets = tokens[1..].to_vec();
                            // println!("DEBUG: PPL target tokens: {:?}", targets);
                            targets
                        } else {
                            Vec::new()
                        };

                        // 调试：检查 GPU logits tensor 信息
                        // let logits_shape = logits.shape();
                        // println!("DEBUG: GPU logits tensor shape: {:?}", logits_shape);
                        // println!("DEBUG: GPU logits tensor dtype: {:?}", logits.dt());

                        let logprobs = if !target_tokens.is_empty() {
                            compute_log_softmax_on_gpu(&logits, &stream, Some(&target_tokens))
                        } else {
                            compute_log_softmax_on_gpu(&logits, &stream, None)
                        };

                        println!("DEBUG: Computed {} logprobs", logprobs.len());
                        // 将 logprobs 存储到全局存储中
                        store_logprobs(logprobs);
                        println!("DEBUG: Stored logprobs");
                    }

                    // // 跳过采样，创建空的 kv_pairs
                    // let kv_pairs = if logits.dt() == nn::digit_layout::types::F32 {
                    //     println!("Skipping sampling for f32 logits");
                    //     // 创建大小为 0 的 KVPair DevMem
                    //     stream.malloc::<KVPair>(1)
                    // } else {
                    //     // 正常采样
                    //     sample_manager.sample(logits, &input, &sample, &stream)
                    // };

                    // 采样
                    let kv_pairs = sample_manager.sample(logits, &input, &sample, &stream);
                    stream.free(input);
                    stream.memcpy_d2d(&mut pre_kv_pairs[..kv_pairs.len()], &kv_pairs);

                    // 处理推理结束
                    sample_manager.remove(finished.iter().map(|s| s.id));

                    // 生成并发送输出
                    let output = output
                        .into_iter()
                        .filter_map(|(id, len)| if len > 0 { Some((id, len)) } else { None })
                        .collect();
                    let output = Output::Complete {
                        output,
                        kv_pair: kv_pairs.sporulate(),
                        event: stream.record().sporulate(),
                        finished: finished.into(),
                    };
                    if outputs.send(output).is_err() {
                        break;
                    }
                }
            }

            // 通知协处理单元退出
            if let Some(barrier) = &barrier {
                let _ = task_box.write().unwrap().take();
                barrier.wait();
            }

            // 送回存储的会话信息
            for stub in manager.into_stubs() {
                if outputs.send(Output::Removed(stub.session)).is_err() {
                    break;
                }
            }
        })
    }

    // #[cfg(nccl)]
    // fn work(self, mamba: Mamba<Tensor<&[u8], 2>>, comm: Communicator) {
    //     let Self {
    //         dev,
    //         dist,
    //         progress,
    //         config,
    //         max_toks: _max_toks,
    //         barrier,
    //         task_box,
    //         ..
    //     } = self;

    //     let barrier = barrier.unwrap();
    //     dev.set_mempool_threshold(u64::MAX);
    //     dev.retain_primary().apply(|ctx| {
    //         let mut handle = Handle::with_comm(ctx, comm);
    //         let mut models =
    //             ModelGroupMamba::new(mamba, dist, progress, config, &mut handle, Some(&barrier));

    //         let stream = ctx.stream();
    //         loop {
    //             barrier.wait();
    //             match &*task_box.read().unwrap() {
    //                 Some(MambaTask { key, reqs }) => {
    //                     models.share_inputs(*key, &mut handle, &stream);
    //                     // TODO: 需要实现 Mamba 的 launch 方法
    //                     // models.launch_mamba(*key, reqs, &mut handle, &stream);
    //                 }
    //                 None => break,
    //             }
    //         }
    //     })
    // }
}

fn pos<T>(reqs: &[Req<T>]) -> Vec<upos> {
    reqs.iter()
        .flat_map(|req| req.pos..req.pos + req.seq)
        .map(|x| x as _)
        .collect()
}

fn out_idx<T>(reqs: &[Req<T>], outs: impl IntoIterator<Item = usize>) -> Vec<utok> {
    let mut out_idx = Vec::new();

    let mut itok = 0;
    for (req, out) in zip(reqs, outs) {
        for i in req.seq - out..req.seq {
            out_idx.push((itok + i) as _)
        }
        itok += req.seq
    }

    out_idx
}

struct BufN<'ctx, T> {
    buf: HostMem<'ctx>,
    index: usize,
    level: usize,
    _phantom: PhantomData<T>,
}

impl<'ctx, T: Copy> BufN<'ctx, T> {
    fn new(len: usize, level: usize, ctx: &'ctx CurrentCtx) -> Self {
        Self {
            buf: ctx.malloc_host::<T>(len * level),
            index: 0,
            level,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy> BufN<'_, T> {
    fn save(&mut self, data: &[T]) {
        let data = unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) };

        if self.index + 1 == self.level {
            self.index = 0
        } else {
            self.index += 1
        }

        let piece = self.buf.len() / self.level;
        let (data_, padding) = self.buf[self.index * piece..][..piece].split_at_mut(data.len());
        data_.copy_from_slice(data);
        padding.fill(0)
    }

    const fn index(&self) -> usize {
        self.index
    }
}

impl<T> Deref for BufN<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        let piece = self.buf.len() / self.level;
        let (&[], piece, &[]) =
            (unsafe { self.buf[self.index * piece..][..piece].align_to::<T>() })
        else {
            unreachable!()
        };
        piece
    }
}
