#![cfg(detected_cuda)]

mod parameters;

#[macro_use]
extern crate log;

pub use common_nv::cuda;

use ::half::f16;
use common_nv::{
    cuda::{bindings::CUdeviceptr, memcpy_d2h, DevMem, DevMemSpore},
    slice, udim, utok, DataType, LocalSplitable, NvidiaKernels, NvidiaKernelsPtx, Tensor,
};
use cuda::{AsRaw, Context, ContextResource, ContextSpore, Device, Stream, StreamSpore};
use parameters::{LayerParameter, LayersParameters, ModelParameters};
use std::{
    fs::File,
    io::Read,
    sync::{Arc, Mutex},
    time::Instant,
};
use transformer::{pos, Kernels, LayerBuffer, LayerCache, Llama2, Memory, Request, SampleArgs};

pub struct Transformer {
    host: Memory,
    model: ModelParameters,
    layers: Mutex<LayersParameters>,
    context: Arc<Context>,
    transfer: StreamSpore,
    kernels: NvidiaKernels,
}

impl transformer::Transformer for Transformer {
    type Cache = Cache;

    #[inline]
    fn model(&self) -> &dyn Llama2 {
        &self.host
    }

    fn new_cache(&self) -> Vec<LayerCache<Self::Cache>> {
        self.context.apply(|ctx| {
            let stream = unsafe { self.transfer.sprout(ctx) };
            LayerCache::new_layers(&self.host, |dt, shape| {
                tensor_cache(dt, shape, self.context.clone(), &stream)
            })
        })
    }

    fn decode<Id>(
        &self,
        mut requests: Vec<Request<Id, Self::Cache>>,
    ) -> (Vec<Id>, Tensor<Self::Cache>) {
        // 归拢所有纯解码的请求到前面，减少批量解码的拷贝开销
        requests.sort_unstable_by_key(Request::purely_decode);
        self.context.apply(|ctx| {
            let transfer = unsafe { self.transfer.sprout(ctx) };
            let compute = ctx.stream();
            // 生成词嵌入并预分配空间
            let mut x0 = self.token_embed(&requests, &compute);
            let mut x1 =
                Tensor::alloc(x0.data_type(), x0.shape(), |len| transfer.malloc::<u8>(len));
            let mut buf =
                LayerBuffer::alloc(&self.host, &requests, |len| transfer.malloc::<u8>(len));
            // 生成位置张量
            let nt = x0.shape()[0]; // `nt` for number of tokens
            let pos_ = pos(&requests, nt);
            let pos = Tensor::new(DataType::U32, &[nt], transfer.from_host(&pos_));
            // 推理
            compute.wait_for(&transfer.record());
            {
                // 层参数滚动加载是有状态的，必须由一个控制流独占。其他逻辑无状态，可以多流并发
                let mut layers = self.layers.lock().unwrap();
                for layer in 0..self.host.num_hidden_layers() {
                    let params = {
                        layers.load(layer, &self.host, &transfer);
                        layers.sync(layer, &compute)
                    };

                    let (q, k, v) =
                        self.before_att(params, &x0, &mut x1, &mut buf.qkv, &pos, &compute);
                    let o = &mut x1;
                    self.attention(
                        layer,
                        &mut requests,
                        q,
                        k,
                        v,
                        o,
                        &mut buf.q_buf,
                        &mut buf.att_buf,
                        &compute,
                    );
                    self.after_att(params, &mut x0, &mut x1, &mut buf.gate_up, &compute);
                }
            }
            // 解码
            if requests[0].decode() {
                let x = self.move_decode(&requests, x0, &compute);
                let requests = requests.into_iter().map(Request::id).collect();
                // Sample.sample(sample, requests, self.logits(x, &compute))
                (requests, self.logits(x, &compute))
            } else {
                todo!()
            }
        })
    }

    fn sample<Id>(
        &self,
        args: &SampleArgs,
        requests: Vec<Id>,
        logits: Tensor<Self::Cache>,
    ) -> Vec<(Id, utok)> {
        assert_eq!(logits.data_type(), DataType::F16);
        let &[_, voc] = logits.shape() else { panic!() };

        let mut host = vec![f16::ZERO; logits.size()];
        let Cache { context, mem } = logits.physical();
        context.apply(|ctx| memcpy_d2h(&mut host, unsafe { &mem.sprout(ctx) }));

        requests
            .into_iter()
            .enumerate()
            .map(|(i, id)| (id, args.random(&host[i * voc as usize..][..voc as usize])))
            .collect()
    }
}

type Splitable<'ctx> = LocalSplitable<DevMem<'ctx>>;

impl Transformer {
    pub fn new(config: File, mut safetensors: File, preload_layers: usize, dev: Device) -> Self {
        let context = Arc::new(dev.retain_primary());
        let time = Instant::now();
        let mut host = context.apply(|ctx| {
            ctx.malloc_host::<u8>(safetensors.metadata().unwrap().len() as _)
                .sporulate()
        });
        safetensors.read_exact(&mut host).unwrap();
        drop(safetensors);
        info!("read to host {:?}", time.elapsed());

        let host = Memory::load_safetensors(config, host, false).unwrap();
        let load_layers = preload_layers.min(host.num_hidden_layers());

        let (model, layers, kernels, transfer) = context.apply(|ctx| {
            let stream = ctx.stream();
            let block_size = ctx.dev().max_block_dims().0;
            (
                ModelParameters::new(&host, &stream),
                Mutex::new(LayersParameters::new(load_layers, &host, &stream)),
                NvidiaKernelsPtx::new(&host, block_size).load(ctx),
                stream.sporulate(),
            )
        });

        Self {
            host,
            model,
            layers,
            context,
            transfer,
            kernels,
        }
    }

    fn token_embed<'ctx, Id>(
        &self,
        requests: &[Request<Id, Cache>],
        compute: &Stream<'ctx>,
    ) -> Tensor<DevMem<'ctx>> {
        let dt = self.host.data_type();
        let nt = requests.iter().map(Request::seq_len).sum::<udim>();
        let d = self.host.hidden_size() as udim;
        let kernels = self.kernels.on(compute);

        let mut x0 = Tensor::alloc(dt, &[nt, d], |len| compute.malloc::<u8>(len));
        let tokens = requests.iter().flat_map(Request::tokens).copied();
        kernels.gather(&mut x0, &self.host.embed_tokens(), tokens);
        // compute.synchronize();
        // println!("gather:\n{}", map_tensor(&x0));

        x0
    }

    fn before_att<'ctx>(
        &self,
        params: &LayerParameter,
        x0: &Tensor<DevMem>,
        x1: &mut Tensor<DevMem>,
        qkv: &mut Tensor<Splitable<'ctx>>,
        pos: &Tensor<DevMem>,
        compute: &Stream,
    ) -> (
        Tensor<Splitable<'ctx>>,
        Tensor<Splitable<'ctx>>,
        Tensor<Splitable<'ctx>>,
    ) {
        let nt = x0.shape()[0];
        let d = self.host.hidden_size() as udim;
        let nh = self.host.num_attention_heads() as udim;
        let nkvh = self.host.num_key_value_heads() as udim;
        let dh = d / nh;
        let dkv = nkvh * dh;

        let ctx = compute.ctx();
        let kernels = self.kernels.on(compute);
        let input_layernorm = &params.input_layernorm(ctx);
        let w_qkv = &params.w_qkv(ctx);

        kernels.rms_norm(x1, x0, input_layernorm);
        // compute.synchronize();
        // println!("layer {layer} input norm:\n{}", map_tensor(&x1));

        kernels.mat_mul(qkv, 0., x1, w_qkv, 1.);
        let mut qkv = qkv.split(1, &[d as _, dkv as _, dkv as _]);
        let v = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
        let mut k = qkv.pop().unwrap().reshape(&[nt, nkvh, dh]);
        let mut q = qkv.pop().unwrap().reshape(&[nt, nh, dh]);
        // compute.synchronize();
        // println!("layer {layer} q:\n{}", map_tensor(&q));
        // println!("layer {layer} k:\n{}", map_tensor(&k));
        // println!("layer {layer} v:\n{}", map_tensor(&v));

        kernels.rotary_embedding(&mut q, pos);
        kernels.rotary_embedding(&mut k, pos);
        // compute.synchronize();
        // println!("layer {layer} rot q:\n{}", map_tensor(&q));
        // println!("layer {layer} rot k:\n{}", map_tensor(&k));

        (q, k, v)
    }

    fn attention<Id>(
        &self,
        layer: usize,
        requests: &mut [Request<Id, Cache>],
        q: Tensor<Splitable>,
        k: Tensor<Splitable>,
        v: Tensor<Splitable>,
        o: &mut Tensor<DevMem>,
        q_buf: &mut DevMem,
        att_buf: &mut DevMem,
        compute: &Stream,
    ) {
        let dt = self.host.data_type();
        let nt = o.shape()[0];
        let d = self.host.hidden_size() as udim;
        let nh = self.host.num_attention_heads() as udim;
        let nkvh = self.host.num_key_value_heads() as udim;
        let dh = d / nh;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();

        let ctx = compute.ctx();
        let kernels = self.kernels.on(compute);

        let q = q.as_ref().transpose(&[1, 0, 2]);
        let k = k.as_ref().transpose(&[1, 0, 2]);
        let v = v.as_ref().transpose(&[1, 0, 2]);
        let mut o = o.as_mut().reshape(&[nt, nh, dh]).transpose(&[1, 0, 2]);

        let q = unsafe { q.map_physical(|u| &**u) };
        let k = unsafe { k.map_physical(|u| &**u) };
        let v = unsafe { v.map_physical(|u| &**u) };

        // paged attention
        // 先不支持 batch 进行测试
        let mut req = 0;
        let r = &mut requests[0];
        let pos = r.pos();
        let seq_len = r.seq_len();
        let att_len = r.att_len();

        let req_slice = &[slice![all], slice![from req, take seq_len], slice![all]];
        let cat_slice = &[slice![all], slice![from pos, take seq_len], slice![all]];
        let att_slice = &[slice![all], slice![from   0, take att_len], slice![all]];
        req += seq_len;

        let q = q.clone().slice(req_slice);        
        let k = k.clone().slice(req_slice);
        let v = v.clone().slice(req_slice);
        let o = o.as_mut().slice(req_slice);
        let mut o = unsafe { o.map_physical(|u| &mut ***u) };

        let mut q_att = Tensor::new(dt, &[nh, seq_len, dh], &mut **q_buf);
        let (k_cache, v_cache) = r.cache(layer);
        let mut k_cache = unsafe { k_cache.as_mut().map_physical(|s| s.mem.sprout(ctx)) };
        let mut v_cache = unsafe { v_cache.as_mut().map_physical(|s| s.mem.sprout(ctx)) };

        let k_cat = k_cache.as_mut().slice(cat_slice);
        let v_cat = v_cache.as_mut().slice(cat_slice);
        let mut k_cat = unsafe { k_cat.map_physical(|u| &mut **u) };
        let mut v_cat = unsafe { v_cat.map_physical(|u| &mut **u) };
        kernels.reform(&mut q_att, &q);
        kernels.reform(&mut k_cat, &k);
        kernels.reform(&mut v_cat, &v);

        let q_att = q_att.reshape(&[nkvh, head_group * seq_len, dh]);
        // [nkvh, att_len, head_size]
        let k_att = k_cache.slice(att_slice).transpose(&[0, 2, 1]);
        // [nkvh, att_len, head_size]
        let v_att = v_cache.slice(att_slice);   
        // [att_len, nkvh, head_size]
        let k_att = k_att.transpose(&[2, 0, 1]);
        // [att_len, nkvh, head_size]
        let v_att = v_att.transpose(&[1, 0, 2]);
        let num_requests = requests.len();
        let mut max_seq_len = requests.iter().map(|req| req.tokens().len()).max().unwrap_or(0);
        let mut block_tables = vec![vec![0; max_seq_len]; num_requests];
        let mut block_count = 0;

        let mut seq_lens: Vec<usize> = Vec::new();
        
        for (idx, request) in requests.iter_mut().enumerate() {
            let seq_len = request.tokens().len();
            seq_lens.push(seq_len);
            for i in 0..seq_len {
                block_tables[idx][i] = i + block_count;
            }
            block_count += seq_len;
            if seq_len > max_seq_len {
                max_seq_len = seq_len;
            }
        }        
        let block_tables =  block_tables.into_iter().flat_map(|v|v).collect::<Vec<_>>();
        let block_tables = Tensor::new(DataType::U32, &[num_requests as u32, max_seq_len as u32], compute.from_host(&block_tables));
        let seq_lens = Tensor::new(DataType::U32, &[num_requests as u32], compute.from_host(&seq_lens));
        // let mut o = unsafe { o.map_physical(|u| &mut **u) };
        kernels.paged_attention(&mut o, &q, &k_att, &v_att, nkvh, 1.0, &block_tables, &seq_lens, max_seq_len as u32);

        // let mut req = 0;
        // for r in requests.iter_mut() {
        //     let pos = r.pos();
        //     let seq_len = r.seq_len();
        //     let att_len = r.att_len();

        //     let req_slice = &[slice![all], slice![from req, take seq_len], slice![all]];
        //     let cat_slice = &[slice![all], slice![from pos, take seq_len], slice![all]];
        //     let att_slice = &[slice![all], slice![from   0, take att_len], slice![all]];
        //     req += seq_len;

        //     println!("q.shape: {:?}, k.shape: {:?}, v.shape: {:?}, o.shape: {:?}", q.shape(), k.shape(), v.shape(), o.shape());
        //     let q = q.clone().slice(req_slice);
        //     let k = k.clone().slice(req_slice);
        //     let v = v.clone().slice(req_slice);
        //     let o = o.as_mut().slice(req_slice);
        //     let mut o = unsafe { o.map_physical(|u| &mut ***u) };

        //     let mut q_att = Tensor::new(dt, &[nh, seq_len, dh], &mut **q_buf);
        //     let (k_cache, v_cache) = r.cache(layer);
        //     let mut k_cache = unsafe { k_cache.as_mut().map_physical(|s| s.mem.sprout(ctx)) };
        //     let mut v_cache = unsafe { v_cache.as_mut().map_physical(|s| s.mem.sprout(ctx)) };

        //     let k_cat = k_cache.as_mut().slice(cat_slice);
        //     let v_cat = v_cache.as_mut().slice(cat_slice);
        //     let mut k_cat = unsafe { k_cat.map_physical(|u| &mut **u) };
        //     let mut v_cat = unsafe { v_cat.map_physical(|u| &mut **u) };
        //     kernels.reform(&mut q_att, &q);
            // kernels.reform(&mut k_cat, &k);
        //     kernels.reform(&mut v_cat, &v);

        //     let q_att = q_att.reshape(&[nkvh, head_group * seq_len, dh]);
        //     let k_att = k_cache.slice(att_slice).transpose(&[0, 2, 1]);
        //     let v_att = v_cache.slice(att_slice);
        //     // compute.synchronize();
        //     // println!("layer {layer} q attention:\n{}", map_tensor(&q_att));
        //     // println!("layer {layer} k attention:\n{}", map_tensor(&k_att));
        //     // println!("layer {layer} v attention:\n{}", map_tensor(&v_att));

        //     let shape_att0 = &[nkvh, head_group * seq_len, att_len];
        //     let shape_att1 = &[nkvh * head_group, seq_len, att_len];

        //     let mut att = Tensor::new(dt, shape_att0, &mut **att_buf);
        //     kernels.mat_mul(&mut att, 0., &q_att, &k_att, head_div);
        //     let mut att = att.reshape(shape_att1);
        //     kernels.softmax(&mut att);
        //     let mut x2 = q_att;
        //     let att = att.reshape(shape_att0);
        //     kernels.mat_mul(&mut x2, 0., &att, &v_att, 1.);

        //     kernels.reform(&mut o, &x2.reshape(&[nh, seq_len, dh]));
        //     // compute.synchronize();
        //     // println!("layer {layer} after attention:\n{}", map_tensor(&o));
        // }
    }

    fn after_att(
        &self,
        params: &LayerParameter,
        x0: &mut Tensor<DevMem>,
        x1: &mut Tensor<DevMem>,
        gate_up: &mut Tensor<Splitable>,
        compute: &Stream,
    ) {
        let di = self.host.intermediate_size() as udim;

        let ctx = compute.ctx();
        let kernels = self.kernels.on(compute);
        let w_o = &params.w_o(ctx);
        let post_attention_layernorm = &params.post_attention_layernorm(ctx);
        let mlp_gate_up = &params.mlp_gate_up(ctx);
        let mlp_down = &params.mlp_down(ctx);

        kernels.mat_mul(x0, 1., x1, w_o, 1.);
        // compute.synchronize();
        // println!("layer {layer} o_proj:\n{}", map_tensor(&x0));

        kernels.rms_norm(x1, x0, post_attention_layernorm);
        // compute.synchronize();
        // println!("layer {layer} post norm:\n{}", map_tensor(&x1));

        kernels.mat_mul(gate_up, 0., x1, mlp_gate_up, 1.);
        let mut gate_up = gate_up.split(1, &[di as _, di as _]);
        let up = gate_up.pop().unwrap();
        let mut gate = gate_up.pop().unwrap();
        // compute.synchronize();
        // println!("layer {layer} gate:\n{}", map_tensor(&gate));
        // println!("layer {layer} up:\n{}", map_tensor(&up));

        kernels.swiglu(&mut gate, &up);
        // compute.synchronize();
        // println!("layer {layer} swiglu:\n{}", map_tensor(&gate));

        kernels.mat_mul(x0, 1., &gate, mlp_down, 1.);
        // compute.synchronize();
        // println!("layer {layer} down:\n{}", map_tensor(&x0));
    }

    fn move_decode<'ctx, Id>(
        &self,
        requests: &[Request<Id, Cache>],
        x0: Tensor<DevMem<'ctx>>,
        compute: &Stream,
    ) -> Tensor<DevMem<'ctx>> {
        let buf = x0.physical().as_ptr() as CUdeviceptr;
        let len = self.host.hidden_size() * self.host.data_type().size();

        let (head, others) = requests.split_first().unwrap();
        let begin = head.seq_len() as usize - 1;

        let mut src = begin;
        let mut dst = begin;
        for r in others {
            src += r.seq_len() as usize;
            if r.decode() {
                dst += 1;
                if dst < src {
                    cuda::driver!(cuMemcpyDtoDAsync_v2(
                        buf + (dst * len) as CUdeviceptr,
                        buf + (src * len) as CUdeviceptr,
                        len,
                        compute.as_raw()
                    ));
                }
            }
        }

        x0.slice(&[slice![from begin, until dst + 1], slice![all]])
    }

    fn logits(&self, mut x: Tensor<DevMem>, compute: &Stream) -> Tensor<Cache> {
        let dt = self.host.data_type();
        let voc = self.host.vocab_size() as udim;

        let (model_norm, lm_head) = unsafe { self.model.release(compute) };
        let kernels = self.kernels.on(compute);

        let mut logits = tensor_cache(dt, &[x.shape()[0], voc], self.context.clone(), compute);
        // 复制一个 x 以实现原地归一化
        let x_ = unsafe {
            x.as_ref()
                .map_physical(|u| std::slice::from_raw_parts(u.as_ptr(), u.len()))
        };
        kernels.rms_norm(&mut x, &x_, &model_norm);
        // compute.synchronize();
        // println!("model norm:\n{}", map_tensor(&x));

        kernels.mat_mul(
            &mut unsafe {
                logits
                    .as_mut()
                    .map_physical(|c| c.mem.sprout(compute.ctx()))
            },
            0.,
            &x,
            &lm_head,
            1.,
        );
        // compute.synchronize();
        // println!("model norm:\n{}", map_tensor(&logits));

        logits
    }
}

pub struct Cache {
    pub context: Arc<Context>,
    pub mem: DevMemSpore,
}

impl Drop for Cache {
    #[inline]
    fn drop(&mut self) {
        self.context.apply(|ctx| unsafe { self.mem.kill(ctx) });
    }
}

#[inline]
fn tensor_cache(
    dt: DataType,
    shape: &[udim],
    context: Arc<Context>,
    stream: &Stream,
) -> Tensor<Cache> {
    Tensor::alloc(dt, shape, |l| Cache {
        context,
        mem: stream.malloc::<u8>(l).sporulate(),
    })
}
