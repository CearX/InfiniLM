use super::{CacheParts, Progress, model::ModelExec, upos};
use crate::op::EmbeddingQw2vl;
use crate::{batch::Req, handle::Handle, load::load_weight, memory::MemPages, utils::offset_ptr};
use nn::{
    Distribution, Graph, GraphBuilder, LLaMA, NNGraph, Qwen2VLmmproj, Tensor, TensorMeta,
    digit_layout::types, op,
};
use operators::{
    attention::common_cpu::Operator as AttnCpu,
    attention_kv_cached::cuda::Operator as Attn,
    conv::cuda::ConvIm2Col,
    cuda::{DevByte, DevMem, Stream, VirByte},
    rearrange::cuda::Operator as Rearrange,
};
use std::{
    collections::BTreeMap,
    num::{NonZero, NonZeroUsize},
    sync::{Arc, Barrier},
};
use tokeneer::utok;

pub(crate) struct ModelGroup<'ctx> {
    internal: Internal<'ctx>,
    attn: Attn,
    pages: MemPages,
    _weight: DevMem<'ctx>,
}

#[derive(Clone)]
pub(super) struct ModelGroupConfig<T> {
    pub static_model_keys: T,
    pub dyn_cache_size: usize,
    pub use_cuda_graph: bool,
}

impl<'ctx> ModelGroup<'ctx> {
    pub fn new<T: IntoIterator<Item = usize>>(
        llama: LLaMA<Tensor<&[u8], 2>>,
        multimodal: bool,
        dist: Distribution,
        progress: Option<Arc<Progress>>,

        config: ModelGroupConfig<T>,

        attn: Attn,
        handle: &mut Handle<'ctx>,
        barrier: Option<&Barrier>,
    ) -> Self {
        let ModelGroupConfig {
            static_model_keys,
            mut dyn_cache_size,
            use_cuda_graph,
        } = config;

        // 构建计算图
        let NNGraph(Graph { topo, nodes, edges }) = if multimodal {
            builder()
                .build(
                    llama.tensor_parallel(dist),
                    [
                        TensorMeta::new(types::U32, ["n_tok".into()]),
                        TensorMeta::new(types::U32, ["n_tok".into(), 3.into()]),
                    ],
                )
                .unwrap()
        } else {
            builder()
                .build(
                    llama.tensor_parallel(dist),
                    [
                        TensorMeta::new(types::U32, ["n_tok".into()]),
                        TensorMeta::new(types::U32, ["n_tok".into()]),
                    ],
                )
                .unwrap()
        };

        // 加载权重
        let dev = handle.ctx.dev();
        let mut pages = MemPages::new(dev);
        let (_weight, edges) = load_weight(edges, progress, handle.ctx);
        // 构建 cuda graph
        let graph = NNGraph(Graph { topo, nodes, edges });
        let static_models = if use_cuda_graph {
            static_model_keys
                .into_iter()
                .map(|n_tok| {
                    if let Some(b) = barrier {
                        b.wait();
                    }
                    let key = NonZeroUsize::new(n_tok).unwrap();
                    let exec = ModelExec::new(graph.clone(), n_tok, handle, &mut pages, true);
                    (key, exec)
                })
                .collect::<BTreeMap<_, _>>()
        } else {
            dyn_cache_size += static_model_keys.into_iter().count();
            Default::default()
        };

        let models_with_one_dyn = Internal::new(graph, static_models, dyn_cache_size);
        Self {
            internal: models_with_one_dyn,
            attn,
            pages,
            _weight,
        }
    }

    pub fn load_inputs(
        &mut self,
        handle: &mut Handle<'ctx>,
        len: usize,
        tok: &[utok],
        pos: &[upos],
        stream: &Stream<'ctx>,
    ) -> (NonZeroUsize, &mut [DevByte]) {
        let key = self.internal.get_key(NonZeroUsize::new(len).unwrap());
        let model = self.internal.map_exec(key, handle, &mut self.pages, stream);
        stream.memcpy_h2d(model.tok_buf(), &tok[..key.get()]);
        stream.memcpy_h2d(model.pos_buf(), &pos[..key.get()]);
        (key, model.tok_buf())
    }

    pub fn load_inputs_qw2vl_llm(
        &mut self,
        handle: &mut Handle<'ctx>,
        len: usize,
        tok: &[utok],
        pos: &[upos],
        stream: &Stream<'ctx>,
    ) -> (NonZeroUsize, &mut [DevByte]) {
        let key = self.internal.get_key(NonZeroUsize::new(len).unwrap());
        let model = self.internal.map_exec(key, handle, &mut self.pages, stream);
        stream.memcpy_h2d(model.tok_buf(), &tok[..key.get()]);
        stream.memcpy_h2d(model.pos_buf(), &pos[..key.get() * 3]);
        (key, model.tok_buf())
    }

    #[cfg(nccl)]
    pub fn share_inputs(
        &mut self,
        key: NonZeroUsize,
        handle: &mut Handle<'ctx>,
        stream: &Stream<'ctx>,
    ) {
        let model = self.internal.map_exec(key, handle, &mut self.pages, stream);
        if let Some(comm) = &handle.comm {
            comm.broadcast(model.tok_buf(), None, 0, stream);
            comm.broadcast(model.pos_buf(), None, 0, stream);
        }
    }

    pub fn launch(
        &mut self,
        key: NonZeroUsize,
        reqs: &[Req<CacheParts>],
        handle: &mut Handle,
        stream: &Stream<'ctx>,
    ) -> Tensor<*const VirByte, 2> {
        let Self {
            internal,
            attn,
            pages,
            ..
        } = self;

        let mut reqs = reqs
            .iter()
            .map(|req| Req {
                cache: req.cache.0[handle.rank()].lock().unwrap(),
                pos: req.pos,
                seq: req.seq,
            })
            .collect::<Vec<_>>();
        let reqs = reqs
            .iter_mut()
            .map(|req| {
                req.cache.update(req.pos + req.seq, pages);
                Req {
                    cache: req.cache.as_tensor(),
                    pos: req.pos,
                    seq: req.seq,
                }
            })
            .collect::<Vec<_>>();

        internal
            .get_mut(&key)
            .unwrap()
            .launch(attn, handle, &reqs, stream)
    }
}

pub(crate) struct ModelGroupQw2vl<'ctx> {
    internal: Internal<'ctx>,
    attn: AttnCpu,
    conv: ConvIm2Col,
    rearrange: Rearrange,
    pages: MemPages,
    _weight: DevMem<'ctx>,
}

impl<'ctx> ModelGroupQw2vl<'ctx> {
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn new<T: IntoIterator<Item = usize>>(
        qw2vl: Qwen2VLmmproj<Tensor<&[u8], 2>>,
        image_shape: [usize; 4],
        d_patch: usize,
        dist: Distribution,
        progress: Option<Arc<Progress>>,
        config: ModelGroupConfig<T>,

        attn: AttnCpu,
        conv: ConvIm2Col,
        rearrange: Rearrange,
        handle: &mut Handle<'ctx>,
        barrier: Option<&Barrier>,
    ) -> Self {
        let ModelGroupConfig {
            static_model_keys,
            mut dyn_cache_size,
            use_cuda_graph,
        } = config;
        // 构建计算图
        let NNGraph(Graph { topo, nodes, edges }) = builder()
            .build(
                qw2vl.tensor_parallel(dist),
                [
                    TensorMeta::new(
                        types::F16,
                        [
                            "n_img".into(),
                            "channels".into(),
                            "h_img".into(),
                            "w_img".into(),
                        ],
                    ),
                    TensorMeta::new(types::U32, ["patches".into(), 2.into()]),
                ],
            )
            .unwrap();
        // 加载权重
        let dev = handle.ctx.dev();
        let mut pages = MemPages::new(dev);
        let (_weight, edges) = load_weight(edges, progress, handle.ctx);
        // 构建 cuda graph
        let graph = NNGraph(Graph { topo, nodes, edges });
        let static_models = if use_cuda_graph {
            static_model_keys
                .into_iter()
                .map(|n_tok| {
                    if let Some(b) = barrier {
                        b.wait();
                    }
                    let key = NonZeroUsize::new(n_tok).unwrap();
                    let exec = ModelExec::new_qw2vl(
                        graph.clone(),
                        image_shape[0],
                        image_shape[1],
                        image_shape[2],
                        image_shape[3],
                        d_patch,
                        handle,
                        &mut pages,
                        false,
                    );
                    (key, exec)
                })
                .collect::<BTreeMap<_, _>>()
        } else {
            dyn_cache_size += static_model_keys.into_iter().count();
            Default::default()
        };

        let models_with_one_dyn = Internal::new(graph, static_models, dyn_cache_size);
        Self {
            internal: models_with_one_dyn,
            attn,
            rearrange,
            conv,
            pages,
            _weight,
        }
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn load_inputs_qw2vl_mmproj(
        &mut self,
        handle: &mut Handle<'ctx>,
        image_len: usize,
        pos_len: usize,
        image_buf: &[u8],
        pos_buf: &[upos],
        image_shape: [usize; 4],
        d_patch: usize,
        stream: &Stream<'ctx>,
    ) -> (NonZeroUsize, &mut [DevByte]) {
        let key = self.internal.get_key(NonZeroUsize::new(image_len).unwrap());
        let model = self.internal.map_exec_qw2vl(
            key,
            handle,
            &mut self.pages,
            stream,
            image_shape,
            d_patch,
        );
        stream.memcpy_h2d(model.tok_buf(), &image_buf[..key.get()]);
        stream.memcpy_h2d(model.pos_buf(), &pos_buf[..pos_len]);
        (key, model.tok_buf())
    }

    #[allow(dead_code)]
    pub fn launch(
        &mut self,
        key: NonZeroUsize,
        reqs: &[Req<CacheParts>],
        handle: &mut Handle,
        stream: &Stream<'ctx>,
    ) -> Tensor<*const VirByte, 2> {
        let Self {
            internal,
            attn,
            conv,
            rearrange,
            pages,
            ..
        } = self;

        let mut reqs = reqs
            .iter()
            .map(|req| Req {
                cache: req.cache.0[handle.rank()].lock().unwrap(),
                pos: req.pos,
                seq: req.seq,
            })
            .collect::<Vec<_>>();
        let reqs = reqs
            .iter_mut()
            .map(|req| {
                req.cache.update(req.pos + req.seq, pages);
                Req {
                    cache: req.cache.as_tensor(),
                    pos: req.pos,
                    seq: req.seq,
                }
            })
            .collect::<Vec<_>>();

        internal
            .get_mut(&key)
            .unwrap()
            .launch_qw2vl(attn, conv, rearrange, handle, &reqs, stream)
    }
}
struct Internal<'ctx> {
    static_models: BTreeMap<NonZeroUsize, ModelExec<'ctx>>,
    dyn_model_cache: lru::LruCache<NonZeroUsize, ModelExec<'ctx>>,
    mapped: Option<NonZeroUsize>,
    graph: NNGraph<Tensor<*const VirByte, 2>>,
}

impl<'ctx> Internal<'ctx> {
    fn new(
        graph: NNGraph<Tensor<*const VirByte, 2>>,
        static_models: BTreeMap<NonZeroUsize, ModelExec<'ctx>>,
        dyn_cache_size: usize,
    ) -> Self {
        const ONE: NonZeroUsize = NonZeroUsize::new(1).unwrap();
        Self {
            static_models,
            dyn_model_cache: lru::LruCache::new(NonZeroUsize::new(dyn_cache_size).unwrap_or(ONE)),
            mapped: None,
            graph,
        }
    }

    /// 获取不小于 `len` 的最小模型索引，如果不存在，则返回 `len`。
    fn get_key(&self, len: NonZero<usize>) -> NonZeroUsize {
        self.static_models
            .range(len..)
            .next()
            .map_or(len, |(k, _)| *k)
    }

    fn get_mut(&mut self, key: &NonZero<usize>) -> Option<&mut ModelExec<'ctx>> {
        self.static_models
            .get_mut(key)
            .or_else(|| self.dyn_model_cache.get_mut(key))
    }

    fn map_exec(
        &mut self,
        key: NonZero<usize>,
        handle: &mut Handle<'ctx>,
        pages: &mut MemPages,
        stream: &Stream<'ctx>,
    ) -> &mut ModelExec<'ctx> {
        // 检查当前映射的模型
        if let Some(mapped) = self.mapped {
            if mapped == key {
                return self.get_mut(&key).unwrap();
            }
            // 当前映射的模型不是要映射的模型，解映射
            if let Some(mapped) = self.get_mut(&mapped) {
                stream.synchronize();
                mapped.unmap(pages)
            }
        }
        let Self {
            static_models,
            dyn_model_cache,
            mapped,
            graph,
        } = self;
        // 更新记录
        *mapped = Some(key);
        // 查找或新建模型
        let model = static_models.get_mut(&key).unwrap_or_else(|| {
            dyn_model_cache.get_or_insert_mut(key, || {
                log::info!("create modelExec for key {}", key.get());
                ModelExec::new(graph.clone(), key.get(), handle, pages, false)
            })
        });
        // 建立映射
        model.map(pages);
        model
    }

    fn map_exec_qw2vl(
        &mut self,
        key: NonZero<usize>,
        handle: &mut Handle<'ctx>,
        pages: &mut MemPages,
        stream: &Stream<'ctx>,
        image_shape: [usize; 4],
        d_patch: usize,
    ) -> &mut ModelExec<'ctx> {
        // 检查当前映射的模型
        if let Some(mapped) = self.mapped {
            if mapped == key {
                return self.get_mut(&key).unwrap();
            }
            // 当前映射的模型不是要映射的模型，解映射
            if let Some(mapped) = self.get_mut(&mapped) {
                stream.synchronize();
                mapped.unmap(pages)
            }
        }
        let Self {
            static_models,
            dyn_model_cache,
            mapped,
            graph,
        } = self;
        // 更新记录
        *mapped = Some(key);
        // 查找或新建模型
        let model = static_models.get_mut(&key).unwrap_or_else(|| {
            dyn_model_cache.get_or_insert_mut(key, || {
                log::info!("create modelExec for key {}", key.get());
                ModelExec::new_qw2vl(
                    graph.clone(),
                    image_shape[0],
                    image_shape[1],
                    image_shape[2],
                    image_shape[3],
                    d_patch,
                    handle,
                    pages,
                    false,
                )
            })
        });
        // 建立映射
        model.map(pages);
        model
    }
}

fn builder() -> GraphBuilder {
    let mut ans = GraphBuilder::default();
    ans.register_op("embedding", op::embedding::Embedding)
        // .register_op("qw2vl-embedding", Qw2vlEmbedding)
        .register_op("add", op::add::Add)
        .register_op("conv", op::conv::Conv)
        .register_op("layer-norm", op::normalization::LayerNorm)
        .register_op("rms-norm", op::normalization::RmsNorm)
        .register_op("linear", op::linear::Linear)
        .register_op("rope", op::rope::Rope)
        .register_op("mrope", op::mrope::Mrope)
        .register_op("attention", op::attention::Attention)
        .register_op("gelu", op::activation::GeLU)
        .register_op("silu", op::activation::SiLU)
        .register_op("swiglu", op::activation::SwiGLU)
        .register_op("concat", op::concat::Concat)
        .register_op("split", op::split::Split)
        .register_op("tile", op::tile::Tile)
        .register_op("merge", op::merge::Merge)
        .register_op("transpose", op::transpose::Transpose)
        .register_op("all-reduce", op::all_reduce::AllReduce);
    ans
}
