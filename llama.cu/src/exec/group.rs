use super::mamba_cache::MambaCache;
use super::{CacheParts, Progress, model::ModelExec, upos};
use crate::{batch::Req, handle::Handle, load::load_weight, memory::MemPages};
use cuda::{DevByte, DevMem, Stream, VirByte};
use nn::{
    Distribution, Graph, GraphBuilder, LLaMA, Mamba, NNGraph, Tensor, TensorMeta,
    digit_layout::types, op,
};
use std::{
    collections::BTreeMap,
    num::{NonZero, NonZeroUsize},
    sync::{Arc, Barrier},
};
use tokeneer::utok;

pub(crate) struct ModelGroup<'ctx> {
    internal: Internal<'ctx>,
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
        dist: Distribution,
        progress: Option<Arc<Progress>>,

        config: ModelGroupConfig<T>,

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
                llama.tensor_parallel(dist),
                [
                    TensorMeta::new(types::U32, ["n_tok".into()]),
                    TensorMeta::new(types::U32, ["n_tok".into()]),
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
            internal, pages, ..
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
                req.cache
                    .update((req.pos + req.seq).div_ceil(32) * 32, pages);
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
            .launch(handle, &reqs, stream)
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
}

fn builder() -> GraphBuilder {
    let mut ans = GraphBuilder::default();
    ans.register_op("embedding", op::embedding::Embedding)
        .register_op("rms-norm", op::normalization::RmsNorm)
        .register_op("linear", op::linear::Linear)
        .register_op("rope", op::rope::Rope)
        .register_op("attention", op::attention::Attention)
        .register_op("swiglu", op::activation::SwiGLU)
        .register_op("concat", op::concat::Concat)
        .register_op("split", op::split::Split)
        .register_op("tile", op::tile::Tile)
        .register_op("merge", op::merge::Merge)
        .register_op("all-reduce", op::all_reduce::AllReduce);
    ans
}

//  Mamba GraphBuilder
fn builder_mamba() -> GraphBuilder {
    let mut ans = GraphBuilder::default();
    ans.register_op("embedding", op::embedding::Embedding)
        .register_op("rms-norm", op::normalization::RmsNorm)
        .register_op("linear", op::linear::Linear)
        .register_op("silu", op::activation::SiLU)
        .register_op("element-mul", op::element_mul::ElementMul)
        .register_op("split", op::split::Split)
        .register_op("mamba-causal-conv1d", op::mamba::CausalConv1d)
        .register_op("mamba-selective-scan", op::mamba::SelectiveScan);
    ans
}

pub(crate) struct ModelGroupMamba<'ctx> {
    internal: Internal<'ctx>,
    pages: MemPages,
    _weight: DevMem<'ctx>,
    next_pos: u32,
}

impl<'ctx> ModelGroupMamba<'ctx> {
    pub fn new<T: IntoIterator<Item = usize>>(
        mamba: Mamba<Tensor<&[u8], 2>>,
        dist: Distribution,
        progress: Option<Arc<Progress>>,
        config: ModelGroupConfig<T>,
        handle: &mut Handle<'ctx>,
        barrier: Option<&Barrier>,
    ) -> Self {
        let ModelGroupConfig {
            static_model_keys,
            mut dyn_cache_size,
            use_cuda_graph,
        } = config;

        let NNGraph(Graph { topo, nodes, edges }) = builder_mamba()
            .build(
                mamba.tensor_parallel(dist),
                [
                    TensorMeta::new(types::U32, ["n_tok".into()]),
                    TensorMeta::new(types::U32, ["n_tok".into()]),
                    TensorMeta::new(types::U32, ["n_tok".into()]),
                ],
            )
            .unwrap();
        handle.ctx.stream().synchronize();

        let dev = handle.ctx.dev();
        let mut pages = MemPages::new(dev);
        let (_weight, edges) = load_weight(edges, progress, handle.ctx);
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

        let internal = Internal::new(graph, static_models, dyn_cache_size);
        Self {
            internal,
            pages,
            _weight,
            next_pos: 0,
        }
    }

    pub fn load_inputs_mamba_prefill(
        &mut self,
        handle: &mut Handle<'ctx>,
        len: usize,
        tok: &[utok],
        stream: &Stream<'ctx>,
    ) -> (NonZeroUsize, &mut [DevByte]) {
        let key = self.internal.get_key(NonZeroUsize::new(len).unwrap());
        let model = self.internal.map_exec(key, handle, &mut self.pages, stream);
        stream.memcpy_h2d(model.tok_buf(), &tok[..key.get()]);
        let pos: Vec<upos> = (0..key.get()).map(|i| i as upos).collect();
        stream.memcpy_h2d(model.pos_buf(), &pos);
        self.next_pos = key.get() as u32;
        let out_idx: Vec<utok> = (0..key.get()).map(|i| i as utok).collect();
        let buf = model.input_buf_at(2);
        stream.memcpy_h2d(buf, &out_idx);
        (key, model.tok_buf())
    }

    pub fn load_input_mamba_decode(
        &mut self,
        handle: &mut Handle<'ctx>,
        tok: utok,
        stream: &Stream<'ctx>,
    ) -> (NonZeroUsize, &mut [DevByte]) {
        let key = self.internal.get_key(NonZeroUsize::new(1).unwrap());
        let model = self.internal.map_exec(key, handle, &mut self.pages, stream);
        let tok_buf = model.tok_buf();
        stream.memcpy_h2d(tok_buf, &[tok]);
        let pos_buf = model.pos_buf();
        let cur = self.next_pos;
        stream.memcpy_h2d(pos_buf, &[cur]);
        // 更新 next_pos
        self.next_pos = cur.saturating_add(1);
        // decode 时 out_idx 固定为 0
        let out_idx_buf = model.input_buf_at(2);
        stream.memcpy_h2d(out_idx_buf, &[0u32]);
        (key, model.tok_buf())
    }

    pub fn launch_mamba(
        &mut self,
        key: NonZeroUsize,
        cache: &mut MambaCache,
        handle: &mut Handle,
        stream: &Stream<'ctx>,
    ) -> Tensor<*const VirByte, 2> {
        let model = self.internal.get_mut(&key).unwrap();
        model.launch_with_mamba_cache(handle, cache, stream)
    }
}
