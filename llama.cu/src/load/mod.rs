mod loader;
mod range_collector;

use crate::exec::Progress;
use bytesize::ByteSize;
use cuda::{CurrentCtx, DevByte, DevMem, Stream, VirByte};
use log::trace;
use nn::{Edge, TPAction, TPTensor, Tensor};
use range_collector::RangeCollector;
use std::{
    collections::HashSet,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::SeqCst},
    },
};

pub(crate) use loader::WeightLoader;

type HostTPTensor<'a> = TPTensor<Tensor<&'a [u8], 2>>;
type VirTensor = Tensor<*const VirByte, 2>;

pub(crate) fn load_weight<'ctx>(
    edges: Box<[Edge<HostTPTensor>]>,
    progress: Option<Arc<Progress>>,
    ctx: &'ctx CurrentCtx,
) -> (DevMem<'ctx>, Box<[Edge<VirTensor>]>) {
    // 排布权重存储
    let align = Some(ctx.dev().alignment())
        .filter(|&n| n > 0)
        .unwrap_or(512);
    let mut ranges = RangeCollector::new(align);
    for nn::Edge { external, .. } in &edges {
        if let Some(nn::External { item, .. }) = external {
            let TPTensor { act, val } = item;
            let len = match act {
                Some(act) => val.get().len() / act.dist.total * act.dist.len,
                None => val.get().len(),
            };
            ranges.insert((act.clone(), val.get().as_ptr()), len)
        }
    }
    if let Some(progress) = &progress {
        progress.weight_size.get_or_init(|| ranges.size());
    }
    // 权重加载
    let mut weight = ctx.malloc::<u8>(ranges.size());
    let mut loader = WeightLoader::new(
        ranges
            .sizes()
            .filter(|&(_, times)| times < 4)
            .map(|(size, _)| size),
    );

    let stream = ctx.stream();
    let loaded = progress.as_ref().map(|p| &p.weight_loaded);
    let mut copied = HashSet::new();
    let edges = edges
        .into_iter()
        .map(|nn::Edge { meta, external }| nn::Edge {
            meta,
            external: external.map(|external| {
                load_exteranl(
                    external,
                    &mut loader,
                    &ranges,
                    &mut weight,
                    &mut copied,
                    loaded,
                    &stream,
                )
            }),
        })
        .collect::<Box<_>>();
    stream.synchronize();
    if let Some(progress) = progress {
        progress
            .weight_loaded
            .store(*progress.weight_size.wait(), SeqCst)
    }
    (weight, edges)
}

fn load_exteranl<'ctx>(
    external: nn::External<TPTensor<Tensor<&[u8], 2>>>,
    loader: &mut WeightLoader<'ctx>,
    ranges: &RangeCollector<(Option<TPAction>, *const u8)>,
    mapped: &mut [DevByte],
    copied: &mut HashSet<Range<usize>>,
    loaded: Option<&AtomicUsize>,
    stream: &Stream<'ctx>,
) -> nn::External<Tensor<*const VirByte, 2>> {
    let nn::External { name, item } = external;
    trace!(
        "loading weight {:>9} @{} {name}",
        ByteSize::b(item.val.get().len() as _).display(),
        stream.ctx().dev().index(),
    );

    let TPTensor { act, val } = item;
    let range = &ranges[&(act.clone(), val.get().as_ptr())];
    let dev = &mut mapped[range.clone()];
    let ptr = dev.as_ptr().cast();
    nn::External {
        name,
        item: match act.clone() {
            Some(TPAction { wt, dist }) => {
                if copied.insert(range.clone()) {
                    let loaded_ = loader.load(dev, stream, |dst| wt.move_data(dist, dst, &val));
                    if let Some(loaded) = loaded {
                        loaded.fetch_add(loaded_, SeqCst);
                    }
                }
                let shape = wt.split_shape(dist, val.shape());
                Tensor::from_dim_slice(val.dt(), &shape).map(|_| ptr)
            }
            None => {
                if copied.insert(range.clone()) {
                    let loaded_ = loader.load(dev, stream, |dst| dst.copy_from_slice(val.get()));
                    if let Some(loaded) = loaded {
                        loaded.fetch_add(loaded_, SeqCst);
                    }
                }
                val.map(|_| ptr)
            }
        },
    }
}
