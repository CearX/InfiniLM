#![cfg(detected)]

use common::{Contiguous, Distribution, Slab, WeightMemCalculator};
use llama::{LlamaBlkStorage, LlamaBlkWeight, LlamaStorage, Tensor, WeightLoader};
use log::trace;
use operators::{
    all_reduce::{infini::Operator as InfiniAllReduce, AllReduce},
    infini::{Device, InfiniNode},
    infini_rt::{DevBlob, DevByte},
    random_sample::infini::Operator as RandomSampleNpu,
    ByteOf, QueueOf, TopoNode,
};
use std::{
    collections::VecDeque,
    iter::zip,
    marker::PhantomData,
    ops::{Deref, Range},
    time::Instant,
};

pub struct Operators<N = InfiniNode, R = InfiniAllReduce>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Device, RandomSampleNpu>;

macro_rules! op {
    ($name:ident) => {
        operators::$name::infini::Operator
    };
}

impl<N, R> llama::Operators for Operators<N, R>
where
    N: TopoNode<Device>,
    R: AllReduce<Device, N>,
{
    type Hardware = Device;
    type TopoNode = N;
    type RmsNorm = op!(rms_norm);
    type MatMul = op!(mat_mul);
    type Rope = op!(rope);
    type AttnKVCached = op!(attention_kv_cached);
    type Swiglu = op!(swiglu);
    type Rearrange = op!(rearrange);
    type AllReduce = R;

    fn debug<T>(tensor: &Tensor<T>, queue: &QueueOf<Self::Hardware>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        let tensor = tensor.as_ref().map(|s| {
            let mut host = vec![0u8; s.len()];
            queue.get_device().memcpy_d2h(&mut host, s);
            queue.synchronize();
            host
        });
        println!("{tensor}")
    }

    fn memcpy_d2h<T: Copy>(
        dst: &mut [T],
        src: &[ByteOf<Self::Hardware>],
        queue: &QueueOf<Self::Hardware>,
    ) {
        queue.get_device().memcpy_d2h(dst, src)
    }
}

pub struct Weights {
    nexp: usize,
    mem: DevBlob,
    blks: Box<[LlamaBlkStorage<Range<usize>>]>,
    output_norm: Range<usize>,
    output: Range<usize>,
}

impl Weights {
    pub fn new(model: &LlamaStorage<&[u8]>, dist: Distribution, dev: &Device) -> Self {
        let LlamaStorage {
            meta,
            output_norm,
            output,
            blocks,
            ..
        } = model;

        let mut calculator = WeightMemCalculator::new(size_of::<usize>());
        let meta_dist = meta.distribute(dist);
        let blk_size = meta_dist.blk();
        let off_blks = (0..meta_dist.nblk)
            .map(|_| {
                blk_size
                    .clone()
                    .into_vec()
                    .into_iter()
                    .map(|(which, size)| (which, calculator.push(size)))
                    .collect::<LlamaBlkStorage<_>>()
            })
            .collect::<Vec<_>>();
        let off_output_norm = calculator.push(output_norm.len());
        let off_output = calculator.push(output.len());

        let mut mem = dev.malloc::<u8>(calculator.size());
        let mut slab = Slab::<usize, _>::new();
        let mut queue = VecDeque::new();
        let stream = dev.stream();

        macro_rules! host {
            ($l:expr) => {
                slab.take(&$l).unwrap_or_else(|| dev.malloc_host::<u8>($l))
            };
        }

        for (blk, off) in zip(blocks, off_blks.clone()) {
            let blk = blk.clone().into_vec();
            let off = off.into_vec();
            for ((which, data), (which_, off)) in zip(blk, off) {
                assert_eq!(which, which_);
                if off.is_empty() {
                    continue;
                }
                let data = meta.distribute_data(which, data, dist, |l| host!(l));
                let data = match data {
                    Contiguous::Borrowed(data) => {
                        let mut mem = host!(data.len());
                        mem.copy_from_slice(data);
                        mem
                    }
                    Contiguous::Owned(data) => data,
                };
                stream.memcpy_h2d(&mut mem[off], &data);
                let mut event = dev.event();
                stream.record(&mut event);
                queue.push_back((event, Instant::now(), data))
            }

            while let Some((event, _, _)) = queue.front() {
                if event.is_complete() {
                    let (_, time, data) = queue.pop_front().unwrap();
                    trace!("{:>16}bytes copied in {:?}", data.len(), time.elapsed());
                    slab.put(data.len(), data)
                } else {
                    break;
                }
            }
        }
        stream.memcpy_h2d(&mut mem[off_output_norm.clone()], output_norm);
        stream.memcpy_h2d(&mut mem[off_output.clone()], output);

        Self {
            nexp: meta.nexp,
            mem,
            blks: off_blks.into_boxed_slice(),
            output_norm: off_output_norm,
            output: off_output,
        }
    }
}

impl WeightLoader for Weights {
    type Hardware = Device;

    type Weight<'s>
        = &'s [DevByte]
    where
        Self: 's;

    fn load_blk<'a>(
        &'a self,
        which: LlamaBlkWeight,
        iblk: usize,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a> {
        let off = &self.blks[iblk];
        use LlamaBlkWeight as W;
        #[rustfmt::skip]
        let off = match which {
            W::AttnNorm    => &off.attn_norm    ,
            W::AttnQKV     => &off.attn_qkv     ,
            W::AttnQKVBias => &off.attn_qkv_bias,
            W::AttnO       => &off.attn_o       ,
            W::FfnNorm     => &off.ffn_norm     ,
            W::FfnGateInp  => &off.ffn_gate_inp ,
            W::FfnGateUp   => &off.ffn_gate_up  ,
            W::FfnDown     => &off.ffn_down     ,
        };
        &self.mem[off.clone()]
    }

    fn load_moe<'a>(
        &'a self,
        which: LlamaBlkWeight,
        iblk: usize,
        iexp: usize,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a> {
        let off = &self.blks[iblk];
        use LlamaBlkWeight as W;
        #[rustfmt::skip]
        let off = match which {
            W::FfnGateUp => &off.ffn_gate_up,
            W::FfnDown   => &off.ffn_down   ,
            _            => unreachable!()  ,
        };
        let w = &self.mem[off.clone()];
        let one = w.len() / self.nexp;
        &w[iexp * one..][..one]
    }

    fn output_norm<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a> {
        &self.mem[self.output_norm.clone()]
    }

    fn output<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a> {
        &self.mem[self.output.clone()]
    }
}

#[cfg(test)]
mod infer;
