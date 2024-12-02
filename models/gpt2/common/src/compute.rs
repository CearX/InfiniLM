use super::{args::Args, Gpt2Meta};
use gguf::ggml_quants::digit_layout::{types as ty, DigitLayout};
use itertools::izip;
use operators::{
    all_reduce::{self, AllReduce, ReduceOp}, attention_kv_cached::{self, AttnKVCached}, layer_norm::{self, LayerNorm}, mat_mul::{self, MatMul}, mlp::{self, Mlp}, rearrange::{self, Rearrange}, rms_norm::{self, RmsNorm}, rope::{self, Rope, Seq, SinCosTable}, ByteOf, Hardware, LaunchError, Operator, QueueAlloc, QueueOf, TopoNode, Workspace
};
use std::ops::{Deref, DerefMut};
// use tensor::{dt_size, split, Tensor};
use tensor::{split, Tensor};


pub trait Operators {
    type Hardware: Hardware;
    type TopoNode: TopoNode<Self::Hardware>;
    type LayerNorm: LayerNorm<Self::Hardware>;
    type RmsNorm: RmsNorm<Self::Hardware>;
    type MatMul: MatMul<Self::Hardware>;
    type Rope: Rope<Self::Hardware>;
    type AttnKVCached: AttnKVCached<Self::Hardware>;
    type Mlp: Mlp<Self::Hardware>;
    type Rearrange: Rearrange<Self::Hardware>;
    type AllReduce: AllReduce<Self::Hardware, Self::TopoNode>;

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>;

    fn build_sin_cos<QA>(
        dt: DigitLayout,
        nctx: usize,
        dh: usize,
        queue_alloc: &QA,
    ) -> Tensor<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let SinCosTable { nctx, mem } =
            <Self::Rope as Rope<Self::Hardware>>::build_sincos(dt, nctx, dh, queue_alloc);
        Tensor::new(dt, &[2, nctx, dh]).map(|_| mem)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlkWeight {
    AttnNormw,
    AttnNormb,
    AttnQKVw,
    AttnQKVb,
    AttnOw,
    AttnOb,

    FfnNormw,
    FfnNormb,
    FfnUpw,
    FfnUpb,
    FfnDownw,
    FfnDownb,
} 

//weight:cpu -> byte = u8, queue = thisthread
pub trait WeightLoader {    
    type Hardware: Hardware;
    type Memory<'s>: Deref<Target = [ByteOf<Self::Hardware>]> + 's
    where
        Self: 's;

    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        queue: &QueueOf<Self::Hardware>,
    ) -> Self::Memory<'_>;

    fn output_norm_weight(&self, queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_>;
    fn output_norm_bias(&self, queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_>;
    fn output(&self, queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_>;
}
pub struct Gpt2Worker<Ops: Operators, W> {
    meta: Gpt2Meta,
    weights: WeightDecorator<W>,
    layer_norm: Ops::LayerNorm,
    rms_norm: Ops::RmsNorm,
    mat_mul: Ops::MatMul,
    rope: Ops::Rope,
    attn_kv_cached: Ops::AttnKVCached,
    mlp: Ops::Mlp,
    rearrange: Ops::Rearrange,
    all_reduce: Ops::AllReduce,
    residual: bool,
    pub debug: bool,
}
// worker: meta + weight + operators

impl<Ops: Operators, W> Gpt2Worker<Ops, W> {
    pub fn new(node: &Ops::TopoNode, meta: Gpt2Meta, weights: W, residual: bool) -> Self {
        let processor = node.processor();
        Self {
            weights: meta.decorator(weights), // meta.decorator
            meta,
            layer_norm: Ops::LayerNorm::new(processor),
            rms_norm: Ops::RmsNorm::new(processor),
            mat_mul: Ops::MatMul::new(processor),
            rope: Ops::Rope::new(processor),
            attn_kv_cached: Ops::AttnKVCached::new(processor),
            mlp: Ops::Mlp::new(processor),
            rearrange: Ops::Rearrange::new(processor),
            all_reduce: Ops::AllReduce::new(node),
            residual,
            debug: true,
        }
    }

    #[inline]
    pub const fn meta(&self) -> &Gpt2Meta {
        &self.meta
    }
    // meta

    pub fn workspace_size(&self, nt: usize, max_seq_len: usize, max_att_len: usize) -> usize {
        let Gpt2Meta {
            dt_mat,
            nh,
            nkvh,
            d,
            // dh,
            di,
            ..
        } = self.meta;

        // let ele = dt_size(dt_mat);
        let ele = dt_mat.nbytes();
        let embd = nt * d * ele;
        // let qkv = nt * (nh + nkvh + nkvh) * dh * ele;
        let qkv = nt * (nh + nkvh + nkvh) * ele;
        let gate_up = nt * di * 2 * ele;
        // let q = max_seq_len * nh * dh * ele;
        let q = max_seq_len * nh * ele;
        let att = nkvh * max_seq_len * max_att_len * ele;

        embd + qkv.max(gate_up) + q + att
    }
    // 计算一些需要用到的size
}

//launch!!!
impl<Ops, W> Gpt2Worker<Ops, W>
where
    Ops: Operators,
    W: WeightLoader<Hardware = Ops::Hardware>,
    ByteOf<Ops::Hardware>: 'static,
{
    pub fn launch<QA>(
        &mut self,
        args: Args<Ops::Hardware>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        let Args {
            embd,
            sin_cos,
            mut logits,
            mut requests,
            num_tokens: nt,
            max_seq_len,
            max_att_len,
        } = args;

        let Gpt2Meta {
            dt_embd,
            nblk,
            nh,
            nkvh,
            // dh,
            ..
        } = self.meta;
        let beta = if self.residual { 1. } else { 0. };
        let workspace_size = self.workspace_size(nt, max_seq_len, max_att_len);
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size);
        
        let mut x = embd;
        let x1 = Tensor::new(dt_embd, x.shape());
        let (buf, workspace) = workspace.split_at_mut(*x1.get());
        let mut x1 = x1.map(|_| buf);

        // let qkv = Tensor::new(dt_embd, &[nt, (nh + nkvh + nkvh) * dh]);
        let qkv = Tensor::new(dt_embd, &[nt, (nh + nkvh + nkvh) * 64]);

        let sin = sin_cos.clone().index(0, 0);
        let cos = sin_cos.index(0, 1);

        let pos = Tensor::new(ty::U32, &[nt]).map(|_| {
            Ops::Rope::build_pos(
                ty::U32,
                nt,
                requests.iter().map(|req| Seq {
                    pos: req.pos,
                    len: req.seq_len,
                }),
                queue_alloc,
            )
        });

        let req_split = requests.iter().map(|req| req.seq_len).collect::<Vec<_>>();

        let queue = queue_alloc.queue();
        for iblk in 0..nblk {
            {
                let w = self.weights.attn_norm_weight(iblk, queue);
                self.rms_norm(&mut x1, &x, &w, workspace, queue_alloc)?;

                let (buf, workspace) = workspace.split_at_mut(*qkv.get());
                let mut qkv = qkv.clone().map(|_| buf);

                let w = self.weights.attn_qkv_weight(iblk, queue);
                self.mat_mul(&mut qkv, 0., &x1, &w, 1., workspace, queue_alloc)?; // qkv: tensor, workspace: size， queue: thisthread

                // let qkv = qkv.tile(1, &[nh + nkvh + nkvh, dh]); // dh.rope = 64
                let qkv = qkv.tile(1, &[nh + nkvh + nkvh, 64]);

                split!(qkv => q, k, v; [nh, nkvh, nkvh] @ 1);
                let mut q = q;
                let mut k = k;
                let v = v;

                self.rope(&mut q, &pos, &sin, &cos, workspace, queue_alloc)?;
                self.rope(&mut k, &pos, &sin, &cos, workspace, queue_alloc)?;

                {
                    let q = q.map_slice_mut().transpose(&[1, 0]);
                    let k = k.map_slice().transpose(&[1, 0]);
                    let v = v.map_slice().transpose(&[1, 0]);
                    let q = q.split(1, &req_split);
                    let k = k.split(1, &req_split);
                    let v = v.split(1, &req_split);

                    for (mut q, k, v, req) in izip!(q, k, v, &mut requests) {
                        let cache = req
                            .cache
                            .as_mut() // [buf, nblk, 2, nkvh, dh]
                            .index(1, iblk) // [buf, 2, nkvh, dh]
                            .transpose(&[2, 0]) // [nkvh, 2, buf, dh]
                            .map(|t| &mut t[..]);

                        split!(cache => kc, vc; [1, 1] @ 1);
                        let mut o = unsafe { q.map_slice_static_mut() };
                        self.attn_kv_cached(
                            &mut q,
                            &k,
                            &v,
                            &mut o,
                            &mut kc.index(1, 0),
                            &mut vc.index(1, 0),
                            req.pos,
                            workspace,
                            queue_alloc,
                        )?;
                    }
                }

                let o = q.merge(1..3).unwrap();
                drop(w);
                let w = self.weights.attn_o_weight(iblk, queue);
                self.mat_mul(&mut x, beta, &o, &w, 1., workspace, queue_alloc)?;

                self.all_reduce(&mut x, workspace, queue_alloc)?;
            }
            {
                let w = self.weights.ffn_norm_weight(iblk, queue);
                self.rms_norm(&mut x1, &x, &w, workspace, queue_alloc)?;

                self.mlp(&mut x, &x1, iblk, self.residual, workspace, queue_alloc)?;

                self.all_reduce(&mut x, workspace, queue_alloc)?;
            }
        }

        // 集中要采样的 token
        // NOTICE: 输入之前将请求按 seq len 升序排列可降低移动开销
        let mut dst = 0;
        let mut src = 0;
        for req in &requests {
            src += req.seq_len;
            for src in src - req.out_len..src {
                if src != dst {
                    let src = unsafe { x.map_slice_static() }.index(0, src);
                    let mut dst = x.map_slice_mut().index(0, dst);
                    self.rearrange(&mut dst, &src, workspace, queue_alloc)?;
                }
                dst += 1;
            }
        }
        assert_eq!(dst, logits.shape()[0]);

        let w = self.weights.output_norm_weight(queue);
        let mut x = x.map_slice_mut().slice(0, 0, 1, dst);
        let x_ = unsafe { x.map_slice_static() };
        self.rms_norm(&mut x, &x_, &w, workspace, queue_alloc)?;

        let output = self.weights.output_weight(queue);
        self.mat_mul(&mut logits, 0., &x, &output, 1., workspace, queue_alloc)
    }
}

//operators
#[allow(clippy::too_many_arguments)]
impl<Ops, W> Gpt2Worker<Ops, W>
where
    Ops: Operators,
    W: WeightLoader<Hardware = Ops::Hardware>,
{
    fn rms_norm<Y, X, W_, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        w: &Tensor<W_>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        X: Deref<Target = [ByteOf<Ops::Hardware>]>,
        W_: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.rms_norm.launch(
            &rms_norm::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                x_layout: x.layout(),
                x_base: x.base(),
                w_layout: w.layout(),
                w_base: w.base(),
                epsilon: self.meta.epsilon,
            },
            workspace,
            queue_alloc,
        )
    }

    fn layer_norm<Y, X, W_, B, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        s: &Tensor<W_>,
        b: &Tensor<B>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        X: Deref<Target = [ByteOf<Ops::Hardware>]>,
        W_: Deref<Target = [ByteOf<Ops::Hardware>]>,
        B: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.layer_norm.launch(
             &layer_norm::Args {
                y_layout: y.layout(),  
                y_base: y.base_mut(),   
                x_layout: x.layout(), 
                x_base: x.base(),           
                scale_layout: s.layout(), 
                scale_base: s.base(),           
                bias_layout: b.layout(),  
                bias_base: b.base(),           
                epsilon: self.meta.epsilon,    
            },
            workspace,
            queue_alloc,
        )
    }

    fn mat_mul<C, A, B, QA>(
        &self,
        c: &mut Tensor<C>,
        beta: f32,
        a: &Tensor<A>,
        b: &Tensor<B>,
        alpha: f32,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        C: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        A: Deref<Target = [ByteOf<Ops::Hardware>]>,
        B: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: c.layout(),
                c_base: c.base_mut(),
                beta,
                a_layout: a.layout(),
                a_base: a.base(),
                b_layout: b.layout(),
                b_base: b.base(),
                alpha,
            },
            workspace,
            queue_alloc,
        )
    }

    fn rope<T, P, Sin, Cos, QA>(
        &self,
        t: &mut Tensor<T>,
        p: &Tensor<P>,
        sin: &Tensor<Sin>,
        cos: &Tensor<Cos>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        T: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        P: Deref<Target = [ByteOf<Ops::Hardware>]>,
        Sin: Deref<Target = [ByteOf<Ops::Hardware>]>,
        Cos: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.rope.launch(
            &rope::Args {
                t_layout: t.layout(),
                t_base: t.base_mut(),
                p_layout: p.layout(),
                p_base: p.base(),
                sin_layout: sin.layout(),
                sin_base: sin.base(),
                cos_layout: cos.layout(),
                cos_base: cos.base(),
                theta: self.meta.theta,
            },
            workspace,
            queue_alloc,
        )
    }

    fn attn_kv_cached<Q, K, V, O, KC, VC, QA>(
        &self,
        q: &mut Tensor<Q>,
        k: &Tensor<K>,
        v: &Tensor<V>,
        o: &mut Tensor<O>,
        kc: &mut Tensor<KC>,
        vc: &mut Tensor<VC>,
        pos: usize,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Q: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        K: Deref<Target = [ByteOf<Ops::Hardware>]>,
        V: Deref<Target = [ByteOf<Ops::Hardware>]>,
        O: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        KC: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        VC: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.attn_kv_cached.launch(
            &attention_kv_cached::Args {
                q_layout: q.layout(),
                q_base: q.base_mut(),
                k_layout: k.layout(),
                k_base: k.base(),
                v_layout: v.layout(),
                v_base: v.base(),
                o_layout: o.layout(),
                o_base: o.base_mut(),
                k_cache_layout: kc.layout(),
                k_cache_base: kc.base_mut(),
                v_cache_layout: vc.layout(),
                v_cache_base: vc.base_mut(),
                pos: pos.into(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn mlp<Y, X, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        iblk: usize,
        residual: bool,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        X: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        let queue = queue_alloc.queue();
        // let w_gate_up = self.weights.ffn_gate_up(iblk, queue);
        // let w_down = self.weights.ffn_down(iblk, queue);
        let w_gate_up = self.weights.ffn_up_weight(iblk, queue);
        let w_down = self.weights.ffn_down_weight(iblk, queue);

        self.mlp.launch(
            &mlp::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                x_layout: x.layout(),
                x_base: x.base(),
                w_gate_up_layout: w_gate_up.layout(),
                w_gate_up_base: w_gate_up.base(),
                w_down_layout: w_down.layout(),
                w_down_base: w_down.base(),
                down_alpha: 1.,
                residual,
            },
            workspace,
            queue_alloc,
        )
    }

    fn rearrange<Y, X, QA>(
        &self,
        dst: &mut Tensor<Y>,
        src: &Tensor<X>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        X: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: dst.layout(),
                dst_base: dst.base_mut(),
                src_layout: src.layout(),
                src_base: src.base(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn all_reduce<X, QA>(
        &self,
        x: &mut Tensor<X>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        X: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.all_reduce.launch(
            &all_reduce::Args {
                pair: rearrange::Args {
                    dst_layout: x.layout(),
                    dst_base: x.base_mut(),
                    src_layout: x.layout(),
                    src_base: x.base(),
                },
                op: ReduceOp::Sum,
            },
            workspace,
            queue_alloc,
        )
    }
}


struct WeightDecorator<W> {
    attn_norm_weight: Tensor<usize>,
    attn_norm_bias: Tensor<usize>,
    attn_qkv_weight: Tensor<usize>,
    attn_qkv_bias: Tensor<usize>,
    attn_o_weight: Tensor<usize>,
    attn_o_bias: Tensor<usize>,

    ffn_norm_weight: Tensor<usize>,
    ffn_norm_bias: Tensor<usize>,
    ffn_up_weight: Tensor<usize>,
    ffn_up_bias: Tensor<usize>,
    ffn_down_weight: Tensor<usize>,
    ffn_down_bias: Tensor<usize>,
    
    output_norm_weight: Tensor<usize>,
    output_norm_bias: Tensor<usize>,
    output_weight: Tensor<usize>,

    weights: W,
}
// tensor<usize> + weight

impl Gpt2Meta {
    fn decorator<W>(&self, weights: W) -> WeightDecorator<W> {
        use crate::TensorUsage::Computation;
        WeightDecorator {
            attn_norm_weight: self.attn_norm_weight(),
            attn_norm_bias: self.attn_norm_bias(),
            attn_qkv_weight: self.attn_qkv_weight(Computation),
            attn_qkv_bias: self.attn_qkv_bias(),
            attn_o_weight: self.attn_o_weight(Computation),
            attn_o_bias: self.attn_o_bias(),

            ffn_norm_weight: self.ffn_norm_weight(),
            ffn_norm_bias: self.ffn_norm_bias(),
            ffn_up_weight: self.ffn_up_weight(Computation),
            ffn_up_bias: self.ffn_up_bias(),
            ffn_down_weight: self.ffn_down_weight(Computation),
            ffn_down_bias: self.ffn_down_bias(),

            output_norm_weight: self.output_norm_weight(),
            output_norm_bias: self.output_norm_bias(),
            output_weight: self.output_weight(),

            weights,
        }
    }
}
// decorator，new tensor<usize>

// Tensor<usize> -> Tensor<Memory>
impl<W: WeightLoader> WeightDecorator<W> {
    #[inline]
    pub fn attn_norm_weight(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.attn_norm_weight
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::AttnNormw, iblk, queue))
    }

    #[inline]
    pub fn attn_norm_bias(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.attn_norm_bias
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::AttnNormb, iblk, queue))
    }

    #[inline]
    pub fn attn_qkv_weight(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.attn_qkv_weight
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::AttnQKVw, iblk, queue))
    }

    #[inline]
    pub fn attn_qkv_bias(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.attn_qkv_bias
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::AttnQKVb, iblk, queue))
    }

    #[inline]
    pub fn attn_o_weight(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.attn_o_weight
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::AttnOw, iblk, queue))
    }

    #[inline]
    pub fn attn_o_bias(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.attn_o_bias
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::AttnOb, iblk, queue))
    }

    #[inline]
    pub fn ffn_norm_weight(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.ffn_norm_weight
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::FfnNormw, iblk, queue))
    }

    #[inline]
    pub fn ffn_norm_bias(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.ffn_norm_bias
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::FfnNormb, iblk, queue))
    }

    #[inline]
    pub fn ffn_up_weight(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.ffn_up_weight
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::FfnUpw, iblk, queue))
    }
    
    #[inline]
    pub fn ffn_up_bias(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.ffn_up_bias
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::FfnUpb, iblk, queue))
    }

    #[inline]
    pub fn ffn_down_weight(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.ffn_down_weight
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::FfnDownw, iblk, queue))
    }

    #[inline]
    pub fn ffn_down_bias(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.ffn_down_bias
            .clone()
            .map(|_| self.weights.load_blk(BlkWeight::FfnDownb, iblk, queue))
    }

    #[inline]
    pub fn output_norm_weight(&self, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.output_norm_weight
            .clone()
            .map(|_| self.weights.output_norm_weight(queue))
    }

    #[inline]
    pub fn output_norm_bias(&self, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.output_norm_bias
            .clone()
            .map(|_| self.weights.output_norm_bias(queue))
    }

    #[inline]
    pub fn output_weight(&self, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.output_weight.clone().map(|_| self.weights.output(queue))
    }
}