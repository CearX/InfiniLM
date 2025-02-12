use super::{args::Args, ClipMeta};
use itertools::izip;
use operators::{
    add::{self, Add},
    add_rows::{self, AddRows},
    attention::{self, Attention},
    conv::{self, Conv},
    gelu::{self, Gelu},
    layer_norm::{self, LayerNorm},
    mat_mul::{self, MatMul},
    rearrange::{self, Rearrange},
    ByteOf, Hardware, LaunchError, Operator, QueueAlloc, QueueOf, TopoNode, Workspace,
};
use std::{
    ops::{Deref, DerefMut},
    time::Instant,
};
use tensor::{split, Tensor};

pub trait Operators {
    type Hardware: Hardware;
    type TopoNode: TopoNode<Self::Hardware>;
    type Conv: Conv<Self::Hardware>;
    type AddRows: AddRows<Self::Hardware>;
    type LayerNorm: LayerNorm<Self::Hardware>;
    type MatMul: MatMul<Self::Hardware>;
    type Attention: Attention<Self::Hardware>;
    type Gelu: Gelu<Self::Hardware>;
    type Add: Add<Self::Hardware>;
    type Rearrange: Rearrange<Self::Hardware>;

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlkWeight {
    AttnNorm,
    AttnQKV,
    AttnO,
    FfnNorm,
    FfnUp,
    FfnDown,
}

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
    ) -> [Self::Memory<'_>; 2];

    fn patch_embd<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> [Self::Memory<'a>; 2];
    fn pos_embd<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a>;
    fn pre_norm<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Option<[Self::Memory<'a>; 2]>;
    fn post_norm<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>)
        -> Option<[Self::Memory<'a>; 2]>;

    // resampler
    fn r_q<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a>;
    fn r_q_ln_wb<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> [Self::Memory<'a>; 2];
    fn r_kv_w<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a>;
    fn r_kv_ln_wb<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> [Self::Memory<'a>; 2];
    fn r_pos_k<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a>;
    fn r_attn_qkv_wb<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> [Self::Memory<'a>; 2];
    fn r_attn_o_wb<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> [Self::Memory<'a>; 2];
    fn r_ln_post_wb<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> [Self::Memory<'a>; 2];
    fn r_proj<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a>;
}

pub struct ClipWorker<Ops: Operators, W> {
    meta: ClipMeta,
    weights: WeightDecorator<W>,
    conv: Ops::Conv,
    add_rows: Ops::AddRows,
    layer_norm: Ops::LayerNorm,
    mat_mul: Ops::MatMul,
    attention: Ops::Attention,
    gelu: Ops::Gelu,
    add: Ops::Add,
    rearrange: Ops::Rearrange,
    pub debug: bool,
}

impl<Ops: Operators, W> ClipWorker<Ops, W> {
    pub fn new(node: &Ops::TopoNode, meta: ClipMeta, weights: W) -> Self {
        let processor = node.processor();
        Self {
            weights: meta.decorator(weights),
            meta,
            conv: Ops::Conv::new(processor),
            add_rows: Ops::AddRows::new(processor),
            layer_norm: Ops::LayerNorm::new(processor),
            mat_mul: Ops::MatMul::new(processor),
            attention: Ops::Attention::new(processor),
            gelu: Ops::Gelu::new(processor),
            add: Ops::Add::new(processor),
            rearrange: Ops::Rearrange::new(processor),
            debug: true,
        }
    }

    #[inline]
    pub const fn meta(&self) -> &ClipMeta {
        &self.meta
    }

    pub fn workspace_size(&self, np: usize) -> usize {
        let ClipMeta {
            nh, nkvh, dh, di, ..
        } = self.meta;

        let embd = self.meta.embd(np);
        let dt = embd.dt();
        let embd = embd.take();

        let qkv = Tensor::new(dt, &[np * (nh + nkvh + nkvh), dh]).take();
        let q = Tensor::new(dt, &[np, nh, dh]).take();
        let att = Tensor::new(dt, &[nh, np, np]).take();

        let up = Tensor::new(dt, &[np, di]).take();
        embd + (qkv + q + att).max(up)
    }
}

impl<Ops, W> ClipWorker<Ops, W>
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
        let time = Instant::now();
        let Args { raw, pos } = args;
        let ClipMeta {
            dt,
            nblk,
            nh,
            nkvh,
            dh,
            di,
            ..
        } = self.meta;

        let queue = queue_alloc.queue();

        let [k, b] = self.weights.patch_embd(queue);
        let &[n, _, h, w] = raw.shape() else {
            unreachable!()
        };
        let &[m, _, hk, wk] = k.shape() else {
            unreachable!()
        };

        let mut embd = Tensor::new(dt, &[n, m, h / hk, w / wk]).map(|s| queue_alloc.alloc(s));
        self.conv(&mut embd, &raw, &k, &b, workspace, queue_alloc)?;
        drop(k);
        drop(b);

        let embd_ = embd.merge(2..4).unwrap().transpose(&[2, 1]);
        let mut embd = Tensor::new(embd_.dt(), embd_.shape()).map(|s| queue_alloc.alloc(s));
        self.rearrange(&mut embd, &embd_, workspace, queue_alloc)?;

        {
            let pos_embd = self.weights.pos_embd(queue);
            self.add_rows(&mut embd, &pos_embd, &pos, workspace, queue_alloc)?; // embd:[8, 1026, 1152]; pos_embd: [4900, 1152]; pos: [8, 1026]
        }

        let &[batch, size, _] = embd.shape() else {
            unreachable!()
        };
        let batch_split = vec![size; batch];

        let np = batch * size;
        let mut x = embd.merge(0..2).unwrap();
        let x1 = Tensor::new(x.dt(), x.shape());
        let qkv = Tensor::new(x.dt(), &[np, (nh + nkvh + nkvh) * dh]);
        let up = Tensor::new(x.dt(), &[np, di]);

        let workspace_size = self.workspace_size(np);
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size);
        let (buf, workspace) = workspace.split_at_mut(*x1.get());
        let mut x1 = x1.map(|_| buf);

        if let Some(wb) = self.weights.pre_norm(queue) {
            let inplace = unsafe { x.map_slice_static() };
            self.layer_norm(&mut x, &inplace, wb, workspace, queue_alloc)?
        }

        for iblk in 0..nblk {
            {
                let wb = self.weights.attn_norm(iblk, queue);
                self.layer_norm(&mut x1, &x, wb, workspace, queue_alloc)?;

                let (buf, workspace) = workspace.split_at_mut(*qkv.get());
                let mut qkv = qkv.clone().map(|_| buf);

                let [w, b] = self.weights.attn_qkv(iblk, queue);
                self.mat_mul(&mut qkv, &x1, (w, Some(b)), workspace, queue_alloc)?;

                let qkv = qkv.tile(1, &[nh + nkvh + nkvh, dh]);
                split!(qkv => q, k, v; [nh, nkvh, nkvh] @ 1);
                let mut q = q;
                let k = k;
                let v = v;
                {
                    let q = q.map_slice_mut().transpose(&[1, 0]);
                    let k = k.map_slice().transpose(&[1, 0]);
                    let v = v.map_slice().transpose(&[1, 0]);
                    let q = q.split(1, &batch_split);
                    let k = k.split(1, &batch_split);
                    let v = v.split(1, &batch_split);

                    for (mut q, k, v) in izip!(q, k, v) {
                        let mut o = unsafe { q.map_slice_static_mut() };
                        self.attn(&mut q, &k, &v, &mut o, workspace, queue_alloc)?
                    }
                }
                let o = q.map_slice().merge(1..3).unwrap();
                let [w, b] = self.weights.attn_o(iblk, queue);
                self.mat_mul(&mut x1, &o, (w, Some(b)), workspace, queue_alloc)?
            }
            let inplace = unsafe { x.map_slice_static() };
            self.add(&mut x, &inplace, &x1, workspace, queue_alloc)?;

            let wb = self.weights.ffn_norm(iblk, queue);
            self.layer_norm(&mut x1, &x, wb, workspace, queue_alloc)?;
            {
                let (buf, workspace) = workspace.split_at_mut(*up.get());
                let mut up = up.clone().map(|_| buf);

                let [w, b] = self.weights.ffn_up(iblk, queue);
                self.mat_mul(&mut up, &x1, (w, Some(b)), workspace, queue_alloc)?;

                self.gelu(&mut up, workspace, queue_alloc)?;

                let [w, b] = self.weights.ffn_down(iblk, queue);
                self.mat_mul(&mut x1, &up, (w, Some(b)), workspace, queue_alloc)?
            }
            let inplace = unsafe { x.map_slice_static() };
            self.add(&mut x, &inplace, &x1, workspace, queue_alloc)?
        }

        if let Some(wb) = self.weights.post_norm(queue) {
            let inplace = unsafe { x.map_slice_static() };
            self.layer_norm(&mut x, &inplace, wb, workspace, queue_alloc)?
        }
        // println!("x.ln_f().shape(): {:?}", &x.shape()); // 1 * 350 x 560: [1000, 1152]; 4 * 350 x 560: [4000, 1152]
        if self.debug {
            println!("encode {n} x {h} x {w} image in {:?}", time.elapsed())
        }

        // Resampler:
        // image_embd = [1000, 1152] -> [64, 3584], [4000, 3584] -> [256, 3584]

        // config
        // batch = 4; np = 4000;
        let query = self.meta.n_patch(); // 64
        let hidden_size = self.meta.n_mmproj_embd(); // 3584
        let head_dim = 128;
        let num_heads = hidden_size / head_dim; // 28

        // q
        let q = Tensor::new(x.dt(), &[query, hidden_size]);
        let (buf, workspace) = workspace.split_at_mut(*q.get());
        let mut q = q.map(|_| buf);

        let r_q = self.weights.r_q(queue); // [64, 3584];

        // q_ln
        let inplace = unsafe { r_q.map_slice_static() };
        let wb = self.weights.r_q_ln_wb(queue);
        self.layer_norm(&mut q, &inplace, wb, workspace, queue_alloc)?; //  [64, 3584]

        // q_boardcast
        let q_batch = Tensor::new(x.dt(), &[batch, query, hidden_size]);
        let (buf, workspace) = workspace.split_at_mut(*q_batch.get());
        let mut q_batch = q_batch.map(|_| buf);

        let q = q.tile(0, &[1, query]); // [1, 64, 3584]; strides: [458752, 7168, 2]
        let q = q.broadcast(0, batch); // [4, 64, 3584]; strides: [0, 7168, 2]
        self.rearrange(&mut q_batch, &q, workspace, queue_alloc)?;

        let mut q = q_batch.map_slice_mut().merge(0..2).unwrap(); // [4*64, 3584]

        // v = x * weight.kv
        let v = Tensor::new(x.dt(), &[np, hidden_size]); // [4000, 3584];
        let (buf, workspace) = workspace.split_at_mut(*v.get());
        let mut v = v.map(|_| buf);

        let r_kv_w = self.weights.r_kv_w(queue); // [1152, 3584]
        self.mat_mul(&mut v, &x, (r_kv_w, None), workspace, queue_alloc)?; // [4000, 3584] = [4000, 1152] * [1152, 3584]

        // v_ln
        let inplace = unsafe { v.map_slice_static() };
        let wb = self.weights.r_kv_ln_wb(queue);
        self.layer_norm(&mut v, &inplace, wb, workspace, queue_alloc)?;

        // k = v + pos
        let k = Tensor::new(v.dt(), &[np, hidden_size]);
        let (buf, workspace) = workspace.split_at_mut(*k.get());
        let mut k = k.map(|_| buf);

        self.rearrange(&mut k, &v, workspace, queue_alloc)?;

        let mut k = k.tile(0, &[batch, size]); // [8208, 3584] -> [8, 1026, 3584]

        let pos_embd = self.weights.r_pos_k(queue);
        self.add_rows(&mut k, &pos_embd, &pos, workspace, queue_alloc)?; // k:[8, 1026, 3584]; pos_embd: [4900, 3584]; pos: [8, 1026]

        let mut k = k.map_slice_mut().merge(0..2).unwrap(); // K [8208, 3584]

        // attn

        // Q, K, V = weight.qkv.split()
        let [r_qkv_w, r_qkv_b] = self.weights.r_attn_qkv_wb(queue);

        let r_qkv_w = r_qkv_w.map_slice();
        let r_qkv_b = r_qkv_b.map_slice();
        split!(r_qkv_w=> q_w, k_w, v_w; [hidden_size, hidden_size, hidden_size] @ 1); // [3584, 10752] -> [3584, 3584]*3
        split!(r_qkv_b => q_b, k_b, v_b; [1, 1, 1] @ 0);

        // q = q*Q, k = k*K, v = v*V
        let inplace_q = unsafe { q.map_slice_static() };
        let inplace_k = unsafe { k.map_slice_static() };
        let inplace_v = unsafe { v.map_slice_static() };
        self.mat_mul(&mut q, &inplace_q, (q_w, Some(q_b)), workspace, queue_alloc)?; // [256, 3584]
        self.mat_mul(&mut k, &inplace_k, (k_w, Some(k_b)), workspace, queue_alloc)?; // [4000, 3584]
        self.mat_mul(&mut v, &inplace_v, (v_w, Some(v_b)), workspace, queue_alloc)?; // [4000, 3584]

        // tile
        let mut q = q.tile(1, &[num_heads, head_dim]); // [256, 28, 128]
        let k = k.tile(1, &[num_heads, head_dim]); // [4000, 28, 128]
        let v = v.tile(1, &[num_heads, head_dim]); // [4000, 28, 128]

        // attn(qkv)
        let q_batch_split = vec![batch; query];
        {
            let q = q.map_slice_mut().transpose(&[1, 0]); // [28, 256, 128]
            let k = k.map_slice().transpose(&[1, 0]); //  [28, 4000, 128]
            let v = v.map_slice().transpose(&[1, 0]); // [28, 4000, 128]
            let q = q.split(1, &q_batch_split); // [64, 64, 64, 64]
            let k = k.split(1, &batch_split);
            let v = v.split(1, &batch_split);

            for (mut q, k, v) in izip!(q, k, v) {
                let mut o = unsafe { q.map_slice_static_mut() };
                self.attn(&mut q, &k, &v, &mut o, workspace, queue_alloc)?
            }
        }
        let o = q.map_slice().merge(1..3).unwrap(); // [64, 3584]; [64*4, 3584]

        // attn_o
        let [w, b] = self.weights.r_attn_o_wb(queue); // [3584, 3584]; [3584]

        let output = Tensor::new(x.dt(), &[batch * query, hidden_size]); // [64*4, 3584]
        let (buf, workspace) = workspace.split_at_mut(*output.get());
        let mut output = output.map(|_| buf);

        let b = b.tile(0, &[1, hidden_size]); // w = [3584, 3584]; b = [1, 3584]
        self.mat_mul(&mut output, &o, (w, Some(b)), workspace, queue_alloc)?; // output = [64, 3584]; [256, 3584]

        // ln_norm
        let inplace = unsafe { output.map_slice_static() };
        let wb = self.weights.r_ln_post_wb(queue);
        self.layer_norm(&mut output, &inplace, wb, workspace, queue_alloc)?;

        // project
        let inplace = unsafe { output.map_slice_static() };
        let w = self.weights.r_proj_w(queue);
        self.mat_mul(&mut output, &inplace, (w, None), workspace, queue_alloc)?;

        println!("resampler_output.shape(): {:?}", &output.shape()); // [64, 3584]; [256, 2584]

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
impl<Ops, W> ClipWorker<Ops, W>
where
    Ops: Operators,
    W: WeightLoader<Hardware = Ops::Hardware>,
{
    fn conv<Y, X, W_, B, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        w: &Tensor<W_>,
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
        self.conv.launch(
            &conv::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                x_layout: x.layout(),
                x_base: x.base(),
                w_layout: w.layout(),
                w_base: w.base(),
                b_layout: b.layout(),
                b_base: b.base(),
                strides: [self.meta.d_patch; 2],
                dilations: [1; 2],
                pads: [0; 4],
            },
            workspace,
            queue_alloc,
        )
    }

    fn add_rows<Dst, Src, Idx, QA>(
        &self,
        dst: &mut Tensor<Dst>,
        src: &Tensor<Src>,
        idx: &Tensor<Idx>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Dst: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        Src: Deref<Target = [ByteOf<Ops::Hardware>]>,
        Idx: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.add_rows.launch(
            &add_rows::Args {
                dst_layout: dst.layout(),
                dst_base: dst.base_mut(),
                src_layout: src.layout(),
                src_base: src.base(),
                idx_layout: idx.layout(),
                idx_base: idx.base(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn layer_norm<Y, X, WB, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        [w, b]: [Tensor<WB>; 2],
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        X: Deref<Target = [ByteOf<Ops::Hardware>]>,
        WB: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.layer_norm.launch(
            &layer_norm::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                x_layout: x.layout(),
                x_base: x.base(),
                scale_layout: w.layout(),
                scale_base: w.base(),
                bias_layout: b.layout(),
                bias_base: b.base(),
                epsilon: self.meta.epsilon,
            },
            workspace,
            queue_alloc,
        )
    }

    fn mat_mul<C, A, WB, QA>(
        &self,
        c: &mut Tensor<C>,
        a: &Tensor<A>,
        (w, b): (Tensor<WB>, Option<Tensor<WB>>),
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        C: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        A: Deref<Target = [ByteOf<Ops::Hardware>]>,
        WB: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        let beta = if let Some(b) = b {
            let n = c.shape()[0];
            let b = b.broadcast(0, n);
            self.rearrange(c, &b, workspace, queue_alloc)?;
            1.
        } else {
            0.
        };
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: c.layout(),
                c_base: c.base_mut(),
                beta,
                a_layout: a.layout(),
                a_base: a.base(),
                b_layout: w.layout(),
                b_base: w.base(),
                alpha: 1.,
            },
            workspace,
            queue_alloc,
        )
    }

    fn attn<Q, K, V, O, QA>(
        &self,
        q: &mut Tensor<Q>,
        k: &Tensor<K>,
        v: &Tensor<V>,
        o: &mut Tensor<O>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Q: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        K: Deref<Target = [ByteOf<Ops::Hardware>]>,
        V: Deref<Target = [ByteOf<Ops::Hardware>]>,
        O: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.attention.launch(
            &attention::Args {
                q_layout: q.layout(),
                q_base: q.base_mut(),
                k_layout: k.layout(),
                k_base: k.base(),
                v_layout: v.layout(),
                v_base: v.base(),
                o_layout: o.layout(),
                o_base: o.base_mut(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn gelu<X, QA>(
        &self,
        x: &mut Tensor<X>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        X: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.gelu.launch(
            &gelu::Args {
                layout: x.layout(),
                base: x.base_mut(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn add<C, A, B, QA>(
        &self,
        c: &mut Tensor<C>,
        a: &Tensor<A>,
        b: &Tensor<B>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        C: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        A: Deref<Target = [ByteOf<Ops::Hardware>]>,
        B: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.add.launch(
            &add::Args {
                c_layout: c.layout(),
                c_base: c.base_mut(),
                a_layout: a.layout(),
                a_base: a.base(),
                b_layout: b.layout(),
                b_base: b.base(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn rearrange<Dst, Src, QA>(
        &self,
        dst: &mut Tensor<Dst>,
        src: &Tensor<Src>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Dst: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        Src: Deref<Target = [ByteOf<Ops::Hardware>]>,
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
}

struct WeightDecorator<W> {
    patch_embd_w: Tensor<usize>,
    patch_embd_b: Tensor<usize>,
    pos_embd: Tensor<usize>,
    norm: Tensor<usize>,

    attn_qkv_w: Tensor<usize>,
    attn_qkv_b: Tensor<usize>,
    attn_o_w: Tensor<usize>,
    attn_o_b: Tensor<usize>,

    ffn_up_w: Tensor<usize>,
    ffn_up_b: Tensor<usize>,
    ffn_down_w: Tensor<usize>,
    ffn_down_b: Tensor<usize>,

    // resampler
    r_q: Tensor<usize>,
    r_norm: Tensor<usize>,
    r_kv_w: Tensor<usize>,
    r_pos_k: Tensor<usize>,
    r_attn_qkv_w: Tensor<usize>,
    r_attn_qkv_b: Tensor<usize>,
    r_attn_o_w: Tensor<usize>,
    r_attn_o_b: Tensor<usize>,
    r_proj_w: Tensor<usize>,

    weights: W,
}

impl ClipMeta {
    fn decorator<W>(&self, weights: W) -> WeightDecorator<W> {
        WeightDecorator {
            patch_embd_w: self.patch_embd_w(),
            patch_embd_b: self.patch_embd_b(),
            pos_embd: self.pos_embd(),
            norm: self.norm(),

            attn_qkv_w: self.attn_qkv_w(),
            attn_qkv_b: self.attn_qkv_b(),
            attn_o_w: self.attn_o_w(),
            attn_o_b: self.attn_o_b(),
            ffn_up_w: self.ffn_up_w(),
            ffn_up_b: self.ffn_up_b(),
            ffn_down_w: self.ffn_down_w(),
            ffn_down_b: self.ffn_down_b(),

            // resampler
            r_q: self.r_q(),
            r_norm: self.r_norm(),
            r_kv_w: self.r_kv_w(),
            r_pos_k: self.r_pos_k(),
            r_attn_qkv_w: self.r_attn_qkv_w(),
            r_attn_qkv_b: self.r_attn_qkv_b(),
            r_attn_o_w: self.r_attn_o_w(),
            r_attn_o_b: self.r_attn_o_b(),
            r_proj_w: self.r_proj_w(),

            weights,
        }
    }
}

impl<W: WeightLoader> WeightDecorator<W> {
    #[inline]
    pub fn patch_embd<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> [Tensor<W::Memory<'a>>; 2] {
        let [w, b] = self.weights.patch_embd(queue);
        [
            self.patch_embd_w.clone().map(|_| w),
            self.patch_embd_b.clone().map(|_| b),
        ]
    }

    #[inline]
    pub fn pos_embd<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> Tensor<W::Memory<'a>> {
        let pos_embd = self.weights.pos_embd(queue);
        self.pos_embd.clone().map(|_| pos_embd)
    }

    #[inline]
    pub fn pre_norm<'a>(
        &'a self,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Option<[Tensor<W::Memory<'a>>; 2]> {
        self.weights
            .pre_norm(queue)
            .map(|pair| pair.map(|w| self.norm.clone().map(|_| w)))
    }

    #[inline]
    pub fn post_norm<'a>(
        &'a self,
        queue: &'a QueueOf<W::Hardware>,
    ) -> Option<[Tensor<W::Memory<'a>>; 2]> {
        self.weights
            .post_norm(queue)
            .map(|pair| pair.map(|w| self.norm.clone().map(|_| w)))
    }

    pub fn attn_norm(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::AttnNorm, iblk, queue);
        [self.norm.clone().map(|_| w), self.norm.clone().map(|_| b)]
    }

    pub fn attn_qkv(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::AttnQKV, iblk, queue);
        [
            self.attn_qkv_w.clone().map(|_| w),
            self.attn_qkv_b.clone().map(|_| b),
        ]
    }

    pub fn attn_o(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::AttnO, iblk, queue);
        [
            self.attn_o_w.clone().map(|_| w),
            self.attn_o_b.clone().map(|_| b),
        ]
    }

    pub fn ffn_norm(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::FfnNorm, iblk, queue);
        [self.norm.clone().map(|_| w), self.norm.clone().map(|_| b)]
    }

    pub fn ffn_up(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::FfnUp, iblk, queue);
        [
            self.ffn_up_w.clone().map(|_| w),
            self.ffn_up_b.clone().map(|_| b),
        ]
    }

    pub fn ffn_down(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::FfnDown, iblk, queue);
        [
            self.ffn_down_w.clone().map(|_| w),
            self.ffn_down_b.clone().map(|_| b),
        ]
    }

    #[inline]
    pub fn r_q<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> Tensor<W::Memory<'a>> {
        let r_q = self.weights.r_q(queue);
        self.r_q.clone().map(|_| r_q)
    }

    #[inline]
    pub fn r_q_ln_wb<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> [Tensor<W::Memory<'a>>; 2] {
        let [w, b] = self.weights.r_q_ln_wb(queue);
        [
            self.r_norm.clone().map(|_| w),
            self.r_norm.clone().map(|_| b),
        ]
    }

    #[inline]
    pub fn r_kv_w<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> Tensor<W::Memory<'a>> {
        let r_kv_w = self.weights.r_kv_w(queue);
        self.r_kv_w.clone().map(|_| r_kv_w)
    }

    #[inline]
    pub fn r_kv_ln_wb<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> [Tensor<W::Memory<'a>>; 2] {
        let [w, b] = self.weights.r_kv_ln_wb(queue);
        [
            self.r_norm.clone().map(|_| w),
            self.r_norm.clone().map(|_| b),
        ]
    }

    #[inline]
    pub fn r_pos_k<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> Tensor<W::Memory<'a>> {
        let r_pos_k = self.weights.r_pos_k(queue);
        self.r_pos_k.clone().map(|_| r_pos_k)
    }

    #[inline]
    pub fn r_attn_qkv_wb<'a>(
        &'a self,
        queue: &'a QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'a>>; 2] {
        let [w, b] = self.weights.r_attn_qkv_wb(queue);
        [
            self.r_attn_qkv_w.clone().map(|_| w),
            self.r_attn_qkv_b.clone().map(|_| b),
        ]
    }

    #[inline]
    pub fn r_attn_o_wb<'a>(
        &'a self,
        queue: &'a QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'a>>; 2] {
        let [w, b] = self.weights.r_attn_o_wb(queue);
        [
            self.r_attn_o_w.clone().map(|_| w),
            self.r_attn_o_b.clone().map(|_| b),
        ]
    }

    #[inline]
    pub fn r_ln_post_wb<'a>(
        &'a self,
        queue: &'a QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'a>>; 2] {
        let [w, b] = self.weights.r_ln_post_wb(queue);
        [
            self.r_norm.clone().map(|_| w),
            self.r_norm.clone().map(|_| b),
        ]
    }

    #[inline]
    pub fn r_proj_w<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> Tensor<W::Memory<'a>> {
        let r_proj_w = self.weights.r_proj(queue);
        self.r_proj_w.clone().map(|_| r_proj_w)
    }
}
