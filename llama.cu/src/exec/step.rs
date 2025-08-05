use crate::{
    batch::Req,
    handle::Handle,
    op::{self, Operator as _},
    utils::{Blob, destruct, layout, offset_ptr},
};
use ndarray_layout::{ArrayLayout, Endian};
use nn::{Arg, Named, Tensor};
use operators::{
    Operator as _, TensorLayout,
    attention::Args as AttnNoKvArgs,
    attention::common_cpu::Operator as AttnCpu,
    attention_kv_cached::{Args as AttnArgs, cuda::Operator as Attn},
    conv::{Args as ConvArgs, cuda::ConvIm2Col},
    cuda::{CaptureStream, DevByte, GraphExec, Stream, VirByte, memcpy_d2h, memcpy_h2d},
    rearrange::{Args as RearrArgs, cuda::Operator as Rearr},
};
use regex::Regex;
use std::{fmt, sync::LazyLock};

#[allow(dead_code)]
pub(super) enum Step<'ctx> {
    Graph(GraphExec<'ctx>, Box<[Tensor<*const VirByte, 2>]>),
    Attention(Box<Attention>),
    Rearrange(Box<Rearrange>),
    Conv(Box<Conv>),
    Exec(nn::Exec<*const VirByte>),
}

pub(super) struct Rearrange {
    pub dst: Tensor<*const VirByte, 2>,
    pub src: Tensor<*const VirByte, 2>,
}

pub(super) struct Conv {
    pub y: Tensor<*const VirByte, 2>,
    pub x: Tensor<*const VirByte, 2>,
    pub w: Tensor<*const VirByte, 2>,
    pub b: Option<Tensor<*const VirByte, 2>>,
    pub d_patch: usize,
}

pub(super) struct Attention {
    pub iblk: usize,
    pub q: Tensor<*const VirByte, 2>,
    pub k: Tensor<*const VirByte, 2>,
    pub v: Tensor<*const VirByte, 2>,
    pub o: Tensor<*const VirByte, 2>,
}

impl<'ctx> Handle<'ctx> {
    pub(super) fn build_steps(
        &mut self,
        exec: impl IntoIterator<Item = nn::Exec<*const VirByte>>,
        use_cuda_graph: bool,
    ) -> Box<[Step<'ctx>]> {
        let mut stream: Option<CaptureStream<'_>> = None;
        let mut exec_ = Vec::new();
        for exec in exec {
            if exec.node.value.name == "attention" {
                static REGEX: LazyLock<Regex> =
                    LazyLock::new(|| Regex::new(r"^Ω\.blk(\d+)\.attn:attention$").unwrap());

                if let Some(stream) = stream.take() {
                    exec_.push(Step::Graph(
                        self.ctx.instantiate(&stream.end()),
                        Default::default(),
                    ))
                }

                let nn::Exec {
                    node: Named { name, value: op },
                    inputs,
                    outputs,
                } = exec;

                destruct!([q, k, v] = inputs);
                destruct!([o] = outputs);
                let Some(nn::Arg::Int(dh)) = op.arg else {
                    panic!()
                };
                let dh = dh as usize;
                // [n, nh * dh] -> [n, nh, dh] -> [nh, n, dh]
                let transform = |t: Tensor<*const VirByte, 2>| {
                    t.transform(|layout| {
                        layout
                            .tile_be(1, &[layout.shape()[1] / dh, dh])
                            .transpose(&[1, 0])
                    })
                };
                let q = transform(q);
                let k = transform(k);
                let v = transform(v);
                let o = transform(o);

                let iblk = {
                    let (_, [iblk]) = REGEX.captures(&name).unwrap().extract();
                    iblk.parse().unwrap()
                };
                exec_.push(Step::Attention(Box::new(Attention { iblk, q, k, v, o })));
                continue;
            }
            if exec.node.value.name == "merge" {
                if let Some(stream) = stream.take() {
                    exec_.push(Step::Graph(
                        self.ctx.instantiate(&stream.end()),
                        Default::default(),
                    ))
                }

                let nn::Exec {
                    node: Named { name: _, value: _ },
                    inputs,
                    outputs,
                } = exec;

                destruct!([src] = inputs);
                destruct!([dst] = outputs);

                exec_.push(Step::Rearrange(Box::new(Rearrange { dst, src })));
                continue;
            }
            if exec.node.value.name == "conv" {
                if let Some(stream) = stream.take() {
                    exec_.push(Step::Graph(
                        self.ctx.instantiate(&stream.end()),
                        Default::default(),
                    ))
                }

                let nn::Exec {
                    node: Named { name: _, value: op },
                    inputs,
                    outputs,
                } = exec;

                let Some(nn::Arg::Bool(bias)) = op.arg else {
                    panic!()
                };
                let (x, w, b) = if bias {
                    destruct!([x, w, b] = inputs);
                    (x, w, Some(b))
                } else {
                    destruct!([x, w] = inputs);
                    (x, w, None)
                };
                destruct!([y] = outputs);

                exec_.push(Step::Conv(Box::new(Conv {
                    y,
                    x,
                    w,
                    b,
                    d_patch: 14, // todo: from model
                })));
                continue;
            }
            if use_cuda_graph {
                self.launch_nn_exec(
                    &exec,
                    stream.get_or_insert_with(|| self.ctx.stream().capture()),
                )
            } else {
                exec_.push(Step::Exec(exec))
            }
        }
        if let Some(stream) = stream.take() {
            exec_.push(Step::Graph(
                self.ctx.instantiate(&stream.end()),
                Default::default(),
            ))
        }
        exec_.into()
    }

    pub(super) fn launch_nn_exec(&mut self, exec: &nn::Exec<*const VirByte>, stream: &Stream) {
        let nn::Exec {
            node,
            inputs,
            outputs,
        } = exec;
        let op = &node.value;
        macro_rules! launch {
            ($op:ident) => {
                op::$op::launch(
                    self,
                    op.arg.clone(),
                    inputs.clone(),
                    outputs.clone(),
                    &stream,
                )
            };
        }
        match &*op.name {
            "embedding" => {
                if let Some(Arg::Arr(_)) = &op.arg {
                    launch!(EmbeddingQw2vl)
                } else {
                    launch!(Embedding)
                }
            }
            "rms-norm" => launch!(RmsNorm),
            "layer-norm" => launch!(LayerNorm),
            "linear" => launch!(Linear),
            "add" => match inputs[0].shape().len() {
                2 => launch!(Add),
                4 => launch!(Add4d),
                _ => panic!("add: unsupported shape"),
            },
            "rope" => launch!(Rope),
            "mrope" => match inputs[1].shape()[1] {
                2 => launch!(MRope),
                3 => launch!(MRope3d),
                _ => panic!("mrope: unsupported shape"),
            },
            "gelu" => launch!(Gelu),
            "silu" => launch!(Silu),
            "swiglu" => launch!(Swiglu),
            #[cfg(nccl)]
            "all-reduce" => launch!(AllReduce),
            "empty" => {}
            _ => panic!(
                "{}",
                ErrorFmt {
                    name: &node.name,
                    ty: &op.name,
                    arg: &op.arg,
                    inputs,
                    outputs,
                }
            ),
        }
    }

    pub(super) fn launch_attn(
        &mut self,
        op: &Attn,
        attn: &Attention,
        reqs: &[Req<Tensor<*const VirByte, 2>>],
        stream: &Stream,
    ) {
        let Attention { iblk, q, k, v, o } = attn;
        let mut start = 0;
        for req in reqs {
            // [nkvh, 2, nctx, dh]
            let cache = req.cache.clone();
            let cache = cache.transform(|layout| layout.index(1, *iblk));
            let k_cache = cache.clone().transform(|layout| layout.index(1, 0));
            let v_cache = cache.clone().transform(|layout| layout.index(1, 1));
            // [nh, n, dh]
            let len = req.seq;
            let q = q.clone().transform(|layout| layout.slice(1, start, 1, len));
            let k = k.clone().transform(|layout| layout.slice(1, start, 1, len));
            let v = v.clone().transform(|layout| layout.slice(1, start, 1, len));
            let o = o.clone().transform(|layout| layout.slice(1, start, 1, len));
            start += len;
            op.launch(
                &AttnArgs {
                    q_layout: layout(&q),
                    q_base: offset_ptr(&q).cast_mut().cast(),
                    k_layout: layout(&k),
                    k_base: offset_ptr(&k).cast(),
                    v_layout: layout(&v),
                    v_base: offset_ptr(&v).cast(),
                    o_layout: layout(&o),
                    o_base: offset_ptr(&o).cast_mut().cast(),
                    k_cache_layout: layout(&k_cache),
                    k_cache_base: offset_ptr(&k_cache).cast_mut().cast(),
                    v_cache_layout: layout(&v_cache),
                    v_cache_base: offset_ptr(&v_cache).cast_mut().cast(),
                    mask: operators::fuesd_softmax::AttnMask::Causal,
                    pos: req.pos as _,
                },
                &mut [],
                stream,
            )
            .unwrap()
        }
    }

    pub(super) fn launch_attn_qw2vl(
        &mut self,
        op: &AttnCpu,
        attn: &Attention,
        _reqs: &[Req<Tensor<*const VirByte, 2>>],
        _stream: &Stream,
    ) {
        let Attention {
            iblk: _,
            q,
            k,
            v,
            o,
        } = attn;

        // d2h
        let d2h = |tensor: &Tensor<*const VirByte, 2>| {
            let mem_range = tensor.layout().data_range();
            let ptr = tensor.get().cast::<DevByte>();
            let len = *mem_range.end() as usize + tensor.dt().nbytes();
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut host = Blob::new(len);
            memcpy_d2h(&mut host, slice);
            tensor.as_ref().map(|_| host)
        };
        let (q_host, k_host, v_host, o_host) = (d2h(q), d2h(k), d2h(v), d2h(o));
        let [q_, k_, v_, o_] = [&q_host, &k_host, &v_host, &o_host]
            .map(|h| h.as_deref().map(|t| t.as_ptr() as *const VirByte));

        // cpu cal
        op.launch(
            &AttnNoKvArgs {
                q_layout: layout(q),
                q_base: offset_ptr(&q_).cast_mut().cast(),
                k_layout: layout(k),
                k_base: offset_ptr(&k_).cast(),
                v_layout: layout(v),
                v_base: offset_ptr(&v_).cast(),
                o_layout: layout(o),
                o_base: offset_ptr(&o_).cast_mut().cast(),
                mask: operators::fuesd_softmax::AttnMask::None,
            },
            &mut [],
            &operators::common_cpu::ThisThread,
        )
        .unwrap();

        // h2d
        let h2d = |tensor: &Tensor<*const VirByte, 2>, host: &Tensor<*const VirByte, 2>| {
            let mem_range = tensor.layout().data_range();
            let ptr = tensor.get().cast::<DevByte>().cast_mut();
            let len = *mem_range.end() as usize + tensor.dt().nbytes();
            let host = unsafe { std::slice::from_raw_parts(host.get().cast::<u8>(), len) };
            let dev = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            memcpy_h2d(dev, host);
        };
        h2d(o, &o_);
    }

    pub(super) fn launch_conv(
        &mut self,
        op: &ConvIm2Col,
        conv: &Conv,
        _reqs: &[Req<Tensor<*const VirByte, 2>>],
        stream: &Stream,
    ) {
        let Conv {
            y,
            x,
            w,
            b,
            d_patch,
        } = conv;

        op.launch(
            &ConvArgs {
                y_layout: layout(y),
                y_base: offset_ptr(y).cast_mut().cast(),
                x_layout: layout(x),
                x_base: offset_ptr(x).cast(),
                w_layout: layout(w),
                w_base: offset_ptr(w).cast(),
                b_layout: b.as_ref().map(layout),
                b_base: b.as_ref().map(|b| offset_ptr(b).cast()),
                strides: [*d_patch; 2],
                dilations: [1; 2],
                pads: [0; 4],
            },
            &mut [],
            stream,
        )
        .unwrap()
    }

    pub(super) fn launch_rearrange(
        &mut self,
        op: &Rearr,
        rearrange: &Rearrange,
        _reqs: &[Req<Tensor<*const VirByte, 2>>],
        stream: &Stream,
    ) {
        let Rearrange { dst, src } = rearrange;
        op.launch(
            &RearrArgs {
                dst_layout: TensorLayout {
                    dt: src.dt(),
                    layout: ArrayLayout::<2>::new_contiguous(src.shape(), Endian::BigEndian, 2)
                        .to_inline_size(),
                },
                dst_base: offset_ptr(dst).cast_mut().cast(),
                src_layout: layout(src),
                src_base: offset_ptr(src).cast(),
            },
            &mut [],
            stream,
        )
        .unwrap();
    }
}

struct ErrorFmt<'a> {
    name: &'a str,
    ty: &'a str,
    arg: &'a Option<Arg>,
    inputs: &'a [Tensor<*const VirByte, 2>],
    outputs: &'a [Tensor<*const VirByte, 2>],
}

impl fmt::Display for ErrorFmt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Self {
            name,
            ty,
            arg,
            inputs,
            outputs,
        } = self;
        write!(f, "todo! [{ty}] {name} ({arg:?})")?;
        for t in inputs {
            write!(f, " {}{:?}", t.dt(), t.shape())?
        }
        write!(f, " ->")?;
        for t in outputs {
            write!(f, " {}{:?}", t.dt(), t.shape())?
        }
        writeln!(f)
    }
}
