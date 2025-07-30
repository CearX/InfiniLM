use crate::{
    batch::Req,
    handle::Handle,
    op::{self, ModuleKey, Operator as _},
    utils::{destruct, distinct, offset_ptr, strides},
};
use cuda::{CaptureStream, GraphExec, Module, Stream, VirByte};
use flash_attn::attention::{AttnType, FlashAttnCfg, KVPage, KernelReq, Strides2D};
use ggus::ggml_quants::f16;
use nn::{Arg, Named, Tensor, digit_layout::types};
use regex::Regex;
use std::{fmt, ptr::null, sync::LazyLock};

pub(super) enum Step<'ctx> {
    Graph(GraphExec<'ctx>, Box<[Tensor<*const VirByte, 2>]>),
    Attention(Box<Attention>),
    Exec(nn::Exec<*const VirByte>),
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
            "embedding" => launch!(Embedding),
            "rms-norm" => launch!(RmsNorm),
            "layer-norm" => launch!(LayerNorm),
            "linear" => launch!(Linear),
            "rope" => launch!(Rope),
            "mrope" => launch!(MRope),
            "gelu" => launch!(Gelu),
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
        attn: &Attention,
        reqs: &[Req<Tensor<*const VirByte, 2>>],
        stream: &Stream,
    ) {
        use ::flash_attn::attention::cuda::code as flash_attn_code;
        let Attention { q, k, v, o, .. } = attn;
        let dt = distinct(&[q.dt(), k.dt(), v.dt(), o.dt()]).unwrap();
        let d = *q.shape().last().unwrap();
        // 编译
        let key = [
            ModuleKey::Text("flash-attn"),
            ModuleKey::Type(dt),
            ModuleKey::Size(d),
        ]
        .into_iter();
        match dt {
            types::F16 => {
                let module = self.compile(key.collect(), || {
                    flash_attn_code::<f16>(d, stream.ctx().dev().warp_size())
                });
                launch_attn_typed::<f16>(attn, reqs, module, stream)
            }
            _ => todo!(),
        }
    }
}

fn launch_attn_typed<T: ::flash_attn::attention::cuda::NVDT>(
    attn: &Attention,
    reqs: &[Req<Tensor<*const VirByte, 2>>],
    module: &Module,
    stream: &Stream,
) {
    const TILE_SEQ: usize = 8;
    const TILE_CTX: usize = 32;

    let Attention { iblk, q, k, v, o } = attn;
    // 取参数
    destruct!([nh_q, seq_q, dh_q] = q.shape());
    destruct!([nkvh_k, seq_k, dh_k] = k.shape());
    destruct!([nkvh_v, seq_v, dh_v] = v.shape());
    destruct!([nh_o, seq_o, dh_o] = o.shape());
    let h = *distinct(&[nh_q, nh_o]).unwrap();
    let kvh = *distinct(&[nkvh_k, nkvh_v]).unwrap();
    let _seq = *distinct(&[seq_q, seq_k, seq_v, seq_o]).unwrap();
    let d = *distinct(&[dh_q, dh_k, dh_v, dh_o]).unwrap();
    let cfg = FlashAttnCfg {
        h,
        kvh,
        d,
        tile_seq: TILE_SEQ,
        tile_ctx: TILE_CTX,
    };
    let q_strides = {
        strides!([head, seq, _] = q);
        Strides2D { head, seq }
    };
    let k_strides = {
        strides!([head, seq, _] = k);
        Strides2D { head, seq }
    };
    let v_strides = {
        strides!([head, seq, _] = v);
        Strides2D { head, seq }
    };
    let o_strides = {
        strides!([head, seq, _] = o);
        Strides2D { head, seq }
    };

    // 生成所有页指针
    let cache_pages = reqs
        .iter()
        .flat_map(|req| {
            let Req { cache, pos, seq: n } = req;
            (0..(pos + n).div_ceil(TILE_CTX)).map(|i| {
                let cache = cache
                    .clone()
                    .transform(|layout| layout.index(1, *iblk).index(2, i * TILE_CTX));
                let base = *cache.get();
                let k = cache
                    .clone()
                    .transform(|layout| layout.index(1, 0))
                    .offset();
                let v = cache
                    .clone()
                    .transform(|layout| layout.index(1, 1))
                    .offset();
                KVPage::<T> {
                    k: unsafe { base.byte_offset(k).cast_mut().cast() },
                    v: unsafe { base.byte_offset(v).cast_mut().cast() },
                }
            })
        })
        .collect::<Box<_>>();
    // 为每个请求的每个头生成 block
    let reqs_ = reqs
        .iter()
        .scan((0, 0), |(seq, page), req| {
            let &Req {
                ref cache,
                pos,
                seq: n,
            } = req;
            let kv_strides = {
                strides!([head, _, _, seq, _] = cache);
                Strides2D { head, seq }
            };

            let seq_start = *seq;
            *seq += n;
            let pages_start = *page as _;
            *page += (pos + n).div_ceil(TILE_CTX);

            let q = q
                .clone()
                .transform(|layout| layout.slice(1, seq_start, 1, n));
            let k = k
                .clone()
                .transform(|layout| layout.slice(1, seq_start, 1, n));
            let v = v
                .clone()
                .transform(|layout| layout.slice(1, seq_start, 1, n));
            let o = o
                .clone()
                .transform(|layout| layout.slice(1, seq_start, 1, n));

            Some(KernelReq::<T> {
                q: offset_ptr(&q).cast(),
                q_strides,
                k: offset_ptr(&k).cast(),
                k_strides,
                v: offset_ptr(&v).cast(),
                v_strides,
                pages_start,
                kv_strides,
                o: offset_ptr(&o).cast_mut().cast(),
                o_strides,
                n,
                s: pos + n,
                ty: AttnType::Causal,
                mask: null(),
            })
        })
        .collect::<Box<_>>();

    cfg.compute_cuda::<T>(&cache_pages, &reqs_, module, stream);
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
