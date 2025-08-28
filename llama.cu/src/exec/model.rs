use super::mamba_cache::MambaCache;
use super::step::Step;
use crate::op::Operator;
use crate::op::conv1d::CausalConv1dUnified;
use crate::op::scan::SelectiveScanWithWriteback;
use crate::{
    batch::Req,
    handle::Handle,
    memory::MemPages,
    utils::{self, destruct},
};
use bytesize::ByteSize;
use cuda::{DevByte, Stream, VirByte, VirMem};
use log::trace;
use nn::{NNGraph, Tensor};
use std::time::Instant;

pub(super) struct ModelExec<'ctx> {
    execs: Box<[Step<'ctx>]>,
    workspace: VirMem,
    inputs: Box<[Tensor<*const VirByte, 2>]>,
    outputs: Box<[Tensor<*const VirByte, 2>]>,
}

impl<'ctx> ModelExec<'ctx> {
    pub fn new(
        graph: NNGraph<Tensor<*const VirByte, 2>>,
        n_tok: usize,
        handle: &mut Handle<'ctx>,
        pages: &mut MemPages,
        use_cuda_graph: bool,
    ) -> Self {
        let graph = graph.lower(&[("n_tok", n_tok)].into(), |t| t);

        let mem_range_map = graph.mem_range_map(8 << 30, 512);

        let mut workspace = pages.reserve_vir(mem_range_map.range.len());
        let ptr = workspace.as_ptr();
        let graph = graph.lower(
            |key| unsafe { ptr.byte_add(mem_range_map.map[&key].start) },
            |&data| data,
        );
        let inputs: Box<[Tensor<*const VirByte, 2>]> = graph
            .0
            .topo
            .global_inputs()
            .map(|i| graph.0.edges[i].clone())
            .collect::<Box<_>>();
        let outputs = graph
            .0
            .topo
            .global_outputs()
            .iter()
            .map(|&i| graph.0.edges[i].clone())
            .collect::<Box<_>>();
        let exec = graph.into_exec();

        // memcpy node 要求当时虚地址有对应的物理页
        pages.map(&mut workspace, ..);

        // 构造 cuda graph
        let time = Instant::now();
        let execs = handle.build_steps(exec, use_cuda_graph);
        trace!(
            "model compiled @{} in {:.2?}, seq len = {n_tok}, workspace = {}",
            handle.ctx.dev().index(),
            time.elapsed(),
            ByteSize::b(workspace.len() as _).display(),
        );

        // 解除映射回收物理页
        pages.unmap(&mut workspace, ..);

        Self {
            execs,
            workspace,
            inputs,
            outputs,
        }
    }
}

impl ModelExec<'_> {
    /// 映射虚页
    pub fn map(&mut self, pages: &mut MemPages) {
        pages.map(&mut self.workspace, ..)
    }

    /// 解映射虚页
    pub fn unmap(&mut self, pages: &mut MemPages) {
        pages.unmap(&mut self.workspace, ..)
    }

    pub fn tok_buf(&mut self) -> &mut [DevByte] {
        as_mapped(&self.inputs[0])
    }

    pub fn pos_buf(&mut self) -> &mut [DevByte] {
        as_mapped(&self.inputs[1])
    }

    /// 获取第 idx 个全局输入缓冲区用于 Mamba
    pub fn input_buf_at(&mut self, idx: usize) -> &mut [DevByte] {
        as_mapped(&self.inputs[idx])
    }

    pub fn launch(
        &mut self,
        handle: &mut Handle,
        reqs: &[Req<Tensor<*const VirByte, 2>>],
        stream: &Stream,
    ) -> Tensor<*const VirByte, 2> {
        // 执行
        for exec in &self.execs {
            match exec {
                Step::Graph(graph, stub) => {
                    stream.launch_graph(graph);
                    if !stub.is_empty() {
                        for t in stub {
                            utils::fmt(t, stream.ctx())
                        }
                        std::process::exit(0);
                    }
                }
                Step::Attention(box_) => handle.launch_attn(box_, reqs, stream),
                Step::Exec(exec) => handle.launch_nn_exec(exec, stream),
            }
        }
        destruct!([x] = self.outputs.clone());
        x
    }

    // 捕获 conv 和 ssm exec, 更新 & 使用 mamba cache
    pub fn launch_with_mamba_cache(
        &mut self,
        handle: &mut Handle,
        cache: &mut MambaCache,
        stream: &Stream,
    ) -> Tensor<*const VirByte, 2> {
        for exec in &self.execs {
            match exec {
                Step::Graph(graph, stub) => {
                    stream.launch_graph(graph);
                    if !stub.is_empty() {
                        for t in stub {
                            utils::fmt(t, stream.ctx())
                        }
                        std::process::exit(0);
                    }
                }
                Step::Attention(_box_) => {}
                Step::Exec(exec) => {
                    fn parse_layer_index(name: &str) -> usize {
                        name.chars()
                            .skip_while(|c| !c.is_ascii_digit())
                            .take_while(|c| c.is_ascii_digit())
                            .collect::<String>()
                            .parse()
                            .unwrap_or_else(|_| {
                                panic!("missing layer index in node name: {}", name)
                            })
                    }

                    match exec.node.value.name.as_str() {
                        "mamba-causal-conv1d" => {
                            let iblk = parse_layer_index(&exec.node.name);
                            let inputs = [
                                exec.inputs[0].clone(),  // x [n,d]
                                exec.inputs[1].clone(),  // w [d,k]
                                exec.inputs[2].clone(),  // b [d]
                                cache.conv_tensor(iblk), // state [d,k]
                            ];
                            let outputs = [exec.outputs[0].clone()];
                            CausalConv1dUnified::launch(handle, None, inputs, outputs, stream);
                        }
                        "mamba-selective-scan" => {
                            let iblk = parse_layer_index(&exec.node.name);
                            let inputs = [
                                exec.inputs[0].clone(), // u [n,d]
                                exec.inputs[1].clone(), // delta [n,d]
                                exec.inputs[2].clone(), // A [d,n_state]
                                exec.inputs[3].clone(), // B [n,n_state]
                                exec.inputs[4].clone(), // C [n,n_state]
                                exec.inputs[5].clone(), // D [d]
                                cache.ssm_tensor(iblk), // state [d,n_state]
                            ];
                            let outputs = [exec.outputs[0].clone()];
                            SelectiveScanWithWriteback::launch(
                                handle, None, inputs, outputs, stream,
                            );
                        }
                        _ => {
                            handle.launch_nn_exec(exec, stream);
                        }
                    }
                }
            }
        }
        destruct!([x] = self.outputs.clone());
        x
    }
}

#[allow(clippy::mut_from_ref)]
fn as_mapped(input: &Tensor<*const VirByte, 2>) -> &mut [DevByte] {
    let ptr = input.get().cast_mut();
    let len = Tensor::use_info(input).take();
    unsafe { std::slice::from_raw_parts_mut(ptr.cast(), len) }
}
