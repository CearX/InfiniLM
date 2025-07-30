//! <https://zhuanlan.zhihu.com/p/667025336>

use crate::utils::offset_ptr;
use cuda::{CurrentCtx, DevByte, DevMem, Module, Rtc, Stream, VirByte, params};
use nn::Tensor;
use std::ffi::c_uint;
use tokeneer::utok;

pub(crate) struct LogitsModifier<'ctx> {
    module: Module<'ctx>,
    n: usize,
    eos: utok,
}

impl<'ctx> LogitsModifier<'ctx> {
    pub fn new(n: usize, eos: utok, ctx: &'ctx CurrentCtx) -> Self {
        Self {
            module: Self::compile(ctx),
            n,
            eos,
        }
    }
}

impl LogitsModifier<'_> {
    pub fn new_state<'ctx>(&self, stream: &Stream<'ctx>) -> DevMem<'ctx> {
        stream.malloc::<u32>(self.n)
    }

    pub unsafe fn next<const N: usize>(
        &self,
        logits: &Tensor<*const VirByte, N>,
        records: *mut DevByte,
        tok: *const DevByte,
        mut temperature: f32,
        penalty: f32,
        stream: &Stream,
    ) {
        let n = self.n as c_uint;
        if temperature == 0. {
            temperature = 1.
        }
        stream.launch(
            &self.module.get_kernel(c"next"),
            (n.div_ceil(256), 256, 0),
            &params![
                offset_ptr(logits),
                records,
                n,
                self.eos,
                temperature,
                penalty,
                tok
            ]
            .to_ptrs(),
        );
    }

    fn compile<'ctx>(ctx: &'ctx CurrentCtx) -> Module<'ctx> {
        const CODE: &str = include_str!("modify.cuh");
        let code = format!(
            r#"{CODE}

extern "C" __global__ void next(
    // 采样分布和状态
    half *logits,          // 概率分布
    unsigned int *records, // 每个 token 的出现次数
    // 词表信息
    unsigned int const n,   // 词表长度
    unsigned int const eos, // 结束符
    // 采样参数
    float const temperature, // 温度
    float const penalty,     // 重复惩罚
    unsigned int const *tok  // 上一次采样结果
) {{
    next_kernel(logits, records, n, eos, temperature, penalty, tok);
}}"#
        );
        let program = Rtc::new()
            .arch(ctx.dev().compute_capability())
            .compile(&code)
            .unwrap();
        ctx.load(&program)
    }
}
