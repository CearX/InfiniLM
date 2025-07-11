use super::{Handle, ModuleKey, Operator, cuda_type, gcd};
use crate::utils::{destruct, dims, offset_ptr, strides};
use nn::{
    Tensor,
    digit_layout::{DigitLayout, types},
};
use operators::cuda::{Stream, VirByte, params};
use std::ffi::c_uint;

pub struct Gelu;

impl Operator for Gelu {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        assert!(arg.is_none());

        destruct!([data] = inputs);
        destruct!([out] = outputs);

        // 检查维度
        dims!([n, d] = data);
        dims!([n2, d2] = out);

        assert_eq!(n, n2);
        assert_eq!(d, d2);

        // 检查类型
        let dt = data.dt();
        assert_eq!(types::F16, dt);
        assert_eq!(out.dt(), dt);

        // 获取stride
        strides!([_s_n_data, s_d_data] = data);
        strides!([_s_n_out, s_d_out] = out);

        // 确保stride符合期望
        let unit = dt.nbytes() as isize;
        assert_eq!(s_d_data, unit);
        assert_eq!(s_d_out, unit);

        // 获取最大线程数
        let max_threads_block = handle.ctx.dev().block_limit().max_threads;

        // 编译内核
        let key = [ModuleKey::Text("gelu"), ModuleKey::Type(dt)].into_iter();
        let module = handle.compile(key.collect(), || code(dt));
        let kernel = module.get_kernel(c"gelu");

        // 准备参数
        let params = params![offset_ptr(&out), offset_ptr(&data)];

        // 计算线程块配置
        let block = gcd(max_threads_block, d);

        // 启动内核
        stream.launch(
            &kernel,
            (((n * d).div_ceil(block) as c_uint), block as c_uint, 0),
            &params.to_ptrs(),
        );
    }
}

fn code(dt: DigitLayout) -> String {
    const CODE: &str = include_str!("gelu.cuh");
    let dt = cuda_type(dt);

    format!(
        r#"{CODE}

extern "C" __global__ void gelu(
    {dt} *__restrict__ out,
    {dt} const *__restrict__ data
){{
    kernel(out, data);
}}"#
    )
}
