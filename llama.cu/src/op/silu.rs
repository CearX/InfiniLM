use super::{Handle, ModuleKey, Operator, cuda_type, gcd};
use crate::utils::{destruct, dims, offset_ptr, strides};
use nn::{Tensor, digit_layout::DigitLayout};
use operators::cuda::{Stream, VirByte, params};
use std::ffi::{c_int, c_uint};

pub struct Silu;

impl Operator for Silu {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        assert!(arg.is_none());

        destruct!([up] = inputs);
        destruct!([out] = outputs);

        // 检查维度
        dims!([n, d] = up);
        dims!([n2, d2] = out);

        assert_eq!(n, n2);
        assert_eq!(d, d2);

        // 检查类型
        let dt = up.dt();
        assert_eq!(out.dt(), dt);

        // 获取 stride
        strides!([s_n_up, s_d_up] = up);
        strides!([s_n_out, s_d_out] = out);

        // 确保 stride 符合期望
        let unit = dt.nbytes() as isize;
        assert_eq!(s_d_up, unit);
        assert_eq!(s_d_out, unit);

        // 获取最大线程数
        let max_threads_block = handle.ctx.dev().block_limit().max_threads;

        // 编译内核
        let key = [ModuleKey::Text("silu"), ModuleKey::Type(dt)].into_iter();
        let module = handle.compile(key.collect(), || code(dt));
        let kernel = module.get_kernel(c"silu");

        // 准备参数
        let params = params![
            offset_ptr(&out),
            (s_n_out / unit) as c_int,
            offset_ptr(&up),
            (s_n_up / unit) as c_int
        ];

        // 计算线程块配置
        let block = gcd(max_threads_block, d);

        // 启动内核
        stream.launch(
            &kernel,
            ((n as c_uint, (d / block) as c_uint), block as c_uint, 0),
            &params.to_ptrs(),
        );
    }
}

fn code(dt: DigitLayout) -> String {
    const CODE: &str = include_str!("silu.cuh");
    let dt = cuda_type(dt);

    format!(
        r#"{CODE}

extern "C" __global__ void silu(
    {dt} *__restrict__ out,
    int const stride_out,
    {dt} const *__restrict__ up,
    int const stride_up
){{
    kernel(out, stride_out, up, stride_up);
}}"#
    )
}
