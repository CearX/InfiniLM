use super::{Handle, ModuleKey, Operator, cuda_type, gcd};
use crate::utils::{destruct, dims, offset_ptr, strides};
use nn::{Tensor, digit_layout::DigitLayout};
use operators::cuda::{Stream, VirByte, params};
use std::ffi::{c_int, c_uint};

pub struct Add4d;

impl Operator for Add4d {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        assert!(arg.is_none());

        destruct!([y] = outputs);
        destruct!([x, b] = inputs);
        // 检查维度
        dims!([n, m, hp, wp] = y);
        dims!([n2, m2, hp2, wp2] = x);
        dims!([n3, m3, hp3, wp3] = b);

        assert_eq!(n, n2);
        assert_eq!(n, n3);
        assert_eq!(m, m2);
        assert_eq!(m, m3);
        assert_eq!(hp, hp2);
        assert_eq!(hp, hp3);
        assert_eq!(wp, wp2);
        assert_eq!(wp, wp3);
        // 检查类型
        let dt = y.dt();
        assert_eq!(x.dt(), dt);
        assert_eq!(b.dt(), dt);
        // 获取 stride
        strides!([sny, smy, shy, swy] = y);
        strides!([snx, smx, shx, swx] = x);
        strides!([snb, smb, shb, swb] = b);
        // 获取最大线程数
        let max_threads_block = handle.ctx.dev().block_limit().max_threads;
        // 编译内核
        let key = [ModuleKey::Text("add4d"), ModuleKey::Type(dt)].into_iter();
        let module = handle.compile(key.collect(), || code(dt));
        let kernel = module.get_kernel(c"add4d");
        // 准备参数
        let unit = dt.nbytes() as isize;
        let params = params![
            offset_ptr(&y),
            (sny / unit) as c_int,
            (smy / unit) as c_int,
            (shy / unit) as c_int,
            (swy / unit) as c_int,
            offset_ptr(&x),
            (snx / unit) as c_int,
            (smx / unit) as c_int,
            (shx / unit) as c_int,
            (swx / unit) as c_int,
            offset_ptr(&b),
            (snb / unit) as c_int,
            (smb / unit) as c_int,
            (shb / unit) as c_int,
            (swb / unit) as c_int
        ];
        // 计算线程块配置
        let block = gcd(max_threads_block, wp);
        // 启动内核
        stream.launch(
            &kernel,
            ((n as c_uint, m as c_uint, hp as c_uint), block as c_uint, 0),
            &params.to_ptrs(),
        );
    }
}

fn code(dt: DigitLayout) -> String {
    const CODE: &str = include_str!("add4d.cuh");
    let dt = cuda_type(dt);

    format!(
        r#"{CODE}

extern "C" __global__ void add4d(
    {dt} *__restrict__ y,
    int const sny,
    int const smy,
    int const shy,
    int const swy,
    {dt} const *__restrict__ x,
    int const snx,
    int const smx,
    int const shx,
    int const swx,
    {dt} const *__restrict__ b,
    int const snb,
    int const smb,
    int const shb,
    int const swb
){{
    kernel(y, sny, smy, shy, swy,
           x, snx, smx, shx, swx,
           b, snb, smb, shb, swb);
}}"#
    )
}
