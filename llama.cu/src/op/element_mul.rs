use super::{Handle, ModuleKey, Operator, cuda_type, gcd};
use crate::utils::{destruct, dims, offset_ptr, strides};
use cuda::{Stream, VirByte, params};
use nn::{Tensor, digit_layout::DigitLayout};
use std::ffi::{c_int, c_uint};

pub struct ElementMul;

impl Operator for ElementMul {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        _arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        destruct!([a, b] = inputs);
        destruct!([y] = outputs);

        dims!([n, d] = a);
        dims!([n2, d2] = b);
        dims!([n3, d3] = y);
        assert_eq!(n, n2);
        assert_eq!(d, d2);
        assert_eq!(n, n3);
        assert_eq!(d, d3);

        let dt = a.dt();
        assert_eq!(b.dt(), dt);
        assert_eq!(y.dt(), dt);

        strides!([s_n_a, s_d_a] = a);
        strides!([s_n_b, s_d_b] = b);
        strides!([s_n_y, s_d_y] = y);
        let unit = dt.nbytes() as isize;
        assert_eq!(s_d_a, unit);
        assert_eq!(s_d_b, unit);
        assert_eq!(s_d_y, unit);

        let max_threads_block = handle.ctx.dev().block_limit().max_threads;
        let key = [ModuleKey::Text("element-mul"), ModuleKey::Type(dt)].into_iter();
        let module = handle.compile(key.collect(), || code(dt));
        let kernel = module.get_kernel(c"element_mul");

        let params = params![
            offset_ptr(&y),
            (s_n_y / unit) as c_int,
            offset_ptr(&a),
            (s_n_a / unit) as c_int,
            offset_ptr(&b),
            (s_n_b / unit) as c_int,
            n as c_int,
            d as c_int
        ];

        let block = gcd(max_threads_block, d);
        stream.launch(
            &kernel,
            (((d / block) as c_uint, n as c_uint), block as c_uint, 0),
            &params.to_ptrs(),
        );
    }
}

fn code(dt: DigitLayout) -> String {
    const CODE: &str = include_str!("element_mul.cuh");
    let dt = cuda_type(dt);
    format!(
        r#"{CODE}

extern "C" __global__ void element_mul(
    {dt} *__restrict__ y,
    int const stride_y_n,
    {dt} const *__restrict__ a,
    int const stride_a_n,
    {dt} const *__restrict__ b,
    int const stride_b_n,
    int const n,
    int const d
){{
    kernel(y, stride_y_n, a, stride_a_n, b, stride_b_n, n, d);
}}
"#
    )
}
