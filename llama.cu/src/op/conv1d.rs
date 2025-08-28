use super::{Handle, ModuleKey, Operator, cuda_type, gcd};
use crate::utils::{destruct, dims, offset_ptr, strides};
use cuda::{Stream, VirByte, params};
use nn::{Tensor, digit_layout::DigitLayout};
use std::ffi::{c_int, c_uint};

pub struct CausalConv1dUnified;

impl Operator for CausalConv1dUnified {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        _arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        destruct!([x, w, b, state] = inputs);
        destruct!([y] = outputs);

        dims!([n_x, d_x] = x);
        dims!([d_w, k] = w);
        dims!([d_b] = b);
        dims!([n_y, d_y] = y);
        dims!([d_state, k_state] = state);

        assert_eq!(n_x, n_y);
        assert_eq!(d_x, d_w);
        assert_eq!(d_x, d_b);
        assert_eq!(d_x, d_y);
        assert_eq!(d_state, d_x);
        assert_eq!(k_state, k);

        let dt = x.dt();
        assert_eq!(y.dt(), dt);
        let dt_wb = w.dt();
        assert_eq!(b.dt(), dt_wb);
        assert_eq!(state.dt(), nn::digit_layout::types::F32);

        strides!([s_n_x, s_d_x] = x);
        strides!([s_n_y, s_d_y] = y);
        strides!([s_d_w, s_k_w] = w);
        strides!([s_d_b] = b);
        strides!([s_state_c, s_state_k] = state);

        let kernel_size = k as c_int;
        let padding = kernel_size - 1;

        let key = [
            ModuleKey::Text("conv1d_unified"),
            ModuleKey::Type(dt),
            ModuleKey::Type(dt_wb),
        ]
        .into_iter();
        let max_threads_block = handle.ctx.dev().block_limit().max_threads;
        let module = handle.compile(key.collect(), || code(dt, dt_wb));
        let kernel = module.get_kernel(c"conv1d_unified");

        let params = params![
            offset_ptr(&y),
            (s_n_y / dt.nbytes() as isize) as c_int,
            (s_d_y / dt.nbytes() as isize) as c_int,
            offset_ptr(&x),
            (s_n_x / dt.nbytes() as isize) as c_int,
            (s_d_x / dt.nbytes() as isize) as c_int,
            offset_ptr(&w),
            (s_d_w / dt_wb.nbytes() as isize) as c_int,
            (s_k_w / dt_wb.nbytes() as isize) as c_int,
            offset_ptr(&b),
            (s_d_b / dt_wb.nbytes() as isize) as c_int,
            offset_ptr(&state),
            (s_state_c / 4) as c_int,
            (s_state_k / 4) as c_int,
            n_x as c_int,
            d_x as c_int,
            kernel_size,
            padding
        ];

        let block = gcd(max_threads_block, if n_x == 1 { d_x } else { n_x });

        let grid = if n_x == 1 {
            // decode 模式：每个channel一个block
            ((1 as c_uint, d_x as c_uint), block as c_uint, 0)
        } else {
            // prefill 模式：标准2D grid
            (((n_x / block) as c_uint, d_x as c_uint), block as c_uint, 0)
        };

        stream.launch(&kernel, grid, &params.to_ptrs());
    }
}

fn code(dt: DigitLayout, dt_wb: DigitLayout) -> String {
    const CODE: &str = include_str!("conv1d.cuh");
    let dt = cuda_type(dt);
    let dt_wb = cuda_type(dt_wb);
    format!(
        r#"{CODE}

extern "C" __global__ void conv1d_unified(
    {dt} *__restrict__ y,
    int const s_n_y,
    int const s_d_y,
    {dt} const *__restrict__ x,
    int const s_n_x,
    int const s_d_x,
    {dt_wb} const *__restrict__ w,
    int const s_d_w,
    int const s_k_w,
    {dt_wb} const *__restrict__ b,
    int const s_d_b,
    float* __restrict__ state,
    int const s_state_c,
    int const s_state_k,
    int const n,
    int const d,
    int const kernel_size,
    int const padding) {{
    kernel_unified<{dt}, {dt_wb}>(
        y, s_n_y, s_d_y,
        x, s_n_x, s_d_x,
        w, s_d_w, s_k_w,
        b, s_d_b,
        state, s_state_c, s_state_k,
        n, d, kernel_size, padding);
}}
"#,
    )
}
