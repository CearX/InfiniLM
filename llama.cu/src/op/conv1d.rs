use super::{Handle, ModuleKey, Operator, cuda_type, gcd};
use crate::utils::{destruct, dims, offset_ptr, strides};
use cuda::{Stream, VirByte, params};
use nn::{Tensor, digit_layout::DigitLayout};
use std::ffi::{c_int, c_uint};

pub struct Conv1d;

impl Operator for Conv1d {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        _arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        destruct!([x, w, b] = inputs);
        destruct!([y] = outputs);

        dims!([n_x, d_x] = x);
        dims!([d_w, k] = w);
        dims!([d_b] = b);
        dims!([n_y, d_y] = y);

        assert_eq!(n_x, n_y);
        assert_eq!(d_x, d_w);
        assert_eq!(d_x, d_b);
        assert_eq!(d_x, d_y);
        let kernel_size = k as c_int;
        let padding = kernel_size - 1;

        let dt = x.dt();
        assert_eq!(y.dt(), dt);
        let dt_wb = w.dt();
        assert_eq!(b.dt(), dt_wb);

        strides!([s_n_x, s_d_x] = x);
        strides!([s_n_y, s_d_y] = y);
        strides!([s_d_w, s_k_w] = w);
        strides!([s_d_b] = b);

        let unit = dt.nbytes() as isize;
        let unit_w = w.dt().nbytes() as isize;
        assert_eq!(s_d_x, unit);
        assert_eq!(s_d_y, unit);
        assert_eq!(s_k_w, unit_w);
        assert_eq!(s_d_b, unit_w);

        let max_threads_block = handle.ctx.dev().block_limit().max_threads;

        let key = [
            ModuleKey::Text("conv1d"),
            ModuleKey::Type(dt),
            ModuleKey::Type(dt_wb),
        ]
        .into_iter();
        let module = handle.compile(key.collect(), || code(dt, dt_wb));
        let kernel = module.get_kernel(c"conv1d");

        let params = params![
            offset_ptr(&y),
            (s_n_y / unit) as c_int,
            (s_d_y / unit) as c_int,
            offset_ptr(&x),
            (s_n_x / unit) as c_int,
            (s_d_x / unit) as c_int,
            offset_ptr(&w),
            (s_d_w / unit_w) as c_int,
            (s_k_w / unit_w) as c_int,
            offset_ptr(&b),
            (s_d_b / unit_w) as c_int,
            kernel_size,
            padding
        ];

        let block = gcd(max_threads_block, n_y);

        stream.launch(
            &kernel,
            (
                ((n_y / block) as c_uint, (d_b as c_uint)),
                block as c_uint,
                0,
            ),
            &params.to_ptrs(),
        );
    }
}

fn code(dt: DigitLayout, dt_wb: DigitLayout) -> String {
    const CODE: &str = include_str!("conv1d.cuh");
    let dt = cuda_type(dt);
    let dt_wb = cuda_type(dt_wb);
    format!(
        r#"{CODE}

extern "C" __global__ void conv1d(
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
    int const kernel_size,
    int const padding) {{
    kernel(y, s_n_y, s_d_y,
           x, s_n_x, s_d_x,
           w, s_d_w, s_k_w,
           b, s_d_b,
           kernel_size, padding);
}}

extern "C" __global__ void conv1d_step(
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
    int const kernel_size,
    int const k_minus_1,
    int const d_channels) {{
    kernel_step<{dt}, {dt_wb}>(
        y, s_n_y, s_d_y,
        x, s_n_x, s_d_x,
        w, s_d_w, s_k_w,
        b, s_d_b,
        state, s_state_c, s_state_k,
        kernel_size, k_minus_1, d_channels);
}}

extern "C" __global__ void conv1d_write_state(
    {dt} const *__restrict__ x,
    int const s_n_x,
    int const s_d_x,
    float* __restrict__ state,
    int const s_state_c,
    int const s_state_k,
    int const n,
    int const d,
    int const k) {{
    kernel_write_state<{dt}>(x, s_n_x, s_d_x, state, s_state_c, s_state_k, n, d, k);
}}
"#,
    )
}

pub struct CausalConv1dStep;

impl Operator for CausalConv1dStep {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        _arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        destruct!([x_t, w, b, state] = inputs);
        destruct!([y_t] = outputs);

        dims!([n_x, d_x] = x_t);
        dims!([d_w, k] = w);
        dims!([d_b] = b);
        dims!([n_y, d_y] = y_t);
        dims!([d_state, k_state] = state);

        assert_eq!(n_x, 1);
        assert_eq!(n_y, 1);
        assert_eq!(d_x, d_w);
        assert_eq!(d_x, d_b);
        assert_eq!(d_x, d_y);
        assert_eq!(d_state, d_x);
        assert_eq!(k_state, k);

        let dt = x_t.dt();
        assert_eq!(y_t.dt(), dt);
        let dt_wb = w.dt();
        assert_eq!(b.dt(), dt_wb);
        assert_eq!(state.dt(), nn::digit_layout::types::F32);

        strides!([s_n_x, s_d_x] = x_t);
        strides!([s_n_y, s_d_y] = y_t);
        strides!([s_d_w, s_k_w] = w);
        strides!([s_d_b] = b);
        strides!([s_state_c, s_state_k] = state);

        let max_threads_block = handle.ctx.dev().block_limit().max_threads;
        let key = [
            ModuleKey::Text("conv1d"),
            ModuleKey::Type(dt),
            ModuleKey::Type(dt_wb),
        ]
        .into_iter();
        let module = handle.compile(key.collect(), || code(dt, dt_wb));
        let kernel = module.get_kernel(c"conv1d_step");

        let params = params![
            offset_ptr(&y_t),
            (s_n_y / dt.nbytes() as isize) as c_int,
            (s_d_y / dt.nbytes() as isize) as c_int,
            offset_ptr(&x_t),
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
            k as c_int,
            (k - 1) as c_int,
            d_y as c_int
        ];

        let block = gcd(max_threads_block, d_y);
        let grid_x: usize = d_y.div_ceil(block);
        // (grid.y, grid.x)
        stream.launch(
            &kernel,
            ((1 as c_uint, grid_x as c_uint), block as c_uint, 0),
            &params.to_ptrs(),
        );
    }
}

pub struct Conv1dWriteStateStep;

impl Operator for Conv1dWriteStateStep {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        _arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        _outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        destruct!([x, state] = inputs);
        let dt = x.dt();
        dims!([n, d] = x);
        dims!([d_state, k] = state);
        assert_eq!(d_state, d);
        assert_eq!(state.dt(), nn::digit_layout::types::F32);
        strides!([s_n_x, s_d_x] = x);
        strides!([s_state_c, s_state_k] = state);

        let key = [ModuleKey::Text("conv1d"), ModuleKey::Type(dt)].into_iter();
        let max_threads_block = handle.ctx.dev().block_limit().max_threads;
        let module = handle.compile(key.collect(), || code(dt, dt));
        let kernel = module.get_kernel(c"conv1d_write_state");

        let params = params![
            offset_ptr(&x),
            (s_n_x / dt.nbytes() as isize) as c_int,
            (s_d_x / dt.nbytes() as isize) as c_int,
            offset_ptr(&state),
            (s_state_c / 4) as c_int,
            (s_state_k / 4) as c_int,
            n as c_int,
            d as c_int,
            k as c_int
        ];

        let block = gcd(max_threads_block, d);
        let grid_x: usize = d.div_ceil(block);
        stream.launch(
            &kernel,
            ((1 as c_uint, grid_x as c_uint), block as c_uint, 0),
            &params.to_ptrs(),
        );
    }
}
