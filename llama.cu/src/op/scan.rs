use super::{Handle, ModuleKey, Operator, cuda_type};
use crate::utils::{destruct, dims, offset_ptr, strides};
use cuda::{Stream, VirByte, params};
use nn::{Tensor, digit_layout::DigitLayout};
use std::ffi::{c_int, c_uint};

pub struct SelectiveScanWithWriteback;
#[allow(non_snake_case)]
impl Operator for SelectiveScanWithWriteback {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        _arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>, // [u,delta,A,B,C,D,state]
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>, // [out]
        stream: &Stream,
    ) {
        destruct!([u, delta, A, B, C, D, state] = inputs);
        destruct!([out] = outputs);

        dims!([n, d] = u);
        dims!([n_delta, d_delta] = delta);
        dims!([d_A, n_state] = A);
        dims!([n_B, n_state_B] = B);
        dims!([n_C, n_state_C] = C);
        dims!([d_D] = D);
        dims!([n_out, d_out] = out);
        dims!([d_state, n_state2] = state);
        assert_eq!(n, n_delta);
        assert_eq!(d, d_delta);
        assert_eq!(n, n_B);
        assert_eq!(n, n_C);
        assert_eq!(d, d_A);
        assert_eq!(d, d_D);
        assert_eq!(n, n_out);
        assert_eq!(d, d_out);
        assert_eq!(d_state, d);
        assert_eq!(n_state, n_state_B);
        assert_eq!(n_state, n_state_C);
        assert_eq!(n_state, n_state2);
        assert_eq!(state.dt(), nn::digit_layout::types::F32);
        assert_eq!(state.dt(), A.dt());
        assert_eq!(d_state, 5120);
        assert_eq!(n_state2, 16);
        assert_eq!(d, 5120);

        let dt = u.dt();
        let dt_AD = A.dt();
        strides!([s_n_u, s_d_u] = u);
        strides!([s_n_delta, s_d_delta] = delta);
        strides!([s_d_A, s_s_A] = A);
        strides!([s_n_B, s_s_B] = B);
        strides!([s_n_C, s_s_C] = C);
        strides!([s_d_D] = D);
        strides!([s_n_out, s_d_out] = out);
        strides!([s_state_d, s_state_n] = state);
        assert_eq!(s_state_d / 4, n_state as isize);
        assert_eq!(s_state_n / 4, 1);

        let unit = dt.nbytes() as isize;
        let unit_AD = dt_AD.nbytes() as isize;
        let key = [
            ModuleKey::Text("scan"),
            ModuleKey::Type(dt),
            ModuleKey::Type(dt_AD),
            ModuleKey::Size(n_state),
        ]
        .into_iter();
        let module = handle.compile(key.collect(), || code(dt, dt_AD, n_state));
        let kernel = module.get_kernel(c"scan_with_writeback");

        let params = params![
            offset_ptr(&out),
            (s_n_out / unit) as c_int,
            (s_d_out / unit) as c_int,
            offset_ptr(&u),
            (s_n_u / unit) as c_int,
            (s_d_u / unit) as c_int,
            offset_ptr(&delta),
            (s_n_delta / unit) as c_int,
            (s_d_delta / unit) as c_int,
            offset_ptr(&A),
            (s_d_A / unit_AD) as c_int,
            (s_s_A / unit_AD) as c_int,
            offset_ptr(&B),
            (s_n_B / unit) as c_int,
            (s_s_B / unit) as c_int,
            offset_ptr(&C),
            (s_n_C / unit) as c_int,
            (s_s_C / unit) as c_int,
            offset_ptr(&D),
            (s_d_D / unit_AD) as c_int,
            n as c_int,
            d as c_int,
            n_state as c_int,
            offset_ptr(&state),
            (s_state_d / 4) as c_int,
            (s_state_n / 4) as c_int
        ];

        let block = 1u32;
        let shared_mem_bytes = n_state * 4usize;
        // (grid.y, grid.x)
        stream.launch(
            &kernel,
            (
                (1 as c_uint, d as c_uint),
                block as c_uint,
                shared_mem_bytes,
            ),
            &params.to_ptrs(),
        );
    }
}

#[allow(non_snake_case)]
pub(crate) fn code(dt: DigitLayout, dt_AD: DigitLayout, _state: usize) -> String {
    const CODE: &str = include_str!("scan.cuh");
    let dt = cuda_type(dt);
    let dt_AD = cuda_type(dt_AD);
    format!(
        r#"{CODE}

extern "C" __global__ void scan_with_writeback(
    {dt} *__restrict__ out,
    int const s_n_out,
    int const s_d_out,
    {dt} const *__restrict__ u,
    int const s_n_u,
    int const s_d_u,
    {dt} const *__restrict__ delta,
    int const s_n_delta,
    int const s_d_delta,
    {dt_AD} const *__restrict__ A,
    int const s_d_A,
    int const s_s_A,
    {dt} const *__restrict__ B,
    int const s_n_B,
    int const s_s_B,
    {dt} const *__restrict__ C,
    int const s_n_C,
    int const s_s_C,
    {dt_AD} const *__restrict__ D,
    int const s_d_D,
    int const n,
    int const d,
    int const state,
    float *__restrict__ state_out,
    int const s_state_d,
    int const s_state_n) {{
    kernel_with_writeback(out, s_n_out, s_d_out,
           u, s_n_u, s_d_u,
           delta, s_n_delta, s_d_delta,
           A, s_d_A, s_s_A,
           B, s_n_B, s_s_B,
           C, s_n_C, s_s_C,
           D, s_d_D,
           n, d, state,
           state_out, s_state_d, s_state_n);
}}
"#
    )
}
