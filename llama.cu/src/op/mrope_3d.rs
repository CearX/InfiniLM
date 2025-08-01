use super::{Handle, ModuleKey, Operator, cuda_type};
use crate::utils::{destruct, dims, offset_ptr, strides};
use nn::{
    Arg, Tensor,
    digit_layout::{DigitLayout, types},
};
use operators::cuda::{Stream, VirByte, params};
use std::ffi::c_uint;

#[allow(dead_code)]
pub struct MRope3d;

impl Operator for MRope3d {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        let Some(Arg::Arr(mrope_section)) = arg else {
            panic!()
        };
        let mrope_section = mrope_section
            .into_iter()
            .map(|x| x.to_usize() as u32)
            .collect::<Vec<u32>>();
        let mrope_section_acc = mrope_section
            .iter()
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect::<Vec<u32>>();
        let mrope_section = stream.from_host(mrope_section_acc.as_slice());

        destruct!([x, pos, sin, cos] = inputs);
        destruct!([y] = outputs);

        //检查dim
        dims!([n, dh_mut_dhead] = x);
        dims!([n2, dim] = pos);
        dims!([nctx, dh_2] = sin);
        dims!([nctx2, dh_2_] = cos);
        dims!([n3, dh_mut_dhead_] = y);

        assert_eq!(dim, 3); // 3d pos_ids
        assert_eq!(n, n2);
        assert_eq!(n, n3);
        assert_eq!(dh_mut_dhead, dh_mut_dhead_);
        assert_eq!(dh_2, dh_2_);
        assert_eq!(nctx, nctx2);

        let dh = dh_2 * 2;
        let d_head = dh_mut_dhead / dh;
        assert_eq!(dh_mut_dhead % dh, 0);

        //检查type
        let dt_t = x.dt();
        let dt_p = pos.dt();

        assert_eq!(y.dt(), dt_t);
        assert_eq!(sin.dt(), types::F32);
        assert_eq!(cos.dt(), types::F32);

        //获取stride
        strides!([s_n_y, s_dh_mut_dhead_y] = y);
        strides!([s_n_x, s_dh_mut_dhead_x] = x);
        let stride_token_y = (s_n_y / dt_t.nbytes() as isize) as i32;
        let stride_head_y = (s_dh_mut_dhead_y / dt_t.nbytes() as isize * dh as isize) as i32;

        let stride_token_x = (s_n_x / dt_t.nbytes() as isize) as i32;
        let stride_head_x = (s_dh_mut_dhead_x / dt_t.nbytes() as isize * dh as isize) as i32;

        // 计算线程块配置，参考mod.rs中的逻辑
        let dh_div_2 = dh / 2;
        // 先获取最大线程数，避免后面借用冲突
        let max_threads_block = handle.ctx.dev().block_limit().max_threads;

        // 确保线程块大小不超过设备限制
        assert!(max_threads_block >= dh_div_2);

        // 计算合适的nh_l和nh_h
        let max_nh_l = std::cmp::min(max_threads_block / dh_div_2, d_head);
        let nh_l = (1..=max_nh_l)
            .rev()
            .find(|nhl| d_head % nhl == 0)
            .unwrap_or(1);
        let nh_h = d_head / nh_l;

        let key = [
            ModuleKey::Text("mrope-3d"),
            ModuleKey::Type(dt_t),
            ModuleKey::Type(dt_p),
        ]
        .into_iter();
        let module = handle.compile(key.collect(), || code(dt_p, dt_t));
        let kernel = module.get_kernel(c"mrope_3d");

        let params = params![
            offset_ptr(&y),
            stride_token_y,
            stride_head_y,
            offset_ptr(&x),
            stride_token_x,
            stride_head_x,
            offset_ptr(&pos),
            offset_ptr(&sin),
            offset_ptr(&cos),
            mrope_section.as_ptr() as *const u32
        ];

        // dim3 grid(nh_h, n);
        // dim3 block(dh_div_2, nh_l);
        stream.launch(
            &kernel,
            (
                (n as c_uint, nh_h as c_uint),
                (nh_l as c_uint, dh_div_2 as c_uint),
                0,
            ),
            &params.to_ptrs(),
        );
    }
}

fn code(tp: DigitLayout, ta: DigitLayout) -> String {
    const CODE: &str = include_str!("mrope_3d.cuh");
    let ta = cuda_type(ta);
    let tp = cuda_type(tp);

    // mrope操作直接使用padding模板函数
    let body = format!(
        "padding<{tp}, {ta}>(y, stride_token_y, stride_head_y, x, stride_token_x, stride_head_x, pos, sin_table, cos_table, mrope_section)"
    );

    let code = format!(
        r#"{CODE}

extern "C" __global__ void mrope_3d(
    {ta} *__restrict__ y,
    int const stride_token_y,
    int const stride_head_y,
    {ta} const *__restrict__ x,
    int const stride_token_x,
    int const stride_head_x,
    {tp} const *__restrict__ pos,
    float const *__restrict__ sin_table,
    float const *__restrict__ cos_table,
    {tp} const *__restrict__ mrope_section
){{
    {body};
}}"#
    );
    code
}
