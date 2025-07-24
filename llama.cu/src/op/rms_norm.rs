use super::{Handle, ModuleKey, Operator, cuda_type};
use crate::utils::{destruct, dims, offset_ptr};
use cuda::{Device, Stream, VirByte, params};
use nn::{Arg, Tensor, digit_layout::DigitLayout};
use std::ffi::c_int;

pub struct RmsNorm;

impl Operator for RmsNorm {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        destruct!([x, w] = inputs);
        destruct!([y] = outputs);
        let Some(Arg::Float(epsilon)) = arg else {
            panic!()
        };
        let ta = x.dt();
        let tw = w.dt();
        assert_eq!(y.dt(), ta);

        let unit = ta.nbytes() as isize;

        // 支持二维和三维输入
        match x.shape().len() {
            2 => {
                dims!([n, d] = x);
                dims!([n_, d_] = y);
                dims!([d__] = w);
                assert_eq!(n_, n);
                assert_eq!(d_, d);
                assert_eq!(d__, d);

                let (code, block_dim) = code_2d(&handle.ctx.dev(), ta, tw, d);
                let key = [
                    ModuleKey::Text("rms-norm-2d"),
                    ModuleKey::Type(ta),
                    ModuleKey::Type(tw),
                    ModuleKey::Size(d),
                ]
                .into_iter();
                let module = handle.compile(key.collect(), || code);
                let kernel = module.get_kernel(c"rms_norm_2d");

                let params = params![
                    offset_ptr(&y),
                    (y.strides()[0] / unit) as c_int,
                    offset_ptr(&x),
                    (x.strides()[0] / unit) as c_int,
                    offset_ptr(&w),
                    epsilon as f32
                ];

                stream.launch(&kernel, (n as u32, block_dim as u32, 0), &params.to_ptrs());
            }
            3 => {
                dims!([batch, seq, d] = x);
                dims!([batch_, seq_, d_] = y);
                dims!([d__] = w);
                assert_eq!(batch_, batch);
                assert_eq!(seq_, seq);
                assert_eq!(d_, d);
                assert_eq!(d__, d);

                let (code, block_dim) = code_3d(&handle.ctx.dev(), ta, tw, d);
                let key = [
                    ModuleKey::Text("rms-norm-3d"),
                    ModuleKey::Type(ta),
                    ModuleKey::Type(tw),
                    ModuleKey::Size(d),
                ]
                .into_iter();
                let module = handle.compile(key.collect(), || code);
                let kernel = module.get_kernel(c"rms_norm_3d");

                let params = params![
                    offset_ptr(&y),
                    (y.strides()[0] / unit) as c_int, // batch stride
                    (y.strides()[1] / unit) as c_int, // seq stride
                    offset_ptr(&x),
                    (x.strides()[0] / unit) as c_int, // batch stride
                    (x.strides()[1] / unit) as c_int, // seq stride
                    offset_ptr(&w),
                    epsilon as f32
                ];

                stream.launch(
                    &kernel,
                    ((seq as u32, batch as u32), block_dim as u32, 0),
                    &params.to_ptrs(),
                );
            }
            _ => panic!("RmsNorm only supports 2D or 3D input tensors"),
        };
    }
}

fn code_2d(dev: &Device, ta: DigitLayout, tw: DigitLayout, d: usize) -> (String, usize) {
    const CODE: &str = include_str!("rms_norm.cuh");
    let ta = cuda_type(ta);
    let tw = cuda_type(tw);
    let block_size = dev.block_limit().max_threads;
    let (body, n_thread_block) = if d <= block_size {
        (
            format!("padding_2d<{d}>(y, stride_y, x, stride_x, w, epsilon)"),
            d,
        )
    } else {
        let n_threads_warp = dev.warp_size();
        assert_eq!(d % n_threads_warp, 0);
        let max_num_warp_block = block_size / n_threads_warp;
        let num_warps_block = max_num_warp_block;
        let num_threads_block = n_threads_warp * num_warps_block;
        let num_items_thread = (d / n_threads_warp).div_ceil(num_warps_block);
        (
            format!(
                "folding_2d<{num_threads_block}, {num_items_thread}>(y, stride_y, x, stride_x, w, epsilon, {d})"
            ),
            num_threads_block,
        )
    };
    let code = format!(
        r#"{CODE}

extern "C" __global__ void rms_norm_2d(
    {ta} *__restrict__ y,
    int  const stride_y,
    {ta} const *__restrict__ x,
    int  const stride_x,
    {tw} const *__restrict__ w,
    float epsilon
){{
    {body};
}}"#
    );
    (code, n_thread_block)
}

fn code_3d(dev: &Device, ta: DigitLayout, tw: DigitLayout, d: usize) -> (String, usize) {
    const CODE: &str = include_str!("rms_norm.cuh");
    let ta = cuda_type(ta);
    let tw = cuda_type(tw);
    let block_size = dev.block_limit().max_threads;
    let (body, n_thread_block) = if d <= block_size {
        (
            format!(
                "padding_3d<{d}>(y, stride_y_batch, stride_y_seq, x, stride_x_batch, stride_x_seq, w, epsilon)"
            ),
            d,
        )
    } else {
        let n_threads_warp = dev.warp_size();
        assert_eq!(d % n_threads_warp, 0);
        let max_num_warp_block = block_size / n_threads_warp;
        let num_warps_block = max_num_warp_block;
        let num_threads_block = n_threads_warp * num_warps_block;
        let num_items_thread = (d / n_threads_warp).div_ceil(num_warps_block);
        (
            format!(
                "folding_3d<{num_threads_block}, {num_items_thread}>(y, stride_y_batch, stride_y_seq, x, stride_x_batch, stride_x_seq, w, epsilon, {d})"
            ),
            num_threads_block,
        )
    };
    let code = format!(
        r#"{CODE}

extern "C" __global__ void rms_norm_3d(
    {ta} *__restrict__ y,
    int  const stride_y_batch,
    int  const stride_y_seq,
    {ta} const *__restrict__ x,
    int  const stride_x_batch,
    int  const stride_x_seq,
    {tw} const *__restrict__ w,
    float epsilon
){{
    {body};
}}"#
    );
    (code, n_thread_block)
}
