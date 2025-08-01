use super::{Handle, ModuleKey, Operator, cuda_type, move_type};
use crate::utils::{destruct, dims, offset_ptr};
use nn::{Arg, Tensor, digit_layout::DigitLayout};
use operators::cuda::{DevByte, Stream, VirByte, params};
use std::ffi::c_uint;

pub struct EmbeddingQw2vl;

impl Operator for EmbeddingQw2vl {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        let Some(Arg::Arr(img_info)) = arg else {
            panic!()
        };
        let img_info = img_info
            .iter()
            .map(|x| x.to_usize())
            .collect::<Vec<usize>>();
        let (img_embd_ptr, img_token_len, img_start_pos) = (img_info[0], img_info[1], img_info[2]);

        destruct!([token_embd, tokens] = inputs);
        destruct!([x] = outputs);
        let tval = x.dt();
        let tidx = tokens.dt();
        assert_eq!(token_embd.dt(), tval);

        // 创建图像嵌入tensor
        let img_embd = Tensor::from_raw_parts(
            tval,
            token_embd.layout().clone(),
            img_embd_ptr as *const VirByte,
        );

        dims!([n] = tokens);
        dims!([n_, d] = x);
        dims!([_, d_] = token_embd);
        dims!([_, d_img] = img_embd);
        assert_eq!(n, n_);
        assert_eq!(d, d_);
        assert_eq!(d, d_img);

        let line = d * tval.nbytes();
        let unit = (0..=5)
            .rev()
            .map(|i| 1 << i)
            .find(|unit| line % unit == 0)
            .unwrap();

        // 1. 执行常规的embedding计算
        let key = [
            ModuleKey::Text("embedding"),
            ModuleKey::Type(tidx),
            ModuleKey::Size(unit),
        ]
        .into_iter();
        let module = handle.compile(key.collect(), || code(unit, tidx));
        let kernel = module.get_kernel(c"embedding");
        let params = params![offset_ptr(&x), offset_ptr(&token_embd), offset_ptr(&tokens)];

        stream.launch(
            &kernel,
            (n as c_uint, (line / unit) as c_uint, 0),
            &params.to_ptrs(),
        );

        // 2. 执行图像嵌入替换
        if img_token_len > 0 {
            let src_slice = unsafe {
                std::slice::from_raw_parts(
                    offset_ptr(&img_embd).cast::<DevByte>(),
                    img_token_len * tval.nbytes(),
                )
            };

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    (offset_ptr(&x).cast::<DevByte>().cast_mut())
                        .add(img_start_pos * tval.nbytes()),
                    img_token_len * tval.nbytes(),
                )
            };

            // 3. 执行内存复制
            stream.memcpy_d2d(dst_slice, src_slice);
        }
    }
}

fn code(unit: usize, tidx: DigitLayout) -> String {
    const CODE: &str = include_str!("embedding.cuh");
    let tval = move_type(unit);
    let tidx = cuda_type(tidx);
    format!(
        r#"{CODE}

extern "C" __global__ void embedding(
    {tval} *__restrict__ out,
    {tval} const *__restrict__ table,
    {tidx} const *__restrict__ index
) {{
    kernel(out, table, index);
}}"#
    )
}
