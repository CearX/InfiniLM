use crate::{
    handle::Handle,
    load::WeightLoader,
    op::{self, Operator as _},
    utils::dims,
};
use cuda::{CurrentCtx, DevMem, Stream, VirByte};
use nn::{Arg, Linear, NormType, Normalization, Tensor, digit_layout::types};
use tokeneer::utok;

pub(super) struct OutputHead<'ctx> {
    norm: Tensor<DevMem<'ctx>, 2>,
    linear: Tensor<DevMem<'ctx>, 2>,
    epsilon: Option<Arg>,
}

impl<'ctx> OutputHead<'ctx> {
    pub fn new(nn: nn::OutputHead<Tensor<&[u8], 2>>, ctx: &'ctx CurrentCtx) -> Self {
        let nn::OutputHead {
            out_norm: Normalization { items, epsilon, .. },
            lm_head: Linear { weight, .. },
        } = nn;
        let norm = match items {
            NormType::RmsNorm { scale, .. } => scale,
            NormType::LayerNorm { .. } => todo!(),
        };

        let stream = ctx.stream();
        let mut loader = WeightLoader::new([]);
        let mut load = |t: Tensor<&[u8], 2>| {
            let dst = stream.malloc::<u8>(t.get().len());
            let (host, mut ans) = t.replace(dst);
            loader.load(ans.get_mut(), &stream, |inter| {
                inter.copy_from_slice(host);
            });
            ans
        };

        Self {
            norm: load(norm),
            linear: load(weight),
            epsilon: Some(epsilon.into()),
        }
    }
}

impl OutputHead<'_> {
    pub fn nvoc(&self) -> usize {
        dims!([ans, _] = self.linear);
        ans
    }

    pub fn launch<'ctx>(
        &mut self,
        x: Tensor<*const VirByte, 2>,
        out_idx: &[utok],
        handle: &mut Handle,
        stream: &Stream<'ctx>,
    ) -> Tensor<DevMem<'ctx>, 2> {
        let Self {
            norm,
            linear,
            epsilon,
        } = self;
        dims!([_, d] = x);
        let out_len = out_idx.len();
        let out_idx_ = stream.from_host(out_idx);
        let out_idx =
            Tensor::from_dim_slice(types::U32, [out_len]).map(|_| out_idx_.as_ptr().cast());
        // gather
        let mut out_ = Tensor::new(x.dt(), [out_len, d]).map(|len| stream.malloc::<u8>(len));
        let out = out_.as_mut().map(|mem| mem.as_ptr().cast());
        op::Embedding::launch(handle, None, [x, out_idx], [out.clone()], stream);
        stream.free(out_idx_);
        // norm
        let scale = norm.as_ref().map(|mem| mem.as_ptr().cast());
        op::RmsNorm::launch(
            handle,
            epsilon.clone(),
            [out.clone(), scale],
            [out.clone()],
            stream,
        );
        // linear
        dims!([nvoc, _] = linear);
        let mut logits_ =
            Tensor::new(out.dt(), [out_len, nvoc]).map(|len| stream.malloc::<u8>(len));
        let logits = logits_.as_mut().map(|mem| mem.as_ptr().cast());
        let lm_head = linear.as_ref().map(|mem| mem.as_ptr().cast());
        op::Linear::launch(handle, Some(false.into()), [out, lm_head], [logits], stream);
        stream.free(out_.take());
        logits_
    }
}
