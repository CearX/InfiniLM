use crate::{
    SessionId,
    batch::SampleInfo,
    op::random_sample::{KV_PAIR, KVPair, LogitsModifier, RandomSample},
    utils::dims,
};
use cuda::{CurrentCtx, DevByte, DevMem, Stream};
use nn::Tensor;
use std::{collections::BTreeMap, ptr::null};
use tokeneer::utok;

pub(super) struct SampleManager<'ctx> {
    sample: RandomSample<'ctx>,
    modifier: LogitsModifier<'ctx>,
    state: BTreeMap<SessionId, DevMem<'ctx>>,
}

impl<'ctx> SampleManager<'ctx> {
    pub fn new(nvoc: usize, eos: utok, ctx: &'ctx CurrentCtx) -> Self {
        Self {
            sample: RandomSample::new(nvoc, ctx),
            modifier: LogitsModifier::new(nvoc, eos, ctx),
            state: Default::default(),
        }
    }

    pub fn sample(
        &mut self,
        mut logits_: Tensor<DevMem<'_>, 2>,
        input: &[DevByte],
        config: &[(SessionId, SampleInfo)],
        stream: &Stream<'ctx>,
    ) -> DevMem<'ctx> {
        let Self {
            sample,
            modifier,
            state,
        } = self;
        let logits = logits_.as_mut().map(|mem| mem.as_ptr().cast());
        dims!([out_len, _nvoc] = logits);

        let kv_pair_template = Tensor::from_dim_slice(KV_PAIR, []);
        let kv_pair = stream.malloc::<KVPair>(out_len);
        for (i, (id, info)) in config.iter().enumerate() {
            let logits = logits.clone().transform(|layout| layout.index(0, i));
            let SampleInfo {
                args,
                input_idx,
                decode_len,
            } = info;

            let state = state
                .entry(*id)
                .or_insert_with(|| modifier.new_state(stream));
            let tok = if *decode_len == 0 {
                null()
            } else {
                input[..input_idx * size_of::<utok>()].as_ptr()
            };

            unsafe {
                modifier.next(
                    &logits,
                    state.as_mut_ptr(),
                    tok,
                    args.temperature,
                    args.repetition_penalty,
                    stream,
                )
            }

            let kv_pair = kv_pair_template
                .as_ref()
                .map(|_| kv_pair[i * size_of::<KVPair>()..].as_ptr().cast());
            if args.is_argmax() {
                sample.argmax(kv_pair, logits, stream)
            } else {
                sample.sample(kv_pair, logits, *args, rand::random(), stream)
            }
        }
        stream.free(logits_.take());
        kv_pair
    }

    pub fn remove(&mut self, id: impl IntoIterator<Item = SessionId>) {
        for id in id {
            self.state.remove(&id);
        }
    }
}
