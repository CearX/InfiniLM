use crate::{Operators, RandomSample, Weights};
use gguf::GGufModel;
use gpt2::{
    ext::ggml_quants::f16, LlamaArgs, Gpt2Meta, LlamaRequest, Storage, Gpt2Worker, Tensor, 
};
use operators::{
    common_cpu::{Cpu, ThisThread},
    random_sample::{KVPair, SampleArgs},
    Blob,
};
use std::slice::from_raw_parts_mut;
use test_utils::{Inference, TokenizerAndPrompt};

type Worker<'w> = Gpt2Worker<Operators, Weights<'w>>;


#[test]
fn test_encoder_forward(){
    // out = wte + wpe
    let Some(Inference {
        model,
        prompt,
        as_user,
        temperature,
        top_p,
        top_k,
        max_steps,
    }) = Inference::load()
    else {
        return;
    };
    let gguf = GGufModel::read(model.iter().map(|s| &**s));

    let model = Storage::from_gguf(&gguf);
    println!("{:?}", model.meta);

    let B: usize = 4; 
    let T: usize = 64; 
    let C: usize = 768;

    // let shape_wte = [V, C] = [50257, 768];
    let shape_wte = [1, 768];
    let wte = Tensor::new(model.meta.dt_embd, &shape_wte).map(|_| model.token_embd);
    println!("tensor = {wte}");
    
    // let shape_wpe = [T, C] = [64, 768];
    let shape_wpe = [1, 768];
    let wpe = Tensor::new(model.meta.dt_embd, &shape_wpe).map(|_| model.position_embd);
    println!("tensor = {wpe}");

}


#[test]
fn todoo(){
    // encoder_forward
    // layernorm_forward
        // operator
        // launch
    // matmul_forward
    // attention_forward
    
    // train, batch 
}

#[test]
fn test_storage(){
    // fromgguf
    // storage, blkstorage
    // gpt2meta, impl meta, tensor<usize>
    // --> print.meta
}

#[test]
fn test_weight(){
    // weight
    // weight.new.distribute.contiguous.blob.NonullPtr.alloc
    // weightloader: memory-byte-queue, operator.rs
    // weightdecorator: tensor<usize> + weight<W>
    // tensor
    // --> print.tensor
}

#[test]
fn test_tokenizer(){
    // lpe & titokenizer
    // --> Once upon a time,
}

#[test]
fn test_worker(){
    // worker：tensor<operator>, matmul, qkv, queue
    // operator, weight & bias
    // launch
    // --> encoder_forward
    // --> layernorm_forward
}

#[test]
fn test_inferr(){
    // args
    // cache
    // dequant
    // sample
    // launch!!!
    // --> infer
}

#[test]
fn test_train() {
    // gradient*2
    // backward_operator
    // train
    // --> train    
}
#[test]
fn test_meta() {
    let Some(Inference {
        model,
        prompt,
        as_user,
        temperature,
        top_p,
        top_k,
        max_steps,
    }) = Inference::load()
    else {
        return;
    };
    let gguf = GGufModel::read(model.iter().map(|s| &**s));

    let model = Storage::from_gguf(&gguf);
    println!("{:?}", model.meta);

    use gpt2::TensorUsage::Computation;
    let mut meta = model.meta.clone();
    let size_qkv = meta.attn_qkv_weight(Computation).take();
    let size_o = meta.attn_o_weight(Computation).take();
    let size_gate_up = meta.ffn_up_weight(Computation).take();
    let size_down = meta.ffn_down_weight(Computation).take();
    
    println!("meta = {meta:?}");
    println!("size_qkv = {size_qkv}");
    println!("size_o = {size_o}");
    println!("size_gate_up = {size_gate_up}");
    println!("size_down = {size_down}");
}

#[test]
fn test_gguf_bin_tensor_compare() {
    let Some(Inference {
        model,
        prompt,
        as_user,
        temperature,
        top_p,
        top_k,
        max_steps,
    }) = Inference::load()
    else {
        return;
    };
    let gguf = GGufModel::read(model.iter().map(|s| &**s));

    let model = Storage::from_gguf(&gguf);
    println!("{:?}", model.meta);

    // let shape_wtemb = [50257, 768];
    let shape_wtemb = [2, 1, 768];
    let t = Tensor::new(model.meta.dt_embd, &shape_wtemb).map(|_| model.token_embd);
    println!("tensor = {t}");
    // --> wte_compare_success!

    // let shape_qkvw = [768, 2304];
    // let t = Tensor::new(model.meta.dt_embd, &shape_qkvw).map(|_| model.blocks.get(0).unwrap().attn_qkv_weight);
    // println!("tensor = {t}");
}

#[test]
fn test_infer() {
    let Some(Inference {
        model,
        prompt,
        as_user,
        temperature,
        top_p,
        top_k,
        max_steps,
    }) = Inference::load()
    else {
        return;
    };
    let gguf = GGufModel::read(model.iter().map(|s| &**s));

    let TokenizerAndPrompt {
        eos,
        tokenizer,
        prompt,
    } = TokenizerAndPrompt::new(&gguf, prompt, as_user);

    let model = Storage::from_gguf(&gguf);
    println!("{:?}", model.meta);

    let sample_args = SampleArgs::new(temperature, top_p, top_k).expect("invalid sample args");
    println!("{sample_args:?}");

    let &Gpt2Meta {
        dt_embd,
        nctx,
        nvoc,
        // dh,
        ..
    } = &model.meta;

    let weights = Weights::new(&model, .., 1);
    let mut worker = Worker::new(&Cpu, model.meta.clone(), weights, true);
    let mut cache = model.meta.kv_cache(nctx).map(Blob::new);
    // let sin_cos = <Operators as gpt2::Operators>::build_sin_cos(dt_embd, nctx, dh, &ThisThread);
    let sin_cos = <Operators as gpt2::Operators>::build_sin_cos(dt_embd, nctx, 64, &ThisThread);
    let indices = RandomSample::build_indices(nvoc, &ThisThread);

    let sample = RandomSample::new(&Cpu);

    test_utils::test_infer(eos, tokenizer, &prompt, max_steps, |input, pos| {
        let mut embd = model.meta.embd(input.len()).map(Blob::new);
        let mut logits = model.meta.logits(1).map(Blob::new);

        let d = embd.get().len() / input.len();
        for (i, &tok) in input.iter().enumerate() {
            embd.get_mut()[i * d..][..d]
                .copy_from_slice(&model.token_embd[tok as usize * d..][..d]);
        }

        worker
            .launch(
                LlamaArgs {
                    embd: embd.map_slice_mut(),
                    logits: logits.map_slice_mut(),
                    sin_cos: sin_cos.map_slice(),
                    requests: vec![LlamaRequest {
                        cache: cache.map_slice_mut(),
                        seq_len: input.len(),
                        out_len: 1,
                        pos,
                    }],
                    num_tokens: input.len(),
                    max_seq_len: input.len(),
                    max_att_len: pos + input.len(),
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let mut pair = KVPair::new(0, f16::ZERO);
        let mut pairs = Tensor::kv_pair_vec(1, |_| unsafe {
            from_raw_parts_mut(&mut pair as *mut _ as _, size_of_val(&pair))
        });

        sample
            .launch(
                &mut pairs,
                &logits,
                &indices,
                sample_args,
                &mut [],
                &ThisThread,
            )
            .unwrap();

        pair.idx() as _
    });
}
