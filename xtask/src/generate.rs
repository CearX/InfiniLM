use crate::{BaseArgs, macros::print_now, progress_bar};
use llama_cu::{
    Message, Received, Service, Session, SessionId, TextBuf, build_3d_pos_ids, image_from_env,
    model_from_env, qw2vl_infer,
};
use log::info;
use std::time::{Duration, Instant};

#[derive(Args)]
pub struct GenerateArgs {
    #[clap(flatten)]
    base: BaseArgs,
    #[clap(short, long)]
    prompt: Option<String>,
    #[clap(short = 't', long)]
    use_template: bool,
    #[clap(short = 'm', long)]
    multimodal: bool,
}

impl GenerateArgs {
    pub fn generate(self) {
        let Self {
            base,
            prompt,
            use_template,
            multimodal,
        } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();
        let sample_args = base.sample_args();
        let mut prompt = if multimodal {
            prompt.unwrap_or(
                "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>
<|im_start|>assistant
"
                .into(),
            )
        } else {
            prompt.unwrap_or("Once upon a time,".into())
        };

        // 保持图像嵌入数据的生命周期，确保指针在推理过程中保持有效
        let img_embd_holder = if multimodal {
            let model = model_from_env();
            let image = image_from_env();
            let (img_embd, img_info_0) = qw2vl_infer(model, image, false);
            let [t, h, w, d_patch, img_token_len] = img_info_0;
            let mrope_3d_pos_ids = build_3d_pos_ids(t, h, w, d_patch, 15, 10);
            let img_info = [
                img_embd.as_ptr() as *const u8 as usize,
                img_token_len,
                15, // image_token start position
            ];
            println!("img_info: {:?}", img_info);

            Some((img_embd, img_token_len, img_info, mrope_3d_pos_ids))
        } else {
            None
        };

        let (img_token_len, img_info, mrope_3d_pos_ids) =
            if let Some((_, img_token_len, img_info, mrope_3d_pos_ids)) = &img_embd_holder {
                (
                    Some(*img_token_len),
                    Some(*img_info),
                    Some(mrope_3d_pos_ids.clone()),
                )
            } else {
                (None, None, None)
            };

        let mut service = Service::new(
            base.model,
            img_info,
            mrope_3d_pos_ids,
            multimodal,
            &gpus,
            !base.no_cuda_graph,
        );
        progress_bar(&mut service);

        let term = service.terminal();

        if use_template {
            prompt = term.render(&[Message::user(&prompt)])
        }
        print_now!("{prompt}");

        let session = Session {
            id: SessionId(0),
            sample_args,
            cache: term.new_cache(),
        };
        let mut tokens = term.tokenize(&prompt);
        println!("tokens: {:?}", tokens);
        println!("tokens.len(): {:?}", tokens.len());
        println!("img_token_len: {:?}", img_token_len);
        // 扩充图像占位符到图像嵌入长度
        if multimodal {
            let placeholder_token: u32 = 151655;

            for i in 0..tokens.len() {
                if tokens[i] == placeholder_token {
                    tokens.remove(i);
                    for j in 0..img_token_len.unwrap() {
                        tokens.insert(i + j, placeholder_token);
                    }
                    break;
                }
            }
        }

        term.start(session, &tokens, max_steps);

        let mut prefill = Duration::ZERO;
        let mut decode = Duration::ZERO;
        let mut ntoks = 0;
        let mut buf = TextBuf::new();
        loop {
            let time = Instant::now();
            let Received { sessions, outputs } = service.recv(Duration::from_millis(50));
            if prefill.is_zero() {
                prefill = time.elapsed()
            } else {
                decode += time.elapsed()
            }

            for (_, tokens) in outputs {
                ntoks += tokens.len();
                print_now!("{}", service.terminal().decode(&tokens, &mut buf))
            }
            if !sessions.is_empty() {
                break;
            }
        }
        println!();
        info!("prefill = {prefill:?}, decode = {decode:?}");
        info!(
            "n toks = {ntoks}, perf: {:?}/tok, {}tok/s",
            decode / ntoks as _,
            ntoks as f64 / decode.as_secs_f64()
        )
    }
}
