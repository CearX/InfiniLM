use crate::{BaseArgs, macros::print_now, progress_bar};
use llama_cu::qw2vl_image_preprocess as get_image;
use llama_cu::{Message, Received, Service, Session, SessionId, TextBuf};
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
            prompt.unwrap_or("Describe this image,".into())
        } else {
            prompt.unwrap_or("Once upon a time,".into())
        };

        let mut service = Service::new(base.model, &gpus, !base.no_cuda_graph, multimodal);
        progress_bar(&mut service);

        let session = Session {
            id: SessionId(0),
            sample_args,
            cache: service.terminal().new_cache(),
        };
        print_now!("{prompt}");
        if use_template {
            prompt = service.terminal().render(&[Message::user(&prompt)])
        }
        if multimodal {
            service.terminal().start(
                session,
                &service.terminal().tokenize(&prompt),
                max_steps,
                Some(get_image()),
            );

            // qw2vl todo
        } else {
            service.terminal().start(
                session,
                &service.terminal().tokenize(&prompt),
                max_steps,
                None,
            );

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
}
