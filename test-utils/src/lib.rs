use gguf::{
    ext::{utok, Mmap},
    map_files, GGufMetaMapExt, GGufModel, Message, Tokenizer,
};
use std::{
    env::{var, var_os},
    fmt,
    iter::zip,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{
        mpsc::{self, Sender},
        Once,
    },
    time::{Duration, Instant},
};

pub struct Inference {
    pub model: Box<[Mmap]>,
    pub devices: Option<String>,
    pub prompt: String,
    pub as_user: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_steps: usize,
}

impl Inference {
    pub fn load() -> Option<Self> {
        static ONCE: Once = Once::new();
        ONCE.call_once(env_logger::init);

        let Some(path) = var_os("TEST_MODEL") else {
            println!("TEST_MODEL not set");
            return None;
        };
        let path = Path::new(&path);
        if !path.is_file() {
            println!("{path:?} not found");
            return None;
        }

        fn parse<T: FromStr>(name: &str, default: T) -> T {
            var(name)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default)
        }

        Some(Self {
            model: map_files(path),
            devices: var("DEVICES").ok(),
            prompt: var("PROMPT").unwrap_or_else(|_| String::from("Once upon a time,")),
            as_user: var("AS_USER").ok().is_some_and(|s| !s.is_empty()),
            temperature: parse("TEMPERATURE", 0.),
            top_p: parse("TOP_P", 1.),
            top_k: parse("TOP_K", usize::MAX),
            max_steps: parse("MAX_STEPS", usize::MAX),
        })
    }
}

pub fn load_roll_cache_size() -> usize {
    var("ROLL_CACHE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX)
}

pub fn image() -> Option<PathBuf> {
    var_os("TEST_IMAGE").map(PathBuf::from)
}

pub struct TokenizerAndPrompt {
    pub eos: utok,
    pub tokenizer: Tokenizer,
    pub prompt: String,
}

impl TokenizerAndPrompt {
    pub fn new(gguf: &GGufModel, mut prompt: String, as_user: bool) -> Self {
        let eos = gguf.tokenizer_ggml_eos_token_id().unwrap();
        let tokenizer = gguf.tokenizer();
        if as_user {
            if let Some(template) = gguf.chat_template(&tokenizer) {
                prompt = template
                    .render(
                        &[Message {
                            role: "user",
                            content: &prompt,
                        }],
                        true,
                    )
                    .unwrap()
            }
        }
        Self {
            eos,
            tokenizer,
            prompt,
        }
    }
}

pub fn test_infer(
    eos: utok,
    tokenizer: Tokenizer,
    prompt: &str,
    max_steps: usize,
    mut lm: impl FnMut(&[utok], usize) -> utok,
) {
    use cli_table::{format::Justify, print_stdout, Cell, CellStruct, Style, Table};
    macro_rules! print_now {
        ($($arg:tt)*) => {{
            use std::io::Write;

            print!($($arg)*);
            std::io::stdout().flush().unwrap();
        }};
    }

    print_now!("{prompt}");

    let mut tokens = tokenizer.encode(prompt);
    let num_prompt_tokens = tokens.len();

    let mut prefill = Duration::ZERO;
    let mut decode = Duration::ZERO;

    let mut pos = 0;
    for _ in 0..max_steps {
        let time = Instant::now();
        let next = lm(&tokens, pos);
        let time = time.elapsed();

        if prefill.is_zero() {
            prefill = time;
        } else {
            decode += time;
        }

        pos += tokens.len();
        if next == eos {
            break;
        }

        let piece = tokenizer.decode(next);
        print_now!("{piece}");
        tokens = vec![next];
    }

    let table = [
        row("total", prefill + decode, pos),
        row("prefill", prefill, num_prompt_tokens),
        row("decode", decode, pos - num_prompt_tokens),
    ]
    .table()
    .title(
        ["\\", "num tokens", "elapse", "time per token"]
            .into_iter()
            .map(|s| cell(s).bold(true)),
    )
    .bold(true);

    println!();
    println!();
    assert!(print_stdout(table).is_ok());

    fn cell(x: impl fmt::Display) -> CellStruct {
        x.cell().justify(Justify::Center)
    }
    fn row(name: &str, time: Duration, n: usize) -> [CellStruct; 4] {
        [
            cell(name).bold(true),
            cell(n),
            cell(format!("{:.3?}", time)),
            cell(format!("{:.3?}", time.div_f64(n as _))),
        ]
    }
}

#[cfg(feature = "llama")]
pub fn test_infer_paralle<'w>(
    model: &llama::LlamaStorage<&'w [u8]>,
    senders: Box<[mpsc::Sender<Task>]>,
    eos: utok,
    tokenizer: Tokenizer,
    prompt: &str,
    max_steps: usize,
) {
    use tensor::Blob;

    let (next, next_recv) = mpsc::channel();
    test_infer(eos, tokenizer, prompt, max_steps, |input, pos| {
        let mut embd = model.meta.embd(input.len()).map(Blob::new).take();

        let d = embd.len() / input.len();
        for (i, &tok) in input.iter().enumerate() {
            embd[i * d..][..d].copy_from_slice(&model.token_embd[tok as usize * d..][..d]);
        }

        for sender in &senders {
            sender
                .send(Task {
                    nt: input.len(),
                    pos,
                    embd: embd.as_ptr(),
                    next: next.clone(),
                })
                .unwrap()
        }
        next_recv.recv().unwrap()
    });
}

pub struct Task {
    pub nt: usize,
    pub pos: usize,
    pub embd: *const u8,
    pub next: mpsc::Sender<utok>,
}

unsafe impl Send for Task {}

pub struct WorkerSeed<N> {
    pub tasks: mpsc::Receiver<Task>,
    pub node: N,
}

impl<N> WorkerSeed<N> {
    pub fn new(nodes: Vec<N>) -> (Vec<Self>, Vec<Sender<Task>>) {
        let n = nodes.len();

        let mut tasks = Vec::with_capacity(n);
        let mut senders = Vec::with_capacity(n);
        for _ in 0..n {
            let (sender, receiver) = std::sync::mpsc::channel();
            tasks.push(receiver);
            senders.push(sender);
        }
        (
            zip(nodes, tasks)
                .map(|(node, tasks)| Self { node, tasks })
                .collect(),
            senders,
        )
    }
}
