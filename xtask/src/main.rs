mod bench;
mod chat;
mod generate;
mod logger;
mod service;

use bytesize::ByteSize;
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressState, ProgressStyle};
use llama_cu::{SampleArgs, Service};
use log::info;
use regex::Regex;
use std::{
    collections::HashMap,
    ffi::c_int,
    fmt::Write,
    path::PathBuf,
    sync::LazyLock,
    time::{Duration, Instant},
};

#[macro_use]
extern crate clap;

fn main() {
    logger::init();
    use Commands::*;
    match Cli::parse().command {
        Generate(args) => args.generate(),
        Chat(args) => args.chat(),
        Service(args) => args.service(),
        Bench(args) => args.bench(),
    }
}

#[derive(Parser)]
#[clap(name = "InfiniLM")]
#[clap(version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// text generation
    Generate(generate::GenerateArgs),
    /// chat in console
    Chat(chat::ChatArgs),
    /// web service
    Service(service::ServiceArgs),
    /// batched benchmark
    Bench(bench::BenchArgs),
}

#[derive(Args)]
struct BaseArgs {
    model: PathBuf,
    #[clap(long)]
    gpus: Option<String>,
    #[clap(long)]
    max_steps: Option<usize>,
    #[clap(long)]
    no_cuda_graph: bool,
    #[clap(long)]
    temperature: Option<f32>,
    #[clap(long)]
    top_p: Option<f32>,
}

impl BaseArgs {
    fn gpus(&self) -> Box<[c_int]> {
        parse_gpus(self.gpus.as_deref())
    }

    fn max_steps(&self) -> usize {
        self.max_steps.unwrap_or(1000)
    }

    fn sample_args(&self) -> SampleArgs {
        SampleArgs::new(
            self.temperature.unwrap_or(0.),
            self.top_p.unwrap_or(1.),
            usize::MAX,
        )
        .unwrap()
    }
}

fn parse_gpus(config: Option<&str>) -> Box<[c_int]> {
    config
        .as_ref()
        .map(|devices| {
            static NUM_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());
            NUM_REGEX
                .find_iter(devices)
                .map(|c| c.as_str().parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|| [0].into())
}

fn progress_bar(service: &mut Service) {
    let m = MultiProgress::new();
    let style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:50}] {bytes}/{total_bytes} ({eta})",
    )
    .unwrap()
    .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
        write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
    });

    let mut pbs = HashMap::new();

    let time = Instant::now();
    service.wait_loading(Duration::from_millis(40), |p| {
        for &(id, pos, len) in p {
            if len > 0 {
                pbs.entry(id)
                    .or_insert_with(|| {
                        m.add(ProgressBar::new((len + 1) as _).with_style(style.clone()))
                    })
                    .set_position(pos as _)
            }
        }
    });
    let time = time.elapsed();
    service.wait_until_ready();
    m.clear().unwrap();

    let size = pbs.values().map(|pb| pb.length().unwrap()).sum::<u64>();
    let speed = size as f64 / time.as_secs_f64();
    info!(
        "weight loaded to {} gpus in {time:.2?}, total = {}, speed = {}/s",
        pbs.len(),
        ByteSize::b(size as _).display(),
        ByteSize::b(speed as _).display(),
    )
}

mod macros {
    macro_rules! print_now {
        ($($arg:tt)*) => {{
            use std::io::Write;

            print!($($arg)*);
            std::io::stdout().flush().unwrap();
        }};
    }

    pub(crate) use print_now;
}
