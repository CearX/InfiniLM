mod app_session;
mod tui;

use crate::{BaseArgs, macros::print_now, progress_bar};
use llama_cu::{Message, Received, SampleArgs, Service, Session, SessionId, TextBuf};
use std::time::Duration;

#[derive(Args)]
pub struct ChatArgs {
    #[clap(flatten)]
    base: BaseArgs,
    #[clap(long)]
    tui: bool,
}

impl ChatArgs {
    pub fn chat(self) {
        let Self {
            base,
            tui: advanced,
        } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();
        let sample_args = base.sample_args();

        let mut service = Service::new(base.model, &gpus, !base.no_cuda_graph, false);
        progress_bar(&mut service);

        if !advanced {
            simple(service, max_steps, sample_args)
        } else {
            let terminal = ratatui::init();
            let result = tui::App::new(service, max_steps).run(terminal);
            ratatui::restore();
            result.unwrap()
        }
    }
}

fn simple(mut service: Service, max_steps: usize, sample_args: SampleArgs) {
    let mut session = Some(Session {
        id: SessionId(0),
        sample_args,
        cache: service.terminal().new_cache(),
    });

    let mut line = String::new();
    loop {
        line.clear();
        while line.is_empty() {
            print_now!("user> ");
            std::io::stdin().read_line(&mut line).unwrap();
            assert_eq!(line.pop(), Some('\n'));
        }

        {
            let t = service.terminal();
            let text = t.render(&[Message::user(&line)]);
            let tokens = t.tokenize(&text);
            t.start(session.take().unwrap(), &tokens, max_steps, None);
        }

        let mut buf = TextBuf::new();
        print_now!("assistant> ");
        for _ in 0..max_steps {
            let Received { sessions, outputs } = service.recv(Duration::MAX);

            for (_, tokens) in outputs {
                print_now!("{}", service.terminal().decode(&tokens, &mut buf))
            }
            if let Some((s, _)) = sessions.into_iter().next() {
                session = Some(s);
                break;
            }
        }
        println!();
        println!("=== over ===")
    }
}
