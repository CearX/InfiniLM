mod blacklist_checker;
mod cache_manager;
mod error;
mod model;
mod openai;
mod response;

use crate::{
    parse_gpus,
    service::{
        openai::{chat_completion_response, chat_completion_response_stream, completion_response},
        response::text_stream,
    },
};
use error::*;
use http_body_util::{BodyExt, combinators::BoxBody};
use hyper::{
    Request, Response,
    body::{Bytes, Incoming},
    server::conn::http1,
    service::Service as HyperService,
};
use hyper_util::rt::TokioIo;
use llama_cu::Service;
use log::{info, warn};
use model::Model;
use openai::create_models;
use openai_struct::{CreateChatCompletionRequest, CreateCompletionRequest};
use response::error;
use response::json;
use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
    time::{SystemTime, UNIX_EPOCH},
};
use std::{ffi::c_int, fs::read_to_string, path::Path};
use std::{
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    pin::Pin,
    sync::Arc,
};
use tokio::net::TcpListener;
use tokio_stream::{StreamExt, wrappers::UnboundedReceiverStream};

#[derive(Args)]
pub struct ServiceArgs {
    file: String,

    #[clap(short, long)]
    port: u16,
    #[clap(long)]
    no_cuda_graph: bool,

    #[clap(long)]
    name: Option<String>,
    #[clap(long)]
    gpus: Option<String>,
    #[clap(long)]
    max_tokens: Option<usize>,
    #[clap(long)]
    temperature: Option<f32>,
    #[clap(long)]
    top_p: Option<f32>,
    #[clap(long)]
    repetition_penalty: Option<f32>,
    #[clap(long)]
    think: bool,
}

#[derive(serde::Deserialize, Debug)]
pub struct ModelConfig {
    pub path: String,
    pub gpus: Option<Box<[c_int]>>,
    #[serde(rename = "max-tokens")]
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    #[serde(rename = "top-p")]
    pub top_p: Option<f32>,
    #[serde(rename = "repetition-penalty")]
    pub repetition_penalty: Option<f32>,
    pub think: Option<bool>,
    pub blacklist: Option<Vec<String>>,
}

impl ServiceArgs {
    pub fn service(self) {
        let Self {
            file,
            port,
            no_cuda_graph,
            name,
            gpus,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            think,
        } = self;

        let path = Path::new(&file);
        let model_configs: HashMap<_, _> = match path.extension().map(|s| s.to_str()) {
            Some(Some("toml")) => toml::from_str(&read_to_string(path).unwrap()).unwrap(),
            Some(Some("gguf")) => [(
                name.as_deref()
                    .unwrap_or_else(|| path.file_stem().unwrap().to_str().unwrap())
                    .to_string(),
                ModelConfig {
                    path: file.clone(),
                    gpus: Some(parse_gpus(gpus.as_deref())),
                    max_tokens,
                    temperature,
                    top_p,
                    repetition_penalty,
                    think: Some(think),
                    blacklist: None,
                },
            )]
            .into(),
            _ => panic!("file must be a gguf model or a toml config"),
        };
        for (name, cfg) in &model_configs {
            info!("{name}: {cfg:?}")
        }

        let mut handles = Vec::with_capacity(model_configs.len());
        let models = model_configs
            .into_iter()
            .map(|(name, config)| {
                let (model, service) = Model::new(config, !no_cuda_graph);
                let model = Arc::new(model);
                handles.push((model.clone(), service));
                (name, model)
            })
            .collect();

        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(start_infer_service(models, handles, port))
            .unwrap()
    }
}

async fn start_infer_service(
    models: HashMap<String, Arc<Model>>,
    handles: Vec<(Arc<Model>, Service)>,
    port: u16,
) -> std::io::Result<()> {
    let app = App(Arc::new(models));

    let _handles = handles
        .into_iter()
        .map(|(model, mut service)| tokio::task::spawn_blocking(move || model.serve(&mut service)))
        .collect::<Vec<_>>();

    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");

    let listener = TcpListener::bind(addr).await?;
    loop {
        let app = app.clone();
        info!("ready to accept");
        let (stream, x) = listener.accept().await?;
        info!("listen from {x}");
        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), app)
                .await
            {
                warn!("Error serving connection: {err:?}")
            }
        });
    }
}

#[derive(Clone)]
struct App(Arc<HashMap<String, Arc<Model>>>);

impl HyperService<Request<Incoming>> for App {
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        match (req.method(), req.uri().path()) {
            openai::GET_MODELS => {
                let json = json(create_models(self.0.keys().cloned()));
                Box::pin(async move { Ok(json) })
            }
            openai::POST_COMPLETIONS => {
                let models = self.0.clone();
                Box::pin(async move {
                    let whole_body = req.collect().await?.to_bytes();
                    let req: CreateCompletionRequest = match serde_json::from_slice(&whole_body) {
                        Ok(req) => req,
                        Err(e) => return Ok(error(Error::WrongJson(e))),
                    };
                    let model_name = match &req.model {
                        serde_json::Value::String(s) => s.clone(),
                        _ => {
                            return Ok(error(Error::ModelNotFound(
                                "model field must be a string".to_string(),
                            )));
                        }
                    };
                    let stream = req.stream.unwrap_or(true);
                    let model = match models.get(&model_name) {
                        Some(model) => model,
                        None => return Ok(error(Error::ModelNotFound(model_name))),
                    };
                    let mut receiver = match model.complete(req) {
                        Ok(receiver) => receiver,
                        Err(e) => return Ok(error(e)),
                    };

                    static ID: AtomicUsize = AtomicUsize::new(0);
                    let id = ID.fetch_add(1, SeqCst);
                    let created = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs() as i32;

                    if stream {
                        return Ok(text_stream(UnboundedReceiverStream::new(receiver).map(
                            move |output| {
                                let response = match output {
                                    model::Output::Text { content, .. } => completion_response(
                                        id,
                                        created,
                                        model_name.clone(),
                                        content,
                                        None,
                                    ),
                                    model::Output::Finish { reason, .. } => completion_response(
                                        id,
                                        created,
                                        model_name.clone(),
                                        String::new(),
                                        Some(reason),
                                    ),
                                };
                                serde_json::to_string(&response).unwrap()
                            },
                        )));
                    }

                    let mut think_ = String::new();
                    let mut content_ = String::new();
                    let mut reason_ = None;
                    while let Some(output) = receiver.recv().await {
                        match output {
                            model::Output::Text { think, content } => {
                                think_.push_str(&think);
                                content_.push_str(&content);
                            }
                            model::Output::Finish { reason, .. } => {
                                assert!(reason_.replace(reason).is_none())
                            }
                        }
                    }

                    let response = completion_response(id, created, model_name, content_, reason_);
                    Ok(json(response))
                })
            }
            openai::POST_CHAT_COMPLETIONS => {
                let models = self.0.clone();
                Box::pin(async move {
                    let whole_body = req.collect().await?.to_bytes();

                    let req: CreateChatCompletionRequest = match serde_json::from_slice(&whole_body)
                    {
                        Ok(req) => req,
                        Err(e) => return Ok(error(Error::WrongJson(e))),
                    };

                    let model_name = req.model.clone();
                    let stream = req.stream.unwrap_or(true);

                    let model = match models.get(&model_name) {
                        Some(model) => model,
                        None => return Ok(error(Error::ModelNotFound(model_name))),
                    };

                    let mut receiver = match model.complete_chat(req) {
                        Ok(receiver) => receiver,
                        Err(e) => return Ok(error(e)),
                    };

                    static ID: AtomicUsize = AtomicUsize::new(0);

                    let id = ID.fetch_add(1, SeqCst);
                    let created = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs() as i32;

                    if stream {
                        return Ok(text_stream(UnboundedReceiverStream::new(receiver).map(
                            move |output| {
                                let response = match output {
                                    model::Output::Text { think, content } => {
                                        chat_completion_response_stream(
                                            id,
                                            created,
                                            model_name.clone(),
                                            Some(think).filter(|s| !s.is_empty()),
                                            Some(content).filter(|s| !s.is_empty()),
                                            None,
                                        )
                                    }
                                    model::Output::Finish { reason, .. } => {
                                        chat_completion_response_stream(
                                            id,
                                            created,
                                            model_name.clone(),
                                            None,
                                            None,
                                            Some(reason),
                                        )
                                    }
                                };
                                serde_json::to_string(&response).unwrap()
                            },
                        )));
                    }

                    let mut think_ = String::new();
                    let mut content_ = String::new();
                    let mut reason_ = None;
                    let mut num_tokens_ = [0, 0];
                    while let Some(output) = receiver.recv().await {
                        match output {
                            model::Output::Text { think, content } => {
                                think_.push_str(&think);
                                content_.push_str(&content);
                            }
                            model::Output::Finish { reason, num_tokens } => {
                                assert!(reason_.replace(reason).is_none());
                                num_tokens_ = num_tokens
                            }
                        }
                    }

                    let response = chat_completion_response(
                        id,
                        created,
                        model_name,
                        Some(think_).filter(|s| !s.is_empty()),
                        Some(content_).filter(|s| !s.is_empty()),
                        num_tokens_,
                        reason_,
                    );
                    Ok(json(response))
                })
            }
            // Return 404 Not Found for other routes.
            (method, uri) => {
                let msg = Error::not_found(method, uri);
                Box::pin(async move { Ok(error(msg)) })
            }
        }
    }
}

#[cfg(test)]
mod blacklist_integration_test;
#[cfg(test)]
mod client;
