use super::{blacklist_checker::BlacklistChecker, cache_manager::CacheManager, error::Error};
use crate::{
    progress_bar,
    service::{ModelConfig, openai::BLACKLISTED_SIGNAL},
};
use llama_cu::{
    Message, Received, ReturnReason, SampleArgs, Service, SessionId, Terminal, TextBuf, utok,
};
use log::{debug, info};
use openai_struct::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest, CreateCompletionRequest,
    FinishReason,
};
use serde_json::Value;
use std::{collections::BTreeMap, sync::Mutex, time::Duration};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

pub(super) struct Model {
    max_tokens: usize,
    sampling: SampleArgs,
    think: [utok; 2],
    terminal: Terminal,
    sessions: Mutex<BTreeMap<SessionId, SessionInfo>>,
    cache_manager: Mutex<CacheManager>,
    blacklist_checker: Option<BlacklistChecker>,
}

pub(super) enum Output {
    Text { think: String, content: String },
    Finish(FinishReason),
}

struct SessionInfo {
    sender: UnboundedSender<Output>,
    buf: TextBuf,
    think: bool,
    tokens: Vec<utok>,
    accumulated_content: String, // Track all generated content for blacklist detection
}

impl Model {
    pub fn new(config: ModelConfig, use_cuda_graph: bool) -> (Self, Service) {
        let ModelConfig {
            path,
            gpus,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            think,
            blacklist,
        } = config;

        let mut service = Service::new(
            path,
            None,
            None,
            false,
            &gpus.unwrap_or(Box::new([0])),
            use_cuda_graph,
        );
        progress_bar(&mut service);

        let think = if think.unwrap_or(false) {
            let &[think] = &*service.terminal().encode("<think>") else {
                unreachable!()
            };
            let &[_think] = &*service.terminal().encode("</think>") else {
                unreachable!()
            };
            [think, _think]
        } else {
            [utok::MAX; 2]
        };

        let blacklist = blacklist
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect::<Vec<String>>();

        let model = Model {
            max_tokens: max_tokens.unwrap_or(2 << 10),
            sampling: SampleArgs::new(
                temperature.unwrap_or(0.),
                top_p.unwrap_or(1.),
                usize::MAX,
                repetition_penalty.unwrap_or(1.),
            )
            .unwrap(),
            think,
            terminal: service.terminal().clone(),
            sessions: Default::default(),
            cache_manager: Default::default(),
            blacklist_checker: if blacklist.is_empty() {
                None
            } else {
                Some(BlacklistChecker::new(blacklist))
            },
        };

        (model, service)
    }

    pub fn serve(&self, service: &mut Service) {
        let [think, _think] = self.think;
        loop {
            let Received { sessions, outputs } = service.recv(Duration::from_millis(10));

            let mut sessions_guard = self.sessions.lock().unwrap();
            // 先处理输出
            for (session_id, tokens) in outputs {
                if tokens.is_empty() {
                    continue;
                }

                let session_info = sessions_guard.get_mut(&session_id).unwrap();
                // 更新 session_info
                session_info.tokens.extend(&tokens);

                let mut tokens = &tokens[..];
                if tokens.first().is_some_and(|t| t == &think) {
                    session_info.think = true;
                    tokens = &tokens[1..]
                }
                let think = if session_info.think {
                    if let Some(_think) = tokens.iter().position(|t| *t == _think) {
                        session_info.think = false;
                        let think = &tokens[.._think];
                        tokens = &tokens[_think + 1..];
                        think
                    } else {
                        let think = tokens;
                        tokens = &[];
                        think
                    }
                } else {
                    &[]
                };

                let think = self.terminal.decode(think, &mut session_info.buf);
                let content = self.terminal.decode(tokens, &mut session_info.buf);
                debug!("解码完成：{tokens:?} -> {think:?} | {content:?}");

                // Truncate accumulated_content to save memory, keeping a suffix long enough
                // for the longest blacklisted word.
                let max_word_len = self.get_max_blacklist_word_length();
                if max_word_len > 0 {
                    // Accumulate content for blacklist detection only when blacklist is not empty
                    session_info.accumulated_content.push_str(&content);

                    let current_len = session_info.accumulated_content.len();
                    if current_len > max_word_len {
                        let start_pos = current_len - max_word_len;
                        // Find the first character boundary at or after `start_pos`.
                        // This is guaranteed to succeed because `is_char_boundary(len)` is always true.
                        let truncate_pos = (start_pos..=current_len)
                            .find(|&i| session_info.accumulated_content.is_char_boundary(i))
                            .unwrap();
                        session_info.accumulated_content.drain(..truncate_pos);
                    }
                }

                // Check for blacklisted content in the accumulated content
                if self.contains_blacklisted_word(&session_info.accumulated_content) {
                    debug!(
                        "Blacklisted content detected in session {:?}: {}",
                        session_id, session_info.accumulated_content
                    );

                    // Send BLACKLISTED_SIGNAL before stopping
                    if session_info
                        .sender
                        .send(Output::Text {
                            think: String::new(),
                            content: BLACKLISTED_SIGNAL.to_string(),
                        })
                        .is_err()
                    {
                        info!("{session_id:?} 客户端连接已关闭");
                    }

                    // Stop the session immediately
                    self.terminal.stop(session_id);
                    // Send finish signal
                    if session_info
                        .sender
                        .send(Output::Finish(FinishReason::Stop))
                        .is_err()
                    {
                        info!("{session_id:?} 客户端连接已关闭");
                    }
                    continue;
                }

                if session_info
                    .sender
                    .send(Output::Text { think, content })
                    .is_err()
                {
                    info!("{session_id:?} 客户端连接已关闭");
                    self.terminal.stop(session_id);
                }
            }

            // 处理会话结束
            if !sessions.is_empty() {
                for (session, reason) in sessions {
                    let SessionInfo { tokens, sender, .. } =
                        sessions_guard.remove(&session.id).unwrap();
                    let reason = match reason {
                        ReturnReason::Finish => {
                            // 正常完成，插回 cache
                            self.cache_manager
                                .lock()
                                .unwrap()
                                .insert(tokens, session.cache);
                            info!("{:?} 正常完成", session.id);
                            FinishReason::Stop
                        }
                        ReturnReason::Length | ReturnReason::CacheOverflow => {
                            info!("{:?} 超长完成", session.id);
                            FinishReason::Length
                        }
                    };

                    sender
                        .send(Output::Finish(reason))
                        .unwrap_or_else(|_| info!("{:?} 发送正常完成失败", session.id));
                }
            }
        }
    }

    pub fn complete_chat(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<UnboundedReceiver<Output>, Error> {
        let CreateChatCompletionRequest {
            messages,
            max_tokens,
            temperature,
            top_p,
            frequency_penalty,
            ..
        } = req;
        let (sender, receiver) = mpsc::unbounded_channel();

        let max_tokens = max_tokens.map_or(self.max_tokens, |n| n as _);
        let sample_args = SampleArgs::new(
            temperature.unwrap_or(self.sampling.temperature),
            top_p.unwrap_or(self.sampling.top_p),
            self.sampling.top_k,
            frequency_penalty.unwrap_or(self.sampling.repetition_penalty),
        )
        .unwrap();

        debug!("received completions: {messages:#?}");

        // 用于持有所有权
        let mut content_list = Vec::with_capacity(messages.len());
        for msg in &messages {
            let msg = match msg {
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(msg),
                    ..
                }) => msg,
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(msg),
                    ..
                }) => msg,
                ChatCompletionRequestMessage::Assistant(
                    ChatCompletionRequestAssistantMessage {
                        content: Some(ChatCompletionRequestAssistantMessageContent::Text(msg)),
                        ..
                    },
                ) => msg,
                msg => return Err(Error::msg_not_supported(msg)),
            };
            content_list.push(msg)
        }

        let messages = messages
            .iter()
            .zip(&content_list)
            .map(|(message, content)| match message {
                ChatCompletionRequestMessage::User(_) => Message::user(content.as_str()),
                ChatCompletionRequestMessage::System(_) => Message::system(content.as_str()),
                ChatCompletionRequestMessage::Assistant(_) => Message::assistant(content.as_str()),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        debug!("received messages: {messages:#?}");
        let text = self.terminal.render(&messages);
        debug!("received prompt: {text}");
        let tokens = self.terminal.tokenize(&text);

        let (id, tokens) = self.cache_manager.lock().unwrap().send(
            &self.terminal,
            tokens,
            sample_args,
            max_tokens,
        );

        let session_info = SessionInfo {
            sender,
            tokens,
            buf: TextBuf::new(),
            think: false,
            accumulated_content: String::new(),
        };
        assert!(
            self.sessions
                .lock()
                .unwrap()
                .insert(id, session_info,)
                .is_none()
        );

        Ok(receiver)
    }

    pub fn complete(
        &self,
        req: CreateCompletionRequest,
    ) -> Result<UnboundedReceiver<Output>, Error> {
        let CreateCompletionRequest {
            prompt,
            max_tokens,
            temperature,
            top_p,
            frequency_penalty,
            ..
        } = req;

        let prompt_text = match &prompt {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            _ => return Err(Error::msg_not_supported(&prompt)),
        };

        let max_tokens = max_tokens.map_or(self.max_tokens, |n| n as _);
        let sample_args = SampleArgs::new(
            temperature.unwrap_or(self.sampling.temperature),
            top_p.unwrap_or(self.sampling.top_p),
            self.sampling.top_k,
            frequency_penalty.unwrap_or(self.sampling.repetition_penalty),
        )
        .unwrap();

        debug!("received completion prompt: {prompt_text:?}");
        let tokens = self.terminal.tokenize(&prompt_text);

        let (sender, receiver) = mpsc::unbounded_channel();

        let (id, tokens) = self.cache_manager.lock().unwrap().send(
            &self.terminal,
            tokens,
            sample_args,
            max_tokens,
        );

        let session_info = SessionInfo {
            sender,
            tokens,
            buf: TextBuf::new(),
            think: false,
            accumulated_content: String::new(),
        };
        assert!(
            self.sessions
                .lock()
                .unwrap()
                .insert(id, session_info,)
                .is_none()
        );

        Ok(receiver)
    }

    /// Get the maximum length of any blacklisted word (in characters, not bytes)
    pub fn get_max_blacklist_word_length(&self) -> usize {
        self.blacklist_checker
            .as_ref()
            .map(|checker| checker.get_max_word_length())
            .unwrap_or(0)
    }

    /// Check if text contains any blacklisted words
    pub fn contains_blacklisted_word(&self, text: &str) -> bool {
        self.blacklist_checker
            .as_ref()
            .map(|checker| checker.contains_word(text))
            .unwrap_or(false)
    }
}
