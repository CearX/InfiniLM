use hyper::Method;
use openai_struct::{
    ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta, CompletionUsage,
    CreateChatCompletionResponse, CreateChatCompletionResponseChoices,
    CreateChatCompletionStreamResponse, CreateChatCompletionStreamResponseChoices,
    CreateCompletionResponse, CreateCompletionResponseChoices, CreateCompletionResponseLogprobs,
    FinishReason, Model,
};
use serde::Serialize;

const CHAT_COMPLETION_OBJECT: &str = "chat.completion.chunk";
const COMPLETION_OBJECT: &str = "text_completion";
pub(crate) const GET_MODELS: (&Method, &str) = (&Method::GET, "/models");
pub(crate) const POST_CHAT_COMPLETIONS: (&Method, &str) = (&Method::POST, "/chat/completions");
pub(crate) const POST_COMPLETIONS: (&Method, &str) = (&Method::POST, "/completions");

pub(crate) const BLACKLISTED_SIGNAL: &str = "<Blacklisted>";

pub(crate) fn create_models(models: impl IntoIterator<Item = String>) -> impl Serialize {
    #[derive(Serialize)]
    struct Response {
        object: &'static str,
        data: Vec<Model>,
    }

    Response {
        object: "list",
        data: models
            .into_iter()
            .map(|id| Model {
                id,
                object: "model".into(),
                owned_by: "QYLab".into(),
                created: 0,
            })
            .collect(),
    }
}

pub(crate) fn chat_completion_response(
    id: usize,
    created: i32,
    model: String,
    think: Option<String>,
    answer: Option<String>,
    [prompt_tokens, total_tokens]: [usize; 2],
    finish_reason: Option<FinishReason>,
) -> CreateChatCompletionResponse {
    let choices = vec![CreateChatCompletionResponseChoices {
        message: ChatCompletionResponseMessage {
            content: answer.unwrap_or_default(),
            reasoning_content: think,
            ..Default::default()
        },
        finish_reason,
        ..Default::default()
    }];
    CreateChatCompletionResponse {
        id: format!("InfiniLM-Service-chatcmpl-{id:#08x}"),
        object: CHAT_COMPLETION_OBJECT.to_string(),
        model,
        choices,
        created,
        usage: Some(CompletionUsage {
            completion_tokens: (total_tokens - prompt_tokens) as _,
            prompt_tokens: prompt_tokens as _,
            total_tokens: total_tokens as _,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        }),
        ..Default::default()
    }
}

pub(crate) fn chat_completion_response_stream(
    id: usize,
    created: i32,
    model: String,
    think: Option<String>,
    answer: Option<String>,
    finish_reason: Option<FinishReason>,
) -> CreateChatCompletionStreamResponse {
    let choices = vec![CreateChatCompletionStreamResponseChoices {
        delta: ChatCompletionStreamResponseDelta {
            reasoning_content: think,
            content: answer,
            ..Default::default()
        },
        finish_reason,
        ..Default::default()
    }];
    CreateChatCompletionStreamResponse {
        id: format!("InfiniLM-Service-chatcmpl-{id:#08x}"),
        object: CHAT_COMPLETION_OBJECT.to_string(),
        created,
        model,
        choices,
        ..Default::default()
    }
}

pub(crate) fn completion_response(
    id: usize,
    created: i32,
    model: String,
    text: String,
    finish_reason: Option<FinishReason>,
) -> CreateCompletionResponse {
    let finish_reason = match finish_reason {
        Some(FinishReason::Stop) => "stop",
        Some(FinishReason::Length) => "length",
        Some(FinishReason::ContentFilter) => "content_filter",
        Some(FinishReason::ToolCalls) => "tool_calls",
        Some(FinishReason::FunctionCall) => "function_call",
        None => "",
    }
    .to_string();
    let choices = vec![CreateCompletionResponseChoices {
        text,
        finish_reason,
        index: 0,
        logprobs: CreateCompletionResponseLogprobs {
            text_offset: None,
            token_logprobs: None,
            tokens: None,
            top_logprobs: None,
        },
    }];
    CreateCompletionResponse {
        id: format!("InfiniLM-Service-cmpl-{id:#08x}"),
        object: COMPLETION_OBJECT.to_string(),
        created,
        model,
        choices,
        system_fingerprint: None,
        usage: None,
    }
}
