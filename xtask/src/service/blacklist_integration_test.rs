use super::client::{create_client_with_headers, send_single_request};
use super::{ModelConfig, ServiceArgs};
use crate::logger;
use crate::service::openai::BLACKLISTED_SIGNAL;
use log::info;
use openai_struct::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent,
    CreateChatCompletionRequest,
};
use std::time::Duration;
use tokio::runtime::Runtime;
use tokio::time::timeout;

/// Integration test demonstrating blacklist functionality
/// This test shows how to:
/// 1. Configure a model with blacklist
/// 2. Start a service with blacklist enabled
/// 3. Test blacklist detection through the API
#[test]
#[ignore = "Requires TEST_PORT environment variable and running service"]
fn test_blacklist_integration() {
    // Skip if no test port is configured
    let port = match std::env::var("TEST_PORT") {
        Ok(port) => port.parse().unwrap(),
        Err(_) => {
            println!("Skipping blacklist integration test - no TEST_PORT configured");
            return;
        }
    };

    logger::init();

    Runtime::new().unwrap().block_on(async move {
        info!("Testing blacklist integration with service");

        let (client, headers) = create_client_with_headers();

        // Test scenarios for blacklist functionality
        test_blacklist_scenarios(port, &client, &headers).await;

        info!("Blacklist integration test completed");
    });
}

async fn test_blacklist_scenarios(
    port: u16,
    client: &reqwest::Client,
    headers: &reqwest::header::HeaderMap,
) {
    info!("Running blacklist scenarios");

    // Scenario 1: Test normal content (should not be blocked)
    info!("Scenario 1: Testing normal content");
    let normal_prompts = vec![
        "Tell me a story about a cat",
        "What is the weather like?",
        "Explain photosynthesis",
        "Write a poem about nature",
        "你好，世界！",       // Hello world in Chinese
        "今天天气很好",       // The weather is nice today in Chinese
        "请介绍一下人工智能", // Please introduce artificial intelligence in Chinese
    ];

    for prompt in normal_prompts {
        test_prompt(port, client, headers, prompt, false).await;
    }

    // Scenario 2: Test potentially blocked content
    info!("Scenario 2: Testing potentially blocked content");
    let potentially_blocked_prompts = vec![
        "Tell me about dangerous activities",
        "How to leak information",
        "Explain bad words",
        "Describe dangerous weapons",
        "请介绍一下敏感词的内容", // Please introduce sensitive word content in Chinese
        "什么是违禁词？",         // What are forbidden words? in Chinese
        "如何避免不良信息",       // How to avoid inappropriate content in Chinese
        "非法内容的定义",         // Definition of illegal content in Chinese
    ];

    for prompt in potentially_blocked_prompts {
        test_prompt(port, client, headers, prompt, true).await;
    }

    // Scenario 3: Test edge cases
    info!("Scenario 3: Testing edge cases");
    let edge_case_prompts = vec![
        "The word 'danger' appears in this sentence",
        "Information can leak through various channels",
        "Some words might be considered bad",
        "This is a safe sentence with no issues",
        "这个句子包含敏感词", // This sentence contains sensitive word in Chinese
        "违禁词的定义是什么？", // What is the definition of forbidden word? in Chinese
        "如何识别不良信息",   // How to identify inappropriate content in Chinese
        "非法内容的危害",     // The harm of illegal content in Chinese
    ];

    for prompt in edge_case_prompts {
        test_prompt(port, client, headers, prompt, true).await;
    }

    // Scenario 4: Test longer blacklisted words (suffix optimization)
    info!("Scenario 4: Testing longer blacklisted words (suffix optimization)");
    let longer_word_prompts = vec![
        "This content contains verylongblacklistedword in the middle",
        "The word extremelylongblacklistedcontent should be detected",
        "supercalifragilisticexpialidocious is a very long word",
        "pneumonoultramicroscopicsilicovolcanoconiosiss is the longest word",
        "Normal content that ends with verylongblacklistedword",
        "A sentence with extremelylongblacklistedcontent at the end",
        "Content with supercalifragilisticexpialidocious split across chunks",
    ];

    for prompt in longer_word_prompts {
        test_prompt(port, client, headers, prompt, true).await;
    }

    // Scenario 5: Test Chinese longer content
    info!("Scenario 5: Testing Chinese longer content");
    let chinese_longer_prompts = vec![
        "这是一段很长的内容，其中包含敏感词，应该被检测到",
        "在长文本中，违禁词的出现应该被正确识别",
        "不良信息可能隐藏在很长的段落中",
        "非法内容有时候会出现在文本的末尾部分",
        "正常的中文内容，不包含任何违禁词汇",
        "包含敏感词的长段落测试",
    ];

    for prompt in chinese_longer_prompts {
        test_prompt(port, client, headers, prompt, true).await;
    }
}

/// Create a proper OpenAI API request body for chat completions
fn create_openai_chat_request(prompt: &str, model_name: &str) -> String {
    serde_json::to_string(&CreateChatCompletionRequest {
        model: model_name.to_string(),
        messages: vec![ChatCompletionRequestMessage::User(
            openai_struct::ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(prompt.to_string()),
                name: None,
            },
        )],
        metadata: None,
        service_tier: None,
        audio: None,
        function_call: None,
        functions: None,
        max_completion_tokens: None,
        max_tokens: Some(256),
        modalities: None,
        n: None,
        parallel_tool_calls: None,
        prediction: None,
        reasoning_effort: None,
        response_format: None,
        store: None,
        tool_choice: None,
        tools: None,
        top_logprobs: None,
        web_search_options: None,
        frequency_penalty: None,
        logit_bias: None,
        logprobs: None,
        presence_penalty: None,
        seed: None,
        stop: None,
        stream: Some(true),
        stream_options: None,
        temperature: None,
        top_p: None,
        user: None,
    })
    .unwrap()
}

async fn test_prompt(
    port: u16,
    client: &reqwest::Client,
    headers: &reqwest::header::HeaderMap,
    prompt: &str,
    might_be_blocked: bool,
) {
    let req_body = create_openai_chat_request(prompt, "minicpm3-1b-awq-4");

    match timeout(
        Duration::from_secs(30),
        send_single_request(port, client, headers, req_body, None),
    )
    .await
    {
        Ok(Ok((_, _, content, duration))) => {
            let is_blocked =
                content.contains(BLACKLISTED_SIGNAL) || content.is_empty() || content.len() < 10;

            if might_be_blocked && is_blocked {
                info!("Correctly blocked: '{prompt}' (took {duration:?}) (content: '{content}')");
            } else if !might_be_blocked && !is_blocked {
                info!("Correctly allowed: '{prompt}' (took {duration:?}) (content: '{content}')");
            } else if might_be_blocked && !is_blocked {
                info!("Not blocked as expected: '{prompt}' (content: '{content}')");
            } else {
                info!("Unexpectedly blocked: '{prompt}' (content: '{content}')");
            }
        }
        Ok(Err(e)) => {
            info!("Request failed for '{prompt}': {e}");
        }
        Err(_) => {
            info!("Request timeout for '{prompt}'");
        }
    }
}

/// Test helper to create a model configuration with blacklist
fn create_blacklist_config() -> ModelConfig {
    ModelConfig {
        path: "test_model.gguf".to_string(),
        gpus: Some(Box::new([0])),
        max_tokens: Some(200),
        temperature: Some(0.7),
        top_p: Some(1.0),
        repetition_penalty: Some(1.0),
        think: Some(false),
        blacklist: Some(vec![
            // English blacklisted words
            "danger".to_string(),
            "leak".to_string(),
            "badword".to_string(),
            "weapon".to_string(),
            "explosive".to_string(),
            // Chinese blacklisted words
            "敏感词".to_string(),   // sensitive word
            "违禁词".to_string(),   // forbidden word
            "不良信息".to_string(), // inappropriate content
            "非法内容".to_string(), // illegal content
            // Longer words to test suffix optimization
            "verylongblacklistedword".to_string(),
            "extremelylongblacklistedcontent".to_string(),
            "supercalifragilisticexpialidocious".to_string(),
            "pneumonoultramicroscopicsilicovolcanoconiosiss".to_string(),
        ]),
    }
}

/// Test helper to demonstrate blacklist configuration
#[test]
#[ignore = "Integration test - requires service setup"]
fn test_blacklist_configuration() {
    logger::init();

    info!("Testing blacklist configuration");

    let config = create_blacklist_config();

    // Verify blacklist configuration
    assert!(config.blacklist.is_some());
    let blacklist = config.blacklist.as_ref().unwrap();

    // Test English words
    assert!(blacklist.contains(&"danger".to_string()));
    assert!(blacklist.contains(&"leak".to_string()));
    assert!(blacklist.contains(&"badword".to_string()));

    // Test Chinese words
    assert!(blacklist.contains(&"敏感词".to_string()));
    assert!(blacklist.contains(&"违禁词".to_string()));
    assert!(blacklist.contains(&"不良信息".to_string()));
    assert!(blacklist.contains(&"非法内容".to_string()));

    // Test longer words
    assert!(blacklist.contains(&"verylongblacklistedword".to_string()));
    assert!(blacklist.contains(&"extremelylongblacklistedcontent".to_string()));
    assert!(blacklist.contains(&"supercalifragilisticexpialidocious".to_string()));
    assert!(blacklist.contains(&"pneumonoultramicroscopicsilicovolcanoconiosiss".to_string()));

    info!("Blacklist configuration is correct");

    // Test case sensitivity for English words
    let test_words = vec!["DANGER", "LeAk", "BADWORD"];
    for word in test_words {
        let lower_word = word.to_lowercase();
        assert!(
            blacklist.iter().any(|bw| bw == &lower_word),
            "Blacklist should contain '{word}' (lowercase: '{lower_word}')"
        );
    }

    info!("Blacklist case sensitivity test passed");

    // Test that we have words of varying lengths
    let word_lengths: Vec<usize> = blacklist.iter().map(|w| w.chars().count()).collect();
    let max_length = word_lengths.iter().max().unwrap();
    let min_length = word_lengths.iter().min().unwrap();

    info!("Blacklist word length range: {min_length} to {max_length} characters");
    assert!(
        *max_length > 10,
        "Should have words longer than 10 characters for suffix optimization test"
    );

    info!("Blacklist word length test passed");
}

/// Test helper to demonstrate service arguments with blacklist
#[test]
#[ignore = "Integration test - requires service setup"]
fn test_service_args_with_blacklist() {
    logger::init();

    info!("Testing service arguments with blacklist");

    // This would be how you'd configure the service with blacklist
    // In a real scenario, you'd load this from a TOML config file
    let service_args = ServiceArgs {
        file: "model.gguf".to_string(),
        port: 8080,
        no_cuda_graph: false,
        name: Some("test-model".to_string()),
        gpus: Some("0".to_string()),
        max_tokens: Some(200),
        temperature: Some(0.7),
        top_p: Some(1.0),
        repetition_penalty: Some(1.0),
        think: false,
    };

    info!("Service arguments configured");
    info!("Model file: {}", service_args.file);
    info!("Port: {}", service_args.port);
    info!("Max tokens: {:?}", service_args.max_tokens);

    // Note: In a real implementation, you'd need to modify ServiceArgs
    // to include blacklist configuration, or load it from a config file
}

/// Example TOML configuration for blacklist
#[test]
#[ignore = "Integration test - requires service setup"]
fn test_toml_blacklist_config() {
    logger::init();

    info!("Testing TOML blacklist configuration");

    // Example TOML configuration that would be used in practice
    let toml_config = r#"
[model]
path = "model.gguf"
gpus = [0]
max-tokens = 200
temperature = 0.7
top-p = 1.0
repetition-penalty = 1.0
think = false
blacklist = [
    # English blacklisted words
    "danger",
    "leak",
    "badword",
    "weapon",
    "explosive",
    # Chinese blacklisted words
    "敏感词",
    "违禁词",
    "不良信息",
    "非法内容",
    # Longer words to test suffix optimization
    "verylongblacklistedword",
    "extremelylongblacklistedcontent",
    "supercalifragilisticexpialidocious",
    "pneumonoultramicroscopicsilicovolcanoconiosiss"
]
"#;

    info!("Example TOML configuration:");
    info!("{toml_config}");

    // In practice, you'd parse this with:
    // let config: ModelConfig = toml::from_str(toml_config).unwrap();

    info!("TOML configuration example created");
}
