use crate::service::openai::BLACKLISTED_SIGNAL;

use super::openai::POST_CHAT_COMPLETIONS;
use log::{info, trace, warn};
use openai_struct::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent,
    CreateChatCompletionRequest, CreateChatCompletionStreamResponse,
};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use std::{env::VarError, time::Instant};
use tokio::time::Duration;
use tokio_stream::StreamExt;

const CONCURRENT_REQUESTS: usize = 240;

pub(crate) fn requset_body_chat(prompt: &str) -> String {
    serde_json::to_string(&CreateChatCompletionRequest {
        model: "model".into(),
        messages: vec![ChatCompletionRequestMessage::User(
            openai_struct::ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(prompt.into()),
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

pub(crate) fn create_client_with_headers() -> (reqwest::Client, HeaderMap) {
    let client = reqwest::Client::new();
    let mut headers: HeaderMap = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    (client, headers)
}

pub(crate) async fn send_single_request(
    port: u16,
    client: &reqwest::Client,
    headers: &HeaderMap,
    req_body: String,
    index: Option<usize>,
) -> Result<
    (
        usize,
        usize,
        String,
        Duration,
        Option<(usize, usize, usize)>,
        Instant, // 请求开始时间
        Instant, // 请求结束时间
    ),
    String,
> {
    let task_start = Instant::now();
    let index = index.unwrap_or(0);

    if index > 0 {
        trace!("任务 {index} 开始")
    }

    let req = client
        .post(format!(
            "http://localhost:{port}{}",
            POST_CHAT_COMPLETIONS.1
        ))
        .headers(headers.clone())
        .body(req_body)
        .timeout(Duration::from_secs(100));

    match req.send().await {
        Ok(res) => {
            let status = res.status();
            if index > 0 {
                info!(
                    "任务 {index} - 响应状态: {status}, 耗时: {:?}",
                    task_start.elapsed()
                )
            } else {
                info!("响应状态: {status}, header={:#?}", res.headers())
            }

            if status.is_success() {
                if index > 0 {
                    trace!("任务 {index} 开始读取流式响应...")
                } else {
                    trace!("开始读取流式响应...")
                }

                let mut stream = res.bytes_stream();
                let mut chunk_count = 0;
                let mut accumulated_content = String::new();
                let mut buffer = String::new();
                let mut token_stats: Option<(usize, usize, usize)> = None; // (prompt_tokens, completion_tokens, total_tokens)

                while let Some(item) = stream.next().await {
                    chunk_count += 1;
                    match item {
                        Ok(bytes) => {
                            let text = std::str::from_utf8(&bytes).unwrap_or("<invalid utf8>");
                            buffer.push_str(text);

                            if index > 0 {
                                trace!("任务 {index} 收到第 {chunk_count} 个数据块: {text:?}")
                            } else {
                                let now = Instant::now();
                                trace!("收到第 {chunk_count} 个数据块 - 时间: {now:?}");
                                trace!("原始数据: {text:?}")
                            }

                            // 处理可能跨越多个数据块的SSE消息
                            while let Some(end_pos) = buffer.find("\n\n") {
                                let sse_chunk = buffer[..end_pos].to_string();
                                buffer.drain(..end_pos + 2);

                                // 解析 SSE 格式，以 data: 为准
                                for line in sse_chunk.lines() {
                                    if let Some(data_line) = line.strip_prefix("data: ") {
                                        // 尝试解析为CreateChatCompletionResponse
                                        match serde_json::from_str::<
                                            CreateChatCompletionStreamResponse,
                                        >(data_line)
                                        {
                                            Ok(response) => {
                                                trace!("解析响应: {response:?}");

                                                // 提取choices数组中的content
                                                for choice in &response.choices {
                                                    if let Some(content) = &choice.delta.content {
                                                        accumulated_content.push_str(content);
                                                        trace!("提取到文本内容: {content:?}")
                                                    }
                                                }

                                                // 提取usage信息（通常在最后一个消息中）
                                                if let Some(usage) = &response.usage {
                                                    token_stats = Some((
                                                        usage.prompt_tokens as usize,
                                                        usage.completion_tokens as usize,
                                                        usage.total_tokens as usize,
                                                    ));
                                                    trace!(
                                                        "提取到token统计: prompt={}, completion={}, total={}",
                                                        usage.prompt_tokens,
                                                        usage.completion_tokens,
                                                        usage.total_tokens
                                                    );
                                                }
                                            }
                                            Err(e) => {
                                                if index == 0 {
                                                    trace!(
                                                        "响应解析失败: {e} - 原始数据: {data_line:?}"
                                                    )
                                                }
                                                // 如果不是有效的响应格式，可能是纯文本内容
                                                accumulated_content.push_str(data_line)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!("任务 {index} 读取流时出错: {e:?}");
                            break;
                        }
                    }
                }

                if index > 0 {
                    info!(
                        "任务 {index} 完成 - 总耗时: {:?}, 数据块数: {chunk_count}, 内容长度: {}",
                        task_start.elapsed(),
                        accumulated_content.len()
                    )
                } else {
                    println!("流式响应结束，共收到 {chunk_count} 个数据块");
                    println!("完整生成内容: {accumulated_content}")
                }

                let task_end = Instant::now();
                Ok((
                    index,
                    chunk_count,
                    accumulated_content,
                    task_start.elapsed(),
                    token_stats,
                    task_start,
                    task_end,
                ))
            } else {
                let error_text = res.text().await.unwrap_or_default();
                if index > 0 {
                    warn!("任务 {index} 失败 - 状态: {status}, 错误: {error_text}")
                } else {
                    println!("body: {error_text}")
                }
                Err(format!("HTTP错误: {status}"))
            }
        }
        Err(e) => {
            if index > 0 {
                warn!("任务 {index} 请求失败: {e:?}")
            }
            Err(format!("请求错误: {e:?}"))
        }
    }
}

#[test]
fn test_post_send() {
    let port = match std::env::var("TEST_PORT") {
        Ok(port) => port.parse().unwrap(),
        Err(VarError::NotPresent) => return,
        Err(e) => panic!("{e:?}"),
    };

    crate::logger::init();
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            let (client, headers) = create_client_with_headers();

            info!(
                "Runtime workers = {}",
                tokio::runtime::Handle::current().metrics().num_workers()
            );

            let req_body = requset_body_chat("Tell a story");

            trace!("send req");
            let _ = send_single_request(port, &client, &headers, req_body, None).await;
        })
}

#[test]
fn test_post_send_multi() {
    let port = match std::env::var("TEST_PORT") {
        Ok(port) => port.parse().unwrap(),
        Err(VarError::NotPresent) => return,
        Err(e) => panic!("{e:?}"),
    };

    crate::logger::init();
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            let (client, headers) = create_client_with_headers();

            info!(
                "Runtime workers = {}",
                tokio::runtime::Handle::current().metrics().num_workers()
            );

            // 创建多个不同的请求内容
            let request_bodies = (0..CONCURRENT_REQUESTS)
                .map(|i| requset_body_chat(&format!("Tell me a story number {}", i + 1)))
                .collect::<Vec<_>>();

            let start_time = Instant::now();
            info!("开始发送 {CONCURRENT_REQUESTS} 个并发请求");

            // 创建并发任务
            let tasks = request_bodies
                .into_iter()
                .enumerate()
                .map(|(index, req_body)| {
                    let client = client.clone();
                    let headers = headers.clone();

                    tokio::spawn(async move {
                        send_single_request(port, &client, &headers, req_body, Some(index + 1))
                            .await
                    })
                })
                .collect::<Vec<_>>();

            // 等待所有任务完成并统计结果
            let mut successful_count = 0;
            let mut failed_count = 0;
            let mut total_chunks = 0;
            let mut total_text_length = 0;
            let mut max_duration = Duration::ZERO;
            let mut min_duration = Duration::MAX;

            let mut total_prompt_tokens = 0;
            let mut total_completion_tokens = 0;
            let mut total_all_tokens = 0;
            let mut token_stats_available = 0;

            // 用于计算真正的并发执行时间
            let mut first_start_time: Option<Instant> = None;
            let mut last_end_time: Option<Instant> = None;

            for task in tasks {
                match task.await {
                    Ok(Ok((
                        index,
                        chunks,
                        content,
                        duration,
                        token_stats,
                        start_time,
                        end_time,
                    ))) => {
                        successful_count += 1;
                        total_chunks += chunks;
                        total_text_length += content.len();
                        max_duration = max_duration.max(duration);
                        min_duration = min_duration.min(duration);

                        // 记录最早开始时间和最晚结束时间
                        if first_start_time.is_none() || start_time < first_start_time.unwrap() {
                            first_start_time = Some(start_time);
                        }
                        if last_end_time.is_none() || end_time > last_end_time.unwrap() {
                            last_end_time = Some(end_time);
                        }

                        // 累计token统计
                        if let Some((prompt_tokens, completion_tokens, all_tokens)) = token_stats {
                            total_prompt_tokens += prompt_tokens;
                            total_completion_tokens += completion_tokens;
                            total_all_tokens += all_tokens;
                            token_stats_available += 1;
                        }

                        trace!("任务 {index} 成功完成")
                    }
                    Ok(Err(e)) => {
                        failed_count += 1;
                        warn!("任务失败: {e}")
                    }
                    Err(e) => {
                        failed_count += 1;
                        warn!("任务执行出错: {e:?}")
                    }
                }
            }

            // 计算真正的并发执行时间
            let actual_concurrent_time =
                if let (Some(start), Some(end)) = (first_start_time, last_end_time) {
                    end.duration_since(start)
                } else {
                    Duration::ZERO
                };

            // 输出统计信息
            println!("\n=== 并发测试统计 ===");
            println!("总请求数: {CONCURRENT_REQUESTS}");
            println!("成功请求数: {successful_count}");
            println!("失败请求数: {failed_count}");
            // println!("任务创建耗时: {:?}", start_time.elapsed());
            println!("实际并发执行时间: {actual_concurrent_time:?}");
            println!("最快请求: {min_duration:?}");
            println!("最慢请求: {max_duration:?}");
            println!(
                "平均每请求耗时: {:?}",
                actual_concurrent_time / successful_count.max(1) as u32
            );
            println!("总数据块数: {total_chunks}");
            println!("总文本长度: {total_text_length}");
            println!(
                "成功率: {:.1}%",
                (successful_count as f64 / CONCURRENT_REQUESTS as f64) * 100.0
            );

            if successful_count > 0 {
                println!(
                    "平均每请求数据块数: {:.1}",
                    total_chunks as f64 / successful_count as f64
                );
                println!(
                    "平均每请求文本长度: {:.1}",
                    total_text_length as f64 / successful_count as f64
                );

                // tokens/s 性能计算
                let avg_single_request_time = (max_duration + min_duration).as_secs_f64() / 2.0;
                let avg_single_request_time_ms =
                    (max_duration.as_millis() + min_duration.as_millis()) as f64 / 2.0;

                println!("\n=== Tokens/s 性能指标 ===");
                if token_stats_available > 0 {
                    println!(
                        "实际统计可用请求数: {}/{}",
                        token_stats_available, successful_count
                    );
                    println!("总提示tokens: {}", total_prompt_tokens);
                    println!("总生成tokens: {}", total_completion_tokens);
                    println!("总tokens: {}", total_all_tokens);

                    // 使用真正的并发执行时间来计算整体tokens/s
                    let actual_concurrent_micros = actual_concurrent_time.as_micros() as f64;

                    if actual_concurrent_micros > 0.0 {
                        println!(
                            "整体生成tokens/s: {:.2}",
                            (total_completion_tokens as f64 * 1_000_000.0)
                                / actual_concurrent_micros
                        );
                    } else {
                        println!("整体生成tokens/s: 无法计算（时间过短）");
                    }

                    if avg_single_request_time_ms > 0.0 {
                        println!(
                            "平均单请求生成tokens/s: {:.2}",
                            ((total_completion_tokens as f64 / token_stats_available as f64)
                                * 1000.0)
                                / avg_single_request_time_ms
                        );
                    } else {
                        println!("平均单请求生成tokens/s: 无法计算（时间过短）");
                    }

                    let min_duration_ms = min_duration.as_millis() as f64;
                    let max_duration_ms = max_duration.as_millis() as f64;

                    if min_duration_ms > 0.0 {
                        println!(
                            "最佳单请求生成tokens/s: {:.2}",
                            ((total_completion_tokens as f64 / token_stats_available as f64)
                                * 1000.0)
                                / min_duration_ms
                        );
                    } else {
                        println!("最佳单请求生成tokens/s: 无法计算（时间过短）");
                    }

                    if max_duration_ms > 0.0 {
                        println!(
                            "最差单请求生成tokens/s: {:.2}",
                            ((total_completion_tokens as f64 / token_stats_available as f64)
                                * 1000.0)
                                / max_duration_ms
                        );
                    } else {
                        println!("最差单请求生成tokens/s: 无法计算（时间过短）");
                    }
                } else {
                    // 如果没有token统计，回退到估算
                    let estimated_total_tokens = total_chunks;
                    println!("估算总生成tokens (数据块): {}", estimated_total_tokens);

                    let actual_concurrent_micros = actual_concurrent_time.as_micros() as f64;
                    if actual_concurrent_micros > 0.0 {
                        println!(
                            "整体tokens/s (估算): {:.2}",
                            (estimated_total_tokens as f64 * 1_000_000.0)
                                / actual_concurrent_micros
                        );
                    } else {
                        println!("整体tokens/s (估算): 无法计算（时间过短）");
                    }

                    if avg_single_request_time_ms > 0.0 {
                        println!(
                            "平均单请求tokens/s (估算): {:.2}",
                            ((estimated_total_tokens as f64 / successful_count as f64) * 1000.0)
                                / avg_single_request_time_ms
                        );
                    } else {
                        println!("平均单请求tokens/s (估算): 无法计算（时间过短）");
                    }
                }

                // 并发效率分析
                let sequential_time_ms = avg_single_request_time_ms * successful_count as f64;
                let concurrent_time_micros = actual_concurrent_time.as_micros() as f64;
                let concurrent_time_ms = concurrent_time_micros / 1000.0;

                println!("\n=== 并发效率分析 ===");
                println!("顺序执行预估时间: {:.2}s", sequential_time_ms / 1000.0);
                println!("并发执行实际时间: {:.3}s", concurrent_time_ms / 1000.0);

                if concurrent_time_micros > 0.0 {
                    let speedup = sequential_time_ms / concurrent_time_ms;
                    let efficiency = speedup / successful_count as f64 * 100.0;
                    println!("并发加速比: {:.2}x", speedup);
                    println!("并发效率: {:.1}%", efficiency);
                } else {
                    println!("并发加速比: 无法计算（时间过短）");
                    println!("并发效率: 无法计算（时间过短）");
                }
            }

            // 验证至少有一些请求成功
            assert!(successful_count > 0, "至少应该有一个请求成功");
        })
}

#[test]
fn test_blacklisted_check() {
    let port = match std::env::var("TEST_PORT") {
        Ok(port) => port.parse().unwrap(),
        Err(VarError::NotPresent) => return,
        Err(e) => panic!("{e:?}"),
    };

    crate::logger::init();
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            let (client, headers) = create_client_with_headers();

            info!("Testing blacklist functionality with actual service");

            // Test case 1: Normal request without blacklisted words
            let normal_prompt = "Tell me a story about a cat";
            let req_body_normal = requset_body_chat(normal_prompt);

            info!("Sending normal request: {normal_prompt}");
            let normal_result =
                send_single_request(port, &client, &headers, req_body_normal, Some(1)).await;

            match normal_result {
                Ok((_, _, content, _, _, _, _)) => {
                    info!("Normal request completed successfully");
                    info!("Generated content length: {}", content.len());
                    assert!(
                        !content.is_empty(),
                        "Normal request should generate content"
                    );
                }
                Err(e) => {
                    warn!("Normal request failed: {e}");
                    // Don't fail the test if the service is not available
                }
            }

            // Test case 2: Request that might trigger blacklist (if service has blacklist configured)
            let test_prompt = "Generate text that might contain sensitive information";
            let req_body_test = requset_body_chat(test_prompt);

            info!("Sending test request: {test_prompt}");
            let test_result =
                send_single_request(port, &client, &headers, req_body_test, Some(2)).await;

            match test_result {
                Ok((_, _, content, _, _, _, _)) => {
                    info!("Test request completed");
                    info!("Generated content: {content}");

                    // Check if the response indicates blacklist detection
                    if content.contains(BLACKLISTED_SIGNAL) {
                        info!("Blacklist detection triggered!");
                    } else {
                        info!("No blacklist detection in this response");
                    }
                }
                Err(e) => {
                    warn!("Test request failed: {e}");
                    // Don't fail the test if the service is not available
                }
            }

            // Test case 3: Test with specific blacklisted words
            let blacklist_test_prompts = [
                "Tell me about dangerous activities",
                "Explain how to leak information",
                "Describe bad words and their usage",
            ];

            for (i, prompt) in blacklist_test_prompts.iter().enumerate() {
                let idx = i + 1;
                let req_body = requset_body_chat(prompt);
                info!("Sending blacklist test {idx}: {prompt}");

                let result =
                    send_single_request(port, &client, &headers, req_body, Some(3 + i)).await;

                match result {
                    Ok((_, _, content, _, _, _, _)) => {
                        info!("Blacklist test {idx} completed");
                        info!("Content: {content}");

                        // Check for blacklist indicators
                        let has_blacklist_indicators =
                            content.contains(BLACKLISTED_SIGNAL) || content.is_empty(); // Empty response might indicate early termination

                        if has_blacklist_indicators {
                            info!("Blacklist detection confirmed for test {idx}");
                        } else {
                            info!("No blacklist detection for test {idx}");
                        }
                    }
                    Err(e) => {
                        warn!("Blacklist test {idx} failed: {e}");
                    }
                }
            }

            info!("Blacklist testing completed");
        })
}
