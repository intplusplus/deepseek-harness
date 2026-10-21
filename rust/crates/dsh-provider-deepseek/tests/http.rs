//! End-to-end tests against a local fake HTTP/1.1 + SSE endpoint.
//!
//! Nothing here touches the network: every response is scripted by
//! [`support::serve`], so the tests assert the provider's real transport
//! behaviour — headers, body framing, chunked termination, error classification
//! — without a key or an internet connection.

mod support;

use std::time::Duration;

use dsh_llm::{
    ContentBlock, FinishReason, LlmCallConfig, Message, MessageId, TokenUsage, ToolCallId,
    ToolSchema, UserMessage,
};
use dsh_provider::{LlmProvider, LlmRequest};
use dsh_provider_deepseek::{DeepSeekConfig, DeepSeekProvider};
use serde_json::json;
use support::{MockServer, Response};

/// A one-message request against `deepseek-chat`.
fn request() -> LlmRequest {
    LlmRequest {
        config: LlmCallConfig {
            provider: "deepseek".to_string(),
            model: "deepseek-chat".to_string(),
            reasoning_effort: None,
            temperature: None,
            max_tokens: None,
            stop: None,
        },
        messages: vec![Message::User(UserMessage::new(
            MessageId::new("user-1"),
            vec![ContentBlock::text("hello")],
        ))],
        tools: vec![ToolSchema::new(
            "lookup",
            "Find a value",
            json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        )],
    }
}

/// A provider pointed at `server`, with deadlines short enough to fail fast.
fn provider(server: &MockServer, stream: bool) -> DeepSeekProvider {
    DeepSeekProvider::new(
        DeepSeekConfig::new("test-key")
            .with_base_url(&server.base_url)
            .with_stream(stream)
            .with_request_timeout(Duration::from_secs(5))
            .with_stream_idle_timeout(Duration::from_secs(5)),
    )
    .expect("provider builds")
}

/// One SSE frame carrying `data: {json}`.
fn sse(body: serde_json::Value) -> String {
    format!("data: {body}\n\n")
}

/// The terminal SSE frame.
fn done() -> String {
    "data: [DONE]\n\n".to_string()
}

#[tokio::test]
async fn non_streaming_completion_round_trips() {
    let server = support::serve(Response::json(
        200,
        json!({
            "id": "chatcmpl-1",
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "the answer"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        })
        .to_string(),
    ))
    .await;

    let response = provider(&server, false)
        .generate(&request())
        .await
        .expect("completion");

    assert_eq!(
        response.message.content,
        vec![ContentBlock::text("the answer")]
    );
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(
        response.usage,
        Some(TokenUsage {
            input_tokens: 7,
            output_tokens: 3,
            total_tokens: Some(10),
            ..TokenUsage::default()
        })
    );

    let recorded = server.only_request();
    assert_eq!(
        recorded.request_line(),
        "POST /chat/completions HTTP/1.1",
        "the configured base URL is the endpoint root"
    );
    assert_eq!(
        recorded.header("authorization").as_deref(),
        Some("Bearer test-key")
    );
    assert_eq!(
        recorded.header("content-type").as_deref(),
        Some("application/json")
    );
    let body = recorded.json();
    assert_eq!(body["model"], json!("deepseek-chat"));
    assert_eq!(body["stream"], json!(false));
    assert_eq!(
        body["messages"],
        json!([{"role": "user", "content": "hello"}])
    );
    assert_eq!(
        body["tools"],
        json!([{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find a value",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }])
    );
    assert!(body.get("stream_options").is_none());
}

#[tokio::test]
async fn streaming_completion_assembles_blocks_tool_calls_and_usage() {
    let server = support::serve(Response::sse(vec![
        sse(json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})),
        sse(json!({"choices": [{"index": 0, "delta": {"reasoning_content": "plan"}}]})),
        sse(json!({"choices": [{"index": 0, "delta": {"content": "Hi "}}]})),
        sse(json!({"choices": [{"index": 0, "delta": {"content": "there"}}]})),
        sse(json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": "{\"q\":"},
        }]}}]})),
        sse(json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": "\"x\"}"},
        }]}}]})),
        sse(json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})),
        sse(json!({"choices": [], "usage": {
            "prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14,
        }})),
        done(),
    ]))
    .await;

    let response = provider(&server, true)
        .generate(&request())
        .await
        .expect("stream");

    assert_eq!(
        response.message.content,
        vec![
            ContentBlock::reasoning("plan"),
            ContentBlock::text("Hi there"),
            ContentBlock::tool_call(ToolCallId::new("call_1"), "lookup", "{\"q\":\"x\"}"),
        ]
    );
    assert_eq!(response.finish_reason, FinishReason::ToolCalls);
    assert_eq!(
        response.usage,
        Some(TokenUsage {
            input_tokens: 9,
            output_tokens: 5,
            total_tokens: Some(14),
            ..TokenUsage::default()
        })
    );
    assert!(!response.stream.is_empty(), "the timed stream is preserved");

    let recorded = server.only_request();
    assert_eq!(
        recorded.header("accept").as_deref(),
        Some("text/event-stream")
    );
    let body = recorded.json();
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
}

#[tokio::test]
async fn rate_limit_carries_the_retry_delay_and_request_id() {
    let server = support::serve(Response {
        status: 429,
        headers: vec![
            ("content-type".to_string(), "application/json".to_string()),
            ("retry-after".to_string(), "3".to_string()),
            ("x-request-id".to_string(), "req-abc".to_string()),
        ],
        frames: vec![
            json!({"error": {"message": "Rate limit reached", "type": "rate_limit_error"}})
                .to_string(),
        ],
        ending: support::Ending::Length,
    })
    .await;

    let failure = provider(&server, true)
        .generate(&request())
        .await
        .expect_err("rate limited");

    assert_eq!(failure.code, "RATE_LIMIT");
    assert_eq!(failure.status, Some(429));
    assert_eq!(failure.provider_retry_after_ms, Some(3000));
    assert_eq!(
        failure.request_id.as_ref().map(|id| id.as_str()),
        Some("req-abc")
    );
    assert_eq!(failure.message, "Rate limit reached");
}

#[tokio::test]
async fn bad_request_reports_the_provider_message() {
    let server = support::serve(Response::json(
        400,
        json!({
            "error": {
                "message": "messages must alternate between user and assistant",
                "type": "invalid_request_error",
            },
        })
        .to_string(),
    ))
    .await;

    let failure = provider(&server, false)
        .generate(&request())
        .await
        .expect_err("invalid request");

    assert_eq!(failure.code, "INVALID_REQUEST");
    assert_eq!(failure.status, Some(400));
    assert_eq!(
        failure.message,
        "messages must alternate between user and assistant"
    );
    assert_eq!(failure.provider_retry_after_ms, None);
}

#[tokio::test]
async fn server_error_is_retryable_classified() {
    let server = support::serve(Response::json(
        503,
        json!({"error": {"message": "overloaded", "type": "overloaded_error"}}).to_string(),
    ))
    .await;

    let failure = provider(&server, true)
        .generate(&request())
        .await
        .expect_err("server error");
    assert_eq!(failure.code, "SERVER");
    assert_eq!(failure.status, Some(503));
}

#[tokio::test]
async fn interrupted_stream_reports_transport() {
    let server = support::serve(Response::truncated_sse(vec![sse(
        json!({"choices": [{"index": 0, "delta": {"content": "half"}}]}),
    )]))
    .await;

    let failure = provider(&server, true)
        .generate(&request())
        .await
        .expect_err("broken body");
    assert_eq!(failure.code, "TRANSPORT", "{}", failure.message);
    assert_eq!(failure.status, None);
}

#[tokio::test]
async fn clean_eof_without_done_reports_a_closed_stream() {
    let server = support::serve(Response::sse_until_close(vec![sse(
        json!({"choices": [{"index": 0, "delta": {"content": "half"}}]}),
    )]))
    .await;

    let failure = provider(&server, true)
        .generate(&request())
        .await
        .expect_err("closed stream");
    assert_eq!(failure.code, "STREAM_CLOSED");
}

#[tokio::test]
async fn malformed_json_frame_reports_malformed_response() {
    let server = support::serve(Response::sse(vec![
        "data: {\"choices\": [\n\n".to_string(),
        done(),
    ]))
    .await;

    let failure = provider(&server, true)
        .generate(&request())
        .await
        .expect_err("malformed frame");
    assert_eq!(failure.code, "MALFORMED_RESPONSE");
}

#[tokio::test]
async fn a_missing_key_fails_before_any_request() {
    let server = support::serve(Response::json(200, "{}")).await;
    let provider = DeepSeekProvider::new(
        DeepSeekConfig::new("")
            .with_base_url(&server.base_url)
            .with_stream(false),
    )
    .expect("provider builds");

    let failure = provider
        .generate(&request())
        .await
        .expect_err("no credential");
    assert_eq!(failure.code, "MISSING_CREDENTIAL");
    assert!(server.requests().is_empty());
}

#[test]
fn debug_output_never_contains_the_api_key() {
    let config = DeepSeekConfig::new("sk-super-secret").with_base_url("http://127.0.0.1:1");
    let rendered = format!("{config:?}");
    assert!(!rendered.contains("sk-super-secret"), "{rendered}");
    assert!(rendered.contains("<redacted>"), "{rendered}");
    assert!(rendered.contains("127.0.0.1:1"), "{rendered}");
}
