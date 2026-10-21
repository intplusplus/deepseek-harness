//! Vocabulary behavior: wire fidelity of the message and block unions,
//! extension slots, and field-wise config equality.

use dsh_llm::*;
use serde_json::{json, Map};

#[test]
fn role_messages_round_trip_with_the_role_tag() {
    let message = Message::User(UserMessage::new(
        MessageId::new("m1"),
        vec![ContentBlock::text("hi")],
    ));
    let text = serde_json::to_string(&message).unwrap();
    assert!(text.contains("\"role\":\"user\""), "{text}");
    assert!(text.contains("\"id\":\"m1\""), "{text}");
    let back: Message = serde_json::from_str(&text).unwrap();
    assert_eq!(back, message);
}

#[test]
fn assistant_sources_carry_provider_and_model_identity() {
    let message = Message::Assistant(AssistantMessage::new(
        MessageId::new("m2"),
        vec![],
        "deepseek",
        "chat",
    ));
    let text = serde_json::to_string(&message).unwrap();
    assert!(text.contains("\"kind\":\"model\""), "{text}");
    assert!(text.contains("\"provider\":\"deepseek\""), "{text}");
    let back: Message = serde_json::from_str(&text).unwrap();
    assert_eq!(back.source().kind(), "model");
}

#[test]
fn tool_results_pair_their_source_with_the_answered_call_id() {
    let message = ToolResultMessage::new(
        MessageId::new("m3"),
        vec![],
        ToolCallId::new("call-1"),
        true,
    );
    let text = serde_json::to_string(&Message::Tool(message.clone())).unwrap();
    assert!(text.contains("\"isError\":true"), "{text}");
    assert!(text.contains("\"toolCallId\":\"call-1\""), "{text}");
    let back: Message = serde_json::from_str(&text).unwrap();
    assert_eq!(back, Message::Tool(message));
}

#[test]
fn system_messages_default_to_the_system_prompt_producer() {
    let message = SystemMessage::new(MessageId::new("m4"), vec![ContentBlock::text("be terse")]);
    assert_eq!(message.source.kind(), "system-prompt");
}

#[test]
fn extension_content_blocks_survive_verbatim() {
    let mut payload = Map::new();
    payload.insert("custom".to_string(), json!({"nested": [1, 2]}));
    let block = ContentBlock::extension("demo/chart", payload);
    let text = serde_json::to_string(&block).unwrap();
    assert!(text.contains("\"type\":\"demo/chart\""), "{text}");
    let back: ContentBlock = serde_json::from_str(&text).unwrap();
    assert_eq!(back, block);
    assert_eq!(back.block_type(), "demo/chart");
}

#[test]
fn extension_message_sources_fall_through_unknown_kinds() {
    let mut fields = Map::new();
    fields.insert("origin".to_string(), json!("cron"));
    let source = MessageSource::Extension(ExtensionSource::new("cron", fields));
    let text = serde_json::to_string(&source).unwrap();
    let back: MessageSource = serde_json::from_str(&text).unwrap();
    assert_eq!(back.kind(), "cron");
    assert_eq!(back, source);
}

#[test]
fn malformed_core_blocks_error_instead_of_demoting_to_extension() {
    let text = r#"{"type":"text"}"#; // missing `text`
    let result: Result<ContentBlock, _> = serde_json::from_str(text);
    assert!(result.is_err());
}

#[test]
fn every_core_block_type_routes_to_the_core_union() {
    // Guards the tag list against drift from the enum.
    for block in [
        ContentBlock::text("t"),
        ContentBlock::reasoning("r"),
        ContentBlock::tool_call(ToolCallId::new("c"), "echo", "{}"),
        ContentBlock::tool_addition("echo"),
        ContentBlock::tool_removal("echo"),
    ] {
        let text = serde_json::to_string(&block).unwrap();
        let back: ContentBlock = serde_json::from_str(&text).unwrap();
        assert!(matches!(back, ContentBlock::Core(_)), "{text}");
        assert_eq!(back, block);
    }
}

#[test]
fn call_config_equality_is_field_wise() {
    let base = LlmCallConfig {
        provider: "p".to_string(),
        model: "m".to_string(),
        reasoning_effort: None,
        temperature: Some(0.5),
        max_tokens: None,
        stop: Some(vec!["a".to_string()]),
    };
    assert!(call_config_equals(&base, &base.clone()));
    let mut changed_stop = base.clone();
    changed_stop.stop = Some(vec!["a".to_string(), "b".to_string()]);
    assert!(!call_config_equals(&base, &changed_stop));
    let mut changed_temperature = base.clone();
    changed_temperature.temperature = Some(0.6);
    assert!(!call_config_equals(&base, &changed_temperature));
}

#[test]
fn assistant_stream_records_round_trip() {
    let record = AssistantStreamRecord::ToolCallChunks {
        time0: 3,
        index: 1,
        dt: vec![0, 2],
        id: ToolCallId::new("c1"),
        name: Some("echo".to_string()),
        args: vec!["{\"te".to_string(), "xt\":1}".to_string()],
    };
    let text = serde_json::to_string(&record).unwrap();
    let back: AssistantStreamRecord = serde_json::from_str(&text).unwrap();
    assert_eq!(back, record);
}

#[test]
fn finish_reasons_carry_structured_failures() {
    let reason = FinishReason::error(LlmFailure {
        message: "boom".to_string(),
        code: "UNKNOWN".to_string(),
        status: Some(500),
        provider_retry_after_ms: None,
        request_id: None,
        offload_images: None,
    });
    let text = serde_json::to_string(&reason).unwrap();
    assert!(text.contains("\"kind\":\"error\""), "{text}");
    let back: FinishReason = serde_json::from_str(&text).unwrap();
    assert_eq!(back, reason);
}

#[test]
fn token_usage_omits_absent_cache_fields() {
    let usage = TokenUsage {
        input_tokens: 10,
        output_tokens: 2,
        ..Default::default()
    };
    let text = serde_json::to_string(&usage).unwrap();
    assert!(!text.contains("cacheReadTokens"), "{text}");
    assert!(text.contains("\"inputTokens\":10"), "{text}");
}
