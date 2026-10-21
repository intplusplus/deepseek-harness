//! Event wire format, field order, and strict decode behaviour.
//!
//! Ported from `packages/compaction/compaction/tests/compaction.spec.ts` and the
//! `SessionEventMap` declarations in `packages/compaction/compaction/src/types.ts`.

use dsh_compaction::{
    CompactionEnd, CompactionEvent, CompactionId, CompactionPrune, CompactionResult,
    CompactionStart, CompactionSummary, CompactionTurn, ShadowedRange, COMPACTION_END_TAG,
    COMPACTION_PRUNE_TAG, COMPACTION_START_TAG, COMPACTION_SUMMARY_TAG,
};
use dsh_llm::{ContentBlock, TokenUsage};
use dsh_session::{SessionEvent, SessionSeq};
use serde_json::{json, Map, Value};

fn seq(value: u64) -> SessionSeq {
    SessionSeq::new(value).unwrap()
}

fn text(value: &str) -> ContentBlock {
    ContentBlock::text(value)
}

fn summary() -> CompactionSummary {
    CompactionSummary {
        compaction_id: CompactionId::new("c1"),
        source_command_id: None,
        summary: vec![text("sum")],
        raw_output: Some(vec![text("raw")]),
        llm_stream_call: true,
        shadowed_range: ShadowedRange::new(seq(2), seq(3)),
        shadowed_seqs: vec![seq(2), seq(3)],
        shadowed_token_count: 9,
        provider: "p".to_string(),
        model: "m".to_string(),
        max_tokens: Some(128),
        usage: Some(TokenUsage {
            input_tokens: 1,
            output_tokens: 2,
            ..TokenUsage::default()
        }),
    }
}

fn encode(event: CompactionEvent) -> String {
    serde_json::to_string(&event.encode().unwrap()).unwrap()
}

fn extension(kind: &str, payload: Value) -> SessionEvent {
    let Value::Object(payload) = payload else {
        panic!("test payload must be an object");
    };
    SessionEvent::extension_event(kind, payload)
}

#[test]
fn compaction_start_serializes_type_first_and_turn_null() {
    let event = CompactionEvent::Start(CompactionStart::new(
        CompactionId::new("c1"),
        None,
        CompactionTurn::standalone(),
    ));
    assert_eq!(
        encode(event),
        r#"{"type":"compaction/start","compactionId":"c1","turn":null}"#
    );
}

#[test]
fn compaction_start_places_source_command_before_turn() {
    let event = CompactionEvent::Start(CompactionStart::new(
        CompactionId::new("c1"),
        Some(dsh_compaction::CommandId::new("cmd")),
        CompactionTurn::new(Some(3)),
    ));
    assert_eq!(
        encode(event),
        r#"{"type":"compaction/start","compactionId":"c1","sourceCommandId":"cmd","turn":3}"#
    );
}

#[test]
fn compaction_end_serializes_error_after_turn() {
    let event = CompactionEvent::End(CompactionEnd::new(
        CompactionId::new("c1"),
        None,
        CompactionTurn::new(Some(3)),
        Some("boom".to_string()),
    ));
    assert_eq!(
        encode(event),
        r#"{"type":"compaction/end","compactionId":"c1","turn":3,"error":"boom"}"#
    );
}

#[test]
fn compaction_prune_serializes_the_surface_span() {
    let event = CompactionEvent::Prune(CompactionPrune::new(
        ShadowedRange::new(seq(2), seq(2)),
        vec![seq(2)],
        7,
    ));
    assert_eq!(
        encode(event),
        r#"{"type":"compaction/prune","shadowedRange":{"start":2,"end":2},"shadowedSeqs":[2],"shadowedTokenCount":7}"#
    );
}

#[test]
fn compaction_summary_follows_the_typescript_producer_field_order() {
    assert_eq!(
        encode(CompactionEvent::Summary(Box::new(summary()))),
        r#"{"type":"compaction/summary","compactionId":"c1","summary":[{"type":"text","text":"sum"}],"rawOutput":[{"type":"text","text":"raw"}],"llmStreamCall":true,"shadowedRange":{"start":2,"end":3},"shadowedSeqs":[2,3],"shadowedTokenCount":9,"provider":"p","model":"m","maxTokens":128,"usage":{"inputTokens":1,"outputTokens":2}}"#
    );
}

#[test]
fn compaction_summary_omits_unmarked_optional_fields() {
    let event = CompactionSummary {
        compaction_id: CompactionId::new("c1"),
        source_command_id: None,
        summary: vec![text("sum")],
        raw_output: None,
        llm_stream_call: false,
        shadowed_range: ShadowedRange::new(seq(2), seq(2)),
        shadowed_seqs: vec![seq(2)],
        shadowed_token_count: 4,
        provider: "p".to_string(),
        model: "m".to_string(),
        max_tokens: None,
        usage: None,
    };
    assert_eq!(
        encode(CompactionEvent::Summary(Box::new(event))),
        r#"{"type":"compaction/summary","compactionId":"c1","summary":[{"type":"text","text":"sum"}],"shadowedRange":{"start":2,"end":2},"shadowedSeqs":[2],"shadowedTokenCount":4,"provider":"p","model":"m"}"#
    );
}

#[test]
fn compaction_summary_keeps_raw_output_without_llm_stream_call() {
    let event = CompactionSummary {
        compaction_id: CompactionId::new("c1"),
        source_command_id: Some(dsh_compaction::CommandId::new("cmd")),
        summary: vec![text("sum")],
        raw_output: Some(vec![text("raw")]),
        llm_stream_call: false,
        shadowed_range: ShadowedRange::new(seq(2), seq(2)),
        shadowed_seqs: vec![seq(2)],
        shadowed_token_count: 4,
        provider: "p".to_string(),
        model: "m".to_string(),
        max_tokens: None,
        usage: None,
    };
    assert_eq!(
        encode(CompactionEvent::Summary(Box::new(event))),
        r#"{"type":"compaction/summary","compactionId":"c1","sourceCommandId":"cmd","summary":[{"type":"text","text":"sum"}],"rawOutput":[{"type":"text","text":"raw"}],"shadowedRange":{"start":2,"end":2},"shadowedSeqs":[2],"shadowedTokenCount":4,"provider":"p","model":"m"}"#
    );
}

#[test]
fn compaction_summary_round_trips() {
    let event = CompactionEvent::Summary(Box::new(summary()));
    let encoded = event.encode().unwrap();
    let decoded = CompactionEvent::decode(&encoded).unwrap().unwrap();
    assert_eq!(decoded, event);
}

#[test]
fn compaction_summary_rejects_llm_stream_call_without_raw_output_on_encode() {
    let mut event = summary();
    event.raw_output = None;
    let error = CompactionEvent::Summary(Box::new(event))
        .encode()
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("llmStreamCall requires rawOutput"));
}

#[test]
fn decode_rejects_llm_stream_call_true_without_raw_output() {
    let event = extension(
        COMPACTION_SUMMARY_TAG,
        json!({
            "type": COMPACTION_SUMMARY_TAG,
            "compactionId": "c1",
            "summary": [],
            "llmStreamCall": true,
            "shadowedRange": { "start": 1, "end": 1 },
            "shadowedSeqs": [1],
            "shadowedTokenCount": 0,
            "provider": "p",
            "model": "m"
        }),
    );
    let error = CompactionEvent::decode(&event).unwrap().unwrap_err();
    assert!(error
        .to_string()
        .contains("llmStreamCall requires rawOutput"));
}

#[test]
fn decode_rejects_llm_stream_call_false() {
    let event = extension(
        COMPACTION_SUMMARY_TAG,
        json!({
            "type": COMPACTION_SUMMARY_TAG,
            "compactionId": "c1",
            "summary": [],
            "llmStreamCall": false,
            "shadowedRange": { "start": 1, "end": 1 },
            "shadowedSeqs": [1],
            "shadowedTokenCount": 0,
            "provider": "p",
            "model": "m"
        }),
    );
    let error = CompactionEvent::decode(&event).unwrap().unwrap_err();
    assert!(error.to_string().contains("llmStreamCall must be true"));
}

#[test]
fn decode_returns_none_for_a_foreign_tag() {
    let event = extension("llm/retry", json!({ "type": "llm/retry" }));
    assert!(CompactionEvent::decode(&event).is_none());
}

#[test]
fn decode_returns_none_for_a_core_event() {
    let event = SessionEvent::user_message(dsh_llm::UserMessage::new(
        dsh_llm::MessageId::new("m1"),
        vec![text("hi")],
    ));
    assert!(CompactionEvent::decode(&event).is_none());
}

#[test]
fn decode_known_tag_with_bad_payload_is_an_error() {
    let event = extension(
        COMPACTION_START_TAG,
        json!({ "type": COMPACTION_START_TAG, "compactionId": "c1" }),
    );
    assert!(CompactionEvent::decode(&event).unwrap().is_err());
}

#[test]
fn compaction_start_requires_an_explicit_turn_key() {
    let event = extension(
        COMPACTION_START_TAG,
        json!({ "type": COMPACTION_START_TAG, "compactionId": "c1" }),
    );
    let error = CompactionEvent::decode(&event).unwrap().unwrap_err();
    assert!(error.to_string().contains("missing field `turn`"));
}

#[test]
fn compaction_turn_preserves_null_and_number() {
    assert!(CompactionTurn::standalone().is_standalone());
    assert_eq!(CompactionTurn::new(Some(4)).get(), Some(4));
    assert_eq!(CompactionTurn::new(None).get(), None);
}

#[test]
fn compaction_event_kind_matches_its_tag() {
    assert_eq!(
        CompactionEvent::Start(CompactionStart::new(
            CompactionId::new("c1"),
            None,
            CompactionTurn::standalone()
        ))
        .kind(),
        COMPACTION_START_TAG
    );
    assert_eq!(
        CompactionEvent::Summary(Box::new(summary())).kind(),
        COMPACTION_SUMMARY_TAG
    );
    assert_eq!(
        CompactionEvent::End(CompactionEnd::new(
            CompactionId::new("c1"),
            None,
            CompactionTurn::standalone(),
            None
        ))
        .kind(),
        COMPACTION_END_TAG
    );
    assert_eq!(
        CompactionEvent::Prune(CompactionPrune::new(
            ShadowedRange::new(seq(1), seq(1)),
            vec![seq(1)],
            0
        ))
        .kind(),
        COMPACTION_PRUNE_TAG
    );
}

#[test]
fn compaction_result_serializes_the_declared_order() {
    let result = CompactionResult {
        compaction_id: CompactionId::new("c1"),
        source_command_id: None,
        start_seq: seq(1),
        summary_seq: seq(2),
        end_seq: seq(3),
        summary: vec![text("sum")],
        shadowed_range: ShadowedRange::new(seq(4), seq(5)),
        shadowed_seqs: vec![seq(4), seq(5)],
        shadowed_token_count: 6,
    };
    assert_eq!(
        serde_json::to_string(&result).unwrap(),
        r#"{"compactionId":"c1","startSeq":1,"summarySeq":2,"endSeq":3,"summary":[{"type":"text","text":"sum"}],"shadowedRange":{"start":4,"end":5},"shadowedSeqs":[4,5],"shadowedTokenCount":6}"#
    );
}

#[test]
fn an_extension_event_round_trips_through_the_log_encoder() {
    let event = CompactionEvent::Summary(Box::new(summary()))
        .encode()
        .unwrap();
    let decoded: SessionEvent =
        serde_json::from_str(&serde_json::to_string(&event).unwrap()).unwrap();
    assert_eq!(decoded, event);
}

#[test]
fn a_manual_summary_payload_can_be_built_without_serde_helpers() {
    let mut fields = Map::new();
    fields.insert("compactionId".to_string(), Value::String("c1".to_string()));
    fields.insert("summary".to_string(), Value::Array(Vec::new()));
    fields.insert("shadowedRange".to_string(), json!({ "start": 1, "end": 1 }));
    fields.insert("shadowedSeqs".to_string(), json!([1]));
    fields.insert("shadowedTokenCount".to_string(), json!(0));
    fields.insert("provider".to_string(), Value::String("p".to_string()));
    fields.insert("model".to_string(), Value::String("m".to_string()));
    let event = SessionEvent::extension_event(COMPACTION_SUMMARY_TAG, fields);
    assert!(CompactionEvent::decode(&event).unwrap().is_ok());
}
