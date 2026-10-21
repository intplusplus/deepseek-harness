//! Shared event builders for the compaction integration tests.

#![forbid(unsafe_code)]
#![allow(dead_code)]

use dsh_compaction::{
    CompactionEnd, CompactionEvent, CompactionId, CompactionPrune, CompactionStart,
    CompactionSummary, CompactionTurn, ShadowedRange,
};
use dsh_llm::{
    AssistantMessage, ContentBlock, DeveloperMessage, MessageId, MessageSource, SystemMessage,
    ToolCallId, ToolResultMessage, UserMessage,
};
use dsh_session::{CoreEvent, SessionEvent, SessionSeq};
use serde_json::{json, Map, Value};

/// A safe-integer test sequence.
pub fn seq(value: u64) -> SessionSeq {
    SessionSeq::new(value).expect("test sequence is within the safe-integer bound")
}

/// A plain text block.
pub fn text(value: impl Into<String>) -> ContentBlock {
    ContentBlock::text(value)
}

/// A system message at turn 0, step 0.
pub fn system(id: &str) -> SessionEvent {
    SessionEvent::system_message(
        0,
        0,
        SystemMessage::new(MessageId::new(id), vec![text("system")]),
    )
}

/// A direct user message.
pub fn user(id: &str, content: Vec<ContentBlock>) -> SessionEvent {
    SessionEvent::user_message(UserMessage::new(MessageId::new(id), content))
}

/// A user message carrying an explicit source.
pub fn user_with_source(
    id: &str,
    content: Vec<ContentBlock>,
    source: MessageSource,
) -> SessionEvent {
    SessionEvent::user_message(UserMessage::with_source(
        MessageId::new(id),
        content,
        source,
    ))
}

/// A developer message.
pub fn developer(id: &str) -> SessionEvent {
    SessionEvent::developer_message(
        0,
        0,
        DeveloperMessage::new(MessageId::new(id), vec![text("dev")], MessageSource::User),
        None,
    )
}

/// An assistant message with no tool calls.
pub fn assistant(id: &str) -> SessionEvent {
    SessionEvent::assistant_message(
        0,
        0,
        AssistantMessage::new(MessageId::new(id), vec![text("answer")], "mock", "mock"),
        Vec::new(),
        None,
        false,
    )
}

/// An assistant message carrying one `tool-call` block per call id.
pub fn assistant_with_calls(id: &str, calls: &[&str]) -> SessionEvent {
    let content = calls
        .iter()
        .map(|call| ContentBlock::tool_call(ToolCallId::new(*call), "bash", "{}"))
        .collect();
    SessionEvent::assistant_message(
        0,
        0,
        AssistantMessage::new(MessageId::new(id), content, "mock", "mock"),
        Vec::new(),
        None,
        false,
    )
}

/// A successful tool result at turn 0, step 0.
pub fn tool_result(id: &str, call: &str, content: Vec<ContentBlock>) -> SessionEvent {
    SessionEvent::tool_result(
        0,
        0,
        ToolResultMessage::new(MessageId::new(id), content, ToolCallId::new(call), false),
    )
}

/// A failed tool result carrying durable error and metadata.
pub fn tool_result_with_details(id: &str, call: &str, content: Vec<ContentBlock>) -> SessionEvent {
    SessionEvent::tool_result_full(
        2,
        3,
        ToolResultMessage::new(MessageId::new(id), content, ToolCallId::new(call), true),
        Some(dsh_session::ToolErrorInfo {
            name: "ExitError".to_string(),
            code: "EXIT_1".to_string(),
            reason: None,
        }),
        Some(json!({ "diff": ["a", "b"] })),
    )
}

/// A `turn/start` event.
pub fn turn_start(turn: u64) -> SessionEvent {
    SessionEvent::turn_start(turn)
}

/// A completed `turn/end` event.
pub fn turn_end(turn: u64) -> SessionEvent {
    SessionEvent::turn_end(turn, dsh_session::TurnEndReason::Completed)
}

/// A `session/end-seed` boundary.
pub fn end_seed() -> SessionEvent {
    SessionEvent::Core(CoreEvent::EndSeed { inherited: false })
}

/// A plugin-declared event.
pub fn extension(kind: &str, payload: Value) -> SessionEvent {
    let Value::Object(payload) = payload else {
        panic!("test payload must be an object");
    };
    SessionEvent::extension_event(kind, payload)
}

/// Encode one compaction event into its durable log row.
pub fn encoded(event: CompactionEvent) -> SessionEvent {
    event.encode().expect("test event encodes")
}

/// A `compaction/start` row.
pub fn start(compaction_id: &str, turn: Option<u64>) -> SessionEvent {
    encoded(CompactionEvent::Start(CompactionStart::new(
        CompactionId::new(compaction_id),
        None,
        CompactionTurn::new(turn),
    )))
}

/// A `compaction/end` row.
pub fn end(compaction_id: &str, turn: Option<u64>) -> SessionEvent {
    encoded(CompactionEvent::End(CompactionEnd::new(
        CompactionId::new(compaction_id),
        None,
        CompactionTurn::new(turn),
        None,
    )))
}

/// A `compaction/summary` row for a non-empty shadowed span.
pub fn summary(
    compaction_id: &str,
    shadowed_seqs: &[SessionSeq],
    shadowed_token_count: u64,
) -> SessionEvent {
    let first = *shadowed_seqs.first().expect("a test summary shadows nodes");
    let last = *shadowed_seqs.last().expect("a test summary shadows nodes");
    encoded(CompactionEvent::Summary(Box::new(CompactionSummary {
        compaction_id: CompactionId::new(compaction_id),
        source_command_id: None,
        summary: vec![text("summary")],
        raw_output: None,
        llm_stream_call: false,
        shadowed_range: ShadowedRange::new(first, last),
        shadowed_seqs: shadowed_seqs.to_vec(),
        shadowed_token_count,
        provider: "mock".to_string(),
        model: "mock".to_string(),
        max_tokens: None,
        usage: None,
    })))
}

/// A `compaction/summary` row with an explicit declared range.
pub fn summary_with_range(
    compaction_id: &str,
    range: ShadowedRange,
    shadowed_seqs: &[SessionSeq],
    shadowed_token_count: u64,
) -> SessionEvent {
    encoded(CompactionEvent::Summary(Box::new(CompactionSummary {
        compaction_id: CompactionId::new(compaction_id),
        source_command_id: None,
        summary: vec![text("summary")],
        raw_output: None,
        llm_stream_call: false,
        shadowed_range: range,
        shadowed_seqs: shadowed_seqs.to_vec(),
        shadowed_token_count,
        provider: "mock".to_string(),
        model: "mock".to_string(),
        max_tokens: None,
        usage: None,
    })))
}

/// A checkpoint user message that names `compaction_id`.
pub fn checkpoint(compaction_id: &str, id: &str) -> SessionEvent {
    let source = dsh_compaction::compact_checkpoint_source(CompactionId::new(compaction_id), None);
    user_with_source(id, vec![text("summary")], source.to_message_source())
}

/// A `compaction/prune` row for one shadowed surface node.
pub fn prune(original: SessionSeq, shadowed_token_count: u64) -> SessionEvent {
    encoded(CompactionEvent::Prune(CompactionPrune::new(
        ShadowedRange::new(original, original),
        vec![original],
        shadowed_token_count,
    )))
}

/// A raw `compaction/prune` row with several shadowed nodes.
pub fn prune_many(shadowed_seqs: &[SessionSeq], shadowed_token_count: u64) -> SessionEvent {
    let first = *shadowed_seqs.first().expect("a test prune shadows nodes");
    let last = *shadowed_seqs.last().expect("a test prune shadows nodes");
    encoded(CompactionEvent::Prune(CompactionPrune::new(
        ShadowedRange::new(first, last),
        shadowed_seqs.to_vec(),
        shadowed_token_count,
    )))
}

/// A raw extension row with an explicit tag and payload map.
pub fn extension_fields(kind: &str, fields: &[(&str, Value)]) -> SessionEvent {
    let mut payload = Map::new();
    for (key, value) in fields {
        payload.insert((*key).to_string(), value.clone());
    }
    SessionEvent::extension_event(kind, payload)
}
