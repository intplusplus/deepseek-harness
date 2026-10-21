//! The v0 -> v1 edge: normalize pre-react-loop payloads into the released v1
//! shape (`origin/packages/session/session-format-v0-to-v1/src/migration.ts`).
//!
//! v0 and v1 share one physical layout; the edge only repairs payloads the
//! react-loop rewrite replaced: `compact/*` renames, `steering/message`
//! collapse, turn-reason shapes, `request/header.messagePrefix` removal, and
//! identity synthesis for messages, retries, and compactions.

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::types::SessionError;

use super::support::{
    assert_envelope, assert_payload, count, fail, field, keys, non_empty, object, object_mut,
    string,
};
use super::{Envelope, Log};

/// State threaded through the v0 normalization.
#[derive(Default)]
struct V0State {
    /// Message ids synthesized for legacy payloads, keyed by source position.
    message_ids: HashMap<u64, String>,
    /// Retry ids shared by one `(turn, step, provider, policyKey)` chain.
    retry_ids: HashMap<String, String>,
    compaction_id: Option<String>,
}

/// Promote a released v0 session into v1.
pub fn v0_to_v1(mut log: Log) -> Result<Log, SessionError> {
    if log.version()? != 0 {
        return Err(fail("expected format v0 header".to_string()));
    }
    if let Some(header) = log.header.as_object_mut() {
        header.insert("version".to_string(), json!(1));
    }
    let session_id = log
        .header
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let mut state = V0State::default();
    let mut events = Vec::with_capacity(log.events.len());
    let inherited_prefix = log.header.get("parentSession").is_some();
    let cut = log.cut;
    for (index, envelope) in log.events.into_iter().enumerate() {
        assert_envelope(&envelope, index)?;
        let normalized = normalize_event(envelope, &session_id, &mut state)?;
        assert_source_delivery_marker(&normalized, inherited_prefix, cut, &session_id)?;
        events.push(normalized);
    }
    log.events = events;
    Ok(log)
}

/// A delivery marker written by the current generation must name this session.
fn assert_source_delivery_marker(
    event: &Envelope,
    inherited_prefix: bool,
    cut: u64,
    session_id: &str,
) -> Result<(), SessionError> {
    if event.r#type != "session-log-deepseek/delivery-accepted" {
        return Ok(());
    }
    let accepted = event
        .data
        .get("sessionFormatVersion")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let inherited = inherited_prefix && event.seq < cut;
    if accepted == 0
        && !inherited
        && event.data.get("sessionId").and_then(Value::as_str) != Some(session_id)
    {
        return Err(fail(
            "current-generation delivery marker names the wrong Session".to_string(),
        ));
    }
    Ok(())
}

fn normalize_event(
    mut envelope: Envelope,
    session_id: &str,
    state: &mut V0State,
) -> Result<Envelope, SessionError> {
    envelope.r#type = rename_compaction_type(&envelope.r#type);
    assert_supported_legacy_type(&envelope.r#type, &envelope.data)?;

    if envelope.r#type == "steering/message" {
        return normalize_steering(envelope, session_id);
    }
    normalize_turn_start(&mut envelope);
    normalize_turn_end(&mut envelope)?;
    normalize_request_header(&mut envelope)?;
    normalize_retry(&mut envelope, session_id, state);
    normalize_compaction(&mut envelope, session_id, state)?;
    normalize_message(&mut envelope, session_id, state)?;

    // `assistant/chunk` keeps its raw chunk; the v1->v2 stage validates it.
    if envelope.r#type != "assistant/chunk" {
        assert_payload(&envelope.r#type, &envelope.data, envelope.seq, 0)?;
    }
    if let Some(id) = event_message_id(&envelope) {
        state.message_ids.insert(envelope.seq, id);
    }
    Ok(envelope)
}

/// `compact/*` became `compaction/*`.
fn rename_compaction_type(event_type: &str) -> String {
    match event_type {
        "compact/start" => "compaction/start",
        "compact/summary" => "compaction/summary",
        "compact/end" => "compaction/end",
        "compact/prune" => "compaction/prune",
        other => other,
    }
    .to_string()
}

/// Events the v1 reader never learned to replay.
fn assert_supported_legacy_type(
    event_type: &str,
    data: &Map<String, Value>,
) -> Result<(), SessionError> {
    if event_type == "request/header-delta" || event_type == "mode/set" {
        return Err(fail(format!(
            "unsupported legacy {event_type} event cannot be migrated"
        )));
    }
    if event_type == "request/header"
        && data.get("reason").and_then(Value::as_str) == Some("fallback")
    {
        return Err(fail(
            "unsupported request/header reason `fallback` cannot be migrated".to_string(),
        ));
    }
    Ok(())
}

/// A legacy `trigger` is dropped; the current start carries only the turn.
fn normalize_turn_start(envelope: &mut Envelope) {
    if envelope.r#type != "turn/start" || !envelope.data.contains_key("trigger") {
        return;
    }
    let turn = envelope.data.get("turn").cloned().unwrap_or(Value::Null);
    let mut data = Map::new();
    data.insert("turn".to_string(), turn);
    envelope.data = data;
}

fn normalize_turn_end(envelope: &mut Envelope) -> Result<(), SessionError> {
    if envelope.r#type != "turn/end" {
        return Ok(());
    }
    let seq = envelope.seq;
    let data = &envelope.data;
    count(
        field(data, "turn", "turn/end")?,
        &format!("turn/end {seq} turn"),
    )?;
    let reason = field(data, "reason", "turn/end")?;
    let reason = object(reason, &format!("turn/end {seq} reason"))?;
    let kind = string(
        field(reason, "kind", "turn/end reason")?,
        &format!("turn/end {seq} reason kind"),
    )?
    .to_string();
    let label = format!("turn/end {seq} reason");
    let replacement = match kind.as_str() {
        "completed" | "blocked" | "max-tokens" | "interrupted" => {
            keys(reason, &["kind"], &[], &label)?;
            None
        }
        "aborted" => {
            if reason.contains_key("reason") {
                None
            } else {
                keys(reason, &["kind"], &[], &label)?;
                Some(json!({ "kind": "aborted", "reason": { "kind": "legacy" } }))
            }
        }
        "disposed" => {
            keys(reason, &["kind"], &[], &label)?;
            Some(json!({ "kind": "aborted", "reason": { "kind": "disposed" } }))
        }
        "error" => {
            if reason.contains_key("error") {
                None
            } else {
                Some(json!({
                    "kind": "error",
                    "error": normalize_legacy_error(reason, seq)?,
                }))
            }
        }
        _ => None,
    };
    if let Some(replacement) = replacement {
        envelope.data.insert("reason".to_string(), replacement);
    }
    Ok(())
}

/// Lift a pre-react-loop `{message, code?}` failure into the structured record.
fn normalize_legacy_error(
    reason: &Map<String, Value>,
    seq: u64,
) -> Result<Map<String, Value>, SessionError> {
    let label = format!("turn/end {seq} reason");
    count(field(reason, "step", &label)?, &format!("{label} step"))?;
    if let Some(failure) = reason.get("failure") {
        keys(reason, &["kind", "step", "failure"], &[], &label)?;
        let failure = object(failure, &format!("{label} failure"))?;
        keys(
            failure,
            &["message", "code"],
            &["status", "providerRetryAfterMs", "requestId"],
            &format!("{label} failure"),
        )?;
        non_empty(
            field(failure, "message", &label)?,
            &format!("{label} failure message"),
        )?;
        non_empty(
            field(failure, "code", &label)?,
            &format!("{label} failure code"),
        )?;
        return Ok(failure.clone());
    }
    keys(reason, &["kind", "step", "message"], &["code"], &label)?;
    let message = string(
        field(reason, "message", &label)?,
        &format!("{label} message"),
    )?;
    let code = match reason.get("code") {
        Some(code) => string(code, &format!("{label} code"))?.to_string(),
        None => "UNKNOWN".to_string(),
    };
    let mut error = Map::new();
    error.insert("message".to_string(), json!(message));
    error.insert("code".to_string(), json!(code));
    Ok(error)
}

fn normalize_request_header(envelope: &mut Envelope) -> Result<(), SessionError> {
    if envelope.r#type != "request/header" {
        return Ok(());
    }
    let seq = envelope.seq;
    let Some(value) = envelope.data.get_mut("header") else {
        return Err(fail(format!("request/header {seq} requires a header")));
    };
    let header = object_mut(value, &format!("request/header {seq} header"))?;
    if !header.contains_key("messagePrefix") {
        return Ok(());
    }
    if !header["messagePrefix"].is_array() {
        return Err(fail(format!(
            "malformed request/header messagePrefix at seq {seq}"
        )));
    }
    header.shift_remove("messagePrefix");
    Ok(())
}

fn chain_key(data: &Map<String, Value>) -> String {
    ["turn", "step", "provider", "policyKey"]
        .iter()
        .map(|key| {
            data.get(*key)
                .map(std::string::ToString::to_string)
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join("\u{0}")
}

fn normalize_retry(envelope: &mut Envelope, session_id: &str, state: &mut V0State) {
    if envelope.r#type != "llm/retry" {
        return;
    }
    let key = chain_key(&envelope.data);
    match envelope.data.get("retryId").and_then(Value::as_str) {
        Some(existing) if !existing.is_empty() => {
            state.retry_ids.insert(key, existing.to_string());
        }
        _ => {
            if envelope.data.contains_key("retryId") {
                return;
            }
            let migrated = state
                .retry_ids
                .get(&key)
                .cloned()
                .unwrap_or_else(|| format!("legacy-retry:{session_id}:{}", envelope.seq));
            state.retry_ids.insert(key, migrated.clone());
            envelope.data.insert("retryId".to_string(), json!(migrated));
        }
    }
}

fn normalize_compaction(
    envelope: &mut Envelope,
    session_id: &str,
    state: &mut V0State,
) -> Result<(), SessionError> {
    match envelope.r#type.as_str() {
        "session/end-seed" => {
            state.compaction_id = None;
        }
        "compaction/start" => match envelope.data.get("compactionId").and_then(Value::as_str) {
            Some(existing) if !existing.is_empty() => {
                state.compaction_id = Some(existing.to_string());
            }
            _ => {
                if envelope.data.contains_key("compactionId") {
                    return Ok(());
                }
                let id = format!("legacy-compaction:{session_id}:{}", envelope.seq);
                envelope.data.insert("compactionId".to_string(), json!(id));
                state.compaction_id = Some(id);
            }
        },
        _ => {
            let Some(compaction_id) = state.compaction_id.clone() else {
                return Ok(());
            };
            if envelope.r#type == "compaction/summary" || envelope.r#type == "compaction/end" {
                if !envelope.data.contains_key("compactionId") {
                    envelope
                        .data
                        .insert("compactionId".to_string(), json!(compaction_id));
                }
                if envelope.r#type == "compaction/end" {
                    state.compaction_id = None;
                }
                return Ok(());
            }
            if envelope.r#type != "user/message" {
                return Ok(());
            }
            let Some(source) = envelope
                .data
                .get_mut("source")
                .and_then(Value::as_object_mut)
            else {
                return Ok(());
            };
            let is_compact = source.get("kind").and_then(Value::as_str) == Some("plugin")
                && source.get("plugin").and_then(Value::as_str) == Some("compact");
            if is_compact && !source.contains_key("compactionId") {
                source.insert("compactionId".to_string(), json!(compaction_id));
            }
        }
    }
    Ok(())
}

/// A `steering/message` becomes a `user/message`.
fn normalize_steering(mut envelope: Envelope, session_id: &str) -> Result<Envelope, SessionError> {
    let seq = envelope.seq;
    let label = format!("steering/message {seq} data");
    if let Some(wrapped) = envelope.data.get("message") {
        keys(&envelope.data, &["turn", "message"], &[], &label)?;
        count(
            field(&envelope.data, "turn", &label)?,
            &format!("{label} turn"),
        )?;
        let message = object(wrapped, &format!("{label} message"))?.clone();
        envelope.r#type = "user/message".to_string();
        envelope.data = message;
        assert_payload("user/message", &envelope.data, seq, 0)?;
        return Ok(envelope);
    }
    keys(&envelope.data, &["turn", "content", "source"], &[], &label)?;
    count(
        field(&envelope.data, "turn", &label)?,
        &format!("{label} turn"),
    )?;
    envelope.data.shift_remove("turn");
    envelope.r#type = "user/message".to_string();
    envelope
        .data
        .insert("id".to_string(), json!(legacy_message_id(session_id, seq)));
    envelope.data.insert("role".to_string(), json!("user"));
    assert_payload("user/message", &envelope.data, seq, 0)?;
    Ok(envelope)
}

fn normalize_message(
    envelope: &mut Envelope,
    session_id: &str,
    state: &V0State,
) -> Result<(), SessionError> {
    match envelope.r#type.as_str() {
        "user/message" => {
            let has = |key: &str| envelope.data.contains_key(key);
            if has("id") || has("role") || has("message") || !has("content") || !has("source") {
                return Ok(());
            }
            envelope.data.insert(
                "id".to_string(),
                json!(legacy_message_id(session_id, envelope.seq)),
            );
            envelope.data.insert("role".to_string(), json!("user"));
        }
        "assistant/message" => {
            let has = |key: &str| envelope.data.contains_key(key);
            if has("message") || !has("content") || !has("provenance") {
                return Ok(());
            }
            let content = envelope.data.shift_remove("content").unwrap_or(Value::Null);
            let provenance = envelope
                .data
                .shift_remove("provenance")
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            let mut source = provenance;
            source.insert("kind".to_string(), json!("model"));
            let mut message = Map::new();
            message.insert(
                "id".to_string(),
                json!(legacy_message_id(session_id, envelope.seq)),
            );
            message.insert("role".to_string(), json!("assistant"));
            message.insert("content".to_string(), content);
            message.insert("source".to_string(), Value::Object(source));
            envelope
                .data
                .insert("message".to_string(), Value::Object(message));
        }
        "tool/result" => {
            let has = |key: &str| envelope.data.contains_key(key);
            if has("message") || !has("callId") || !has("content") || !has("isError") {
                return Ok(());
            }
            let Some(call_id) = envelope.data.get("callId").and_then(Value::as_str) else {
                return Ok(());
            };
            let call_id = call_id.to_string();
            let Some(is_error) = envelope.data.get("isError").and_then(Value::as_bool) else {
                return Ok(());
            };
            let content = envelope.data.shift_remove("content").unwrap_or(Value::Null);
            envelope.data.shift_remove("callId");
            envelope.data.shift_remove("isError");
            let message_id = match replacement_start(envelope) {
                Some(start) => state.message_ids.get(&start).cloned().ok_or_else(|| {
                    fail(format!(
                        "tool/result {} replacement cites a message without identity",
                        envelope.seq
                    ))
                })?,
                None => legacy_message_id(session_id, envelope.seq),
            };
            let mut block = Map::new();
            block.insert("type".to_string(), json!("tool-result"));
            block.insert("toolCallId".to_string(), json!(call_id));
            block.insert("content".to_string(), content);
            block.insert("isError".to_string(), json!(is_error));
            let mut source = Map::new();
            source.insert("kind".to_string(), json!("tool"));
            source.insert("callId".to_string(), json!(call_id));
            let mut message = Map::new();
            message.insert("id".to_string(), json!(message_id));
            message.insert("role".to_string(), json!("user"));
            message.insert(
                "content".to_string(),
                Value::Array(vec![Value::Object(block)]),
            );
            message.insert("source".to_string(), Value::Object(source));
            envelope
                .data
                .insert("message".to_string(), Value::Object(message));
        }
        _ => {}
    }
    Ok(())
}

/// The replacement start of one `surfaceOp: replace` envelope.
fn replacement_start(envelope: &Envelope) -> Option<u64> {
    let operation = envelope.surface_op.as_ref()?.as_object()?;
    if operation.get("op").and_then(Value::as_str) != Some("replace") {
        return None;
    }
    operation.get("start").and_then(Value::as_u64)
}

/// The message identity one normalized event carries, if any.
fn event_message_id(envelope: &Envelope) -> Option<String> {
    let message = if envelope.r#type == "user/message" {
        &envelope.data
    } else {
        envelope.data.get("message")?.as_object()?
    };
    message
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn legacy_message_id(session_id: &str, seq: u64) -> String {
    format!("legacy-message:{session_id}:{seq}")
}
