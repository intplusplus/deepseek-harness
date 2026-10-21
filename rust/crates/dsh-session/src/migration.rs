//! Historical session format migration chain (v0 → v1 → v2 → v3 → v4).
//!
//! The TypeScript harness kept four generations of the session wire format.
//! Any older log (v0–v3) can be loaded and normalized up to the current
//! [`SESSION_FORMAT_VERSION`] before it is folded into a [`Session`][crate::log::Session].
//!
//! Each migration step is modeled after the corresponding TypeScript migration
//! package (`session-format-v0-to-v1` … `session-format-v3-to-v4`). The work
//! happens at the JSON `Value` level because the envelope (`type`/`seq`/`time`/
//! `data`) is a TypeScript-physical concern: the Rust core models only the
//! flattened `data` payload, so the chain produces those flat events and lets
//! [`Session::append`][crate::log::Session::append] re-admit them by position.
//!
//! Two Rust-specific normalizations happen at the very end, because the Rust
//! `SessionEvent` vocabulary differs slightly from the TypeScript wire shape:
//!
//! 1. TypeScript leaves `user/message` *inline* (its payload **is** the
//!    `UserMessage`); the Rust [`CoreEvent::UserMessage`][crate::event::CoreEvent]
//!    wraps it under `message`. The chain wraps it.
//! 2. TypeScript messages carry a `role` discriminator that the Rust message
//!    types drop (the role is derived from the event tag). Roles are stripped so
//!    the migrated event round-trips losslessly through
//!    [`Session::append`][crate::log::Session::append]'s JSON check.

#![forbid(unsafe_code)]

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use super::types::{SessionError, SESSION_FORMAT_VERSION};

/// One stored session event together with its TypeScript-physical envelope.
///
/// `data` is the event payload exactly as the TypeScript `data` field carried
/// it (or, for a Rust-native flat log, the whole event minus its `type` tag).
/// The envelope metadata (`seq`/`time`/`ignorable`/`sourceEventSeqs`/
/// `surfaceOp`) is surface-layer bookkeeping the Rust core does not model, so it
/// is dropped on the way out.
#[derive(Clone, Debug)]
pub struct Envelope {
    pub r#type: String,
    pub seq: u64,
    pub time: u64,

    pub data: Map<String, Value>,

    #[allow(dead_code)]
    pub ignorable: Option<bool>,
    pub source_event_seqs: Option<Vec<u64>>,
    pub surface_op: Option<Value>,
}

impl Envelope {
    fn new(r#type: impl Into<String>, data: Map<String, Value>) -> Self {
        Self {
            r#type: r#type.into(),
            seq: 0,
            time: 0,
            data,

            ignorable: None,
            source_event_seqs: None,
            surface_op: None,
        }
    }
}

/// The format version stamped in a header value, if present.
pub fn detect_version(header: &Value) -> Option<u32> {
    header.get("version").and_then(Value::as_u64).map(|v| v as u32)
}

// ---------------------------------------------------------------------------
// Value accessors
// ---------------------------------------------------------------------------

fn as_u64(value: &Value) -> Option<u64> {
    value.as_u64()
}

fn as_bool(value: &Value) -> Option<bool> {
    value.as_bool()
}

fn as_array(value: &Value) -> Option<&Vec<Value>> {
    value.as_array()
}

fn as_object(value: &Value) -> Option<&Map<String, Value>> {
    value.as_object()
}

fn as_object_mut(value: &mut Value) -> Option<&mut Map<String, Value>> {
    value.as_object_mut()
}

fn str_of<'a>(map: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    map.get(key).and_then(Value::as_str)
}

/// Read a `u64` coordinate from a value, defaulting to `0`.
fn coord(value: Option<&Value>) -> u64 {
    value.and_then(as_u64).unwrap_or(0)
}

fn is_finish(chunk: &Value) -> bool {
    as_object(chunk)
        .and_then(|m| m.get("type"))
        .and_then(Value::as_str)
        == Some("finish")
}

fn legacy_message_id(session_id: &str, seq: u64) -> String {
    format!("legacy-message:{session_id}:{seq}")
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the full chain on a stored log: a header plus its envelopes.
///
/// Returns the normalized v4 header and the flat (Rust-native) events ready for
/// [`Session::restore`][crate::log::Session::restore]. Any generation below the
/// current version is migrated; an equal version is returned unchanged (already
/// v4), and a higher version is refused.
pub fn migrate_session(
    mut header: Value,
    envelopes: Vec<Envelope>,
    children: &[Value],
) -> Result<(Value, Vec<Value>), SessionError> {
    let version = detect_version(&header).ok_or_else(|| {
        SessionError::Json("session header is missing a format version".to_string())
    })?;

    if version > SESSION_FORMAT_VERSION {
        return Err(SessionError::UnsupportedVersion {
            found: version,
            current: SESSION_FORMAT_VERSION,
        });
    }
    if version == SESSION_FORMAT_VERSION {
        // Already current; only the final flattening is needed.
        let flat = envelopes.into_iter().map(emit_flat).collect();
        finalize_header(&mut header);
        return Ok((header, flat));
    }

    let events = v0_to_v1(&mut header, envelopes)?;
    let events = v1_to_v2(&mut header, events)?;
    let events = v2_to_v3(&mut header, events)?;
    let events = v3_to_v4(&mut header, events, children)?;

    finalize_header(&mut header);
    let flat = events.into_iter().map(emit_flat).collect();
    Ok((header, flat))
}

/// Ensure the header is a clean v4 Rust-native header: drop the physical
/// `type` discriminator, pin the version, and guarantee `isSeeded` exists.
fn finalize_header(header: &mut Value) {
    if let Some(map) = as_object_mut(header) {
        map.remove("type");
        map.insert("version".into(), json!(SESSION_FORMAT_VERSION));
        if !map.contains_key("isSeeded") {
            map.insert("isSeeded".into(), json!(false));
        }
    }
}

/// Flatten one migrated envelope into a Rust-native flat event.
///
/// This is the single place where the Rust vocabulary diverges from the
/// TypeScript `data` payload: `user/message` is wrapped under `message`, and
/// `role` discriminators are stripped from message objects (they are derived
/// from the event tag in Rust and would otherwise break the lossless round-trip
/// check in [`Session::append`][crate::log::Session::append]).
fn emit_flat(envelope: Envelope) -> Value {
    let mut object = Map::new();
    object.insert("type".into(), Value::String(envelope.r#type.clone()));
    let mut data = envelope.data;

    match envelope.r#type.as_str() {
        "user/message" => {
            // TypeScript leaves user/message inline; Rust wraps it.
            data.remove("role");
            object.insert("message".into(), Value::Object(data));
        }
        "assistant/message" | "tool/result" | "developer/message" | "system/message" => {
            strip_role_from_message(&mut data);
            object.extend(data);
        }
        _ => {
            strip_role_from_message(&mut data);
            object.extend(data);
        }
    }

    Value::Object(object)
}

fn strip_role_from_message(data: &mut Map<String, Value>) {
    if let Some(message) = data.get_mut("message") {
        if let Some(message_map) = as_object_mut(message) {
            message_map.remove("role");
        }
    }
}

fn set_version(header: &mut Value, version: u32) {
    if let Some(map) = as_object_mut(header) {
        map.insert("version".into(), json!(version));
    }
}

// ---------------------------------------------------------------------------
// v0 → v1 : normalize pre-react-loop / released-v0 payloads
// ---------------------------------------------------------------------------

/// State threaded through the v0 → v1 normalization.
#[derive(Default)]
struct V0State {
    message_ids: HashMap<u64, String>,
    retry_ids: HashMap<String, String>,
    compaction_id: Option<String>,
}

/// Promote a released v0 session into v1.
pub fn v0_to_v1(header: &mut Value, envelopes: Vec<Envelope>) -> Result<Vec<Envelope>, SessionError> {
    set_version(header, 1);
    let session_id = str_of(
        as_object(header).unwrap_or(&Map::new()),
        "id",
    )
    .unwrap_or_default()
    .to_string();
    let mut state = V0State::default();
    let mut out = Vec::with_capacity(envelopes.len());
    for envelope in envelopes {
        let (r#type, data) =
            normalize_v0_event(envelope.r#type, envelope.seq, envelope.data, &session_id, &mut state)?;
        let mut migrated = Envelope::new(r#type, data);
        migrated.seq = envelope.seq;
        migrated.time = envelope.time;
        out.push(migrated);
    }
    Ok(out)
}

fn normalize_v0_event(
    r#type: String,
    seq: u64,
    data: Map<String, Value>,
    session_id: &str,
    state: &mut V0State,
) -> Result<(String, Map<String, Value>), SessionError> {
    let r#type = rename_compaction_type(&r#type);
    assert_supported_legacy_type(&r#type, &data)?;

    if r#type == "steering/message" {
        return normalize_steering(data, session_id, seq);
    }

    let mut data = data;
    data = normalize_turn_start(&r#type, data);
    data = normalize_turn_end(&r#type, data)?;
    data = normalize_request_header(&r#type, data);
    data = normalize_retry(&r#type, data, session_id, seq, state);
    data = normalize_compaction(&r#type, data, session_id, seq, state)?;
    data = normalize_message(&r#type, data, session_id, seq, state)?;
    Ok((r#type, data))
}

/// `compact/*` became `compaction/*`.
fn rename_compaction_type(r#type: &str) -> String {
    match r#type {
        "compact/start" => "compaction/start",
        "compact/summary" => "compaction/summary",
        "compact/end" => "compaction/end",
        "compact/prune" => "compaction/prune",
        other => other,
    }
    .to_string()
}

fn assert_supported_legacy_type(r#type: &str, data: &Map<String, Value>) -> Result<(), SessionError> {
    if r#type == "request/header-delta" || r#type == "mode/set" {
        return Err(SessionError::Migration(format!(
            "unsupported legacy event `{type}` cannot be migrated"
        )));
    }
    if r#type == "request/header" && str_of(data, "reason") == Some("fallback") {
        return Err(SessionError::Migration(
            "unsupported request/header reason `fallback` cannot be migrated".to_string(),
        ));
    }
    Ok(())
}

fn normalize_turn_start(r#type: &str, data: Map<String, Value>) -> Map<String, Value> {
    if r#type == "turn/start" && data.contains_key("trigger") {
        let turn = data.get("turn").cloned().unwrap_or(Value::Null);
        let mut out = Map::new();
        out.insert("turn".into(), turn);
        return out;
    }
    data
}

fn normalize_turn_end(r#type: &str, mut data: Map<String, Value>) -> Result<Map<String, Value>, SessionError> {
    if r#type != "turn/end" {
        return Ok(data);
    }
    let Some(reason) = data.get("reason").cloned() else {
        return Ok(data);
    };
    let Value::Object(reason_map) = &reason else {
        return Ok(data);
    };
    let kind = str_of(reason_map, "kind");
    let replacement: Option<Value> = match kind {
        Some("completed") | Some("blocked") | Some("max-tokens") | Some("interrupted") => None,
        Some("aborted") => {
            if reason_map.contains_key("reason") {
                None
            } else {
                Some(json!({ "kind": "aborted", "reason": { "kind": "legacy" } }))
            }
        }
        Some("disposed") => Some(json!({ "kind": "aborted", "reason": { "kind": "disposed" } })),
        Some("error") => {
            if reason_map.contains_key("error") {
                None
            } else {
                Some(json!({ "kind": "error", "error": normalize_legacy_error_reason(reason_map)? }))
            }
        }
        _ => None,
    };
    if let Some(replacement) = replacement {
        data.insert("reason".into(), replacement);
    }
    Ok(data)
}

fn normalize_legacy_error_reason(reason: &Map<String, Value>) -> Result<Map<String, Value>, SessionError> {
    // A v0 error reason without an `error` record carries `message`/`code`
    // (and optionally `code`) at its own level.
    let message = str_of(reason, "message").unwrap_or_default().to_string();
    let code = match str_of(reason, "code") {
        Some(code) => code.to_string(),
        None => "UNKNOWN".to_string(),
    };
    let mut error = Map::new();
    error.insert("message".into(), json!(message));
    error.insert("code".into(), json!(code));
    let mut out = Map::new();
    out.insert("kind".into(), json!("error"));
    out.insert("error".into(), Value::Object(error));
    Ok(out)
}

fn normalize_request_header(r#type: &str, mut data: Map<String, Value>) -> Map<String, Value> {
    if r#type != "request/header" {
        return data;
    }
    if let Some(header) = data.get_mut("header").and_then(as_object_mut) {
        header.remove("messagePrefix");
    }
    data
}

fn chain_key(data: &Map<String, Value>) -> String {
    ["turn", "step", "provider", "policyKey"]
        .iter()
        .map(|key| data.get(*key).map(Value::to_string).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\u{0}")
}

fn normalize_retry(
    r#type: &str,
    mut data: Map<String, Value>,
    session_id: &str,
    seq: u64,
    state: &mut V0State,
) -> Map<String, Value> {
    if r#type != "llm/retry" {
        return data;
    }
    match str_of(&data, "retryId") {
        Some(existing) if !existing.is_empty() => {
            state.retry_ids.insert(chain_key(&data), existing.to_string());
        }
        _ => {
            if !data.contains_key("retryId") {
                let migrated = state
                    .retry_ids
                    .get(&chain_key(&data))
                    .cloned()
                    .unwrap_or_else(|| format!("legacy-retry:{session_id}:{seq}"));
                state.retry_ids.insert(chain_key(&data), migrated.clone());
                data.insert("retryId".into(), json!(migrated));
            }
        }
    }
    data
}

fn normalize_compaction(
    r#type: &str,
    mut data: Map<String, Value>,
    session_id: &str,
    seq: u64,
    state: &mut V0State,
) -> Result<Map<String, Value>, SessionError> {
    match r#type {
        "session/end-seed" => {
            state.compaction_id = None;
        }
        "compaction/start" => match str_of(&data, "compactionId") {
            Some(existing) if !existing.is_empty() => {
                state.compaction_id = Some(existing.to_string());
            }
            _ => {
                if !data.contains_key("compactionId") {
                    let id = format!("legacy-compaction:{session_id}:{seq}");
                    data.insert("compactionId".into(), json!(id.clone()));
                    state.compaction_id = Some(id);
                }
            }
        },
        _ => {
            if let Some(compaction_id) = state.compaction_id.clone() {
                if r#type == "compaction/summary" || r#type == "compaction/end" {
                    if !data.contains_key("compactionId") {
                        data.insert("compactionId".into(), json!(compaction_id.clone()));
                    }
                    if r#type == "compaction/end" {
                        state.compaction_id = None;
                    }
                } else if r#type == "user/message" {
                    if let Some(source) = data.get_mut("source").and_then(as_object_mut) {
                        let is_compact = str_of(source, "kind") == Some("plugin")
                            && str_of(source, "plugin") == Some("compact");
                        if is_compact && !source.contains_key("compactionId") {
                            source.insert("compactionId".into(), json!(compaction_id));
                        }
                    }
                }
            }
        }
    }
    Ok(data)
}

fn normalize_steering(
    data: Map<String, Value>,
    session_id: &str,
    seq: u64,
) -> Result<(String, Map<String, Value>), SessionError> {
    // A `steering/message` becomes a `user/message`.
    if data.contains_key("message") {
        // Already wrapped: forward the inner message as a `user/message`.
        let message = data.get("message").cloned().unwrap_or(Value::Object(Map::new()));
        let mut out = Map::new();
        out.insert("message".into(), message);
        return Ok(("user/message".to_string(), out));
    }
    let mut out = Map::new();
    for (key, value) in data {
        if key == "turn" {
            continue;
        }
        out.insert(key, value);
    }
    out.insert("id".into(), json!(legacy_message_id(session_id, seq)));
    out.insert("role".into(), json!("user"));
    Ok(("user/message".to_string(), out))
}

fn normalize_message(
    r#type: &str,
    mut data: Map<String, Value>,
    session_id: &str,
    seq: u64,
    _state: &mut V0State,
) -> Result<Map<String, Value>, SessionError> {
    let has = |key: &str| data.contains_key(key);
    match r#type {
        "user/message" => {
            // Already normalized (released) shape keeps its inline payload.
            if has("id") || has("role") || has("message") || !has("content") || !has("source") {
                return Ok(data);
            }
            data.insert("id".into(), json!(legacy_message_id(session_id, seq)));
            data.insert("role".into(), json!("user"));
            Ok(data)
        }
        "assistant/message" => {
            if has("message") || !has("content") || !has("provenance") {
                return Ok(data);
            }
            let content = data.remove("content").unwrap_or(Value::Array(Vec::new()));
            let provenance = match data.remove("provenance") {
                Some(Value::Object(map)) => map,
                _ => Map::new(),
            };
            let mut source = Map::new();
            source.insert("kind".into(), json!("model"));
            if let Some(provider) = provenance.get("provider") {
                source.insert("provider".into(), provider.clone());
            }
            if let Some(model) = provenance.get("model") {
                source.insert("model".into(), model.clone());
            }
            let mut message = Map::new();
            message.insert("id".into(), json!(legacy_message_id(session_id, seq)));
            message.insert("role".into(), json!("assistant"));
            message.insert("content".into(), content);
            message.insert("source".into(), Value::Object(source));
            data.insert("message".into(), Value::Object(message));
            Ok(data)
        }
        "tool/result" => {
            if has("message") || !has("callId") || !has("content") || !has("isError") {
                return Ok(data);
            }
            let call_id = data.get("callId").cloned().unwrap_or(Value::String(String::new()));
            let content = data.remove("content").unwrap_or(Value::Array(Vec::new()));
            let is_error = data.get("isError").cloned().unwrap_or(Value::Bool(false));
            let mut block = Map::new();
            block.insert("type".into(), json!("tool-result"));
            block.insert("toolCallId".into(), call_id);
            block.insert("content".into(), content);
            block.insert("isError".into(), is_error);
            let mut message = Map::new();
            message.insert("id".into(), json!(legacy_message_id(session_id, seq)));
            message.insert("role".into(), json!("user"));
            message.insert("content".into(), Value::Array(vec![Value::Object(block)]));
            let mut source = Map::new();
            source.insert("kind".into(), json!("tool"));
            message.insert("source".into(), Value::Object(source));
            data.insert("message".into(), Value::Object(message));
            Ok(data)
        }
        _ => Ok(data),
    }
}

// ---------------------------------------------------------------------------
// v1 → v2 : embed released-v1 top-level assistant chunks into v2 attempts
// ---------------------------------------------------------------------------

struct PendingAttempt {
    turn: u64,
    step: u64,
    stream: Vec<Value>,
    original_seqs: Vec<u64>,
    terminal: bool,
    after_last: Vec<Envelope>,
}

impl PendingAttempt {
    fn new(turn: u64, step: u64) -> Self {
        Self {
            turn,
            step,
            stream: Vec::new(),

            original_seqs: Vec::new(),
            terminal: false,
            after_last: Vec::new(),
        }
    }
}

/// Embed released-v1 top-level `assistant/chunk` runs into v2 `assistant/attempt`
/// events, and attach the stream to the consuming `assistant/message`.
pub fn v1_to_v2(
    header: &mut Value,
    envelopes: Vec<Envelope>,
) -> Result<Vec<Envelope>, SessionError> {
    set_version(header, 2);
    let mut out = Vec::new();
    let mut pending: Option<PendingAttempt> = None;

    for mut envelope in envelopes {
        match envelope.r#type.as_str() {
            "assistant/chunk" => {
                let turn = coord(envelope.data.get("turn"));
                let step = coord(envelope.data.get("step"));
                if let Some(pending) = &mut pending {
                    if pending.terminal || pending.turn != turn || pending.step != step {
                        flush_attempt(&mut out, pending.take());
                    }
                }
                let pending = pending.get_or_insert_with(|| PendingAttempt::new(turn, step));
                let chunk = envelope.data.get("chunk").cloned().unwrap_or(Value::Null);
                pending.stream.push(json!({
                    "type": "chunk",
                    "time": envelope.time,
                    "chunk": chunk,
                }));
                pending.original_seqs.push(envelope.seq);
                if is_finish(&chunk) {
                    pending.terminal = true;
                }
            }
            "assistant/message" => {
                let sources = envelope
                    .data
                    .get("sourceEventSeqs")
                    .and_then(as_array)
                    .map(|values| values.iter().filter_map(as_u64).collect::<Vec<_>>());
                let consumes = match (&mut pending, &sources) {
                    (Some(pending), Some(sources)) => {
                        *sources == pending.original_seqs && pending.terminal
                    }
                    _ => false,
                };
                if consumes {
                    let pending = pending.take().expect("pending present");
                    for after in pending.after_last {
                        out.push(after);
                    }
                    envelope
                        .data
                        .insert("stream".into(), Value::Array(pending.stream));
                    envelope.data.remove("sourceEventSeqs");
                    envelope.data.remove("surfaceOp");
                    out.push(envelope);
                    continue;
                }
                if !envelope.data.contains_key("stream") {
                    envelope.data.insert("stream".into(), Value::Array(Vec::new()));
                }
                envelope.data.remove("sourceEventSeqs");
                envelope.data.remove("surfaceOp");
                out.push(envelope);
            }
            _ => {
                let closes = matches!(
                    envelope.r#type.as_str(),
                    "turn/end" | "step/end" | "llm/retry" | "llm/retry-started"
                );
                if closes {
                    if let Some(pending) = pending.take() {
                        flush_attempt(&mut out, Some(pending));
                    }
                } else if let Some(pending) = &mut pending {
                    pending.after_last.push(envelope);
                    continue;
                }
                envelope.data.remove("sourceEventSeqs");
                envelope.data.remove("surfaceOp");
                out.push(envelope);
            }
        }
    }
    if let Some(pending) = pending.take() {
        flush_attempt(&mut out, Some(pending));
    }
    Ok(out)
}

fn flush_attempt(out: &mut Vec<Envelope>, pending: Option<PendingAttempt>) {
    if let Some(pending) = pending {
        let mut data = Map::new();
        data.insert("turn".into(), json!(pending.turn));
        data.insert("step".into(), json!(pending.step));
        data.insert("stream".into(), Value::Array(pending.stream));
        out.push(Envelope::new("assistant/attempt", data));
        for after in pending.after_last {
            out.push(after);
        }
    }
}

// ---------------------------------------------------------------------------
// v2 → v3 : promote system prompts, canonicalize, rename PTC vocabulary
// ---------------------------------------------------------------------------

/// Promote streaming system prompts into `system/message` events, strip the
/// `system` string from `request/header`, and rename the PTC (`tools-code-mode`
/// → `tools-ptc`, `tool/code-dispatch*` → `tool/ptc-dispatch*`) vocabulary.
pub fn v2_to_v3(
    header: &mut Value,
    envelopes: Vec<Envelope>,
) -> Result<Vec<Envelope>, SessionError> {
    set_version(header, 3);

    if let Some(map) = as_object_mut(header) {
        if str_of(map, "agentPreset") == Some("code") {
            map.insert("agentPreset".into(), json!("ptc"));
        }
    }

    let mut prompt = String::new();
    let mut step: Option<(u64, u64)> = None;
    let mut out = Vec::new();

    for mut envelope in envelopes {
        envelope.r#type = rename_ptc_event_type(&envelope.r#type);
        if envelope.r#type == "agent-preset/selected" {
            if str_of(&envelope.data, "agentPreset") == Some("code") {
                envelope
                    .data
                    .insert("agentPreset".into(), json!("ptc"));
            }
        }
        if envelope.r#type == "request/header" {
            if let Some(header_obj) = envelope.data.get_mut("header").and_then(as_object_mut) {
                if let Some(system) = header_obj.remove("system") {
                    let next = match &system {
                        Value::String(s) => s.clone(),
                        _ => String::new(),
                    };
                    if next != prompt {
                        if let Some(st) = step {
                            out.push(make_system_message(&envelope, st, &next));
                        }
                        prompt = next;
                    }
                }
            }
        }

        if envelope.r#type == "step/start" {
            step = Some((coord(envelope.data.get("turn")), coord(envelope.data.get("step"))));
        } else if matches!(envelope.r#type.as_str(), "step/end" | "turn/end") {
            step = None;
        }

        rename_message_source_plugin(&mut envelope.data);
        out.push(envelope);
    }
    Ok(out)
}

fn rename_ptc_event_type(r#type: &str) -> String {
    match r#type {
        "tool/code-dispatch-start" => "tool/ptc-dispatch-start",
        "tool/code-dispatch" => "tool/ptc-dispatch",
        other => other,
    }
    .to_string()
}

fn make_system_message(anchor: &Envelope, step: (u64, u64), prompt: &str) -> Envelope {
    let session_id = String::new();
    let id = format!(
        "v2-to-v3-system-{}-{}-{}",
        session_id, anchor.seq, anchor.r#type
    );
    let mut content = Vec::new();
    if !prompt.is_empty() {
        let mut block = Map::new();
        block.insert("type".into(), json!("text"));
        block.insert("text".into(), json!(prompt));
        content.push(Value::Object(block));
    }
    let mut message = Map::new();
    message.insert("id".into(), json!(id));
    message.insert("role".into(), json!("system"));
    message.insert("content".into(), Value::Array(content));
    let mut source = Map::new();
    source.insert("kind".into(), json!("plugin"));
    source.insert("plugin".into(), json!("@deepseek-ai/dsh-system-prompt"));
    message.insert("source".into(), Value::Object(source));

    let mut data = Map::new();
    data.insert("turn".into(), json!(step.0));
    data.insert("step".into(), json!(step.1));
    data.insert("message".into(), Value::Object(message));
    data.insert("surfaceOp".into(), json!("append"));
    Envelope::new("system/message", data)
}

fn rename_message_source_plugin(data: &mut Map<String, Value>) {
    // Rename a `tools-code-mode` source plugin to `tools-ptc` wherever a message
    // (or a list of messages) declares it.
    let rewrite = |source: &mut Map<String, Value>| {
        if str_of(source, "kind") == Some("plugin")
            && str_of(source, "plugin") == Some("tools-code-mode")
        {
            source.insert("plugin".into(), json!("tools-ptc"));
        }
    };

    if let Some(message) = data.get_mut("message").and_then(as_object_mut) {
        if let Some(source) = message.get_mut("source").and_then(as_object_mut) {
            rewrite(source);
        }
    }
    for key in ["inserted", "messages"] {
        if let Some(list) = data.get_mut(key).and_then(as_array) {
            for item in list.iter_mut() {
                if let Some(item_map) = as_object_mut(item) {
                    if let Some(source) = item_map.get_mut("source").and_then(as_object_mut) {
                        rewrite(source);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v3 → v4 : append historical child facts, lift tool results, rewrite sources
// ---------------------------------------------------------------------------

/// Names of event types that exist in released v3. Unknown (downstream plugin)
/// types are left untouched and surface as [`ExtensionEvent`]s in Rust.
const RELEASED_V3_EVENT_TYPES: &[&str] = &[
    "turn/start",
    "turn/end",
    "step/start",
    "step/end",
    "user/message",
    "developer/message",
    "system/message",
    "assistant/message",
    "assistant/attempt",
    "tool/call",
    "tool/result",
    "request/header",
    "request/context",
    "session/end-seed",
    "assistant/chunk",
    "llm/retry",
    "llm/retry-started",
    "compaction/start",
    "compaction/summary",
    "compaction/end",
    "compaction/prune",
    "session/title-llm-request",
    "session/title",
    "agent/inbox/spliced",
    "subagent/catalog",
    "goal/change",
    "command/done",
    "session-log-deepseek/delivery-accepted",
];

/// Convert a released v3 session into v4.
///
/// `children` is the explicit historical child evidence the TypeScript v3→v4
/// migration requires; an empty slice declares no children (so no
/// `subagent/catalog` events are synthesized).
pub fn v3_to_v4(
    header: &mut Value,
    envelopes: Vec<Envelope>,
    children: &[Value],
) -> Result<Vec<Envelope>, SessionError> {
    set_version(header, 4);

    let mut out = Vec::new();
    for mut envelope in envelopes {
        if !RELEASED_V3_EVENT_TYPES.contains(&envelope.r#type.as_str()) {
            // Unknown / downstream plugin event: keep verbatim (the Rust core
            // models it as an `ExtensionEvent`). The TypeScript migration only
            // re-admits it; no payload change is attempted here.
            out.push(envelope);
            continue;
        }
        if envelope.r#type == "tool/result" {
            lift_tool_result(&mut envelope.data);
        }
        rewrite_v3_message_source(&mut envelope.data);
        out.push(envelope);
    }

    // Synthesize `subagent/catalog` events for declared children the parent
    // catalog did not already record.
    let mut catalog_ids = std::collections::HashSet::new();
    for envelope in &out {
        if envelope.r#type == "subagent/catalog" {
            if let Some(id) = envelope.data.get("childId").and_then(Value::as_str) {
                catalog_ids.insert(id.to_string());
            }
        }
    }
    for child in children {
        let Some(child_map) = as_object(child) else {
            continue;
        };
        let child_id = match str_of(child_map, "childId") {
            Some(id) => id.to_string(),
            None => continue,
        };
        if catalog_ids.contains(&child_id) {
            continue;
        }
        let mut data = Map::new();
        data.insert("version".into(), json!(1));
        data.insert("childId".into(), json!(child_id));
        data.insert(
            "childCreatedAt".into(),
            json!(child_map
                .get("childCreatedAt")
                .and_then(as_u64)
                .unwrap_or(0)),
        );
        data.insert(
            "mode".into(),
            json!(str_of(child_map, "mode").unwrap_or("unknown")),
        );
        out.push(Envelope::new("subagent/catalog", data));
        catalog_ids.insert(child_id);
    }

    Ok(out)
}

/// Lift `isError` / `callId` from a v3 `tool/result` payload into its `message`.
fn lift_tool_result(data: &mut Map<String, Value>) {
    if let Some(message) = data.get_mut("message").and_then(as_object_mut) {
        if let Some(is_error) = data.get("isError").cloned() {
            message.insert("isError".into(), is_error);
        }
        if let Some(call_id) = data.get("callId").cloned() {
            message.insert("toolCallId".into(), call_id);
        }
    }
}

fn rewrite_v3_message_source(data: &mut Map<String, Value>) {
    rename_message_source_plugin(data);
}

// ---------------------------------------------------------------------------
// Parsing a stored log into envelopes
// ---------------------------------------------------------------------------

/// Parse one event line into an [`Envelope`].
///
/// A TypeScript-physical line carries a `{type, seq, time, data, ...}` envelope;
/// a Rust-native flat line is the event itself (`{type, ...payload}`). Both are
/// accepted so a single migration path serves every historical generation.
pub fn parse_event_line(line: &str, index: usize) -> Result<Envelope, SessionError> {
    let value: Value =
        serde_json::from_str(line).map_err(|error| SessionError::Json(error.to_string()))?;
    let Some(object) = as_object(&value) else {
        return Err(SessionError::Json("session event must be a JSON object".to_string()));
    };
    if let Some(data) = object.get("data") {
        let data = data.as_object().cloned().unwrap_or_default();
        let seq = object.get("seq").and_then(as_u64).unwrap_or(index as u64);
        let time = object.get("time").and_then(as_u64).unwrap_or(0);
        let r#type = str_of(object, "type")
            .ok_or_else(|| SessionError::Json("session event is missing a `type` tag".to_string()))?
            .to_string();
        let ignorable = object.get("ignorable").and_then(as_bool);
        let source_event_seqs = object
            .get("sourceEventSeqs")
            .and_then(as_array)
            .map(|values| values.iter().filter_map(as_u64).collect());
        let surface_op = object.get("surfaceOp").cloned();
        Ok(Envelope {
            r#type,
            seq,
            time,

            data,

            ignorable,
            source_event_seqs,
            surface_op,
        })
    } else {
        let r#type = str_of(object, "type")
            .ok_or_else(|| SessionError::Json("session event is missing a `type` tag".to_string()))?
            .to_string();
        let mut data = object.clone();
        data.remove("type");
        Ok(Envelope {
            r#type,
            seq: index as u64,
            time: 0,

            data,

            ignorable: None,
            source_event_seqs: None,
            surface_op: None,
        })
    }
}
