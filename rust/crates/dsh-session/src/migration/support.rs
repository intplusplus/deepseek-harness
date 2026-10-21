//! Shared admission helpers for the session-format migration chain.
//!
//! Every rule here is transcribed from the TypeScript reference implementation
//! (`origin/packages/session/session-format-*`). The helpers are deliberately
//! strict: a historical row that the TypeScript reader refuses must be refused
//! here too, because silently dropping or guessing at a payload would make the
//! Rust reader accept a session the writer never produced.

use serde_json::{json, Map, Value};

use crate::types::{SessionError, MAX_SAFE_INTEGER};

use super::Envelope;

/// A JSON object payload.
pub(crate) type JsonObject = Map<String, Value>;

/// One refused historical artifact.
pub(crate) fn fail(message: impl Into<String>) -> SessionError {
    SessionError::Migration(message.into())
}

/// Require one JSON object.
pub(crate) fn object<'a>(value: &'a Value, label: &str) -> Result<&'a JsonObject, SessionError> {
    value
        .as_object()
        .ok_or_else(|| fail(format!("{label} must be a JSON object")))
}

/// Require one mutable JSON object.
pub(crate) fn object_mut<'a>(
    value: &'a mut Value,
    label: &str,
) -> Result<&'a mut JsonObject, SessionError> {
    value
        .as_object_mut()
        .ok_or_else(|| fail(format!("{label} must be a JSON object")))
}

/// Require one JSON array.
pub(crate) fn array<'a>(value: &'a Value, label: &str) -> Result<&'a [Value], SessionError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| fail(format!("{label} must be an array")))
}

/// Require one JSON array, mutably.
pub(crate) fn array_mut<'a>(
    value: &'a mut Value,
    label: &str,
) -> Result<&'a mut Vec<Value>, SessionError> {
    value
        .as_array_mut()
        .ok_or_else(|| fail(format!("{label} must be an array")))
}

/// Require a non-negative safe integer.
///
/// JavaScript treats `1.0` as `Number.isSafeInteger(1)`, and `serde_json`
/// keeps an integral float as `f64`, so an integral `1.0` is accepted.
pub(crate) fn count(value: &Value, label: &str) -> Result<u64, SessionError> {
    if let Some(number) = value.as_u64() {
        if number <= MAX_SAFE_INTEGER {
            return Ok(number);
        }
    } else if let Some(number) = value.as_f64() {
        if number.fract() == 0.0 && (0.0..=MAX_SAFE_INTEGER as f64).contains(&number) {
            return Ok(number as u64);
        }
    }
    Err(fail(format!("{label} must be a non-negative safe integer")))
}

/// Require a strictly positive safe integer.
pub(crate) fn positive(value: &Value, label: &str) -> Result<u64, SessionError> {
    let number = count(value, label)?;
    if number == 0 {
        return Err(fail(format!("{label} must be positive")));
    }
    Ok(number)
}

/// Require a finite JSON number.
pub(crate) fn finite(value: &Value, label: &str) -> Result<f64, SessionError> {
    match value.as_f64() {
        Some(number) if number.is_finite() => Ok(number),
        _ => Err(fail(format!("{label} must be a finite number"))),
    }
}

/// Require a JSON string.
pub(crate) fn string<'a>(value: &'a Value, label: &str) -> Result<&'a str, SessionError> {
    value
        .as_str()
        .ok_or_else(|| fail(format!("{label} must be a string")))
}

/// Require a non-empty JSON string.
pub(crate) fn non_empty<'a>(value: &'a Value, label: &str) -> Result<&'a str, SessionError> {
    let text = string(value, label)?;
    if text.is_empty() {
        return Err(fail(format!("{label} must be a non-empty string")));
    }
    Ok(text)
}

/// Require a JSON boolean.
pub(crate) fn boolean(value: &Value, label: &str) -> Result<bool, SessionError> {
    value
        .as_bool()
        .ok_or_else(|| fail(format!("{label} must be a boolean")))
}

/// Require one of a fixed literal set.
pub(crate) fn literal<'a>(
    value: &'a Value,
    allowed: &[&str],
    label: &str,
) -> Result<&'a str, SessionError> {
    let text = string(value, label)?;
    if !allowed.contains(&text) {
        return Err(fail(format!(
            "{label} must be one of {}",
            allowed.join(", ")
        )));
    }
    Ok(text)
}

/// Require one of a fixed numeric literal set.
pub(crate) fn literal_number(
    value: &Value,
    allowed: &[u64],
    label: &str,
) -> Result<u64, SessionError> {
    let number = count(value, label)?;
    if !allowed.contains(&number) {
        return Err(fail(format!("{label} must be one of {:?}", allowed)));
    }
    Ok(number)
}

/// Require a member that must exist.
pub(crate) fn field<'a>(
    map: &'a JsonObject,
    key: &str,
    label: &str,
) -> Result<&'a Value, SessionError> {
    map.get(key)
        .ok_or_else(|| fail(format!("{label} lacks required member `{key}`")))
}

/// Require an object member that must exist.
pub(crate) fn field_object<'a>(
    map: &'a JsonObject,
    key: &str,
    label: &str,
) -> Result<&'a JsonObject, SessionError> {
    object(field(map, key, label)?, &format!("{label} {key}"))
}

/// Require an array member that must exist.
pub(crate) fn field_array<'a>(
    map: &'a JsonObject,
    key: &str,
    label: &str,
) -> Result<&'a [Value], SessionError> {
    array(field(map, key, label)?, &format!("{label} {key}"))
}

/// Require a string member that must exist.
pub(crate) fn field_str<'a>(
    map: &'a JsonObject,
    key: &str,
    label: &str,
) -> Result<&'a str, SessionError> {
    string(field(map, key, label)?, &format!("{label} {key}"))
}

/// Require every named member and no member outside the admitted set.
pub(crate) fn keys(
    map: &JsonObject,
    required: &[&str],
    optional: &[&str],
    label: &str,
) -> Result<(), SessionError> {
    if let Some(key) = required.iter().find(|key| !map.contains_key(**key)) {
        return Err(fail(format!("{label} lacks required member `{key}`")));
    }
    if let Some(key) = map
        .keys()
        .find(|key| !required.contains(&key.as_str()) && !optional.contains(&key.as_str()))
    {
        return Err(fail(format!("{label} has unexpected member `{key}`")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Envelope admission
// ---------------------------------------------------------------------------

/// Require one dense logical event envelope.
///
/// `index` is the event's position in the artifact, which the released reader
/// requires `seq` to equal.
pub(crate) fn assert_envelope(envelope: &Envelope, index: usize) -> Result<(), SessionError> {
    let label = format!("{} {}", envelope.r#type, envelope.seq);
    if envelope.seq != index as u64 {
        return Err(fail(format!(
            "Session event {index} has non-dense seq {}",
            envelope.seq
        )));
    }
    if envelope.ignorable == Some(false) {
        return Err(fail(format!("{label} ignorable must be true when present")));
    }
    if let Some(op) = &envelope.surface_op {
        assert_surface_op(op, envelope.seq, &label)?;
    } else if envelope.physical && is_surface_type(&envelope.r#type) {
        return Err(fail(format!("{label} requires a surfaceOp marker")));
    }
    Ok(())
}

/// Whether one event type is placed on the model-visible surface.
pub(crate) fn is_surface_type(event_type: &str) -> bool {
    matches!(
        event_type,
        "user/message" | "assistant/message" | "tool/result" | "system/message"
    )
}

/// Reject surface enumerations that point at themselves.
pub(crate) fn assert_surface_op(
    operation: &Value,
    seq: u64,
    label: &str,
) -> Result<(), SessionError> {
    if operation.as_str() == Some("append") {
        return Ok(());
    }
    let map = object(operation, &format!("{label} surfaceOp"))?;
    let (start_key, end_key) = if map.contains_key("start") {
        keys(
            map,
            &["op", "start", "end"],
            &[],
            &format!("{label} surfaceOp"),
        )?;
        ("start", "end")
    } else {
        keys(
            map,
            &["op", "startSeq", "endSeq"],
            &[],
            &format!("{label} surfaceOp"),
        )?;
        ("startSeq", "endSeq")
    };
    if map["op"].as_str() != Some("replace") {
        return Err(fail(format!("{label} surfaceOp must be replace")));
    }
    for key in [start_key, end_key] {
        if count(&map[key], &format!("{label} surface {key}"))? >= seq {
            return Err(fail(format!("{label} has an invalid surface replacement")));
        }
    }
    Ok(())
}

/// Validate one event's `sourceEventSeqs` list.
pub(crate) fn assert_source_seqs(
    envelope: &Envelope,
    allow_empty: bool,
) -> Result<(), SessionError> {
    let Some(sources) = &envelope.source_event_seqs else {
        return Ok(());
    };
    if sources.is_empty() && !allow_empty {
        return Err(fail(format!(
            "{} {} sourceEventSeqs must be non-empty",
            envelope.r#type, envelope.seq
        )));
    }
    let mut seen = Vec::with_capacity(sources.len());
    for source in sources {
        if *source >= envelope.seq || seen.contains(source) {
            return Err(fail(format!(
                "{} {} sourceEventSeqs must be unique earlier seqs",
                envelope.r#type, envelope.seq
            )));
        }
        seen.push(*source);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Frozen payload inventories
// ---------------------------------------------------------------------------

/// One frozen top-level payload disposition.
#[derive(Clone, Copy)]
pub(crate) struct Disposition {
    pub required: &'static [&'static str],
    pub optional: &'static [&'static str],
}

const fn disposition(
    required: &'static [&'static str],
    optional: &'static [&'static str],
) -> Disposition {
    Disposition { required, optional }
}

/// The released v0/v1 event inventory.
pub(crate) fn v0_disposition(event_type: &str) -> Option<Disposition> {
    Some(match event_type {
        "agent-preset/selected" => disposition(&["agentPreset"], &[]),
        "agent/inbox/spliced" => disposition(
            &["target", "start", "inserted"],
            &["removedCount", "outcome"],
        ),
        "approval/asked" => disposition(&["id", "toolName"], &["callId", "reason"]),
        "approval/decided" => disposition(&["id", "outcome"], &[]),
        "approval/policy" => disposition(&["policy"], &["source"]),
        "assistant/chunk" => disposition(&["turn", "step", "chunk"], &[]),
        "assistant/message" => disposition(&["turn", "step", "message"], &["usage", "interrupted"]),
        "command/done" => disposition(&["commandId", "kind"], &["text", "sourceEventSeq"]),
        "command/run" => disposition(&["commandId", "name", "source"], &["args"]),
        "compaction/end" => disposition(&["compactionId", "turn"], &["sourceCommandId", "error"]),
        "compaction/prune" => disposition(
            &["shadowedRange", "shadowedSeqs", "shadowedTokenCount"],
            &[],
        ),
        "compaction/start" => disposition(&["compactionId", "turn"], &["sourceCommandId"]),
        "compaction/summary" => disposition(
            &[
                "compactionId",
                "summary",
                "shadowedRange",
                "shadowedSeqs",
                "shadowedTokenCount",
                "provider",
                "model",
            ],
            &[
                "sourceCommandId",
                "maxTokens",
                "usage",
                "rawOutput",
                "llmStreamCall",
            ],
        ),
        "feedback/record" => disposition(&["text"], &[]),
        "goal/change" => disposition(
            &["kind", "version", "operation"],
            &[
                "goal",
                "roundsStarted",
                "createdAt",
                "updatedAt",
                "cleared",
                "clearedAt",
            ],
        ),
        "hook/invoked" => disposition(&["turn", "point", "dialect", "handlerId"], &["matcher"]),
        "hook/result" => disposition(
            &["turn", "point", "handlerId", "decision", "durationMs"],
            &["exitCode", "stderrSummary"],
        ),
        "llm/retry" => disposition(
            &[
                "retryId",
                "turn",
                "step",
                "provider",
                "mode",
                "policyKey",
                "retry",
                "delayMs",
                "failure",
            ],
            &["maxRetries"],
        ),
        "llm/retry-started" => disposition(&["retryId", "turn", "step", "retry"], &[]),
        "model/selection" => disposition(&["provider", "model"], &["reasoningEffort"]),
        "permission/preset" => disposition(&["preset"], &[]),
        "plan/mode" => disposition(&["active"], &[]),
        "request/context" => disposition(&["provider", "model"], &["contextWindow"]),
        "request/header" => disposition(&["header", "reason"], &["startsSeries"]),
        "sandbox/mode" => disposition(&["mode"], &["source"]),
        "schedule/change" => {
            disposition(&["version", "operation"], &["schedule", "id", "acceptedAt"])
        }
        "session-log-deepseek/delivery-accepted" => disposition(&["sessionId", "throughSeq"], &[]),
        "session/end-seed" => disposition(&[], &[]),
        "session/title" => disposition(&["title", "messageSeqs", "source"], &[]),
        "session/title-llm-request" => disposition(
            &[
                "titleProvider",
                "messageSeqs",
                "route",
                "system",
                "messages",
                "maxTokens",
            ],
            &[],
        ),
        "step/end" | "step/start" => disposition(&["turn", "step"], &[]),
        "subagent/descriptor" => disposition(
            &["mode", "version", "provider"],
            &[
                "label",
                "agentProvider",
                "agentModel",
                "agentReasoningEffort",
                "persona",
                "toolFilter",
            ],
        ),
        "subagent/model-selection-policy" => disposition(&["allowedModels"], &[]),
        "team/member" => disposition(&["version", "teamId", "member"], &[]),
        "team/message/delivered" => {
            disposition(&["version", "teamId", "messageId", "targetId"], &[])
        }
        "team/message/queued" => disposition(&["version", "teamId", "message"], &[]),
        "team/task" => disposition(&["version", "teamId", "task"], &[]),
        "todo/write" => disposition(&["todos"], &[]),
        "tool-workflow/agent-end" => disposition(&["runId", "seq", "outcome"], &[]),
        "tool-workflow/agent-start" => {
            disposition(&["runId", "seq", "label", "childId"], &["phase"])
        }
        "tool-workflow/run-end" => disposition(&["runId", "stopReason"], &[]),
        "tool-workflow/run-start" => disposition(&["runId", "name"], &[]),
        "tool/call" => disposition(&["turn", "step", "callId", "name", "arguments"], &[]),
        "tool/code-dispatch" => disposition(
            &[
                "rootCallId",
                "parentCallId",
                "subCallId",
                "name",
                "arguments",
                "isError",
                "content",
            ],
            &[],
        ),
        "tool/code-dispatch-start" => disposition(
            &[
                "rootCallId",
                "parentCallId",
                "subCallId",
                "name",
                "arguments",
            ],
            &[],
        ),
        "tool/result" => disposition(&["turn", "step", "message"], &["error", "meta"]),
        "turn/end" => disposition(&["turn", "reason"], &[]),
        "turn/start" => disposition(&["turn"], &[]),
        "user/message" => disposition(&["role", "id", "content", "source"], &[]),
        "web/deepseek-search-llm-request" => disposition(&["endpoint", "apiVersion", "body"], &[]),
        _ => return None,
    })
}

/// The released v2 event inventory: v1 minus the chunk/attempt collapse.
pub(crate) fn v2_disposition(event_type: &str) -> Option<Disposition> {
    Some(match event_type {
        "assistant/attempt" => disposition(&["turn", "step", "stream"], &[]),
        "assistant/message" => disposition(
            &["turn", "step", "message", "stream"],
            &["usage", "interrupted"],
        ),
        "session-log-deepseek/delivery-accepted" => {
            disposition(&["sessionId", "throughSeq"], &["sessionFormatVersion"])
        }
        "session/end-seed" => disposition(&[], &["inherited"]),
        "assistant/chunk" => return None,
        other => return v0_disposition(other),
    })
}

/// The released v3 event inventory: v2 plus promoted system messages, native
/// PTC dispatch tags, and feedback rows.
pub(crate) fn v3_disposition(event_type: &str) -> Option<Disposition> {
    Some(match event_type {
        "system/message" => disposition(&["turn", "step", "message"], &[]),
        "tool/ptc-dispatch" => disposition(
            &[
                "rootCallId",
                "parentCallId",
                "subCallId",
                "name",
                "arguments",
                "isError",
                "content",
            ],
            &[],
        ),
        "tool/ptc-dispatch-start" => disposition(
            &[
                "rootCallId",
                "parentCallId",
                "subCallId",
                "name",
                "arguments",
            ],
            &[],
        ),
        "feedback/message-put" => disposition(&["sessionId", "item"], &[]),
        "feedback/message-delete" => disposition(&["sessionId", "messageId"], &[]),
        other => return v2_disposition(other),
    })
}

// ---------------------------------------------------------------------------
// Payload semantics
// ---------------------------------------------------------------------------

/// Validate one known released payload after normalization.
///
/// `version` selects the versioned members (v0 has no
/// `sessionFormatVersion`, v2 adds `stream`, v3 adds promoted system messages).
pub(crate) fn assert_payload(
    event_type: &str,
    data: &JsonObject,
    seq: u64,
    version: u32,
) -> Result<(), SessionError> {
    let label = format!("{event_type} {seq}");
    let disposition = match version {
        0 | 1 => v0_disposition(event_type),
        2 => v2_disposition(event_type),
        _ => v3_disposition(event_type),
    }
    .ok_or_else(|| {
        fail(format!(
            "format v{version} contains unknown event type `{event_type}` at seq {seq}"
        ))
    })?;
    let optional = if event_type == "session-log-deepseek/delivery-accepted" && version == 1 {
        let mut optional = disposition.optional.to_vec();
        optional.push("sessionFormatVersion");
        optional
    } else {
        disposition.optional.to_vec()
    };
    keys(
        data,
        disposition.required,
        &optional,
        &format!("{label} data"),
    )?;
    assert_semantics(event_type, data, seq, version, &label)
}

/// Validate nested payload semantics for one admitted event.
fn assert_semantics(
    event_type: &str,
    data: &JsonObject,
    seq: u64,
    version: u32,
    label: &str,
) -> Result<(), SessionError> {
    match event_type {
        "agent-preset/selected" => {
            string(
                field(data, "agentPreset", label)?,
                &format!("{label} agentPreset"),
            )?;
        }
        "agent/inbox/spliced" => {
            literal(
                field(data, "target", label)?,
                &["next-turn", "next-step"],
                &format!("{label} target"),
            )?;
            count(field(data, "start", label)?, &format!("{label} start"))?;
            if let Some(removed) = data.get("removedCount") {
                count(removed, &format!("{label} removedCount"))?;
            }
            for (index, message) in field_array(data, "inserted", label)?.iter().enumerate() {
                assert_message(
                    message,
                    &format!("{label} inserted[{index}]"),
                    version,
                    Some("user"),
                )?;
            }
            if let Some(outcome) = data.get("outcome") {
                literal(outcome, &["canceled"], &format!("{label} outcome"))?;
            }
        }
        "approval/asked" => {
            non_empty(field(data, "id", label)?, &format!("{label} id"))?;
            non_empty(
                field(data, "toolName", label)?,
                &format!("{label} toolName"),
            )?;
            if let Some(call) = data.get("callId") {
                non_empty(call, &format!("{label} callId"))?;
            }
            if let Some(reason) = data.get("reason") {
                string(reason, &format!("{label} reason"))?;
            }
        }
        "approval/decided" => {
            non_empty(field(data, "id", label)?, &format!("{label} id"))?;
            literal(
                field(data, "outcome", label)?,
                &["allowed-once", "rejected", "cancelled", "unavailable"],
                &format!("{label} outcome"),
            )?;
        }
        "approval/policy" => {
            literal(
                field(data, "policy", label)?,
                &["ask", "never"],
                &format!("{label} policy"),
            )?;
            if let Some(source) = data.get("source") {
                literal(source, &["delegation"], &format!("{label} source"))?;
            }
        }
        "assistant/chunk" => {
            coordinate_pair(data, label)?;
            assert_stream_chunk(field(data, "chunk", label)?, &format!("{label} chunk"))?;
        }
        "assistant/message" => {
            coordinate_pair(data, label)?;
            let message = field(data, "message", label)?;
            assert_message(
                message,
                &format!("{label} message"),
                version,
                Some("assistant"),
            )?;
            if let Some(usage) = data.get("usage") {
                assert_usage(usage, &format!("{label} usage"))?;
            }
            if let Some(interrupted) = data.get("interrupted") {
                if interrupted.as_bool() != Some(true) {
                    return Err(fail(format!("{label} interrupted must be true")));
                }
            }
            if data.contains_key("stream") {
                assert_stream(field(data, "stream", label)?, &format!("{label} stream"))?;
            }
        }
        "assistant/attempt" => {
            positive(field(data, "turn", label)?, &format!("{label} turn"))?;
            positive(field(data, "step", label)?, &format!("{label} step"))?;
            assert_stream(field(data, "stream", label)?, &format!("{label} stream"))?;
        }
        "command/done" => {
            non_empty(
                field(data, "commandId", label)?,
                &format!("{label} commandId"),
            )?;
            literal(
                field(data, "kind", label)?,
                &["success", "error"],
                &format!("{label} kind"),
            )?;
            if let Some(text) = data.get("text") {
                string(text, &format!("{label} text"))?;
            }
            if let Some(source) = data.get("sourceEventSeq") {
                earlier(source, seq, &format!("{label} sourceEventSeq"))?;
            }
        }
        "command/run" => {
            non_empty(
                field(data, "commandId", label)?,
                &format!("{label} commandId"),
            )?;
            non_empty(field(data, "name", label)?, &format!("{label} name"))?;
            if let Some(args) = data.get("args") {
                string(args, &format!("{label} args"))?;
            }
            let source = field_object(data, "source", label)?;
            keys(source, &["kind"], &[], &format!("{label} source"))?;
            literal(
                field(source, "kind", &format!("{label} source"))?,
                &["user"],
                &format!("{label} source kind"),
            )?;
        }
        "compaction/start" | "compaction/end" => {
            non_empty(
                field(data, "compactionId", label)?,
                &format!("{label} compactionId"),
            )?;
            if let Some(command) = data.get("sourceCommandId") {
                non_empty(command, &format!("{label} sourceCommandId"))?;
            }
            match field(data, "turn", label)? {
                Value::Null => {}
                value => {
                    count(value, &format!("{label} turn"))?;
                }
            }
            if let Some(error) = data.get("error") {
                string(error, &format!("{label} error"))?;
            }
        }
        "compaction/prune" => assert_shadowed(data, seq, label)?,
        "compaction/summary" => {
            if data.get("llmStreamCall").and_then(Value::as_bool) == Some(true)
                && !data.contains_key("rawOutput")
            {
                return Err(fail(format!("{label} llmStreamCall requires rawOutput")));
            }
            non_empty(
                field(data, "compactionId", label)?,
                &format!("{label} compactionId"),
            )?;
            if let Some(command) = data.get("sourceCommandId") {
                non_empty(command, &format!("{label} sourceCommandId"))?;
            }
            assert_content(
                field(data, "summary", label)?,
                &format!("{label} summary"),
                version,
            )?;
            assert_shadowed(data, seq, label)?;
            non_empty(
                field(data, "provider", label)?,
                &format!("{label} provider"),
            )?;
            non_empty(field(data, "model", label)?, &format!("{label} model"))?;
            if let Some(max) = data.get("maxTokens") {
                count(max, &format!("{label} maxTokens"))?;
            }
            if let Some(usage) = data.get("usage") {
                assert_usage(usage, &format!("{label} usage"))?;
            }
            if let Some(raw) = data.get("rawOutput") {
                assert_content(raw, &format!("{label} rawOutput"), version)?;
            }
            if let Some(stream) = data.get("llmStreamCall") {
                if stream.as_bool() != Some(true) {
                    return Err(fail(format!("{label} llmStreamCall must be true")));
                }
            }
        }
        "feedback/record" => {
            non_empty(field(data, "text", label)?, &format!("{label} text"))?;
        }
        "goal/change" => assert_goal_change(data, label)?,
        "hook/invoked" => {
            count(field(data, "turn", label)?, &format!("{label} turn"))?;
            non_empty(field(data, "point", label)?, &format!("{label} point"))?;
            literal(
                field(data, "dialect", label)?,
                &["claude-code", "codex"],
                &format!("{label} dialect"),
            )?;
            if let Some(matcher) = data.get("matcher") {
                string(matcher, &format!("{label} matcher"))?;
            }
            non_empty(
                field(data, "handlerId", label)?,
                &format!("{label} handlerId"),
            )?;
        }
        "hook/result" => {
            count(field(data, "turn", label)?, &format!("{label} turn"))?;
            non_empty(field(data, "point", label)?, &format!("{label} point"))?;
            non_empty(
                field(data, "handlerId", label)?,
                &format!("{label} handlerId"),
            )?;
            non_empty(
                field(data, "decision", label)?,
                &format!("{label} decision"),
            )?;
            if let Some(code) = data.get("exitCode") {
                count(code, &format!("{label} exitCode"))?;
            }
            if let Some(summary) = data.get("stderrSummary") {
                string(summary, &format!("{label} stderrSummary"))?;
            }
            if finite(
                field(data, "durationMs", label)?,
                &format!("{label} durationMs"),
            )? < 0.0
            {
                return Err(fail(format!("{label} durationMs must be non-negative")));
            }
        }
        "llm/retry" => assert_retry(data, label)?,
        "llm/retry-started" => {
            non_empty(field(data, "retryId", label)?, &format!("{label} retryId"))?;
            coordinate_pair(data, label)?;
            positive(field(data, "retry", label)?, &format!("{label} retry"))?;
        }
        "model/selection" => {
            non_empty(
                field(data, "provider", label)?,
                &format!("{label} provider"),
            )?;
            non_empty(field(data, "model", label)?, &format!("{label} model"))?;
            if let Some(effort) = data.get("reasoningEffort") {
                non_empty(effort, &format!("{label} reasoningEffort"))?;
            }
        }
        "permission/preset" => {
            non_empty(field(data, "preset", label)?, &format!("{label} preset"))?;
        }
        "plan/mode" => {
            boolean(field(data, "active", label)?, &format!("{label} active"))?;
        }
        "request/context" => {
            non_empty(
                field(data, "provider", label)?,
                &format!("{label} provider"),
            )?;
            non_empty(field(data, "model", label)?, &format!("{label} model"))?;
            if let Some(window) = data.get("contextWindow") {
                positive(window, &format!("{label} contextWindow"))?;
            }
        }
        "request/header" => assert_request_header(data, label)?,
        "sandbox/mode" => {
            literal(
                field(data, "mode", label)?,
                &["read-only", "workspace-write", "danger-full-access"],
                &format!("{label} mode"),
            )?;
            if let Some(source) = data.get("source") {
                literal(source, &["delegation"], &format!("{label} source"))?;
            }
        }
        "schedule/change" => assert_schedule_change(data, label)?,
        "session-log-deepseek/delivery-accepted" => {
            let accepted = match data.get("sessionFormatVersion") {
                Some(version) => count(version, &format!("{label} sessionFormatVersion"))?,
                None => 0,
            };
            if accepted == u64::from(version) {
                non_empty(
                    field(data, "sessionId", label)?,
                    &format!("{label} sessionId"),
                )?;
                earlier(
                    field(data, "throughSeq", label)?,
                    seq,
                    &format!("{label} throughSeq"),
                )?;
            }
        }
        "session/end-seed" => {
            if let Some(inherited) = data.get("inherited") {
                if inherited.as_bool() != Some(true) {
                    return Err(fail(format!("{label} inherited must be true")));
                }
            }
        }
        "session/title" => {
            non_empty(field(data, "title", label)?, &format!("{label} title"))?;
            assert_seq_array(data, "messageSeqs", seq, label, false)?;
            let source = field_object(data, "source", label)?;
            if source.get("kind").and_then(Value::as_str) == Some("provider") {
                keys(
                    source,
                    &["kind", "provider"],
                    &["model"],
                    &format!("{label} source"),
                )?;
                non_empty(
                    field(source, "provider", label)?,
                    &format!("{label} provider"),
                )?;
            } else {
                keys(source, &["kind"], &[], &format!("{label} source"))?;
                literal(
                    field(source, "kind", label)?,
                    &["fallback", "user"],
                    &format!("{label} source kind"),
                )?;
            }
        }
        "session/title-llm-request" => {
            non_empty(
                field(data, "titleProvider", label)?,
                &format!("{label} titleProvider"),
            )?;
            assert_seq_array(data, "messageSeqs", seq, label, true)?;
            let route = field_object(data, "route", label)?;
            keys(
                route,
                &["provider", "model"],
                &[],
                &format!("{label} route"),
            )?;
            non_empty(
                field(route, "provider", label)?,
                &format!("{label} route provider"),
            )?;
            non_empty(
                field(route, "model", label)?,
                &format!("{label} route model"),
            )?;
            string(field(data, "system", label)?, &format!("{label} system"))?;
            for (index, message) in field_array(data, "messages", label)?.iter().enumerate() {
                assert_message(
                    message,
                    &format!("{label} messages[{index}]"),
                    version,
                    None,
                )?;
            }
            positive(
                field(data, "maxTokens", label)?,
                &format!("{label} maxTokens"),
            )?;
        }
        "step/end" | "step/start" => {
            coordinate_pair(data, label)?;
        }
        "subagent/descriptor" => assert_subagent_descriptor(data, label)?,
        "subagent/model-selection-policy" => {
            let routes = field_array(data, "allowedModels", label)?;
            if routes.is_empty() {
                return Err(fail(format!("{label} allowedModels must be non-empty")));
            }
            for (index, route) in routes.iter().enumerate() {
                let route = object(route, &format!("{label} allowedModels[{index}]"))?;
                keys(
                    route,
                    &["provider", "model"],
                    &[],
                    &format!("{label} allowedModels[{index}]"),
                )?;
                non_empty(
                    field(route, "provider", label)?,
                    &format!("{label} allowedModels[{index}] provider"),
                )?;
                non_empty(
                    field(route, "model", label)?,
                    &format!("{label} allowedModels[{index}] model"),
                )?;
            }
        }
        "team/member" | "team/message/delivered" | "team/message/queued" | "team/task" => {
            literal_number(
                field(data, "version", label)?,
                &[1],
                &format!("{label} version"),
            )?;
            non_empty(field(data, "teamId", label)?, &format!("{label} teamId"))?;
        }
        "todo/write" => {
            for (index, todo) in field_array(data, "todos", label)?.iter().enumerate() {
                let item = object(todo, &format!("{label} todos[{index}]"))?;
                keys(
                    item,
                    &["content", "status"],
                    &[],
                    &format!("{label} todos[{index}]"),
                )?;
                string(
                    field(item, "content", label)?,
                    &format!("{label} todos[{index}] content"),
                )?;
                literal(
                    field(item, "status", label)?,
                    &["pending", "in_progress", "completed"],
                    &format!("{label} todos[{index}] status"),
                )?;
            }
        }
        "tool-workflow/agent-end" => {
            non_empty(field(data, "runId", label)?, &format!("{label} runId"))?;
            positive(field(data, "seq", label)?, &format!("{label} seq"))?;
            literal(
                field(data, "outcome", label)?,
                &["completed", "failed", "cancelled"],
                &format!("{label} outcome"),
            )?;
        }
        "tool-workflow/agent-start" => {
            non_empty(field(data, "runId", label)?, &format!("{label} runId"))?;
            positive(field(data, "seq", label)?, &format!("{label} seq"))?;
            string(field(data, "label", label)?, &format!("{label} label"))?;
            if let Some(phase) = data.get("phase") {
                string(phase, &format!("{label} phase"))?;
            }
            non_empty(field(data, "childId", label)?, &format!("{label} childId"))?;
        }
        "tool-workflow/run-end" => {
            non_empty(field(data, "runId", label)?, &format!("{label} runId"))?;
            literal(
                field(data, "stopReason", label)?,
                &["completed", "cancelled", "error"],
                &format!("{label} stopReason"),
            )?;
        }
        "tool-workflow/run-start" => {
            non_empty(field(data, "runId", label)?, &format!("{label} runId"))?;
            non_empty(field(data, "name", label)?, &format!("{label} name"))?;
        }
        "tool/call" => {
            coordinate_pair(data, label)?;
            non_empty(field(data, "callId", label)?, &format!("{label} callId"))?;
            non_empty(field(data, "name", label)?, &format!("{label} name"))?;
            string(
                field(data, "arguments", label)?,
                &format!("{label} arguments"),
            )?;
        }
        "tool/code-dispatch"
        | "tool/code-dispatch-start"
        | "tool/ptc-dispatch"
        | "tool/ptc-dispatch-start" => {
            for key in ["rootCallId", "parentCallId", "subCallId", "name"] {
                non_empty(field(data, key, label)?, &format!("{label} {key}"))?;
            }
            if event_type.ends_with("dispatch") {
                boolean(field(data, "isError", label)?, &format!("{label} isError"))?;
                assert_content(
                    field(data, "content", label)?,
                    &format!("{label} content"),
                    version,
                )?;
            }
        }
        "tool/result" => {
            coordinate_pair(data, label)?;
            assert_message(
                field(data, "message", label)?,
                &format!("{label} message"),
                version,
                Some("tool"),
            )?;
            if let Some(error) = data.get("error") {
                let error = object(error, &format!("{label} error"))?;
                keys(error, &["name", "code"], &[], &format!("{label} error"))?;
                non_empty(field(error, "name", label)?, &format!("{label} error name"))?;
                non_empty(field(error, "code", label)?, &format!("{label} error code"))?;
            }
        }
        "turn/end" => {
            count(field(data, "turn", label)?, &format!("{label} turn"))?;
            assert_turn_end_reason(field(data, "reason", label)?, &format!("{label} reason"))?;
        }
        "turn/start" => {
            count(field(data, "turn", label)?, &format!("{label} turn"))?;
        }
        "user/message" => {
            assert_message(&Value::Object(data.clone()), label, version, Some("user"))?;
        }
        "system/message" => {
            positive(field(data, "turn", label)?, &format!("{label} turn"))?;
            positive(field(data, "step", label)?, &format!("{label} step"))?;
            let message = field_object(data, "message", label)?;
            keys(
                message,
                &["id", "role", "source", "content"],
                &[],
                &format!("{label} message"),
            )?;
            non_empty(field(message, "id", label)?, &format!("{label} message id"))?;
            if field(message, "role", label)?.as_str() != Some("system") {
                return Err(fail(format!("{label} message role must be system")));
            }
            let source = field_object(message, "source", label)?;
            if source.get("kind").and_then(Value::as_str) != Some("plugin") {
                return Err(fail(format!("{label} message requires plugin source")));
            }
            assert_content(
                field(message, "content", label)?,
                &format!("{label} message content"),
                version,
            )?;
        }
        "web/deepseek-search-llm-request" => {
            non_empty(
                field(data, "endpoint", label)?,
                &format!("{label} endpoint"),
            )?;
            non_empty(
                field(data, "apiVersion", label)?,
                &format!("{label} apiVersion"),
            )?;
            object(field(data, "body", label)?, &format!("{label} body"))?;
        }
        other => {
            return Err(fail(format!(
                "released payload validator is missing event `{other}`"
            )))
        }
    }
    Ok(())
}

fn coordinate_pair(data: &JsonObject, label: &str) -> Result<(), SessionError> {
    count(field(data, "turn", label)?, &format!("{label} turn"))?;
    count(field(data, "step", label)?, &format!("{label} step"))?;
    Ok(())
}

fn earlier(value: &Value, seq: u64, label: &str) -> Result<u64, SessionError> {
    let position = count(value, label)?;
    if position >= seq {
        return Err(fail(format!("{label} must identify an earlier event")));
    }
    Ok(position)
}

fn assert_seq_array(
    data: &JsonObject,
    key: &str,
    seq: u64,
    label: &str,
    require_non_empty: bool,
) -> Result<(), SessionError> {
    let values = field_array(data, key, label)?;
    if require_non_empty && values.is_empty() {
        return Err(fail(format!("{label} {key} must be non-empty")));
    }
    let mut seen: Vec<u64> = Vec::with_capacity(values.len());
    for value in values {
        let member = earlier(value, seq, &format!("{label} {key}"))?;
        if seen.contains(&member) {
            return Err(fail(format!("{label} {key} repeats seq {member}")));
        }
        seen.push(member);
    }
    Ok(())
}

fn assert_usage(value: &Value, label: &str) -> Result<(), SessionError> {
    let usage = object(value, label)?;
    keys(
        usage,
        &["inputTokens", "outputTokens"],
        &[
            "totalTokens",
            "cacheReadTokens",
            "cacheWriteTokens",
            "reasoningTokens",
        ],
        label,
    )?;
    for (key, member) in usage {
        count(member, &format!("{label} {key}"))?;
    }
    Ok(())
}

fn assert_retry(data: &JsonObject, label: &str) -> Result<(), SessionError> {
    non_empty(field(data, "retryId", label)?, &format!("{label} retryId"))?;
    coordinate_pair(data, label)?;
    non_empty(
        field(data, "provider", label)?,
        &format!("{label} provider"),
    )?;
    let mode = literal(
        field(data, "mode", label)?,
        &["normal", "always"],
        &format!("{label} mode"),
    )?;
    non_empty(
        field(data, "policyKey", label)?,
        &format!("{label} policyKey"),
    )?;
    let retry = positive(field(data, "retry", label)?, &format!("{label} retry"))?;
    if mode == "normal" {
        let max = positive(
            field(data, "maxRetries", label)?,
            &format!("{label} maxRetries"),
        )?;
        if retry > max {
            return Err(fail(format!("{label} retry exceeds maxRetries")));
        }
    } else if data.contains_key("maxRetries") {
        return Err(fail(format!("{label} always mode must omit maxRetries")));
    }
    let delay = finite(field(data, "delayMs", label)?, &format!("{label} delayMs"))?;
    if delay < 0.0 {
        return Err(fail(format!("{label} delayMs must be non-negative")));
    }
    if delay > 2_147_483_647.0 {
        return Err(fail(format!("{label} delayMs exceeds the timer range")));
    }
    assert_failure(field(data, "failure", label)?, &format!("{label} failure"))
}

fn assert_failure(value: &Value, label: &str) -> Result<(), SessionError> {
    let failure = object(value, label)?;
    keys(
        failure,
        &["message", "code"],
        &["status", "providerRetryAfterMs", "requestId"],
        label,
    )?;
    non_empty(
        field(failure, "message", label)?,
        &format!("{label} message"),
    )?;
    non_empty(field(failure, "code", label)?, &format!("{label} code"))?;
    if let Some(status) = failure.get("status") {
        let status = count(status, &format!("{label} status"))?;
        if !(100..=599).contains(&status) {
            return Err(fail(format!("{label} status must be 100 through 599")));
        }
    }
    if let Some(after) = failure.get("providerRetryAfterMs") {
        if finite(after, &format!("{label} providerRetryAfterMs"))? <= 0.0 {
            return Err(fail(format!(
                "{label} providerRetryAfterMs must be positive"
            )));
        }
    }
    if let Some(request) = failure.get("requestId") {
        non_empty(request, &format!("{label} requestId"))?;
    }
    Ok(())
}

fn assert_turn_end_reason(value: &Value, label: &str) -> Result<(), SessionError> {
    let reason = object(value, label)?;
    let kind = string(field(reason, "kind", label)?, &format!("{label} kind"))?;
    match kind {
        "completed" | "blocked" | "max-tokens" | "interrupted" => {
            keys(reason, &["kind"], &[], label)?;
        }
        "aborted" => {
            keys(reason, &["kind", "reason"], &[], label)?;
            let cause = object(
                field(reason, "reason", label)?,
                &format!("{label} abort cause"),
            )?;
            if cause.get("kind").and_then(Value::as_str) == Some("hook") {
                keys(
                    cause,
                    &["kind", "reason"],
                    &[],
                    &format!("{label} abort cause"),
                )?;
                string(
                    field(cause, "reason", label)?,
                    &format!("{label} abort reason"),
                )?;
            } else {
                keys(cause, &["kind"], &[], &format!("{label} abort cause"))?;
                literal(
                    field(cause, "kind", label)?,
                    &["user", "parent", "disposed", "legacy"],
                    &format!("{label} abort kind"),
                )?;
            }
        }
        "error" => {
            keys(reason, &["kind", "error"], &[], label)?;
            assert_failure(field(reason, "error", label)?, &format!("{label} error"))?;
        }
        _ => {}
    }
    Ok(())
}

fn assert_request_header(data: &JsonObject, label: &str) -> Result<(), SessionError> {
    literal(
        field(data, "reason", label)?,
        &["initial", "resume", "change", "series"],
        &format!("{label} reason"),
    )?;
    if let Some(starts) = data.get("startsSeries") {
        if starts.as_bool() != Some(true) {
            return Err(fail(format!("{label} startsSeries must be true")));
        }
    }
    let header = field_object(data, "header", label)?;
    keys(
        header,
        &["config"],
        &["adapterDefaults", "system", "tools"],
        &format!("{label} header"),
    )?;
    let config = object(
        field(header, "config", label)?,
        &format!("{label} header config"),
    )?;
    keys(
        config,
        &["provider", "model"],
        &["reasoningEffort", "temperature", "maxTokens", "stop"],
        &format!("{label} header config"),
    )?;
    non_empty(
        field(config, "provider", label)?,
        &format!("{label} header provider"),
    )?;
    non_empty(
        field(config, "model", label)?,
        &format!("{label} header model"),
    )?;
    if let Some(effort) = config.get("reasoningEffort") {
        non_empty(effort, &format!("{label} reasoningEffort"))?;
    }
    if let Some(temperature) = config.get("temperature") {
        finite(temperature, &format!("{label} temperature"))?;
    }
    if let Some(max) = config.get("maxTokens") {
        positive(max, &format!("{label} maxTokens"))?;
    }
    if let Some(stop) = config.get("stop") {
        for (index, member) in array(stop, &format!("{label} stop"))?.iter().enumerate() {
            string(member, &format!("{label} stop[{index}]"))?;
        }
    }
    if let Some(defaults) = header.get("adapterDefaults") {
        let defaults = object(defaults, &format!("{label} adapterDefaults"))?;
        keys(
            defaults,
            &[],
            &["reasoningEffort", "maxTokens"],
            &format!("{label} adapterDefaults"),
        )?;
        for key in defaults.keys() {
            if defaults[key].as_bool() != Some(true) {
                return Err(fail(format!("{label} adapterDefaults {key} must be true")));
            }
            if !config.contains_key(key) {
                return Err(fail(format!(
                    "{label} adapter default {key} lacks config value"
                )));
            }
        }
    }
    if let Some(system) = header.get("system") {
        string(system, &format!("{label} header system"))?;
    }
    if let Some(tools) = header.get("tools") {
        for (index, tool) in array(tools, &format!("{label} tools"))?.iter().enumerate() {
            let tool = object(tool, &format!("{label} tools[{index}]"))?;
            keys(
                tool,
                &["name", "description", "parameters"],
                &["deferLoading"],
                &format!("{label} tools[{index}]"),
            )?;
            non_empty(
                tool.get("name").unwrap_or(&Value::Null),
                &format!("{label} tools[{index}] name"),
            )?;
            string(
                field(tool, "description", label)?,
                &format!("{label} tools[{index}] description"),
            )?;
            object(
                field(tool, "parameters", label)?,
                &format!("{label} tools[{index}] parameters"),
            )?;
        }
    }
    Ok(())
}

fn assert_goal_change(data: &JsonObject, label: &str) -> Result<(), SessionError> {
    literal(
        field(data, "kind", label)?,
        &["goal/change"],
        &format!("{label} kind"),
    )?;
    literal_number(
        field(data, "version", label)?,
        &[1],
        &format!("{label} version"),
    )?;
    if data.get("operation").and_then(Value::as_str) == Some("clear") {
        keys(
            data,
            &["kind", "version", "operation", "cleared", "clearedAt"],
            &[],
            &format!("{label} data"),
        )?;
        let cleared = field_object(data, "cleared", label)?;
        keys(
            cleared,
            &["id", "revision"],
            &[],
            &format!("{label} cleared"),
        )?;
        non_empty(field(cleared, "id", label)?, &format!("{label} cleared id"))?;
        positive(
            field(cleared, "revision", label)?,
            &format!("{label} cleared revision"),
        )?;
        count(
            field(data, "clearedAt", label)?,
            &format!("{label} clearedAt"),
        )?;
        return Ok(());
    }
    literal(
        field(data, "operation", label)?,
        &["create", "edit", "pause", "resume", "complete", "block"],
        &format!("{label} operation"),
    )?;
    let goal = field_object(data, "goal", label)?;
    keys(
        goal,
        &["id", "revision", "objective", "phase", "maxGoalRounds"],
        &["blockedReason"],
        &format!("{label} goal"),
    )?;
    non_empty(field(goal, "id", label)?, &format!("{label} goal id"))?;
    positive(
        field(goal, "revision", label)?,
        &format!("{label} goal revision"),
    )?;
    non_empty(
        field(goal, "objective", label)?,
        &format!("{label} goal objective"),
    )?;
    literal(
        field(goal, "phase", label)?,
        &["active", "paused", "blocked", "complete"],
        &format!("{label} goal phase"),
    )?;
    positive(
        field(goal, "maxGoalRounds", label)?,
        &format!("{label} goal maxGoalRounds"),
    )?;
    for key in ["roundsStarted", "createdAt", "updatedAt"] {
        count(field(data, key, label)?, &format!("{label} {key}"))?;
    }
    Ok(())
}

fn assert_schedule_change(data: &JsonObject, label: &str) -> Result<(), SessionError> {
    literal_number(
        field(data, "version", label)?,
        &[1],
        &format!("{label} version"),
    )?;
    let operation = literal(
        field(data, "operation", label)?,
        &["create", "delete", "dispatch"],
        &format!("{label} operation"),
    )?;
    if operation == "create" {
        keys(
            data,
            &["version", "operation", "schedule"],
            &[],
            &format!("{label} data"),
        )?;
        let schedule = field_object(data, "schedule", label)?;
        let kind = literal(
            field(schedule, "kind", label)?,
            &["after", "at", "every"],
            &format!("{label} schedule kind"),
        )?;
        let allowed: &[&str] = match kind {
            "after" => &["id", "kind", "prompt", "afterSeconds", "scheduledAt"],
            "every" => &["id", "kind", "prompt", "everySeconds", "scheduledAt"],
            _ => &["id", "kind", "prompt", "scheduledAt"],
        };
        keys(schedule, allowed, &[], &format!("{label} schedule"))?;
        non_empty(
            field(schedule, "id", label)?,
            &format!("{label} schedule id"),
        )?;
        non_empty(
            field(schedule, "prompt", label)?,
            &format!("{label} schedule prompt"),
        )?;
        if kind == "after" {
            positive(
                field(schedule, "afterSeconds", label)?,
                &format!("{label} afterSeconds"),
            )?;
        }
        if kind == "every" {
            let seconds = positive(
                field(schedule, "everySeconds", label)?,
                &format!("{label} everySeconds"),
            )?;
            if seconds < 300 {
                return Err(fail(format!("{label} everySeconds must be at least 300")));
            }
        }
        instant(
            field(schedule, "scheduledAt", label)?,
            &format!("{label} scheduledAt"),
        )?;
        return Ok(());
    }
    keys(
        data,
        &["version", "operation", "id"],
        if operation == "dispatch" {
            &["acceptedAt"]
        } else {
            &[]
        },
        &format!("{label} data"),
    )?;
    let id = non_empty(field(data, "id", label)?, &format!("{label} id"))?;
    if id.trim() != id {
        return Err(fail(format!(
            "{label} id must not have surrounding whitespace"
        )));
    }
    if let Some(accepted) = data.get("acceptedAt") {
        instant(accepted, &format!("{label} acceptedAt"))?;
    }
    Ok(())
}

/// Require a canonical UTC instant (`YYYY-MM-DDTHH:MM:SS.sssZ`).
fn instant(value: &Value, label: &str) -> Result<(), SessionError> {
    let text = string(value, label)?;
    let bytes = text.as_bytes();
    let canonical = bytes.len() == 24
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[10] == b'T'
        && bytes[13] == b':'
        && bytes[16] == b':'
        && bytes[19] == b'.'
        && bytes[23] == b'Z'
        && bytes[..4].iter().all(u8::is_ascii_digit)
        && bytes[5..7].iter().all(u8::is_ascii_digit)
        && bytes[8..10].iter().all(u8::is_ascii_digit)
        && bytes[11..13].iter().all(u8::is_ascii_digit)
        && bytes[14..16].iter().all(u8::is_ascii_digit)
        && bytes[17..19].iter().all(u8::is_ascii_digit)
        && bytes[20..23].iter().all(u8::is_ascii_digit)
        && (1..=12).contains(&text[5..7].parse::<u32>().unwrap_or(0))
        && (1..=31).contains(&text[8..10].parse::<u32>().unwrap_or(0))
        && text[11..13].parse::<u32>().unwrap_or(99) <= 23
        && text[14..16].parse::<u32>().unwrap_or(99) <= 59
        && text[17..19].parse::<u32>().unwrap_or(99) <= 59;
    if !canonical {
        return Err(fail(format!("{label} must be a canonical UTC instant")));
    }
    Ok(())
}

fn assert_shadowed(data: &JsonObject, seq: u64, label: &str) -> Result<(), SessionError> {
    let range = field_object(data, "shadowedRange", label)?;
    keys(
        range,
        &["start", "end"],
        &[],
        &format!("{label} shadowedRange"),
    )?;
    let start = earlier(
        field(range, "start", label)?,
        seq,
        &format!("{label} shadowedRange start"),
    )?;
    let end = earlier(
        field(range, "end", label)?,
        seq,
        &format!("{label} shadowedRange end"),
    )?;
    let seqs = field_array(data, "shadowedSeqs", label)?;
    if seqs.is_empty() {
        return Err(fail(format!("{label} shadowedSeqs must be non-empty")));
    }
    let mut seen = Vec::with_capacity(seqs.len());
    for member in seqs {
        let member = earlier(member, seq, &format!("{label} shadowedSeqs"))?;
        if seen.contains(&member) {
            return Err(fail(format!("{label} shadowedSeqs repeats seq {member}")));
        }
        seen.push(member);
    }
    if seen.first() != Some(&start) || seen.last() != Some(&end) {
        return Err(fail(format!(
            "{label} shadowedRange must match shadowedSeqs endpoints"
        )));
    }
    count(
        field(data, "shadowedTokenCount", label)?,
        &format!("{label} shadowedTokenCount"),
    )?;
    Ok(())
}

fn assert_subagent_descriptor(data: &JsonObject, label: &str) -> Result<(), SessionError> {
    literal_number(
        field(data, "version", label)?,
        &[3],
        &format!("{label} version"),
    )?;
    non_empty(
        field(data, "provider", label)?,
        &format!("{label} provider"),
    )?;
    let mode = literal(
        field(data, "mode", label)?,
        &["one-shot", "continuable"],
        &format!("{label} mode"),
    )?;
    if mode == "one-shot" {
        if let Some(name) = data.get("label") {
            string(name, &format!("{label} label"))?;
        }
        return Ok(());
    }
    non_empty(field(data, "label", label)?, &format!("{label} label"))?;
    for key in [
        "agentProvider",
        "agentModel",
        "agentReasoningEffort",
        "persona",
    ] {
        if let Some(value) = data.get(key) {
            non_empty(value, &format!("{label} {key}"))?;
        }
    }
    if data.contains_key("agentProvider") != data.contains_key("agentModel") {
        return Err(fail(format!(
            "{label} agentProvider and agentModel must be paired"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Messages, content, and stream records
// ---------------------------------------------------------------------------

/// Validate one message payload.
pub(crate) fn assert_message(
    value: &Value,
    label: &str,
    version: u32,
    expected_role: Option<&str>,
) -> Result<(), SessionError> {
    let message = object(value, label)?;
    keys(message, &["id", "role", "content", "source"], &[], label)?;
    non_empty(field(message, "id", label)?, &format!("{label} id"))?;
    let role = string(field(message, "role", label)?, &format!("{label} role"))?;
    // A released `tool/result` carries a user-role message with one
    // `tool-result` wrapper; only the v4 lift gives it the tool role.
    let expected = match expected_role {
        Some("tool") => Some("user"),
        other => other,
    };
    match expected {
        Some(expected) if expected != role => {
            return Err(fail(format!("{label} role must be {expected}")));
        }
        Some(_) => {}
        None => {
            literal(
                field(message, "role", label)?,
                &["system", "user", "assistant"],
                &format!("{label} role"),
            )?;
        }
    }
    assert_content(
        field(message, "content", label)?,
        &format!("{label} content"),
        version,
    )?;
    let source = field_object(message, "source", label)?;
    if version < 2
        && expected_role == Some("user")
        && source.get("kind").and_then(Value::as_str) == Some("goal")
        && source.contains_key("change")
    {
        legacy_goal_message(message, source, label)?;
    } else {
        assert_source(source, &format!("{label} source"), version, expected_role)?;
    }
    if expected_role == Some("tool") {
        let content = field_array(message, "content", label)?;
        let source = field_object(message, "source", label)?;
        let call_id = field_str(source, "callId", &format!("{label} source"))?;
        let wrapper = match content {
            [only] => object(only, &format!("{label} tool result"))?,
            _ => {
                return Err(fail(format!(
                    "{label} must contain exactly one tool-result block"
                )))
            }
        };
        if wrapper.get("type").and_then(Value::as_str) != Some("tool-result")
            || wrapper.get("toolCallId").and_then(Value::as_str) != Some(call_id)
        {
            return Err(fail(format!(
                "{label} must contain exactly one tool-result block"
            )));
        }
    }
    Ok(())
}

/// Validate a pre-v2 goal message whose source still carried its change record.
///
/// The v1 reader accepted one rendered `<goal_state>` payload, so the edge can
/// split the change out and keep the message lossless.
fn legacy_goal_message(
    message: &JsonObject,
    source: &JsonObject,
    label: &str,
) -> Result<(), SessionError> {
    keys(
        source,
        &["kind", "goalId", "revision", "round", "change"],
        &[],
        &format!("{label} source"),
    )?;
    non_empty(field(source, "goalId", label)?, &format!("{label} goalId"))?;
    let revision = positive(
        field(source, "revision", label)?,
        &format!("{label} revision"),
    )?;
    if count(field(source, "round", label)?, &format!("{label} round"))? != 0 {
        return Err(fail(format!("{label} legacy goal source round must be 0")));
    }
    let change = field_object(source, "change", label)?;
    assert_goal_change(change, &format!("{label} source change"))?;
    let reference = match change.get("operation").and_then(Value::as_str) {
        Some("clear") => field_object(change, "cleared", label)?,
        _ => field_object(change, "goal", label)?,
    };
    let reference_id = string(
        field(reference, "id", label)?,
        &format!("{label} change ref id"),
    )?;
    let reference_revision = positive(
        field(reference, "revision", label)?,
        &format!("{label} change ref revision"),
    )?;
    if source.get("goalId").and_then(Value::as_str) != Some(reference_id)
        || revision != reference_revision
    {
        return Err(fail(format!(
            "{label} legacy goal source does not match its change"
        )));
    }
    let payload = if change.get("operation").and_then(Value::as_str) == Some("clear") {
        json!({
            "cleared": change.get("cleared").cloned().unwrap_or(Value::Null),
            "clearedAt": change.get("clearedAt").cloned().unwrap_or(Value::Null),
        })
    } else {
        json!({
            "goal": change.get("goal").cloned().unwrap_or(Value::Null),
            "roundsStarted": change.get("roundsStarted").cloned().unwrap_or(Value::Null),
            "createdAt": change.get("createdAt").cloned().unwrap_or(Value::Null),
            "updatedAt": change.get("updatedAt").cloned().unwrap_or(Value::Null),
        })
    };
    let text = format!("<goal_state>{payload}</goal_state>");
    let expected = json!([{ "type": "text", "text": text }]);
    if message.get("content") != Some(&expected) {
        return Err(fail(format!(
            "{label} legacy goal content does not match its change"
        )));
    }
    Ok(())
}

/// Validate one message source.
pub(crate) fn assert_source(
    source: &JsonObject,
    label: &str,
    version: u32,
    expected_role: Option<&str>,
) -> Result<(), SessionError> {
    let kind = non_empty(field(source, "kind", label)?, &format!("{label} kind"))?;
    if expected_role == Some("assistant") && kind != "model" {
        return Err(fail(format!("{label} must be model source")));
    }
    if expected_role == Some("tool") && kind != "tool" {
        return Err(fail(format!("{label} must be tool source")));
    }
    match kind {
        "user" => {
            keys(source, &["kind"], &["rpcId", "clientTimeZone"], label)?;
            for key in ["rpcId", "clientTimeZone"] {
                if let Some(value) = source.get(key) {
                    non_empty(value, &format!("{label} {key}"))?;
                }
            }
        }
        "plugin" => {
            let plugin = non_empty(field(source, "plugin", label)?, &format!("{label} plugin"))?;
            let optional: &[&str] = if plugin == "compact" {
                &[
                    "form",
                    "sections",
                    "summary",
                    "compactionId",
                    "sourceCommandId",
                ]
            } else {
                &["form", "sections", "summary"]
            };
            keys(source, &["kind", "plugin"], optional, label)?;
            if plugin == "compact" {
                non_empty(
                    field(source, "compactionId", label)?,
                    &format!("{label} compactionId"),
                )?;
                if let Some(command) = source.get("sourceCommandId") {
                    non_empty(command, &format!("{label} sourceCommandId"))?;
                }
            }
            if let Some(form) = source.get("form") {
                let form = literal(
                    form,
                    &[
                        "instructions",
                        "catalog",
                        "snapshot",
                        "notice",
                        "relay",
                        "recall",
                    ],
                    &format!("{label} form"),
                )?;
                if form == "snapshot" {
                    let sections = field_array(source, "sections", label)?;
                    for (index, section) in sections.iter().enumerate() {
                        let section = object(section, &format!("{label} sections[{index}]"))?;
                        keys(
                            section,
                            &["name", "text"],
                            &[],
                            &format!("{label} sections[{index}]"),
                        )?;
                        non_empty(
                            field(section, "name", label)?,
                            &format!("{label} sections[{index}] name"),
                        )?;
                        string(
                            field(section, "text", label)?,
                            &format!("{label} sections[{index}] text"),
                        )?;
                    }
                } else if source.contains_key("sections") {
                    return Err(fail(format!("{label} sections require snapshot form")));
                }
                if form == "notice" {
                    string(
                        field(source, "summary", label)?,
                        &format!("{label} summary"),
                    )?;
                } else if source.contains_key("summary") {
                    return Err(fail(format!("{label} summary requires notice form")));
                }
            }
        }
        "model" => {
            keys(
                source,
                &["kind", "provider", "model"],
                &["replayState"],
                label,
            )?;
            non_empty(
                field(source, "provider", label)?,
                &format!("{label} provider"),
            )?;
            non_empty(field(source, "model", label)?, &format!("{label} model"))?;
        }
        "tool" => {
            keys(source, &["kind", "callId"], &[], label)?;
            non_empty(field(source, "callId", label)?, &format!("{label} callId"))?;
        }
        "goal" => {
            keys(source, &["kind", "goalId", "revision", "round"], &[], label)?;
            non_empty(field(source, "goalId", label)?, &format!("{label} goalId"))?;
            positive(
                field(source, "revision", label)?,
                &format!("{label} revision"),
            )?;
            positive(field(source, "round", label)?, &format!("{label} round"))?;
        }
        "skill-invocation" => {
            keys(source, &["kind", "name", "form"], &[], label)?;
            non_empty(field(source, "name", label)?, &format!("{label} name"))?;
            literal(
                field(source, "form", label)?,
                &["instructions"],
                &format!("{label} form"),
            )?;
        }
        "coordinator" | "subagent-report" => {
            keys(source, &["kind", "form", "senderSessionId"], &[], label)?;
            literal(
                field(source, "form", label)?,
                &["relay"],
                &format!("{label} form"),
            )?;
            non_empty(
                field(source, "senderSessionId", label)?,
                &format!("{label} senderSessionId"),
            )?;
        }
        "subagent-settled" => {
            keys(
                source,
                &["kind", "form", "summary", "senderSessionId"],
                &[],
                label,
            )?;
            literal(
                field(source, "form", label)?,
                &["notice"],
                &format!("{label} form"),
            )?;
            string(
                field(source, "summary", label)?,
                &format!("{label} summary"),
            )?;
            non_empty(
                field(source, "senderSessionId", label)?,
                &format!("{label} senderSessionId"),
            )?;
        }
        "team-message" => {
            keys(
                source,
                &["kind", "teamId", "messageId", "senderId", "senderName"],
                &[],
                label,
            )?;
            for key in ["teamId", "messageId", "senderId"] {
                non_empty(field(source, key, label)?, &format!("{label} {key}"))?;
            }
            string(
                field(source, "senderName", label)?,
                &format!("{label} senderName"),
            )?;
        }
        // `agent-instructions`, `session-reference`, `skill-catalog`, `webhook`,
        // `agent-message`, and producer extensions carry nested structures whose
        // exact shape is validated by their owner; the migration only needs the
        // discriminant. See the crate docs for the coverage table.
        _ => {}
    }
    let _ = version;
    Ok(())
}

/// Validate one content array.
pub(crate) fn assert_content(value: &Value, label: &str, version: u32) -> Result<(), SessionError> {
    for (index, block) in array(value, label)?.iter().enumerate() {
        assert_block(block, &format!("{label}[{index}]"), version)?;
    }
    Ok(())
}

/// Validate one content block.
pub(crate) fn assert_block(value: &Value, label: &str, version: u32) -> Result<(), SessionError> {
    let block = object(value, label)?;
    let kind = non_empty(field(block, "type", label)?, &format!("{label} type"))?;
    match kind {
        "text" | "reasoning" => {
            keys(block, &["type", "text"], &[], label)?;
            string(field(block, "text", label)?, &format!("{label} text"))?;
        }
        "image" => {
            keys(block, &["type", "attachment"], &["offloaded"], label)?;
            let attachment = field_object(block, "attachment", label)?;
            keys(
                attachment,
                &["attachmentId", "mediaType", "bytes", "width", "height"],
                &["name", "originalDimensions"],
                &format!("{label} attachment"),
            )?;
            non_empty(
                field(attachment, "attachmentId", label)?,
                &format!("{label} attachmentId"),
            )?;
            literal(
                field(attachment, "mediaType", label)?,
                &["image/png", "image/jpeg", "image/webp", "image/gif"],
                &format!("{label} mediaType"),
            )?;
            count(
                field(attachment, "bytes", label)?,
                &format!("{label} bytes"),
            )?;
            positive(
                field(attachment, "width", label)?,
                &format!("{label} width"),
            )?;
            positive(
                field(attachment, "height", label)?,
                &format!("{label} height"),
            )?;
            if let Some(name) = attachment.get("name") {
                string(name, &format!("{label} name"))?;
            }
            if let Some(dimensions) = attachment.get("originalDimensions") {
                let dimensions = object(dimensions, &format!("{label} originalDimensions"))?;
                keys(
                    dimensions,
                    &["width", "height"],
                    &[],
                    &format!("{label} originalDimensions"),
                )?;
                positive(
                    field(dimensions, "width", label)?,
                    &format!("{label} original width"),
                )?;
                positive(
                    field(dimensions, "height", label)?,
                    &format!("{label} original height"),
                )?;
            }
        }
        "tool-call" => {
            keys(block, &["type", "id", "name", "arguments"], &[], label)?;
            non_empty(field(block, "id", label)?, &format!("{label} id"))?;
            non_empty(field(block, "name", label)?, &format!("{label} name"))?;
            string(
                field(block, "arguments", label)?,
                &format!("{label} arguments"),
            )?;
        }
        "tool-result" => {
            keys(
                block,
                &["type", "toolCallId", "content"],
                &["isError"],
                label,
            )?;
            non_empty(
                field(block, "toolCallId", label)?,
                &format!("{label} toolCallId"),
            )?;
            assert_content(
                field(block, "content", label)?,
                &format!("{label} content"),
                version,
            )?;
            if let Some(is_error) = block.get("isError") {
                boolean(is_error, &format!("{label} isError"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Validate one raw stream chunk.
pub(crate) fn assert_stream_chunk(value: &Value, label: &str) -> Result<(), SessionError> {
    let chunk = object(value, label)?;
    let kind = string(field(chunk, "type", label)?, &format!("{label} type"))?;
    match kind {
        "block-start" => {
            keys(chunk, &["type", "index", "blockType"], &[], label)?;
            count(field(chunk, "index", label)?, &format!("{label} index"))?;
            non_empty(
                field(chunk, "blockType", label)?,
                &format!("{label} blockType"),
            )?;
        }
        "text-delta" | "reasoning-delta" => {
            keys(chunk, &["type", "index", "text"], &[], label)?;
            count(field(chunk, "index", label)?, &format!("{label} index"))?;
            string(field(chunk, "text", label)?, &format!("{label} text"))?;
        }
        "tool-call-delta" => {
            keys(
                chunk,
                &["type", "index", "id", "argumentsDelta"],
                &["name"],
                label,
            )?;
            count(field(chunk, "index", label)?, &format!("{label} index"))?;
            non_empty(field(chunk, "id", label)?, &format!("{label} id"))?;
            if let Some(name) = chunk.get("name") {
                string(name, &format!("{label} name"))?;
            }
            string(
                field(chunk, "argumentsDelta", label)?,
                &format!("{label} argumentsDelta"),
            )?;
        }
        "block-end" => {
            keys(chunk, &["type", "index", "block"], &[], label)?;
            count(field(chunk, "index", label)?, &format!("{label} index"))?;
            assert_block(field(chunk, "block", label)?, &format!("{label} block"), 1)?;
        }
        "usage" => {
            keys(chunk, &["type", "usage"], &[], label)?;
            assert_usage(field(chunk, "usage", label)?, &format!("{label} usage"))?;
        }
        "finish" => {
            keys(chunk, &["type", "reason"], &["replayState"], label)?;
            let reason = field_object(chunk, "reason", label)?;
            let kind = non_empty(
                field(reason, "kind", label)?,
                &format!("{label} reason kind"),
            )?;
            if kind == "aborted" || kind == "error" {
                keys(
                    reason,
                    &["kind", "failure"],
                    &[],
                    &format!("{label} reason"),
                )?;
                assert_failure(
                    field(reason, "failure", label)?,
                    &format!("{label} failure"),
                )?;
            } else if ["stop", "tool-calls", "max-tokens"].contains(&kind) {
                keys(reason, &["kind"], &[], &format!("{label} reason"))?;
            }
        }
        other => {
            return Err(fail(format!(
                "{label} has unknown stream chunk type `{other}`"
            )))
        }
    }
    Ok(())
}

/// Validate one compact assistant stream.
pub(crate) fn assert_stream(value: &Value, label: &str) -> Result<(), SessionError> {
    for (index, record) in array(value, label)?.iter().enumerate() {
        let path = format!("{label}[{index}]");
        let record = object(record, &path)?;
        let kind = string(field(record, "type", &path)?, &format!("{path} type"))?;
        match kind {
            "chunk" => {
                keys(record, &["type", "time", "chunk"], &[], &path)?;
                count(field(record, "time", &path)?, &format!("{path} time"))?;
                assert_stream_chunk(field(record, "chunk", &path)?, &format!("{path} chunk"))?;
            }
            "text-chunks" | "reasoning-chunks" => {
                keys(
                    record,
                    &["type", "time0", "index", "dt", "texts"],
                    &[],
                    &path,
                )?;
                assert_run(record, "texts", &path)?;
            }
            "tool-call-chunks" => {
                keys(
                    record,
                    &["type", "time0", "index", "dt", "id", "args"],
                    &["name"],
                    &path,
                )?;
                assert_run(record, "args", &path)?;
                non_empty(field(record, "id", &path)?, &format!("{path} id"))?;
                if let Some(name) = record.get("name") {
                    non_empty(name, &format!("{path} name"))?;
                }
            }
            other => {
                return Err(fail(format!(
                    "{path} has unsupported record type `{other}`"
                )))
            }
        }
    }
    Ok(())
}

fn assert_run(record: &JsonObject, members: &str, label: &str) -> Result<(), SessionError> {
    count(field(record, "time0", label)?, &format!("{label} time0"))?;
    count(field(record, "index", label)?, &format!("{label} index"))?;
    let values = field_array(record, members, label)?;
    if values.is_empty() {
        return Err(fail(format!("{label} {members} must be non-empty")));
    }
    for (index, value) in values.iter().enumerate() {
        string(value, &format!("{label} {members}[{index}]"))?;
    }
    let gaps = field_array(record, "dt", label)?;
    if gaps.len() + 1 != values.len() {
        return Err(fail(format!(
            "{label} dt length must be one less than its members"
        )));
    }
    for gap in gaps {
        count(gap, &format!("{label} dt member"))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reference remapping
// ---------------------------------------------------------------------------

/// Source-to-target event positions accumulated while one stage streams.
///
/// Indexed by the *source* position, because one source position may emit no
/// event at all (a chunk folded into an attempt) while another emits two.
#[derive(Default)]
pub(crate) struct Mapping {
    targets: Vec<Option<u64>>,
}

impl Mapping {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record the target position of one source event.
    pub(crate) fn record(&mut self, source: u64, target: u64) {
        let index = source as usize;
        if self.targets.len() <= index {
            self.targets.resize(index + 1, None);
        }
        self.targets[index] = Some(target);
    }

    /// Map one earlier source position.
    pub(crate) fn one(&self, value: &Value, label: &str) -> Result<u64, SessionError> {
        let source = count(value, label)?;
        self.targets
            .get(source as usize)
            .copied()
            .flatten()
            .ok_or_else(|| fail(format!("{label} targets a consumed or future event")))
    }

    /// Map a list of earlier source positions.
    pub(crate) fn list(&self, value: &Value, label: &str) -> Result<Vec<Value>, SessionError> {
        let values = array(value, label)?;
        values
            .iter()
            .map(|member| self.one(member, label).map(Value::from))
            .collect()
    }

    /// Map the endpoints of one `{start, end}` range.
    pub(crate) fn range(&self, value: &Value, label: &str) -> Result<Value, SessionError> {
        let range = object(value, label)?;
        let mut mapped = range.clone();
        for key in ["start", "end"] {
            let current = field(range, key, label)?.clone();
            mapped.insert(key.to_string(), Value::from(self.one(&current, label)?));
        }
        Ok(Value::Object(mapped))
    }
}

/// Rewrite the payload members that name earlier events of the same artifact.
///
/// Delivery watermarks, capture watermarks, workflow sequence numbers, stream
/// block indices, turn/step coordinates, and numeric tool JSON are *not*
/// session positions and stay untouched.
pub(crate) fn remap_payload_references(
    event_type: &str,
    data: &mut JsonObject,
    mapping: &Mapping,
) -> Result<(), SessionError> {
    match event_type {
        "command/done" => {
            if let Some(source) = data.get("sourceEventSeq").cloned() {
                let mapped = mapping.one(&source, "command/done sourceEventSeq")?;
                data.insert("sourceEventSeq".to_string(), Value::from(mapped));
            }
        }
        "compaction/summary" | "compaction/prune" => {
            if let Some(range) = data.get("shadowedRange").cloned() {
                let mapped = mapping.range(&range, &format!("{event_type} shadowedRange"))?;
                data.insert("shadowedRange".to_string(), mapped);
            }
            if let Some(seqs) = data.get("shadowedSeqs").cloned() {
                let mapped = mapping.list(&seqs, &format!("{event_type} shadowedSeqs"))?;
                data.insert("shadowedSeqs".to_string(), Value::Array(mapped));
            }
        }
        "session/title" | "session/title-llm-request" => {
            if let Some(seqs) = data.get("messageSeqs").cloned() {
                let mapped = mapping.list(&seqs, &format!("{event_type} messageSeqs"))?;
                data.insert("messageSeqs".to_string(), Value::Array(mapped));
            }
        }
        "image/offload" => {
            if let Some(targets) = data.get("targets").and_then(Value::as_array).cloned() {
                let mut mapped = Vec::with_capacity(targets.len());
                for target in targets {
                    let target = object(&target, "image/offload target")?.clone();
                    let mut rewritten = target.clone();
                    if let Some(seq) = target.get("seq").cloned() {
                        let mapped_seq = mapping.one(&seq, "image/offload target seq")?;
                        rewritten.insert("seq".to_string(), Value::from(mapped_seq));
                    }
                    mapped.push(Value::Object(rewritten));
                }
                data.insert("targets".to_string(), Value::Array(mapped));
            }
        }
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SHA-256, for the generated v2→v3 system-message identity
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Lowercase hexadecimal SHA-256 of `bytes`.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut state: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let mut message = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in message.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for (index, word) in chunk.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(SHA256_K[index])
                .wrapping_add(schedule[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }

    let mut output = String::with_capacity(64);
    for word in state {
        output.push_str(&format!("{word:08x}"));
    }
    output
}

/// Quote one string the way `JSON.stringify` does, for the migration identity.
pub(crate) fn json_quote(text: &str) -> String {
    let mut output = String::with_capacity(text.len() + 2);
    output.push('"');
    for character in text.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if (character as u32) < 0x20 => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

#[cfg(test)]
mod tests {
    use super::{json_quote, sha256_hex};

    #[test]
    fn sha256_matches_the_published_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn json_quote_escapes_like_javascript() {
        assert_eq!(json_quote("a\"b\\c\n"), r#""a\"b\\c\n""#);
        assert_eq!(json_quote("\u{1}"), r#""\u0001""#);
        assert_eq!(json_quote("é"), "\"é\"");
    }
}
