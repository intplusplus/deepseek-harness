//! The v3 → v4 edge: first-class tool-role messages, producer-owned sources,
//! namespaced extension tags, and synthesized historical child catalogs
//! (`origin/packages/session/session-format-v3-to-v4/src/migration.ts`).
//!
//! The edge also owns the one structural insertion left in the chain: a resume
//! whose next turn starts after a `next-turn` inbox splice owes the previous
//! turn an implicit `turn/end`, which shifts every later position and therefore
//! remaps payload references.

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::types::SessionError;

use super::support::{
    array_mut, assert_envelope, assert_payload, assert_source_seqs, boolean, count, fail, field,
    literal, non_empty, object, object_mut, remap_payload_references, v3_disposition, Mapping,
};
use super::{Envelope, Log};

/// First-party event names understood by the released v3 reader.
const RELEASED_V3_EVENT_TYPES: [&str; 58] = [
    "agent-preset/selected",
    "agent/inbox/spliced",
    "approval/asked",
    "approval/decided",
    "approval/policy",
    "assistant/attempt",
    "assistant/message",
    "command/done",
    "command/run",
    "compaction/end",
    "compaction/prune",
    "compaction/start",
    "compaction/summary",
    "deliverables/presented",
    "feedback/message-delete",
    "feedback/message-put",
    "feedback/record",
    "goal/change",
    "hook/invoked",
    "hook/result",
    "image/offload",
    "llm/retry",
    "llm/retry-started",
    "model/selection",
    "permission/preset",
    "plan/mode",
    "request/context",
    "request/header",
    "sandbox/mode",
    "schedule/change",
    "session-log-deepseek/delivery-accepted",
    "session/end-seed",
    "session/title",
    "session/title-llm-request",
    "step/end",
    "step/start",
    "subagent/catalog",
    "subagent/descriptor",
    "subagent/model-selection-policy",
    "system/message",
    "team/member",
    "team/message/delivered",
    "team/message/queued",
    "team/task",
    "todo/write",
    "tool-workflow/agent-end",
    "tool-workflow/agent-start",
    "tool-workflow/run-end",
    "tool-workflow/run-start",
    "tool/call",
    "tool/ptc-dispatch",
    "tool/ptc-dispatch-start",
    "tool/result",
    "turn/end",
    "turn/start",
    "user/message",
    "web/deepseek-search-llm-request",
    "workspace/changes",
];

/// Released v3 plugin names whose current producer kind differs.
const RENAMED_PRODUCERS: [(&str, &str); 5] = [
    ("compact", "compact-checkpoint"),
    ("tools-code-mode", "ptc-mode"),
    ("tools-ptc", "ptc-mode"),
    ("dsh-compaction-basic", "compact-basic"),
    ("@deepseek-ai/dsh-system-prompt", "runtime-context"),
];

/// First-party v3 plugin identities that keep their own name.
const SAME_NAME_PRODUCERS: [&str; 25] = [
    "agent-instructions",
    "session-reference",
    "team-message",
    "goal",
    "skill-invocation",
    "skill-catalog",
    "coordinator",
    "subagent-report",
    "subagent-settled",
    "webhook",
    "agent-message",
    "model-selection",
    "plan-mode",
    "time-context",
    "tmux-context",
    "user-approval",
    "repeat-tool-reminder",
    "tool-cordis",
    "cordis-host-runner",
    "tool-goal",
    "tool-jobs",
    "hooks-codex",
    "hooks-claude-code",
    "schedule",
    "dsh-session-title-llm",
];

/// One streaming v3 → v4 stage.
struct Stage {
    session_id: String,
    is_seeded: bool,
    candidates: Vec<Map<String, Value>>,
    mapping: Mapping,
    next_seq: u64,
    cut: Option<u64>,
    source_cut: Option<u64>,
    turn: Option<u64>,
    step_open: bool,
    next_turn_spliced: bool,
    time: u64,
    foreign_delivery: Option<u64>,
    catalogs: Vec<Value>,
    out: Vec<Envelope>,
}

/// Convert a released v3 session into v4.
///
/// `children` is the explicit historical child evidence retained by the caller;
/// an empty slice declares a parent without children. Reading the child
/// sessions themselves is storage work and deliberately stays outside this
/// crate — see the module documentation.
pub fn v3_to_v4(mut log: Log, children: &[Value]) -> Result<Log, SessionError> {
    if log.version()? != 3 {
        return Err(fail("expected format v3 header".to_string()));
    }
    let session_id = log
        .header
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let is_seeded = log
        .header
        .get("isSeeded")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let created_at = log
        .header
        .get("createdAt")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if let Some(header) = log.header.as_object_mut() {
        header.insert("version".to_string(), json!(4));
    }

    let mut stage = Stage {
        session_id,
        is_seeded,
        candidates: catalog_candidates(children)?,
        mapping: Mapping::new(),
        next_seq: 0,
        cut: if is_seeded { None } else { Some(0) },
        source_cut: if is_seeded { None } else { Some(0) },
        turn: None,
        step_open: false,
        next_turn_spliced: false,
        time: created_at,
        foreign_delivery: None,
        catalogs: Vec::new(),
        out: Vec::new(),
    };
    for (index, event) in log.events.into_iter().enumerate() {
        stage.push(event, index)?;
    }
    let cut = stage.finish(log.cut)?;
    log.events = stage.out;
    log.cut = cut;
    Ok(log)
}

impl Stage {
    fn push(&mut self, mut event: Envelope, index: usize) -> Result<(), SessionError> {
        assert_envelope(&event, index)?;
        let known = RELEASED_V3_EVENT_TYPES.contains(&event.r#type.as_str());
        if known {
            if event.r#type == "assistant/message" && event.source_event_seqs.is_some() {
                return Err(fail(format!(
                    "assistant/message {} embeds its stream and cannot carry sourceEventSeqs",
                    event.seq
                )));
            }
            assert_source_seqs(&event, false)?;
            if v3_disposition(&event.r#type).is_some() {
                assert_payload(&event.r#type, &event.data, event.seq, 3)?;
            }
        } else if event.ignorable != Some(true) {
            return Err(fail(format!(
                "format v3 contains unknown event type `{}` at seq {}",
                event.r#type, event.seq
            )));
        }

        let interrupted = self.observe_restart(&event);
        if let Some(turn) = interrupted {
            let seq = self.next_seq;
            self.next_seq += 1;
            let mut data = Map::new();
            data.insert("turn".to_string(), json!(turn));
            data.insert("reason".to_string(), json!({ "kind": "interrupted" }));
            self.out
                .push(Envelope::at("turn/end", seq, event.time, data));
        }
        let target = self.next_seq;
        self.next_seq += 1;
        self.time = event.time;

        if event.r#type == "session/end-seed"
            && event.data.get("inherited").and_then(Value::as_bool) == Some(true)
        {
            if !self.is_seeded {
                return Err(fail(
                    "unseeded format v3 Session contains an inherited end-seed marker".to_string(),
                ));
            }
            self.source_cut = Some(event.seq);
            self.cut = Some(target);
            self.catalogs.clear();
        } else if event.r#type == "subagent/catalog" {
            let fact = catalog_fact(&Value::Object(event.data.clone()), "subagent/catalog")?;
            self.catalogs.push(Value::Object(fact));
        }

        if event.r#type == "session-log-deepseek/delivery-accepted" {
            let claimed = event
                .data
                .get("sessionFormatVersion")
                .and_then(Value::as_u64);
            if claimed == Some(4) {
                return Err(fail(
                    "format v3 delivery marker claims target format v4".to_string(),
                ));
            }
            if claimed == Some(3)
                && event.data.get("sessionId").and_then(Value::as_str)
                    != Some(self.session_id.as_str())
            {
                self.foreign_delivery = Some(event.seq);
            }
        }

        if !known {
            // An unknown ignorable event keeps its unvalidated envelope
            // references; only its name and position change.
            let source = event.seq;
            event.r#type = format!("plugin:{}", event.r#type);
            event.seq = target;
            event.source_event_seqs = None;
            event.surface_op = None;
            event.physical = false;
            self.mapping.record(source, target);
            self.out.push(event);
            return Ok(());
        }

        let source = event.seq;
        let event_type = event.r#type.clone();
        remap_payload_references(&event_type, &mut event.data, &self.mapping)?;
        self.mapping.record(source, target);
        rewrite_message_sources(&event_type, &mut event.data, event.seq)?;
        lift_tool_result(&event_type, &mut event.data, event.seq)?;
        migrate_content(&event_type, &mut event.data, event.seq)?;
        event.seq = target;
        event.source_event_seqs = None;
        event.surface_op = None;
        event.physical = false;
        self.out.push(event);
        Ok(())
    }

    /// The implicit `turn/end` a resume splice owes the previous turn.
    fn observe_restart(&mut self, event: &Envelope) -> Option<u64> {
        let next = self.turn.map(|turn| turn + 1);
        let interrupted = if event.r#type == "turn/start"
            && !self.step_open
            && self.next_turn_spliced
            && event.data.get("turn").and_then(Value::as_u64) == next
        {
            self.turn
        } else {
            None
        };
        self.next_turn_spliced = event.r#type == "agent/inbox/spliced"
            && event.data.get("target").and_then(Value::as_str) == Some("next-turn")
            && event
                .data
                .get("inserted")
                .and_then(Value::as_array)
                .is_some_and(|inserted| !inserted.is_empty());
        match event.r#type.as_str() {
            "turn/start" => {
                if let Some(turn) = event.data.get("turn").and_then(Value::as_u64) {
                    self.turn = Some(turn);
                }
            }
            "turn/end" => self.turn = None,
            "step/start" => self.step_open = true,
            "step/end" => self.step_open = false,
            _ => {}
        }
        interrupted
    }

    fn finish(&mut self, source_cut: u64) -> Result<u64, SessionError> {
        let cut = self.cut.unwrap_or(0);
        let cut_in_source = self.source_cut.unwrap_or(0);
        let mut existing: HashMap<String, Map<String, Value>> = HashMap::new();
        for data in std::mem::take(&mut self.catalogs) {
            let fact = catalog_fact(&data, "subagent/catalog")?;
            let id = fact
                .get("childId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if existing.insert(id.clone(), fact).is_some() {
                return Err(fail(format!("duplicate catalog child {id}")));
            }
        }
        if cut_in_source != source_cut {
            return Err(fail(
                "format v3 inherited cut disagrees with its source marker".to_string(),
            ));
        }
        if let Some(foreign) = self.foreign_delivery {
            if !self.is_seeded || foreign >= cut_in_source {
                return Err(fail(
                    "current-generation delivery marker names the wrong Session".to_string(),
                ));
            }
        }

        for source in std::mem::take(&mut self.candidates) {
            let id = source
                .get("childId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let fact = child_catalog_fact(&source)?;
            if let Some(current) = existing.get(&id) {
                if current.get("childCreatedAt") != source.get("childCreatedAt") {
                    return Err(fail(format!(
                        "Session {id} conflicts with its parent catalog"
                    )));
                }
                if let Some(fact) = &fact {
                    if ["mode", "label"]
                        .iter()
                        .any(|key| current.get(*key) != fact.get(*key))
                    {
                        return Err(fail(format!(
                            "Session {id} conflicts with its parent catalog"
                        )));
                    }
                }
                continue;
            }
            let entry = fact.unwrap_or_else(|| {
                let mut entry = Map::new();
                entry.insert("version".to_string(), json!(1));
                entry.insert("childId".to_string(), json!(id));
                entry.insert(
                    "childCreatedAt".to_string(),
                    source.get("childCreatedAt").cloned().unwrap_or(Value::Null),
                );
                entry.insert("mode".to_string(), json!("unknown"));
                entry
            });
            existing.insert(id, entry.clone());
            let seq = self.next_seq;
            self.next_seq += 1;
            self.out
                .push(Envelope::at("subagent/catalog", seq, self.time, entry));
        }
        Ok(cut)
    }
}

// ---------------------------------------------------------------------------
// Catalog facts
// ---------------------------------------------------------------------------

/// Validate and sort the caller's child evidence by creation time, then id.
fn catalog_candidates(children: &[Value]) -> Result<Vec<Map<String, Value>>, SessionError> {
    let mut candidates = Vec::with_capacity(children.len());
    for child in children {
        candidates.push(child_catalog_source(child)?);
    }
    candidates.sort_by(|left, right| {
        let left_time = left
            .get("childCreatedAt")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let right_time = right
            .get("childCreatedAt")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        left_time.cmp(&right_time).then_with(|| {
            let left_id = left
                .get("childId")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let right_id = right
                .get("childId")
                .and_then(Value::as_str)
                .unwrap_or_default();
            left_id.cmp(right_id)
        })
    });
    Ok(candidates)
}

/// Validate one supplemental child-evidence record.
fn child_catalog_source(value: &Value) -> Result<Map<String, Value>, SessionError> {
    let source = object(value, "catalog child evidence")?;
    if !source.contains_key("descriptor") {
        return Err(fail(
            "catalog migration requires historical child identity and descriptor evidence"
                .to_string(),
        ));
    }
    non_empty(
        field(source, "childId", "catalog child evidence")?,
        "catalog child id",
    )?;
    count(
        field(source, "childCreatedAt", "catalog child evidence")?,
        "catalog child creation time",
    )?;
    count(
        field(source, "descriptorCount", "catalog child evidence")?,
        "child descriptor count",
    )?;
    Ok(source.clone())
}

/// The catalog fact one child's descriptor supports, when it supports one.
fn child_catalog_fact(
    source: &Map<String, Value>,
) -> Result<Option<Map<String, Value>>, SessionError> {
    let id = source
        .get("childId")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let count = source.get("descriptorCount").and_then(Value::as_u64);
    let descriptor = source.get("descriptor").and_then(Value::as_object);
    let version = descriptor
        .and_then(|descriptor| descriptor.get("version"))
        .and_then(Value::as_u64);
    if count != Some(1) || descriptor.is_none() || !matches!(version, Some(1..=3)) {
        return Ok(None);
    }
    let descriptor = descriptor.expect("descriptor is present");
    non_empty(
        field(descriptor, "provider", "subagent descriptor")?,
        "subagent descriptor provider",
    )?;
    let version = version.unwrap_or(1);
    let mode = if version == 1 {
        // V1 described only continuable children and carried no mode field.
        "continuable".to_string()
    } else {
        literal(
            field(descriptor, "mode", "subagent descriptor")?,
            &["continuable", "one-shot"],
            "subagent descriptor mode",
        )?
        .to_string()
    };
    let mut fact = Map::new();
    fact.insert("version".to_string(), json!(0));
    fact.insert("childId".to_string(), json!(id));
    fact.insert(
        "childCreatedAt".to_string(),
        source.get("childCreatedAt").cloned().unwrap_or(Value::Null),
    );
    fact.insert("mode".to_string(), json!(mode));
    if let Some(label) = descriptor.get("label") {
        non_empty(label, "subagent descriptor label")?;
        fact.insert("label".to_string(), label.clone());
    }
    catalog_fact(&Value::Object(fact), "subagent/catalog").map(Some)
}

/// Validate one `subagent/catalog` payload.
fn catalog_fact(value: &Value, subject: &str) -> Result<Map<String, Value>, SessionError> {
    let fact = object(value, subject)?;
    let version = fact.get("version").and_then(Value::as_u64);
    let mode = fact.get("mode").and_then(Value::as_str);
    let valid = matches!(version, Some(0..=2))
        && matches!(
            mode,
            Some("continuable" | "one-shot" | "external" | "unknown")
        )
        && (version == Some(2)) == (mode == Some("external"))
        && !(version == Some(0) && mode == Some("unknown"))
        && (mode != Some("continuable") || fact.get("label").is_some_and(Value::is_string))
        && fact.get("label").is_none_or(Value::is_string);
    if !valid {
        return Err(fail(format!(
            "{subject} requires a supported versioned catalog fact"
        )));
    }
    non_empty(
        field(fact, "childId", subject)?,
        &format!("{subject} childId"),
    )?;
    count(
        field(fact, "childCreatedAt", subject)?,
        "catalog child creation time",
    )?;
    Ok(fact.clone())
}

// ---------------------------------------------------------------------------
// Source attribution
// ---------------------------------------------------------------------------

/// The current producer kind for one released v3 plugin string.
fn producer_kind(plugin: &str, role: Option<&str>) -> String {
    if plugin == "@deepseek-ai/dsh-system-prompt" && role == Some("system") {
        return "system-prompt".to_string();
    }
    if let Some((_, renamed)) = RENAMED_PRODUCERS
        .iter()
        .find(|(released, _)| *released == plugin)
    {
        return (*renamed).to_string();
    }
    if SAME_NAME_PRODUCERS.contains(&plugin) {
        return plugin.to_string();
    }
    format!("plugin:{plugin}")
}

/// Rewrite every declared message slot's source.
fn rewrite_message_sources(
    event_type: &str,
    data: &mut Map<String, Value>,
    _seq: u64,
) -> Result<(), SessionError> {
    map_messages(event_type, data, &mut |message| {
        let Some(source) = message.get("source").and_then(Value::as_object) else {
            return Ok(());
        };
        if source.get("kind").and_then(Value::as_str) != Some("plugin") {
            return Ok(());
        }
        let plugin = non_empty(
            source.get("plugin").ok_or_else(|| {
                fail("plugin source is not canonical: plugin requires a string".to_string())
            })?,
            "plugin source plugin",
        )?
        .to_string();
        let role = message.get("role").and_then(Value::as_str);
        let kind = producer_kind(&plugin, role);
        let mut rewritten = Map::new();
        for (key, value) in source {
            match key.as_str() {
                "plugin" => {}
                "kind" => {
                    rewritten.insert(key.clone(), json!(kind));
                }
                _ => {
                    rewritten.insert(key.clone(), value.clone());
                }
            }
        }
        message.insert("source".to_string(), Value::Object(rewritten));
        Ok(())
    })
}

/// Visit only the messages carried by first-party payload slots.
fn map_messages(
    event_type: &str,
    data: &mut Map<String, Value>,
    transform: &mut impl FnMut(&mut Map<String, Value>) -> Result<(), SessionError>,
) -> Result<(), SessionError> {
    match event_type {
        "user/message" => transform(data),
        "developer/message" | "system/message" | "assistant/message" | "tool/result" => {
            let message = data
                .get_mut("message")
                .ok_or_else(|| fail(format!("{event_type} requires a message")))?;
            transform(object_mut(message, &format!("{event_type} message"))?)
        }
        "agent/inbox/spliced" | "session/title-llm-request" => {
            let key = if event_type == "agent/inbox/spliced" {
                "inserted"
            } else {
                "messages"
            };
            let messages = data
                .get_mut(key)
                .ok_or_else(|| fail(format!("{event_type} requires a message array")))?;
            for (index, message) in array_mut(messages, key)?.iter_mut().enumerate() {
                transform(object_mut(
                    message,
                    &format!("{event_type} {key}[{index}]"),
                )?)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Tool-role lifting
// ---------------------------------------------------------------------------

/// Lift a released wrapper `tool/result` row into a first-class tool message.
fn lift_tool_result(
    event_type: &str,
    data: &mut Map<String, Value>,
    seq: u64,
) -> Result<(), SessionError> {
    if event_type != "tool/result" {
        return Ok(());
    }
    let Some(message) = data.get("message").and_then(Value::as_object) else {
        return Ok(());
    };
    if message.get("role").and_then(Value::as_str) != Some("user") {
        return Ok(());
    }
    let subject = format!("format v3 tool/result at seq {seq}");
    let source = message
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            fail(format!(
                "{subject} requires exactly one tool-result wrapper matching its tool source"
            ))
        })?;
    let call_id = source
        .get("callId")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let id = message
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let wrapper = message
        .get("content")
        .and_then(Value::as_array)
        .filter(|content| content.len() == 1)
        .and_then(|content| content[0].as_object());
    let valid = !id.is_empty()
        && source.get("kind").and_then(Value::as_str) == Some("tool")
        && !call_id.is_empty()
        && wrapper.is_some_and(|wrapper| {
            wrapper.get("type").and_then(Value::as_str) == Some("tool-result")
                && wrapper.get("toolCallId").and_then(Value::as_str) == Some(call_id)
        });
    if !valid {
        return Err(fail(format!(
            "{subject} requires exactly one tool-result wrapper matching its tool source"
        )));
    }
    let wrapper = wrapper.expect("wrapper is present");
    let is_error = match wrapper.get("isError") {
        Some(value) => Some(boolean(value, &format!("{subject} tool-result isError"))?),
        None => None,
    };
    let content = wrapper
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| fail(format!("{subject} tool-result content must be an array")))?;
    if content
        .iter()
        .any(|block| block.get("type").and_then(Value::as_str) == Some("tool-result"))
    {
        return Err(fail(format!(
            "{subject} contains a nested tool-result unsupported by this converter"
        )));
    }

    let mut target = Map::new();
    target.insert("role".to_string(), json!("tool"));
    target.insert("source".to_string(), Value::Object(source.clone()));
    target.insert("toolCallId".to_string(), json!(call_id));
    target.insert("content".to_string(), Value::Array(content.clone()));
    if let Some(is_error) = is_error {
        target.insert("isError".to_string(), json!(is_error));
    }
    target.insert("id".to_string(), json!(id));
    for (key, value) in message {
        if ["id", "role", "source", "content"].contains(&key.as_str()) {
            continue;
        }
        target.insert(format!("plugin:message:{key}"), value.clone());
    }
    for (key, value) in wrapper {
        if ["type", "toolCallId", "content", "isError"].contains(&key.as_str()) {
            continue;
        }
        target.insert(format!("plugin:result:{key}"), value.clone());
    }
    data.insert("message".to_string(), Value::Object(target));
    Ok(())
}

// ---------------------------------------------------------------------------
// Content tags
// ---------------------------------------------------------------------------

const V3_BLOCK_TYPES: [&str; 6] = [
    "text",
    "reasoning",
    "image",
    "file",
    "tool-call",
    "tool-result",
];

/// Convert declared content tags, keeping unknown tags as `plugin:` extensions.
fn migrate_content(
    event_type: &str,
    data: &mut Map<String, Value>,
    seq: u64,
) -> Result<(), SessionError> {
    let subject = format!("format v3 {event_type} at seq {seq}");
    map_messages(event_type, data, &mut |message| {
        if let Some(content) = message.get_mut("content") {
            migrate_content_array(content, &subject)?;
        }
        Ok(())
    })?;
    match event_type {
        "compaction/summary" => {
            if let Some(summary) = data.get_mut("summary") {
                migrate_content_array(summary, &subject)?;
            }
            if let Some(raw) = data.get_mut("rawOutput") {
                migrate_content_array(raw, &subject)?;
            }
        }
        "tool/ptc-dispatch" => {
            if let Some(content) = data.get_mut("content") {
                migrate_content_array(content, &subject)?;
            }
        }
        "team/message/queued" => {
            if let Some(message) = data.get_mut("message").and_then(Value::as_object_mut) {
                if let Some(content) = message.get_mut("content") {
                    migrate_content_array(content, &subject)?;
                }
            }
        }
        _ => {}
    }
    if event_type == "request/header" {
        if let Some(tools) = data
            .get("header")
            .and_then(Value::as_object)
            .and_then(|header| header.get("tools"))
            .and_then(Value::as_array)
        {
            for (index, tool) in tools.iter().enumerate() {
                if tool.get("deferLoading").is_some() {
                    return Err(fail(format!(
                        "{subject}.header.tools[{index}] contains deferLoading, which is only defined in V4"
                    )));
                }
            }
        }
    }
    if event_type == "assistant/message" || event_type == "assistant/attempt" {
        if let Some(stream) = data.get_mut("stream").and_then(Value::as_array_mut) {
            for (index, entry) in stream.iter_mut().enumerate() {
                if entry.get("type").and_then(Value::as_str) != Some("chunk") {
                    continue;
                }
                if let Some(chunk) = entry.get_mut("chunk").and_then(Value::as_object_mut) {
                    migrate_chunk_tag(chunk, &format!("{subject}.stream[{index}]"))?;
                }
            }
        }
    }
    Ok(())
}

fn migrate_content_array(value: &mut Value, subject: &str) -> Result<(), SessionError> {
    let Some(blocks) = value.as_array_mut() else {
        return Err(fail(format!("{subject} content must be an array")));
    };
    for (index, block) in blocks.iter_mut().enumerate() {
        let block = block
            .as_object_mut()
            .ok_or_else(|| fail(format!("{subject}[{index}] requires content blocks")))?;
        let Some(kind) = block.get("type").and_then(Value::as_str) else {
            return Err(fail(format!(
                "{subject}[{index}] requires content blocks with string type tags"
            )));
        };
        if V3_BLOCK_TYPES.contains(&kind) {
            continue;
        }
        let namespaced = format!("plugin:{kind}");
        block.insert("type".to_string(), json!(namespaced));
    }
    Ok(())
}

fn migrate_chunk_tag(chunk: &mut Map<String, Value>, subject: &str) -> Result<(), SessionError> {
    match chunk.get("type").and_then(Value::as_str) {
        Some("block-end") => {
            if let Some(block) = chunk.get_mut("block").and_then(Value::as_object_mut) {
                let Some(kind) = block.get("type").and_then(Value::as_str) else {
                    return Err(fail(format!("{subject}.block requires a type tag")));
                };
                if !V3_BLOCK_TYPES.contains(&kind) {
                    let namespaced = format!("plugin:{kind}");
                    block.insert("type".to_string(), json!(namespaced));
                }
            }
        }
        Some("block-start") => {
            let Some(kind) = chunk.get("blockType").and_then(Value::as_str) else {
                return Err(fail(format!("{subject} blockType must be a string")));
            };
            if !V3_BLOCK_TYPES.contains(&kind) {
                let namespaced = format!("plugin:{kind}");
                chunk.insert("blockType".to_string(), json!(namespaced));
            }
        }
        _ => {}
    }
    Ok(())
}
