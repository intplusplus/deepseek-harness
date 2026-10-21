//! The v2 -> v3 edge: promote streamed system prompts into `system/message`
//! events and adopt the v3 vocabulary
//! (`origin/packages/session/session-format-v2-to-v3/src/migration.ts`).
//!
//! The system prompt used to ride inside `request/header.header.system`. v3
//! makes it derived history, so every prompt *change* becomes a surface event
//! placed immediately before the header that carried it, chained through
//! `surfaceOp: replace` so the model history keeps one system slot.

use std::collections::HashSet;

use serde_json::{json, Map, Value};

use crate::types::SessionError;

use super::support::{
    assert_envelope, assert_payload, assert_source_seqs, count, fail, field, is_surface_type,
    json_quote, object, remap_payload_references, sha256_hex, Mapping,
};
use super::{Envelope, Log};

/// One streaming v2 -> v3 stage.
struct Stage {
    session_id: String,
    is_seeded: bool,
    mapping: Mapping,
    target_seq: u64,
    source_cut: Option<u64>,
    target_cut: Option<u64>,
    last_foreign_delivery: Option<u64>,
    step: Option<(u64, u64)>,
    head: Option<u64>,
    prompt: String,
    original_ids: HashSet<String>,
    generated_ids: HashSet<String>,
    out: Vec<Envelope>,
}

/// Promote system prompts, remap references, and canonicalize v3 envelopes.
pub fn v2_to_v3(mut log: Log) -> Result<Log, SessionError> {
    if log.version()? != 2 {
        return Err(fail("expected format v2 header".to_string()));
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
    if let Some(header) = log.header.as_object_mut() {
        header.insert("version".to_string(), json!(3));
        if header.get("agentPreset").and_then(Value::as_str) == Some("code") {
            header.insert("agentPreset".to_string(), json!("ptc"));
        }
    }

    let mut stage = Stage {
        session_id,
        is_seeded,
        mapping: Mapping::new(),
        target_seq: 0,
        source_cut: if is_seeded { None } else { Some(0) },
        target_cut: if is_seeded { None } else { Some(0) },
        last_foreign_delivery: None,
        step: None,
        head: None,
        prompt: String::new(),
        original_ids: HashSet::new(),
        generated_ids: HashSet::new(),
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
        if event.r#type == "assistant/message" && event.source_event_seqs.is_some() {
            return Err(fail(format!(
                "assistant/message {} retains obsolete chunk references",
                event.seq
            )));
        }
        assert_source_seqs(&event, false)?;
        assert_payload(&event.r#type, &event.data, event.seq, 2)?;
        self.observe_message_ids(&event)?;

        if event.r#type == "request/header" {
            let mut prompt = String::new();
            if let Some(header) = event.data.get_mut("header").and_then(Value::as_object_mut) {
                if let Some(system) = header.shift_remove("system") {
                    if let Some(text) = system.as_str() {
                        prompt = text.to_string();
                    }
                }
            }
            if prompt != self.prompt {
                self.emit_system(&prompt, &event)?;
            }
        }

        if is_surface_type(&event.r#type) && self.head.is_none() {
            return Err(fail(
                "format v2 surface before first step cannot acquire a system head without changing chronology"
                    .to_string(),
            ));
        }
        if event.r#type == "session/end-seed"
            && event.data.get("inherited").and_then(Value::as_bool) == Some(true)
        {
            if !self.is_seeded {
                return Err(fail(
                    "format v2 unseeded Session contains an inherited end-seed marker".to_string(),
                ));
            }
            self.source_cut = Some(event.seq);
            self.target_cut = Some(self.target_seq);
        }
        if event.r#type == "session-log-deepseek/delivery-accepted" {
            let claimed = event
                .data
                .get("sessionFormatVersion")
                .and_then(Value::as_u64);
            if claimed == Some(3) {
                return Err(fail(
                    "format v2 delivery marker claims target format v3".to_string(),
                ));
            }
            if claimed == Some(2)
                && event.data.get("sessionId").and_then(Value::as_str) != Some(&self.session_id)
            {
                self.last_foreign_delivery = Some(event.seq);
            }
        }

        let source = event.seq;
        let target = self.target_seq;
        remap_payload_references(&event.r#type, &mut event.data, &self.mapping)?;
        self.mapping.record(source, target);
        self.target_seq += 1;
        event.seq = target;
        event.source_event_seqs = None;
        event.surface_op = None;
        rename_ptc(&mut event);
        canonicalize(&mut event)?;
        assert_payload(&event.r#type, &event.data, event.seq, 3)?;

        let event_type = event.r#type.clone();
        let turn = event.data.get("turn").and_then(Value::as_u64).unwrap_or(0);
        let step = event.data.get("step").and_then(Value::as_u64).unwrap_or(0);
        self.out.push(event);
        if event_type == "step/start" {
            self.step = Some((turn, step));
            if self.head.is_none() {
                let anchor = self.out.last().cloned().expect("just pushed");
                self.emit_system("", &anchor)?;
            }
        } else if event_type == "step/end" || event_type == "turn/end" {
            self.step = None;
        }
        Ok(())
    }

    fn observe_message_ids(&mut self, event: &Envelope) -> Result<(), SessionError> {
        let ids: Vec<String> = match event.r#type.as_str() {
            "user/message" => event
                .data
                .get("id")
                .and_then(Value::as_str)
                .map(|id| vec![id.to_string()])
                .unwrap_or_default(),
            "assistant/message" | "tool/result" => event
                .data
                .get("message")
                .and_then(|message| message.get("id"))
                .and_then(Value::as_str)
                .map(|id| vec![id.to_string()])
                .unwrap_or_default(),
            "agent/inbox/spliced" => message_ids(event.data.get("inserted")),
            "session/title-llm-request" => message_ids(event.data.get("messages")),
            _ => Vec::new(),
        };
        for id in ids {
            if self.generated_ids.contains(&id) {
                return Err(fail(
                    "source message id collides with a generated system message id".to_string(),
                ));
            }
            self.original_ids.insert(id);
        }
        Ok(())
    }

    /// Emit one promoted `system/message` before its anchor.
    fn emit_system(&mut self, prompt: &str, anchor: &Envelope) -> Result<(), SessionError> {
        let Some((turn, step)) = self.step else {
            return Err(fail(
                "format v2 changed request prompt outside an open step cannot retain source chronology"
                    .to_string(),
            ));
        };
        let identity = format!(
            "[{},{},{},{}]",
            json_quote("session-format-v2-to-v3"),
            json_quote(&self.session_id),
            anchor.seq,
            json_quote(&anchor.r#type),
        );
        let id = format!("v2-to-v3-system-{}", sha256_hex(identity.as_bytes()));
        if self.original_ids.contains(&id) || self.generated_ids.contains(&id) {
            return Err(fail(
                "generated system message id collides with an existing message id".to_string(),
            ));
        }
        self.generated_ids.insert(id.clone());

        let mut message = Map::new();
        message.insert("id".to_string(), json!(id));
        message.insert("role".to_string(), json!("system"));
        let mut source = Map::new();
        source.insert("kind".to_string(), json!("plugin"));
        source.insert(
            "plugin".to_string(),
            json!("@deepseek-ai/dsh-system-prompt"),
        );
        message.insert("source".to_string(), Value::Object(source));
        message.insert(
            "content".to_string(),
            if prompt.is_empty() {
                Value::Array(Vec::new())
            } else {
                json!([{ "type": "text", "text": prompt }])
            },
        );

        let mut data = Map::new();
        data.insert("turn".to_string(), json!(turn));
        data.insert("step".to_string(), json!(step));
        data.insert("message".to_string(), Value::Object(message));

        let seq = self.target_seq;
        self.target_seq += 1;
        let mut event = Envelope::at("system/message", seq, anchor.time, data);
        event.surface_op = Some(match self.head {
            None => json!("append"),
            Some(head) => json!({ "op": "replace", "startSeq": head, "endSeq": head }),
        });
        if let Some(head) = self.head {
            event.source_event_seqs = Some(vec![head]);
        }
        self.out.push(event);
        self.head = Some(seq);
        self.prompt = prompt.to_string();
        Ok(())
    }

    fn finish(&mut self, source_cut: u64) -> Result<u64, SessionError> {
        let cut = self.source_cut.unwrap_or(0);
        if cut != source_cut {
            return Err(fail(
                "format v2 inherited end-seed marker disagrees with its source cut".to_string(),
            ));
        }
        if let Some(foreign) = self.last_foreign_delivery {
            if !self.is_seeded || foreign >= cut {
                return Err(fail(
                    "current-generation delivery marker names the wrong Session".to_string(),
                ));
            }
        }
        if self.is_seeded {
            return self.target_cut.ok_or_else(|| {
                fail("format v3 seeded Session lacks an inherited cut".to_string())
            });
        }
        Ok(0)
    }
}

/// The message identities carried by one message array.
fn message_ids(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|messages| {
            messages
                .iter()
                .filter_map(|message| message.get("id").and_then(Value::as_str))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Replace the retired v2 vocabulary with the v3 names.
fn rename_ptc(event: &mut Envelope) {
    match event.r#type.as_str() {
        "agent-preset/selected" => {
            if event.data.get("agentPreset").and_then(Value::as_str) == Some("code") {
                event.data.insert("agentPreset".to_string(), json!("ptc"));
            }
        }
        "tool/code-dispatch-start" => event.r#type = "tool/ptc-dispatch-start".to_string(),
        "tool/code-dispatch" => event.r#type = "tool/ptc-dispatch".to_string(),
        "user/message" => {
            if let Some(source) = event.data.get_mut("source").and_then(Value::as_object_mut) {
                rename_message_source(source);
            }
        }
        "agent/inbox/spliced" => {
            if let Some(messages) = event.data.get_mut("inserted").and_then(Value::as_array_mut) {
                for message in messages {
                    if let Some(source) = message.get_mut("source").and_then(Value::as_object_mut) {
                        rename_message_source(source);
                    }
                }
            }
        }
        "session/title-llm-request" => {
            if let Some(messages) = event.data.get_mut("messages").and_then(Value::as_array_mut) {
                for message in messages {
                    if let Some(source) = message.get_mut("source").and_then(Value::as_object_mut) {
                        rename_message_source(source);
                    }
                }
            }
        }
        _ => {}
    }
}

/// `tools-code-mode` became `tools-ptc`; the kind stays `plugin` until v4.
fn rename_message_source(source: &mut Map<String, Value>) {
    if source.get("kind").and_then(Value::as_str) == Some("plugin")
        && source.get("plugin").and_then(Value::as_str) == Some("tools-code-mode")
    {
        source.insert("plugin".to_string(), json!("tools-ptc"));
    }
}

/// Canonicalize the v3 envelope: replacement ranges and empty header members.
fn canonicalize(event: &mut Envelope) -> Result<(), SessionError> {
    if let Some(operation) = event.surface_op.take() {
        if operation.as_str() == Some("replace") {
            let map = object(&operation, "surfaceOp")?;
            let start = count(field(map, "start", "surfaceOp")?, "surfaceOp start")?;
            let end = count(field(map, "end", "surfaceOp")?, "surfaceOp end")?;
            event.surface_op = Some(json!({
                "op": "replace",
                "startSeq": start,
                "endSeq": end,
            }));
        } else {
            event.surface_op = Some(operation);
        }
    }
    if event.r#type == "request/header" {
        if let Some(header) = event.data.get_mut("header").and_then(Value::as_object_mut) {
            if header
                .get("tools")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
            {
                header.shift_remove("tools");
            }
            if header
                .get("adapterDefaults")
                .and_then(Value::as_object)
                .is_some_and(Map::is_empty)
            {
                header.shift_remove("adapterDefaults");
            }
        }
    }
    if event.r#type != "tool/result" {
        return Ok(());
    }
    if !event.data.contains_key("error") {
        return Ok(());
    }
    let message = event
        .data
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| fail("tool/result requires a message".to_string()))?;
    let content = message
        .get("content")
        .and_then(Value::as_array)
        .filter(|content| content.len() == 1)
        .and_then(|content| content[0].as_object());
    let is_error = content
        .and_then(|block| block.get("isError"))
        .and_then(Value::as_bool);
    if content
        .and_then(|block| block.get("type"))
        .and_then(Value::as_str)
        != Some("tool-result")
        || is_error != Some(true)
    {
        return Err(fail(
            "tool/result carries error metadata for a non-error tool result".to_string(),
        ));
    }
    Ok(())
}
