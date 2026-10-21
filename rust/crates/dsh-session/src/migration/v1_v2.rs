//! The v1 → v2 edge: embed released top-level Assistant chunks into attempt
//! events (`origin/packages/session/session-format-v1-to-v2/src/migration.ts`).
//!
//! The edge is the only one that renumbers events: five chunk rows become one
//! attempt row, so every later payload reference (`messageSeqs`,
//! `shadowedSeqs`, `sourceEventSeq`) is remapped into target coordinates, and a
//! seeded session gains its `session/end-seed` marker at the inherited cut.

use serde_json::{json, Map, Value};

use crate::types::SessionError;

use super::support::{
    assert_envelope, assert_payload, assert_source_seqs, assert_stream_chunk, count, fail, field,
    object, remap_payload_references, Mapping,
};
use super::{Envelope, Log};

/// One packed delta run inside a compact stream.
#[derive(Clone, Debug)]
enum Run {
    Text {
        time0: u64,
        index: u64,
        dt: Vec<u64>,
        texts: Vec<String>,
        last_time: u64,
    },
    Reasoning {
        time0: u64,
        index: u64,
        dt: Vec<u64>,
        texts: Vec<String>,
        last_time: u64,
    },
    ToolCall {
        time0: u64,
        index: u64,
        dt: Vec<u64>,
        id: String,
        name: Option<String>,
        args: Vec<String>,
        last_time: u64,
    },
}

/// One stream record: a packed run, or a raw chunk.
#[derive(Clone, Debug)]
enum StreamRecord {
    Run(Run),
    Chunk { time: u64, chunk: Value },
}

impl StreamRecord {
    fn last_time(&self) -> u64 {
        match self {
            Self::Run(Run::Text { last_time, .. })
            | Self::Run(Run::Reasoning { last_time, .. })
            | Self::Run(Run::ToolCall { last_time, .. }) => *last_time,
            Self::Chunk { time, .. } => *time,
        }
    }

    fn to_value(&self) -> Value {
        match self {
            Self::Chunk { time, chunk } => json!({ "type": "chunk", "time": time, "chunk": chunk }),
            Self::Run(Run::Text {
                time0,
                index,
                dt,
                texts,
                ..
            }) => json!({
                "type": "text-chunks",
                "time0": time0,
                "index": index,
                "dt": dt,
                "texts": texts,
            }),
            Self::Run(Run::Reasoning {
                time0,
                index,
                dt,
                texts,
                ..
            }) => json!({
                "type": "reasoning-chunks",
                "time0": time0,
                "index": index,
                "dt": dt,
                "texts": texts,
            }),
            Self::Run(Run::ToolCall {
                time0,
                index,
                dt,
                id,
                name,
                args,
                ..
            }) => match name {
                Some(name) => json!({
                    "type": "tool-call-chunks",
                    "time0": time0,
                    "index": index,
                    "dt": dt,
                    "id": id,
                    "name": name,
                    "args": args,
                }),
                None => json!({
                    "type": "tool-call-chunks",
                    "time0": time0,
                    "index": index,
                    "dt": dt,
                    "id": id,
                    "args": args,
                }),
            },
        }
    }
}

/// One in-flight attempt.
struct Attempt {
    turn: u64,
    step: u64,
    /// Contiguous source spans consumed by this attempt, in order.
    spans: Vec<(u64, u64)>,
    chunk_count: u64,
    last_chunk_seq: u64,
    last_chunk_time: u64,
    terminal: bool,
    accumulator: Vec<StreamRecord>,
    stream: Vec<StreamRecord>,
    after_last: Vec<Envelope>,
}

impl Attempt {
    fn new(turn: u64, step: u64) -> Self {
        Self {
            turn,
            step,
            spans: Vec::new(),
            chunk_count: 0,
            last_chunk_seq: 0,
            last_chunk_time: 0,
            terminal: false,
            accumulator: Vec::new(),
            stream: Vec::new(),
            after_last: Vec::new(),
        }
    }

    /// Fold one chunk into the accumulator, packing delta runs.
    fn accumulate(&mut self, time: u64, chunk: &Value) -> Result<(), SessionError> {
        let map = object(chunk, "assistant/chunk chunk")?;
        match map.get("type").and_then(Value::as_str) {
            Some("text-delta") | Some("reasoning-delta") => {
                let reasoning = map.get("type").and_then(Value::as_str) == Some("reasoning-delta");
                let index = count(
                    field(map, "index", "assistant/chunk chunk")?,
                    "assistant/chunk index",
                )?;
                let text = super::support::string(
                    field(map, "text", "assistant/chunk chunk")?,
                    "assistant/chunk text",
                )?
                .to_string();
                self.push_delta(index, reasoning, text, time);
            }
            Some("tool-call-delta") => {
                let index = count(
                    field(map, "index", "assistant/chunk chunk")?,
                    "assistant/chunk index",
                )?;
                let id = super::support::non_empty(
                    field(map, "id", "assistant/chunk chunk")?,
                    "assistant/chunk id",
                )?
                .to_string();
                let name = match map.get("name") {
                    Some(value) => {
                        Some(super::support::string(value, "assistant/chunk name")?.to_string())
                    }
                    None => None,
                };
                let delta = super::support::string(
                    field(map, "argumentsDelta", "assistant/chunk chunk")?,
                    "assistant/chunk argumentsDelta",
                )?
                .to_string();
                if id.is_empty() || name.as_deref() == Some("") {
                    self.accumulator.push(StreamRecord::Chunk {
                        time,
                        chunk: chunk.clone(),
                    });
                    return Ok(());
                }
                self.push_tool_delta(index, id, name, delta, time);
            }
            _ => self.accumulator.push(StreamRecord::Chunk {
                time,
                chunk: chunk.clone(),
            }),
        }
        Ok(())
    }

    fn push_delta(&mut self, index: u64, reasoning: bool, text: String, time: u64) {
        if let Some(StreamRecord::Run(run)) = self.accumulator.last_mut() {
            let gap = match run {
                Run::Text {
                    index: previous,
                    last_time,
                    ..
                } if !reasoning && *previous == index && *last_time <= time => {
                    Some(time - *last_time)
                }
                Run::Reasoning {
                    index: previous,
                    last_time,
                    ..
                } if reasoning && *previous == index && *last_time <= time => {
                    Some(time - *last_time)
                }
                _ => None,
            };
            if let Some(gap) = gap {
                match run {
                    Run::Text {
                        dt,
                        texts,
                        last_time,
                        ..
                    }
                    | Run::Reasoning {
                        dt,
                        texts,
                        last_time,
                        ..
                    } => {
                        dt.push(gap);
                        texts.push(text);
                        *last_time = time;
                    }
                    Run::ToolCall { .. } => {}
                }
                return;
            }
        }
        let records = if reasoning {
            StreamRecord::Run(Run::Reasoning {
                time0: time,
                index,
                dt: Vec::new(),
                texts: vec![text],
                last_time: time,
            })
        } else {
            StreamRecord::Run(Run::Text {
                time0: time,
                index,
                dt: Vec::new(),
                texts: vec![text],
                last_time: time,
            })
        };
        self.accumulator.push(records);
    }

    fn push_tool_delta(
        &mut self,
        index: u64,
        id: String,
        name: Option<String>,
        delta: String,
        time: u64,
    ) {
        if let Some(StreamRecord::Run(Run::ToolCall {
            index: previous,
            dt,
            id: previous_id,
            name: previous_name,
            args,
            last_time,
            ..
        })) = self.accumulator.last_mut()
        {
            if *previous == index
                && *previous_id == id
                && *previous_name == name
                && *last_time <= time
            {
                dt.push(time - *last_time);
                args.push(delta);
                *last_time = time;
                return;
            }
        }
        self.accumulator.push(StreamRecord::Run(Run::ToolCall {
            time0: time,
            index,
            dt: Vec::new(),
            id,
            name,
            args: vec![delta],
            last_time: time,
        }));
    }

    /// Flush packed runs into the durable stream, merging adjacent runs.
    fn flush_accumulator(&mut self) {
        for record in std::mem::take(&mut self.accumulator) {
            let last_time = record.last_time();
            push_merging(&mut self.stream, record, last_time);
        }
    }

    fn stream(&mut self) -> Vec<Value> {
        self.flush_accumulator();
        self.stream.iter().map(StreamRecord::to_value).collect()
    }
}

/// Append one run, merging it into the previous run when the shapes line up.
fn push_merging(records: &mut Vec<StreamRecord>, source: StreamRecord, last_time: u64) {
    let merge = match (records.last(), &source) {
        (
            Some(StreamRecord::Run(Run::Text {
                index,
                last_time: previous,
                ..
            })),
            StreamRecord::Run(Run::Text {
                time0, index: next, ..
            }),
        ) => index == next && *previous <= *time0,
        (
            Some(StreamRecord::Run(Run::Reasoning {
                index,
                last_time: previous,
                ..
            })),
            StreamRecord::Run(Run::Reasoning {
                time0, index: next, ..
            }),
        ) => index == next && *previous <= *time0,
        (
            Some(StreamRecord::Run(Run::ToolCall {
                index,
                id,
                name,
                last_time: previous,
                ..
            })),
            StreamRecord::Run(Run::ToolCall {
                time0,
                index: next,
                id: next_id,
                name: next_name,
                ..
            }),
        ) => index == next && id == next_id && name == next_name && *previous <= *time0,
        _ => false,
    };
    if merge {
        if let Some(previous) = records.last_mut() {
            match (previous, &source) {
                (
                    StreamRecord::Run(Run::Text {
                        dt,
                        texts,
                        last_time: previous,
                        ..
                    }),
                    StreamRecord::Run(Run::Text {
                        time0,
                        dt: gaps,
                        texts: values,
                        ..
                    }),
                ) => {
                    dt.push(time0 - *previous);
                    dt.extend(gaps);
                    texts.extend(values.iter().cloned());
                    *previous = last_time;
                }
                (
                    StreamRecord::Run(Run::Reasoning {
                        dt,
                        texts,
                        last_time: previous,
                        ..
                    }),
                    StreamRecord::Run(Run::Reasoning {
                        time0,
                        dt: gaps,
                        texts: values,
                        ..
                    }),
                ) => {
                    dt.push(time0 - *previous);
                    dt.extend(gaps);
                    texts.extend(values.iter().cloned());
                    *previous = last_time;
                }
                (
                    StreamRecord::Run(Run::ToolCall {
                        dt,
                        args,
                        last_time: previous,
                        ..
                    }),
                    StreamRecord::Run(Run::ToolCall {
                        time0,
                        dt: gaps,
                        args: values,
                        ..
                    }),
                ) => {
                    dt.push(time0 - *previous);
                    dt.extend(gaps);
                    args.extend(values.iter().cloned());
                    *previous = last_time;
                }
                _ => {}
            }
        }
        return;
    }
    records.push(source);
}

/// Turn/step nesting observed while the edge runs.
#[derive(Default)]
struct LegacyTurn {
    open_turn: Option<u64>,
    open_step: Option<u64>,
    previous: Option<Envelope>,
}

/// One streaming v1 → v2 stage.
struct Stage {
    session_id: String,
    is_seeded: bool,
    source_cut: u64,
    mapping: Mapping,
    target_seq: u64,
    target_cut: Option<u64>,
    last_time: u64,
    pending: Option<Attempt>,
    legacy: LegacyTurn,
    out: Vec<Envelope>,
}

/// Embed released v1 Assistant chunks into v2 attempt events.
pub fn v1_to_v2(mut log: Log) -> Result<Log, SessionError> {
    if log.version()? != 1 {
        return Err(fail("expected format v1 header".to_string()));
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
        header.insert("version".to_string(), json!(2));
    }

    let mut stage = Stage {
        session_id,
        is_seeded,
        source_cut: log.cut,
        mapping: Mapping::new(),
        target_seq: 0,
        target_cut: if is_seeded { None } else { Some(0) },
        last_time: created_at,
        pending: None,
        legacy: LegacyTurn::default(),
        out: Vec::new(),
    };
    let source_cut = stage.source_cut;
    if source_cut > log.events.len() as u64 {
        return Err(fail(
            "format v1 inherited event count exceeds its events".to_string(),
        ));
    }
    for (index, envelope) in log.events.into_iter().enumerate() {
        stage.push(envelope, index)?;
    }
    let cut = stage.finish()?;
    log.events = stage.out;
    log.cut = cut;
    Ok(log)
}

impl Stage {
    fn push(&mut self, event: Envelope, index: usize) -> Result<(), SessionError> {
        assert_envelope(&event, index)?;
        if super::support::v0_disposition(&event.r#type).is_none() {
            return Err(fail(format!(
                "format v1 contains unknown event type `{}` at seq {index}",
                event.r#type
            )));
        }
        assert_source_seqs(&event, event.r#type == "assistant/message")?;
        assert_payload(&event.r#type, &event.data, event.seq, 1)?;

        let interrupted = self.legacy_interrupted_turn(&event);
        if event.r#type == "turn/start" && self.legacy.open_turn.is_some() && interrupted.is_none()
        {
            return Err(fail(format!(
                "turn/start {index} does not close the prior turn"
            )));
        }
        self.assert_source_delivery_marker(&event)?;
        self.observe_legacy_turn(&event);
        self.last_time = event.time;

        if let Some(turn) = interrupted {
            self.finish_attempt()?;
            let mut data = Map::new();
            data.insert("turn".to_string(), json!(turn));
            data.insert("reason".to_string(), json!({ "kind": "interrupted" }));
            self.emit_generated(event.seq, Envelope::at("turn/end", 0, event.time, data))?;
        }

        if event.r#type == "user/message" {
            if let Some((change, message)) = split_legacy_goal(&event)? {
                self.emit_generated(event.seq, change)?;
                self.emit_source(message)?;
                return Ok(());
            }
        }

        if event.r#type == "assistant/chunk" {
            return self.transform_chunk(event);
        }
        if event.r#type == "assistant/message" {
            return self.transform_message(event);
        }
        if matches!(
            event.r#type.as_str(),
            "turn/end" | "step/end" | "llm/retry" | "llm/retry-started"
        ) {
            self.finish_attempt()?;
            return self.emit_source(event);
        }
        if let Some(pending) = &mut self.pending {
            pending.after_last.push(event);
            return Ok(());
        }
        self.emit_source(event)
    }

    fn assert_source_delivery_marker(&self, event: &Envelope) -> Result<(), SessionError> {
        if event.r#type != "session-log-deepseek/delivery-accepted" {
            return Ok(());
        }
        let inherited = self.is_seeded && event.seq < self.source_cut;
        if event
            .data
            .get("sessionFormatVersion")
            .and_then(Value::as_u64)
            == Some(1)
            && !inherited
            && event.data.get("sessionId").and_then(Value::as_str) != Some(&self.session_id)
        {
            return Err(fail(
                "current-generation delivery marker names the wrong Session".to_string(),
            ));
        }
        Ok(())
    }

    /// The implicit `turn/end` a resume splice owes the previous turn.
    fn legacy_interrupted_turn(&self, event: &Envelope) -> Option<u64> {
        if event.r#type != "turn/start" {
            return None;
        }
        let open_turn = self.legacy.open_turn?;
        if self.legacy.open_step.is_some() {
            return None;
        }
        let turn = event.data.get("turn").and_then(Value::as_u64)?;
        if turn != open_turn + 1 {
            return None;
        }
        let previous = self.legacy.previous.as_ref()?;
        if previous.r#type != "agent/inbox/spliced" {
            return None;
        }
        if previous.data.get("target").and_then(Value::as_str) != Some("next-turn") {
            return None;
        }
        let inserted = previous.data.get("inserted").and_then(Value::as_array)?;
        if inserted.is_empty() {
            return None;
        }
        Some(open_turn)
    }

    fn observe_legacy_turn(&mut self, event: &Envelope) {
        match event.r#type.as_str() {
            "turn/start" => {
                self.legacy.open_turn = event.data.get("turn").and_then(Value::as_u64);
                self.legacy.open_step = None;
            }
            "turn/end" => {
                self.legacy.open_turn = None;
                self.legacy.open_step = None;
            }
            "step/start" => {
                self.legacy.open_step = event.data.get("step").and_then(Value::as_u64);
            }
            "step/end" => self.legacy.open_step = None,
            _ => {}
        }
        self.legacy.previous = Some(event.clone());
    }

    fn transform_chunk(&mut self, event: Envelope) -> Result<(), SessionError> {
        let turn = count(
            field(&event.data, "turn", "assistant/chunk")?,
            "assistant/chunk turn",
        )?;
        let step = count(
            field(&event.data, "step", "assistant/chunk")?,
            "assistant/chunk step",
        )?;
        let chunk = field(&event.data, "chunk", "assistant/chunk")?.clone();
        assert_stream_chunk(&chunk, &format!("assistant/chunk {} chunk", event.seq))?;

        let split = self.pending.as_ref().is_some_and(|pending| {
            pending.terminal || pending.turn != turn || pending.step != step
        });
        if split {
            self.finish_attempt()?;
        } else if self.pending.is_some() {
            self.flush_buffered()?;
        }
        if self.pending.is_none() {
            self.pending = Some(Attempt::new(turn, step));
        }
        if let Some(attempt) = &self.pending {
            self.assert_attempt_cut(attempt, event.seq)?;
        }
        let pending = self.pending.as_mut().expect("attempt is present");
        pending.accumulate(event.time, &chunk)?;
        pending.record_span(event.seq, 1, event.time);
        if chunk.get("type").and_then(Value::as_str) == Some("finish") {
            pending.terminal = true;
        }
        Ok(())
    }

    fn transform_message(&mut self, event: Envelope) -> Result<(), SessionError> {
        let turn = count(
            field(&event.data, "turn", "assistant/message")?,
            "assistant/message turn",
        )?;
        let step = count(
            field(&event.data, "step", "assistant/message")?,
            "assistant/message step",
        )?;
        let sources = event.source_event_seqs.clone();

        if let Some(pending) = &self.pending {
            if pending.turn != turn || pending.step != step {
                self.finish_attempt()?;
                return self.emit_message(event, Attempt::new(turn, step));
            }
        }
        match sources {
            None => {
                if self.pending.is_some() {
                    return Err(fail(format!(
                        "assistant/message {} does not cite its complete v1 chunk attempt",
                        event.seq
                    )));
                }
                self.emit_message(event, Attempt::new(turn, step))
            }
            Some(sources) if sources.is_empty() => {
                self.finish_attempt()?;
                self.emit_message(event, Attempt::new(turn, step))
            }
            Some(sources) => {
                let Some(mut pending) = self.pending.take() else {
                    return Err(fail(format!(
                        "assistant/message {} chunk references are not one complete ordered attempt",
                        event.seq
                    )));
                };
                if !matches_sources(&pending, &sources) {
                    return Err(fail(format!(
                        "assistant/message {} chunk references are not one complete ordered attempt",
                        event.seq
                    )));
                }
                self.assert_attempt_cut(&pending, event.seq)?;
                pending.terminal = true;
                for buffered in std::mem::take(&mut pending.after_last) {
                    self.emit_source(buffered)?;
                }
                self.emit_message(event, pending)
            }
        }
    }

    fn emit_message(
        &mut self,
        mut event: Envelope,
        mut attempt: Attempt,
    ) -> Result<(), SessionError> {
        let stream = attempt.stream();
        event
            .data
            .insert("stream".to_string(), Value::Array(stream));
        event.source_event_seqs = None;
        event.surface_op = None;
        self.emit_source(event)
    }

    fn flush_buffered(&mut self) -> Result<(), SessionError> {
        let Some(pending) = &mut self.pending else {
            return Ok(());
        };
        for event in std::mem::take(&mut pending.after_last) {
            self.emit_source(event)?;
        }
        Ok(())
    }

    fn finish_attempt(&mut self) -> Result<(), SessionError> {
        let Some(mut pending) = self.pending.take() else {
            return Ok(());
        };
        let stream = pending.stream();
        let mut data = Map::new();
        data.insert("turn".to_string(), json!(pending.turn));
        data.insert("step".to_string(), json!(pending.step));
        data.insert("stream".to_string(), Value::Array(stream));
        let generated = Envelope::at("assistant/attempt", 0, pending.last_chunk_time, data);
        self.emit_generated(pending.last_chunk_seq, generated)?;
        for event in std::mem::take(&mut pending.after_last) {
            self.emit_source(event)?;
        }
        Ok(())
    }

    fn assert_attempt_cut(&self, attempt: &Attempt, member: u64) -> Result<(), SessionError> {
        let first = attempt
            .spans
            .first()
            .map(|(first, _)| *first)
            .unwrap_or(member);
        if (first < self.source_cut) != (member < self.source_cut) {
            return Err(fail(format!(
                "inherited Session cut {} splits one Assistant attempt",
                self.source_cut
            )));
        }
        Ok(())
    }

    fn emit_source(&mut self, mut event: Envelope) -> Result<(), SessionError> {
        if self.is_seeded && event.seq == self.source_cut && event.r#type == "session/end-seed" {
            let mut data = Map::new();
            data.insert("inherited".to_string(), json!(true));
            event.data = data;
        }
        self.ensure_target_cut(event.seq, event.time, &event.r#type);
        let source = event.seq;
        let target = self.target_seq;
        let event_type = event.r#type.clone();
        remap_payload_references(&event_type, &mut event.data, &self.mapping)?;
        self.mapping.record(source, target);
        event.seq = target;
        event.source_event_seqs = None;
        event.surface_op = None;
        event.physical = false;
        self.out.push(event);
        self.target_seq += 1;
        Ok(())
    }

    fn emit_generated(&mut self, origin: u64, mut event: Envelope) -> Result<(), SessionError> {
        self.ensure_target_cut(origin, event.time, &event.r#type);
        let target = self.target_seq;
        let event_type = event.r#type.clone();
        remap_payload_references(&event_type, &mut event.data, &self.mapping)?;
        event.seq = target;
        self.out.push(event);
        self.target_seq += 1;
        Ok(())
    }

    /// Place the inherited cut before the first post-cut event.
    fn ensure_target_cut(&mut self, origin: u64, time: u64, event_type: &str) {
        if !self.is_seeded || self.target_cut.is_some() || origin < self.source_cut {
            return;
        }
        self.target_cut = Some(self.target_seq);
        if origin == self.source_cut && event_type == "session/end-seed" {
            return;
        }
        let mut data = Map::new();
        data.insert("inherited".to_string(), json!(true));
        let seq = self.target_seq;
        self.out
            .push(Envelope::at("session/end-seed", seq, time, data));
        self.target_seq += 1;
    }

    fn finish(&mut self) -> Result<u64, SessionError> {
        self.finish_attempt()?;
        if self.is_seeded && self.target_cut.is_none() {
            self.target_cut = Some(self.target_seq);
            let mut data = Map::new();
            data.insert("inherited".to_string(), json!(true));
            let seq = self.target_seq;
            self.out
                .push(Envelope::at("session/end-seed", seq, self.last_time, data));
            self.target_seq += 1;
        }
        Ok(self.target_cut.unwrap_or(0))
    }
}

impl Attempt {
    fn record_span(&mut self, first_seq: u64, count: u64, last_time: u64) {
        match self.spans.last_mut() {
            Some((first, current)) if *first + *current == first_seq => *current += count,
            _ => self.spans.push((first_seq, count)),
        }
        self.chunk_count += count;
        self.last_chunk_seq = first_seq + count - 1;
        self.last_chunk_time = last_time;
    }
}

fn matches_sources(attempt: &Attempt, sources: &[u64]) -> bool {
    if sources.len() as u64 != attempt.chunk_count {
        return false;
    }
    let mut index = 0;
    for (first, count) in &attempt.spans {
        for offset in 0..*count {
            if sources.get(index) != Some(&(first + offset)) {
                return false;
            }
            index += 1;
        }
    }
    true
}

/// Split a legacy `user/message` whose source carries a goal change.
fn split_legacy_goal(event: &Envelope) -> Result<Option<(Envelope, Envelope)>, SessionError> {
    let source = match event.data.get("source").and_then(Value::as_object) {
        Some(source) => source,
        None => return Ok(None),
    };
    if source.get("kind").and_then(Value::as_str) != Some("goal") {
        return Ok(None);
    }
    let Some(change) = source.get("change") else {
        return Ok(None);
    };
    let change_event = Envelope::at(
        "goal/change",
        0,
        event.time,
        object(change, "goal/change")?.clone(),
    );
    let mut message = event.clone();
    let mut replacement = Map::new();
    replacement.insert("kind".to_string(), json!("plugin"));
    replacement.insert("plugin".to_string(), json!("goal"));
    message
        .data
        .insert("source".to_string(), Value::Object(replacement));
    Ok(Some((change_event, message)))
}
