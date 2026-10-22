//! The released v4 physical envelope: header framing, row framing, and the
//! vocabulary admission the TypeScript codec applies on read.
//!
//! This is the transcription of the two released physical codecs:
//!
//! - `packages/session/session-format-v1-to-v2/src/codec.ts` — the physical
//!   row envelope (`type`, `seq`, `time`, `data` plus the optional `ignorable`,
//!   `sourceEventSeqs`, `surfaceOp`), the canonical header key order, and the
//!   `sourceEventSeqs` range codec.
//! - `packages/session/session-format-v3-to-v4/src/codec.ts` — the v4 stamp and
//!   the v4 row admission that delegates to the v2 codec.
//!
//! # Why this module exists
//!
//! The Rust core models the *flattened* event payload: [`SessionEvent`]
//! serializes as `{ "type": "...", ...payload }`, with no `seq`/`time`/`data`
//! envelope. The TypeScript writer stores the physical envelope instead, and
//! the two layouts are not interchangeable on the wire — a physical row read
//! as a flat event fails with `missing field 'turn'`.
//!
//! So the Rust side keeps two layouts and needs both a reader and a writer for
//! the physical one:
//!
//! - [`decode_physical_log`] reads a TypeScript-written v4 log.
//! - [`encode_physical_log`] writes a log the shipped v4 decoder accepts.
//!
//! The two are exact inverses for the events the core vocabulary covers.
//!
//! # Scope and deliberate omissions
//!
//! | Rule | Status |
//! |------|--------|
//! | Physical row envelope (`type`/`seq`/`time`/`data` + optionals) | implemented |
//! | Header `type: "session"` discriminator and canonical key order | implemented |
//! | `delegationDepth` required in the physical header | implemented |
//! | `ignorable` must be `true` when present | implemented |
//! | `sourceEventSeqs` range encoding and decoding | implemented |
//! | Unknown event types require `ignorable: true` | implemented |
//! | `surfaceOp` placement metadata | read and preserved, not interpreted by the core |
//! | Message `role` discriminator | emitted by the writer (see below) |
//! | `isError: false` on tool results | emitted by the writer (see below) |
//! | Cross-event relationship audit (turn/step nesting, tool lifecycle) | **not** implemented |
//!
//! The Rust message types in `dsh-llm` derive `role` from the event tag and
//! serialize `isError` only when it is `true`, and `serde_json` writes a whole
//! `f64` as `1.0` where JavaScript writes `1`. The **core flat serialization**
//! keeps that Rust-native shape (it is byte-locked by the existing tests), but
//! the physical **writer** restores the three fields the shipped TypeScript
//! decoder expects — `role` on every message row, `isError: false` on
//! non-error `tool/result` rows, and integer-valued `f64` written as integers —
//! so the decoder accepts every row the writer emits. The physical reader
//! accepts both shapes.

#![forbid(unsafe_code)]

use serde_json::{json, Map, Value};

use super::event::SessionEvent;
use super::log::Session;
use super::types::{SessionError, SessionHeader, MAX_SAFE_INTEGER, SESSION_FORMAT_VERSION};

/// The physical header's `type` discriminator. Every stored session header
/// carries it, and a header without it is not a session header.
pub const PHYSICAL_HEADER_TYPE: &str = "session";

/// The canonical physical header key order (TS `codec.ts:164-175`).
///
/// The order is observable: `serde_json`'s `preserve_order` keeps insertion
/// order on the wire, so the reader must build the header in exactly this
/// order for the bytes to match a TypeScript-written header.
const HEADER_KEY_ORDER: [&str; 10] = [
    "type",
    "version",
    "id",
    "createdAt",
    "cwd",
    "parentSession",
    "isSeeded",
    "origin",
    "delegationDepth",
    "agentPreset",
];

/// The physical row's required members (TS `codec.ts:21`).
const ROW_REQUIRED: [&str; 4] = ["type", "seq", "time", "data"];

/// The physical row's optional members (TS `codec.ts:22`).
const ROW_OPTIONAL: [&str; 3] = ["ignorable", "sourceEventSeqs", "surfaceOp"];

/// The event types the core vocabulary owns. A row whose type is not in this
/// set and which does not carry `ignorable: true` is refused, exactly as the
/// TypeScript current-mode admission refuses it
/// (`packages/session/session-format-v1-to-v2/src/validation.ts:69-76`).
fn is_known_type(event_type: &str) -> bool {
    super::event::CORE_EVENT_TAGS.contains(&event_type)
}

/// The event types a reader understands, on top of the core vocabulary.
///
/// The TypeScript reader takes the installed event types as an argument
/// (`knownEventTypes` in `validation.ts`): a `compaction/start` row is
/// admissible to a deployment that mounts the compaction plugin, and
/// inadmissible to one that does not. The Rust core owns only its own
/// vocabulary, so the rest is supplied by the caller.
pub trait KnownEventTypes {
    /// Whether the reader understands `event_type`.
    fn is_known(&self, event_type: &str) -> bool;
}

impl KnownEventTypes for [&str] {
    fn is_known(&self, event_type: &str) -> bool {
        self.contains(&event_type)
    }
}

impl KnownEventTypes for std::collections::BTreeSet<String> {
    fn is_known(&self, event_type: &str) -> bool {
        self.contains(event_type)
    }
}

/// One decoded physical row: the event plus the envelope metadata the core
/// does not model.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalRow {
    /// The event's `type` tag.
    pub r#type: String,
    /// The event's position in the stored artifact.
    pub seq: u64,
    /// Unix epoch milliseconds of the event.
    pub time: u64,
    /// The event payload.
    pub data: Map<String, Value>,
    /// `ignorable: true`, when the writer marked an unknown row safe to skip.
    pub ignorable: Option<bool>,
    /// Earlier source positions this event consumed, when the writer recorded
    /// them. The ranges are already expanded into a flat list.
    pub source_event_seqs: Option<Vec<u64>>,
    /// Surface placement (`append` or a `replace` range), preserved verbatim.
    pub surface_op: Option<Value>,
}

impl PhysicalRow {
    /// The row's payload as a JSON value, ready to be deserialized into a
    /// [`SessionEvent`].
    ///
    /// The physical row keeps the `type` tag at the row level and the payload
    /// under `data`, so the tag is stamped back onto the payload before it is
    /// handed to the event decoder — the Rust event vocabulary derives its
    /// shape from that tag.
    ///
    /// ## The `user/message` shape difference
    ///
    /// The TypeScript `user/message` payload *is* the message: `id`, `role`,
    /// `content`, and `source` sit directly under `data`. The Rust
    /// [`CoreEvent::UserMessage`] wraps the message under a `message` member
    /// instead. This is the one payload whose physical shape differs, so the
    /// reader lifts the flat members into the nested shape here. Every other
    /// core payload already matches — `assistant/message` and `tool/result`
    /// carry a nested `message` in TypeScript too.
    pub fn payload(&self) -> Value {
        let mut map = self.data.clone();
        if self.r#type == "user/message" && !map.contains_key("message") {
            let mut message = Map::new();
            for key in ["id", "role", "content", "source"] {
                if let Some(value) = map.remove(key) {
                    message.insert(key.to_string(), value);
                }
            }
            map.insert("message".to_string(), Value::Object(message));
        }
        map.insert("type".to_string(), json!(self.r#type));
        Value::Object(map)
    }
}

/// Decode one physical row against the core vocabulary alone.
///
/// A reader that understands only the core event types calls this; a reader
/// that also understands plugin vocabularies calls [`decode_row`] with the
/// full set.
pub fn decode_row(value: &Value, index: usize) -> Result<PhysicalRow, SessionError> {
    decode_row_with(value, index, &[] as &[&str])
}

/// Decode one physical row.
///
/// The row must carry exactly the required members plus a subset of the
/// optional ones; an unexpected member is a refusal, not a silent drop
/// (TS `codec.ts:134-139`).
///
/// The `seq` must be dense — the caller checks it against the running event
/// count, because density is a property of the *artifact*, not of one row.
///
/// `known` supplies the event types the reader understands on top of the core
/// vocabulary; see [`KnownEventTypes`].
pub fn decode_row_with<K: KnownEventTypes + ?Sized>(
    value: &Value,
    index: usize,
    known: &K,
) -> Result<PhysicalRow, SessionError> {
    let map = value.as_object().ok_or_else(|| {
        SessionError::Json(format!("released Session row {index} must be an object"))
    })?;

    for required in ROW_REQUIRED {
        if !map.contains_key(required) {
            return Err(SessionError::Json(format!(
                "released Session row {index} lacks required field {required}"
            )));
        }
    }
    for key in map.keys() {
        if !ROW_REQUIRED.contains(&key.as_str()) && !ROW_OPTIONAL.contains(&key.as_str()) {
            return Err(SessionError::Json(format!(
                "released Session row {index} has unexpected field {key}"
            )));
        }
    }

    let event_type = map
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            SessionError::Json(format!(
                "released Session row {index} type must be a string"
            ))
        })?
        .to_string();

    let seq = read_count(map.get("seq"), &format!("released Session row {index} seq"))?;
    let time = read_count(
        map.get("time"),
        &format!("released Session row {index} time"),
    )?;

    let ignorable = match map.get("ignorable") {
        None => None,
        Some(value) => {
            let flag = value.as_bool().ok_or_else(|| {
                SessionError::Json(format!(
                    "released Session row {index} ignorable must be true when present"
                ))
            })?;
            if !flag {
                return Err(SessionError::Json(format!(
                    "released Session row {index} ignorable must be true when present"
                )));
            }
            Some(true)
        }
    };

    let data = map
        .get("data")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            SessionError::Json(format!(
                "released Session row {index} data must be an object"
            ))
        })?
        .clone();

    let source_event_seqs = match map.get("sourceEventSeqs") {
        None => None,
        Some(value) => Some(decode_seq_ranges(value, seq, index)?),
    };

    // Current-mode admission: an unknown type is only acceptable when the
    // writer marked it ignorable. The TypeScript reader applies the same rule
    // against the *installed* event types; the core vocabulary is the Rust
    // equivalent of that set, and `known` carries the rest.
    if !is_known_type(&event_type) && !known.is_known(&event_type) && ignorable != Some(true) {
        return Err(SessionError::Json(format!(
            "released Session row {index} contains unknown event type {}",
            json!(event_type)
        )));
    }

    Ok(PhysicalRow {
        r#type: event_type,
        seq,
        time,
        data,
        ignorable,
        source_event_seqs,
        surface_op: map.get("surfaceOp").cloned(),
    })
}

/// Decode a `sourceEventSeqs` value into the flat list of earlier positions it
/// stands for (TS `decodeSeqRanges`).
///
/// The wire form is a list whose members are either a single position or a
/// `[start, end]` pair. The expanded list must be strictly increasing, must
/// contain no duplicate, and every member must be below the row's own `seq`.
pub fn decode_seq_ranges(
    value: &Value,
    max_entries: u64,
    index: usize,
) -> Result<Vec<u64>, SessionError> {
    let entries = value
        .as_array()
        .ok_or_else(|| SessionError::Json("sourceEventSeqs must be an array".to_string()))?;
    let mut output = Vec::new();
    let mut has_range = false;
    for entry in entries {
        let Some(range) = entry.as_array() else {
            output.push(read_count(
                Some(entry),
                &format!("released Session row {index} sourceEventSeqs member"),
            )?);
            continue;
        };
        if range.len() != 2 {
            return Err(SessionError::Json(
                "sourceEventSeqs range must be a [start, end] pair".to_string(),
            ));
        }
        let start = read_count(Some(&range[0]), "sourceEventSeqs range start")?;
        let end = read_count(Some(&range[1]), "sourceEventSeqs range end")?;
        if start > end || end >= max_entries || end - start + 1 > max_entries - output.len() as u64
        {
            return Err(SessionError::Json(
                "sourceEventSeqs range exceeds its event seq".to_string(),
            ));
        }
        for current in start..=end {
            output.push(current);
        }
        has_range = true;
    }
    let mut seen = std::collections::BTreeSet::new();
    for source in &output {
        if *source >= max_entries || !seen.insert(*source) {
            return Err(SessionError::Json(
                "sourceEventSeqs ranges must contain unique earlier seqs".to_string(),
            ));
        }
    }
    if has_range && output.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(SessionError::Json(
            "sourceEventSeqs ranges must be strictly increasing".to_string(),
        ));
    }
    Ok(output)
}

/// Encode a strictly increasing position list back into the wire form
/// (TS `encodeSeqRanges`).
///
/// A run of three or more consecutive positions becomes a `[start, end]` pair;
/// a run of exactly two becomes two singletons; anything else stays a
/// singleton.
pub fn encode_seq_ranges(values: &[u64]) -> Vec<Value> {
    let strictly_increasing = values.windows(2).all(|pair| pair[1] > pair[0]);
    if !strictly_increasing {
        return values.iter().map(|value| json!(value)).collect();
    }
    let mut output: Vec<Value> = Vec::new();
    let mut index = 0;
    while index < values.len() {
        let start = values[index];
        let mut end = start;
        while index + 1 < values.len() && values[index + 1] == end + 1 {
            index += 1;
            end += 1;
        }
        if end - start >= 2 {
            output.push(json!([start, end]));
        } else {
            output.push(json!(start));
        }
        if end - start == 1 {
            output.push(json!(end));
        }
        index += 1;
    }
    output
}

/// Decode a physical v4 header into the logical [`SessionHeader`].
///
/// The physical header carries the `type: "session"` discriminator and a
/// required `delegationDepth`; the logical header drops the discriminator and
/// keeps the rest (TS `decodePhysicalHeader`).
pub fn decode_physical_header(value: &Value) -> Result<SessionHeader, SessionError> {
    let map = value.as_object().ok_or_else(|| {
        SessionError::Json("released v4 physical header must be a JSON object".to_string())
    })?;

    if map.get("type").and_then(Value::as_str) != Some(PHYSICAL_HEADER_TYPE) {
        return Err(SessionError::Json(
            "expected released v4 physical Session header".to_string(),
        ));
    }
    let version = map
        .get("version")
        .and_then(Value::as_u64)
        .filter(|value| *value <= MAX_SAFE_INTEGER)
        .ok_or_else(|| {
            SessionError::Json(
                "Session format version must be a non-negative safe integer".to_string(),
            )
        })?;
    if version != u64::from(SESSION_FORMAT_VERSION) {
        return Err(SessionError::UnsupportedVersion {
            found: version as u32,
            current: SESSION_FORMAT_VERSION,
        });
    }

    let id = map
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| SessionError::Json("released v4 header id must be a string".to_string()))?
        .to_string();
    let created_at = read_count(map.get("createdAt"), "released v4 header createdAt")?;
    let is_seeded = map
        .get("isSeeded")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            SessionError::Json("released v4 header isSeeded must be boolean".to_string())
        })?;
    let delegation_depth = read_count(
        map.get("delegationDepth"),
        "released v4 header delegationDepth",
    )?;

    let cwd = read_optional_string(map, "cwd")?;
    let parent_session = read_optional_string(map, "parentSession")?;
    let agent_preset = read_optional_string(map, "agentPreset")?;
    let origin = match map.get("origin") {
        None => None,
        Some(value) => {
            let origin = value.as_str().ok_or_else(|| {
                SessionError::Json("released v4 header origin must be \"subagent\"".to_string())
            })?;
            if origin != "subagent" {
                return Err(SessionError::Json(
                    "released v4 header origin must be \"subagent\"".to_string(),
                ));
            }
            Some(super::types::SessionOrigin::Subagent)
        }
    };

    Ok(SessionHeader {
        version: SESSION_FORMAT_VERSION,
        id: super::types::SessionId::new(id).map_err(|_| {
            SessionError::Json("released v4 header id must be a non-empty string".to_string())
        })?,
        created_at,
        cwd,
        parent_session: parent_session
            .map(super::types::SessionId::new)
            .transpose()
            .map_err(|_| {
                SessionError::Json(
                    "released v4 header parentSession must be a non-empty string".to_string(),
                )
            })?,
        is_seeded,
        origin,
        delegation_depth: Some(delegation_depth as u32),
        agent_preset,
    })
}

/// Encode a logical [`SessionHeader`] into the physical v4 header, in the
/// canonical key order (TS `encodeHeader`).
pub fn encode_physical_header(header: &SessionHeader) -> Value {
    let mut map = Map::new();
    map.insert("type".to_string(), json!(PHYSICAL_HEADER_TYPE));
    map.insert("version".to_string(), json!(header.version));
    map.insert("id".to_string(), json!(header.id.as_str()));
    map.insert("createdAt".to_string(), json!(header.created_at));
    if let Some(cwd) = &header.cwd {
        map.insert("cwd".to_string(), json!(cwd));
    }
    if let Some(parent) = &header.parent_session {
        map.insert("parentSession".to_string(), json!(parent.as_str()));
    }
    map.insert("isSeeded".to_string(), json!(header.is_seeded));
    if header.origin.is_some() {
        map.insert("origin".to_string(), json!("subagent"));
    }
    // The physical header requires the depth even when the logical header
    // treats it as optional: a top-level session has depth zero.
    map.insert(
        "delegationDepth".to_string(),
        json!(header.delegation_depth.unwrap_or(0)),
    );
    if let Some(preset) = &header.agent_preset {
        map.insert("agentPreset".to_string(), json!(preset));
    }
    debug_assert_eq!(map.len(), HEADER_KEY_ORDER.len().min(map.len()));
    Value::Object(map)
}

/// Encode one event into its physical v4 row.
///
/// The row's `seq` is the event's position and `time` is supplied by the
/// caller: the Rust core does not keep a timestamp per event, because the
/// flat layout has nowhere to put one.
///
/// ## The `user/message` shape difference
///
/// The Rust [`CoreEvent::UserMessage`] nests the message under a `message`
/// member, while the TypeScript physical payload *is* the message. The
/// writer flattens it back, so the shipped TypeScript decoder accepts the
/// row. See [`PhysicalRow::payload`] for the reader's mirror of this.
///
/// ## Fields the shipped decoder expects but the core omits
///
/// The core flat serialization derives a message's `role` from its event tag
/// and serializes `tool/result.isError` only when it is `true`; `serde_json`
/// also writes a whole `f64` as `1.0` rather than `1`. The writer restores the
/// three so the shipped v4 decoder accepts the row:
///
/// - `role` is stamped into the message (`user/message` keeps it at the top
///   level; the other message rows carry it inside `message`), placed right
///   after `id` to match the TypeScript wire order `id, role, content, source`.
/// - `tool/result` always carries `isError`; a non-error result gains an
///   explicit `isError: false` (it is the last member of the message, matching
///   the TypeScript shape `…, toolCallId, isError`).
/// - integer-valued `f64` numbers are written as integers, mirroring
///   `JSON.stringify` (see [`normalize_integer_f64`]).
pub fn encode_physical_row(event: &SessionEvent, seq: u64, time: u64) -> Value {
    // The flat event serializes as `{ "type": "...", ...payload }`; the
    // physical row needs the payload *without* the tag under `data`.
    let flat = match serde_json::to_value(event) {
        Ok(value) => value,
        Err(_) => Value::Null,
    };
    let event_type = flat
        .as_object()
        .and_then(|map| map.get("type"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    // Build `data` from the flat event, dropping the `type` tag while preserving
    // the remaining field order. serde_json's `Map` is an `IndexMap` whose
    // `remove` is a *swap*-remove: removing the leading `type` would swap it with
    // the last field and scramble the wire bytes the TypeScript writer and
    // decoder expect. Rebuilding the map keeps the order intact.
    let mut data = Map::new();
    if let Some(map) = flat.as_object() {
        for (key, value) in map {
            if key != "type" {
                data.insert(key.clone(), value.clone());
            }
        }
    }

    // The TypeScript `user/message` payload *is* the message (no `message`
    // wrapper); the Rust core nests it. Flatten it back so the shipped decoder
    // accepts the row, preserving the field order.
    if event_type == "user/message" {
        if let Some(Value::Object(message)) = data.remove("message") {
            for (key, value) in message {
                data.insert(key, value);
            }
        }
    }

    // Restore the three fields the shipped v4 decoder expects but the core flat
    // serialization omits. See the module documentation and
    // [`shape_physical_payload`].
    shape_physical_payload(&event_type, &mut data);
    let mut data_value = Value::Object(data);
    normalize_integer_f64(&mut data_value);

    let mut row = Map::new();
    row.insert("type".to_string(), json!(event_type));
    row.insert("seq".to_string(), json!(seq));
    row.insert("time".to_string(), json!(time));
    row.insert("data".to_string(), data_value);
    Value::Object(row)
}

/// Apply the three field-shaping steps the shipped v4 decoder expects but the
/// core flat serialization omits:
///
/// - stamp `role` onto every message row (`user/message` keeps it at the top
///   level; the other message rows carry it inside `message`, placed right after
///   `id` to match the TypeScript wire order `id, role, content, source`);
/// - write `tool/result.isError: false` explicitly when the core omitted it
///   (it is the last member of the message, matching the TypeScript shape
///   `…, toolCallId, isError`);
/// - integer-valued `f64` are normalized by [`normalize_integer_f64`].
///
/// `data` is the event payload *without* its `type` tag. For `user/message` the
/// message is already flat at the top level (the caller flattens it before
/// calling); for the other message rows it is nested under `message`.
fn shape_physical_payload(event_type: &str, data: &mut Map<String, Value>) {
    match event_type {
        "user/message" => insert_role_after_id(data, "user"),
        "assistant/message" | "system/message" | "developer/message" => {
            let role = match event_type {
                "assistant/message" => "assistant",
                "system/message" => "system",
                _ => "developer",
            };
            if let Some(Value::Object(message)) = data.get_mut("message") {
                insert_role_after_id(message, role);
            }
        }
        "tool/result" => {
            if let Some(Value::Object(message)) = data.get_mut("message") {
                insert_role_after_id(message, "tool");
                // The core serializes `isError` only when true; the shipped
                // decoder requires it on every tool result, so a missing field
                // is an explicit `false`.
                if !message.contains_key("isError") {
                    message.insert("isError".to_string(), Value::Bool(false));
                }
            }
        }
        _ => {}
    }
}

/// Largest integer exactly representable as a JavaScript number; integer-valued
/// `f64` within this bound is normalized to an integer on the wire.
const JS_MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

/// Stamp `role` onto a message object immediately after its `id` member,
/// matching the TypeScript wire order `id, role, content, source, …`.
///
/// The Rust core serializes a message as `id, content, source, …` and derives
/// the role from the enclosing event tag, so the writer must add it. The `role`
/// is inserted right after `id` rather than appended, because TypeScript's
/// `MessageBase` order is `id → role → source`. If the object has no `id`, the
/// role is placed first so the row is still well formed.
fn insert_role_after_id(message: &mut Map<String, Value>, role: &str) {
    if message.contains_key("role") {
        return;
    }
    let mut rebuilt = Map::new();
    if let Some(id) = message.get("id").cloned() {
        rebuilt.insert("id".to_string(), id);
        rebuilt.insert("role".to_string(), Value::String(role.to_string()));
    }
    for (key, value) in message.iter() {
        if key != "id" && key != "role" {
            rebuilt.insert(key.clone(), value.clone());
        }
    }
    if rebuilt.get("role").is_none() {
        rebuilt.insert("role".to_string(), Value::String(role.to_string()));
    }
    *message = rebuilt;
}

/// Rewrite integer-valued `f64` numbers as integers, mirroring JavaScript's
/// `JSON.stringify` (which writes `500`, not `500.0`).
///
/// The walk is a no-op for numbers that are already integral-typed (`u64` /
/// `i64`) or non-integral, so it changes only the `1.0`-style leaves that
/// `serde_json` would otherwise emit with a trailing `.0`. Values outside
/// JavaScript's safe-integer range keep their `f64` rendering (for example
/// `1e+21`), exactly as the TypeScript writer and `dsh-retry::json::number`
/// do.
fn normalize_integer_f64(value: &mut Value) {
    match value {
        Value::Number(number) => {
            if let Some(float) = number.as_f64() {
                if !number.is_i64()
                    && !number.is_u64()
                    && float.is_finite()
                    && float.fract() == 0.0
                    && float.abs() <= JS_MAX_SAFE_INTEGER
                {
                    *value = Value::Number((float as i64).into());
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                normalize_integer_f64(item);
            }
        }
        Value::Object(map) => {
            for item in map.values_mut() {
                normalize_integer_f64(item);
            }
        }
        _ => {}
    }
}

/// Decode a TypeScript-written physical v4 log against the core vocabulary
/// alone.
///
/// A reader that understands only the core event types calls this; a reader
/// that also understands plugin vocabularies calls [`decode_physical_log_with`].
pub fn decode_physical_log(text: &str) -> Result<Session, SessionError> {
    decode_physical_log_with(text, &[] as &[&str])
}

/// Decode a TypeScript-written physical v4 log into a [`Session`].
///
/// The header must be the released v4 physical header; every event row is
/// decoded through [`decode_row_with`] and admitted into the session in order.
/// The row `seq` must be dense, matching the artifact's event count.
///
/// `known` supplies the event types the reader understands on top of the core
/// vocabulary. A deployment that mounts the compaction or retry plugin passes
/// those tags; a reader that does not know a tag must see the row refused
/// unless the writer marked it ignorable.
pub fn decode_physical_log_with<K: KnownEventTypes + ?Sized>(
    text: &str,
    known: &K,
) -> Result<Session, SessionError> {
    let mut lines = text.lines();
    let header_line = lines.next().ok_or(SessionError::EmptyLog)?;
    let header_value: Value =
        serde_json::from_str(header_line).map_err(|error| SessionError::Json(error.to_string()))?;
    let header = decode_physical_header(&header_value)?;

    let mut events: Vec<SessionEvent> = Vec::new();
    for (index, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let value: Value =
            serde_json::from_str(line).map_err(|error| SessionError::Json(error.to_string()))?;
        let row = decode_row_with(&value, index, known)?;
        if row.seq != events.len() as u64 {
            return Err(SessionError::Json(format!(
                "released Session row {index} has seq gap (expected {}, got {})",
                events.len(),
                row.seq
            )));
        }
        let event: SessionEvent = serde_json::from_value(row.payload())
            .map_err(|error| SessionError::Json(error.to_string()))?;
        events.push(event);
    }

    Session::restore(header, events)
}
/// Encode a session as a TypeScript-readable physical v4 log.
///
/// The event timestamps are not retained by the Rust core, so the caller
/// supplies one timestamp per event through [`SessionTimes`]. A session
/// without timestamps is encoded with `time: 0`, which the TypeScript reader
/// accepts (it validates the *shape* of `time`, not its value).
pub fn encode_physical_log(session: &Session, times: &[u64]) -> Result<String, SessionError> {
    let header = encode_physical_header(session.header());
    let mut text =
        serde_json::to_string(&header).map_err(|error| SessionError::Json(error.to_string()))?;
    text.push('\n');
    for (seq, event) in session.events().iter().enumerate() {
        let time = times.get(seq).copied().unwrap_or(0);
        let row = encode_physical_row(event, seq as u64, time);
        text.push_str(
            &serde_json::to_string(&row).map_err(|error| SessionError::Json(error.to_string()))?,
        );
        text.push('\n');
    }
    Ok(text)
}

/// Read a non-negative safe integer from a JSON value.
fn read_count(value: Option<&Value>, label: &str) -> Result<u64, SessionError> {
    let value = value.unwrap_or(&Value::Null);
    value
        .as_u64()
        .filter(|number| *number <= MAX_SAFE_INTEGER)
        .ok_or_else(|| SessionError::Json(format!("{label} must be a non-negative safe integer")))
}

/// Read an optional string member of a header.
fn read_optional_string(
    map: &Map<String, Value>,
    key: &str,
) -> Result<Option<String>, SessionError> {
    match map.get(key) {
        None => Ok(None),
        Some(value) => {
            let text = value.as_str().ok_or_else(|| {
                SessionError::Json(format!("released v4 header {key} must be a string"))
            })?;
            Ok(Some(text.to_string()))
        }
    }
}
