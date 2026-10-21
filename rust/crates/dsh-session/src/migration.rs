//! Historical session format migration chain (v0 -> v1 -> v2 -> v3 -> v4).
//!
//! The TypeScript harness kept four generations of the session wire format.
//! Any older log (v0-v3) can be loaded and normalized up to the current
//! [`SESSION_FORMAT_VERSION`] before it is folded into a [`Session`][crate::log::Session].
//!
//! Each step mirrors one TypeScript package (`session-format-v0-to-v1` ...
//! `session-format-v3-to-v4`); the comments name the rule being transcribed so
//! the two readers can be diffed. The work happens at the JSON `Value` level
//! because the envelope (`type`/`seq`/`time`/`data`) is a physical concern: the
//! Rust core models only the flattened payload, so the chain produces flat
//! events and lets [`Session::restore`][crate::log::Session::restore] re-admit
//! them by position.
//!
//! # Coverage
//!
//! | Edge | Implemented | Not implemented (reason) |
//! |------|-------------|--------------------------|
//! | v0->v1 | type renames, steering collapse, turn reason shape, `messagePrefix` removal, retry/compaction identity synthesis, legacy message/provenance/`tool/result` wrapping, exact payload inventories, payload semantics | deep nested validation of `session-reference`, `skill-catalog`, `agent-instructions`, `webhook`, team and workflow payloads (owner-owned shapes; only the discriminant is admitted); cross-event relationship auditing (turn/step nesting, tool lifecycle, surface spans) |
//! | v1->v2 | attempt grouping, chunk packing, `assistant/message` stream attachment, `goal/change` split, legacy interrupted-turn restart, inherited-cut insertion, reference remapping | compact runs are expanded and re-packed rather than carried through (`SessionFormatEventRun` has no Rust counterpart) |
//! | v2->v3 | system-prompt promotion with the TypeScript SHA-256 identity, PTC renames, empty header field canonicalization, reference remapping | none |
//! | v3->v4 | tool-role lift, plugin-source producer rewrite, content-tag namespacing, opaque-event namespacing, interrupted-turn insertion, catalog facts from explicit child evidence | reading child sessions from storage (`v3_to_v4` takes the evidence as a slice); the full `relationships.ts` audit and `fork-result` validation |
//!
//! Everything left out is a *validation* rule that needs either a complete
//! artifact view or storage access. None of them change the emitted events, so
//! a migrated log is still byte-compatible with the TypeScript chain for the
//! rules that are implemented.
//!
//! # Rust-specific finalization
//!
//! Two normalizations happen at the very end, because the Rust `SessionEvent`
//! vocabulary differs slightly from the TypeScript wire shape:
//!
//! 1. TypeScript leaves `user/message` *inline* (its payload **is** the
//!    message); [`CoreEvent::UserMessage`][crate::event::CoreEvent] wraps it
//!    under `message`.
//! 2. TypeScript messages carry a `role` discriminator that the Rust message
//!    types derive from the event tag; roles are stripped so the migrated event
//!    round-trips losslessly through [`Session::append`][crate::log::Session::append].

#![forbid(unsafe_code)]

use serde_json::{json, Map, Value};

use super::types::{SessionError, SESSION_FORMAT_VERSION};

mod support;
mod v0_v1;
mod v1_v2;
mod v2_v3;
mod v3_v4;

pub use v0_v1::v0_to_v1;
pub use v1_v2::v1_to_v2;
pub use v2_v3::v2_to_v3;
pub use v3_v4::v3_to_v4;

use support::{count, fail, object, string};

/// Physical row types that pack one Assistant delta run into one JSONL line.
const PACKED_TAGS: [&str; 3] = ["text-chunks", "reasoning-chunks", "tool-call-chunks"];

/// Event-envelope members every physical row may carry.
const ENVELOPE_REQUIRED: [&str; 4] = ["type", "seq", "time", "data"];
const ENVELOPE_OPTIONAL: [&str; 3] = ["ignorable", "sourceEventSeqs", "surfaceOp"];

/// One stored session event together with its physical envelope.
///
/// `data` is the event payload exactly as the TypeScript `data` field carried
/// it (or, for a Rust-native flat log, the whole event minus its `type` tag).
/// The envelope metadata is surface-layer bookkeeping the Rust core does not
/// model, so it is dropped on the way out, but the chain needs it while it
/// renumbers events.
#[derive(Clone, Debug)]
pub struct Envelope {
    /// The event's `type` tag.
    pub r#type: String,
    /// The event's position in the source artifact.
    pub seq: u64,
    /// Unix epoch milliseconds of the event.
    pub time: u64,
    /// The event payload.
    pub data: Map<String, Value>,
    /// `ignorable: true`, when the writer marked an unknown row safe to skip.
    pub ignorable: Option<bool>,
    /// Earlier source positions this event consumed, when the writer recorded them.
    pub source_event_seqs: Option<Vec<u64>>,
    /// Surface placement (`append` or a `replace` range).
    pub surface_op: Option<Value>,
    /// Whether the row came from the physical JSONL framing.
    ///
    /// A flat Rust-native row carries no surface metadata, so envelope rules
    /// that depend on `surfaceOp`/`sourceEventSeqs` only apply to physical rows.
    pub physical: bool,
}

impl Envelope {
    /// Build one migrated envelope at an explicit position.
    fn at(r#type: impl Into<String>, seq: u64, time: u64, data: Map<String, Value>) -> Self {
        Self {
            r#type: r#type.into(),
            seq,
            time,
            data,
            ignorable: None,
            source_event_seqs: None,
            surface_op: None,
            physical: false,
        }
    }
}

/// The format version stamped in a header value, if present.
pub fn detect_version(header: &Value) -> Option<u32> {
    header
        .get("version")
        .and_then(Value::as_u64)
        .map(|version| version as u32)
}

/// One stored log: a logical header plus its events, in source coordinates.
#[derive(Clone, Debug)]
pub struct Log {
    /// The logical header for the log's stored version.
    pub header: Value,
    /// The events, in stored order.
    pub events: Vec<Envelope>,
    /// The fork-inherited prefix length in the log's own coordinates.
    pub cut: u64,
}

impl Log {
    /// Build a log from an already-decoded header and events.
    pub fn new(header: Value, events: Vec<Envelope>, cut: u64) -> Self {
        Self {
            header,
            events,
            cut,
        }
    }

    /// The stored format version.
    pub fn version(&self) -> Result<u32, SessionError> {
        detect_version(&self.header).ok_or_else(|| {
            SessionError::Json("session header is missing a format version".to_string())
        })
    }

    /// Parse one stored log: the header line, then one physical row per line.
    ///
    /// Empty lines are ignored. A future generation is refused before any event
    /// is decoded, exactly as the TypeScript catalog classifies headers first.
    pub fn parse(text: &str) -> Result<Self, SessionError> {
        let mut lines = text.lines();
        let header_line = lines.next().ok_or(SessionError::EmptyLog)?;
        if header_line.trim().is_empty() {
            return Err(SessionError::EmptyLog);
        }
        let raw: Value = serde_json::from_str(header_line)
            .map_err(|error| SessionError::Json(error.to_string()))?;
        let version = detect_version(&raw).ok_or_else(|| {
            SessionError::Json("session header is missing a format version".to_string())
        })?;
        if version > SESSION_FORMAT_VERSION {
            return Err(SessionError::UnsupportedVersion {
                found: version,
                current: SESSION_FORMAT_VERSION,
            });
        }
        let (header, seed_length, is_seeded) = decode_header(&raw, version)?;
        let mut events = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let index = events.len();
            events.extend(parse_event_line(line, index)?);
        }
        let cut = if version <= 1 {
            seed_length
        } else {
            inherited_cut(&events, is_seeded)?
        };
        if cut > events.len() as u64 {
            return Err(fail(
                "Session inheritedEventCount exceeds its event count".to_string(),
            ));
        }
        Ok(Self {
            header,
            events,
            cut,
        })
    }
}

/// Migrate one stored log to the current generation.
///
/// `children` is the explicit historical child evidence the v3->v4 edge needs;
/// an empty slice declares a parent without children.
pub fn migrate_session(log: Log, children: &[Value]) -> Result<(Value, Vec<Value>), SessionError> {
    let version = log.version()?;
    if version > SESSION_FORMAT_VERSION {
        return Err(SessionError::UnsupportedVersion {
            found: version,
            current: SESSION_FORMAT_VERSION,
        });
    }
    if version == SESSION_FORMAT_VERSION {
        let mut header = log.header;
        finalize_header(&mut header);
        return Ok((header, log.events.into_iter().map(emit_flat).collect()));
    }

    let mut log = log;
    if version == 0 {
        log = v0_to_v1(log)?;
    }
    if version <= 1 {
        log = v1_to_v2(log)?;
    }
    if version <= 2 {
        log = v2_to_v3(log)?;
    }
    log = v3_to_v4(log, children)?;

    let mut header = log.header;
    finalize_header(&mut header);
    Ok((header, log.events.into_iter().map(emit_flat).collect()))
}

/// Pin the current version and guarantee the members the Rust header requires.
fn finalize_header(header: &mut Value) {
    if let Some(map) = header.as_object_mut() {
        map.shift_remove("type");
        map.insert("version".to_string(), json!(SESSION_FORMAT_VERSION));
        if !map.contains_key("isSeeded") {
            map.insert("isSeeded".to_string(), json!(false));
        }
    }
}

/// Flatten one migrated envelope into a Rust-native flat event.
fn emit_flat(envelope: Envelope) -> Value {
    let mut object = Map::new();
    object.insert("type".to_string(), Value::String(envelope.r#type.clone()));
    let mut data = envelope.data;
    data.shift_remove("ignorable");
    if envelope.r#type == "user/message" {
        // TypeScript keeps the message inline; Rust wraps it.
        data.shift_remove("role");
        object.insert("message".to_string(), Value::Object(data));
        return Value::Object(object);
    }
    // The role of a wrapped message is derived from the event tag.
    if let Some(message) = data.get_mut("message").and_then(Value::as_object_mut) {
        message.shift_remove("role");
    }
    object.extend(data);
    Value::Object(object)
}

// ---------------------------------------------------------------------------
// Header decoding
// ---------------------------------------------------------------------------

/// Decode one stored header into the canonical logical header for `version`.
///
/// Returns the logical header, the physical fork seed length (v0/v1 only), and
/// whether the header declares a seeded prefix.
fn decode_header(raw: &Value, version: u32) -> Result<(Value, u64, bool), SessionError> {
    let map = object(raw, "Session header")?;
    if let Some(tag) = map.get("type") {
        if tag.as_str() != Some("session") {
            return Err(fail("Session header type must be `session`".to_string()));
        }
    }
    let required: &[&str] = if version <= 1 {
        &["version", "id", "createdAt", "delegationDepth"]
    } else if version == SESSION_FORMAT_VERSION {
        &["version", "id", "createdAt", "isSeeded"]
    } else {
        &["version", "id", "createdAt", "isSeeded", "delegationDepth"]
    };
    // `seedLength` belongs to the v0/v1 physical framing only.
    let mut optional: Vec<&str> = vec![
        "type",
        "cwd",
        "parentSession",
        "isSeeded",
        "origin",
        "agentPreset",
    ];
    if version <= 1 {
        optional.push("seedLength");
    }
    if version >= 2 {
        optional.push("delegationDepth");
    }
    support::keys(
        map,
        required,
        &optional,
        &format!("format v{version} header"),
    )?;
    if count(map.get("version").unwrap_or(&Value::Null), "header version")? != u64::from(version) {
        return Err(fail(format!("expected format v{version} header")));
    }
    let id = string(
        map.get("id")
            .ok_or_else(|| fail("header id must be a string"))?,
        "header id",
    )?;
    let created_at = count(
        map.get("createdAt").unwrap_or(&Value::Null),
        "header createdAt",
    )?;
    let delegation_depth = match map.get("delegationDepth") {
        Some(depth) => Some(count(depth, "header delegationDepth")?),
        None => None,
    };
    let seed_length = match map.get("seedLength") {
        Some(seed) => count(seed, "header seedLength")?,
        None => 0,
    };
    let is_seeded = match map.get("isSeeded") {
        Some(value) => support::boolean(value, "header isSeeded")?,
        None if version <= 1 => map.contains_key("seedLength"),
        None => return Err(fail("header isSeeded must be a boolean".to_string())),
    };
    if let Some(cwd) = map.get("cwd") {
        let cwd = string(cwd, "header cwd")?;
        if !is_absolute_like_typescript(cwd) {
            return Err(fail("header cwd must be absolute".to_string()));
        }
    }
    if let Some(parent) = map.get("parentSession") {
        string(parent, "header parentSession")?;
    }
    if let Some(origin) = map.get("origin") {
        if origin.as_str() != Some("subagent") {
            return Err(fail("header origin must be `subagent`".to_string()));
        }
    }
    if let Some(preset) = map.get("agentPreset") {
        string(preset, "header agentPreset")?;
    }

    // Canonical member order, matching the TypeScript logical header.
    let mut header = Map::new();
    header.insert("version".to_string(), json!(version));
    header.insert("id".to_string(), Value::String(id.to_string()));
    header.insert("createdAt".to_string(), json!(created_at));
    if let Some(cwd) = map.get("cwd") {
        header.insert("cwd".to_string(), cwd.clone());
    }
    if let Some(parent) = map.get("parentSession") {
        header.insert("parentSession".to_string(), parent.clone());
    }
    header.insert("isSeeded".to_string(), json!(is_seeded));
    if let Some(origin) = map.get("origin") {
        header.insert("origin".to_string(), origin.clone());
    }
    if let Some(depth) = delegation_depth {
        header.insert("delegationDepth".to_string(), json!(depth));
    }
    if let Some(preset) = map.get("agentPreset") {
        header.insert("agentPreset".to_string(), preset.clone());
    }
    Ok((Value::Object(header), seed_length, is_seeded))
}

/// `node:path.isAbsolute` for the paths a session records.
fn is_absolute_like_typescript(path: &str) -> bool {
    if path.starts_with('/') || path.starts_with('\\') {
        return true;
    }
    let bytes = path.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
}

/// The fork-inherited cut named by the last `session/end-seed` marker.
fn inherited_cut(events: &[Envelope], is_seeded: bool) -> Result<u64, SessionError> {
    let marker = events
        .iter()
        .filter(|event| event.r#type == "session/end-seed")
        .filter(|event| event.data.get("inherited").and_then(Value::as_bool) == Some(true))
        .map(|event| event.seq)
        .next_back();
    match (is_seeded, marker) {
        (true, Some(seq)) => Ok(seq),
        (true, None) => Err(fail(
            "seeded Session lacks an inherited end-seed marker".to_string(),
        )),
        (false, Some(_)) => Err(fail(
            "unseeded Session contains an inherited end-seed marker".to_string(),
        )),
        (false, None) => Ok(0),
    }
}

// ---------------------------------------------------------------------------
// Row parsing
// ---------------------------------------------------------------------------

/// Parse one stored row into one or more envelopes.
///
/// A TypeScript-physical line carries `{type, seq, time, data, ...}`; a
/// Rust-native flat line is the event itself (`{type, ...payload}`). A packed
/// Assistant delta run expands into one envelope per member so every later
/// stage sees logical events only.
pub fn parse_event_line(line: &str, index: usize) -> Result<Vec<Envelope>, SessionError> {
    let value: Value =
        serde_json::from_str(line).map_err(|error| SessionError::Json(error.to_string()))?;
    let map = object(&value, "Session event")?;
    let event_type = string(
        map.get("type").ok_or_else(|| {
            SessionError::Json("session event is missing a `type` tag".to_string())
        })?,
        "Session event type",
    )?
    .to_string();

    if !map.contains_key("data") {
        let mut data = map.clone();
        data.shift_remove("type");
        return Ok(vec![Envelope {
            r#type: event_type,
            seq: index as u64,
            time: 0,
            data,
            ignorable: None,
            source_event_seqs: None,
            surface_op: None,
            physical: false,
        }]);
    }

    if PACKED_TAGS.contains(&event_type.as_str()) {
        return expand_packed_run(map, &event_type);
    }

    support::keys(
        map,
        &ENVELOPE_REQUIRED,
        &ENVELOPE_OPTIONAL,
        &format!("released Session row {index}"),
    )?;
    let seq = count(
        map.get("seq").unwrap_or(&Value::Null),
        &format!("released Session row {index} seq"),
    )?;
    let time = count(
        map.get("time").unwrap_or(&Value::Null),
        &format!("released Session row {index} time"),
    )?;
    let data = object(
        map.get("data").unwrap_or(&Value::Null),
        &format!("released Session row {index} data"),
    )?
    .clone();
    let ignorable = match map.get("ignorable") {
        Some(value) => Some(support::boolean(
            value,
            &format!("released Session row {index} ignorable"),
        )?),
        None => None,
    };
    if ignorable == Some(false) {
        return Err(fail(format!(
            "released Session row {index} ignorable must be true when present"
        )));
    }
    let source_event_seqs = match map.get("sourceEventSeqs") {
        Some(value) => Some(decode_source_event_seqs(value, seq)?),
        None => None,
    };
    Ok(vec![Envelope {
        r#type: event_type,
        seq,
        time,
        data,
        ignorable,
        source_event_seqs,
        surface_op: map.get("surfaceOp").cloned(),
        physical: true,
    }])
}

/// Expand the three packed Assistant row shapes into per-chunk envelopes.
fn expand_packed_run(
    map: &Map<String, Value>,
    event_type: &str,
) -> Result<Vec<Envelope>, SessionError> {
    let label = format!("released {event_type} row");
    support::keys(map, &["type", "seq0", "time0", "data"], &[], &label)?;
    let seq0 = count(
        map.get("seq0").unwrap_or(&Value::Null),
        &format!("{label} seq0"),
    )?;
    let time0 = count(
        map.get("time0").unwrap_or(&Value::Null),
        &format!("{label} time0"),
    )?;
    let data = object(
        map.get("data").unwrap_or(&Value::Null),
        &format!("{label} data"),
    )?;
    let is_tool = event_type == "tool-call-chunks";
    let required: &[&str] = if is_tool {
        &["turn", "step", "index", "id", "dt", "args"]
    } else {
        &["turn", "step", "index", "dt", "texts"]
    };
    support::keys(
        data,
        required,
        if is_tool { &["name"] } else { &[] },
        &format!("{label} data"),
    )?;
    let members = support::field_array(data, if is_tool { "args" } else { "texts" }, &label)?;
    if members.is_empty() {
        return Err(fail(format!("{label} payload must be a non-empty array")));
    }
    for member in members {
        string(member, &format!("{label} payload member"))?;
    }
    let gaps = support::field_array(data, "dt", &label)?;
    if gaps.len() + 1 != members.len() {
        return Err(fail(format!("{label} dt length must match its payload")));
    }
    let turn = count(
        data.get("turn").unwrap_or(&Value::Null),
        &format!("{label} turn"),
    )?;
    let step = count(
        data.get("step").unwrap_or(&Value::Null),
        &format!("{label} step"),
    )?;
    let chunk_index = count(
        data.get("index").unwrap_or(&Value::Null),
        &format!("{label} index"),
    )?;
    let id = if is_tool {
        Some(
            support::non_empty(
                data.get("id").unwrap_or(&Value::Null),
                &format!("{label} id"),
            )?
            .to_string(),
        )
    } else {
        None
    };
    let name = if is_tool {
        match data.get("name") {
            Some(value) => Some(string(value, &format!("{label} name"))?.to_string()),
            None => None,
        }
    } else {
        None
    };

    let mut envelopes = Vec::with_capacity(members.len());
    let mut time = time0;
    for (offset, member) in members.iter().enumerate() {
        if offset > 0 {
            time += count(&gaps[offset - 1], &format!("{label} dt member"))?;
        }
        let chunk = if is_tool {
            let mut chunk = Map::new();
            chunk.insert("type".to_string(), json!("tool-call-delta"));
            chunk.insert("index".to_string(), json!(chunk_index));
            chunk.insert("id".to_string(), json!(id));
            if let Some(name) = &name {
                chunk.insert("name".to_string(), json!(name));
            }
            chunk.insert("argumentsDelta".to_string(), member.clone());
            Value::Object(chunk)
        } else {
            let mut chunk = Map::new();
            chunk.insert(
                "type".to_string(),
                json!(if event_type == "text-chunks" {
                    "text-delta"
                } else {
                    "reasoning-delta"
                }),
            );
            chunk.insert("index".to_string(), json!(chunk_index));
            chunk.insert("text".to_string(), member.clone());
            Value::Object(chunk)
        };
        let mut payload = Map::new();
        payload.insert("turn".to_string(), json!(turn));
        payload.insert("step".to_string(), json!(step));
        payload.insert("chunk".to_string(), chunk);
        envelopes.push(Envelope::at(
            "assistant/chunk",
            seq0 + offset as u64,
            time,
            payload,
        ));
    }
    Ok(envelopes)
}

/// Decode `[seq, ...]` or `[[start, end], ...]` source-event references.
fn decode_source_event_seqs(value: &Value, max_entries: u64) -> Result<Vec<u64>, SessionError> {
    let entries = value
        .as_array()
        .ok_or_else(|| fail("sourceEventSeqs must be an array".to_string()))?;
    let mut output: Vec<u64> = Vec::new();
    let mut has_range = false;
    for entry in entries {
        if let Some(member) = entry.as_u64() {
            if output.len() as u64 >= max_entries {
                return Err(fail("sourceEventSeqs exceeds its event seq".to_string()));
            }
            output.push(member);
            continue;
        }
        let pair = entry
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| fail("sourceEventSeqs range must be a [start, end] pair".to_string()))?;
        let start = count(&pair[0], "sourceEventSeqs range start")?;
        let end = count(&pair[1], "sourceEventSeqs range end")?;
        if end < start || end - start + 1 > max_entries - output.len() as u64 {
            return Err(fail(
                "sourceEventSeqs range exceeds its event seq".to_string(),
            ));
        }
        for seq in start..=end {
            output.push(seq);
        }
        has_range = true;
    }
    if output.iter().any(|seq| *seq >= max_entries) {
        return Err(fail("sourceEventSeqs must name earlier events".to_string()));
    }
    if has_range && output.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(fail(
            "sourceEventSeqs ranges must be strictly increasing".to_string(),
        ));
    }
    let mut seen: Vec<u64> = Vec::with_capacity(output.len());
    for seq in &output {
        if seen.contains(seq) {
            return Err(fail("sourceEventSeqs must contain unique seqs".to_string()));
        }
        seen.push(*seq);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{detect_version, parse_event_line, Log};
    use serde_json::json;

    #[test]
    fn detect_version_reads_only_the_version_member() {
        assert_eq!(detect_version(&json!({ "version": 2 })), Some(2));
        assert_eq!(detect_version(&json!({ "type": "session" })), None);
    }

    #[test]
    fn a_flat_row_takes_its_position_from_the_index() {
        let rows = parse_event_line(r#"{"type":"turn/start","turn":1}"#, 3).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].seq, 3);
        assert_eq!(rows[0].data["turn"], 1);
        assert!(!rows[0].physical);
    }

    #[test]
    fn a_packed_row_expands_into_every_delta() {
        let row = r#"{"type":"text-chunks","seq0":4,"time0":10,"data":{"turn":1,"step":1,"index":0,"dt":[2,3],"texts":["a","b","c"]}}"#;
        let rows = parse_event_line(row, 0).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].seq, 4);
        assert_eq!(rows[1].time, 12);
        assert_eq!(rows[2].time, 15);
        assert_eq!(rows[2].data["chunk"]["text"], "c");
    }

    #[test]
    fn a_header_without_a_version_is_refused() {
        assert!(Log::parse("{\"id\":\"s\"}\n").is_err());
    }

    #[test]
    fn a_future_generation_is_refused_before_the_body() {
        let error = Log::parse(
            "{\"version\":5,\"id\":\"s\",\"createdAt\":0,\"isSeeded\":false,\"delegationDepth\":0}\n",
        )
        .unwrap_err();
        assert!(matches!(
            error,
            crate::types::SessionError::UnsupportedVersion { found: 5, .. }
        ));
    }

    #[test]
    fn an_empty_log_is_refused() {
        assert!(matches!(
            Log::parse(""),
            Err(crate::types::SessionError::EmptyLog)
        ));
    }
}
