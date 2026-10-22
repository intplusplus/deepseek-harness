//! Golden-fixture comparison against the TypeScript v4 JSONL wire format.
//!
//! `rust/fixtures/v4-physical` holds logs written by the shipped TypeScript
//! encoder; `rust/fixtures/v4-event-data` holds the same rows with the v4
//! envelope removed, which is the payload shape Rust models. Provenance,
//! evidence levels, and regeneration live in `rust/fixtures/README.md`.
//!
//! Two properties are pinned here: which rows reproduce the TypeScript bytes
//! exactly, and precisely how every other row differs. A newly introduced
//! difference fails the table instead of passing silently.
//!
//! Set `DSH_FIXTURE_REPORT` to a path to have [`write_comparison_report`] emit
//! the Rust rendering of every row; `rust/fixtures/compare-rust.mjs` feeds
//! that report back through the shipped TypeScript decoder.

use std::fs;

use dsh_llm::{ContentBlock, MessageId, UserMessage};
use dsh_session::physical::encode_physical_row;
use dsh_session::{jsonl, SessionError, SessionEvent, SessionHeader, SESSION_FORMAT_VERSION};
use serde_json::{Map, Value};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");

const SCENARIOS: [&str; 4] = ["turn-tool-call", "retry", "tool-error", "extension-unicode"];

/// The projection row at `index` of `scenario`, as the shipped encoder wrote it.
fn projection_line(scenario: &str, index: usize) -> String {
    projection_lines(scenario)
        .into_iter()
        .nth(index)
        .unwrap_or_else(|| panic!("{scenario} has no projection row {index}"))
}

fn projection_lines(scenario: &str) -> Vec<String> {
    read_lines(&format!("v4-event-data/{scenario}.jsonl"))
}

fn physical_lines(scenario: &str) -> Vec<String> {
    read_lines(&format!("v4-physical/{scenario}.v4.jsonl"))
}

fn read_lines(relative: &str) -> Vec<String> {
    let path = format!("{FIXTURES}/{relative}");
    let text =
        fs::read_to_string(&path).unwrap_or_else(|error| panic!("cannot read {path}: {error}"));
    text.lines().map(str::to_owned).collect()
}

/// Rust's bytes for one flat TypeScript projection row, or the decode error.
///
/// This renders the *core flat* event — the Rust-native layout that keeps the
/// `role`, `isError`, and integer-`f64` differences the physical writer closes.
/// [`the_projection_table_covers_every_row_and_matches_the_documented_bytes`]
/// pins those differences against the TypeScript bytes.
fn rust_render(ts_line: &str) -> Result<String, String> {
    let event: SessionEvent = serde_json::from_str(ts_line).map_err(|error| error.to_string())?;
    serde_json::to_string(&event).map_err(|error| error.to_string())
}

/// Build a Rust [`SessionEvent`] from a TypeScript flat projection row.
///
/// The TypeScript `user/message` payload is flat (`id`/`role`/`content`/`source`
/// at the top level) while the Rust core nests it under `message`; we lift it
/// into the nested form before deserializing. Every other core event
/// deserializes directly. Deserializing the projection the same way
/// [`rust_render`] does keeps the field order the shipped decoder and the byte
/// comparison expect — round-tripping through a reconstructed physical payload
/// would reorder the tag and scramble the wire bytes.
fn event_from_ts_projection(ts_line: &str) -> Result<SessionEvent, String> {
    let mut value: Value = serde_json::from_str(ts_line).map_err(|error| error.to_string())?;
    if let Some(map) = value.as_object_mut() {
        if map
            .get("type")
            .and_then(Value::as_str)
            .map(|tag| tag == "user/message")
            .unwrap_or(false)
            && !map.contains_key("message")
        {
            let mut message = Map::new();
            for key in ["id", "role", "content", "source"] {
                if let Some(field) = map.remove(key) {
                    message.insert(key.to_string(), field);
                }
            }
            map.insert("message".to_string(), Value::Object(message));
        }
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

/// The bytes the physical v4 writer would emit for one TypeScript projection
/// row, expressed as a flat event so the comparison harness (which feeds a flat
/// event to the shipped decoder) sees the writer's `data`.
///
/// The physical writer closes the three differences the core flat layout keeps:
/// it stamps `role`, writes `tool/result.isError: false` explicitly, and
/// normalizes integer-valued `f64`. Routing the report through it is what makes
/// `compare-rust.mjs` accept every row the Rust writer emits.
fn rust_physical_render(ts_line: &str) -> Result<String, String> {
    let event = event_from_ts_projection(ts_line)?;
    let row = encode_physical_row(&event, 0, 0);
    let row_map = row.as_object().expect("the physical row is an object");
    let mut flat = Map::new();
    flat.insert("type".to_string(), row_map["type"].clone());
    if let Some(data) = row_map.get("data").and_then(Value::as_object) {
        for (key, value) in data {
            flat.insert(key.clone(), value.clone());
        }
    }
    serde_json::to_string(&Value::Object(flat)).map_err(|error| error.to_string())
}

#[derive(Clone, Copy)]
struct Case {
    scenario: &'static str,
    index: usize,
    event: &'static str,
    expected: Expected,
}

#[derive(Clone, Copy)]
enum Expected {
    /// Rust re-serializes the TypeScript row to the same bytes.
    Exact,
    /// Rust rejects the projection row; the fragment identifies the gap.
    Rejected(&'static str),
    /// Rust accepts the row and re-serializes it with exactly these gaps.
    Gaps(&'static [Gap]),
}

/// One documented reason the Rust bytes differ from the TypeScript bytes.
#[derive(Clone, Copy)]
enum Gap {
    /// TypeScript messages carry a `role` discriminator inside the message
    /// payload; Rust derives the role from the event tag and omits it.
    MessageRole(&'static str),
    /// TypeScript writes `isError: false`; Rust serializes the field only when true.
    FalseIsError,
    /// `serde_json` writes a whole `f64` as `1.0`; JavaScript writes `1`.
    IntegerF64,
}

/// Every projection row, in fixture order, with its observed Rust behavior.
///
/// Row 1 of each scenario is a `user/message`: Rust nests the payload under
/// `message`, so the TypeScript row does not even deserialize. That gap is
/// covered by [`a_user_message_event_wraps_its_payload_and_has_no_role`].
const CASES: &[Case] = &[
    Case {
        scenario: "turn-tool-call",
        index: 0,
        event: "turn/start",
        expected: Expected::Exact,
    },
    Case {
        scenario: "turn-tool-call",
        index: 1,
        event: "user/message",
        expected: Expected::Rejected("missing field `message`"),
    },
    Case {
        scenario: "turn-tool-call",
        index: 2,
        event: "step/start",
        expected: Expected::Exact,
    },
    Case {
        scenario: "turn-tool-call",
        index: 3,
        event: "request/header",
        expected: Expected::Gaps(&[Gap::IntegerF64]),
    },
    Case {
        scenario: "turn-tool-call",
        index: 4,
        event: "assistant/message",
        expected: Expected::Gaps(&[Gap::MessageRole("assistant")]),
    },
    Case {
        scenario: "turn-tool-call",
        index: 5,
        event: "tool/call",
        expected: Expected::Exact,
    },
    Case {
        scenario: "turn-tool-call",
        index: 6,
        event: "tool/result",
        expected: Expected::Gaps(&[Gap::MessageRole("tool"), Gap::FalseIsError]),
    },
    Case {
        scenario: "turn-tool-call",
        index: 7,
        event: "step/end",
        expected: Expected::Exact,
    },
    Case {
        scenario: "turn-tool-call",
        index: 8,
        event: "step/start",
        expected: Expected::Exact,
    },
    Case {
        scenario: "turn-tool-call",
        index: 9,
        event: "assistant/message",
        expected: Expected::Gaps(&[Gap::MessageRole("assistant")]),
    },
    Case {
        scenario: "turn-tool-call",
        index: 10,
        event: "step/end",
        expected: Expected::Exact,
    },
    Case {
        scenario: "turn-tool-call",
        index: 11,
        event: "turn/end",
        expected: Expected::Exact,
    },
    Case {
        scenario: "retry",
        index: 0,
        event: "turn/start",
        expected: Expected::Exact,
    },
    Case {
        scenario: "retry",
        index: 1,
        event: "user/message",
        expected: Expected::Rejected("missing field `message`"),
    },
    Case {
        scenario: "retry",
        index: 2,
        event: "step/start",
        expected: Expected::Exact,
    },
    Case {
        scenario: "retry",
        index: 3,
        event: "request/header",
        expected: Expected::Exact,
    },
    Case {
        scenario: "retry",
        index: 4,
        event: "assistant/attempt",
        expected: Expected::Exact,
    },
    Case {
        scenario: "retry",
        index: 5,
        event: "llm/retry",
        expected: Expected::Exact,
    },
    Case {
        scenario: "retry",
        index: 6,
        event: "llm/retry-started",
        expected: Expected::Exact,
    },
    Case {
        scenario: "retry",
        index: 7,
        event: "assistant/message",
        expected: Expected::Gaps(&[Gap::MessageRole("assistant")]),
    },
    Case {
        scenario: "retry",
        index: 8,
        event: "step/end",
        expected: Expected::Exact,
    },
    Case {
        scenario: "retry",
        index: 9,
        event: "turn/end",
        expected: Expected::Exact,
    },
    Case {
        scenario: "tool-error",
        index: 0,
        event: "turn/start",
        expected: Expected::Exact,
    },
    Case {
        scenario: "tool-error",
        index: 1,
        event: "user/message",
        expected: Expected::Rejected("missing field `message`"),
    },
    Case {
        scenario: "tool-error",
        index: 2,
        event: "step/start",
        expected: Expected::Exact,
    },
    Case {
        scenario: "tool-error",
        index: 3,
        event: "tool/call",
        expected: Expected::Exact,
    },
    Case {
        scenario: "tool-error",
        index: 4,
        event: "tool/result",
        expected: Expected::Gaps(&[Gap::MessageRole("tool")]),
    },
    Case {
        scenario: "tool-error",
        index: 5,
        event: "step/end",
        expected: Expected::Exact,
    },
    Case {
        scenario: "tool-error",
        index: 6,
        event: "turn/end",
        expected: Expected::Exact,
    },
    Case {
        scenario: "extension-unicode",
        index: 0,
        event: "turn/start",
        expected: Expected::Exact,
    },
    Case {
        scenario: "extension-unicode",
        index: 1,
        event: "user/message",
        expected: Expected::Rejected("missing field `message`"),
    },
    Case {
        scenario: "extension-unicode",
        index: 2,
        event: "step/start",
        expected: Expected::Exact,
    },
    Case {
        scenario: "extension-unicode",
        index: 3,
        event: "compaction/start",
        expected: Expected::Exact,
    },
    Case {
        scenario: "extension-unicode",
        index: 4,
        event: "compaction/summary",
        expected: Expected::Exact,
    },
    Case {
        scenario: "extension-unicode",
        index: 5,
        event: "compaction/end",
        expected: Expected::Exact,
    },
    Case {
        scenario: "extension-unicode",
        index: 6,
        event: "plugin:fixture/acme",
        expected: Expected::Exact,
    },
    Case {
        scenario: "extension-unicode",
        index: 7,
        event: "user/message",
        expected: Expected::Rejected("missing field `message`"),
    },
    Case {
        scenario: "extension-unicode",
        index: 8,
        event: "step/end",
        expected: Expected::Exact,
    },
    Case {
        scenario: "extension-unicode",
        index: 9,
        event: "turn/end",
        expected: Expected::Exact,
    },
];

/// Apply one documented gap to the TypeScript bytes.
fn apply_gap(ts_line: &str, gap: Gap) -> String {
    let (from, to) = match gap {
        Gap::MessageRole(role) => (format!("\"role\":\"{role}\","), String::new()),
        Gap::FalseIsError => (String::from(",\"isError\":false"), String::new()),
        Gap::IntegerF64 => (
            String::from("\"temperature\":1,"),
            String::from("\"temperature\":1.0,"),
        ),
    };
    assert!(
        ts_line.contains(&from),
        "fixture no longer contains `{from}`; the divergence table is stale"
    );
    ts_line.replace(&from, &to)
}

/// A short window around the first differing character, for assertion messages.
fn divergence(expected: &str, actual: &str) -> String {
    let index = expected
        .chars()
        .zip(actual.chars())
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| expected.chars().count().min(actual.chars().count()));
    let skip = index.saturating_sub(30);
    let expected: String = expected.chars().skip(skip).take(80).collect();
    let actual: String = actual.chars().skip(skip).take(80).collect();
    format!("first difference at character {index}:\n  expected: {expected}\n  actual:   {actual}")
}

#[test]
fn every_fixture_has_one_physical_row_per_projection_row_plus_the_header() {
    for scenario in SCENARIOS {
        let physical = physical_lines(scenario);
        let projection = projection_lines(scenario);
        assert_eq!(physical.len(), projection.len() + 1, "{scenario}");
        assert!(!projection.is_empty(), "{scenario}");
    }
}

#[test]
fn the_physical_header_carries_the_v4_discriminator_and_required_delegation_depth() {
    for scenario in SCENARIOS {
        let physical = physical_lines(scenario);
        let header: Value = serde_json::from_str(&physical[0]).unwrap();
        assert_eq!(header["type"], "session", "{scenario}");
        assert_eq!(
            header["version"].as_u64(),
            Some(u64::from(SESSION_FORMAT_VERSION)),
            "{scenario}"
        );
        assert!(header.get("delegationDepth").is_some(), "{scenario}");

        // Header order from session-format-v1-to-v2/src/codec.ts:164-175.
        let keys: Vec<&str> = header
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let mut expected = vec!["type", "version", "id", "createdAt"];
        for optional in ["cwd", "parentSession"] {
            if header.get(optional).is_some() {
                expected.push(optional);
            }
        }
        expected.push("isSeeded");
        if header.get("origin").is_some() {
            expected.push("origin");
        }
        expected.push("delegationDepth");
        if header.get("agentPreset").is_some() {
            expected.push("agentPreset");
        }
        assert_eq!(keys, expected, "{scenario}");
    }
}

#[test]
fn the_physical_rows_use_the_v4_envelope() {
    const ENVELOPE: [&str; 4] = ["type", "seq", "time", "data"];
    const OPTIONAL: [&str; 3] = ["ignorable", "sourceEventSeqs", "surfaceOp"];

    for scenario in SCENARIOS {
        let physical = physical_lines(scenario);
        let header: Value = serde_json::from_str(&physical[0]).unwrap();
        let created_at = header["createdAt"].as_u64().unwrap();
        for (index, line) in physical[1..].iter().enumerate() {
            let row: Value = serde_json::from_str(line).unwrap();
            let keys: Vec<&str> = row
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            assert_eq!(&keys[..ENVELOPE.len()], ENVELOPE, "{scenario} row {index}");
            assert!(
                keys.iter()
                    .all(|key| ENVELOPE.contains(key) || OPTIONAL.contains(key)),
                "{scenario} row {index} has an unexpected field: {keys:?}"
            );
            assert_eq!(
                row["seq"].as_u64(),
                Some(index as u64),
                "{scenario} row {index}"
            );
            assert_eq!(
                row["time"].as_u64(),
                Some(created_at + index as u64),
                "{scenario} row {index}"
            );
            assert!(row["data"].is_object(), "{scenario} row {index}");
        }
    }
}

#[test]
fn an_unknown_plugin_event_carries_the_ignorable_marker() {
    // KNOWN_SESSION_EVENT_TYPES excludes downstream plugin events; the reader
    // admits one only when the envelope says its omission is safe.
    let physical = physical_lines("extension-unicode");
    let row = physical[1..]
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .find(|row| row["type"] == "plugin:fixture/acme")
        .expect("the fixture has a plugin event");
    assert_eq!(row["ignorable"], true);
}

#[test]
fn rust_decodes_the_header_but_not_the_event_envelope() {
    let physical = physical_lines("turn-tool-call");

    let header: SessionHeader = serde_json::from_str(&physical[0]).unwrap();
    assert_eq!(header.version, SESSION_FORMAT_VERSION);
    assert_eq!(header.delegation_depth, Some(0));
    let rendered = serde_json::to_string(&header).unwrap();
    assert_eq!(
        rendered,
        physical[0].replacen("{\"type\":\"session\",", "{", 1),
        "the Rust header is the TypeScript header without its discriminator"
    );

    // TypeScript requires `delegationDepth` (codec.ts:19); Rust makes it optional.
    let mut without_depth: Value = serde_json::from_str(&physical[0]).unwrap();
    without_depth
        .as_object_mut()
        .unwrap()
        .remove("delegationDepth");
    let header: SessionHeader = serde_json::from_value(without_depth).unwrap();
    assert_eq!(header.delegation_depth, None);
    assert!(!serde_json::to_string(&header)
        .unwrap()
        .contains("delegationDepth"));

    let text = format!("{}\n", physical.join("\n"));
    match jsonl::decode(&text) {
        Err(SessionError::Json(message)) => {
            assert!(message.contains("missing field `turn`"), "{message}");
        }
        other => panic!("expected the v4 envelope to be rejected, got {other:?}"),
    }

    // The same first event, with the envelope removed, is exactly what Rust reads.
    let projection: Value = serde_json::from_str(&projection_line("turn-tool-call", 0)).unwrap();
    let event: SessionEvent = serde_json::from_value(projection).unwrap();
    assert_eq!(event, SessionEvent::turn_start(1));
}

#[test]
fn a_user_message_event_wraps_its_payload_and_has_no_role() {
    let event = SessionEvent::user_message(UserMessage::new(
        MessageId::new("user-1"),
        vec![ContentBlock::text("hi")],
    ));
    assert_eq!(
        serde_json::to_string(&event).unwrap(),
        concat!(
            r#"{"type":"user/message","message":{"id":"user-1","#,
            r#""content":[{"type":"text","text":"hi"}],"source":{"kind":"user"}}}"#,
        )
    );
}

#[test]
fn a_failed_tool_result_keeps_its_error_reason_outside_the_message() {
    let line = projection_line("tool-error", 4);
    assert!(line.contains(r#""isError":true"#), "{line}");
    assert!(
        line.contains(
            r#""error":{"name":"ToolError","code":"ENOENT","reason":"MISSING.md does not exist"}"#
        ),
        "{line}"
    );

    let rust = rust_render(&line).unwrap();
    assert!(rust.contains(r#""isError":true"#), "{rust}");
    assert!(
        rust.contains(
            r#""error":{"name":"ToolError","code":"ENOENT","reason":"MISSING.md does not exist"}"#
        ),
        "{rust}"
    );
}

#[test]
fn optional_markers_are_omitted_when_absent() {
    // `assistant/message.interrupted` is `true`-only (core/session/src/types.ts:349);
    // a message without the marker must not gain `interrupted: false`.
    let assistant = projection_line("turn-tool-call", 4);
    assert!(!assistant.contains("interrupted"), "{assistant}");
    let rust = rust_render(&assistant).unwrap();
    assert!(!rust.contains("interrupted"), "{rust}");

    // `ImageBlock.offloaded` is `true`-only (llm/src/types.ts:88). Removing the
    // marker from the recorded image block must round-trip without it.
    let row: Value = serde_json::from_str(&projection_line("extension-unicode", 7)).unwrap();
    let image = row["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|block| block["type"] == "image")
        .expect("the fixture has an image block")
        .clone();
    assert_eq!(image["offloaded"], true, "{image}");

    let round_trip: ContentBlock = serde_json::from_value(image.clone()).unwrap();
    assert_eq!(
        serde_json::to_string(&round_trip).unwrap(),
        serde_json::to_string(&image).unwrap()
    );

    let mut without = image;
    without.as_object_mut().unwrap().remove("offloaded");
    let decoded: ContentBlock = serde_json::from_value(without.clone()).unwrap();
    let rendered = serde_json::to_string(&decoded).unwrap();
    assert_eq!(rendered, serde_json::to_string(&without).unwrap());
    assert!(!rendered.contains("offloaded"), "{rendered}");
}

#[test]
fn the_projection_table_covers_every_row_and_matches_the_documented_bytes() {
    let mut seen = std::collections::HashSet::new();
    for case in CASES {
        assert!(
            seen.insert((case.scenario, case.index)),
            "duplicate case for {} row {}",
            case.scenario,
            case.index
        );
        let ts_line = projection_line(case.scenario, case.index);
        let row: Value = serde_json::from_str(&ts_line).unwrap();
        assert_eq!(
            row["type"], case.event,
            "{} row {}",
            case.scenario, case.index
        );

        let rust = rust_render(&ts_line);
        match case.expected {
            Expected::Exact => {
                let line = rust.unwrap_or_else(|error| {
                    panic!(
                        "{} row {} ({}) was rejected: {error}",
                        case.scenario, case.index, case.event
                    )
                });
                assert_eq!(
                    line,
                    ts_line,
                    "{} row {} ({})",
                    case.scenario,
                    case.index,
                    divergence(&ts_line, &line)
                );
            }
            Expected::Rejected(fragment) => {
                let error = rust.expect_err(&format!(
                    "{} row {} ({}) decoded, but the table says it cannot",
                    case.scenario, case.index, case.event
                ));
                assert!(
                    error.contains(fragment),
                    "{} row {}: {error}",
                    case.scenario,
                    case.index
                );
            }
            Expected::Gaps(gaps) => {
                let line = rust.unwrap_or_else(|error| {
                    panic!(
                        "{} row {} ({}) was rejected: {error}",
                        case.scenario, case.index, case.event
                    )
                });
                let expected = gaps
                    .iter()
                    .fold(ts_line.clone(), |line, gap| apply_gap(&line, *gap));
                assert_ne!(
                    expected, ts_line,
                    "{} row {}: the gap table no longer describes a difference",
                    case.scenario, case.index
                );
                assert_eq!(
                    line,
                    expected,
                    "{} row {} ({})",
                    case.scenario,
                    case.index,
                    divergence(&expected, &line)
                );
            }
        }
    }

    for scenario in SCENARIOS {
        for index in 0..projection_lines(scenario).len() {
            assert!(
                seen.contains(&(scenario, index)),
                "{scenario} row {index} is missing from the comparison table"
            );
        }
    }
}

/// Emit the Rust rendering of every row for `compare-rust.mjs`.
///
/// A plain `cargo test` does not write anything; the script sets the
/// environment variable and reads the report back.
#[test]
fn write_comparison_report() {
    let Ok(path) = std::env::var("DSH_FIXTURE_REPORT") else {
        return;
    };
    let mut scenarios = Vec::new();
    for scenario in SCENARIOS {
        let mut rows = Vec::new();
        for (index, ts_line) in projection_lines(scenario).iter().enumerate() {
            let row: Value = serde_json::from_str(ts_line).unwrap();
            let rust = match rust_physical_render(ts_line) {
                Ok(line) => serde_json::json!({ "status": "encoded", "line": line }),
                Err(message) => serde_json::json!({ "status": "rejected", "message": message }),
            };
            rows.push(serde_json::json!({
                "seq": index,
                "type": row["type"],
                "rust": rust,
            }));
        }
        scenarios.push(serde_json::json!({ "name": scenario, "rows": rows }));
    }
    let report = serde_json::json!({
        "source": "rust/crates/dsh-session/tests/fixtures.rs",
        "scenarios": scenarios,
    });
    if let Some(parent) = std::path::Path::new(&path).parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(
        &path,
        format!("{}\n", serde_json::to_string_pretty(&report).unwrap()),
    )
    .unwrap();
}
