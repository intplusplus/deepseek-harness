//! End-to-end evidence for the v0 → v4 chain over real fixture files.
//!
//! Two fixtures, two claims (see `fixtures/README.md`):
//!
//! 1. The verbatim released-v0 log is **refused**, with the same diagnosis the
//!    TypeScript reader produces. Accepting it would mean the chain invented
//!    chronology to place a system head.
//! 2. A normalized variant of the same scenario migrates all the way to v4, and
//!    the result satisfies the structural invariants of the Rust vocabulary
//!    independently of the golden file that locks its bytes.

use dsh_session::{decode, encode, CoreEvent, Session, SessionEvent};

fn fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{path}: {error}"))
}

/// The object's keys in serialization order (`serde_json` keeps insertion order).
fn keys(value: &serde_json::Value) -> Vec<&str> {
    value
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect()
}

#[test]
fn the_frozen_released_v0_fixture_is_refused() {
    let error = decode(&fixture("released-v0-real-shapes.jsonl"))
        .expect_err("a surface before the first step cannot acquire a system head");
    let message = error.to_string();
    assert!(
        message.contains("surface before first step"),
        "expected the TypeScript diagnosis, got: {message}"
    );
}

#[test]
fn the_normalized_v0_fixture_migrates_to_v4() {
    let session = decode(&fixture("v0-synthetic-normalized.jsonl")).expect("chain accepts it");

    // Header: version pinned, physical `type` dropped, seeded flag materialized.
    assert_eq!(session.header().version, 4);
    let header = serde_json::to_value(session.header()).unwrap();
    assert!(
        header.get("type").is_none(),
        "physical `type` must be dropped"
    );
    assert_eq!(header["isSeeded"], serde_json::json!(false));

    // Flat rows: every event starts with its tag and parses back into the
    // vocabulary. Seq is positional in this crate, so density is structural —
    // `Session::restore` already rejects a gap, and it accepted this log.
    let rows: Vec<String> = encode(&session)
        .unwrap()
        .lines()
        .skip(1)
        .map(str::to_string)
        .collect();
    for (index, row) in rows.iter().enumerate() {
        assert!(
            row.starts_with("{\"type\":\""),
            "row {index} does not open with its discriminator: {row}"
        );
        let _: SessionEvent = serde_json::from_str(row)
            .unwrap_or_else(|error| panic!("row {index} does not parse: {error}\n{row}"));
    }

    // The streamed system prompt became the surface head, before any message.
    let first_surface = session
        .events()
        .iter()
        .find(|event| {
            matches!(
                event.core(),
                Some(
                    CoreEvent::SystemMessage { .. }
                        | CoreEvent::UserMessage { .. }
                        | CoreEvent::AssistantMessage { .. }
                )
            )
        })
        .expect("a surface event exists");
    assert!(
        matches!(first_surface.core(), Some(CoreEvent::SystemMessage { .. })),
        "node 0 must be the promoted system head, got {first_surface:?}"
    );

    // `role` is derived from the tag in Rust, so it must not survive on the wire.
    for event in session.events() {
        let value = serde_json::to_value(event).unwrap();
        if let Some(message) = value.get("message") {
            assert!(message.get("role").is_none(), "role survived: {value}");
        }
    }

    // The retry pair keeps its TypeScript declaration order
    // (`packages/llm/llm-retry/src/types.ts`).
    let retry = rows
        .iter()
        .find(|row| row.starts_with("{\"type\":\"llm/retry\""))
        .expect("the fixture has a retry");
    let retry: serde_json::Value = serde_json::from_str(retry).unwrap();
    assert_eq!(
        keys(&retry),
        [
            "type",
            "retryId",
            "turn",
            "step",
            "provider",
            "mode",
            "policyKey",
            "retry",
            "maxRetries",
            "delayMs",
            "failure",
        ]
    );

    // Message field order follows `packages/llm/llm/src/message.ts`: the base
    // interface is `id`, `content`, and a specialization appends `source` after
    // the (stripped) `role` discriminator.
    let system = session
        .events()
        .iter()
        .find_map(|event| match event.core() {
            Some(CoreEvent::SystemMessage { message, .. }) => Some(message.as_ref()),
            _ => None,
        })
        .expect("the promoted system head exists");
    let system = serde_json::to_value(system).unwrap();
    assert_eq!(keys(&system), ["id", "content", "source"]);
    assert_eq!(system["source"]["kind"], serde_json::json!("system-prompt"));
    assert_eq!(
        system["content"],
        serde_json::json!([]),
        "an empty prompt sends no content blocks"
    );

    // `assistant/message` keeps `turn, step, message, stream, usage`
    // (`packages/core/session/src/types.ts`).
    let assistant_event = rows
        .iter()
        .find(|row| row.starts_with("{\"type\":\"assistant/message\""))
        .expect("the fixture has an assistant message");
    let assistant_event: serde_json::Value = serde_json::from_str(assistant_event).unwrap();
    assert_eq!(
        keys(&assistant_event),
        ["type", "turn", "step", "message", "stream", "usage"]
    );

    // Streamed chunks fold into one assistant message carrying the run list.
    let assistant = session
        .events()
        .iter()
        .find_map(|event| match event.core() {
            Some(CoreEvent::AssistantMessage { stream, .. }) => Some(stream),
            _ => None,
        })
        .expect("the fixture has an assistant message");
    assert!(
        assistant.len() > 1,
        "the chunk run should fold into several stream records, got {}",
        assistant.len()
    );
}

#[test]
fn the_migrated_rows_match_the_recorded_golden() {
    let session = decode(&fixture("v0-synthetic-normalized.jsonl")).expect("chain accepts it");
    let produced = encode(&session).unwrap();
    let golden_path = format!(
        "{}/tests/fixtures/v0-synthetic-normalized.v4.jsonl",
        env!("CARGO_MANIFEST_DIR")
    );
    // `DSH_UPDATE_GOLDEN=1` rewrites the recorded output. It is a regression
    // lock, not an oracle: read fixtures/README.md first, and only regenerate
    // after confirming the new shape against the TypeScript declarations.
    if std::env::var("DSH_UPDATE_GOLDEN").is_ok_and(|value| value == "1") {
        std::fs::write(&golden_path, &produced).unwrap();
        return;
    }
    let golden = std::fs::read_to_string(&golden_path).unwrap();
    assert_eq!(
        produced.trim_end(),
        golden.trim_end(),
        "the chain's output changed; see fixtures/README.md before updating the golden"
    );
}

#[test]
fn a_current_generation_log_round_trips_without_the_chain() {
    // Guard against the chain being applied to an already-current log.
    let session = decode(&fixture("v0-synthetic-normalized.jsonl")).unwrap();
    let encoded = encode(&session).unwrap();
    let reloaded: Session = decode(&encoded).expect("v4 decodes directly");
    assert_eq!(reloaded, session);
}
