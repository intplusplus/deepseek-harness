//! Golden-fixture checks for the durable retry events.
//!
//! `rust/fixtures/v4-event-data/retry.jsonl` was written by the shipped
//! TypeScript encoder. These tests decode the TypeScript bytes into the Rust
//! vocabulary and re-encode them byte for byte, so the hand-written field order
//! in `events.rs` is checked against the real producer rather than a
//! hand-built expectation.

use std::fs;

use dsh_retry::{
    decode_retry, decode_retry_started, encode_retry, encode_retry_started, LlmRetryEvent,
    RetryMode,
};
use dsh_session::SessionEvent;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/v4-event-data/retry.jsonl"
);

fn fixture_rows() -> Vec<String> {
    let text = fs::read_to_string(FIXTURE)
        .unwrap_or_else(|error| panic!("cannot read {FIXTURE}: {error}"));
    text.lines().map(str::to_owned).collect()
}

fn row_of(tag: &str) -> String {
    fixture_rows()
        .into_iter()
        .find(|line| line.starts_with(&format!("{{\"type\":\"{tag}\"")))
        .unwrap_or_else(|| panic!("the fixture has no {tag} row"))
}

#[test]
fn the_scheduled_retry_re_encodes_to_the_typescript_bytes() {
    let ts_line = row_of("llm/retry");
    let event: SessionEvent = serde_json::from_str(&ts_line).unwrap();
    let retry = decode_retry(&event)
        .expect("the row is not llm/retry")
        .unwrap();
    assert_eq!(retry.mode(), RetryMode::Normal);

    let LlmRetryEvent::Normal {
        retry_id,
        turn,
        step,
        provider,
        policy_key,
        retry: attempt,
        max_retries,
        delay_ms,
        failure,
    } = &retry
    else {
        panic!("the fixture is a normal-mode retry");
    };
    assert_eq!(retry_id.as_str(), "retry-1");
    assert_eq!((*turn, *step, *attempt, *max_retries), (1, 1, 1, 2));
    assert_eq!(provider, "mock");
    assert_eq!(
        policy_key,
        "[\"normal\",2,[\"RATE_LIMIT\",\"SERVER\"],1.5,4,0]"
    );
    assert_eq!(*delay_ms, 1.5);
    assert_eq!(failure.message, "provider rate limited");
    assert_eq!(failure.code, "RATE_LIMIT");
    assert_eq!(failure.status, Some(429));
    assert_eq!(failure.provider_retry_after_ms, Some(2000));

    let encoded = serde_json::to_string(&encode_retry(&retry).unwrap()).unwrap();
    assert_eq!(encoded, ts_line);
}

#[test]
fn the_started_retry_re_encodes_to_the_typescript_bytes() {
    let ts_line = row_of("llm/retry-started");
    let event: SessionEvent = serde_json::from_str(&ts_line).unwrap();
    let started = decode_retry_started(&event)
        .expect("the row is not llm/retry-started")
        .unwrap();

    assert_eq!(started.retry_id.as_str(), "retry-1");
    assert_eq!((started.turn, started.step, started.retry), (1, 1, 1));

    let encoded = serde_json::to_string(&encode_retry_started(&started).unwrap()).unwrap();
    assert_eq!(encoded, ts_line);
}

#[test]
fn each_decoder_ignores_the_other_retry_tag() {
    let scheduled: SessionEvent = serde_json::from_str(&row_of("llm/retry")).unwrap();
    let started: SessionEvent = serde_json::from_str(&row_of("llm/retry-started")).unwrap();
    assert!(decode_retry_started(&scheduled).is_none());
    assert!(decode_retry(&started).is_none());
}
