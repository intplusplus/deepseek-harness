//! Read the TypeScript-written physical v4 fixtures.
//!
//! `fixtures/v4-physical/*.v4.jsonl` are produced by the shipped TypeScript
//! codec (`rust/fixtures/generate.mjs`), not by Rust. Reading them here is the
//! end-to-end evidence that the Rust physical reader agrees with the writer
//! that will eventually replace it.
//!
//! The fixtures are the regression lock, not the oracle: the *rules* are
//! asserted by `physical.rs`, and the byte-level comparison against the
//! TypeScript row order lives in `fixtures.rs` and `fixtures/compare-rust.mjs`.

use dsh_session::physical::decode_physical_log_with;
use std::collections::BTreeSet;
use std::path::Path;

/// The plugin event types the shipped fixtures carry: the compaction seam and
/// the retry seam. The Rust core owns only its own vocabulary, so a reader
/// that wants these rows must say it understands them — exactly the
/// `knownEventTypes` argument the TypeScript reader takes.
fn known_types() -> BTreeSet<String> {
    [
        "compaction/start",
        "compaction/end",
        "compaction/summary",
        "llm/retry",
        "llm/retry-started",
    ]
    .iter()
    .map(|tag| (*tag).to_string())
    .collect()
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    // `CARGO_MANIFEST_DIR` is `rust/crates/dsh-session`; the fixtures live
    // two levels up, in `rust/fixtures`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/v4-physical")
        .join(format!("{name}.v4.jsonl"))
}

fn read_fixture(name: &str) -> String {
    let path = fixture_path(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

#[test]
fn the_typescript_physical_fixture_reads_back() {
    let text = read_fixture("turn-tool-call");
    let session = decode_physical_log_with(&text, &known_types())
        .unwrap_or_else(|error| panic!("cannot decode the fixture: {error}"));

    assert_eq!(session.header().id.as_str(), "fixture-turn-tool-call");
    assert_eq!(session.header().created_at, 1760000000000);
    assert_eq!(
        session.header().cwd.as_deref(),
        Some("/workspace/dsh-fixtures")
    );
    assert!(!session.header().is_seeded);
    assert_eq!(session.header().delegation_depth, Some(0));
    // The fixture covers a full turn: twelve events over thirteen physical
    // lines (the header plus twelve rows).
    assert_eq!(session.events().len(), 12);
}

#[test]
fn the_typescript_retry_fixture_reads_back() {
    let text = read_fixture("retry");
    let session = decode_physical_log_with(&text, &known_types())
        .unwrap_or_else(|error| panic!("cannot decode the fixture: {error}"));
    assert_eq!(session.events().len(), 10);
}

#[test]
fn the_typescript_tool_error_fixture_reads_back() {
    let text = read_fixture("tool-error");
    let session = decode_physical_log_with(&text, &known_types())
        .unwrap_or_else(|error| panic!("cannot decode the fixture: {error}"));
    assert_eq!(session.events().len(), 7);
}

#[test]
fn the_typescript_extension_fixture_reads_back() {
    let text = read_fixture("extension-unicode");
    let session = decode_physical_log_with(&text, &known_types())
        .unwrap_or_else(|error| panic!("cannot decode the fixture: {error}"));
    assert_eq!(session.events().len(), 10);
}

#[test]
fn a_reader_without_the_plugin_vocabulary_refuses_the_plugin_rows() {
    // The same fixture, read by a reader that understands only the core
    // vocabulary: the compaction and retry rows are refused, because the
    // TypeScript reader does the same against its installed event types.
    let text = read_fixture("retry");
    let error =
        decode_physical_log_with(&text, &[] as &[&str]).expect_err("the plugin rows are refused");
    assert!(
        error.to_string().contains("unknown event type"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn the_unknown_plugin_event_needs_its_ignorable_marker() {
    // The `extension-unicode` fixture carries an unknown `plugin:fixture/acme`
    // row. It is admitted because the TypeScript writer marked it ignorable;
    // removing that marker must make the reader refuse the row, exactly as
    // `packages/session/session-format-v1-to-v2/src/validation.ts:69-76`
    // requires.
    let text = read_fixture("extension-unicode");
    let stripped: String = text
        .lines()
        .map(|line| {
            if line.contains("plugin:fixture/acme") {
                line.replace(",\"ignorable\":true", "")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let error =
        decode_physical_log_with(&stripped, &known_types()).expect_err("the row is refused");
    assert!(
        error.to_string().contains("unknown event type"),
        "unexpected diagnostic: {error}"
    );
}
