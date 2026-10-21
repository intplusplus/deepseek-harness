//! Codec dispatch: registration validation, version classification, and the
//! JSONL read path. Refusal texts are the TypeScript catalog's diagnostics,
//! asserted verbatim where the TS tests pin them.

use dsh_session::migrate_session;
use dsh_session::migration::Log;
use dsh_session_format_catalog::{
    CatalogSpec, HeaderReadStatus, SessionFormatCatalog, HISTORICAL_CURRENT_VERSION,
};
use serde_json::{json, Value};

/// A minimal stored header for `version`, in the physical framing.
fn header(version: u32) -> Value {
    let mut header = json!({
        "type": "session",
        "version": version,
        "id": "catalog",
        "createdAt": 1,
    });
    if version <= 1 {
        header["delegationDepth"] = json!(0);
    } else {
        header["isSeeded"] = json!(false);
        header["delegationDepth"] = json!(0);
    }
    header
}

fn feedback_row(seq: u64) -> String {
    json!({"type": "feedback/record", "seq": seq, "time": 1, "data": {"text": "retained"}})
        .to_string()
}

fn jsonl(version: u32, rows: &[String]) -> String {
    let mut text = header(version).to_string();
    for row in rows {
        text.push('\n');
        text.push_str(row);
    }
    text
}

// ---------------------------------------------------------------------------
// readHeader: the version dispatch matrix
// ---------------------------------------------------------------------------

#[test]
fn headers_below_current_are_classified_migration_required_with_the_migrated_header() {
    let catalog = SessionFormatCatalog::shared();
    for stored in 0..4 {
        let result = catalog.read_header(&header(stored));
        assert_eq!(
            result.status,
            HeaderReadStatus::MigrationRequired,
            "v{stored}"
        );
        assert_eq!(result.stored_version, Some(stored));
        assert_eq!(result.target_version, 4);
        let migrated = result.header.expect("a migrated header");
        assert_eq!(migrated["version"], json!(4));
        assert_eq!(migrated["id"], json!("catalog"));
        assert!(
            migrated.get("type").is_none(),
            "the physical tag is dropped"
        );
    }
}

#[test]
fn a_current_header_is_classified_current() {
    let result = SessionFormatCatalog::shared().read_header(&header(4));
    assert_eq!(result.status, HeaderReadStatus::Current);
    assert_eq!(result.stored_version, Some(4));
    assert_eq!(result.header.expect("migrated")["version"], json!(4));
}

#[test]
fn a_future_generation_is_classified_unsupported_with_the_ts_text() {
    let result = SessionFormatCatalog::shared().read_header(&header(5));
    assert_eq!(result.status, HeaderReadStatus::Unsupported);
    assert_eq!(
        result.reason.as_deref(),
        Some("stored Session uses newer format v5; this build writes v4")
    );
}

#[test]
fn a_non_object_header_is_malformed_with_the_ts_text() {
    let catalog = SessionFormatCatalog::shared();
    for value in [json!(["session"]), json!("session"), json!(null)] {
        let result = catalog.read_header(&value);
        assert_eq!(result.status, HeaderReadStatus::Malformed);
        assert_eq!(
            result.reason.as_deref(),
            Some("Session header must be a JSON object")
        );
    }
}

#[test]
fn a_missing_or_invalid_version_is_malformed_with_the_ts_text() {
    let catalog = SessionFormatCatalog::shared();
    let candidates = [
        json!({"id": "catalog"}),
        json!({"version": "2", "id": "catalog"}),
        json!({"version": -1, "id": "catalog"}),
        json!({"version": 1.5, "id": "catalog"}),
    ];
    for value in candidates {
        let result = catalog.read_header(&value);
        assert_eq!(result.status, HeaderReadStatus::Malformed);
        assert_eq!(
            result.reason.as_deref(),
            Some("Session format version must be a non-negative safe integer"),
        );
    }
}

// ---------------------------------------------------------------------------
// restore_jsonl: the codec-selection read path
// ---------------------------------------------------------------------------

#[test]
fn a_future_generation_log_is_refused_before_any_row_is_opened() {
    let text = jsonl(5, &[feedback_row(0)]);
    let error = SessionFormatCatalog::shared()
        .restore_jsonl(&text)
        .unwrap_err();
    assert!(error.is_unsupported_migration());
    assert_eq!(
        error.message(),
        "stored Session uses newer format v5; this build writes v4",
    );
}

#[test]
fn an_empty_log_is_refused_with_the_ts_text() {
    let error = SessionFormatCatalog::shared()
        .restore_jsonl("")
        .unwrap_err();
    assert_eq!(error.message(), "stored log is empty");
}

#[test]
fn a_v1_log_migrates_end_to_end_and_matches_the_core_chain_header() {
    let text = jsonl(1, &[feedback_row(0)]);
    let artifact = SessionFormatCatalog::shared()
        .restore_jsonl(&text)
        .expect("v1 migrates");
    assert_eq!(artifact.header["version"], json!(4));
    assert_eq!(artifact.inherited_event_count, 0);
    assert_eq!(artifact.events.len(), 1);
    assert_eq!(artifact.events[0].r#type, "feedback/record");
    assert_eq!(artifact.events[0].seq, 0);
    assert_eq!(artifact.events[0].data["text"], json!("retained"));

    let log = Log::parse(&text).unwrap();
    let (core_header, _) = migrate_session(log.clone(), &[]).unwrap();
    assert_eq!(
        artifact.header, core_header,
        "dispatch parity with the core chain"
    );
}

#[test]
fn a_v2_log_acquires_the_promoted_system_head_through_the_catalog() {
    let text = jsonl(
        2,
        &[
            r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#.to_string(),
            r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#.to_string(),
            feedback_row(2),
        ],
    );
    let artifact = SessionFormatCatalog::shared()
        .restore_jsonl(&text)
        .expect("v2 migrates");
    let types: Vec<&str> = artifact
        .events
        .iter()
        .map(|event| event.r#type.as_str())
        .collect();
    assert_eq!(
        types,
        [
            "turn/start",
            "step/start",
            "system/message",
            "feedback/record"
        ]
    );
    assert_eq!(artifact.events[2].seq, 2);
    assert!(
        artifact.events[2].data["message"]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("v2-to-v3-system-")),
        "the system head carries the TypeScript identity"
    );
    assert_eq!(artifact.events[3].seq, 3, "later rows are renumbered");
}

#[test]
fn a_v2_delivery_marker_claiming_v3_is_refused_with_the_wrapped_ts_text() {
    let text = jsonl(
        2,
        &[
            feedback_row(0),
            r#"{"type":"session-log-deepseek/delivery-accepted","seq":1,"time":2,"data":{"sessionId":"catalog","throughSeq":0,"sessionFormatVersion":3}}"#.to_string(),
        ],
    );
    let error = SessionFormatCatalog::shared()
        .restore_jsonl(&text)
        .unwrap_err();
    assert!(error.is_unsupported_migration());
    assert_eq!(
        error.message(),
        "@deepseek-ai/dsh-session-format-v2-to-v3 refuses this format v2 Session: \
         format v2 delivery marker claims target format v3",
    );
}

#[test]
fn a_v2_input_carrying_v3_dispatch_tags_is_refused_with_the_ts_text() {
    for tag in ["tool/ptc-dispatch-start", "tool/ptc-dispatch"] {
        let row = json!({"type": tag, "seq": 0, "time": 1, "data": {}}).to_string();
        let error = SessionFormatCatalog::shared()
            .restore_jsonl(&jsonl(2, &[row]))
            .unwrap_err();
        assert!(
            error.message().contains("format v2") && error.message().contains("ptc-dispatch"),
            "expected the TS diagnosis, got: {error}",
        );
    }
}

#[test]
fn restore_log_dispatches_an_already_parsed_log() {
    let text = jsonl(1, &[feedback_row(0)]);
    let log = Log::parse(&text).unwrap();
    let artifact = SessionFormatCatalog::shared()
        .restore_log(log)
        .expect("migrates");
    assert_eq!(artifact.header["version"], json!(4));
    assert_eq!(artifact.events.len(), 1);
}

// ---------------------------------------------------------------------------
// Registration validation (the catalog constructor checks)
// ---------------------------------------------------------------------------

fn spec(current_version: u32, codecs: Vec<u32>, edges: Vec<(u32, u32)>) -> CatalogSpec {
    CatalogSpec {
        current_version,
        codec_versions: codecs,
        migration_edges: edges,
    }
}

fn full_edges(current: u32) -> Vec<(u32, u32)> {
    (0..current).map(|from| (from, from + 1)).collect()
}

#[test]
fn a_duplicated_codec_is_refused_with_the_ts_text() {
    let error =
        SessionFormatCatalog::assemble(spec(4, vec![0, 1, 2, 2, 3, 4], full_edges(4)), Vec::new())
            .unwrap_err();
    assert_eq!(error.message(), "Session format codec v2 is duplicated");
}

#[test]
fn a_missing_codec_is_refused_with_the_ts_text() {
    let error =
        SessionFormatCatalog::assemble(spec(2, vec![0, 2], full_edges(2)), Vec::new()).unwrap_err();
    assert_eq!(error.message(), "Session format codec v1 is missing");
}

#[test]
fn a_codec_newer_than_current_is_refused_with_the_ts_text() {
    let error =
        SessionFormatCatalog::assemble(spec(4, vec![0, 1, 2, 3, 4, 5], full_edges(4)), Vec::new())
            .unwrap_err();
    assert_eq!(
        error.message(),
        "Session format codec v5 is newer than current v4"
    );
}

#[test]
fn a_missing_migration_edge_is_refused_with_the_ts_text() {
    let error =
        SessionFormatCatalog::assemble(spec(3, vec![0, 1, 2, 3], vec![(0, 1), (1, 2)]), Vec::new())
            .unwrap_err();
    assert!(error.is_unsupported_migration());
    assert_eq!(error.message(), "Session migration v2->v3 is missing");
}

#[test]
fn a_duplicated_migration_edge_is_refused_with_the_ts_text() {
    let error = SessionFormatCatalog::assemble(
        spec(2, vec![0, 1, 2], vec![(0, 1), (1, 2), (1, 2)]),
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(error.message(), "Session migration v1->v2 is duplicated");
}

#[test]
fn a_non_adjacent_migration_edge_is_refused_with_the_ts_text() {
    let error =
        SessionFormatCatalog::assemble(spec(2, vec![0, 1, 2], vec![(0, 1), (1, 3)]), Vec::new())
            .unwrap_err();
    assert_eq!(error.message(), "Session migration v1->v3 must be adjacent");
}

#[test]
fn an_edge_that_does_not_lead_to_current_is_refused_with_the_ts_text() {
    let error = SessionFormatCatalog::assemble(
        spec(
            4,
            vec![0, 1, 2, 3, 4],
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        ),
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(
        error.message(),
        "Session migration from v4 does not lead to current v4",
    );
}

// ---------------------------------------------------------------------------
// The historical prerequisite catalog (TS historicalSessionFormatCatalog)
// ---------------------------------------------------------------------------

#[test]
fn the_historical_catalog_stops_at_v3() {
    let catalog = SessionFormatCatalog::historical();
    assert_eq!(catalog.current_version(), HISTORICAL_CURRENT_VERSION);
    assert_eq!(catalog.codec_versions().len(), 4);

    let v3 = catalog.read_header(&header(3));
    assert_eq!(v3.status, HeaderReadStatus::Current);
    assert_eq!(v3.header.expect("migrated")["version"], json!(3));

    let v2 = catalog.read_header(&header(2));
    assert_eq!(v2.status, HeaderReadStatus::MigrationRequired);
    assert_eq!(v2.header.expect("migrated")["version"], json!(3));

    let v4 = catalog.read_header(&header(4));
    assert_eq!(v4.status, HeaderReadStatus::Unsupported);
    assert_eq!(
        v4.reason.as_deref(),
        Some("stored Session uses newer format v4; this build writes v3"),
    );
}

#[test]
fn the_historical_catalog_restores_a_v2_log_to_v3() {
    let text = jsonl(2, &[feedback_row(0)]);
    let artifact = SessionFormatCatalog::historical()
        .restore_jsonl(&text)
        .expect("migrates to v3");
    assert_eq!(artifact.header["version"], json!(3));
    assert_eq!(artifact.events.len(), 1);
}

#[test]
fn child_evidence_bound_into_the_edge_does_not_change_plain_migration() {
    // An empty slice declares a parent without children (TS
    // `createSessionFormatCatalogWithChildren([])`); plain migration is
    // identical to the shared catalog.
    let text = jsonl(1, &[feedback_row(0)]);
    let with_children = SessionFormatCatalog::with_children(Vec::new());
    let plain = SessionFormatCatalog::shared().restore_jsonl(&text).unwrap();
    let bound = with_children.restore_jsonl(&text).unwrap();
    assert_eq!(plain.header, bound.header);
    assert_eq!(plain.events.len(), bound.events.len());
}

#[test]
fn malformed_child_evidence_is_refused_with_the_ts_text() {
    // The v3→v4 edge demands explicit, well-formed child facts; anything else
    // is a migration refusal naming the edge.
    let text = jsonl(3, &[feedback_row(0)]);
    let bound = SessionFormatCatalog::with_children(vec![json!({"unrelated": true})]);
    let error = bound.restore_jsonl(&text).unwrap_err();
    assert!(error.is_unsupported_migration());
    assert_eq!(
        error.message(),
        "@deepseek-ai/dsh-session-format-v3-to-v4 refuses this format v3 Session: \
         catalog migration requires historical child identity and descriptor evidence",
    );
}

// ---------------------------------------------------------------------------
// Chain-linkage refusals exercised through dispatch
// ---------------------------------------------------------------------------

#[test]
fn an_unknown_required_v3_event_type_is_refused_with_the_ts_text() {
    let row =
        json!({"type": "tool/code-dispatch-start", "seq": 0, "time": 1, "data": {}}).to_string();
    let error = SessionFormatCatalog::shared()
        .restore_jsonl(&jsonl(3, &[row]))
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("unknown event type `tool/code-dispatch-start`"),
        "expected the TS diagnosis, got: {error}",
    );
}

#[test]
fn a_damaged_row_is_classified_format_not_migration_refusal() {
    let text = jsonl(
        2,
        &[json!({
            "type": "feedback/record",
            "seq": 0,
            "time": 1,
            "data": {"text": "kept"},
            "ignorable": false,
        })
        .to_string()],
    );
    let error = SessionFormatCatalog::shared()
        .restore_jsonl(&text)
        .unwrap_err();
    assert!(!error.is_unsupported_migration());
    assert!(
        error.message().contains("ignorable"),
        "the envelope audit fires: {error}",
    );
}

#[test]
fn the_v0_fixture_migrates_through_the_catalog_with_chain_parity() {
    // The recorded synthetic v0 fixture (owned by `dsh-session`, read-only)
    // goes through catalog dispatch and lands on the same artifact the core
    // chain produces.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../dsh-session/tests/fixtures/v0-synthetic-normalized.jsonl"
    );
    let text = std::fs::read_to_string(path).expect("the fixture is readable");
    let artifact = SessionFormatCatalog::shared()
        .restore_jsonl(&text)
        .expect("the normalized fixture migrates");
    let log = Log::parse(&text).unwrap();
    let (core_header, core_events) = migrate_session(log.clone(), &[]).unwrap();
    assert_eq!(
        artifact.header, core_header,
        "header parity with the core chain"
    );
    assert_eq!(artifact.events.len(), core_events.len());
    // The same text restores through the installed reader, which runs the
    // chain and then re-admits every event through the runtime vocabulary.
    let session = dsh_session::decode(&text).expect("the migrated log reopens");
    assert_eq!(session.header().version, 4);
}
