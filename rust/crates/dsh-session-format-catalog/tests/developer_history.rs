//! Developer history through the catalog dispatch (TS
//! `session-format-catalog/tests/developer-history.spec.ts` and the plugin
//! namespacing slice of `catalog.spec.ts`): ignorable developer events are
//! deferred to vocabulary restoration, and the installed catalog namespaces
//! the v3-era plugin names.

use std::iter::once;

use dsh_session_format_catalog::SessionFormatCatalog;
use serde_json::{json, Value};

fn v3_header() -> String {
    json!({
        "type": "session",
        "version": 3,
        "id": "old-extension",
        "createdAt": 1,
        "isSeeded": false,
        "delegationDepth": 0,
    })
    .to_string()
}

fn v3_rows() -> Vec<String> {
    vec![
        json!({
            "type": "feedback/record",
            "seq": 0,
            "time": 1,
            "data": {"text": "saved"},
        })
        .to_string(),
        // `developer/message` entered the installed vocabulary after v3: the
        // row is ignorable and carries vendor-only members.
        json!({
            "type": "developer/message",
            "seq": 1,
            "time": 2,
            "ignorable": true,
            "surfaceOp": "append",
            "sourceEventSeqs": [0],
            "data": {
                "content": [{"type": "tool-result", "toolCallId": "opaque", "content": []}],
                "vendor": true,
            },
        })
        .to_string(),
    ]
}

#[test]
fn the_installed_catalog_namespaces_the_v3_developer_tag() {
    let text = [v3_header(), v3_rows().join("\n")].join("\n");
    let artifact = SessionFormatCatalog::shared()
        .restore_jsonl(&text)
        .expect("v3 migrates");
    assert_eq!(artifact.events.len(), 2);
    let renamed = &artifact.events[1];
    assert_eq!(renamed.r#type, "plugin:developer/message");
    assert_eq!(renamed.seq, 1);
    // The unvalidated payload survives untouched; only the name changed.
    assert_eq!(renamed.data["vendor"], json!(true));
    assert_eq!(renamed.data["content"][0]["type"], json!("tool-result"),);
}

#[test]
fn the_historical_catalog_keeps_the_developer_tag_unnamed() {
    // The prerequisite catalog stops at v3, before the tag became known: the
    // event is retained exactly as stored.
    let text = [v3_header(), v3_rows().join("\n")].join("\n");
    let artifact = SessionFormatCatalog::historical()
        .restore_jsonl(&text)
        .expect("migrates to v3");
    assert_eq!(artifact.header["version"], json!(3));
    assert_eq!(artifact.events[1].r#type, "developer/message");
    assert_eq!(artifact.events[1].data["vendor"], json!(true));
}

#[test]
fn an_ignorable_v3_event_still_requires_the_ignorable_flag() {
    // The same tag without `ignorable` is refused, in either catalog.
    let rows = v3_rows();
    let required = json!({
        "type": "developer/message",
        "seq": 1,
        "time": 2,
        "surfaceOp": "append",
        "sourceEventSeqs": [0],
        "data": {
            "content": [{"type": "tool-result", "toolCallId": "opaque", "content": []}],
            "vendor": true,
        },
    })
    .to_string();
    let text = [v3_header(), rows[0].clone(), required].join("\n");
    let error = SessionFormatCatalog::shared()
        .restore_jsonl(&text)
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("unknown event type `developer/message`"),
        "expected the TS diagnosis, got: {error}",
    );
}

#[test]
fn a_current_log_round_trips_through_dispatch_without_the_chain() {
    // A v4 log selects the current codec directly: rows pass through the
    // envelope audit unchanged, ready for the installed vocabulary restore.
    let header = json!({
        "type": "session",
        "version": 4,
        "id": "current-developer",
        "createdAt": 1,
        "isSeeded": false,
        "delegationDepth": 0,
    });
    let events = [
        json!({
            "type": "request/header",
            "seq": 0,
            "time": 1,
            "data": {"reason": "initial", "header": {"config": {"provider": "test", "model": "test"}, "tools": [{"name": "search", "description": "Initial", "parameters": {}}]}},
        }),
        json!({
            "type": "developer/message",
            "seq": 1,
            "time": 2,
            "data": {"turn": 1, "step": 1, "headerSeq": 0, "message": {"id": "dev-1", "content": [{"type": "tool-addition", "toolName": "search"}], "source": {"kind": "tool-registry"}}},
        }),
    ];
    let text = once(header.to_string())
        .chain(events.iter().map(Value::to_string))
        .collect::<Vec<_>>()
        .join("\n");
    let artifact = SessionFormatCatalog::shared()
        .restore_jsonl(&text)
        .expect("v4 reads directly");
    assert_eq!(artifact.header["version"], json!(4));
    assert_eq!(artifact.events.len(), 2);
    assert_eq!(artifact.events[0].r#type, "request/header");
    assert_eq!(
        artifact.events[0].data["header"]["tools"][0]["name"],
        json!("search")
    );
    assert_eq!(artifact.events[1].r#type, "developer/message");
    assert_eq!(
        artifact.events[1].data["message"]["content"][0]["toolName"],
        json!("search"),
    );
    assert_eq!(artifact.inherited_event_count, 0);
}
