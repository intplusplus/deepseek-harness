//! `tool/ptc-dispatch*` wire format and strict decode.
//!
//! Ported from the `SessionEventMap` declaration in
//! `packages/core/tools/src/types.ts` and the two appends in
//! `packages/core/tools/src/ptc.ts`. The byte assertions exist because field
//! order is what a session log compares against the TypeScript producer.

use dsh_llm::{ContentBlock, ToolCallId};
use dsh_ptc::{PtcDispatch, PtcDispatchStart, PtcEvent, PTC_DISPATCH_START_TAG, PTC_DISPATCH_TAG};
use dsh_session::{SessionEvent, ToolErrorInfo};
use serde_json::{json, Map, Value};

fn id(value: &str) -> ToolCallId {
    ToolCallId::new(value)
}

fn encoded(event: PtcEvent) -> String {
    serde_json::to_string(&event.encode().expect("encodable")).expect("serializable")
}

fn extension(kind: &str, payload: Value) -> SessionEvent {
    let Value::Object(payload) = payload else {
        panic!("a test payload must be an object");
    };
    SessionEvent::extension_event(kind, payload)
}

fn start_event() -> PtcDispatchStart {
    PtcDispatchStart::new(
        id("call-1"),
        id("call-1"),
        id("call-1:ptc:1"),
        "read",
        json!({ "path": "a.ts" }),
    )
}

#[test]
fn a_start_record_serializes_type_first_then_the_pairing_ids() {
    assert_eq!(
        encoded(PtcEvent::Start(start_event())),
        r#"{"type":"tool/ptc-dispatch-start","rootCallId":"call-1","parentCallId":"call-1","subCallId":"call-1:ptc:1","name":"read","arguments":{"path":"a.ts"}}"#
    );
}

#[test]
fn a_successful_settle_record_omits_the_optional_error_and_meta() {
    let event = PtcDispatch::success(
        id("call-1"),
        id("call-1"),
        id("call-1:ptc:1"),
        "read",
        json!({ "path": "a.ts" }),
        vec![ContentBlock::text("ok")],
    );
    assert_eq!(
        encoded(PtcEvent::Settle(Box::new(event))),
        r#"{"type":"tool/ptc-dispatch","rootCallId":"call-1","parentCallId":"call-1","subCallId":"call-1:ptc:1","name":"read","arguments":{"path":"a.ts"},"isError":false,"content":[{"type":"text","text":"ok"}]}"#
    );
}

#[test]
fn a_failed_settle_record_places_error_between_is_error_and_content() {
    let event = PtcDispatch::failure(
        id("call-1"),
        id("call-1"),
        id("call-1:ptc:2"),
        "write",
        json!({ "path": "b.ts", "text": "x" }),
        vec![ContentBlock::text("Error: denied")],
    )
    .with_error(ToolErrorInfo {
        name: "SandboxDeniedError".into(),
        code: "SANDBOX_DENIED".into(),
        reason: Some("read-only".into()),
    })
    .with_meta(json!({ "spill": { "bytes": 9 } }));

    assert_eq!(
        encoded(PtcEvent::Settle(Box::new(event))),
        r#"{"type":"tool/ptc-dispatch","rootCallId":"call-1","parentCallId":"call-1","subCallId":"call-1:ptc:2","name":"write","arguments":{"path":"b.ts","text":"x"},"isError":true,"error":{"name":"SandboxDeniedError","code":"SANDBOX_DENIED","reason":"read-only"},"content":[{"type":"text","text":"Error: denied"}],"meta":{"spill":{"bytes":9}}}"#
    );
}

#[test]
fn both_records_round_trip_through_the_log() {
    let start = PtcEvent::Start(start_event());
    let settle = PtcEvent::Settle(Box::new(PtcDispatch::success(
        id("call-1"),
        id("call-1"),
        id("call-1:ptc:1"),
        "read",
        json!({ "path": "a.ts" }),
        vec![ContentBlock::text("ok")],
    )));
    for event in [start, settle] {
        let encoded_event = event.encode().expect("encodable");
        match PtcEvent::decode(&encoded_event) {
            Some(Ok(decoded)) => assert_eq!(decoded, event),
            Some(Err(error)) => panic!("{} did not decode: {error}", event.kind()),
            None => panic!("{} was not recognized", event.kind()),
        }
        assert_eq!(event.sub_call_id().as_str(), "call-1:ptc:1");
    }
}

#[test]
fn the_two_tags_are_the_whole_vocabulary() {
    assert_eq!(PTC_DISPATCH_START_TAG, "tool/ptc-dispatch-start");
    assert_eq!(PTC_DISPATCH_TAG, "tool/ptc-dispatch");
    // The run's own boundary is the enclosing `run_code` tool/result pair, so
    // a third "end" tag would be a second source of truth for it.
    assert_eq!(
        dsh_ptc::PTC_EVENT_TAGS,
        ["tool/ptc-dispatch-start", "tool/ptc-dispatch"]
    );
}

#[test]
fn a_foreign_tag_is_not_this_crates_event() {
    assert!(PtcEvent::decode(&extension("tool/result", json!({}))).is_none());
    assert!(PtcEvent::decode(&SessionEvent::turn_start(1)).is_none());
}

#[test]
fn a_known_tag_with_an_unexpected_payload_fails_loudly() {
    // A missing required field must not degrade to an opaque event.
    let decoded = PtcEvent::decode(&extension(
        PTC_DISPATCH_TAG,
        json!({
            "rootCallId": "call-1",
            "parentCallId": "call-1",
            "subCallId": "call-1:ptc:1",
            "name": "read",
            "arguments": {}
        }),
    ))
    .expect("a known tag");
    assert!(
        decoded.is_err(),
        "a malformed settle record must not decode"
    );

    // An undeclared field is refused, not ignored: the log's shape is a
    // contract both sides can rely on.
    let decoded = PtcEvent::decode(&extension(
        PTC_DISPATCH_START_TAG,
        json!({
            "rootCallId": "call-1",
            "parentCallId": "call-1",
            "subCallId": "call-1:ptc:1",
            "name": "read",
            "arguments": {},
            "parentToken": "leaked"
        }),
    ))
    .expect("a known tag");
    assert!(decoded.is_err(), "an undeclared field must not decode");
}

#[test]
fn the_error_field_is_optional_inside_a_failed_record() {
    let event = PtcDispatch::failure(
        id("call-1"),
        id("call-1"),
        id("call-1:ptc:3"),
        "read",
        json!({}),
        vec![ContentBlock::text("Error: boom")],
    );
    let encoded = PtcEvent::Settle(Box::new(event))
        .encode()
        .expect("encodable");
    let payload: Map<String, Value> = encoded.extension().expect("extension").payload().clone();
    assert!(!payload.contains_key("error"));
    assert_eq!(payload["isError"], json!(true));
    // `type` is stamped first and the payload keeps its own field order.
    assert_eq!(
        payload.keys().collect::<Vec<_>>(),
        [
            "type",
            "rootCallId",
            "parentCallId",
            "subCallId",
            "name",
            "arguments",
            "isError",
            "content"
        ]
    );
}
