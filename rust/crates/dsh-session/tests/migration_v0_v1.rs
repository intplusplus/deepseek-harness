//! v0 -> v1 normalization: the released react-loop rewrite.
//!
//! Every expectation here is taken from
//! `origin/packages/session/session-format-v0-to-v1`.

use dsh_session::{v0_to_v1, Log, SessionError};
use serde_json::{json, Value};

const V0: &str = r#"{"type":"session","version":0,"id":"s1","createdAt":1,"delegationDepth":0}"#;

fn log(rows: &[&str]) -> Log {
    let mut text = String::from(V0);
    text.push('\n');
    for row in rows {
        text.push_str(row);
        text.push('\n');
    }
    Log::parse(&text).unwrap()
}

fn error(rows: &[&str]) -> SessionError {
    match v0_to_v1(log(rows)) {
        Ok(_) => panic!("the edge accepted a log the v1 reader refuses"),
        Err(error) => error,
    }
}

#[test]
fn legacy_turn_shapes_are_normalized() {
    let log = v0_to_v1(log(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1,"trigger":{"kind":"user"}}}"#,
        r#"{"type":"turn/end","seq":1,"time":2,"data":{"turn":1,"reason":{"kind":"disposed"}}}"#,
        r#"{"type":"turn/end","seq":2,"time":3,"data":{"turn":1,"reason":{"kind":"error","step":1,"message":"boom"}}}"#,
        r#"{"type":"turn/end","seq":3,"time":4,"data":{"turn":1,"reason":{"kind":"aborted"}}}"#,
    ]))
    .unwrap();

    assert_eq!(
        log.events[0].data,
        json!({ "turn": 1 }).as_object().unwrap().clone()
    );
    assert_eq!(
        log.events[1].data["reason"],
        json!({ "kind": "aborted", "reason": { "kind": "disposed" } })
    );
    assert_eq!(
        log.events[2].data["reason"],
        json!({ "kind": "error", "error": { "message": "boom", "code": "UNKNOWN" } })
    );
    assert_eq!(
        log.events[3].data["reason"],
        json!({ "kind": "aborted", "reason": { "kind": "legacy" } })
    );
}

#[test]
fn steering_messages_become_user_messages() {
    let log = v0_to_v1(log(&[
        r#"{"type":"steering/message","seq":0,"time":1,"data":{"turn":1,"content":[{"type":"text","text":"hi"}],"source":{"kind":"user"}},"surfaceOp":"append"}"#,
        r#"{"type":"steering/message","seq":1,"time":2,"data":{"turn":2,"message":{"id":"m2","role":"user","content":[],"source":{"kind":"user"}}},"surfaceOp":"append"}"#,
    ]))
    .unwrap();

    assert_eq!(log.events[0].r#type, "user/message");
    assert_eq!(
        Value::Object(log.events[0].data.clone()),
        json!({
            "content": [{ "type": "text", "text": "hi" }],
            "source": { "kind": "user" },
            "id": "legacy-message:s1:0",
            "role": "user",
        })
    );
    assert_eq!(
        Value::Object(log.events[1].data.clone()),
        json!({
            "id": "m2",
            "role": "user",
            "content": [],
            "source": { "kind": "user" },
        })
    );
}

#[test]
fn legacy_messages_receive_stable_identities() {
    let log = v0_to_v1(log(&[
        r#"{"type":"user/message","seq":0,"time":1,"data":{"content":[{"type":"text","text":"hi"}],"source":{"kind":"user"}},"surfaceOp":"append"}"#,
        r#"{"type":"assistant/message","seq":1,"time":2,"data":{"turn":1,"step":1,"content":[{"type":"text","text":"yo"}],"provenance":{"provider":"p","model":"m"}},"surfaceOp":"append"}"#,
        r#"{"type":"tool/result","seq":2,"time":3,"data":{"turn":1,"step":1,"callId":"c1","content":[{"type":"text","text":"ok"}],"isError":false},"surfaceOp":{"op":"replace","start":0,"end":0}}"#,
    ]))
    .unwrap();

    assert_eq!(log.events[0].data["id"], json!("legacy-message:s1:0"));
    assert_eq!(log.events[0].data["role"], json!("user"));

    let assistant = log.events[1].data["message"].as_object().unwrap();
    assert_eq!(assistant["id"], json!("legacy-message:s1:1"));
    assert_eq!(assistant["role"], json!("assistant"));
    assert_eq!(
        assistant["source"],
        json!({ "provider": "p", "model": "m", "kind": "model" })
    );

    // The replacement inherits the identity of the message it replaced.
    let result = log.events[2].data["message"].as_object().unwrap();
    assert_eq!(result["id"], json!("legacy-message:s1:0"));
    assert_eq!(result["role"], json!("user"));
    assert_eq!(result["source"], json!({ "kind": "tool", "callId": "c1" }));
    assert_eq!(
        result["content"],
        json!([{
            "type": "tool-result",
            "toolCallId": "c1",
            "content": [{ "type": "text", "text": "ok" }],
            "isError": false,
        }])
    );
    assert!(!log.events[2].data.contains_key("callId"));
    assert!(!log.events[2].data.contains_key("isError"));
}

#[test]
fn unsupported_legacy_events_are_refused() {
    let delta = error(&[r#"{"type":"request/header-delta","seq":0,"time":1,"data":{}}"#]);
    assert!(
        delta.to_string().contains("request/header-delta"),
        "{delta}"
    );

    let mode = error(&[r#"{"type":"mode/set","seq":0,"time":1,"data":{}}"#]);
    assert!(mode.to_string().contains("mode/set"), "{mode}");

    let fallback = error(&[
        r#"{"type":"request/header","seq":0,"time":1,"data":{"header":{"config":{"provider":"p","model":"m"}},"reason":"fallback"}}"#,
    ]);
    assert!(fallback.to_string().contains("fallback"), "{fallback}");
}

#[test]
fn compact_events_are_renamed_and_labelled() {
    let log = v0_to_v1(log(&[
        r#"{"type":"compact/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"compact/end","seq":1,"time":2,"data":{"turn":1}}"#,
    ]))
    .unwrap();

    assert_eq!(log.events[0].r#type, "compaction/start");
    assert_eq!(
        log.events[0].data["compactionId"],
        json!("legacy-compaction:s1:0")
    );
    assert_eq!(log.events[1].r#type, "compaction/end");
    assert_eq!(
        log.events[1].data["compactionId"],
        json!("legacy-compaction:s1:0")
    );

    // A second end has no open compaction to name, so the payload is incomplete.
    let error = error(&[
        r#"{"type":"compact/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"compact/end","seq":1,"time":2,"data":{"turn":1}}"#,
        r#"{"type":"compact/end","seq":2,"time":3,"data":{"turn":1}}"#,
    ]);
    assert!(error.to_string().contains("compactionId"), "{error}");
}

#[test]
fn malformed_payloads_are_refused() {
    let missing = error(&[
        r#"{"type":"user/message","seq":0,"time":1,"data":{"id":"x","role":"user","content":[]},"surfaceOp":"append"}"#,
    ]);
    assert!(missing.to_string().contains("source"), "{missing}");

    let unexpected = error(&[
        r#"{"type":"user/message","seq":0,"time":1,"data":{"id":"x","role":"user","content":[],"source":{"kind":"user"},"extra":1},"surfaceOp":"append"}"#,
    ]);
    assert!(unexpected.to_string().contains("extra"), "{unexpected}");
}

#[test]
fn retry_identity_is_shared_across_one_policy_chain() {
    let log = v0_to_v1(log(&[
        r#"{"type":"llm/retry","seq":0,"time":1,"data":{"turn":1,"step":1,"provider":"p","mode":"normal","policyKey":"k","retry":1,"maxRetries":3,"delayMs":0,"failure":{"message":"m","code":"c"}}}"#,
        r#"{"type":"llm/retry","seq":1,"time":2,"data":{"turn":1,"step":1,"provider":"p","mode":"normal","policyKey":"k","retry":2,"maxRetries":3,"delayMs":0,"failure":{"message":"m","code":"c"}}}"#,
    ]))
    .unwrap();

    assert_eq!(log.events[0].data["retryId"], json!("legacy-retry:s1:0"));
    assert_eq!(log.events[1].data["retryId"], json!("legacy-retry:s1:0"));
}

#[test]
fn a_seeded_header_records_its_cut() {
    let header = r#"{"type":"session","version":0,"id":"s1","createdAt":1,"parentSession":"p","seedLength":1,"delegationDepth":1}"#;
    let text = format!(
        "{header}\n{}\n{}\n",
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"turn/end","seq":1,"time":2,"data":{"turn":1,"reason":{"kind":"completed"}}}"#,
    );
    let log = Log::parse(&text).unwrap();
    assert!(log.header["isSeeded"].as_bool().unwrap());
    assert_eq!(log.cut, 1);
    assert!(!log.header.as_object().unwrap().contains_key("seedLength"));
}
