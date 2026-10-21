//! v2 -> v3: system-prompt promotion and the v3 vocabulary
//! (`origin/packages/session/session-format-v2-to-v3`).

use dsh_session::{v2_to_v3, Log, SessionError};
use serde_json::json;

const V2: &str = r#"{"version":2,"id":"s1","createdAt":1,"isSeeded":false,"delegationDepth":0}"#;

fn log(rows: &[&str]) -> Log {
    let mut text = String::from(V2);
    text.push('\n');
    for row in rows {
        text.push_str(row);
        text.push('\n');
    }
    Log::parse(&text).unwrap()
}

fn error(rows: &[&str]) -> SessionError {
    match v2_to_v3(log(rows)) {
        Ok(_) => panic!("the edge accepted a log the v3 reader refuses"),
        Err(error) => error,
    }
}

fn types(log: &Log) -> Vec<&str> {
    log.events
        .iter()
        .map(|event| event.r#type.as_str())
        .collect()
}

#[test]
fn system_prompts_are_promoted_before_their_header() {
    let log = v2_to_v3(log(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"request/header","seq":2,"time":3,"data":{"header":{"config":{"provider":"p","model":"m"},"system":"prompt A"},"reason":"initial"}}"#,
        r#"{"type":"request/header","seq":3,"time":4,"data":{"header":{"config":{"provider":"p","model":"m"},"system":"prompt B"},"reason":"change"}}"#,
        r#"{"type":"step/end","seq":4,"time":5,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"turn/end","seq":5,"time":6,"data":{"turn":1,"reason":{"kind":"completed"}}}"#,
    ]))
    .unwrap();

    assert_eq!(
        types(&log),
        vec![
            "turn/start",
            "step/start",
            "system/message",
            "system/message",
            "request/header",
            "system/message",
            "request/header",
            "step/end",
            "turn/end",
        ]
    );

    // The step opens an empty system head; each prompt change replaces it.
    assert_eq!(log.events[2].data["message"]["content"], json!([]));
    assert_eq!(log.events[2].surface_op, Some(json!("append")));
    assert_eq!(
        log.events[3].data["message"]["content"],
        json!([{ "type": "text", "text": "prompt A" }])
    );
    assert_eq!(
        log.events[3].surface_op,
        Some(json!({ "op": "replace", "startSeq": 2, "endSeq": 2 }))
    );
    assert_eq!(log.events[3].source_event_seqs, Some(vec![2]));
    assert_eq!(
        log.events[5].data["message"]["content"],
        json!([{ "type": "text", "text": "prompt B" }])
    );
    assert!(!log.events[4].data["header"]
        .as_object()
        .unwrap()
        .contains_key("system"));

    // TypeScript identity: "v2-to-v3-system-" + sha256(JSON.stringify([...])).
    let id = log.events[3].data["message"]["id"].as_str().unwrap();
    assert!(id.starts_with("v2-to-v3-system-"), "{id}");
    assert_eq!(id.len(), "v2-to-v3-system-".len() + 64);
}

#[test]
fn ptc_vocabulary_is_renamed() {
    let log = v2_to_v3(log(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"agent-preset/selected","seq":2,"time":3,"data":{"agentPreset":"code"}}"#,
        r#"{"type":"tool/code-dispatch-start","seq":3,"time":4,"data":{"rootCallId":"r","parentCallId":"r","subCallId":"c","name":"n","arguments":"{}"}}"#,
        r#"{"type":"user/message","seq":4,"time":5,"data":{"id":"u1","role":"user","content":[],"source":{"kind":"plugin","plugin":"tools-code-mode"}},"surfaceOp":"append"}"#,
    ]))
    .unwrap();

    assert_eq!(log.events[3].data["agentPreset"], json!("ptc"));
    assert_eq!(log.events[4].r#type, "tool/ptc-dispatch-start");
    assert_eq!(
        log.events[5].data["source"],
        json!({ "kind": "plugin", "plugin": "tools-ptc" })
    );
}

#[test]
fn empty_optional_header_members_are_removed() {
    let log = v2_to_v3(log(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"request/header","seq":2,"time":3,"data":{"header":{"config":{"provider":"p","model":"m"},"tools":[],"adapterDefaults":{}},"reason":"initial"}}"#,
    ]))
    .unwrap();

    let header = log.events[3].data["header"].as_object().unwrap();
    assert!(!header.contains_key("tools"));
    assert!(!header.contains_key("adapterDefaults"));
}

#[test]
fn a_surface_event_before_the_first_step_is_refused() {
    let error = error(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"user/message","seq":1,"time":2,"data":{"id":"u1","role":"user","content":[],"source":{"kind":"user"}},"surfaceOp":"append"}"#,
    ]);
    assert!(
        error.to_string().contains("surface before first step"),
        "{error}"
    );
}
