//! v3 -> v4: tool-role lifting, producer sources, extension tags, and child
//! catalogs (`origin/packages/session/session-format-v3-to-v4`).

use dsh_session::{migrate_session, v3_to_v4, Log, SessionError};
use serde_json::{json, Value};

const V3: &str = r#"{"version":3,"id":"s1","createdAt":1,"isSeeded":false,"delegationDepth":0}"#;

fn log(rows: &[&str]) -> Log {
    let mut text = String::from(V3);
    text.push('\n');
    for row in rows {
        text.push_str(row);
        text.push('\n');
    }
    Log::parse(&text).unwrap()
}

fn migrated(rows: &[&str], children: &[Value]) -> Log {
    v3_to_v4(log(rows), children).unwrap()
}

fn error(rows: &[&str], children: &[Value]) -> SessionError {
    match v3_to_v4(log(rows), children) {
        Ok(_) => panic!("the edge accepted a log the v4 reader refuses"),
        Err(error) => error,
    }
}

const STEP: [&str; 2] = [
    r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
    r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
];

#[test]
fn tool_results_lift_to_first_class_tool_messages() {
    let log = migrated(
        &[
            STEP[0],
            STEP[1],
            r#"{"type":"tool/call","seq":2,"time":3,"data":{"turn":1,"step":1,"callId":"c1","name":"n","arguments":"{}"}}"#,
            r#"{"type":"tool/result","seq":3,"time":4,"data":{"turn":1,"step":1,"message":{"id":"r1","role":"user","content":[{"type":"tool-result","toolCallId":"c1","content":[{"type":"text","text":"ok"}],"isError":true}],"source":{"kind":"tool","callId":"c1"}}},"surfaceOp":"append"}"#,
        ],
        &[],
    );

    assert_eq!(
        log.events[3].data["message"],
        json!({
            "role": "tool",
            "source": { "kind": "tool", "callId": "c1" },
            "toolCallId": "c1",
            "content": [{ "type": "text", "text": "ok" }],
            "isError": true,
            "id": "r1",
        })
    );
}

#[test]
fn plugin_sources_become_producer_sources() {
    let log = migrated(
        &[
            STEP[0],
            STEP[1],
            r#"{"type":"user/message","seq":2,"time":3,"data":{"id":"u1","role":"user","content":[],"source":{"kind":"plugin","plugin":"compact","compactionId":"c"}},"surfaceOp":"append"}"#,
            r#"{"type":"user/message","seq":3,"time":4,"data":{"id":"u2","role":"user","content":[],"source":{"kind":"plugin","plugin":"acme"}},"surfaceOp":"append"}"#,
            r#"{"type":"user/message","seq":4,"time":5,"data":{"id":"u3","role":"user","content":[],"source":{"kind":"plugin","plugin":"tools-ptc"}},"surfaceOp":"append"}"#,
            r#"{"type":"system/message","seq":5,"time":6,"data":{"turn":1,"step":1,"message":{"id":"s1m","role":"system","content":[],"source":{"kind":"plugin","plugin":"@deepseek-ai/dsh-system-prompt"}}},"surfaceOp":"append"}"#,
        ],
        &[],
    );

    assert_eq!(
        log.events[2].data["source"],
        json!({ "kind": "compact-checkpoint", "compactionId": "c" })
    );
    assert_eq!(
        log.events[3].data["source"],
        json!({ "kind": "plugin:acme" })
    );
    assert_eq!(log.events[4].data["source"], json!({ "kind": "ptc-mode" }));
    assert_eq!(
        log.events[5].data["message"]["source"],
        json!({ "kind": "system-prompt" })
    );
}

#[test]
fn extension_content_tags_are_namespaced() {
    let log = migrated(
        &[
            STEP[0],
            STEP[1],
            r#"{"type":"user/message","seq":2,"time":3,"data":{"id":"u1","role":"user","content":[{"type":"text","text":"a"},{"type":"acme-widget","value":1}],"source":{"kind":"user"}},"surfaceOp":"append"}"#,
        ],
        &[],
    );

    assert_eq!(
        log.events[2].data["content"],
        json!([
            { "type": "text", "text": "a" },
            { "type": "plugin:acme-widget", "value": 1 },
        ])
    );
}

#[test]
fn unknown_ignorable_events_are_namespaced_and_required_ones_refused() {
    let log = migrated(
        &[
            STEP[0],
            r#"{"type":"acme/event","seq":1,"time":2,"data":{"x":1},"ignorable":true}"#,
        ],
        &[],
    );
    assert_eq!(log.events[1].r#type, "plugin:acme/event");
    assert_eq!(log.events[1].data["x"], json!(1));

    let error = error(
        &[
            STEP[0],
            r#"{"type":"acme/event","seq":1,"time":2,"data":{}}"#,
        ],
        &[],
    );
    assert!(error.to_string().contains("unknown event type"), "{error}");
}

#[test]
fn catalogs_are_synthesized_from_child_evidence() {
    let children = [
        json!({
            "childId": "b",
            "childCreatedAt": 2,
            "descriptorCount": 1,
            "descriptor": { "version": 3, "mode": "continuable", "provider": "p", "label": "B" },
        }),
        json!({
            "childId": "a",
            "childCreatedAt": 1,
            "descriptorCount": 1,
            "descriptor": { "version": 3, "mode": "one-shot", "provider": "p" },
        }),
    ];
    let log = migrated(
        &[
            r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
            r#"{"type":"turn/end","seq":1,"time":2,"data":{"turn":1,"reason":{"kind":"completed"}}}"#,
        ],
        &children,
    );

    assert_eq!(log.events.len(), 4);
    assert_eq!(log.events[2].r#type, "subagent/catalog");
    assert_eq!(
        Value::Object(log.events[2].data.clone()),
        json!({
            "version": 0,
            "childId": "a",
            "childCreatedAt": 1,
            "mode": "one-shot",
        })
    );
    assert_eq!(
        Value::Object(log.events[3].data.clone()),
        json!({
            "version": 0,
            "childId": "b",
            "childCreatedAt": 2,
            "mode": "continuable",
            "label": "B",
        })
    );
}

#[test]
fn child_evidence_without_a_descriptor_still_gets_a_catalog_row() {
    let children = [json!({
        "childId": "a",
        "childCreatedAt": 5,
        "descriptorCount": 0,
        "descriptor": null,
    })];
    let log = migrated(
        &[
            r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
            r#"{"type":"turn/end","seq":1,"time":2,"data":{"turn":1,"reason":{"kind":"completed"}}}"#,
        ],
        &children,
    );

    assert_eq!(
        Value::Object(log.events[2].data.clone()),
        json!({
            "version": 1,
            "childId": "a",
            "childCreatedAt": 5,
            "mode": "unknown",
        })
    );
}

#[test]
fn a_conflicting_catalog_fact_is_refused() {
    let children = [json!({
        "childId": "a",
        "childCreatedAt": 9,
        "descriptorCount": 1,
        "descriptor": { "version": 3, "mode": "one-shot", "provider": "p" },
    })];
    let error = error(
        &[
            r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
            r#"{"type":"subagent/catalog","seq":1,"time":2,"data":{"version":0,"childId":"a","childCreatedAt":1,"mode":"one-shot"}}"#,
        ],
        &children,
    );
    assert!(error.to_string().contains("conflicts"), "{error}");
}

#[test]
fn a_resume_splice_inserts_the_missing_turn_end() {
    let log = migrated(
        &[
            r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
            r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
            r#"{"type":"step/end","seq":2,"time":3,"data":{"turn":1,"step":1}}"#,
            r#"{"type":"agent/inbox/spliced","seq":3,"time":4,"data":{"target":"next-turn","start":0,"inserted":[{"id":"i1","role":"user","content":[],"source":{"kind":"user"}}]}}"#,
            r#"{"type":"turn/start","seq":4,"time":5,"data":{"turn":2}}"#,
        ],
        &[],
    );

    let types: Vec<&str> = log
        .events
        .iter()
        .map(|event| event.r#type.as_str())
        .collect();
    assert_eq!(
        types,
        vec![
            "turn/start",
            "step/start",
            "step/end",
            "agent/inbox/spliced",
            "turn/end",
            "turn/start"
        ]
    );
    assert_eq!(log.events[4].data["turn"], json!(1));
    assert_eq!(
        log.events[4].data["reason"],
        json!({ "kind": "interrupted" })
    );
    assert_eq!(log.events[4].seq, 4);
    assert_eq!(log.events[5].seq, 5);
}

#[test]
fn the_lifted_row_is_flat_and_byte_stable() {
    let text = format!(
        "{V3}\n{}\n{}\n{}\n{}\n",
        STEP[0],
        STEP[1],
        r#"{"type":"tool/result","seq":2,"time":3,"data":{"turn":1,"step":1,"message":{"id":"r1","role":"user","content":[{"type":"tool-result","toolCallId":"c1","content":[{"type":"text","text":"ok"}],"isError":true}],"source":{"kind":"tool","callId":"c1"}}},"surfaceOp":"append"}"#,
        r#"{"type":"turn/end","seq":3,"time":4,"data":{"turn":1,"reason":{"kind":"completed"}}}"#,
    );
    let log = Log::parse(&text).unwrap();
    let (_, flat) = migrate_session(log, &[]).unwrap();
    assert_eq!(
        serde_json::to_string(&flat[2]).unwrap(),
        r#"{"type":"tool/result","turn":1,"step":1,"message":{"source":{"kind":"tool","callId":"c1"},"toolCallId":"c1","content":[{"type":"text","text":"ok"}],"isError":true,"id":"r1"}}"#
    );
}
