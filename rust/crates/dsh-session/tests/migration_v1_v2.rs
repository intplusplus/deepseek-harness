//! v1 -> v2: Assistant chunk attempts, reference remapping, and the resume
//! insertion (`origin/packages/session/session-format-v1-to-v2`).

use dsh_session::{v1_to_v2, Log, SessionError};
use serde_json::json;

const V1: &str = r#"{"version":1,"id":"s1","createdAt":1,"isSeeded":false,"delegationDepth":0}"#;

fn log(rows: &[&str]) -> Log {
    let mut text = String::from(V1);
    text.push('\n');
    for row in rows {
        text.push_str(row);
        text.push('\n');
    }
    Log::parse(&text).unwrap()
}

fn error(rows: &[&str]) -> SessionError {
    match v1_to_v2(log(rows)) {
        Ok(_) => panic!("the edge accepted a log the v2 reader refuses"),
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
fn chunks_fold_into_one_consumed_attempt() {
    let log = v1_to_v2(log(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"assistant/chunk","seq":2,"time":3,"data":{"turn":1,"step":1,"chunk":{"type":"text-delta","index":0,"text":"a"}}}"#,
        r#"{"type":"assistant/chunk","seq":3,"time":4,"data":{"turn":1,"step":1,"chunk":{"type":"text-delta","index":0,"text":"b"}}}"#,
        r#"{"type":"assistant/chunk","seq":4,"time":5,"data":{"turn":1,"step":1,"chunk":{"type":"finish","reason":{"kind":"stop"}}}}"#,
        r#"{"type":"assistant/message","seq":5,"time":6,"data":{"turn":1,"step":1,"message":{"id":"a1","role":"assistant","content":[{"type":"text","text":"ab"}],"source":{"kind":"model","provider":"p","model":"m"}}},"sourceEventSeqs":[2,3,4],"surfaceOp":"append"}"#,
        r#"{"type":"step/end","seq":6,"time":7,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"turn/end","seq":7,"time":8,"data":{"turn":1,"reason":{"kind":"completed"}}}"#,
    ]))
    .unwrap();

    assert_eq!(
        types(&log),
        vec![
            "turn/start",
            "step/start",
            "assistant/message",
            "step/end",
            "turn/end"
        ]
    );
    assert_eq!(
        log.events[2].data["stream"],
        json!([
            { "type": "text-chunks", "time0": 3, "index": 0, "dt": [1], "texts": ["a", "b"] },
            { "type": "chunk", "time": 5, "chunk": { "type": "finish", "reason": { "kind": "stop" } } },
        ])
    );
    assert!(log.events[2].source_event_seqs.is_none());
    assert!(log.events[2].surface_op.is_none());
}

#[test]
fn an_unconsumed_attempt_is_emitted_at_its_last_chunk() {
    let log = v1_to_v2(log(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"assistant/chunk","seq":2,"time":3,"data":{"turn":1,"step":1,"chunk":{"type":"text-delta","index":0,"text":"a"}}}"#,
        r#"{"type":"assistant/chunk","seq":3,"time":4,"data":{"turn":1,"step":1,"chunk":{"type":"finish","reason":{"kind":"stop"}}}}"#,
        r#"{"type":"step/end","seq":4,"time":5,"data":{"turn":1,"step":1}}"#,
    ]))
    .unwrap();

    assert_eq!(
        types(&log),
        vec!["turn/start", "step/start", "assistant/attempt", "step/end"]
    );
    assert_eq!(log.events[2].data["turn"], json!(1));
    assert_eq!(log.events[2].data["step"], json!(1));
    assert_eq!(
        log.events[2].data["stream"],
        json!([
            { "type": "text-chunks", "time0": 3, "index": 0, "dt": [], "texts": ["a"] },
            { "type": "chunk", "time": 4, "chunk": { "type": "finish", "reason": { "kind": "stop" } } },
        ])
    );
}

#[test]
fn attempts_split_across_steps() {
    let log = v1_to_v2(log(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"assistant/chunk","seq":2,"time":3,"data":{"turn":1,"step":1,"chunk":{"type":"text-delta","index":0,"text":"a"}}}"#,
        r#"{"type":"assistant/chunk","seq":3,"time":4,"data":{"turn":1,"step":1,"chunk":{"type":"finish","reason":{"kind":"stop"}}}}"#,
        r#"{"type":"step/end","seq":4,"time":5,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"step/start","seq":5,"time":6,"data":{"turn":1,"step":2}}"#,
        r#"{"type":"assistant/chunk","seq":6,"time":7,"data":{"turn":1,"step":2,"chunk":{"type":"text-delta","index":0,"text":"b"}}}"#,
        r#"{"type":"assistant/chunk","seq":7,"time":8,"data":{"turn":1,"step":2,"chunk":{"type":"finish","reason":{"kind":"stop"}}}}"#,
    ]))
    .unwrap();

    let attempts: Vec<_> = log
        .events
        .iter()
        .filter(|event| event.r#type == "assistant/attempt")
        .collect();
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].data["step"], json!(1));
    assert_eq!(attempts[1].data["step"], json!(2));
}

#[test]
fn a_message_without_chunk_references_gets_an_empty_stream() {
    let log = v1_to_v2(log(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"assistant/message","seq":2,"time":3,"data":{"turn":1,"step":1,"message":{"id":"a1","role":"assistant","content":[],"source":{"kind":"model","provider":"p","model":"m"}}},"surfaceOp":"append"}"#,
    ]))
    .unwrap();

    assert_eq!(
        types(&log),
        vec!["turn/start", "step/start", "assistant/message"]
    );
    assert_eq!(log.events[2].data["stream"], json!([]));
}

#[test]
fn mismatched_chunk_references_are_refused() {
    let error = error(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"assistant/chunk","seq":2,"time":3,"data":{"turn":1,"step":1,"chunk":{"type":"text-delta","index":0,"text":"a"}}}"#,
        r#"{"type":"assistant/chunk","seq":3,"time":4,"data":{"turn":1,"step":1,"chunk":{"type":"finish","reason":{"kind":"stop"}}}}"#,
        r#"{"type":"assistant/message","seq":4,"time":5,"data":{"turn":1,"step":1,"message":{"id":"a1","role":"assistant","content":[],"source":{"kind":"model","provider":"p","model":"m"}}},"sourceEventSeqs":[2],"surfaceOp":"append"}"#,
    ]);
    assert!(
        error.to_string().contains("complete ordered attempt"),
        "{error}"
    );
}

#[test]
fn a_packed_delta_row_round_trips_through_the_accumulator() {
    let log = v1_to_v2(log(&[
        r#"{"type":"text-chunks","seq0":0,"time0":1,"data":{"turn":1,"step":1,"index":0,"dt":[7],"texts":["a","b"]}}"#,
        r#"{"type":"assistant/message","seq":2,"time":8,"data":{"turn":1,"step":1,"message":{"id":"a1","role":"assistant","content":[{"type":"text","text":"ab"}],"source":{"kind":"model","provider":"p","model":"m"}}},"sourceEventSeqs":[0,1],"surfaceOp":"append"}"#,
    ]))
    .unwrap();

    assert_eq!(
        log.events[0].data["stream"],
        json!([{ "type": "text-chunks", "time0": 1, "index": 0, "dt": [7], "texts": ["a", "b"] }])
    );
}

#[test]
fn a_legacy_goal_change_splits_out_of_the_user_message() {
    let goal = json!({
        "id": "G",
        "revision": 1,
        "objective": "o",
        "phase": "active",
        "maxGoalRounds": 3,
    });
    let change = json!({
        "kind": "goal/change",
        "version": 1,
        "operation": "create",
        "goal": goal.clone(),
        "roundsStarted": 0,
        "createdAt": 1,
        "updatedAt": 1,
    });
    let payload = json!({
        "goal": goal,
        "roundsStarted": 0,
        "createdAt": 1,
        "updatedAt": 1,
    });
    let text = format!("<goal_state>{payload}</goal_state>");
    let row = json!({
        "type": "user/message",
        "seq": 0,
        "time": 1,
        "data": {
            "id": "g1",
            "role": "user",
            "content": [{ "type": "text", "text": text }],
            "source": {
                "kind": "goal",
                "goalId": "G",
                "revision": 1,
                "round": 0,
                "change": change,
            },
        },
        "surfaceOp": "append",
    })
    .to_string();
    let log = v1_to_v2(log(&[&row])).unwrap();

    assert_eq!(types(&log), vec!["goal/change", "user/message"]);
    assert_eq!(log.events[0].data["operation"], json!("create"));
    assert_eq!(
        log.events[1].data["source"],
        json!({ "kind": "plugin", "plugin": "goal" })
    );
}

#[test]
fn references_are_remapped_after_the_collapse() {
    let log = v1_to_v2(log(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"user/message","seq":2,"time":3,"data":{"id":"u1","role":"user","content":[{"type":"text","text":"q"}],"source":{"kind":"user"}},"surfaceOp":"append"}"#,
        r#"{"type":"assistant/chunk","seq":3,"time":4,"data":{"turn":1,"step":1,"chunk":{"type":"text-delta","index":0,"text":"a"}}}"#,
        r#"{"type":"assistant/chunk","seq":4,"time":5,"data":{"turn":1,"step":1,"chunk":{"type":"finish","reason":{"kind":"stop"}}}}"#,
        r#"{"type":"assistant/message","seq":5,"time":6,"data":{"turn":1,"step":1,"message":{"id":"a1","role":"assistant","content":[],"source":{"kind":"model","provider":"p","model":"m"}}},"sourceEventSeqs":[3,4],"surfaceOp":"append"}"#,
        r#"{"type":"step/end","seq":6,"time":7,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"user/message","seq":7,"time":8,"data":{"id":"u2","role":"user","content":[{"type":"text","text":"late"}],"source":{"kind":"user"}},"surfaceOp":"append"}"#,
        r#"{"type":"session/title","seq":8,"time":9,"data":{"title":"t","messageSeqs":[7],"source":{"kind":"fallback"}}}"#,
        r#"{"type":"compaction/start","seq":9,"time":10,"data":{"compactionId":"c","turn":1}}"#,
        r#"{"type":"compaction/summary","seq":10,"time":11,"data":{"compactionId":"c","summary":[{"type":"text","text":"s"}],"shadowedRange":{"start":2,"end":7},"shadowedSeqs":[2,7],"shadowedTokenCount":2,"provider":"p","model":"m"}}"#,
        r#"{"type":"compaction/end","seq":11,"time":12,"data":{"compactionId":"c","turn":1}}"#,
        r#"{"type":"turn/end","seq":12,"time":13,"data":{"turn":1,"reason":{"kind":"completed"}}}"#,
    ]))
    .unwrap();

    // Source 2 -> 2, source 5 (message) -> 3, source 7 -> 5.
    assert_eq!(log.events[6].data["messageSeqs"], json!([5]));
    assert_eq!(log.events[8].data["shadowedSeqs"], json!([2, 5]));
    assert_eq!(
        log.events[8].data["shadowedRange"],
        json!({ "start": 2, "end": 5 })
    );
}

#[test]
fn a_resume_splice_inserts_the_missing_turn_end() {
    let log = v1_to_v2(log(&[
        r#"{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}"#,
        r#"{"type":"step/start","seq":1,"time":2,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"step/end","seq":2,"time":3,"data":{"turn":1,"step":1}}"#,
        r#"{"type":"agent/inbox/spliced","seq":3,"time":4,"data":{"target":"next-turn","start":0,"inserted":[{"id":"i1","role":"user","content":[{"type":"text","text":"next"}],"source":{"kind":"user"}}]}}"#,
        r#"{"type":"turn/start","seq":4,"time":5,"data":{"turn":2}}"#,
    ]))
    .unwrap();

    assert_eq!(
        types(&log),
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
    assert_eq!(log.events[5].data["turn"], json!(2));
}
