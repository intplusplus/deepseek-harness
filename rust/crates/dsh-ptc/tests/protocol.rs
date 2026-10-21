//! The message vocabulary inside a frame.
//!
//! Field order and strictness come from
//! `packages/ptc-runtime/ptc-runtime-node/src/protocol.ts` and the receive
//! paths of `persistent.ts`; the byte-level assertions pin the producer order.

use dsh_ptc::reserved::BindingErrorClass;
use dsh_ptc::{
    HostToProgram, NamespaceDeclaration, ProgramBootData, ProgramEnvelope, ProgramFailure,
    ProgramFailureKind, ProgramToHost, ProtocolError, Reply,
};
use serde_json::{json, Value};

const LIMIT: usize = 4_096;

fn encode(message: &ProgramToHost) -> String {
    message.to_json().to_string()
}

fn decode(value: Value) -> Result<ProgramToHost, ProtocolError> {
    ProgramToHost::from_json(&value)
}

#[test]
fn call_log_and_done_serialize_tag_first_in_producer_order() {
    assert_eq!(
        encode(&ProgramToHost::Call {
            id: 1,
            global: "tools".into(),
            name: "read".into(),
            args: json!(["path"]),
        }),
        r#"{"type":"call","id":1,"global":"tools","name":"read","args":["path"]}"#
    );
    assert_eq!(
        encode(&ProgramToHost::Log { text: "hi".into() }),
        r#"{"type":"log","text":"hi"}"#
    );
    assert_eq!(
        encode(&ProgramToHost::OutputLimit),
        r#"{"type":"output-limit"}"#
    );
    assert_eq!(
        encode(&ProgramToHost::Done {
            value: Some(json!([1])),
            error: None,
        }),
        r#"{"type":"done","value":[1]}"#
    );
    assert_eq!(
        encode(&ProgramToHost::Done {
            value: None,
            error: Some(ProgramFailure {
                kind: ProgramFailureKind::InvalidOutput,
                message: "not JSON".into(),
            }),
        }),
        r#"{"type":"done","error":{"kind":"invalid-output","message":"not JSON"}}"#
    );
}

#[test]
fn host_frames_serialize_tag_first_in_producer_order() {
    let data = ProgramBootData {
        code: "return 1".into(),
        namespaces: vec![NamespaceDeclaration {
            global: "tools".into(),
            names: vec!["read".into()],
            error_class: Some(BindingErrorClass::new("ToolCallError", "toolName")),
        }],
        max_output_bytes: 64,
    };
    let run = HostToProgram::Run {
        id: 7,
        cwd: "/work".into(),
        data: data.clone(),
    };
    assert_eq!(
        run.to_json().to_string(),
        r#"{"type":"run","id":7,"cwd":"/work","data":{"code":"return 1","namespaces":[{"global":"tools","names":["read"],"errorClass":{"name":"ToolCallError","memberNameProperty":"toolName"}}],"maxOutputBytes":64}}"#
    );
    assert_eq!(
        HostToProgram::Ready.to_json().to_string(),
        r#"{"type":"ready"}"#
    );

    assert_eq!(HostToProgram::from_json(&run.to_json()), Ok(run));
    assert_eq!(
        ProtocolError::UnknownField {
            kind: "run.data",
            field: "extra".into()
        },
        HostToProgram::from_json(&json!({
            "type": "run", "id": 1, "cwd": "/w",
            "data": { "code": "", "namespaces": [], "maxOutputBytes": 1, "extra": true }
        }))
        .expect_err("an undeclared field is refused")
    );
}

#[test]
fn replies_serialize_the_outcome_after_the_correlation_id() {
    assert_eq!(
        Reply::Ok {
            id: 2,
            value: json!([1]),
        }
        .to_json()
        .to_string(),
        r#"{"type":"reply","id":2,"ok":true,"value":[1]}"#
    );
    assert_eq!(
        Reply::Error {
            id: 3,
            message: "boom".into(),
        }
        .to_json()
        .to_string(),
        r#"{"type":"reply","id":3,"ok":false,"message":"boom"}"#
    );
    // `ok` is authoritative: the other field is undeclared for that branch.
    assert_eq!(
        Reply::from_json(&json!({ "type": "reply", "id": 2, "ok": true, "message": "x" })),
        Err(ProtocolError::MissingField {
            kind: "reply",
            field: "value"
        })
    );
    assert_eq!(
        Reply::from_json(&json!({ "type": "reply", "id": 2, "ok": false, "value": [1] })),
        Err(ProtocolError::MissingField {
            kind: "reply",
            field: "message"
        })
    );
}

#[test]
fn decoding_is_strict_about_tags_fields_and_types() {
    assert_eq!(
        decode(json!([1])),
        Err(ProtocolError::NotAnObject {
            kind: "program message",
            actual: "array"
        })
    );
    assert_eq!(
        decode(json!({ "type": "shout" })),
        Err(ProtocolError::UnknownType {
            value: "shout".into()
        })
    );
    assert_eq!(
        decode(json!({ "id": 1 })),
        Err(ProtocolError::MissingField {
            kind: "program message",
            field: "type"
        })
    );
    assert_eq!(
        decode(json!({ "type": 7 })),
        Err(ProtocolError::WrongType {
            kind: "program message",
            field: "type",
            expected: "a string"
        })
    );
    assert_eq!(
        decode(json!({ "type": "log" })),
        Err(ProtocolError::MissingField {
            kind: "log",
            field: "text"
        })
    );
    assert_eq!(
        decode(json!({ "type": "log", "text": 1 })),
        Err(ProtocolError::WrongType {
            kind: "log",
            field: "text",
            expected: "a string"
        })
    );
    assert_eq!(
        decode(json!({ "type": "log", "text": "a", "level": "warn" })),
        Err(ProtocolError::UnknownField {
            kind: "log",
            field: "level".into()
        })
    );
    assert_eq!(
        decode(json!({ "type": "call", "id": -1, "global": "tools", "name": "read", "args": [] })),
        Err(ProtocolError::WrongType {
            kind: "call",
            field: "id",
            expected: "a non-negative integer"
        })
    );
    // The single argument is always a wire array, never the value itself.
    assert_eq!(
        decode(json!({ "type": "call", "id": 1, "global": "tools", "name": "read", "args": {} })),
        Err(ProtocolError::WrongType {
            kind: "call",
            field: "args",
            expected: "a flat JSON wire array"
        })
    );
}

#[test]
fn a_completion_value_must_be_a_wire_array_like_every_other_carried_value() {
    // The protocol layer guarantees "this is a flat token array"; the wire
    // layer then validates the tokens themselves.
    assert_eq!(
        decode(json!({ "type": "done", "value": { "a": 1 } })),
        Err(ProtocolError::WrongType {
            kind: "done",
            field: "value",
            expected: "a flat JSON wire array"
        })
    );
    assert_eq!(
        decode(json!({ "type": "done", "value": null })),
        Err(ProtocolError::WrongType {
            kind: "done",
            field: "value",
            expected: "a flat JSON wire array"
        })
    );
    assert_eq!(
        decode(json!({ "type": "done", "value": [1] })),
        Ok(ProgramToHost::Done {
            value: Some(json!([1])),
            error: None
        })
    );
}

#[test]
fn a_program_may_not_report_a_host_owned_failure_class() {
    assert_eq!(
        decode(json!({
            "type": "done",
            "error": { "kind": "timeout", "message": "took too long" }
        })),
        Err(ProtocolError::UnknownFailureKind {
            value: "timeout".into()
        })
    );
    // The three classes a program can know about are accepted.
    for kind in ["exception", "invalid-output", "output-limit"] {
        assert_eq!(
            decode(json!({ "type": "done", "error": { "kind": kind, "message": "m" } })),
            Ok(ProgramToHost::Done {
                value: None,
                error: Some(ProgramFailure {
                    kind: ProgramFailureKind::parse(kind).expect("known kind"),
                    message: "m".into(),
                }),
            })
        );
    }
}

#[test]
fn the_persistent_envelope_carries_the_evaluation_id() {
    let envelope = ProgramEnvelope {
        id: 4,
        payload: ProgramToHost::Log { text: "x".into() },
    };
    assert_eq!(
        envelope.to_json().to_string(),
        r#"{"id":4,"payload":{"type":"log","text":"x"}}"#
    );
    assert_eq!(
        ProgramEnvelope::from_json(&envelope.to_json()),
        Ok(envelope)
    );

    // An envelope without its id is refused, because the host pairs frames to
    // the active evaluation by it.
    assert_eq!(
        ProgramEnvelope::from_json(&json!({ "payload": { "type": "done" } })),
        Err(ProtocolError::MissingField {
            kind: "envelope",
            field: "id"
        })
    );
}

#[test]
fn boot_data_round_trips_without_an_error_class() {
    let data = ProgramBootData {
        code: "1".into(),
        namespaces: vec![NamespaceDeclaration {
            global: "tools".into(),
            names: vec![],
            error_class: None,
        }],
        max_output_bytes: 8,
    };
    assert_eq!(ProgramBootData::from_json(&data.to_json()), Ok(data));
}

#[test]
fn every_message_kind_is_named_for_diagnostics() {
    assert_eq!(ProgramToHost::OutputLimit.kind(), "output-limit");
    assert_eq!(
        ProgramToHost::Done {
            value: None,
            error: None
        }
        .kind(),
        "done"
    );
    assert_eq!(HostToProgram::Ready.kind(), "ready");
    assert_eq!(
        HostToProgram::Reply(Reply::Error {
            id: 1,
            message: "x".into()
        })
        .kind(),
        "reply"
    );
}

#[test]
fn a_message_that_fits_the_frame_limit_is_transportable() {
    let message = ProgramToHost::Log {
        text: "x".repeat(16),
    };
    let frame = dsh_ptc::encode_frame(&message.to_json(), LIMIT).expect("fits");
    let mut decoder = dsh_ptc::FrameDecoder::new(LIMIT);
    let frames = decoder.push(&frame).expect("decodes");
    assert_eq!(frames.len(), 1);
    assert_eq!(decode(frames[0].clone()), Ok(message));
}
