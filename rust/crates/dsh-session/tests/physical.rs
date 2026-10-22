//! The released v4 physical envelope: row framing, header framing, and the
//! `sourceEventSeqs` range codec.
//!
//! The byte-level evidence lives in `fixtures/README.md`; these tests pin the
//! rules the shipped TypeScript codec applies, so a change that breaks the
//! physical framing is caught here rather than in a byte comparison.

use dsh_session::physical::{
    decode_physical_header, decode_physical_log, decode_row, decode_seq_ranges,
    encode_physical_header, encode_physical_log, encode_physical_row, encode_seq_ranges,
    PHYSICAL_HEADER_TYPE,
};
use dsh_session::{SessionEvent, SessionHeader, SessionOrigin};
use serde_json::{json, Value};

fn physical_header_value() -> Value {
    json!({
        "type": "session",
        "version": 4,
        "id": "session-1",
        "createdAt": 1760000000000_u64,
        "isSeeded": false,
        "delegationDepth": 0,
    })
}

#[test]
fn the_physical_header_carries_the_session_discriminator() {
    let header = decode_physical_header(&physical_header_value()).expect("the header decodes");
    assert_eq!(header.id.as_str(), "session-1");
    assert_eq!(header.created_at, 1760000000000);
    assert!(!header.is_seeded);
    // The TypeScript physical header requires `delegationDepth`; the logical
    // header keeps it, so the round trip does not lose it.
    assert_eq!(header.delegation_depth, Some(0));
}

#[test]
fn a_header_without_the_discriminator_is_refused() {
    let mut value = physical_header_value();
    value.as_object_mut().unwrap().remove("type");
    let error = decode_physical_header(&value).expect_err("the header is refused");
    assert!(
        error
            .to_string()
            .contains("expected released v4 physical Session header"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn the_physical_header_requires_the_delegation_depth() {
    let mut value = physical_header_value();
    value.as_object_mut().unwrap().remove("delegationDepth");
    let error = decode_physical_header(&value).expect_err("the header is refused");
    assert!(
        error.to_string().contains("delegationDepth"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn the_physical_header_keeps_the_canonical_key_order() {
    let header = decode_physical_header(&physical_header_value()).expect("the header decodes");
    let encoded = encode_physical_header(&header);
    let keys: Vec<&str> = encoded
        .as_object()
        .expect("the header is an object")
        .keys()
        .map(String::as_str)
        .collect();
    // The TypeScript writer emits `type`, `version`, `id`, `createdAt`, then
    // the optional members, then `isSeeded`, `origin`, `delegationDepth`, and
    // finally `agentPreset`. The order is observable on the wire.
    assert_eq!(
        keys,
        vec![
            "type",
            "version",
            "id",
            "createdAt",
            "isSeeded",
            "delegationDepth",
        ]
    );
    assert_eq!(
        encoded.as_object().unwrap().get("type").unwrap(),
        &json!(PHYSICAL_HEADER_TYPE)
    );
}

#[test]
fn an_optional_header_member_keeps_its_place_in_the_order() {
    let mut value = physical_header_value();
    {
        let map = value.as_object_mut().unwrap();
        map.insert("cwd".to_string(), json!("C:/work"));
        map.insert("parentSession".to_string(), json!("session-0"));
        map.insert("origin".to_string(), json!("subagent"));
        map.insert("agentPreset".to_string(), json!("preset-a"));
    }
    let header = decode_physical_header(&value).expect("the header decodes");
    assert_eq!(header.cwd.as_deref(), Some("C:/work"));
    assert_eq!(
        header.parent_session.as_ref().map(|id| id.as_str()),
        Some("session-0")
    );
    assert_eq!(header.origin, Some(SessionOrigin::Subagent));
    assert_eq!(header.agent_preset.as_deref(), Some("preset-a"));

    let encoded = encode_physical_header(&header);
    let keys: Vec<&str> = encoded
        .as_object()
        .expect("the header is an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        vec![
            "type",
            "version",
            "id",
            "createdAt",
            "cwd",
            "parentSession",
            "isSeeded",
            "origin",
            "delegationDepth",
            "agentPreset",
        ]
    );
}

#[test]
fn an_origin_other_than_subagent_is_refused() {
    let mut value = physical_header_value();
    value
        .as_object_mut()
        .unwrap()
        .insert("origin".to_string(), json!("primary"));
    let error = decode_physical_header(&value).expect_err("the header is refused");
    assert!(
        error.to_string().contains("origin must be \"subagent\""),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn a_physical_row_carries_the_envelope_members() {
    let row = decode_row(
        &json!({
            "type": "turn/start",
            "seq": 0,
            "time": 1760000000000_u64,
            "data": { "turn": 1 },
        }),
        0,
    )
    .expect("the row decodes");
    assert_eq!(row.r#type, "turn/start");
    assert_eq!(row.seq, 0);
    assert_eq!(row.time, 1760000000000);
    assert_eq!(row.data.get("turn"), Some(&json!(1)));
    assert_eq!(row.ignorable, None);
    assert_eq!(row.source_event_seqs, None);
    assert_eq!(row.surface_op, None);
}

#[test]
fn a_physical_row_requires_the_envelope_members() {
    for missing in ["type", "seq", "time", "data"] {
        let mut row = json!({
            "type": "turn/start",
            "seq": 0,
            "time": 1760000000000_u64,
            "data": { "turn": 1 },
        });
        row.as_object_mut().unwrap().remove(missing);
        let error = decode_row(&row, 0).expect_err("the row is refused");
        assert!(
            error
                .to_string()
                .contains(&format!("lacks required field {missing}")),
            "unexpected diagnostic for {missing}: {error}"
        );
    }
}

#[test]
fn a_physical_row_refuses_an_unexpected_member() {
    let row = json!({
        "type": "turn/start",
        "seq": 0,
        "time": 1760000000000_u64,
        "data": { "turn": 1 },
        "surprise": 1,
    });
    let error = decode_row(&row, 0).expect_err("the row is refused");
    assert!(
        error.to_string().contains("has unexpected field surprise"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn ignorable_must_be_true_when_present() {
    let row = json!({
        "type": "turn/start",
        "seq": 0,
        "time": 1760000000000_u64,
        "data": { "turn": 1 },
        "ignorable": false,
    });
    let error = decode_row(&row, 0).expect_err("the row is refused");
    assert!(
        error
            .to_string()
            .contains("ignorable must be true when present"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn an_unknown_event_type_needs_the_ignorable_marker() {
    let row = json!({
        "type": "plugin:fixture/acme",
        "seq": 0,
        "time": 1760000000000_u64,
        "data": { "value": 1 },
    });
    let error = decode_row(&row, 0).expect_err("the row is refused");
    assert!(
        error.to_string().contains("unknown event type"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn an_unknown_event_type_with_the_ignorable_marker_is_admitted() {
    let row = decode_row(
        &json!({
            "type": "plugin:fixture/acme",
            "seq": 0,
            "time": 1760000000000_u64,
            "data": { "value": 1 },
            "ignorable": true,
        }),
        0,
    )
    .expect("the row decodes");
    assert_eq!(row.r#type, "plugin:fixture/acme");
    assert_eq!(row.ignorable, Some(true));
}

#[test]
fn the_source_event_seqs_ranges_expand_to_a_flat_list() {
    let row = decode_row(
        &json!({
            "type": "turn/start",
            "seq": 5,
            "time": 1760000000000_u64,
            "data": { "turn": 1 },
            "sourceEventSeqs": [0, [2, 4]],
        }),
        0,
    )
    .expect("the row decodes");
    assert_eq!(row.source_event_seqs, Some(vec![0, 2, 3, 4]));
}

#[test]
fn the_source_event_seqs_ranges_must_be_strictly_increasing() {
    // The positions are unique but not increasing: the TypeScript reader
    // rejects this list with the "strictly increasing" diagnostic.
    let row = json!({
        "type": "turn/start",
        "seq": 5,
        "time": 1760000000000_u64,
        "data": { "turn": 1 },
        "sourceEventSeqs": [[3, 4], 2],
    });
    let error = decode_row(&row, 0).expect_err("the row is refused");
    assert!(
        error
            .to_string()
            .contains("sourceEventSeqs ranges must be strictly increasing"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn the_source_event_seqs_ranges_must_not_reach_the_rows_own_seq() {
    let row = json!({
        "type": "turn/start",
        "seq": 3,
        "time": 1760000000000_u64,
        "data": { "turn": 1 },
        "sourceEventSeqs": [[1, 3]],
    });
    let error = decode_row(&row, 0).expect_err("the row is refused");
    assert!(
        error
            .to_string()
            .contains("sourceEventSeqs range exceeds its event seq"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn the_source_event_seqs_must_not_repeat_a_position() {
    let row = json!({
        "type": "turn/start",
        "seq": 5,
        "time": 1760000000000_u64,
        "data": { "turn": 1 },
        "sourceEventSeqs": [2, [1, 2]],
    });
    let error = decode_row(&row, 0).expect_err("the row is refused");
    assert!(
        error
            .to_string()
            .contains("sourceEventSeqs ranges must contain unique earlier seqs"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn the_seq_range_codec_round_trips() {
    for values in [
        vec![],
        vec![0],
        vec![0, 1],
        vec![0, 1, 2],
        vec![0, 1, 2, 3],
        vec![1, 5, 9],
        vec![2, 3, 7, 8, 9],
    ] {
        let encoded = encode_seq_ranges(&values);
        let decoded = decode_seq_ranges(&Value::Array(encoded.clone()), 100, 0)
            .expect("the encoded ranges decode");
        assert_eq!(decoded, values, "round trip failed for {values:?}");
    }
}

#[test]
fn the_seq_range_codec_writes_a_pair_for_a_run_of_three() {
    assert_eq!(encode_seq_ranges(&[0, 1, 2]), vec![json!([0, 2])]);
    assert_eq!(encode_seq_ranges(&[0, 1]), vec![json!(0), json!(1)]);
    assert_eq!(encode_seq_ranges(&[0]), vec![json!(0)]);
}

#[test]
fn the_seq_range_codec_keeps_a_non_increasing_list_verbatim() {
    // A list that is not strictly increasing cannot be expressed as ranges;
    // the TypeScript writer keeps it as singletons rather than mis-encoding it.
    assert_eq!(
        encode_seq_ranges(&[3, 1, 2]),
        vec![json!(3), json!(1), json!(2)]
    );
}

#[test]
fn a_physical_log_round_trips_through_the_physical_writer() {
    let header = SessionHeader {
        version: 4,
        id: dsh_session::SessionId::new("session-round-trip").expect("the id is valid"),
        created_at: 1760000000000,
        cwd: None,
        parent_session: None,
        is_seeded: false,
        origin: None,
        delegation_depth: Some(0),
        agent_preset: None,
    };
    let events = vec![
        SessionEvent::turn_start(1),
        SessionEvent::turn_end(1, dsh_session::TurnEndReason::Completed),
    ];
    let session =
        dsh_session::Session::restore(header.clone(), events).expect("the session restores");

    let times = vec![1760000000000_u64, 1760000000001];
    let text = encode_physical_log(&session, &times).expect("the log encodes");
    let decoded = decode_physical_log(&text).expect("the log decodes");

    assert_eq!(decoded.events().len(), session.events().len());
    assert_eq!(decoded.events(), session.events());
    assert_eq!(decoded.header().id.as_str(), session.header().id.as_str());
    assert_eq!(decoded.header().created_at, session.header().created_at);
}

#[test]
fn a_physical_row_carries_the_payload_under_data() {
    let row = encode_physical_row(&SessionEvent::turn_start(7), 3, 1760000000009);
    let map = row.as_object().expect("the row is an object");
    assert_eq!(map.get("type"), Some(&json!("turn/start")));
    assert_eq!(map.get("seq"), Some(&json!(3)));
    assert_eq!(map.get("time"), Some(&json!(1760000000009_u64)));
    // The flat event's `turn` member moves under `data`.
    assert_eq!(map.get("data").unwrap().get("turn"), Some(&json!(7)));
    // The flat event carries no `turn` at the top level.
    assert_eq!(map.get("turn"), None);
}

#[test]
fn a_physical_log_with_a_seq_gap_is_refused() {
    let text = format!(
        "{}\n{}\n",
        physical_header_value(),
        json!({
            "type": "turn/start",
            "seq": 3,
            "time": 1760000000000_u64,
            "data": { "turn": 1 },
        })
    );
    let error = decode_physical_log(&text).expect_err("the log is refused");
    assert!(
        error.to_string().contains("seq gap"),
        "unexpected diagnostic: {error}"
    );
}
