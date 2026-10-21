//! Session log behavior: sequence branding, append admission, the folds, and
//! JSONL generation governance.

use dsh_llm::{ContentBlock, LlmCallConfig, MessageId, SystemMessage, UserMessage};
use dsh_session::*;
use serde_json::{json, Map};

fn demo_session() -> Session {
    Session::create(SessionId::new("s1").unwrap(), SessionMeta::default(), 0)
}

fn config(model: &str) -> LlmCallConfig {
    LlmCallConfig {
        provider: "p".to_string(),
        model: model.to_string(),
        reasoning_effort: None,
        temperature: None,
        max_tokens: None,
        stop: None,
    }
}

#[test]
fn append_assigns_contiguous_sequence_numbers() {
    let mut session = demo_session();
    let first = session.append(SessionEvent::turn_start(1)).unwrap();
    let second = session.append(SessionEvent::step_start(1, 1)).unwrap();
    assert_eq!(first.get(), 0);
    assert_eq!(second.get(), 1);
    assert_eq!(session.len(), 2);
    assert!(!session.is_empty());
    assert_eq!(session.event(first).unwrap(), &SessionEvent::turn_start(1));
}

#[test]
fn session_seq_rejects_values_beyond_the_safe_integer_bound() {
    assert!(matches!(
        SessionSeq::new(MAX_SAFE_INTEGER + 1),
        Err(SessionError::UnsafeInteger { .. })
    ));
    assert!(SessionSeq::new(MAX_SAFE_INTEGER).is_ok());
}

#[test]
fn append_rejects_data_that_is_not_lossless_json() {
    let mut session = demo_session();
    let header = EpochHeader {
        config: LlmCallConfig {
            temperature: Some(f64::NAN),
            ..config("m1")
        },
        adapter_defaults: None,
        tools: None,
    };
    let error = session
        .append(SessionEvent::request_header(
            header,
            RequestHeaderReason::Initial,
        ))
        .unwrap_err();
    assert!(matches!(error, SessionError::NotJson(_)), "{error}");
}

#[test]
fn append_rejects_non_canonical_request_headers() {
    let mut session = demo_session();
    let header = EpochHeader {
        config: config("m1"),
        adapter_defaults: None,
        tools: Some(vec![]),
    };
    let error = session
        .append(SessionEvent::request_header(
            header,
            RequestHeaderReason::Initial,
        ))
        .unwrap_err();
    assert!(
        matches!(error, SessionError::NonCanonicalHeader(_)),
        "{error}"
    );

    let header = EpochHeader {
        config: config("m1"),
        adapter_defaults: Some(dsh_llm::LlmCallConfigAdapterDefaults::default()),
        tools: None,
    };
    let error = session
        .append(SessionEvent::request_header(
            header,
            RequestHeaderReason::Initial,
        ))
        .unwrap_err();
    assert!(
        matches!(error, SessionError::NonCanonicalHeader(_)),
        "{error}"
    );
}

#[test]
fn append_rejects_the_legacy_v0_header_delta_event() {
    let mut session = demo_session();
    let event = SessionEvent::extension_event("request/header-delta", Map::new());
    let error = session.append(event).unwrap_err();
    assert!(matches!(error, SessionError::LegacyEvent(_)), "{error}");
}

#[test]
fn core_events_round_trip_through_json_with_their_wire_tags() {
    let event = SessionEvent::tool_call(
        1,
        2,
        dsh_llm::ToolCallId::new("call-9"),
        "echo",
        "{\"text\":\"x\"}",
    );
    let text = serde_json::to_string(&event).unwrap();
    assert!(text.contains("\"type\":\"tool/call\""), "{text}");
    assert!(text.contains("\"callId\":\"call-9\""), "{text}");
    let back: SessionEvent = serde_json::from_str(&text).unwrap();
    assert_eq!(back, event);
}

#[test]
fn interrupted_and_series_markers_serialize_only_when_true() {
    let event = SessionEvent::assistant_message(
        1,
        1,
        dsh_llm::AssistantMessage::new(MessageId::new("a"), vec![], "p", "m"),
        vec![],
        None,
        true,
    );
    let text = serde_json::to_string(&event).unwrap();
    assert!(text.contains("\"interrupted\":true"), "{text}");
    let event = SessionEvent::request_header_series(
        EpochHeader {
            config: config("m1"),
            adapter_defaults: None,
            tools: None,
        },
        RequestHeaderReason::Series,
    );
    let text = serde_json::to_string(&event).unwrap();
    assert!(text.contains("\"startsSeries\":true"), "{text}");
}

#[test]
fn extension_events_survive_verbatim() {
    let mut payload = Map::new();
    payload.insert("summary".to_string(), json!("folded"));
    let event = SessionEvent::extension_event("compaction/summary", payload);
    let text = serde_json::to_string(&event).unwrap();
    let back: SessionEvent = serde_json::from_str(&text).unwrap();
    assert_eq!(back, event);
    assert_eq!(back.extension().unwrap().kind(), "compaction/summary");
}

/// The `type` discriminator is the **first** key on the wire.
///
/// Every TypeScript row opens with its tag — the format tests are all
/// `{ type: '...', ... }` — and `serde_json`'s `preserve_order` makes insertion
/// order observable, so appending the stamp instead of prepending it would
/// produce a log that round-trips perfectly and still differs byte-for-byte
/// from one TypeScript wrote.
#[test]
fn an_extension_event_opens_with_its_type_tag() {
    let mut payload = Map::new();
    payload.insert("zeta".to_string(), json!(1));
    payload.insert("alpha".to_string(), json!(2));
    let text = serde_json::to_string(&SessionEvent::extension_event(
        "compaction/summary",
        payload,
    ))
    .unwrap();
    assert_eq!(
        text, r#"{"type":"compaction/summary","zeta":1,"alpha":2}"#,
        "the tag comes first and the declared field order is kept, not sorted"
    );
}

#[test]
fn unknown_event_types_are_ignorable_not_fatal() {
    let event: SessionEvent =
        serde_json::from_str(r#"{"type":"hook/invoked","hook":"x"}"#).unwrap();
    assert_eq!(event.extension().unwrap().kind(), "hook/invoked");
}

#[test]
fn fold_request_header_selects_the_latest_snapshot() {
    let mut session = demo_session();
    session
        .append(SessionEvent::request_header(
            EpochHeader {
                config: config("m1"),
                adapter_defaults: None,
                tools: None,
            },
            RequestHeaderReason::Initial,
        ))
        .unwrap();
    session
        .append(SessionEvent::request_header(
            EpochHeader {
                config: config("m2"),
                adapter_defaults: None,
                tools: None,
            },
            RequestHeaderReason::Change,
        ))
        .unwrap();
    let header = fold_request_header(session.events()).unwrap();
    assert_eq!(header.config.model, "m2");
}

#[test]
fn fold_turn_boundary_tracks_the_open_turn_and_the_last_step() {
    let mut session = demo_session();
    session.append(SessionEvent::turn_start(1)).unwrap();
    session.append(SessionEvent::step_start(1, 1)).unwrap();
    let boundary = fold_turn_boundary(session.events());
    assert_eq!(boundary.open_turn_start_seq.unwrap().get(), 0);
    assert_eq!(boundary.last_step_start_seq.unwrap().get(), 1);
    assert_eq!(boundary.last_turn, 1);

    session.append(SessionEvent::step_end(1, 1)).unwrap();
    session
        .append(SessionEvent::turn_end(1, TurnEndReason::Completed))
        .unwrap();
    let boundary = fold_turn_boundary(session.events());
    assert!(boundary.open_turn_start_seq.is_none());
    assert_eq!(
        boundary.last_step_boundary,
        Some(StepBoundary {
            kind: StepBoundaryKind::End,
            seq: SessionSeq::new(2).unwrap()
        })
    );
}

#[test]
fn derive_messages_projects_only_model_visible_events() {
    let mut session = demo_session();
    session.append(SessionEvent::turn_start(1)).unwrap();
    session.append(SessionEvent::step_start(1, 1)).unwrap();
    session
        .append(SessionEvent::user_message(UserMessage::new(
            MessageId::random(),
            vec![ContentBlock::text("hi")],
        )))
        .unwrap();
    session
        .append(SessionEvent::system_message(
            1,
            1,
            SystemMessage::new(MessageId::random(), vec![ContentBlock::text("be terse")]),
        ))
        .unwrap();
    session
        .append(SessionEvent::turn_end(1, TurnEndReason::Completed))
        .unwrap();
    let messages = derive_messages(session.events());
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role(), "user");
    assert_eq!(messages[1].role(), "system");
}

#[test]
fn jsonl_publication_round_trips() {
    let mut session = demo_session();
    session.append(SessionEvent::turn_start(1)).unwrap();
    session
        .append(SessionEvent::user_message(UserMessage::new(
            MessageId::random(),
            vec![ContentBlock::text("hi")],
        )))
        .unwrap();
    let directory = std::env::temp_dir().join(format!("dsh-session-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);

    let path = write_session(&directory, &session).unwrap();
    assert!(path.to_string_lossy().ends_with("session.v4.jsonl"));
    assert_eq!(read_session(&directory).unwrap(), session);

    // Republishing replaces the generation in place.
    write_session(&directory, &session).unwrap();
    assert_eq!(read_session(&directory).unwrap(), session);
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn the_reader_refuses_a_future_generation() {
    let mut session = demo_session();
    session.append(SessionEvent::turn_start(1)).unwrap();
    let directory = std::env::temp_dir().join(format!("dsh-session-future-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();

    let mut header = session.header().clone();
    header.version = 5;
    let mut text = serde_json::to_string(&header).unwrap();
    text.push('\n');
    for event in session.events() {
        text.push_str(&serde_json::to_string(event).unwrap());
        text.push('\n');
    }
    std::fs::write(directory.join("session.v5.jsonl"), text).unwrap();

    let error = read_session(&directory).unwrap_err();
    assert!(
        matches!(error, SessionError::UnsupportedVersion { found: 5, .. }),
        "{error}"
    );
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn generation_names_encode_and_decode() {
    assert_eq!(parse_generation("session.jsonl"), Some(0));
    assert_eq!(parse_generation("session.v4.jsonl"), Some(4));
    assert_eq!(parse_generation("session.v10.jsonl"), Some(10));
    assert_eq!(parse_generation("session.vN.jsonl"), None);
    assert_eq!(parse_generation("other.jsonl"), None);
    assert_eq!(session_file_name(0), "session.jsonl");
    assert_eq!(session_file_name(4), "session.v4.jsonl");
}

#[test]
fn restore_rejects_a_pre_current_generation() {
    let mut header = demo_session().header().clone();
    header.version = 3;
    let error = Session::restore(header, vec![]).unwrap_err();
    assert!(
        matches!(error, SessionError::MigrationRequired { found: 3, .. }),
        "{error}"
    );
}

#[test]
fn session_ids_must_not_be_empty() {
    assert!(matches!(
        SessionId::new(""),
        Err(SessionError::EmptySessionId)
    ));
    assert!(SessionId::new("s").is_ok());
}

/// The log is append-only and unbounded, so an event's inline footprint is
/// paid by every event — including the tiny lifecycle ones. Boxing the heavy
/// payloads keeps a bare `turn/start` from costing as much as a full
/// `assistant/message`; this test locks that in so a future field cannot
/// quietly regress it.
#[test]
fn an_event_stays_small_regardless_of_its_largest_variant() {
    use std::mem::size_of;

    let inline = size_of::<SessionEvent>();
    assert!(
        inline <= 128,
        "SessionEvent grew to {inline} bytes; box the new heavy payload"
    );

    // The heaviest payloads are heap-side, so they cost nothing inline.
    assert!(
        size_of::<EpochHeader>() > inline,
        "EpochHeader should be boxed, not inlined"
    );
}

/// Boxing must stay invisible on the wire: `Box<T>` serializes exactly as `T`.
#[test]
fn boxed_payloads_serialize_identically() {
    let event = SessionEvent::request_header(
        EpochHeader {
            config: config("m"),
            adapter_defaults: None,
            tools: None,
        },
        RequestHeaderReason::Initial,
    );
    let wire =
        serde_json::from_str::<serde_json::Value>(&serde_json::to_string(&event).unwrap()).unwrap();
    assert_eq!(
        wire,
        json!({
            "type": "request/header",
            "header": { "config": { "provider": "p", "model": "m" } },
            "reason": "initial"
        })
    );

    let back: SessionEvent = serde_json::from_value(wire).unwrap();
    assert_eq!(back, event);
}
