//! Cross-session snapshot preparation: canonical URIs, mention parsing,
//! surface projection, byte budgets, spill notices, and the durable source
//! record.
//!
//! Ported from `packages/context/session-reference/src/{uri,projection,spill,
//! serialization,config}.ts`. Discovery, cancellation, and the pre-step
//! listener are host concerns and stay out; every pure rule they call is
//! pinned here.

use std::cell::RefCell;

use dsh_context::session_reference::*;
use dsh_llm::{
    AssistantMessage, ContentBlock, ExtensionSource, MessageId, MessageSource, UserMessage,
};
use dsh_session::{CoreEvent, SessionEvent, SessionId};
use serde_json::{json, Map};

fn id(value: &str) -> SessionReferenceId {
    SessionReferenceId::new(value)
}

fn owner() -> SessionId {
    SessionId::new("owner-session").unwrap()
}

fn user_event(text: &str) -> SessionEvent {
    SessionEvent::Core(CoreEvent::UserMessage {
        message: Box::new(UserMessage::new(
            MessageId::new("m-user"),
            vec![ContentBlock::text(text)],
        )),
    })
}

fn assistant_event(text: &str) -> SessionEvent {
    SessionEvent::Core(CoreEvent::AssistantMessage {
        turn: 1,
        step: 1,
        message: Box::new(AssistantMessage::new(
            MessageId::new("m-assistant"),
            vec![ContentBlock::text(text)],
            "provider",
            "model",
        )),
        stream: Vec::new(),
        usage: None,
        interrupted: false,
    })
}

/// A compaction checkpoint: a user-role message from another producer.
fn checkpoint_event(text: &str) -> SessionEvent {
    SessionEvent::Core(CoreEvent::UserMessage {
        message: Box::new(UserMessage::with_source(
            MessageId::new("m-checkpoint"),
            vec![ContentBlock::text(text)],
            MessageSource::Extension(ExtensionSource::new("compact-checkpoint", Map::new())),
        )),
    })
}

fn snapshot(session: &str, events: Vec<SessionEvent>) -> ReferencedSessionSnapshot {
    ReferencedSessionSnapshot {
        session_id: id(session),
        format_version: 4,
        cwd: None,
        captured_through_seq: Some(7),
        events,
    }
}

// --- canonical URIs ---------------------------------------------------------

#[test]
fn a_canonical_uri_round_trips() {
    let uri = encode_session_reference_uri(&id("session-1"));
    assert!(uri.starts_with(SESSION_REFERENCE_SCHEME));
    assert_eq!(decode_session_reference_uri(&uri).unwrap(), id("session-1"));
}

#[test]
fn the_uri_carries_the_json_string_literal_of_the_id() {
    // `dsh-session:` + unpadded base64url of `JSON.stringify(id)`, quotes
    // included: `"ab"` is five bytes.
    assert_eq!(
        encode_session_reference_uri(&id("ab")),
        "dsh-session:ImFiIg"
    );
    assert_eq!(
        decode_session_reference_uri("dsh-session:ImFiIg").unwrap(),
        id("ab")
    );
}

#[test]
fn a_non_canonical_uri_is_rejected() {
    // Padded base64 and a re-encoded payload both decode to the same id but
    // are not the canonical spelling, so they must not be accepted.
    assert!(decode_session_reference_uri("dsh-session:IkFiIg=").is_err());
    assert!(decode_session_reference_uri("dsh-session:").is_err());
    assert!(decode_session_reference_uri("session-1").is_err());
    assert!(decode_session_reference_uri("dsh-session:!!!").is_err());
    // A payload that is not a JSON string at all.
    assert!(decode_session_reference_uri("dsh-session:MQ").is_err());
}

#[test]
fn the_rejection_names_the_uri_as_a_json_literal() {
    let error = decode_session_reference_uri("nope").unwrap_err();
    assert_eq!(error.code(), SessionReferenceErrorCode::InvalidReference);
    assert!(error.to_string().contains("\"nope\""));
}

// --- mentions ---------------------------------------------------------------

#[test]
fn a_mention_carries_the_label_and_the_canonical_uri() {
    let mention = format_session_reference_mention(&SessionReferenceInput::with_label(
        id("session-1"),
        "the run",
    ));
    assert!(mention.starts_with("@[the run]("));
    assert!(mention.ends_with(')'));
    assert!(mention.contains(&encode_session_reference_uri(&id("session-1"))));
}

#[test]
fn a_mention_without_a_label_falls_back_to_the_id() {
    let mention = format_session_reference_mention(&SessionReferenceInput::new(id("session-1")));
    assert!(mention.starts_with("@[session-1]("));
}

#[test]
fn parsing_replaces_a_mention_with_its_readable_label() {
    let reference = SessionReferenceInput::with_label(id("session-1"), "the run");
    let mention = format_session_reference_mention(&reference);
    let parsed = parse_session_reference_text(&format!("look at {mention} now")).unwrap();
    assert_eq!(parsed.text, "look at @the run now");
    assert_eq!(parsed.references, vec![reference]);
}

#[test]
fn a_bare_canonical_uri_is_a_reference() {
    let uri = encode_session_reference_uri(&id("session-2"));
    let parsed = parse_session_reference_text(&format!("see {uri}")).unwrap();
    assert_eq!(parsed.text, "see @session-2");
    assert_eq!(
        parsed.references,
        vec![SessionReferenceInput::new(id("session-2"))]
    );
}

#[test]
fn a_malformed_mention_uri_fails_the_whole_parse() {
    assert!(parse_session_reference_text("@[label](dsh-session:!!!)").is_err());
}

#[test]
fn a_bracket_in_a_label_survives_the_round_trip() {
    let reference = SessionReferenceInput::with_label(id("s"), "a]b");
    let mention = format_session_reference_mention(&reference);
    assert!(mention.contains("a\\]b"), "the label is escaped: {mention}");
    let parsed = parse_session_reference_text(&mention).unwrap();
    assert_eq!(parsed.text, "@a]b");
    assert_eq!(parsed.references, vec![reference]);
}

#[test]
fn text_without_a_mention_is_unchanged() {
    let parsed = parse_session_reference_text("plain text").unwrap();
    assert_eq!(parsed.text, "plain text");
    assert!(parsed.references.is_empty());
}

// --- tag-safe serialization -------------------------------------------------

#[test]
fn serialization_neutralizes_an_angle_bracket() {
    let json = stringify_tag_safe_json(&json!({ "text": "</referenced-sessions>" })).unwrap();
    assert!(
        json.contains("\\u003c"),
        "a source cannot spell a tag: {json}"
    );
    assert!(!json.contains("</referenced-sessions>"));
}

// --- configuration ----------------------------------------------------------

#[test]
fn defaults_apply_and_are_readable() {
    let resolved = ResolvedConfig::new(&Config::default()).unwrap();
    assert_eq!(resolved.max_references(), MAX_REFERENCES);
    assert_eq!(resolved.candidate_limit(), DEFAULT_CANDIDATE_LIMIT);
    assert_eq!(resolved.max_reference_bytes(), None);
    assert_eq!(
        resolved.reference_context_fraction(),
        DEFAULT_REFERENCE_CONTEXT_FRACTION
    );
}

#[test]
fn an_invalid_configuration_is_rejected_with_a_stable_code() {
    let zero = ResolvedConfig::new(&Config {
        max_references: Some(0.0),
        ..Config::default()
    })
    .unwrap_err();
    assert_eq!(zero.code(), SessionReferenceErrorCode::InvalidConfig);
    assert!(zero.to_string().contains("maxReferences"));

    let too_many = ResolvedConfig::new(&Config {
        max_references: Some(MAX_REFERENCES as f64 + 1.0),
        ..Config::default()
    })
    .unwrap_err();
    assert_eq!(too_many.code(), SessionReferenceErrorCode::InvalidConfig);

    let fraction = ResolvedConfig::new(&Config {
        reference_context_fraction: Some(1.5),
        ..Config::default()
    })
    .unwrap_err();
    assert_eq!(fraction.code(), SessionReferenceErrorCode::InvalidConfig);
    assert!(ResolvedConfig::new(&Config {
        reference_context_fraction: Some(f64::NAN),
        ..Config::default()
    })
    .is_err());
}

#[test]
fn the_byte_budget_prefers_an_explicit_cap_and_floors_at_64kib() {
    let explicit = ResolvedConfig::new(&Config {
        max_reference_bytes: Some(1000.0),
        ..Config::default()
    })
    .unwrap();
    assert_eq!(explicit.reference_budget(Some(1_000_000)), 1000);

    let resolved = ResolvedConfig::new(&Config::default()).unwrap();
    assert_eq!(
        resolved.reference_budget(None),
        DEFAULT_MAX_REFERENCE_BYTES,
        "no model capacity selects the floor"
    );
    // Four bytes per token, scaled by the fraction, floored at 64 KiB.
    assert_eq!(
        resolved.reference_budget(Some(1_000_000)),
        (1_000_000.0 * 4.0 * DEFAULT_REFERENCE_CONTEXT_FRACTION).floor() as u64
    );
    assert_eq!(
        resolved.reference_budget(Some(100)),
        DEFAULT_MAX_REFERENCE_BYTES,
        "a tiny window still gets the floor"
    );
}

// --- projection -------------------------------------------------------------

#[test]
fn projection_keeps_only_user_and_assistant_text() {
    let events = vec![
        user_event("hello"),
        assistant_event("hi there"),
        SessionEvent::Core(CoreEvent::TurnStart { turn: 1 }),
    ];
    let projected = project_conversation(&events);
    assert_eq!(
        projected,
        vec![
            ReferencedConversationItem {
                role: ReferencedConversationRole::User,
                text: "hello".to_string(),
            },
            ReferencedConversationItem {
                role: ReferencedConversationRole::Assistant,
                text: "hi there".to_string(),
            },
        ]
    );
}

#[test]
fn a_compaction_checkpoint_is_a_user_message_and_flags_the_source() {
    let retained = retain_referenced_session(
        &snapshot(
            "s",
            vec![checkpoint_event("summary"), assistant_event("ok")],
        ),
        "label",
        65_536,
    )
    .unwrap();
    assert!(retained.stats.compacted);
    assert_eq!(retained.data.conversation.len(), 2);
    assert_eq!(
        retained.data.conversation[0].role,
        ReferencedConversationRole::User
    );
}

#[test]
fn a_small_budget_drops_the_oldest_non_checkpoint_message_and_keeps_the_newest() {
    let events = vec![
        user_event("first message that is long enough to matter"),
        assistant_event("second message that is long enough to matter"),
        user_event("third message that is long enough to matter"),
    ];
    let retained = retain_referenced_session(&snapshot("s", events), "label", 260).unwrap();
    assert_eq!(retained.stats.original_messages, 3);
    assert!(
        retained.stats.retained_messages < 3,
        "something was dropped"
    );
    assert!(retained.stats.truncated);
    let last = retained.data.conversation.last().unwrap();
    assert_eq!(last.text, "third message that is long enough to matter");
}

#[test]
fn a_checkpoint_is_never_dropped_to_fit_the_budget() {
    let events = vec![
        checkpoint_event("checkpoint text"),
        user_event("later text"),
    ];
    let retained = retain_referenced_session(&snapshot("s", events), "label", 200).unwrap();
    assert!(retained
        .data
        .conversation
        .iter()
        .any(|item| item.text == "checkpoint text"));
}

#[test]
fn an_impossibly_small_budget_returns_nothing() {
    assert!(retain_referenced_session(&snapshot("s", vec![user_event("x")]), "label", 1).is_none());
}

#[test]
fn the_preview_fits_the_budget() {
    let events = vec![user_event(&"a".repeat(5000)), assistant_event("done")];
    let retained = retain_referenced_session(&snapshot("s", events), "label", 512).unwrap();
    let serialized = stringify_tag_safe_json(&retained.data).unwrap();
    assert!(
        serialized.len() as u64 <= 512,
        "the preview fit: {} bytes",
        serialized.len()
    );
    assert!(retained.stats.omitted_bytes > 0);
    assert!(
        retained.full_data.conversation[0].text.len() > retained.data.conversation[0].text.len(),
        "the transcript keeps the full text"
    );
}

// --- truncation -------------------------------------------------------------

#[test]
fn text_inside_the_budget_is_untouched() {
    let truncated = truncate_with_notice("short", 100);
    assert_eq!(truncated.text, "short");
    assert_eq!(truncated.omitted_bytes, 0);
}

#[test]
fn an_excerpt_splits_its_budget_between_head_and_tail() {
    let source = format!("HEAD{}TAIL", "x".repeat(200));
    let truncated = truncate_with_notice(&source, 80);
    assert!(
        truncated.text.starts_with("HEAD"),
        "the head survives: {}",
        truncated.text
    );
    assert!(
        truncated.text.contains("TAIL"),
        "the tail survives: {}",
        truncated.text
    );
    assert!(
        truncated.text.len() as u64 <= 80,
        "the result fits its budget: {}",
        truncated.text.len()
    );
    assert!(
        truncated
            .text
            .contains(&format!("omitted {} UTF-8 bytes", truncated.omitted_bytes)),
        "the notice states the exact omission: {}",
        truncated.text
    );
}

#[test]
fn a_budget_too_small_for_the_notice_yields_no_text() {
    // The notice itself does not fit, so the largest fitting excerpt is the
    // empty one and every byte is reported as omitted.
    let truncated = truncate_with_notice(&"x".repeat(40), 20);
    assert_eq!(truncated.text, "");
    assert_eq!(truncated.omitted_bytes, 40);
    // A source that already fits is never touched, notice included.
    assert_eq!(truncate_with_notice("abcdefghij", 12).text, "abcdefghij");
}

#[test]
fn the_notice_counts_the_bytes_actually_dropped() {
    let text = "x".repeat(1000);
    let truncated = truncate_with_notice(&text, 120);
    assert!(
        truncated.text.len() as u64 <= 120,
        "the excerpt plus notice fit the budget"
    );
    assert!(truncated.omitted_bytes > 0);
    assert!(
        truncated
            .text
            .contains(&format!("omitted {} UTF-8 bytes", truncated.omitted_bytes)),
        "the stated omission is the one that was counted"
    );
}

#[test]
fn a_cut_never_splits_a_multi_byte_character() {
    let source = "中".repeat(40);
    let truncated = truncate_with_notice(&source, 40);
    assert!(
        !truncated.text.contains('\u{fffd}'),
        "no replacement characters: {}",
        truncated.text
    );
    assert!(truncated.text.len() as u64 <= 40);
    assert!(truncated.omitted_bytes > 0);
    assert!(
        truncated.omitted_bytes.is_multiple_of(3),
        "whole three-byte characters are dropped, never partial ones: {}",
        truncated.omitted_bytes
    );
    let retained: String = truncated.text.chars().filter(|c| *c == '中').collect();
    assert!(
        retained.chars().count() >= 2,
        "both cuts keep whole characters: {}",
        truncated.text
    );
}

// --- spill ------------------------------------------------------------------

struct RecordedSink {
    saved: RefCell<Vec<TextSpillRequest>>,
    fails: bool,
}

impl TextSpillSink for RecordedSink {
    fn save_text(&self, request: &TextSpillRequest) -> Result<SpillReference, TextSpillError> {
        self.saved.borrow_mut().push(request.clone());
        if self.fails {
            return Err(TextSpillError::new("disk full"));
        }
        Ok(SpillReference {
            locator: "spill://1".to_string(),
            bytes: 42,
            retrieval_hint: "read it".to_string(),
        })
    }
}

fn truncated_stats() -> ReferenceRetentionStats {
    ReferenceRetentionStats {
        compacted: false,
        original_messages: 5,
        retained_messages: 2,
        omitted_messages: 3,
        omitted_bytes: 90,
        truncated: true,
    }
}

#[test]
fn an_untruncated_preview_spills_nothing() {
    let stats = ReferenceRetentionStats::default();
    let data = ReferencedSessionData {
        session_id: id("s"),
        label: "label".to_string(),
        cwd: None,
        captured_through_seq: None,
        conversation: Vec::new(),
    };
    let sink = RecordedSink {
        saved: RefCell::new(Vec::new()),
        fails: false,
    };
    assert!(
        prepare_reference_omission(Some(&sink), &owner(), &data, &stats, 4, 0)
            .unwrap()
            .is_none()
    );
    assert!(
        sink.saved.borrow().is_empty(),
        "no save without an omission"
    );
}

#[test]
fn a_truncated_preview_without_storage_reports_why() {
    let data = ReferencedSessionData {
        session_id: id("s"),
        label: "label".to_string(),
        cwd: None,
        captured_through_seq: Some(3),
        conversation: Vec::new(),
    };
    let omission = prepare_reference_omission(None, &owner(), &data, &truncated_stats(), 4, 0)
        .unwrap()
        .unwrap();
    assert_eq!(omission.omitted_messages, 3);
    assert_eq!(omission.omitted_bytes, 90);
    assert_eq!(
        omission.full_snapshot,
        FullSnapshot::Unavailable {
            reason: UnavailableReason::StorageNotConfigured
        }
    );
}

#[test]
fn a_failed_save_is_reported_not_claimed() {
    let data = ReferencedSessionData {
        session_id: id("s"),
        label: "label".to_string(),
        cwd: None,
        captured_through_seq: None,
        conversation: Vec::new(),
    };
    let sink = RecordedSink {
        saved: RefCell::new(Vec::new()),
        fails: true,
    };
    let omission =
        prepare_reference_omission(Some(&sink), &owner(), &data, &truncated_stats(), 4, 1)
            .unwrap()
            .unwrap();
    assert_eq!(
        omission.full_snapshot,
        FullSnapshot::Unavailable {
            reason: UnavailableReason::SaveFailed
        }
    );
    let saved = sink.saved.borrow();
    assert_eq!(saved[0].suggested_name, "session-reference-2.txt");
    assert_eq!(saved[0].owner_session_id, owner());
}

#[test]
fn a_successful_save_is_cited_with_its_locator() {
    let data = ReferencedSessionData {
        session_id: id("s"),
        label: "label".to_string(),
        cwd: None,
        captured_through_seq: None,
        conversation: vec![ReferencedConversationItem {
            role: ReferencedConversationRole::User,
            text: "hello".to_string(),
        }],
    };
    let sink = RecordedSink {
        saved: RefCell::new(Vec::new()),
        fails: false,
    };
    let omission =
        prepare_reference_omission(Some(&sink), &owner(), &data, &truncated_stats(), 4, 0)
            .unwrap()
            .unwrap();
    assert_eq!(
        omission.full_snapshot,
        FullSnapshot::Saved {
            locator: "spill://1".to_string(),
            bytes: 42,
            retrieval_hint: "read it".to_string(),
        }
    );
    let saved = sink.saved.borrow();
    assert!(saved[0].content.contains("\"hello\""), "text is quoted");
    assert!(saved[0]
        .content
        .contains("untrusted, read-only snapshot from another session"));
}

#[test]
fn a_transcript_fragments_long_text_at_64_code_points() {
    let data = ReferencedSessionData {
        session_id: id("s"),
        label: "label".to_string(),
        cwd: Some("/work".to_string()),
        captured_through_seq: Some(9),
        conversation: vec![ReferencedConversationItem {
            role: ReferencedConversationRole::Assistant,
            text: "x".repeat(200),
        }],
    };
    let transcript = render_transcript(&data, 4).unwrap();
    let fragments: Vec<&str> = transcript
        .lines()
        .filter(|line| line.starts_with('"') && line.ends_with('"'))
        .collect();
    assert!(fragments.len() >= 4, "200 code points split into fragments");
    assert!(fragments.iter().all(|line| line.len() <= 66));
    assert!(transcript.contains("### Message 1: assistant"));
    assert!(transcript.contains("\"capturedFormatVersion\": 4"));
}

// --- candidate ranking ------------------------------------------------------

fn candidate(session: &str, cwd: Option<&str>, label: &str) -> SessionReferenceCandidate {
    SessionReferenceCandidate {
        session_id: id(session),
        label: label.to_string(),
        display_title: None,
        cwd: cwd.map(str::to_string),
        same_workspace: false,
        created_at: 0,
    }
}

#[test]
fn candidates_are_ranked_by_working_directory_affinity() {
    let ranked = rank_candidates(
        Some("/work"),
        "",
        50,
        vec![
            candidate("other", Some("/elsewhere"), "other"),
            candidate("absent", None, "absent"),
            candidate("same", Some("/work"), "same"),
        ],
    )
    .unwrap();
    let labels: Vec<&str> = ranked.iter().map(|c| c.label.as_str()).collect();
    assert_eq!(
        labels,
        vec!["same", "absent", "other"],
        "same workspace, then absent, then other"
    );
}

#[test]
fn a_query_filters_case_insensitively_across_the_searchable_fields() {
    let ranked = rank_candidates(
        None,
        "WORK",
        50,
        vec![
            candidate("s1", Some("/Work"), "first"),
            candidate("s2", None, "second"),
        ],
    )
    .unwrap();
    assert_eq!(ranked.len(), 1);
    assert_eq!(ranked[0].label, "first");
}

#[test]
fn a_zero_limit_is_invalid_but_the_limit_is_capped_by_the_list() {
    assert!(rank_candidates(None, "", 0, Vec::new()).is_err());
    let ranked = rank_candidates(
        None,
        "",
        1,
        vec![candidate("a", None, "a"), candidate("b", None, "b")],
    )
    .unwrap();
    assert_eq!(ranked.len(), 1);
}

// --- preparation ------------------------------------------------------------

#[test]
fn a_message_without_references_gains_no_context() {
    let prepared = prepare(PrepareRequest {
        target_id: &id("target"),
        content: vec![ContentBlock::text("hello")],
        references: &[],
        max_references: MAX_REFERENCES,
        max_reference_bytes: DEFAULT_MAX_REFERENCE_BYTES,
        snapshots: Vec::new(),
        owner_id: &owner(),
        additional_context_id: MessageId::new("ctx"),
        spill: None,
    })
    .unwrap();
    assert!(prepared.additional_context.is_none());
}

#[test]
fn referencing_the_target_session_fails() {
    let error = prepare(PrepareRequest {
        target_id: &id("target"),
        content: Vec::new(),
        references: &[SessionReferenceInput::new(id("target"))],
        max_references: MAX_REFERENCES,
        max_reference_bytes: DEFAULT_MAX_REFERENCE_BYTES,
        snapshots: Vec::new(),
        owner_id: &owner(),
        additional_context_id: MessageId::new("ctx"),
        spill: None,
    })
    .unwrap_err();
    assert_eq!(error.code(), SessionReferenceErrorCode::SelfReference);
}

#[test]
fn a_missing_snapshot_is_a_read_failure() {
    let error = prepare(PrepareRequest {
        target_id: &id("target"),
        content: Vec::new(),
        references: &[SessionReferenceInput::new(id("missing"))],
        max_references: MAX_REFERENCES,
        max_reference_bytes: DEFAULT_MAX_REFERENCE_BYTES,
        snapshots: Vec::new(),
        owner_id: &owner(),
        additional_context_id: MessageId::new("ctx"),
        spill: None,
    })
    .unwrap_err();
    assert_eq!(error.code(), SessionReferenceErrorCode::ReadFailed);
}

#[test]
fn a_prepared_reference_carries_the_source_record_and_the_snapshot() {
    let sink = RecordedSink {
        saved: RefCell::new(Vec::new()),
        fails: false,
    };
    let prepared = prepare(PrepareRequest {
        target_id: &id("target"),
        content: vec![ContentBlock::text("look at this")],
        references: &[SessionReferenceInput::with_label(id("source"), "the run")],
        max_references: MAX_REFERENCES,
        max_reference_bytes: DEFAULT_MAX_REFERENCE_BYTES,
        snapshots: vec![snapshot(
            "source",
            vec![user_event("question"), assistant_event("answer")],
        )],
        owner_id: &owner(),
        additional_context_id: MessageId::new("ctx"),
        spill: Some(&sink),
    })
    .unwrap();

    let message = prepared.additional_context.unwrap();
    let text = message
        .content
        .iter()
        .filter_map(|block| match block {
            dsh_llm::ContentBlock::Core(dsh_llm::CoreContentBlock::Text { text }) => {
                Some(text.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    assert!(text.contains("<referenced-sessions>"));
    assert!(text.contains("untrusted, read-only snapshot"));
    assert!(text.contains("\"label\":\"the run\""));
    assert!(
        !text.contains("## Reference omissions"),
        "nothing was omitted, so no omission section"
    );
}

#[test]
fn a_duplicate_reference_is_normalized_to_one_source() {
    let normalized = normalize_references(
        &id("target"),
        &[
            SessionReferenceInput::new(id("a")),
            SessionReferenceInput::with_label(id("a"), "second"),
        ],
        MAX_REFERENCES,
    )
    .unwrap();
    assert_eq!(normalized.len(), 1);
    assert_eq!(normalized[0].label, "a", "the first label wins");
}

#[test]
fn a_reference_label_defaults_to_the_session_id() {
    let normalized =
        normalize_references(&id("target"), &[SessionReferenceInput::new(id("a"))], 3).unwrap();
    assert_eq!(normalized[0].label, "a");
}

#[test]
fn more_references_than_allowed_fail() {
    let error = normalize_references(
        &id("target"),
        &[
            SessionReferenceInput::new(id("a")),
            SessionReferenceInput::new(id("b")),
        ],
        1,
    )
    .unwrap_err();
    assert_eq!(error.code(), SessionReferenceErrorCode::TooMany);
}
