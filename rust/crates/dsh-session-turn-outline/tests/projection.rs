//! `turnOutline` fold behavior: boundary anchoring, prompt/response previews,
//! turn lifecycle and statuses, draft-only identity, restored state validation,
//! and the exact client-view wire shape.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use dsh_llm::{
    AssistantMessage, ContentBlock, ExtensionSource, LlmFailure, MessageId, MessageSource,
    ToolCallId, ToolResultMessage, UserMessage,
};
use dsh_session::{
    Session, SessionEvent, SessionId, SessionLogOffset, SessionMeta, SessionSeq, ToolErrorInfo,
    TurnEndCancelCause, TurnEndReason,
};
use dsh_session_projection::{
    ProjectionCheckpoint, ProjectionCheckpointRow, ProjectionCursor, ProjectionError,
    ProjectionRegistry,
};
use dsh_session_turn_outline::{
    outline_projection, TurnOutlineEntry, TurnOutlineState, TurnOutlineStatus, KEY,
    PROMPT_PREVIEW_LIMIT, RESPONSE_PREVIEW_LIMIT, STATE_VERSION,
};
use serde_json::{json, Map, Value};

fn session(id: &str) -> Session {
    Session::create(SessionId::new(id).unwrap(), SessionMeta::default(), 0)
}

fn offset(value: u64) -> SessionLogOffset {
    SessionLogOffset::new(value).unwrap()
}

fn seq(value: u64) -> SessionSeq {
    SessionSeq::new(value).unwrap()
}

fn append(session: &mut Session, event: SessionEvent) -> SessionSeq {
    session.append(event).unwrap()
}

fn human_prompt(text: impl Into<String>) -> SessionEvent {
    SessionEvent::user_message(UserMessage::new(
        MessageId::new("user"),
        vec![ContentBlock::text(text)],
    ))
}

fn prompt_blocks(blocks: Vec<ContentBlock>) -> SessionEvent {
    SessionEvent::user_message(UserMessage::new(MessageId::new("user-blocks"), blocks))
}

fn synthetic_prompt(text: impl Into<String>) -> SessionEvent {
    SessionEvent::user_message(UserMessage::with_source(
        MessageId::new("synthetic"),
        vec![ContentBlock::text(text)],
        MessageSource::Extension(ExtensionSource::new("agent", Map::new())),
    ))
}

fn assistant(turn: u64, step: u64, text: impl Into<String>) -> SessionEvent {
    SessionEvent::assistant_message(
        turn,
        step,
        AssistantMessage::new(
            MessageId::new(format!("assistant-{turn}-{step}")),
            vec![ContentBlock::text(text)],
            "alpha",
            "m1",
        ),
        vec![],
        None,
        false,
    )
}

fn assistant_blocks(turn: u64, step: u64, blocks: Vec<ContentBlock>) -> SessionEvent {
    SessionEvent::assistant_message(
        turn,
        step,
        AssistantMessage::new(
            MessageId::new(format!("assistant-blocks-{turn}-{step}")),
            blocks,
            "alpha",
            "m1",
        ),
        vec![],
        None,
        false,
    )
}

fn tool_call(turn: u64, step: u64, id: &str) -> SessionEvent {
    SessionEvent::tool_call(turn, step, ToolCallId::new(id), "echo", "{}")
}

fn tool_result(turn: u64, step: u64, id: &str, is_error: bool) -> SessionEvent {
    SessionEvent::tool_result(
        turn,
        step,
        ToolResultMessage::new(
            MessageId::new(format!("result-{id}")),
            vec![ContentBlock::text("done")],
            ToolCallId::new(id),
            is_error,
        ),
    )
}

fn tool_failure(turn: u64, step: u64, id: &str) -> SessionEvent {
    SessionEvent::tool_error(
        turn,
        step,
        ToolResultMessage::new(
            MessageId::new(format!("failure-{id}")),
            vec![ContentBlock::text("failed")],
            ToolCallId::new(id),
            false,
        ),
        ToolErrorInfo {
            name: "echo".to_string(),
            code: "FAILED".to_string(),
            reason: None,
        },
    )
}

fn failure() -> LlmFailure {
    LlmFailure {
        message: "model failed".to_string(),
        code: "MODEL_ERROR".to_string(),
        status: None,
        provider_retry_after_ms: None,
        request_id: None,
        offload_images: None,
    }
}

fn outline(registry: &ProjectionRegistry, session: &Session) -> Value {
    registry
        .snapshot(session, offset(0))
        .unwrap()
        .values
        .get(KEY)
        .unwrap()
        .clone()
}

fn state(registry: &ProjectionRegistry, session: &Session) -> Arc<TurnOutlineState> {
    registry
        .state_of::<TurnOutlineState>(session, offset(0), KEY)
        .unwrap()
        .unwrap()
}

fn entry(turn: u64, start: u64, end: Option<u64>, status: TurnOutlineStatus) -> TurnOutlineEntry {
    TurnOutlineEntry {
        turn,
        seq: seq(start),
        prompt: String::new(),
        response: String::new(),
        end_seq: end.map(seq),
        steps: 0,
        tool_calls: 0,
        tool_results: 0,
        tool_errors: 0,
        status,
    }
}

fn state_json(turns: Vec<TurnOutlineEntry>, draft: impl Into<String>) -> Value {
    serde_json::to_value(TurnOutlineState {
        turns: Arc::new(turns),
        draft: draft.into(),
    })
    .unwrap()
}

fn checkpoint(value: Value) -> ProjectionCheckpoint {
    [(
        KEY.to_string(),
        ProjectionCheckpointRow {
            ver: STATE_VERSION,
            seq: ProjectionCursor::BEFORE_LOG,
            val: value,
        },
    )]
    .into_iter()
    .collect()
}

#[test]
fn an_empty_log_serves_an_empty_outline() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let session = session("outline-empty");
    assert_eq!(KEY, "turnOutline");
    assert_eq!(STATE_VERSION, 3);

    let snapshot = registry.snapshot(&session, offset(0)).unwrap();
    assert_eq!(snapshot.as_of_seq, ProjectionCursor::BEFORE_LOG);
    assert_eq!(snapshot.values.get(KEY), Some(&json!([])));

    let rows = registry.checkpoint(&session, offset(0)).unwrap();
    assert_eq!(
        serde_json::to_string(rows.get(KEY).unwrap()).unwrap(),
        r#"{"ver":3,"seq":-1,"val":{"turns":[],"draft":""}}"#
    );
}

#[test]
fn turn_start_appends_an_entry_anchored_at_its_own_seq() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-anchors");
    let first = append(&mut session, SessionEvent::turn_start(1));
    let second = append(&mut session, SessionEvent::turn_start(2));
    append(&mut session, SessionEvent::turn_start(2));
    append(&mut session, SessionEvent::turn_start(1));

    let value = outline(&registry, &session);
    assert_eq!(first, seq(0));
    assert_eq!(second, seq(1));
    assert_eq!(
        serde_json::to_string(&value).unwrap(),
        concat!(
            r#"[{"turn":1,"seq":0,"prompt":"","response":"","steps":0,"toolCalls":0,"#,
            r#""toolResults":0,"toolErrors":0,"status":"open"},"#,
            r#"{"turn":2,"seq":1,"prompt":"","response":"","steps":0,"toolCalls":0,"#,
            r#""toolResults":0,"toolErrors":0,"status":"open"}]"#
        ),
        "entries are strictly increasing and a retried boundary is ignored"
    );
}

#[test]
fn the_first_human_prompt_fills_the_open_turn_once() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-prompt");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, human_prompt("  first   prompt \n line "));
    append(&mut session, human_prompt("steering text"));
    append(&mut session, synthetic_prompt("agent context"));

    let value = outline(&registry, &session);
    assert_eq!(value[0]["prompt"], json!("first prompt line"));
    assert_eq!(
        state(&registry, &session).turns[0].prompt,
        "first prompt line"
    );
}

#[test]
fn a_synthetic_user_role_message_does_not_fill_the_prompt() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-synthetic");
    append(&mut session, synthetic_prompt("agent context"));
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, synthetic_prompt("still not human"));
    append(&mut session, human_prompt("human"));

    assert_eq!(outline(&registry, &session)[0]["prompt"], json!("human"));
}

#[test]
fn previews_join_text_blocks_and_collapse_whitespace() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-preview-blocks");
    append(&mut session, SessionEvent::turn_start(1));
    append(
        &mut session,
        prompt_blocks(vec![
            ContentBlock::text("  alpha \n\t beta "),
            ContentBlock::reasoning("ignored reasoning"),
            ContentBlock::text("gamma"),
        ]),
    );
    append(
        &mut session,
        assistant_blocks(
            1,
            1,
            vec![
                ContentBlock::reasoning("ignored reasoning"),
                ContentBlock::text(" one "),
                ContentBlock::text("two  three "),
            ],
        ),
    );
    append(
        &mut session,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );

    let value = outline(&registry, &session);
    assert_eq!(value[0]["prompt"], json!("alpha beta gamma"));
    assert_eq!(value[0]["response"], json!("one two three"));
}

#[test]
fn previews_clip_to_their_budgets_on_unicode_scalar_boundaries() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut clipped = session("outline-preview-clip");
    append(&mut clipped, SessionEvent::turn_start(1));
    append(&mut clipped, human_prompt("a".repeat(60)));
    append(&mut clipped, assistant(1, 1, "b".repeat(130)));
    append(
        &mut clipped,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );

    let value = outline(&registry, &clipped);
    let prompt = value[0]["prompt"].as_str().unwrap();
    let response = value[0]["response"].as_str().unwrap();
    assert_eq!(prompt.chars().count(), PROMPT_PREVIEW_LIMIT);
    assert!(prompt.ends_with('…'));
    assert_eq!(response.chars().count(), RESPONSE_PREVIEW_LIMIT);
    assert!(response.ends_with('…'));

    let mut multibyte = session("outline-preview-multibyte");
    append(&mut multibyte, SessionEvent::turn_start(1));
    append(&mut multibyte, human_prompt("汉".repeat(60)));
    append(&mut multibyte, assistant(1, 1, "😀".repeat(130)));
    append(
        &mut multibyte,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );
    let value = outline(&registry, &multibyte);
    let prompt = value[0]["prompt"].as_str().unwrap();
    let response = value[0]["response"].as_str().unwrap();
    assert_eq!(prompt.chars().count(), PROMPT_PREVIEW_LIMIT);
    assert_eq!(response.chars().count(), RESPONSE_PREVIEW_LIMIT);
    assert!(prompt.starts_with('汉') && prompt.ends_with('…'));
    assert!(response.starts_with('😀') && response.ends_with('…'));
}

#[test]
fn the_newest_assistant_draft_commits_at_turn_end() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-draft");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, human_prompt("prompt"));
    append(&mut session, assistant(1, 1, "draft one"));
    append(&mut session, assistant(1, 2, "draft two"));

    let open = outline(&registry, &session);
    assert_eq!(open[0]["response"], json!(""));
    assert_eq!(open[0]["status"], json!("open"));
    assert_eq!(state(&registry, &session).draft, "draft two");

    let end = append(
        &mut session,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );
    let value = outline(&registry, &session);
    assert_eq!(value[0]["response"], json!("draft two"));
    assert_eq!(value[0]["endSeq"], json!(end.get()));
    assert_eq!(value[0]["status"], json!("completed"));
    assert_eq!(state(&registry, &session).draft, "");

    append(&mut session, assistant(1, 3, "too late"));
    assert_eq!(
        outline(&registry, &session)[0]["response"],
        json!("draft two")
    );
}

#[test]
fn a_turn_with_no_assistant_text_commits_an_empty_response() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-empty-response");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, human_prompt("prompt"));
    append(
        &mut session,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );

    let value = outline(&registry, &session);
    assert_eq!(value[0]["response"], json!(""));
    assert_eq!(value[0]["status"], json!("completed"));
}

#[test]
fn step_and_tool_counts_update_only_the_open_turn() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-steps-tools");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, SessionEvent::step_end(1, 1));
    append(&mut session, SessionEvent::step_end(1, 2));
    append(&mut session, tool_call(1, 1, "a"));
    append(&mut session, tool_result(1, 1, "a", false));
    append(&mut session, tool_call(1, 2, "b"));
    append(&mut session, tool_failure(1, 2, "b"));
    append(&mut session, SessionEvent::step_end(1, 3));
    append(
        &mut session,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );
    append(&mut session, SessionEvent::step_end(1, 4));
    append(&mut session, tool_call(1, 3, "c"));
    append(&mut session, tool_result(1, 3, "c", false));

    let value = outline(&registry, &session);
    assert_eq!(value[0]["steps"], json!(3));
    assert_eq!(value[0]["toolCalls"], json!(2));
    assert_eq!(value[0]["toolResults"], json!(2));
    assert_eq!(value[0]["toolErrors"], json!(1));
    assert_eq!(value[0]["status"], json!("completed"));
}

#[test]
fn every_turn_end_reason_maps_to_its_status() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-statuses");
    let reasons = vec![
        TurnEndReason::Completed,
        TurnEndReason::Aborted {
            reason: TurnEndCancelCause::User,
        },
        TurnEndReason::Blocked,
        TurnEndReason::error(failure()),
        TurnEndReason::MaxTokens,
        TurnEndReason::Interrupted,
        TurnEndReason::Forked,
    ];
    for (index, reason) in reasons.into_iter().enumerate() {
        let turn = index as u64 + 1;
        append(&mut session, SessionEvent::turn_start(turn));
        append(&mut session, assistant(turn, 1, format!("turn {turn}")));
        append(&mut session, SessionEvent::turn_end(turn, reason));
    }

    let value = outline(&registry, &session);
    let statuses: Vec<&str> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["status"].as_str().unwrap())
        .collect();
    assert_eq!(
        statuses,
        vec![
            "completed",
            "aborted",
            "blocked",
            "error",
            "max-tokens",
            "interrupted",
            "forked"
        ]
    );
    assert_eq!(value[3]["response"], json!("turn 4"));
}

#[test]
fn turn_start_resets_the_draft_and_ignores_non_advancing_boundaries() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-draft-reset");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, assistant(1, 1, "first draft"));
    append(&mut session, SessionEvent::turn_start(2));
    assert_eq!(state(&registry, &session).draft, "");
    append(&mut session, assistant(2, 1, "second draft"));
    append(&mut session, SessionEvent::turn_start(2));
    append(&mut session, SessionEvent::turn_start(1));
    assert_eq!(state(&registry, &session).draft, "second draft");
    append(
        &mut session,
        SessionEvent::turn_end(2, TurnEndReason::Completed),
    );

    let value = outline(&registry, &session);
    assert_eq!(value.as_array().unwrap().len(), 2);
    assert_eq!(value[0]["response"], json!(""));
    assert_eq!(value[1]["response"], json!("second draft"));
}

#[test]
fn draft_only_changes_stay_quiet_while_boundaries_publish() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-change-feed");
    let seen = Rc::new(RefCell::new(Vec::new()));
    let _subscription = registry.on_changed({
        let seen = Rc::clone(&seen);
        move |change| seen.borrow_mut().push(change.value.clone())
    });

    append(&mut session, SessionEvent::turn_start(1));
    registry.drive_event(&session, offset(0), seq(0)).unwrap();
    append(&mut session, human_prompt("prompt"));
    registry.drive_event(&session, offset(0), seq(1)).unwrap();
    append(&mut session, assistant(1, 1, "draft one"));
    registry.drive_event(&session, offset(0), seq(2)).unwrap();
    append(&mut session, assistant(1, 2, "draft two"));
    registry.drive_event(&session, offset(0), seq(3)).unwrap();
    append(&mut session, SessionEvent::step_end(1, 1));
    registry.drive_event(&session, offset(0), seq(4)).unwrap();
    append(
        &mut session,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );
    registry.drive_event(&session, offset(0), seq(5)).unwrap();

    let seen = seen.borrow();
    assert_eq!(
        seen.len(),
        4,
        "boundary, prompt, step count, and committed response"
    );
    assert_eq!(seen[0][0]["status"], json!("open"));
    assert_eq!(seen[1][0]["prompt"], json!("prompt"));
    assert_eq!(seen[2][0]["steps"], json!(1));
    assert_eq!(seen[3][0]["response"], json!("draft two"));
    assert_eq!(seen[3][0]["status"], json!("completed"));
}

#[test]
fn the_view_is_the_turns_array_without_the_internal_draft() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-view");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, assistant(1, 1, "draft"));
    assert_eq!(state(&registry, &session).draft, "draft");
    let value = outline(&registry, &session);
    assert!(value.is_array());
    assert!(value[0].get("draft").is_none());
}

#[test]
fn unrelated_events_keep_the_state_allocation() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-unrelated");
    append(&mut session, SessionEvent::turn_start(1));
    let changes = Rc::new(RefCell::new(0usize));
    let _subscription = registry.on_changed({
        let changes = Rc::clone(&changes);
        move |_change| *changes.borrow_mut() += 1
    });
    registry.drive_event(&session, offset(0), seq(0)).unwrap();
    let before = state(&registry, &session);

    let unrelated = vec![
        SessionEvent::step_start(1, 1),
        SessionEvent::extension_event("compaction/start", Map::new()),
        SessionEvent::turn_start(1),
        assistant(9, 1, "other turn"),
        prompt_blocks(vec![]),
    ];
    for (index, event) in unrelated.into_iter().enumerate() {
        let position = append(&mut session, event);
        assert_eq!(position, seq(index as u64 + 1));
        registry.drive_event(&session, offset(0), position).unwrap();
        let current = state(&registry, &session);
        assert!(
            Arc::ptr_eq(&before, &current),
            "seq {position} is not an outline change"
        );
    }
    assert_eq!(*changes.borrow(), 1);
}

#[test]
fn checkpoint_and_restore_round_trip_a_two_turn_outline() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let mut session = session("outline-restore");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, human_prompt("first"));
    append(&mut session, assistant(1, 1, "answer one"));
    append(
        &mut session,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );
    append(&mut session, SessionEvent::turn_start(2));
    append(&mut session, human_prompt("second"));
    let rows = registry.checkpoint(&session, offset(0)).unwrap();
    assert_eq!(rows.get(KEY).unwrap().ver, STATE_VERSION);
    assert_eq!(rows.get(KEY).unwrap().seq, ProjectionCursor::at(seq(5)));
    assert_eq!(
        rows.get(KEY)
            .unwrap()
            .val
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["turns", "draft"]
    );

    append(&mut session, assistant(2, 1, "answer two"));
    append(
        &mut session,
        SessionEvent::turn_end(2, TurnEndReason::Completed),
    );
    let expected = outline(&registry, &session);

    let restored = registry
        .restore(
            &rows,
            &session.events()[6..],
            offset(6),
            session.header(),
            offset(0),
        )
        .unwrap();
    assert_eq!(restored.snapshot.values.get(KEY), Some(&expected));
    assert_eq!(restored.snapshot.as_of_seq, ProjectionCursor::at(seq(7)));
    assert_eq!(
        restored.checkpoint.get(KEY).unwrap().seq,
        ProjectionCursor::at(seq(7))
    );
    assert_eq!(
        restored.checkpoint.get(KEY).unwrap().val["turns"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        restored.checkpoint.get(KEY).unwrap().val["draft"],
        json!("")
    );

    let state = state(&registry, &session);
    let encoded = serde_json::to_value(&*state).unwrap();
    let decoded: TurnOutlineState = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, *state);
}

#[test]
fn restore_rejects_an_outline_that_violates_its_invariants() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(outline_projection()).unwrap();
    let header = session("outline-invalid").header().clone();
    let cases = vec![
        (
            "descending turns",
            state_json(
                vec![
                    entry(2, 0, None, TurnOutlineStatus::Open),
                    entry(1, 1, None, TurnOutlineStatus::Open),
                ],
                "",
            ),
        ),
        (
            "closed with an open status",
            state_json(vec![entry(1, 0, Some(1), TurnOutlineStatus::Open)], ""),
        ),
        (
            "open with a closed status",
            state_json(vec![entry(1, 0, None, TurnOutlineStatus::Completed)], ""),
        ),
        (
            "end before start",
            state_json(vec![entry(2, 5, Some(3), TurnOutlineStatus::Completed)], ""),
        ),
        (
            "draft beyond the response budget",
            state_json(vec![], "x".repeat(RESPONSE_PREVIEW_LIMIT + 1)),
        ),
    ];

    for (name, value) in cases {
        let rows = checkpoint(value);
        assert!(
            registry.view_checkpoint(&rows, None).unwrap().is_empty(),
            "{name}"
        );
        let error = registry
            .restore(&rows, &[], offset(0), &header, offset(0))
            .unwrap_err();
        assert!(
            matches!(
                error,
                ProjectionError::InvalidState {
                    key: "turnOutline",
                    ..
                }
            ),
            "{name}"
        );
    }

    let valid = checkpoint(state_json(
        vec![entry(1, 0, Some(1), TurnOutlineStatus::Completed)],
        "",
    ));
    assert_eq!(
        registry.view_checkpoint(&valid, None).unwrap().get(KEY),
        Some(&json!([{
            "turn": 1,
            "seq": 0,
            "prompt": "",
            "response": "",
            "endSeq": 1,
            "steps": 0,
            "toolCalls": 0,
            "toolResults": 0,
            "toolErrors": 0,
            "status": "completed"
        }]))
    );
}
