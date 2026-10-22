//! The `userQuestions` projection fold: the pure rules over the session log.

use dsh_interaction::projection::*;
use dsh_interaction::questions::{
    answer_reply_message, AskUserQuestionAnswerItem, PendingUserQuestion,
    UserQuestionProjectionView, UserQuestionState,
};
use dsh_llm::{ContentBlock, LlmCallConfig, MessageId, ToolCallId, ToolResultMessage, ToolSchema};
use dsh_session::{
    EpochHeader, RequestHeaderReason, Session, SessionEvent, SessionId, SessionLogOffset,
    SessionMeta, ToolErrorInfo,
};
use dsh_session_projection::ProjectionRegistry;
use serde_json::json;

const CALL_ID: &str = "call_ask_1";

fn tool_arguments() -> String {
    json!({
        "questions": [{
            "id": "scope",
            "question": "Which scope?",
            "header": "Scope",
            "options": [
                {"label": "Tool only", "description": "Only the tool package."},
                {"label": "Everything"},
            ],
            "multi_select": true,
        }],
    })
    .to_string()
}

fn timed_schema() -> ToolSchema {
    ToolSchema::new(
        "ask_user_question",
        "Ask brief questions.",
        json!({
            "type": "object",
            "properties": {
                "questions": {"type": "array"},
                "timeout": {"type": "integer"},
            },
        }),
    )
}

fn legacy_schema() -> ToolSchema {
    ToolSchema::new(
        "ask_user_question",
        "Ask the user a concise question.",
        json!({
            "type": "object",
            "properties": {"questions": {"type": "array"}},
        }),
    )
}

fn bash_schema() -> ToolSchema {
    ToolSchema::new(
        "bash",
        "Run a command.",
        json!({"type": "object", "properties": {}}),
    )
}

fn header(tools: Option<Vec<ToolSchema>>) -> SessionEvent {
    SessionEvent::request_header(
        EpochHeader {
            config: LlmCallConfig {
                provider: "deepseek-official".to_string(),
                model: "deepseek-v4-flash".to_string(),
                reasoning_effort: None,
                temperature: None,
                max_tokens: None,
                stop: None,
            },
            adapter_defaults: None,
            tools,
        },
        RequestHeaderReason::Initial,
    )
}

fn timed_header() -> SessionEvent {
    header(Some(vec![bash_schema(), timed_schema()]))
}

fn legacy_header() -> SessionEvent {
    header(Some(vec![bash_schema(), legacy_schema()]))
}

fn asked(call_id: &str) -> SessionEvent {
    SessionEvent::tool_call(
        1,
        1,
        ToolCallId::new(call_id),
        "ask_user_question",
        tool_arguments(),
    )
}

fn result_message(text: &str) -> ToolResultMessage {
    ToolResultMessage::new(
        MessageId::new("result-1"),
        vec![ContentBlock::text(text)],
        ToolCallId::new(CALL_ID),
        false,
    )
}

fn resulted(text: &str) -> SessionEvent {
    SessionEvent::tool_result(1, 1, result_message(text))
}

fn errored(text: &str, code: &str) -> SessionEvent {
    SessionEvent::tool_error(
        1,
        1,
        ToolResultMessage::new(
            MessageId::new("result-1"),
            vec![ContentBlock::text(text)],
            ToolCallId::new(CALL_ID),
            true,
        ),
        ToolErrorInfo {
            name: "ToolOutcomeUnknownError".to_string(),
            code: code.to_string(),
            reason: None,
        },
    )
}

fn reply(text: &str, call_id: &str) -> SessionEvent {
    let message = answer_reply_message(
        MessageId::new("reply-1"),
        &ToolCallId::new(call_id),
        &[],
        &dsh_interaction::questions::AskUserQuestionAnswer {
            answers: Vec::new(),
        },
    );
    let _ = text;
    SessionEvent::user_message(message)
}

fn ptc_dispatch(call_id: &str, pending: bool) -> SessionEvent {
    let content = if pending {
        json!({"pending": true, "callId": call_id}).to_string()
    } else {
        json!({"answers": []}).to_string()
    };
    let payload = json!({
        "rootCallId": "run_1",
        "parentCallId": "run_1",
        "subCallId": call_id,
        "name": "ask_user_question",
        "arguments": {"questions": [{"id": "scope", "question": "Which scope?"}]},
        "isError": false,
        "content": [{"type": "text", "text": content}],
    });
    SessionEvent::extension_event("tool/ptc-dispatch", payload.as_object().unwrap().clone())
}

#[test]
fn the_timed_schema_is_recognised_by_its_wait_parameter_alone() {
    let timed = timed_schema();
    let legacy = legacy_schema();
    assert!(is_timed_ask_user_question_schema(
        &timed.name,
        &timed.parameters
    ));
    assert!(!is_timed_ask_user_question_schema(
        &legacy.name,
        &legacy.parameters
    ));
    assert!(!is_timed_ask_user_question_schema(
        "bash",
        &timed.parameters
    ));
    let no_properties = ToolSchema::new("ask_user_question", "Ask.", json!({"type": "object"}));
    assert!(!is_timed_ask_user_question_schema(
        &no_properties.name,
        &no_properties.parameters
    ));
}

#[test]
fn a_call_under_the_timed_schema_opens_a_question() {
    let view = fold_user_questions(&[timed_header(), asked(CALL_ID)]);
    assert_eq!(view.active.len(), 1);
    assert_eq!(view.active[0].call_id, ToolCallId::new(CALL_ID));
    assert_eq!(view.active[0].state, UserQuestionState::Open);
    assert_eq!(view.active[0].questions.len(), 1);
    // The tool vocabulary is mapped into the service vocabulary.
    assert_eq!(view.active[0].questions[0].id, "scope");
    assert_eq!(view.active[0].questions[0].header.as_deref(), Some("Scope"));
    assert_eq!(view.active[0].questions[0].multi_select, Some(true));
    assert_eq!(
        view.active[0].questions[0]
            .options
            .as_ref()
            .unwrap()
            .first()
            .unwrap()
            .label,
        "Tool only"
    );
    assert!(view.settled.is_empty());
}

#[test]
fn a_call_under_the_legacy_schema_is_never_tracked() {
    let view = fold_user_questions(&[legacy_header(), asked(CALL_ID)]);
    assert!(view.active.is_empty());
    assert!(view.settled.is_empty());

    // No header at all, and a header without tools, read as the legacy tool.
    assert!(fold_user_questions(&[asked(CALL_ID)]).active.is_empty());
    assert!(fold_user_questions(&[header(None), asked(CALL_ID)])
        .active
        .is_empty());
}

#[test]
fn the_latest_header_wins() {
    let later = SessionEvent::tool_call(
        2,
        1,
        ToolCallId::new("call_ask_2"),
        "ask_user_question",
        tool_arguments(),
    );
    let view = fold_user_questions(&[
        legacy_header(),
        asked(CALL_ID),
        header(Some(vec![timed_schema()])),
        later.clone(),
    ]);
    assert_eq!(view.active.len(), 1);
    assert_eq!(view.active[0].call_id, ToolCallId::new("call_ask_2"));

    let view = fold_user_questions(&[
        timed_header(),
        asked(CALL_ID),
        header(Some(vec![legacy_schema()])),
        later.clone(),
    ]);
    assert_eq!(view.active.len(), 1);
    assert_eq!(view.active[0].call_id, ToolCallId::new(CALL_ID));
}

#[test]
fn an_in_time_answer_settles_the_question_with_that_batch() {
    let view = fold_user_questions(&[
        timed_header(),
        asked(CALL_ID),
        resulted(r#"{"answers":[{"id":"scope","selected":["Tool only"]}]}"#),
    ]);
    assert!(view.active.is_empty());
    assert_eq!(view.settled.len(), 1);
    assert_eq!(view.settled[0].call_id, ToolCallId::new(CALL_ID));
    assert_eq!(view.settled[0].answers.len(), 1);
    assert_eq!(view.settled[0].answers[0].id, "scope");
    assert_eq!(view.settled[0].answers[0].selected, vec!["Tool only"]);
}

#[test]
fn a_failure_drops_the_question() {
    for text in ["cancelled", "aborted", "boom", "not json at all"] {
        let view = fold_user_questions(&[timed_header(), asked(CALL_ID), resulted(text)]);
        assert!(view.active.is_empty(), "{text} must drop the question");
        assert!(
            view.settled.is_empty(),
            "{text} must not settle the question"
        );
    }
    // A batch spelled inside an error result is a failure, whatever its text
    // says.
    let view = fold_user_questions(&[
        timed_header(),
        asked(CALL_ID),
        errored(r#"{"answers":[]}"#, "UNEXPECTED"),
    ]);
    assert!(view.active.is_empty());
    assert!(view.settled.is_empty());
}

#[test]
fn a_pending_result_keeps_the_question_answerable() {
    let view = fold_user_questions(&[
        timed_header(),
        asked(CALL_ID),
        resulted(r#"{"pending":true,"callId":"call_ask_1"}"#),
    ]);
    assert_eq!(view.active.len(), 1);
    assert_eq!(view.active[0].state, UserQuestionState::Continued);
    assert!(view.settled.is_empty());
}

#[test]
fn a_call_the_process_never_finished_keeps_the_question_answerable() {
    let view = fold_user_questions(&[
        timed_header(),
        asked(CALL_ID),
        errored(
            "The tool call was interrupted after it was recorded.",
            TOOL_OUTCOME_UNKNOWN,
        ),
    ]);
    assert_eq!(view.active.len(), 1);
    assert_eq!(view.active[0].state, UserQuestionState::Continued);
}

#[test]
fn a_late_reply_settles_the_question_only_when_it_is_admitted() {
    let continued = vec![
        timed_header(),
        asked(CALL_ID),
        resulted(r#"{"pending":true,"callId":"call_ask_1"}"#),
    ];
    let mut events = continued.clone();
    events.push(reply("{}", CALL_ID));
    let view = fold_user_questions(&events);
    assert!(view.active.is_empty());
    assert_eq!(view.settled.len(), 1);
    assert!(view.settled[0].answers.is_empty());
}

#[test]
fn an_inbox_splice_alone_settles_nothing() {
    let view = fold_user_questions(&[
        timed_header(),
        asked(CALL_ID),
        resulted(r#"{"pending":true,"callId":"call_ask_1"}"#),
        SessionEvent::extension_event(
            "agent/inbox/spliced",
            json!({"target": "next-step", "start": 0, "inserted": []})
                .as_object()
                .unwrap()
                .clone(),
        ),
    ]);
    assert_eq!(view.active.len(), 1);
    assert_eq!(view.active[0].state, UserQuestionState::Continued);
    assert!(view.settled.is_empty());
}

#[test]
fn a_pending_ptc_sub_call_continues_even_without_the_timed_header() {
    let run_code_header = header(Some(vec![ToolSchema::new(
        "run_code",
        "Run code.",
        json!({"type": "object", "properties": {}}),
    )]));
    let view = fold_user_questions(&[run_code_header.clone(), ptc_dispatch("run_1:ptc:1", true)]);
    assert_eq!(view.active.len(), 1);
    assert_eq!(view.active[0].call_id, ToolCallId::new("run_1:ptc:1"));
    assert_eq!(view.active[0].state, UserQuestionState::Continued);

    // A settled sub-call carries no answer batch and does not enter the fold.
    let view = fold_user_questions(&[run_code_header.clone(), ptc_dispatch("run_1:ptc:1", false)]);
    assert!(view.active.is_empty());
}

#[test]
fn the_questions_are_read_out_of_the_logged_arguments() {
    assert_eq!(questions_of(r#"{"questions":[]}"#), None);
    assert_eq!(questions_of(r#"{"questions":[{"id":"x"}]}"#), None);
    assert_eq!(questions_of("[]"), None);
    assert_eq!(questions_of("{not json"), None);
    let questions = questions_of(r#"{"questions":[{"id":"x","question":"Q?"}]}"#).unwrap();
    assert_eq!(questions.len(), 1);
    assert_eq!(questions[0].id, "x");
    assert_eq!(questions[0].question, "Q?");
    assert!(questions[0].options.is_none());
}

#[test]
fn unrelated_events_leave_the_fold_untouched() {
    let fold = UserQuestionFold::initial();
    assert!(!fold.timed);
    let timed = UserQuestionFold {
        timed: true,
        questions: std::sync::Arc::new(fold_user_questions(&[timed_header(), asked(CALL_ID)])),
    };
    assert!(apply_user_question_event(&timed, &SessionEvent::turn_start(2)).is_none());
    assert!(apply_user_question_event(&timed, &timed_header()).is_none());
    assert!(apply_user_question_event(
        &timed,
        &SessionEvent::tool_call(1, 2, ToolCallId::new("other"), "bash", "{}")
    )
    .is_none());
    assert!(apply_user_question_event(
        &timed,
        &SessionEvent::tool_call(
            1,
            2,
            ToolCallId::new("broken"),
            "ask_user_question",
            "{not json"
        )
    )
    .is_none());
}

#[test]
fn several_calls_are_tracked_at_once_and_a_re_ask_moves_the_call() {
    let other = SessionEvent::tool_call(
        1,
        2,
        ToolCallId::new("call_ask_2"),
        "ask_user_question",
        r#"{"questions":[{"id":"name","question":"Name?"}]}"#,
    );
    let view = fold_user_questions(&[timed_header(), asked(CALL_ID), other.clone()]);
    assert_eq!(view.active.len(), 2);
    assert_eq!(view.active[0].call_id, ToolCallId::new(CALL_ID));
    assert_eq!(view.active[1].call_id, ToolCallId::new("call_ask_2"));

    let continued = fold_user_questions(&[
        timed_header(),
        asked(CALL_ID),
        other.clone(),
        resulted(r#"{"pending":true,"callId":"call_ask_1"}"#),
    ]);
    assert_eq!(continued.active[0].state, UserQuestionState::Continued);
    assert_eq!(continued.active[1].state, UserQuestionState::Open);

    // A re-ask under the same call id replaces the entry in place, at the end.
    let reasked = fold_user_questions(&[
        timed_header(),
        asked(CALL_ID),
        other.clone(),
        resulted(r#"{"pending":true,"callId":"call_ask_1"}"#),
        asked(CALL_ID),
    ]);
    assert_eq!(reasked.active.len(), 2);
    assert_eq!(reasked.active[0].call_id, ToolCallId::new("call_ask_2"));
    assert_eq!(reasked.active[1].call_id, ToolCallId::new(CALL_ID));
    assert_eq!(reasked.active[1].state, UserQuestionState::Open);
}

#[test]
fn a_tracked_call_whose_result_carries_no_text_is_closed() {
    let view = fold_user_questions(&[
        timed_header(),
        asked(CALL_ID),
        SessionEvent::tool_result(
            1,
            1,
            ToolResultMessage::new(
                MessageId::new("result-1"),
                Vec::new(),
                ToolCallId::new(CALL_ID),
                false,
            ),
        ),
    ]);
    assert!(view.active.is_empty());
    assert!(view.settled.is_empty());
}

#[test]
fn the_projection_folds_the_log_in_session_order() {
    let mut session = Session::create(
        SessionId::new("sess-projection").unwrap(),
        SessionMeta::default(),
        0,
    );
    let registry = ProjectionRegistry::new();
    let _registration = registry.register(user_question_projection()).unwrap();
    let inherited = SessionLogOffset::new(0).unwrap();

    let events = vec![
        timed_header(),
        asked(CALL_ID),
        resulted(r#"{"answers":[{"id":"scope","selected":["Tool only"]}]}"#),
    ];
    for event in events {
        let seq = session.append(event).unwrap();
        registry
            .drive_event(&session, inherited, seq)
            .expect("the projection folds the event");
        let state = registry
            .state_of::<UserQuestionProjectionState>(&session, inherited, KEY)
            .expect("the projection has a cell")
            .expect("the projection is registered");
        let expected = fold_user_questions(session.events());
        assert_eq!(state.questions.as_ref(), &expected);
    }

    let state = registry
        .state_of::<UserQuestionProjectionState>(&session, inherited, KEY)
        .unwrap()
        .unwrap();
    assert!(state.questions.active.is_empty());
    assert_eq!(state.questions.settled.len(), 1);
}

#[test]
fn the_projection_state_version_is_two() {
    assert_eq!(STATE_VERSION, 2);
    assert_eq!(KEY, "userQuestions");
}

#[test]
fn a_client_view_refuses_duplicate_call_ids() {
    let view = UserQuestionProjectionView {
        active: vec![
            PendingUserQuestion {
                call_id: ToolCallId::new("call-1"),
                questions: Vec::new(),
                state: UserQuestionState::Open,
            },
            PendingUserQuestion {
                call_id: ToolCallId::new("call-1"),
                questions: Vec::new(),
                state: UserQuestionState::Open,
            },
        ],
        settled: Vec::new(),
    };
    assert!(validate_projection_view(&view).is_err());
}

#[test]
fn an_answer_item_carries_its_selection_and_custom_text() {
    let item = AskUserQuestionAnswerItem {
        id: "scope".to_string(),
        selected: vec!["Tool only".to_string()],
        custom: Some("also docs".to_string()),
    };
    let value = serde_json::to_value(&item).unwrap();
    assert_eq!(value["id"], json!("scope"));
    assert_eq!(value["selected"][0], json!("Tool only"));
    assert_eq!(value["custom"], json!("also docs"));
}
