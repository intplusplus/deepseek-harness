//! The todo tool's behavior, ported from `packages/todo/tool-todo/tests/
//! tool-todo.spec.ts`.
//!
//! The TypeScript suite mounts the real tool on a Cordis registry and a session
//! store. What matters here is the part with rules: the description the model
//! reads, the schema boundary's path-qualified violations, the canonical list,
//! the locked output value, the durable snapshot, and the two-fold boundaries.
//! The append itself belongs to the caller, so these tests assert the event the
//! caller would append and its exact bytes.

use dsh_llm::{ContentBlock, CoreContentBlock};
use dsh_session::SessionEvent;
use dsh_todo::{
    apply_todo_state, decide_todo_write, decode_todo_write, describe, encode_todo_write,
    execute_todo_write, fold_todo_state, output_schema, parameter_schema, present_call,
    validate_arguments, TodoItem, TodoStatus, TodoWriteConfig, TodoWriteError, TodoWriteEvent,
    TODO_WRITE_TAG,
};
use dsh_tools::ToolExecutionResult;
use serde_json::{json, Map, Value};

fn text_of(result: &ToolExecutionResult) -> String {
    match result.content() {
        [ContentBlock::Core(CoreContentBlock::Text { text })] => text.clone(),
        other => format!("{other:?}"),
    }
}

fn failure_text(result: &ToolExecutionResult) -> String {
    text_of(result)
}

#[test]
fn the_description_changes_only_the_active_status_clause() {
    assert_eq!(
        describe(false),
        "Record and update a task list to plan multi-step work and show progress; skip it for \
         trivial single-step tasks. Add one todo per concrete step before you start. While work \
         remains, keep exactly one todo `in_progress`. Mark each todo `completed` as soon as it is \
         done."
    );
    assert_eq!(
        describe(true),
        "Record and update a task list to plan multi-step work and show progress; skip it for \
         trivial single-step tasks. Add one todo per concrete step before you start. While work \
         remains, keep the todos being worked on `in_progress`, several only when work runs in \
         parallel. Mark each todo `completed` as soon as it is done."
    );
}

#[test]
fn the_parameter_schema_is_the_compiled_typescript_bytes() {
    assert_eq!(
        parameter_schema().to_string(),
        concat!(
            r#"{"type":"object","properties":{"todos":{"type":"array","description":"The COMPLETE task list, replacing any previous list.","items":"#,
            r#"{"type":"object","additionalProperties":false,"properties":{"content":{"type":"string","description":"What the task is — a short imperative line."},"#,
            r#""status":{"type":"string","description":"pending (not started) | in_progress (now) | completed (done).","enum":["pending","in_progress","completed"]}},"#,
            r#""required":["content","status"]}}},"required":["todos"]}"#
        )
    );
}

#[test]
fn the_output_schema_closes_every_object_and_requires_counts() {
    assert_eq!(
        output_schema().to_string(),
        concat!(
            r#"{"type":"object","additionalProperties":false,"properties":{"todos":{"type":"array","items":"#,
            r#"{"type":"object","additionalProperties":false,"properties":{"content":{"type":"string"},"status":{"type":"string","enum":["pending","in_progress","completed"]}},"required":["content","status"]}},"#,
            r#""counts":{"type":"object","additionalProperties":false,"properties":{"pending":{"type":"integer"},"inProgress":{"type":"integer"},"completed":{"type":"integer"}},"required":["pending","inProgress","completed"]}},"#,
            r#""required":["todos","counts"]}"#
        )
    );
}

#[test]
fn a_write_trims_content_and_returns_the_locked_value_and_confirmation() {
    let arguments = json!({
        "todos": [
            { "content": "  write the port  ", "status": "in_progress" },
            { "content": "run the tests", "status": "pending" }
        ]
    });
    let call = execute_todo_write(&arguments, false, true);
    assert!(!call.result.is_error());
    assert_eq!(
        text_of(&call.result),
        "Updated todo list: 1 pending, 1 in progress, 0 completed."
    );
    let value = match &call.result {
        ToolExecutionResult::Success { value, .. } => value,
        ToolExecutionResult::Failure { .. } => panic!("expected a successful todo write"),
    };
    assert_eq!(
        value.to_string(),
        concat!(
            r#"{"todos":[{"content":"write the port","status":"in_progress"},"#,
            r#"{"content":"run the tests","status":"pending"}],"#,
            r#""counts":{"pending":1,"inProgress":1,"completed":0}}"#
        )
    );
    let event = call.event.expect("a successful call carries its snapshot");
    assert_eq!(
        serde_json::to_string(&event).expect("the snapshot is serializable"),
        concat!(
            r#"{"type":"todo/write","todos":[{"content":"write the port","status":"in_progress"},"#,
            r#"{"content":"run the tests","status":"pending"}]}"#
        )
    );
}

#[test]
fn an_empty_content_line_is_rejected() {
    for content in ["", "   ", "\t\n"] {
        let arguments = json!({ "todos": [{ "content": content, "status": "pending" }] });
        let call = execute_todo_write(&arguments, false, true);
        assert!(call.result.is_error());
        assert_eq!(
            failure_text(&call.result),
            "Error: invalid todo: `content` must be a non-empty string"
        );
        assert!(call.event.is_none());
    }
}

#[test]
fn duplicate_content_is_rejected_with_the_json_quoted_value() {
    let arguments = json!({
        "todos": [
            { "content": "same", "status": "pending" },
            { "content": "  same  ", "status": "completed" }
        ]
    });
    let call = execute_todo_write(&arguments, false, true);
    assert_eq!(
        failure_text(&call.result),
        "Error: invalid todos: duplicate content \"same\""
    );
}

#[test]
fn a_single_active_deployment_rejects_two_in_progress_entries() {
    let arguments = json!({
        "todos": [
            { "content": "one", "status": "in_progress" },
            { "content": "two", "status": "in_progress" }
        ]
    });
    let call = execute_todo_write(&arguments, false, true);
    assert_eq!(
        failure_text(&call.result),
        "Error: invalid todos: at most one task may be in_progress (got 2)"
    );
    let parallel = execute_todo_write(&arguments, true, true);
    assert!(!parallel.result.is_error());
}

#[test]
fn a_caller_without_an_owning_session_is_rejected_after_canonicalization() {
    let arguments = json!({ "todos": [{ "content": "a", "status": "pending" }] });
    let call = execute_todo_write(&arguments, false, false);
    assert_eq!(
        failure_text(&call.result),
        "Error: todo_write requires an owning agent session"
    );
    // Canonicalization errors still win: the TypeScript body validates first.
    let invalid = execute_todo_write(
        &json!({ "todos": [{ "content": "   ", "status": "pending" }] }),
        false,
        false,
    );
    assert!(invalid.result.is_error());
    assert!(!failure_text(&invalid.result).contains("owning agent session"));
}

#[test]
fn schema_violations_are_path_qualified_and_joined_in_walk_order() {
    assert_eq!(
        validate_arguments(&json!("not an object")),
        vec!["\"arguments\" must be an object".to_string()]
    );
    assert_eq!(
        validate_arguments(&json!({})),
        vec!["missing required property \"todos\"".to_string()]
    );
    assert_eq!(
        validate_arguments(&json!({ "todos": {} })),
        vec!["\"todos\" must be an array".to_string()]
    );
    assert_eq!(
        validate_arguments(&json!({ "todos": [1] })),
        vec!["\"todos[0]\" must be an object".to_string()]
    );
    assert_eq!(
        validate_arguments(&json!({ "todos": [{ "status": "bogus" }] })),
        vec![
            "missing required property \"todos[0].content\"".to_string(),
            "\"todos[0].status\" must be one of [\"pending\",\"in_progress\",\"completed\"]"
                .to_string(),
        ]
    );
    // Required violations are collected before child violations, exactly as the
    // registry's validator walks the object.
    assert_eq!(
        validate_arguments(&json!({ "todos": [{ "content": 1 }] })),
        vec![
            "missing required property \"todos[0].status\"".to_string(),
            "\"todos[0].content\" must be a string".to_string(),
        ]
    );
    assert_eq!(
        validate_arguments(&json!({
            "todos": [{ "content": "a", "status": "pending", "extra": true }]
        })),
        vec![
            "\"todos[0].extra\" is not a declared property (additionalProperties: false)"
                .to_string()
        ]
    );
}

#[test]
fn schema_violations_become_one_invalid_arguments_failure() {
    let call = execute_todo_write(&json!({ "todos": [{ "content": 1 }] }), false, true);
    assert_eq!(
        failure_text(&call.result),
        "Error: invalid arguments: missing required property \"todos[0].status\"; \
         \"todos[0].content\" must be a string"
    );
}

#[test]
fn a_rejected_call_never_produces_a_snapshot() {
    let call = execute_todo_write(&json!({ "todos": "nope" }), false, true);
    assert!(call.event.is_none());
    assert!(call.result.is_error());
}

#[test]
fn the_snapshot_round_trips_through_the_extension_slot() {
    let event = TodoWriteEvent {
        todos: vec![
            TodoItem::new("one", TodoStatus::Pending),
            TodoItem::new("two", TodoStatus::Completed),
        ],
    };
    let encoded = encode_todo_write(&event);
    assert_eq!(
        encoded.extension().map(|extension| extension.kind()),
        Some(TODO_WRITE_TAG)
    );
    match decode_todo_write(&encoded) {
        Some(Ok(decoded)) => assert_eq!(decoded, event),
        other => panic!("expected the snapshot back, got {other:?}"),
    }
}

#[test]
fn a_known_tag_with_a_bad_payload_errors_and_an_unknown_tag_is_ignored() {
    let mut payload = Map::new();
    payload.insert("todos".to_string(), json!("not an array"));
    let malformed = SessionEvent::extension_event(TODO_WRITE_TAG, payload);
    assert!(matches!(decode_todo_write(&malformed), Some(Err(_))));

    let other = SessionEvent::extension_event("someone/else", Map::new());
    assert!(decode_todo_write(&other).is_none());
}

#[test]
fn the_fold_keeps_the_latest_write_and_clears_at_the_next_turn_start() {
    let first = encode_todo_write(&TodoWriteEvent {
        todos: vec![TodoItem::new("first", TodoStatus::Pending)],
    });
    let second = encode_todo_write(&TodoWriteEvent {
        todos: vec![TodoItem::new("second", TodoStatus::Completed)],
    });
    assert_eq!(fold_todo_state(&[]).expect("empty log folds"), None);
    assert_eq!(
        fold_todo_state(std::slice::from_ref(&first)).expect("one write folds"),
        Some(vec![TodoItem::new("first", TodoStatus::Pending)])
    );
    assert_eq!(
        fold_todo_state(&[first, second]).expect("last write wins"),
        Some(vec![TodoItem::new("second", TodoStatus::Completed)])
    );
}

#[test]
fn a_turn_start_clears_the_list_and_a_turn_end_keeps_it() {
    let write = encode_todo_write(&TodoWriteEvent {
        todos: vec![TodoItem::new("task", TodoStatus::InProgress)],
    });
    let turn_start = SessionEvent::turn_start(0);
    let turn_end = SessionEvent::turn_end(0, dsh_session::TurnEndReason::Completed);

    let after_start = apply_todo_state(
        fold_todo_state(std::slice::from_ref(&write)).expect("write folds"),
        &turn_start,
    )
    .expect("turn start folds");
    assert_eq!(after_start, None);

    let after_end = apply_todo_state(
        fold_todo_state(std::slice::from_ref(&write)).expect("write folds"),
        &turn_end,
    )
    .expect("turn end folds");
    assert_eq!(
        after_end,
        Some(vec![TodoItem::new("task", TodoStatus::InProgress)])
    );
}

#[test]
fn unrelated_events_leave_the_state_untouched() {
    let write = encode_todo_write(&TodoWriteEvent {
        todos: vec![TodoItem::new("task", TodoStatus::Pending)],
    });
    let state = fold_todo_state(&[write]).expect("write folds");
    let unrelated = SessionEvent::step_start(0, 0);
    assert_eq!(
        apply_todo_state(state.clone(), &unrelated).expect("step folds"),
        state
    );
}

#[test]
fn a_malformed_snapshot_stops_the_fold() {
    let mut payload = Map::new();
    payload.insert("todos".to_string(), Value::Null);
    let malformed = SessionEvent::extension_event(TODO_WRITE_TAG, payload);
    assert!(fold_todo_state(&[malformed]).is_err());
}

#[test]
fn the_decision_is_pure_and_the_event_matches_its_value() {
    let arguments = json!({ "todos": [{ "content": "a", "status": "completed" }] });
    let decision = decide_todo_write(&arguments, false, true).expect("valid arguments");
    assert_eq!(
        decision.output.todos,
        vec![TodoItem::new("a", TodoStatus::Completed)]
    );
    assert_eq!(decision.output.counts.completed, 1);
    let encoded = serde_json::to_string(&decision.event).expect("event serializes");
    assert_eq!(
        encoded,
        r#"{"type":"todo/write","todos":[{"content":"a","status":"completed"}]}"#
    );
}

#[test]
fn the_config_requires_an_explicit_parallel_policy() {
    let config: TodoWriteConfig =
        serde_json::from_str(r#"{"allowParallelInProgress":true}"#).expect("config parses");
    assert!(config.allows_parallel());
    assert!(serde_json::from_str::<TodoWriteConfig>("{}").is_err());
    assert!(serde_json::from_str::<TodoWriteConfig>(
        r#"{"allowParallelInProgress":true,"extra":1}"#
    )
    .is_err());
}

#[test]
fn presentation_is_the_generic_card_carrying_the_raw_list() {
    let arguments = json!({ "todos": [{ "content": "a", "status": "pending" }] });
    assert_eq!(
        serde_json::to_string(&present_call(&arguments)).expect("presentation serializes"),
        r#"{"card":"generic","title":"Update todo list","kind":"other","rawInput":[{"content":"a","status":"pending"}]}"#
    );
    assert_eq!(
        serde_json::to_string(&present_call(&json!({}))).expect("presentation serializes"),
        r#"{"card":"generic","title":"Update todo list","kind":"other","rawInput":[]}"#
    );
}

#[test]
fn the_error_types_render_the_model_visible_text() {
    assert_eq!(
        TodoWriteError::EmptyContent.to_string(),
        "invalid todo: `content` must be a non-empty string"
    );
    assert_eq!(
        TodoWriteError::DuplicateContent("\"a\"".to_string()).to_string(),
        "invalid todos: duplicate content \"a\""
    );
    assert_eq!(
        TodoWriteError::AtMostOneInProgress(3).to_string(),
        "invalid todos: at most one task may be in_progress (got 3)"
    );
    assert_eq!(
        TodoWriteError::NoOwningSession.to_string(),
        "todo_write requires an owning agent session"
    );
}
