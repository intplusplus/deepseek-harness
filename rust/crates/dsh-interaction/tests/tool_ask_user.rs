//! The model-facing `ask_user_question` tool: its schema, its validation, and
//! the question → answer → result path.

use std::sync::{Arc, Mutex};

use dsh_agent::Agent;
use dsh_interaction::ask_user::*;
use dsh_interaction::questions::*;
use dsh_llm::ToolCallId;
use dsh_session::SessionId;
use dsh_tools::{CancelSignal, ToolCallInput, ToolExecutionResult, ToolHandler};

/// An answerer that records every request it sees.
struct ScriptedAnswerer {
    seen: Mutex<Vec<AskUserQuestionRequest>>,
    answer: AskUserQuestionAnswer,
}

impl ScriptedAnswerer {
    fn new(answer: AskUserQuestionAnswer) -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
            answer,
        }
    }
}

impl UserQuestionAnswerer for ScriptedAnswerer {
    fn ask(
        &self,
        request: &AskUserQuestionRequest,
    ) -> Result<AskUserQuestionAnswer, UserQuestionError> {
        self.seen.lock().unwrap().push(request.clone());
        Ok(self.answer.clone())
    }
}

/// An answerer whose foreground wait always expires.
struct TimingOutAnswerer;

impl UserQuestionAnswerer for TimingOutAnswerer {
    fn ask(
        &self,
        _request: &AskUserQuestionRequest,
    ) -> Result<AskUserQuestionAnswer, UserQuestionError> {
        Err(UserQuestionError::AskTimedOut)
    }
}

/// A live-agent registry with fixed answers.
struct StubAgents;

impl LiveAgentRegistry for StubAgents {
    fn is_live(&self, _agent: &Agent) -> bool {
        true
    }

    fn is_root(&self, _agent: &Agent) -> bool {
        true
    }
}

fn agent(id: &str) -> Agent {
    Agent::new(SessionId::new(id).unwrap())
}

fn input(arguments: serde_json::Value) -> ToolCallInput {
    ToolCallInput {
        call_id: ToolCallId::new("call-1"),
        name: "ask_user_question".to_string(),
        arguments,
    }
}

fn questions_argument() -> serde_json::Value {
    serde_json::json!({"questions": [{"id": "continue", "question": "Continue?"}]})
}

fn answer_for(id: &str, selected: &[&str], custom: Option<&str>) -> AskUserQuestionAnswer {
    AskUserQuestionAnswer {
        answers: vec![AskUserQuestionAnswerItem {
            id: id.to_string(),
            selected: selected.iter().map(|label| (*label).to_string()).collect(),
            custom: custom.map(str::to_string),
        }],
    }
}

fn text_of(result: &ToolExecutionResult) -> String {
    result.content()[0]
        .as_text()
        .unwrap_or_default()
        .to_string()
}

#[test]
fn the_legacy_tool_declares_no_wait_parameter() {
    let service = Arc::new(UserQuestionService::new());
    let tool = AskUserQuestionTool::legacy(service);
    let schema = tool.schema();

    assert_eq!(schema.name, "ask_user_question");
    assert!(schema
        .description
        .contains("Ask the user a concise question"));
    let properties = schema.parameters.get("properties").unwrap();
    assert!(properties.get("timeout").is_none());
    assert!(properties.get("questions").is_some());
}

#[test]
fn the_timed_tool_declares_the_wait_parameter() {
    let service = Arc::new(UserQuestionService::new());
    let tool = AskUserQuestionTool::timed(service, 45).unwrap();
    let schema = tool.schema();

    let parameters = serde_json::to_string(&schema.parameters).unwrap();
    assert!(parameters.contains("timeout"));
    assert!(parameters
        .contains("Wait seconds for the entire batch (default 45); omit unless the user specifies a duration. Use -1 only when an answer is required before proceeding."));
    assert!(!parameters.contains("default 120"));
}

#[test]
fn the_timed_tool_falls_back_to_the_default_wait() {
    let service = Arc::new(UserQuestionService::new());
    let tool = AskUserQuestionTool::timed(service, 120).unwrap();
    assert_eq!(tool.timeout_seconds(), 120);
    let parameters = serde_json::to_string(&tool.schema().parameters).unwrap();
    assert!(parameters.contains("default 120"));
}

#[test]
fn the_tool_returns_the_answer_batch() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for(
        "continue",
        &["yes"],
        None,
    )));
    let service = Arc::new(UserQuestionService::new().with_answerer(answerer.clone()));
    let tool = AskUserQuestionTool::legacy(service);

    let result = tool.call(&input(questions_argument()), &CancelSignal::new());
    assert!(!result.is_error());
    assert_eq!(
        text_of(&result),
        r#"{"answers":[{"id":"continue","selected":["yes"]}]}"#
    );
}

#[test]
fn a_custom_answer_comes_back_verbatim() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for("pkg", &[], Some("bun"))));
    let service = Arc::new(UserQuestionService::new().with_answerer(answerer));
    let tool = AskUserQuestionTool::legacy(service);

    let result = tool.call(&input(questions_argument()), &CancelSignal::new());
    assert_eq!(
        text_of(&result),
        r#"{"answers":[{"id":"pkg","selected":[],"custom":"bun"}]}"#
    );
}

#[test]
fn the_legacy_request_stays_unkeyed() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for(
        "continue",
        &["yes"],
        None,
    )));
    let service = Arc::new(UserQuestionService::new().with_answerer(answerer.clone()));
    let tool = AskUserQuestionTool::legacy(service);

    tool.call(&input(questions_argument()), &CancelSignal::new());

    let seen = answerer.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert!(seen[0].wait.is_none());
    assert!(seen[0].agent.is_none());
    assert_eq!(seen[0].questions[0].id, "continue");
}

#[test]
fn the_timed_request_is_keyed_by_the_call_id() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for(
        "continue",
        &["yes"],
        None,
    )));
    let service = Arc::new(
        UserQuestionService::new()
            .with_answerer(answerer.clone())
            .with_live_agents(Arc::new(StubAgents)),
    );
    let tool = AskUserQuestionTool::timed(service, -1)
        .unwrap()
        .with_agent(agent("root"));

    tool.call(&input(questions_argument()), &CancelSignal::new());

    let seen = answerer.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    let wait = seen[0].wait.as_ref().expect("the timed request is keyed");
    assert_eq!(wait.call_id, ToolCallId::new("call-1"));
    // The indefinite form carries no deadline.
    assert_eq!(wait.timed, None);
}

#[test]
fn the_tool_rejects_a_duplicate_question_id_before_dispatching() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for("scope", &["Tools"], None)));
    let service = Arc::new(UserQuestionService::new().with_answerer(answerer.clone()));
    let tool = AskUserQuestionTool::legacy(service);

    let result = tool.call(
        &input(serde_json::json!({"questions": [
            {"id": "scope", "question": "Choose a scope."},
            {"id": "scope", "question": "Confirm the scope."},
        ]})),
        &CancelSignal::new(),
    );
    assert!(result.is_error());
    assert_eq!(
        text_of(&result),
        "Error: question id \"scope\" must be unique within this call"
    );
    assert!(answerer.seen.lock().unwrap().is_empty());
}

#[test]
fn the_tool_rejects_an_invalid_timeout() {
    let service = Arc::new(UserQuestionService::new());
    let tool = AskUserQuestionTool::legacy(service);

    for timeout in [0.0, 1.5, 2_147_484.0] {
        let mut arguments = questions_argument();
        arguments["timeout"] = serde_json::json!(timeout);
        let result = tool.call(&input(arguments), &CancelSignal::new());
        assert!(result.is_error(), "timeout {timeout} must be rejected");
        assert!(text_of(&result).contains("timeout"));
    }
}

#[test]
fn the_timed_tool_requires_a_live_agent() {
    let service = Arc::new(UserQuestionService::new());
    let tool = AskUserQuestionTool::timed(service, 30).unwrap();

    let result = tool.call(&input(questions_argument()), &CancelSignal::new());
    assert!(result.is_error());
    assert_eq!(
        text_of(&result),
        "Error: timed questions require a live agent"
    );
}

#[test]
fn the_timed_tool_reports_a_pending_question() {
    let service = Arc::new(
        UserQuestionService::new()
            .with_answerer(Arc::new(TimingOutAnswerer))
            .with_live_agents(Arc::new(StubAgents)),
    );
    let tool = AskUserQuestionTool::timed(service, 1)
        .unwrap()
        .with_agent(agent("root"));

    let result = tool.call(&input(questions_argument()), &CancelSignal::new());
    assert!(!result.is_error());

    let text = text_of(&result);
    let payload: serde_json::Value =
        serde_json::from_str(&text).expect("the result is one JSON object");
    assert_eq!(payload["pending"], serde_json::json!(true));
    assert_eq!(payload["callId"], serde_json::json!("call-1"));
    assert!(payload["message"]
        .as_str()
        .unwrap()
        .contains("This is pending, not a skipped answer."));
    assert!(text.contains("No answer batch arrived before the timeout."));
    assert!(text.contains("Do not treat this as permission."));
}

#[test]
fn the_tool_rejects_an_empty_question_batch() {
    let service = Arc::new(UserQuestionService::new());
    let tool = AskUserQuestionTool::legacy(service);

    let result = tool.call(
        &input(serde_json::json!({"questions": []})),
        &CancelSignal::new(),
    );
    assert!(result.is_error());
    assert_eq!(result.error_info().unwrap().code, "EMPTY_QUESTIONS");
    assert_eq!(result.error_info().unwrap().name, "UserQuestionError");
}

#[test]
fn the_tool_reports_a_missing_answerer() {
    let service = Arc::new(UserQuestionService::new());
    let tool = AskUserQuestionTool::legacy(service);

    let result = tool.call(&input(questions_argument()), &CancelSignal::new());
    assert!(result.is_error());
    assert_eq!(result.error_info().unwrap().code, "NO_PROVIDER");
}

#[test]
fn the_tool_rejects_arguments_without_a_questions_array() {
    let service = Arc::new(UserQuestionService::new());
    let tool = AskUserQuestionTool::legacy(service);

    let result = tool.call(&input(serde_json::json!({})), &CancelSignal::new());
    assert!(result.is_error());
    assert!(text_of(&result).contains("questions"));
}

#[test]
fn the_tool_maps_the_model_vocabulary_into_the_service_vocabulary() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for("pkg", &["pnpm"], None)));
    let service = Arc::new(UserQuestionService::new().with_answerer(answerer.clone()));
    let tool = AskUserQuestionTool::legacy(service);

    tool.call(
        &input(serde_json::json!({"questions": [{
            "id": "pkg",
            "question": "Which package manager?",
            "header": "Setup",
            "multi_select": false,
            "options": [
                {"label": "pnpm (Recommended)", "description": "Workspace default."},
                {"label": "npm"},
            ],
        }]})),
        &CancelSignal::new(),
    );

    let seen = answerer.seen.lock().unwrap();
    let question = &seen[0].questions[0];
    assert_eq!(question.id, "pkg");
    assert_eq!(question.header.as_deref(), Some("Setup"));
    assert_eq!(question.multi_select, Some(false));
    assert_eq!(question.options.as_ref().unwrap().len(), 2);
    assert_eq!(
        question.options.as_ref().unwrap()[0].label,
        "pnpm (Recommended)"
    );
}

#[test]
fn a_custom_and_multi_select_answer_survives_the_round_trip() {
    let answerer = Arc::new(ScriptedAnswerer::new(AskUserQuestionAnswer {
        answers: vec![
            AskUserQuestionAnswerItem {
                id: "targets".to_string(),
                selected: vec!["tests".to_string(), "docs".to_string()],
                custom: Some("release notes".to_string()),
            },
            AskUserQuestionAnswerItem {
                id: "optional".to_string(),
                selected: Vec::new(),
                custom: None,
            },
        ],
    }));
    let service = Arc::new(UserQuestionService::new().with_answerer(answerer));
    let tool = AskUserQuestionTool::legacy(service);

    let result = tool.call(&input(questions_argument()), &CancelSignal::new());
    assert_eq!(
        text_of(&result),
        r#"{"answers":[{"id":"targets","selected":["tests","docs"],"custom":"release notes"},{"id":"optional","selected":[]}]}"#
    );
}

#[test]
fn the_timeout_validation_rejects_the_values_outside_the_platform_timer() {
    assert_eq!(validate_timeout_seconds(-1).unwrap(), -1);
    assert_eq!(validate_timeout_seconds(1).unwrap(), 1);
    assert_eq!(validate_timeout_seconds(2_147_483).unwrap(), 2_147_483);
    assert!(validate_timeout_seconds(0).is_err());
    assert!(validate_timeout_seconds(2_147_484).is_err());
}
