//! Turn driving: step structure, tool-driven continuation, and failure ends.

use std::sync::{Arc, Mutex};

use dsh_agent_loop::{AgentLoop, LoopConfig};
use dsh_llm::{ContentBlock, FinishReason, LlmFailure, Message, MessageId, ToolSchema};
use dsh_provider::{
    text_response, tool_call_response, user_message, EchoProvider, LlmResponse, ScriptedProvider,
};
use dsh_session::{CoreEvent, RequestContext, Session, SessionId, SessionMeta};
use dsh_tools::{
    CancelSignal, ToolCallInput, ToolExecutionResult, ToolHandler, ToolRegistry, ToolScheduleConfig,
};

/// A tool that echoes its `text` argument.
struct EchoTool;

impl ToolHandler for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "echo",
            "Echo text back",
            serde_json::json!({ "type": "object", "properties": { "text": { "type": "string" } } }),
        )
    }

    fn call(&self, input: &ToolCallInput, _cancel: &CancelSignal) -> ToolExecutionResult {
        let text = dsh_tools::argument(input, "text")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        ToolExecutionResult::success(vec![ContentBlock::text(format!("echo:{text}"))])
    }
}

/// A tool whose success concludes the turn and defers context.
struct ConcludeTool;

impl ToolHandler for ConcludeTool {
    fn name(&self) -> &str {
        "finish"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "finish",
            "ends the turn",
            serde_json::json!({ "type": "object" }),
        )
    }

    fn call(&self, _input: &ToolCallInput, _cancel: &CancelSignal) -> ToolExecutionResult {
        ToolExecutionResult::success(vec![ContentBlock::text("done")])
            .concluding_turn()
            .with_contexts(vec![user_message("wrap up")])
    }
}

fn session() -> Session {
    Session::create(SessionId::new("t").unwrap(), SessionMeta::default(), 0)
}

fn route() -> RequestContext {
    RequestContext {
        provider: "demo".to_string(),
        model: "demo-1".to_string(),
        context_window: Some(128_000),
        system_prompt_update: None,
    }
}

fn registry_with(tools: Vec<Arc<dyn ToolHandler>>) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for tool in tools {
        registry.register(tool);
    }
    registry
}

/// The sequence of core event tags the log holds.
fn tags(session: &Session) -> Vec<String> {
    session
        .events()
        .iter()
        .filter_map(|event| match event.core() {
            Some(CoreEvent::TurnStart { .. }) => Some("turn/start"),
            Some(CoreEvent::TurnEnd { .. }) => Some("turn/end"),
            Some(CoreEvent::StepStart { .. }) => Some("step/start"),
            Some(CoreEvent::StepEnd { .. }) => Some("step/end"),
            Some(CoreEvent::UserMessage { .. }) => Some("user/message"),
            Some(CoreEvent::AssistantMessage { .. }) => Some("assistant/message"),
            Some(CoreEvent::ToolCall { .. }) => Some("tool/call"),
            Some(CoreEvent::ToolResult { .. }) => Some("tool/result"),
            Some(CoreEvent::RequestHeader { .. }) => Some("request/header"),
            Some(CoreEvent::RequestContext { .. }) => Some("request/context"),
            _ => None,
        })
        .map(str::to_string)
        .collect()
}

fn failure(code: &str) -> LlmFailure {
    LlmFailure {
        message: "boom".to_string(),
        code: code.to_string(),
        status: None,
        provider_retry_after_ms: None,
        request_id: None,
        offload_images: None,
    }
}

#[tokio::test]
async fn a_turn_with_no_tool_calls_completes_in_one_step() {
    let mut session = session();
    let registry = registry_with(vec![Arc::new(EchoTool)]);
    let provider = EchoProvider;
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route());
    driver.send(user_message("hi"));

    let report = driver.run_turn().await.unwrap();

    assert_eq!(report.turn, 1);
    assert_eq!(report.steps, 1);
    assert_eq!(report.reason, dsh_session::TurnEndReason::Completed);
    assert_eq!(
        tags(&session),
        vec![
            "turn/start",
            "user/message",
            "step/start",
            "request/header",
            "request/context",
            "assistant/message",
            "step/end",
            "turn/end",
        ]
    );
}

/// The request the loop builds comes from the log, not from driver state.
#[tokio::test]
async fn the_request_is_derived_from_the_log() {
    let seen: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let provider = dsh_provider::FnProvider::new(move |request| {
        sink.lock().unwrap().push(
            request
                .messages
                .iter()
                .map(Message::role)
                .map(str::to_string)
                .collect(),
        );
        let message = dsh_llm::AssistantMessage::new(
            MessageId::random(),
            vec![ContentBlock::text("ok")],
            &request.config.provider,
            &request.config.model,
        );
        Ok(LlmResponse::new(message))
    });

    let mut session = session();
    let registry = registry_with(vec![]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route());
    driver.send(user_message("hi"));
    driver.run_turn().await.unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(*seen, vec![vec!["user".to_string()]]);
}

#[tokio::test]
async fn a_tool_call_opens_another_step() {
    let mut session = session();
    let registry = registry_with(vec![Arc::new(EchoTool)]);
    let provider = ScriptedProvider::new(vec![
        tool_call_response("demo", "demo-1", "c1", "echo", r#"{"text":"hello"}"#),
        text_response("demo", "demo-1", "all done"),
    ]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route());
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    assert_eq!(report.steps, 2);
    assert_eq!(report.reason, dsh_session::TurnEndReason::Completed);
    assert_eq!(
        tags(&session),
        vec![
            "turn/start",
            "user/message",
            "step/start",
            "request/header",
            "request/context",
            "assistant/message",
            "tool/call",
            "tool/result",
            "step/end",
            "step/start",
            "assistant/message",
            "step/end",
            "turn/end",
        ]
    );
    // The second request saw the tool's result.
    let messages = dsh_session::derive_messages(session.events());
    assert!(messages
        .iter()
        .any(|message| matches!(message, Message::Tool(_))));
}

#[tokio::test]
async fn a_concluding_tool_ends_the_turn_after_one_step() {
    let mut session = session();
    let registry = registry_with(vec![Arc::new(ConcludeTool)]);
    let provider = ScriptedProvider::new(vec![tool_call_response(
        "demo", "demo-1", "c1", "finish", "{}",
    )]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route());
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    assert_eq!(report.steps, 1);
    assert_eq!(report.reason, dsh_session::TurnEndReason::Completed);
    // The deferred context became model-visible input.
    assert_eq!(
        tags(&session)
            .iter()
            .filter(|tag| *tag == "user/message")
            .count(),
        2
    );
    assert_eq!(provider.remaining(), 0);
}

#[tokio::test]
async fn a_provider_failure_ends_the_turn_with_that_failure() {
    let mut session = session();
    let registry = registry_with(vec![]);
    let provider = dsh_provider::FnProvider::new(|_| Err(failure("RATE_LIMITED")));
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route());

    let report = driver.run_turn().await.unwrap();

    match &report.reason {
        dsh_session::TurnEndReason::Error { error } => assert_eq!(error.code, "RATE_LIMITED"),
        other => panic!("expected an error reason, got {other:?}"),
    }
    assert_eq!(report.steps, 1);
}

#[tokio::test]
async fn max_tokens_survives_as_the_turn_reason() {
    let mut session = session();
    let registry = registry_with(vec![]);
    let provider =
        ScriptedProvider::new(vec![text_response("demo", "demo-1", "truncated")
            .with_finish_reason(FinishReason::MaxTokens)]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route());

    let report = driver.run_turn().await.unwrap();
    assert_eq!(report.reason, dsh_session::TurnEndReason::MaxTokens);
}

/// The step guard is a Rust-only bound; it ends the turn instead of hanging.
#[tokio::test]
async fn the_step_limit_guards_a_model_that_never_stops() {
    let mut session = session();
    let registry = registry_with(vec![Arc::new(EchoTool)]);
    let provider = ScriptedProvider::new(
        (1..=50)
            .map(|n| tool_call_response("demo", "demo-1", &format!("c{n}"), "echo", "{}"))
            .collect(),
    );
    let mut driver =
        AgentLoop::new(&mut session, &provider, &registry, route()).with_config(LoopConfig {
            max_steps: 3,
            tool_schedule: ToolScheduleConfig::default(),
            retry: None,
        });

    let report = driver.run_turn().await.unwrap();

    assert_eq!(report.steps, 3);
    match &report.reason {
        dsh_session::TurnEndReason::Error { error } => assert_eq!(error.code, "STEP_LIMIT"),
        other => panic!("expected the step limit, got {other:?}"),
    }
}

#[tokio::test]
async fn queued_input_becomes_user_messages_before_the_first_request() {
    let mut session = session();
    let registry = registry_with(vec![]);
    let provider = EchoProvider;
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route());
    driver.send(user_message("first"));
    driver.send(user_message("second"));

    driver.run_turn().await.unwrap();

    assert_eq!(
        tags(&session)
            .iter()
            .filter(|tag| *tag == "user/message")
            .count(),
        2
    );
}

#[tokio::test]
async fn turns_continue_numbering_across_runs() {
    let mut session = session();
    let registry = registry_with(vec![]);
    let provider = EchoProvider;
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route());

    driver.send(user_message("one"));
    let first = driver.run_turn().await.unwrap();
    driver.send(user_message("two"));
    let second = driver.run_turn().await.unwrap();

    assert_eq!((first.turn, second.turn), (1, 2));
    let turns: Vec<u64> = session
        .events()
        .iter()
        .filter_map(|event| match event.core() {
            Some(CoreEvent::TurnStart { turn }) => Some(*turn),
            _ => None,
        })
        .collect();
    assert_eq!(turns, vec![1, 2]);
}

/// The log the loop produced survives a store-and-reload round trip, which is
/// what makes a replay a re-derivation.
#[tokio::test]
async fn a_driven_turn_round_trips_through_jsonl() {
    let mut session = session();
    let registry = registry_with(vec![Arc::new(EchoTool)]);
    let provider = ScriptedProvider::new(vec![
        tool_call_response("demo", "demo-1", "c1", "echo", r#"{"text":"hello"}"#),
        text_response("demo", "demo-1", "done"),
    ]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route());
    driver.send(user_message("go"));
    driver.run_turn().await.unwrap();

    let directory = std::env::temp_dir().join(format!("dsh-loop-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    dsh_session::jsonl::write_session(&directory, &session).unwrap();
    let reloaded = dsh_session::jsonl::read_session(&directory).unwrap();
    let _ = std::fs::remove_dir_all(&directory);

    assert_eq!(reloaded, session);
}

/// A tool result is paired with the call that requested it, so the derived
/// history never shows an unanswered call.
#[tokio::test]
async fn every_tool_call_in_a_driven_turn_is_answered() {
    let mut session = session();
    let registry = registry_with(vec![Arc::new(EchoTool)]);
    let provider = ScriptedProvider::new(vec![
        tool_call_response("demo", "demo-1", "c1", "echo", "{}"),
        text_response("demo", "demo-1", "done"),
    ]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route());
    driver.run_turn().await.unwrap();

    let mut asked: Vec<String> = Vec::new();
    let mut answered: Vec<String> = Vec::new();
    for event in session.events() {
        match event.core() {
            Some(CoreEvent::ToolCall { call_id, .. }) => asked.push(call_id.as_str().to_string()),
            Some(CoreEvent::ToolResult { message, .. }) => {
                answered.push(message.tool_call_id.as_str().to_string())
            }
            _ => {}
        }
    }
    assert_eq!(asked, answered);
}
