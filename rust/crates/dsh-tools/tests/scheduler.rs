//! Tool scheduling: ordering, barriers, cancellation, and the settle invariant.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use dsh_llm::{ContentBlock, MessageId, ToolCallId, ToolSchema, UserMessage};
use dsh_session::{CoreEvent, Session, SessionId, SessionMeta};
use dsh_tools::*;

/// A tool that echoes a fixed text and records the order it ran in.
struct EchoTool {
    name: String,
    mode: ToolExecutionMode,
    log: Arc<Mutex<Vec<String>>>,
}

impl EchoTool {
    fn new(name: &str, mode: ToolExecutionMode, log: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            name: name.to_string(),
            mode,
            log,
        }
    }
}

impl ToolHandler for EchoTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name.clone(),
            "echo",
            serde_json::json!({ "type": "object", "properties": { "text": { "type": "string" } } }),
        )
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        self.mode
    }

    fn call(&self, input: &ToolCallInput, _cancel: &CancelSignal) -> ToolExecutionResult {
        self.log
            .lock()
            .unwrap()
            .push(input.call_id.as_str().to_string());
        let text = argument(input, "text")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        ToolExecutionResult::success(vec![ContentBlock::text(format!("{}:{}", self.name, text))])
            .with_value(serde_json::json!({ "echoed": text }))
    }
}

/// A tool that raises cancellation from inside its own body.
struct CancelTool {
    log: Arc<Mutex<Vec<String>>>,
}

impl ToolHandler for CancelTool {
    fn name(&self) -> &str {
        "cancel"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "cancel",
            "cancels the step",
            serde_json::json!({ "type": "object" }),
        )
    }

    /// Exclusive so the cancellation raised inside the body is observed before
    /// any sibling starts, preserving the sequential scheduler's mid-group
    /// semantics now that parallel siblings can overlap.
    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Exclusive
    }

    fn call(&self, input: &ToolCallInput, cancel: &CancelSignal) -> ToolExecutionResult {
        self.log
            .lock()
            .unwrap()
            .push(input.call_id.as_str().to_string());
        cancel.cancel();
        ToolExecutionResult::success(vec![ContentBlock::text("cancelled")])
    }
}

/// A tool that concludes the turn and defers context to the next request.
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
            .with_contexts(vec![UserMessage::new(
                MessageId::random(),
                vec![ContentBlock::text("wrapping up")],
            )])
    }
}

/// A tool body that waits, bounded, for a sibling body to be running.
///
/// The shared `peak` records the largest number of bodies active at once: two
/// with at least two scheduling slots, one when the bound serializes them. The
/// wait bound keeps the serialized case from hanging.
struct RendezvousTool {
    active: Arc<(Mutex<usize>, Condvar)>,
    peak: Arc<Mutex<usize>>,
    wait: Duration,
}

impl ToolHandler for RendezvousTool {
    fn name(&self) -> &str {
        "rendezvous"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "rendezvous",
            "waits for a sibling",
            serde_json::json!({ "type": "object" }),
        )
    }

    fn call(&self, input: &ToolCallInput, _cancel: &CancelSignal) -> ToolExecutionResult {
        let (lock, ready) = &*self.active;
        let mut active = lock.lock().unwrap();
        *active += 1;
        {
            let mut peak = self.peak.lock().unwrap();
            if *active > *peak {
                *peak = *active;
            }
        }
        ready.notify_all();
        let deadline = Instant::now() + self.wait;
        loop {
            if *self.peak.lock().unwrap() >= 2 {
                break;
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            let (guard, _) = ready.wait_timeout(active, remaining).unwrap();
            active = guard;
        }
        *active -= 1;
        drop(active);
        ToolExecutionResult::success(vec![ContentBlock::text(input.call_id.as_str())])
    }
}

/// A post-execute policy for the tests: records every call it sees, then
/// answers with a fixed decision and optional finalize content.
struct TestPost {
    decision: PostToolDecision,
    finalize: Option<Vec<ContentBlock>>,
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl PostToolGate for TestPost {
    async fn decide(
        &self,
        call: &ToolCallInput,
        _result: &ToolExecutionResult,
    ) -> PostToolDecision {
        self.seen
            .lock()
            .unwrap()
            .push(call.call_id.as_str().to_string());
        self.decision.clone()
    }

    fn finalize(
        &self,
        _call: &ToolCallInput,
        _result: &ToolExecutionResult,
    ) -> Option<Vec<ContentBlock>> {
        self.finalize.clone()
    }
}

fn session() -> Session {
    Session::create(SessionId::new("t").unwrap(), SessionMeta::default(), 0)
}

fn call(id: &str, name: &str, arguments: &str) -> ContentBlock {
    ContentBlock::tool_call(ToolCallId::new(id), name, arguments)
}

fn schedule<'a>(registry: &'a ToolRegistry, cancel: &'a CancelSignal) -> ToolSchedule<'a> {
    ToolSchedule {
        turn: 1,
        step: 1,
        registry,
        config: ToolScheduleConfig::default(),
        cancel,
    }
}

/// Every committed `tool/result` cites the call id the model asked for, in
/// model order.
///
/// Parallel bodies overlap, so calls and results are no longer adjacent in the
/// log; pairing is by the id each result answers, while both sequences stay in
/// model order.
fn committed_pairs(session: &Session) -> Vec<(String, String)> {
    let calls: Vec<String> = session
        .events()
        .iter()
        .filter_map(|event| match event.core() {
            Some(CoreEvent::ToolCall { call_id, .. }) => Some(call_id.as_str().to_string()),
            _ => None,
        })
        .collect();
    let results: Vec<String> = session
        .events()
        .iter()
        .filter_map(|event| match event.core() {
            Some(CoreEvent::ToolResult { message, .. }) => {
                Some(message.tool_call_id.as_str().to_string())
            }
            _ => None,
        })
        .collect();
    assert_eq!(calls, results, "every call settles once, in model order");
    calls.into_iter().zip(results).collect()
}

/// The error code of every committed result, or `none` when it succeeded.
fn result_codes(session: &Session) -> Vec<String> {
    session
        .events()
        .iter()
        .filter_map(|event| match event.core() {
            Some(CoreEvent::ToolResult { error, .. }) => Some(
                error
                    .as_ref()
                    .map_or_else(|| "none".to_string(), |e| e.code.clone()),
            ),
            _ => None,
        })
        .collect()
}

/// The rendered text of every committed result, in commit order.
fn result_texts(session: &Session) -> Vec<String> {
    session
        .events()
        .iter()
        .filter_map(|event| match event.core() {
            Some(CoreEvent::ToolResult { message, .. }) => Some(
                message
                    .content
                    .iter()
                    .filter_map(ContentBlock::as_text)
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => None,
        })
        .collect()
}

/// Whether each committed result is marked as an error, in commit order.
fn result_is_error(session: &Session) -> Vec<bool> {
    session
        .events()
        .iter()
        .filter_map(|event| match event.core() {
            Some(CoreEvent::ToolResult { message, .. }) => Some(message.is_error),
            _ => None,
        })
        .collect()
}

/// The `info.name` of every committed result's error, in commit order.
fn result_error_names(session: &Session) -> Vec<String> {
    session
        .events()
        .iter()
        .filter_map(|event| match event.core() {
            Some(CoreEvent::ToolResult { error, .. }) => error
                .as_ref()
                .map(|error| error.name.clone())
                .or_else(|| Some("none".to_string())),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn results_commit_in_model_order() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));

    let mut session = session();
    let calls = vec![
        call("c1", "echo", r#"{"text":"a"}"#),
        call("c2", "echo", r#"{"text":"b"}"#),
        call("c3", "echo", r#"{"text":"c"}"#),
    ];
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls(
        &mut session,
        &schedule(&registry, &cancel),
        &calls,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 3);
    assert!(!outcome.aborted);
    let mut bodies = log.lock().unwrap().clone();
    bodies.sort();
    assert_eq!(bodies, vec!["c1", "c2", "c3"], "every body ran");
    assert_eq!(
        committed_pairs(&session),
        vec![
            ("c1".to_string(), "c1".to_string()),
            ("c2".to_string(), "c2".to_string()),
            ("c3".to_string(), "c3".to_string()),
        ]
    );
}

/// Bodies really overlap up to the configured bound: two slots let each body
/// observe its sibling, while one slot serializes them and both time out.
#[tokio::test]
async fn parallel_bodies_overlap_up_to_the_configured_bound() {
    async fn run(max: usize, wait: Duration) -> usize {
        let active = Arc::new((Mutex::new(0usize), Condvar::new()));
        let peak = Arc::new(Mutex::new(0usize));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(RendezvousTool {
            active,
            peak: peak.clone(),
            wait,
        }));
        let mut session = session();
        let cancel = CancelSignal::new();
        let mut schedule = schedule(&registry, &cancel);
        schedule.config = ToolScheduleConfig {
            max_parallel_tool_calls: max,
        };
        let outcome = execute_tool_calls(
            &mut session,
            &schedule,
            &[
                call("c1", "rendezvous", "{}"),
                call("c2", "rendezvous", "{}"),
            ],
            &mut |_| {},
        )
        .await
        .unwrap();
        assert_eq!(outcome.settled, 2);
        assert_eq!(committed_pairs(&session).len(), 2);
        let peak = *peak.lock().unwrap();
        peak
    }

    assert_eq!(
        run(2, Duration::from_secs(5)).await,
        2,
        "two slots run both bodies at the same time"
    );
    assert_eq!(
        run(1, Duration::from_millis(100)).await,
        1,
        "one slot serializes the bodies"
    );
}

/// Every dispatched body passes post-execute, and an accepted content
/// replacement becomes the durable result instead of the body's content.
#[tokio::test]
async fn post_execute_can_rewrite_a_dispatched_result_before_it_commits() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log,
    )));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let post = TestPost {
        decision: PostToolDecision::accept_content(vec![ContentBlock::text("rewritten")]),
        finalize: None,
        seen: seen.clone(),
    };
    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls_with_post_execute(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "echo", r#"{"text":"a"}"#)],
        &AllowAllGate,
        &post,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 1);
    assert_eq!(*seen.lock().unwrap(), vec!["c1"]);
    assert_eq!(result_texts(&session), vec!["rewritten"]);
    assert_eq!(result_is_error(&session), vec![false]);
}

/// A block decision turns a success into a valueless failure: the model sees
/// the corrective feedback, the result is marked as an error, and no
/// structured identity is invented.
#[tokio::test]
async fn post_execute_block_turns_a_success_into_a_valueless_failure() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log,
    )));
    let post = TestPost {
        decision: PostToolDecision::block(vec![ContentBlock::text("output rejected")]),
        finalize: None,
        seen: Arc::new(Mutex::new(Vec::new())),
    };
    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls_with_post_execute(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "echo", r#"{"text":"a"}"#)],
        &AllowAllGate,
        &post,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 1);
    assert_eq!(result_texts(&session), vec!["output rejected"]);
    assert_eq!(result_is_error(&session), vec![true]);
    assert_eq!(result_codes(&session), vec!["none"]);
}

/// Finalize runs after the decision applied and replaces the committed
/// content without changing the result's success or failure.
#[tokio::test]
async fn post_execute_finalize_replaces_the_content_after_the_decision() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log,
    )));
    let post = TestPost {
        decision: PostToolDecision::accept(),
        finalize: Some(vec![ContentBlock::text("final")]),
        seen: Arc::new(Mutex::new(Vec::new())),
    };
    let mut session = session();
    let cancel = CancelSignal::new();
    execute_tool_calls_with_post_execute(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "echo", r#"{"text":"a"}"#)],
        &AllowAllGate,
        &post,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(result_texts(&session), vec!["final"]);
    assert_eq!(result_is_error(&session), vec![false]);
}

/// A gate denial, a gate cancellation, and an unknown tool all settle through
/// post-execute, in model order, before they become durable.
#[tokio::test]
async fn post_execute_sees_gate_denials_cancellations_and_unknown_tools() {
    struct MixedGate;

    impl PreToolGate for MixedGate {
        fn decide(&self, call: &ToolCallInput) -> PreToolDecision {
            match call.name.as_str() {
                "deny" => PreToolDecision::deny("not permitted"),
                "cancel" => PreToolDecision::cancel(),
                _ => PreToolDecision::Allow,
            }
        }
    }

    let registry = ToolRegistry::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let post = TestPost {
        decision: PostToolDecision::accept_content(vec![ContentBlock::text("post-executed")]),
        finalize: None,
        seen: seen.clone(),
    };
    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls_with_post_execute(
        &mut session,
        &schedule(&registry, &cancel),
        &[
            call("c1", "deny", "{}"),
            call("c2", "cancel", "{}"),
            call("c3", "ghost", "{}"),
        ],
        &MixedGate,
        &post,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 3);
    assert_eq!(*seen.lock().unwrap(), vec!["c1", "c2", "c3"]);
    assert_eq!(result_texts(&session), vec!["post-executed"; 3]);
    assert_eq!(
        result_codes(&session),
        vec![
            "none".to_string(),
            TOOL_ABORTED_BEFORE_DISPATCH.to_string(),
            TOOL_UNKNOWN_TOOL.to_string()
        ]
    );
    assert_eq!(result_is_error(&session), vec![true, true, true]);
}

/// A call skipped by cancellation never reaches post-execute: the scheduler
/// records the canonical abort directly, as `appendSkippedToolCall` does.
#[tokio::test]
async fn post_execute_is_not_consulted_for_a_call_skipped_by_cancellation() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log,
    )));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let post = TestPost {
        decision: PostToolDecision::accept_content(vec![ContentBlock::text("post-executed")]),
        finalize: Some(vec![ContentBlock::text("finalized")]),
        seen: seen.clone(),
    };
    let mut session = session();
    let cancel = CancelSignal::new();
    cancel.cancel();
    let outcome = execute_tool_calls_with_post_execute(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "echo", "{}")],
        &AllowAllGate,
        &post,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert!(outcome.aborted);
    assert_eq!(outcome.settled, 1);
    assert!(seen.lock().unwrap().is_empty());
    assert_eq!(
        result_texts(&session),
        vec!["Error: tool call aborted before dispatch"]
    );
    assert_eq!(
        result_codes(&session),
        vec![TOOL_ABORTED_BEFORE_DISPATCH.to_string()]
    );
    assert_eq!(result_error_names(&session), vec!["AbortError"]);
}

/// Replacing the value of a failed result is the TypeScript `TypeError`: the
/// error text becomes the committed failure, and finalize still runs.
#[tokio::test]
async fn a_value_replacement_of_a_failure_commits_the_type_error_and_finalizes() {
    let registry = ToolRegistry::new();
    let post = TestPost {
        decision: PostToolDecision::accept_value(serde_json::json!("replacement")),
        finalize: Some(vec![ContentBlock::text("finalized")]),
        seen: Arc::new(Mutex::new(Vec::new())),
    };
    let mut session = session();
    let cancel = CancelSignal::new();
    execute_tool_calls_with_post_execute(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "ghost", "{}")],
        &AllowAllGate,
        &post,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(result_texts(&session), vec!["finalized"]);
    assert_eq!(result_is_error(&session), vec![true]);
}

/// Barriers change *when* bodies run, never the committed log: an exclusive
/// call interleaved with parallel ones produces the same durable pairs.
#[tokio::test]
async fn barriers_do_not_change_the_committed_log() {
    async fn run(barrier_mode: ToolExecutionMode) -> Session {
        let empty = || Arc::new(Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool::new(
            "parallel",
            ToolExecutionMode::Parallel,
            empty(),
        )));
        registry.register(Arc::new(EchoTool::new("barrier", barrier_mode, empty())));
        let mut session = session();
        let cancel = CancelSignal::new();
        execute_tool_calls(
            &mut session,
            &schedule(&registry, &cancel),
            &[
                call("c1", "parallel", "{}"),
                call("c2", "barrier", "{}"),
                call("c3", "parallel", "{}"),
            ],
            &mut |_| {},
        )
        .await
        .unwrap();
        session
    }

    let exclusive = run(ToolExecutionMode::Exclusive).await;
    let parallel = run(ToolExecutionMode::Parallel).await;
    assert_eq!(committed_pairs(&exclusive), committed_pairs(&parallel));
}

/// A parallel group is bounded by `max_parallel_tool_calls`; the bound never
/// drops a call, it only splits the work into more groups.
#[tokio::test]
async fn a_bounded_group_still_settles_every_call() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));

    let calls: Vec<ContentBlock> = (1..=5)
        .map(|n| call(&format!("c{n}"), "echo", "{}"))
        .collect();
    let mut session = session();
    let cancel = CancelSignal::new();
    let mut schedule = schedule(&registry, &cancel);
    schedule.config = ToolScheduleConfig {
        max_parallel_tool_calls: 2,
    };
    let outcome = execute_tool_calls(&mut session, &schedule, &calls, &mut |_| {})
        .await
        .unwrap();

    assert_eq!(outcome.settled, 5);
    assert_eq!(log.lock().unwrap().len(), 5);
    assert_eq!(committed_pairs(&session).len(), 5);
}

/// Cancellation raised before dispatch still records a durable pair for every
/// model call, so no call is left unanswered.
#[tokio::test]
async fn cancellation_before_dispatch_settles_every_call_with_a_synthetic_error() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));

    let cancel = CancelSignal::new();
    cancel.cancel();
    let mut session = session();
    let outcome = execute_tool_calls(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "echo", "{}"), call("c2", "echo", "{}")],
        &mut |_| {},
    )
    .await
    .unwrap();

    assert!(outcome.aborted);
    assert_eq!(outcome.settled, 2);
    assert!(
        log.lock().unwrap().is_empty(),
        "no body may run after cancellation"
    );
    assert_eq!(committed_pairs(&session).len(), 2);
    assert_eq!(
        result_codes(&session),
        vec![
            TOOL_ABORTED_BEFORE_DISPATCH.to_string(),
            TOOL_ABORTED_BEFORE_DISPATCH.to_string()
        ]
    );
}

/// Cancellation raised mid-group commits the started call, then records
/// synthetic results for the rest of the group.
#[tokio::test]
async fn cancellation_mid_group_drains_the_started_call_and_skips_the_rest() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CancelTool { log: log.clone() }));
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));

    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls(
        &mut session,
        &schedule(&registry, &cancel),
        &[
            call("c1", "cancel", "{}"),
            call("c2", "echo", "{}"),
            call("c3", "echo", "{}"),
        ],
        &mut |_| {},
    )
    .await
    .unwrap();

    assert!(outcome.aborted);
    assert_eq!(outcome.settled, 3);
    assert_eq!(
        *log.lock().unwrap(),
        vec!["c1"],
        "only the cancelling call runs a body"
    );
    assert_eq!(committed_pairs(&session).len(), 3);
    assert_eq!(
        result_codes(&session),
        vec![
            "none".to_string(),
            TOOL_ABORTED_BEFORE_DISPATCH.to_string(),
            TOOL_ABORTED_BEFORE_DISPATCH.to_string()
        ]
    );
}

/// An unknown tool is a failure the model sees, not a scheduler error.
///
/// The message and `info.name` are pinned to the TypeScript
/// `ToolNotFoundError` (`index.ts:511-526`): `unknown tool "<name>"` with
/// double quotes, and the class name TS routes the error under. Only the code
/// was asserted before, which let a wording drift go unnoticed.
#[tokio::test]
async fn an_unknown_tool_fails_its_call_instead_of_stopping_the_step() {
    let registry = ToolRegistry::new();
    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "nope", "{}"), call("c2", "nope", "{}")],
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 2);
    assert!(!outcome.aborted);
    assert_eq!(
        result_codes(&session),
        vec![TOOL_UNKNOWN_TOOL.to_string(), TOOL_UNKNOWN_TOOL.to_string()]
    );
    assert_eq!(
        result_texts(&session),
        vec![
            "Error: unknown tool \"nope\"".to_string(),
            "Error: unknown tool \"nope\"".to_string()
        ]
    );
    assert_eq!(
        result_error_names(&session),
        vec![
            "ToolNotFoundError".to_string(),
            "ToolNotFoundError".to_string()
        ]
    );
}

/// Only a success can conclude a turn, and its deferred contexts reach the
/// caller in call order.
#[tokio::test]
async fn a_concluding_result_stops_the_turn_and_defers_its_contexts() {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(ConcludeTool));
    let mut session = session();

    let mut contexts: Vec<UserMessage> = Vec::new();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "finish", "{}")],
        &mut |context: UserMessage| contexts.push(context),
    )
    .await
    .unwrap();

    assert!(outcome.concluded);
    assert_eq!(contexts.len(), 1);
    assert_eq!(contexts[0].content, vec![ContentBlock::text("wrapping up")]);
    assert!(!ToolExecutionResult::failure("boom")
        .concluding_turn()
        .concludes_turn());
}

#[test]
fn argument_parsing_preserves_invalid_json_as_text() {
    assert_eq!(parse_arguments(""), serde_json::json!({}));
    assert_eq!(parse_arguments(r#"{"a":1}"#), serde_json::json!({ "a": 1 }));
    assert_eq!(parse_arguments("{not json"), serde_json::json!("{not json"));
}

/// A malformed argument string still reaches the tool: the call settles as a
/// normal result carrying the raw text.
#[tokio::test]
async fn malformed_arguments_reach_the_tool_as_text() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));

    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "echo", "{oops")],
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 1);
    assert!(!outcome.aborted);
    assert_eq!(*log.lock().unwrap(), vec!["c1"]);
}

/// The registry keeps schemas deterministic so an assembled `request/header`
/// can be compared across replays.
#[test]
fn registry_schemas_are_ordered_by_tool_name() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "zeta",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));
    registry.register(Arc::new(EchoTool::new(
        "alpha",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));
    let names: Vec<String> = registry
        .schemas()
        .iter()
        .map(|schema| schema.name.clone())
        .collect();
    assert_eq!(names, vec!["alpha".to_string(), "zeta".to_string()]);
    assert_eq!(
        registry.execution_mode("alpha"),
        Some(ToolExecutionMode::Parallel)
    );
    assert_eq!(registry.execution_mode("missing"), None);
}

/// Non-tool-call blocks are ignored rather than scheduled.
#[tokio::test]
async fn blocks_that_are_not_tool_calls_are_ignored() {
    let registry = ToolRegistry::new();
    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls(
        &mut session,
        &schedule(&registry, &cancel),
        &[ContentBlock::text("not a call")],
        &mut |_| {},
    )
    .await
    .unwrap();
    assert_eq!(outcome.settled, 0);
    assert!(session.events().is_empty());
}
