//! Compaction wired into the driver: the step-boundary pressure trigger, the
//! context-overflow recovery path, and the failure semantics that keep a turn
//! alive.
//!
//! Every test pins the exact event-type sequence. Compaction is an ordering
//! contract — the bracket may not straddle step events, and the checkpoint
//! `user/message` must stay adjacent to `compaction/summary` — so "the turn
//! finished" is never the whole assertion.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use dsh_agent_loop::{AgentLoop, CompactionConfig, LoopConfig, TokenMeter};
use dsh_compaction::{
    fold_surface, resolve_config, BasicCompactionConfig, CompactionEvent, CompactionService,
    LlmSummarizationBackend, ModelTarget, ResolvedCompactSpec, ResolvedPruneConfig,
    SummarizationBackend, COMPACTION_SUMMARY_TAG, CONTEXT_WINDOW_EXCEEDED_CODE, PRUNE_MARKER,
};
use dsh_llm::{AssistantMessage, ContentBlock, LlmFailure, MessageId, ToolSchema, UserMessage};
use dsh_provider::{
    text_response, tool_call_response, user_message, FnProvider, LlmProvider, LlmRequest,
    ScriptedProvider,
};
use dsh_session::{RequestContext, Session, SessionEvent, SessionId, SessionMeta, TurnEndReason};
use dsh_tools::{CancelSignal, ToolCallInput, ToolExecutionResult, ToolHandler, ToolRegistry};

/// Text that only the pre-compaction history carries.
const OLD_SENTINEL: &str = "OLD HISTORY SENTINEL";
/// Text the fake summarizer writes.
const CHECKPOINT: &str = "compacted checkpoint text";
/// The oversized result `DumpTool` returns, in code points.
const DUMP_CHARS: usize = 20_000;

fn route() -> RequestContext {
    RequestContext {
        provider: "demo".to_string(),
        model: "demo-1".to_string(),
        context_window: Some(128_000),
        system_prompt_update: None,
    }
}

fn session() -> Session {
    Session::create(SessionId::new("t").unwrap(), SessionMeta::default(), 0)
}

fn registry(tools: Vec<Arc<dyn ToolHandler>>) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for tool in tools {
        registry.register(tool);
    }
    registry
}

/// Every event tag in the log, core and extension alike.
fn tags(session: &Session) -> Vec<String> {
    session
        .events()
        .iter()
        .map(|event| {
            serde_json::to_value(event)
                .expect("an event serializes")
                .get("type")
                .and_then(|tag| tag.as_str())
                .expect("every event carries a type tag")
                .to_string()
        })
        .collect()
}

/// How many events [`seeded_history`] contributes.
const SEED_EVENTS: usize = 4;

/// The tags the driven turn appended, with the seed's history skipped.
fn driven_tags(session: &Session) -> Vec<String> {
    tags(session).into_iter().skip(SEED_EVENTS).collect()
}

/// Two earlier turns' worth of conversation the driven turn inherits.
///
/// Message events only: the pressure trigger reads the surface and the derived
/// history, and the open turn it compacts inside is the one the driver starts.
fn seeded_history() -> Session {
    let mut session = session();
    for turn in 1..=2u64 {
        session
            .append(SessionEvent::user_message(UserMessage::new(
                MessageId::random(),
                vec![ContentBlock::text(format!(
                    "{OLD_SENTINEL} {turn} {}",
                    "old context ".repeat(40)
                ))],
            )))
            .unwrap();
        session
            .append(SessionEvent::assistant_message(
                turn,
                1,
                AssistantMessage::new(
                    MessageId::random(),
                    vec![ContentBlock::text(format!("historical answer {turn}"))],
                    "demo",
                    "demo-1",
                ),
                Vec::new(),
                None,
                false,
            ))
            .unwrap();
    }
    session
}

/// One scripted conversation step.
#[derive(Clone, Copy)]
enum Step {
    /// Answer with text and no tool calls.
    Text(&'static str),
    /// Ask `DumpTool` for its oversized result.
    CallDump(&'static str),
    /// Fail with the canonical context-window overflow.
    Overflow,
}

fn overflow() -> LlmFailure {
    LlmFailure {
        message: "request too large for model context".to_string(),
        code: CONTEXT_WINDOW_EXCEEDED_CODE.to_string(),
        status: None,
        provider_retry_after_ms: None,
        request_id: None,
        offload_images: None,
    }
}

/// Every text block of one request, joined, so a test can grep the history.
fn request_text(request: &LlmRequest) -> String {
    request
        .messages
        .iter()
        .flat_map(|message| message.content())
        .filter_map(ContentBlock::as_text)
        .collect::<Vec<_>>()
        .join("\n")
}

/// A conversation provider that records each request's text and replays `steps`.
///
/// The recorder is what lets a test assert which history the request actually
/// carried, which is the only observable difference compaction makes.
fn conversation(steps: Vec<Step>) -> (impl LlmProvider, Arc<Mutex<Vec<String>>>) {
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = FnProvider::new(move |request: &LlmRequest| {
        sink.lock().unwrap().push(request_text(request));
        let index = calls.fetch_add(1, Ordering::SeqCst);
        match steps.get(index).copied() {
            Some(Step::Text(text)) => Ok(text_response("demo", "demo-1", text)),
            Some(Step::CallDump(call_id)) => {
                Ok(tool_call_response("demo", "demo-1", call_id, "dump", "{}"))
            }
            Some(Step::Overflow) | None => Err(overflow()),
        }
    });
    (provider, seen)
}

/// A scripted stand-in for the summarization model, with `calls` queued answers.
fn summarizer(calls: usize) -> Arc<dyn SummarizationBackend> {
    let responses = (0..calls)
        .map(|_| text_response("demo", "demo-1", CHECKPOINT))
        .collect();
    Arc::new(LlmSummarizationBackend::new(Arc::new(
        ScriptedProvider::new(responses),
    )))
}

/// A summarization backend whose model call always fails.
fn broken_summarizer() -> Arc<dyn SummarizationBackend> {
    Arc::new(LlmSummarizationBackend::new(Arc::new(FnProvider::new(
        |_: &LlmRequest| {
            Err(LlmFailure {
                message: "summarizer is down".to_string(),
                code: "SUMMARY_DOWN".to_string(),
                status: None,
                provider_retry_after_ms: None,
                request_id: None,
                offload_images: None,
            })
        },
    ))))
}

fn engine(backend: Arc<dyn SummarizationBackend>) -> Arc<CompactionService> {
    let config = resolve_config(&BasicCompactionConfig::default()).expect("the default policy");
    Arc::new(CompactionService::new(backend, config, "demo", "demo-1"))
}

fn spec(
    threshold_tokens: u64,
    retain_tokens: u64,
    compaction_retries: u64,
    max_overflow_retries: u64,
) -> ResolvedCompactSpec {
    ResolvedCompactSpec {
        target: ModelTarget::new("demo", "demo-1"),
        context_window: 128_000,
        threshold_ratio: 0.8,
        threshold_tokens,
        retain_tokens,
        summarization_provider: "demo".to_string(),
        summarization_model: "demo-1".to_string(),
        max_tokens: 128,
        compaction_retries,
        max_overflow_retries,
    }
}

/// A host meter with the two readings the trigger compares.
///
/// The real host re-measures the replacement surface and reports the smaller
/// request once a checkpoint has landed; a constant would keep the loop
/// convinced that a successful compaction changed nothing.
fn host_meter(before: u64, after: u64) -> impl TokenMeter {
    move |events: &[SessionEvent]| {
        let compacted = events.iter().any(|event| {
            event.extension().map(|extension| extension.kind()) == Some(COMPACTION_SUMMARY_TAG)
        });
        if compacted {
            after
        } else {
            before
        }
    }
}

/// Wire the capability with a host measurement and a fixed clock.
fn compaction(
    backend: Arc<dyn SummarizationBackend>,
    spec: ResolvedCompactSpec,
    meter: impl TokenMeter + 'static,
) -> CompactionConfig {
    CompactionConfig::new(engine(backend), spec, meter).with_clock(Arc::new(|| 1_000_i64))
}

/// A tool whose result is far larger than the prune budget.
struct DumpTool;

impl ToolHandler for DumpTool {
    fn name(&self) -> &str {
        "dump"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new("dump", "Dump a large payload", serde_json::json!({}))
    }

    fn call(&self, _input: &ToolCallInput, _cancel: &CancelSignal) -> ToolExecutionResult {
        ToolExecutionResult::success(vec![ContentBlock::text("d".repeat(DUMP_CHARS))])
    }
}

/// One delivered text turn, driven with and without the compaction capability.
async fn run_text_turn(config: LoopConfig) -> (Vec<String>, Arc<Mutex<Vec<String>>>) {
    let mut session = session();
    let registry = registry(vec![]);
    let (provider, seen) = conversation(vec![Step::Text("all done")]);
    let mut driver =
        AgentLoop::new(&mut session, &provider, &registry, route()).with_config(config);
    driver.send(user_message("hi"));
    driver.run_turn().await.expect("the turn runs");
    (tags(&session), seen)
}

/// The log shape a turn has when nothing compacts.
fn plain_turn_tags() -> Vec<&'static str> {
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
}

#[tokio::test]
async fn one_token_below_the_threshold_leaves_the_log_unchanged() {
    let (plain, plain_requests) = run_text_turn(LoopConfig::default()).await;
    let (with_compaction, compaction_requests) =
        run_text_turn(LoopConfig::default().with_compaction(compaction(
            summarizer(1),
            spec(8_000, 0, 0, 0),
            // Exactly one token short of the routed threshold.
            host_meter(7_999, 7_999),
        )))
        .await;

    assert_eq!(plain, plain_turn_tags());
    assert_eq!(with_compaction, plain_turn_tags());
    // Installing the capability below the threshold changes nothing at all.
    assert_eq!(
        *plain_requests.lock().unwrap(),
        *compaction_requests.lock().unwrap()
    );
}

#[tokio::test]
async fn pressure_lands_the_full_chain_before_the_request() {
    let mut session = seeded_history();
    let registry = registry(vec![]);
    let (provider, seen) = conversation(vec![Step::Text("done")]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route()).with_config(
        LoopConfig::default().with_compaction(compaction(
            summarizer(1),
            spec(8_000, 0, 0, 0),
            // Pressured before the compaction, comfortable after it.
            host_meter(9_000, 500),
        )),
    );
    driver.send(user_message("continue"));

    let report = driver.run_turn().await.expect("the turn runs");

    assert_eq!(report.steps, 1);
    assert_eq!(report.reason, TurnEndReason::Completed);
    assert_eq!(
        driven_tags(&session),
        vec![
            "turn/start",
            "user/message",
            "step/start",
            "request/header",
            "request/context",
            "compaction/start",
            "compaction/summary",
            // The replacement checkpoint, contractually adjacent to the summary.
            "user/message",
            "compaction/end",
            "assistant/message",
            "step/end",
            "turn/end",
        ]
    );
    // The bracket is effective on replay, which is what "adjacent" buys.
    let surface = fold_surface(session.events()).expect("the surface folds");
    assert_eq!(surface.replacements().len(), 1);

    // The single request already carried the compacted history: compaction ran
    // before the request was derived, not after it was sent.
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].contains(CHECKPOINT));
    assert!(!requests[0].contains(OLD_SENTINEL));
    assert!(requests[0].contains("continue"));
}

#[tokio::test]
async fn a_summary_failure_keeps_the_full_history_and_records_the_error() {
    let mut session = seeded_history();
    let registry = registry(vec![]);
    let (provider, seen) = conversation(vec![Step::Text("done")]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route()).with_config(
        LoopConfig::default().with_compaction(compaction(
            broken_summarizer(),
            spec(8_000, 0, 0, 0),
            host_meter(9_000, 500),
        )),
    );
    driver.send(user_message("continue"));

    let report = driver.run_turn().await.expect("the turn still runs");

    assert_eq!(report.reason, TurnEndReason::Completed);
    assert_eq!(
        driven_tags(&session),
        vec![
            "turn/start",
            "user/message",
            "step/start",
            "request/header",
            "request/context",
            "compaction/start",
            // No summary: the attempt failed and the bracket records it.
            "compaction/end",
            "assistant/message",
            "step/end",
            "turn/end",
        ]
    );
    // The turn continued on the full history rather than failing.
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].contains(OLD_SENTINEL));

    let end = session
        .events()
        .iter()
        .find_map(|event| match CompactionEvent::decode(event) {
            Some(Ok(CompactionEvent::End(end))) => Some(end),
            _ => None,
        })
        .expect("the failed attempt closed its bracket");
    let error = end.error.expect("the failure is recorded");
    assert!(
        error.contains("summarization"),
        "unexpected diagnostic: {error}"
    );
}

#[tokio::test]
async fn no_compactable_range_writes_nothing() {
    // A fresh turn holds one user message, so there is no safe span to replace.
    let (with_compaction, seen) = run_text_turn(LoopConfig::default().with_compaction(compaction(
        summarizer(1),
        spec(8_000, 0, 0, 0),
        host_meter(9_000, 500),
    )))
    .await;

    assert_eq!(with_compaction, plain_turn_tags());
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].contains(CHECKPOINT));
}

#[tokio::test]
async fn an_overflow_forces_one_reduction_and_retries_the_step() {
    let mut session = seeded_history();
    let registry = registry(vec![]);
    let (provider, seen) = conversation(vec![Step::Overflow, Step::Text("recovered")]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route()).with_config(
        LoopConfig::default().with_compaction(compaction(
            summarizer(1),
            // The threshold is out of reach, so only the overflow path can fire.
            spec(1_000_000, 0, 0, 1),
            host_meter(10, 10),
        )),
    );
    driver.send(user_message("continue"));

    let report = driver.run_turn().await.expect("the turn runs");

    assert_eq!(report.steps, 1);
    assert_eq!(report.reason, TurnEndReason::Completed);
    assert_eq!(
        driven_tags(&session),
        vec![
            "turn/start",
            "user/message",
            "step/start",
            "request/header",
            "request/context",
            "compaction/start",
            "compaction/summary",
            "user/message",
            "compaction/end",
            "assistant/message",
            "step/end",
            "turn/end",
        ]
    );

    // The retried request was re-derived from the replacement surface.
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].contains(OLD_SENTINEL));
    assert!(requests[1].contains(CHECKPOINT));
    assert!(!requests[1].contains(OLD_SENTINEL));
}

#[tokio::test]
async fn overflow_recovery_stops_at_its_retry_budget() {
    let mut session = seeded_history();
    let registry = registry(vec![]);
    let (provider, seen) = conversation(vec![Step::Overflow, Step::Overflow, Step::Text("x")]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route()).with_config(
        LoopConfig::default().with_compaction(compaction(
            summarizer(1),
            spec(1_000_000, 0, 0, 1),
            host_meter(10, 10),
        )),
    );
    driver.send(user_message("continue"));

    let report = driver.run_turn().await.expect("the turn ends cleanly");

    // One recovery was spent, so the second overflow ends the turn with the
    // provider's own code rather than looping.
    match &report.reason {
        TurnEndReason::Error { error } => assert_eq!(error.code, CONTEXT_WINDOW_EXCEEDED_CODE),
        other => panic!("expected the overflow failure, got {other:?}"),
    }
    assert_eq!(
        driven_tags(&session),
        vec![
            "turn/start",
            "user/message",
            "step/start",
            "request/header",
            "request/context",
            "compaction/start",
            "compaction/summary",
            "user/message",
            "compaction/end",
            // No assistant message: the retried request failed too.
            "step/end",
            "turn/end",
        ]
    );
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 2);
}

#[tokio::test]
async fn a_prune_phase_lands_each_replacement_with_its_shadow_price() {
    let mut session = session();
    let registry = registry(vec![Arc::new(DumpTool)]);
    let (provider, seen) = conversation(vec![Step::CallDump("c1"), Step::Text("done")]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route()).with_config(
        LoopConfig::default().with_compaction(
            compaction(summarizer(1), spec(8_000, 0, 0, 0), host_meter(9_000, 500)).with_prune(
                ResolvedPruneConfig {
                    threshold_chars: 8_192,
                    head_chars: 4_096,
                    tail_chars: 1_024,
                },
            ),
        ),
    );
    driver.send(user_message("go"));

    let report = driver.run_turn().await.expect("the turn runs");

    assert_eq!(report.steps, 2);
    assert_eq!(report.reason, TurnEndReason::Completed);
    assert_eq!(
        tags(&session),
        vec![
            "turn/start",
            "user/message",
            "step/start",
            "request/header",
            "request/context",
            // Step 1 has nothing to shadow, so the first attempt writes nothing.
            "assistant/message",
            "tool/call",
            "tool/result",
            "step/end",
            "step/start",
            "compaction/start",
            // The model-free pass: shadow price, then its rewritten result.
            "compaction/prune",
            "tool/result",
            "compaction/summary",
            "user/message",
            "compaction/end",
            "assistant/message",
            "step/end",
            "turn/end",
        ]
    );

    // The rewritten result is the pruned one, and the pairs stayed adjacent.
    let results: Vec<_> = session
        .events()
        .iter()
        .filter_map(|event| event.core())
        .filter_map(|core| match core {
            dsh_session::CoreEvent::ToolResult { message, .. } => Some(message.content.clone()),
            _ => None,
        })
        .collect();
    let pruned = results.last().expect("a tool result followed the prune");
    let text = pruned
        .iter()
        .filter_map(ContentBlock::as_text)
        .collect::<String>();
    assert!(text.contains(PRUNE_MARKER));
    assert!(text.chars().count() < DUMP_CHARS);

    // The second request carried the pruned result and the checkpoint.
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].contains(PRUNE_MARKER.trim()));
    assert!(requests[1].contains(CHECKPOINT));
}

#[tokio::test]
async fn a_spent_attempt_budget_records_still_above_threshold() {
    let mut session = seeded_history();
    let registry = registry(vec![]);
    let (provider, _seen) = conversation(vec![Step::Text("done")]);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route()).with_config(
        LoopConfig::default().with_compaction(compaction(
            // Two answers: both attempts summarize successfully.
            summarizer(2),
            spec(8_000, 0, 1, 0),
            // The meter never improves, so neither attempt can clear pressure.
            host_meter(9_000, 9_000),
        )),
    );
    driver.send(user_message("continue"));

    let report = driver.run_turn().await.expect("the turn still completes");

    assert_eq!(report.reason, TurnEndReason::Completed);
    let starts = tags(&session)
        .iter()
        .filter(|tag| *tag == "compaction/start")
        .count();
    // Two successful attempts (the first, then the checkpoint re-compacted
    // alone) plus the failure bracket the spent budget records.
    assert_eq!(starts, 3);

    let ends: Vec<_> = session
        .events()
        .iter()
        .filter_map(|event| match CompactionEvent::decode(event) {
            Some(Ok(CompactionEvent::End(end))) => Some(end),
            _ => None,
        })
        .collect();
    assert_eq!(ends.len(), 3);
    assert!(ends[0].error.is_none());
    assert!(ends[1].error.is_none());
    let error = ends[2]
        .error
        .as_deref()
        .expect("the spent budget is recorded");
    assert!(
        error.contains("still above threshold"),
        "unexpected diagnostic: {error}"
    );
}
