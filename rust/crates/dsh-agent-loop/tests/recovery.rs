//! Recovery: a failed model request is offered to the retry policy before the
//! turn is ended.
//!
//! Ported from `packages/llm/llm-retry/tests/retry.spec.ts`. The TypeScript
//! suite drives a whole Cordis context with fake timers; here the decision
//! lives in `dsh_retry::plan_retry` and only the wiring around it — log order,
//! event identity, attempt bounds — is under test.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use dsh_agent_loop::{AgentLoop, LoopConfig};
use dsh_llm::{LlmFailure, Message, MessageId};
use dsh_provider::{text_response, user_message, FnProvider};
use dsh_retry::{
    constant_random, decode_retry, fold_retry_state, resolve_retry_policy, AlwaysRetryPolicyConfig,
    BackoffConfig, CancelledDelay, DownstreamError, DownstreamRecovery, NoDelay,
    NormalRetryPolicyConfig, RequestErrorAction, ResolvedRetryPolicy, RetryConfig, RetryDelay,
    RetryLifecycle, RetryPolicyConfig, TokioDelay, WaitCancel,
};
use dsh_session::{CoreEvent, RequestContext, Session, SessionEvent, SessionId, SessionMeta};
use tokio::sync::watch;

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

fn registry() -> dsh_tools::ToolRegistry {
    dsh_tools::ToolRegistry::new()
}

/// Every event tag in the log, core and extension alike.
fn tags(session: &Session) -> Vec<String> {
    session
        .events()
        .iter()
        .map(|event| match event.core() {
            Some(CoreEvent::TurnStart { .. }) => "turn/start",
            Some(CoreEvent::TurnEnd { .. }) => "turn/end",
            Some(CoreEvent::StepStart { .. }) => "step/start",
            Some(CoreEvent::StepEnd { .. }) => "step/end",
            Some(CoreEvent::UserMessage { .. }) => "user/message",
            Some(CoreEvent::AssistantMessage { .. }) => "assistant/message",
            Some(CoreEvent::ToolCall { .. }) => "tool/call",
            Some(CoreEvent::ToolResult { .. }) => "tool/result",
            Some(CoreEvent::RequestHeader { .. }) => "request/header",
            Some(CoreEvent::RequestContext { .. }) => "request/context",
            _ => event.extension().map_or("", |extension| extension.kind()),
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

fn normal_policy(max_retries: u64) -> ResolvedRetryPolicy {
    resolve_retry_policy(
        Some(&RetryPolicyConfig::Normal(NormalRetryPolicyConfig {
            max_retries: Some(max_retries),
            retryable_codes: Some(vec!["SERVER".to_string(), "RATE_LIMIT".to_string()]),
            backoff: Some(BackoffConfig {
                initial_delay_ms: Some(500.0),
                max_delay_ms: Some(10_000.0),
                jitter_ratio: Some(0.0),
            }),
        })),
        "test",
    )
    .unwrap()
}

/// A retry policy whose first wait is long enough for a canceller to win.
fn slow_policy() -> ResolvedRetryPolicy {
    resolve_retry_policy(
        Some(&RetryPolicyConfig::Normal(NormalRetryPolicyConfig {
            max_retries: Some(2),
            retryable_codes: Some(vec!["SERVER".to_string()]),
            backoff: Some(BackoffConfig {
                initial_delay_ms: Some(2_000.0),
                max_delay_ms: Some(2_000.0),
                jitter_ratio: Some(0.0),
            }),
        })),
        "test",
    )
    .unwrap()
}

fn always_policy() -> ResolvedRetryPolicy {
    resolve_retry_policy(
        Some(&RetryPolicyConfig::Always(AlwaysRetryPolicyConfig {
            backoff: Some(BackoffConfig {
                initial_delay_ms: Some(1.0),
                max_delay_ms: Some(1.0),
                jitter_ratio: Some(0.0),
            }),
            max_retries: None,
            retryable_codes: None,
        })),
        "test",
    )
    .unwrap()
}

/// Retry configured for the `demo` route, with a wait that costs nothing.
fn retry_for(policy: ResolvedRetryPolicy) -> RetryConfig {
    let mut policies = BTreeMap::new();
    policies.insert("demo".to_string(), policy);
    RetryConfig::new(policies)
        .with_delay(Arc::new(NoDelay))
        .with_random(constant_random(0.5))
}

/// Retry configured for a route other than the one the loop drives.
fn retry_for_other_provider() -> RetryConfig {
    let mut policies = BTreeMap::new();
    policies.insert("other".to_string(), always_policy());
    RetryConfig::new(policies)
        .with_delay(Arc::new(NoDelay))
        .with_random(constant_random(0.5))
}

/// How many events carry `tag`.
fn count(session: &Session, tag: &str) -> usize {
    tags(session).iter().filter(|t| *t == tag).count()
}

fn assert_aborted(reason: &dsh_session::TurnEndReason) {
    assert_eq!(
        reason,
        &dsh_session::TurnEndReason::Aborted {
            reason: dsh_session::TurnEndCancelCause::User
        }
    );
}

// ------------------------------------------------ downstream fixtures -----

/// What a scripted downstream recovery answers.
#[derive(Clone, Copy)]
enum Script {
    /// Answer `retry`.
    Retry,
    /// Answer with no opinion.
    NoOpinion,
    /// Fail, which the negotiation must ignore.
    Fail,
}

/// A scripted downstream recovery that counts its consultations.
struct ScriptedDownstream {
    script: Script,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl DownstreamRecovery for ScriptedDownstream {
    async fn recover(&self) -> Result<Option<RequestErrorAction>, DownstreamError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.script {
            Script::Retry => Ok(Some(RequestErrorAction::Retry)),
            Script::NoOpinion => Ok(None),
            Script::Fail => Err(DownstreamError("downstream recovery failed".into())),
        }
    }
}

fn scripted_downstream(script: Script) -> (ScriptedDownstream, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    (
        ScriptedDownstream {
            script,
            calls: Arc::clone(&calls),
        },
        calls,
    )
}

/// A downstream recovery that raises the plugin lifetime before answering
/// `retry`: the abort must win over the settled decision.
struct AbortingDownstream {
    lifecycle: RetryLifecycle,
}

#[async_trait]
impl DownstreamRecovery for AbortingDownstream {
    async fn recover(&self) -> Result<Option<RequestErrorAction>, DownstreamError> {
        self.lifecycle.lifetime().cancel();
        Ok(Some(RequestErrorAction::Retry))
    }
}

/// A wait that announces it was entered and then holds until its gate opens.
struct GatedDelay {
    entered: watch::Sender<bool>,
    gate: watch::Receiver<bool>,
}

#[async_trait]
impl RetryDelay for GatedDelay {
    async fn wait(&self, _delay_ms: f64) -> bool {
        let _ = self.entered.send(true);
        self.gate.clone().wait_for(|open| *open).await.is_ok()
    }
}

/// A provider that fails `failures` times, then answers with `text`.
fn flaky_provider(
    failures: usize,
    code: &'static str,
    text: &'static str,
) -> FnProvider<
    impl Fn(&dsh_provider::LlmRequest) -> Result<dsh_provider::LlmResponse, LlmFailure> + Send + Sync,
> {
    let seen = Arc::new(Mutex::new(0usize));
    FnProvider::new(move |_| {
        let attempt = {
            let mut seen = seen.lock().unwrap();
            *seen += 1;
            *seen
        };
        if attempt <= failures {
            Err(failure(code))
        } else {
            Ok(text_response("demo", "demo-1", text))
        }
    })
}

#[tokio::test]
async fn a_failed_request_recovers_and_the_turn_completes() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "RATE_LIMIT", "recovered");
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry_for(normal_policy(2))));
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    assert_eq!(report.reason, dsh_session::TurnEndReason::Completed);
    assert_eq!(report.steps, 1);
    // The retry happened inside the step, so the step is not re-opened.
    assert_eq!(
        tags(&session),
        vec![
            "turn/start",
            "user/message",
            "step/start",
            "request/header",
            "request/context",
            "llm/retry",
            "llm/retry-started",
            "assistant/message",
            "step/end",
            "turn/end",
        ]
    );
    // Exactly one committed message: the failed attempt left none behind.
    assert_eq!(
        tags(&session)
            .iter()
            .filter(|t| *t == "assistant/message")
            .count(),
        1
    );
}

#[tokio::test]
async fn the_scheduled_event_is_durable_before_the_wait() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "recovered");
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry_for(normal_policy(2))));
    driver.send(user_message("go"));

    driver.run_turn().await.unwrap();

    let retries: Vec<dsh_retry::LlmRetryEvent> = session
        .events()
        .iter()
        .filter_map(decode_retry)
        .map(|decoded| decoded.unwrap())
        .collect();
    assert_eq!(retries.len(), 1);
    assert_eq!(retries[0].retry(), 1);
    assert_eq!(retries[0].provider(), "demo");
    assert_eq!(retries[0].delay_ms(), 500.0);
    assert_eq!(retries[0].max_retries(), Some(2));
    assert_eq!(retries[0].failure().code, "SERVER");
    // The chain identity survived into the state the next decision reads.
    // Folded over the prefix that ends at the retry event — exactly what the
    // log held when the wait was scheduled, before the turn ended and reset
    // the step scope.
    let upto = session
        .events()
        .iter()
        .position(|event| {
            event
                .extension()
                .is_some_and(|extension| extension.kind() == "llm/retry")
        })
        .expect("the retry event is in the log");
    let state = fold_retry_state(&session.events()[..=upto]).unwrap();
    assert_eq!(state.len(), 1);
    let (_, entry) = state.iter().next().unwrap();
    assert_eq!(entry.retry, 1);
    assert!(!entry.retry_id.as_str().is_empty());
}

/// The retried request must see exactly the context the failed one saw —
/// no failure text, no partial output.
#[tokio::test]
async fn a_retried_request_sees_the_same_context() {
    let mut session = session();
    let registry = registry();
    let seen: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let attempts = Arc::new(Mutex::new(0usize));
    let counter = attempts.clone();
    let provider = FnProvider::new(move |request| {
        sink.lock().unwrap().push(
            request
                .messages
                .iter()
                .map(Message::role)
                .map(str::to_string)
                .collect(),
        );
        let attempt = {
            let mut counter = counter.lock().unwrap();
            *counter += 1;
            *counter
        };
        if attempt == 1 {
            Err(LlmFailure {
                message: "private provider diagnostic must not enter context".to_string(),
                ..failure("SERVER")
            })
        } else {
            Ok(text_response("demo", "demo-1", "recovered"))
        }
    });

    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry_for(normal_policy(2))));
    driver.send(user_message("go"));
    driver.run_turn().await.unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[0], seen[1],
        "the retried request must see the same context"
    );
    let serialized = serde_json::to_string(&seen[1]).unwrap();
    assert!(!serialized.contains("private provider diagnostic"));
}

#[tokio::test]
async fn a_non_retryable_code_ends_the_turn_without_retrying() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "AUTH", "must not run");
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry_for(normal_policy(2))));

    let report = driver.run_turn().await.unwrap();

    match &report.reason {
        dsh_session::TurnEndReason::Error { error } => assert_eq!(error.code, "AUTH"),
        other => panic!("expected the original failure, got {other:?}"),
    }
    assert_eq!(
        tags(&session).iter().filter(|t| *t == "llm/retry").count(),
        0
    );
}

#[tokio::test]
async fn a_spent_budget_ends_the_turn_with_the_last_failure() {
    let mut session = session();
    let registry = registry();
    // Two retries allowed, three failures: the third attempt is the last.
    let provider = flaky_provider(3, "SERVER", "must not run");
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry_for(normal_policy(2))));

    let report = driver.run_turn().await.unwrap();

    assert_eq!(
        tags(&session).iter().filter(|t| *t == "llm/retry").count(),
        2
    );
    assert_eq!(
        tags(&session)
            .iter()
            .filter(|t| *t == "llm/retry-started")
            .count(),
        2
    );
    match &report.reason {
        dsh_session::TurnEndReason::Error { error } => assert_eq!(error.code, "SERVER"),
        other => panic!("expected the last failure, got {other:?}"),
    }
    // The retries did not open extra steps.
    assert_eq!(report.steps, 1);
}

#[tokio::test]
async fn an_always_policy_stops_at_the_drivers_attempt_guard() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(50, "AUTH", "must not run");
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route()).with_config(
        LoopConfig::default().with_retry(retry_for(always_policy()).with_max_attempts(4)),
    );

    let report = driver.run_turn().await.unwrap();

    // Always mode has no budget, so the driver's own ceiling is what stops it.
    assert_eq!(
        tags(&session).iter().filter(|t| *t == "llm/retry").count(),
        3
    );
    match &report.reason {
        dsh_session::TurnEndReason::Error { error } => assert_eq!(error.code, "ATTEMPT_LIMIT"),
        other => panic!("expected the attempt guard, got {other:?}"),
    }
}

#[tokio::test]
async fn a_cancelled_wait_ends_the_turn_as_aborted() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "must not run");
    let mut policies = BTreeMap::new();
    policies.insert("demo".to_string(), normal_policy(2));
    let retry = RetryConfig::new(policies)
        .with_delay(Arc::new(CancelledDelay))
        .with_random(constant_random(0.5));
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry));

    let report = driver.run_turn().await.unwrap();

    assert_eq!(
        report.reason,
        dsh_session::TurnEndReason::Aborted {
            reason: dsh_session::TurnEndCancelCause::User
        }
    );
    // The retry was scheduled but never started.
    assert_eq!(
        tags(&session).iter().filter(|t| *t == "llm/retry").count(),
        1
    );
    assert_eq!(
        tags(&session)
            .iter()
            .filter(|t| *t == "llm/retry-started")
            .count(),
        0
    );
    assert_eq!(
        tags(&session)
            .iter()
            .filter(|t| *t == "assistant/message")
            .count(),
        0
    );
}

/// A cancellation from outside the driver ends the retry wait, even though the
/// delay itself would eventually complete: the wait is raced against the
/// configured source, and no retried request starts.
#[tokio::test]
async fn an_external_cancellation_ends_the_retry_wait() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "must not run");
    let cancel = WaitCancel::new();
    let retry = retry_for(slow_policy())
        .with_delay(Arc::new(TokioDelay))
        .with_cancel(cancel.clone());
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry));
    driver.send(user_message("go"));

    let trigger = cancel.clone();
    let canceller = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        trigger.cancel();
    });
    let report = driver.run_turn().await.unwrap();
    canceller.await.unwrap();

    assert_eq!(
        report.reason,
        dsh_session::TurnEndReason::Aborted {
            reason: dsh_session::TurnEndCancelCause::User
        }
    );
    // The retry was scheduled but never started.
    assert_eq!(
        tags(&session).iter().filter(|t| *t == "llm/retry").count(),
        1
    );
    assert_eq!(
        tags(&session)
            .iter()
            .filter(|t| *t == "llm/retry-started")
            .count(),
        0
    );
    assert_eq!(
        tags(&session)
            .iter()
            .filter(|t| *t == "assistant/message")
            .count(),
        0
    );
}

#[tokio::test]
async fn recovery_is_scoped_to_the_failing_provider() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "must not run");
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry_for_other_provider()));

    let report = driver.run_turn().await.unwrap();

    match &report.reason {
        dsh_session::TurnEndReason::Error { error } => assert_eq!(error.code, "SERVER"),
        other => panic!("expected the original failure, got {other:?}"),
    }
    assert_eq!(
        tags(&session).iter().filter(|t| *t == "llm/retry").count(),
        0
    );
}

#[tokio::test]
async fn retry_state_is_scoped_to_the_step() {
    let mut session = session();
    let registry = registry();
    // The first request of *each* turn fails, so both turns retry once.
    let seen_lengths = Arc::new(Mutex::new(std::collections::HashSet::new()));
    let lengths = seen_lengths.clone();
    let provider = FnProvider::new(move |request: &dsh_provider::LlmRequest| {
        let context = request.messages.len();
        let first = lengths.lock().unwrap().insert(context);
        if first {
            Err(failure("SERVER"))
        } else {
            Ok(text_response("demo", "demo-1", "recovered"))
        }
    });
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry_for(normal_policy(2))));

    driver.send(user_message("first"));
    driver.run_turn().await.unwrap();
    driver.send(user_message("second"));
    driver.run_turn().await.unwrap();

    // Both turns recovered on their first retry: the second turn did not
    // inherit the first turn's spent budget.
    let retries: Vec<u64> = session
        .events()
        .iter()
        .filter_map(decode_retry)
        .map(|decoded| decoded.unwrap().retry())
        .collect();
    assert_eq!(retries, vec![1, 1]);
}

/// A reloaded log must reproduce the retry state, because the state is
/// derived rather than stored.
#[tokio::test]
async fn a_reloaded_log_reproduces_the_retry_state() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(2, "SERVER", "recovered");
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry_for(normal_policy(3))));
    driver.send(user_message("go"));
    driver.run_turn().await.unwrap();

    let before = fold_retry_state(session.events()).unwrap();
    let directory = session_dir();
    let published = dsh_session::write_session(&directory, &session).unwrap();
    // `read_session` takes the directory and picks the highest generation.
    let reloaded = dsh_session::read_session(&directory).unwrap();

    assert_eq!(fold_retry_state(reloaded.events()).unwrap(), before);
    assert_eq!(reloaded.events().len(), session.events().len());
    assert!(published.starts_with(&directory));
}

fn session_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("dsh-rust-recovery-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A committed message identity is irrelevant here; this guards that a
/// retried step still commits exactly one assistant message with a fresh id.
#[tokio::test]
async fn each_attempt_commits_at_most_one_message() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(2, "SERVER", "recovered");
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry_for(normal_policy(3))));
    driver.send(user_message("go"));

    driver.run_turn().await.unwrap();

    let ids: Vec<MessageId> = session
        .events()
        .iter()
        .filter_map(|event| match event.core() {
            Some(CoreEvent::AssistantMessage { message, .. }) => Some(message.id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ids.len(), 1);
}

#[tokio::test]
async fn a_retry_event_round_trips_through_the_log() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "recovered");
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry_for(normal_policy(2))));
    driver.send(user_message("go"));
    driver.run_turn().await.unwrap();

    let raw = serde_json::to_string(
        &session
            .events()
            .iter()
            .find(|event| {
                event
                    .extension()
                    .is_some_and(|extension| extension.kind() == "llm/retry")
            })
            .unwrap(),
    )
    .unwrap();
    // The event is stored verbatim, tag included, and stays decodable.
    assert!(raw.contains("\"llm/retry\""));
    let decoded: SessionEvent = serde_json::from_str(&raw).unwrap();
    assert_eq!(decoded.extension().unwrap().kind(), "llm/retry");
}

// -------------------------------------------- always-mode negotiation ----

/// An always policy adopts a downstream `retry` verbatim: the same step
/// tries again with no `llm/retry` event and no local wait.
#[tokio::test]
async fn an_always_policy_adopts_a_downstream_retry_without_scheduling() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "recovered");
    let (downstream, calls) = scripted_downstream(Script::Retry);
    let retry = retry_for(always_policy()).with_downstream(Arc::new(downstream));
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry));
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    assert_eq!(report.reason, dsh_session::TurnEndReason::Completed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(count(&session, "llm/retry"), 0);
    assert_eq!(count(&session, "llm/retry-started"), 0);
    assert_eq!(count(&session, "assistant/message"), 1);
}

/// A downstream recovery with no opinion falls through to the local
/// schedule: the whole durable chain is written.
#[tokio::test]
async fn an_always_policy_with_no_downstream_opinion_schedules_locally() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "recovered");
    let (downstream, calls) = scripted_downstream(Script::NoOpinion);
    let retry = retry_for(always_policy()).with_downstream(Arc::new(downstream));
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry));
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    assert_eq!(report.reason, dsh_session::TurnEndReason::Completed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(count(&session, "llm/retry"), 1);
    assert_eq!(count(&session, "llm/retry-started"), 1);
}

/// A downstream failure is logged and ignored — never propagated — and the
/// local schedule takes over with the warning text as its diagnostic.
#[tokio::test]
async fn an_ignored_downstream_failure_falls_through_with_a_diagnostic() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "recovered");
    let (downstream, calls) = scripted_downstream(Script::Fail);
    let retry = retry_for(always_policy()).with_downstream(Arc::new(downstream));
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry));
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    // The warning text the driver logs is the TypeScript prefix.
    assert_eq!(
        dsh_retry::ignored_downstream_warning("demo"),
        "llm-retry: provider \"demo\" always policy ignored a downstream recovery failure"
    );
    assert_eq!(report.reason, dsh_session::TurnEndReason::Completed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(count(&session, "llm/retry"), 1);
    assert_eq!(count(&session, "llm/retry-started"), 1);
}

/// A normal policy never negotiates: the downstream is not consulted even
/// when it would answer `retry`.
#[tokio::test]
async fn a_normal_policy_never_negotiates_with_the_downstream() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "recovered");
    let (downstream, calls) = scripted_downstream(Script::Retry);
    let retry = retry_for(normal_policy(2)).with_downstream(Arc::new(downstream));
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry));
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    assert_eq!(report.reason, dsh_session::TurnEndReason::Completed);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(count(&session, "llm/retry"), 1);
    assert_eq!(count(&session, "llm/retry-started"), 1);
}

/// An abort that wins before the downstream is consulted ends the turn as a
/// user abort, and the downstream is never called.
#[tokio::test]
async fn an_abort_before_the_negotiation_ends_the_turn_without_consulting() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "must not run");
    let (downstream, calls) = scripted_downstream(Script::Retry);
    let cancel = WaitCancel::new();
    let retry = retry_for(always_policy())
        .with_downstream(Arc::new(downstream))
        .with_cancel(cancel.clone());
    cancel.cancel();
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry));
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    assert_aborted(&report.reason);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(count(&session, "llm/retry"), 0);
}

/// A downstream that answers `retry` after raising the plugin lifetime
/// loses to the abort: the post-settle check wins over the settled
/// decision.
#[tokio::test]
async fn an_abort_during_the_negotiation_beats_the_downstream_decision() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "must not run");
    let lifecycle = RetryLifecycle::new();
    let downstream = AbortingDownstream {
        lifecycle: lifecycle.clone(),
    };
    let retry = retry_for(always_policy())
        .with_downstream(Arc::new(downstream))
        .with_lifecycle(lifecycle);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry));
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    assert_aborted(&report.reason);
    assert_eq!(count(&session, "llm/retry"), 0);
    assert_eq!(count(&session, "llm/retry-started"), 0);
}

/// A cancelled plugin lifetime ends the always negotiation before the
/// downstream is consulted, even without a disposal.
#[tokio::test]
async fn a_cancelled_lifetime_ends_the_negotiation_before_the_downstream() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "must not run");
    let lifecycle = RetryLifecycle::new();
    lifecycle.lifetime().cancel();
    let (downstream, calls) = scripted_downstream(Script::Retry);
    let retry = retry_for(always_policy())
        .with_downstream(Arc::new(downstream))
        .with_lifecycle(lifecycle);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry));
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    assert_aborted(&report.reason);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(count(&session, "llm/retry"), 0);
}

/// Without a downstream wired, an always policy still schedules locally:
/// the negotiation degenerates to its abort checks and a fall-through.
#[tokio::test]
async fn an_always_policy_without_a_downstream_schedules_locally() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "recovered");
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry_for(always_policy())));
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    assert_eq!(report.reason, dsh_session::TurnEndReason::Completed);
    assert_eq!(count(&session, "llm/retry"), 1);
    assert_eq!(count(&session, "llm/retry-started"), 1);
}

/// A cancelled turn ends an always recovery that has no downstream and
/// schedules nothing.
#[tokio::test]
async fn a_cancelled_turn_ends_an_always_recovery_without_a_downstream() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "must not run");
    let cancel = WaitCancel::new();
    let retry = retry_for(always_policy()).with_cancel(cancel.clone());
    cancel.cancel();
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry));
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    assert_aborted(&report.reason);
    assert_eq!(count(&session, "llm/retry"), 0);
}

// ------------------------------------------------------ the lifecycle ----

/// A recovery captured before disposal is rejected without entering any
/// recovery path — the TypeScript stale callback.
#[tokio::test]
async fn a_disposed_lifecycle_rejects_new_recoveries() {
    let mut session = session();
    let registry = registry();
    let provider = flaky_provider(1, "SERVER", "must not run");
    let lifecycle = RetryLifecycle::new();
    lifecycle.dispose();
    let retry = retry_for(normal_policy(2)).with_lifecycle(lifecycle);
    let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
        .with_config(LoopConfig::default().with_retry(retry));
    driver.send(user_message("go"));

    let report = driver.run_turn().await.unwrap();

    match &report.reason {
        dsh_session::TurnEndReason::Error { error } => assert_eq!(error.code, "SERVER"),
        other => panic!("expected the original failure, got {other:?}"),
    }
    assert_eq!(count(&session, "llm/retry"), 0);
    assert_eq!(count(&session, "llm/retry-started"), 0);
}

/// Disposal aborts the in-flight wait, and the drain settles only once the
/// abandoned recovery has settled: the chain is cut with no
/// `llm/retry-started`.
#[tokio::test]
async fn a_disposed_lifecycle_drains_the_in_flight_recovery() {
    let lifecycle = RetryLifecycle::new();
    let (entered_tx, mut entered_rx) = watch::channel(false);
    let (_gate_tx, gate_rx) = watch::channel(false);
    let mut policies = BTreeMap::new();
    policies.insert("demo".to_string(), slow_policy());
    let retry = RetryConfig::new(policies)
        .with_delay(Arc::new(GatedDelay {
            entered: entered_tx,
            gate: gate_rx,
        }))
        .with_random(constant_random(0.5))
        .with_lifecycle(lifecycle.clone());

    let task = tokio::spawn(async move {
        let mut session = session();
        let registry = registry();
        let provider = flaky_provider(1, "SERVER", "must not run");
        let mut driver = AgentLoop::new(&mut session, &provider, &registry, route())
            .with_config(LoopConfig::default().with_retry(retry));
        driver.send(user_message("go"));
        let report = driver.run_turn().await.unwrap();
        (report, tags(&session))
    });

    // The recovery registered, scheduled its retry, and sits in the wait.
    entered_rx.wait_for(|entered| *entered).await.unwrap();
    let drain = lifecycle.dispose();
    // The drain settles only once the abandoned wait ended the recovery.
    drain.await;

    let (report, log_tags) = task.await.unwrap();
    assert_aborted(&report.reason);
    assert_eq!(log_tags.iter().filter(|t| *t == "llm/retry").count(), 1);
    assert_eq!(
        log_tags
            .iter()
            .filter(|t| *t == "llm/retry-started")
            .count(),
        0
    );
}
