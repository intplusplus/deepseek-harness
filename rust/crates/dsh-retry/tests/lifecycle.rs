//! The recovery lifecycle: downstream negotiation in always mode and the
//! unload drain.
//!
//! Ported from the lifecycle behaviours in
//! `packages/llm/llm-retry/tests/retry.spec.ts` that the pure decision cannot
//! express: `aborts and drains a captured backoff before plugin disposal
//! completes`, `drains delegated recovery before completing plugin disposal`,
//! `fails a captured callback after disposal without entering downstream
//! policy`, and the always-mode downstream composition tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Notify;

use dsh_retry::{
    decode_retry, encode_retry, encode_retry_started, fused_wait, ignored_downstream_warning,
    negotiate_always, plan_retry, resolve_retry_policy, AlwaysNegotiation, DownstreamError,
    DownstreamRecovery, RequestErrorAction, ResolvedRetryPolicy, RetryLifecycle, RetryMode,
    RetryState, TokioDelay, WaitCancel,
};

// ---------------------------------------------------------- fixtures ------

fn always_policy(initial: f64, max: f64) -> ResolvedRetryPolicy {
    resolve_retry_policy(
        Some(&dsh_retry::RetryPolicyConfig::Always(
            dsh_retry::AlwaysRetryPolicyConfig {
                backoff: Some(dsh_retry::BackoffConfig {
                    initial_delay_ms: Some(initial),
                    max_delay_ms: Some(max),
                    jitter_ratio: Some(0.0),
                }),
                max_retries: None,
                retryable_codes: None,
            },
        )),
        "test",
    )
    .unwrap()
}

fn normal_policy() -> ResolvedRetryPolicy {
    resolve_retry_policy(
        Some(&dsh_retry::RetryPolicyConfig::Normal(
            dsh_retry::NormalRetryPolicyConfig {
                max_retries: Some(2),
                retryable_codes: Some(vec!["SERVER".to_string()]),
                backoff: Some(dsh_retry::BackoffConfig {
                    initial_delay_ms: Some(500.0),
                    max_delay_ms: Some(10_000.0),
                    jitter_ratio: Some(0.0),
                }),
            },
        )),
        "test",
    )
    .unwrap()
}

fn failure(code: &str) -> dsh_llm::LlmFailure {
    dsh_llm::LlmFailure {
        message: "busy".to_string(),
        code: code.to_string(),
        status: None,
        provider_retry_after_ms: None,
        request_id: None,
        offload_images: None,
    }
}

/// A downstream recovery that answers `retry`.
struct AllowRetry;

#[async_trait]
impl DownstreamRecovery for AllowRetry {
    async fn recover(&self) -> Result<Option<RequestErrorAction>, DownstreamError> {
        Ok(Some(RequestErrorAction::Retry))
    }
}

/// A downstream recovery with no opinion.
struct NoOpinion;

#[async_trait]
impl DownstreamRecovery for NoOpinion {
    async fn recover(&self) -> Result<Option<RequestErrorAction>, DownstreamError> {
        Ok(None)
    }
}

/// A downstream recovery that fails.
struct Fails;

#[async_trait]
impl DownstreamRecovery for Fails {
    async fn recover(&self) -> Result<Option<RequestErrorAction>, DownstreamError> {
        Err(DownstreamError("downstream recovery failed".into()))
    }
}

/// A downstream recovery that counts its consultations.
struct Counting(AtomicUsize);

#[async_trait]
impl DownstreamRecovery for Counting {
    async fn recover(&self) -> Result<Option<RequestErrorAction>, DownstreamError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Some(RequestErrorAction::Retry))
    }
}

/// A downstream recovery that settles only once its gate is opened.
struct SettleOnNotify {
    gate: Arc<Notify>,
}

#[async_trait]
impl DownstreamRecovery for SettleOnNotify {
    async fn recover(&self) -> Result<Option<RequestErrorAction>, DownstreamError> {
        self.gate.notified().await;
        Ok(Some(RequestErrorAction::Retry))
    }
}

// ------------------------------------------------------- unload drain -----

#[tokio::test]
async fn a_new_recovery_is_rejected_once_disposal_started() {
    // The TS `fails a captured callback after disposal without entering
    // downstream policy`: a stale callback fails without entering the
    // waterfall, so a drain period accepts no new work.
    let lifecycle = RetryLifecycle::new();
    assert!(!lifecycle.aborted());

    lifecycle.dispose();

    assert!(lifecycle.begin().is_none());
    assert!(lifecycle.aborted());
}

#[tokio::test]
async fn a_registration_before_disposal_is_tracked_and_drains_when_settled() {
    let lifecycle = RetryLifecycle::new();
    let guard = lifecycle
        .begin()
        .expect("an undisposed lifetime accepts work");

    let mut drain = lifecycle.dispose();
    // The drain barrier is open only after the tracked recovery settles.
    tokio::time::timeout(Duration::from_millis(20), &mut drain)
        .await
        .expect_err("the drain waits for the active recovery");

    drop(guard);
    tokio::time::timeout(Duration::from_secs(1), drain)
        .await
        .expect("a settled recovery releases the drain");
}

#[tokio::test]
async fn the_drain_blocks_until_every_active_recovery_settles() {
    let lifecycle = RetryLifecycle::new();
    let first = lifecycle.begin().unwrap();
    let second = lifecycle.begin().unwrap();

    let mut drain = lifecycle.dispose();
    tokio::time::timeout(Duration::from_millis(20), &mut drain)
        .await
        .expect_err("two active recoveries hold the drain");

    drop(first);
    tokio::time::timeout(Duration::from_millis(20), &mut drain)
        .await
        .expect_err("one active recovery still holds the drain");

    drop(second);
    tokio::time::timeout(Duration::from_secs(1), drain)
        .await
        .expect("the drain finishes once every recovery settled");
}

#[tokio::test]
async fn the_drain_completes_at_once_when_nothing_is_active() {
    let lifecycle = RetryLifecycle::new();
    tokio::time::timeout(Duration::from_millis(50), lifecycle.dispose())
        .await
        .expect("an idle lifetime drains immediately");
}

#[tokio::test]
async fn disposal_cancels_the_lifetime_signal() {
    // The TS disposal aborts `lifetime` before draining; every scheduled wait
    // fused to that signal must observe the cancellation.
    let lifecycle = RetryLifecycle::new();
    let lifetime = lifecycle.lifetime();
    assert!(!lifetime.is_cancelled());

    lifecycle.dispose();

    assert!(lifetime.is_cancelled());
    assert!(lifecycle.aborted());
}

/// The TS `aborts and drains a captured backoff before plugin disposal
/// completes`: a retry abandoned by the disposal never reaches
/// `llm/retry-started`, and the driver has no dangling wait left.
#[tokio::test]
async fn a_wait_abandoned_by_disposal_writes_no_retry_started() {
    let policy = always_policy(50.0, 50.0);
    let failure = failure("AUTH");
    let plan = plan_retry("mock", &failure, &policy, &RetryState::new(), || 0.5).unwrap();

    // The scheduled event is durable before the wait, exactly as the driver
    // writes it.
    let scheduled = encode_retry(&plan.retry_event(1, 1, failure)).unwrap();
    assert_eq!(decode_retry(&scheduled).unwrap().unwrap().retry(), 1);

    let lifecycle = RetryLifecycle::new();
    let turn_cancel = WaitCancel::new();
    let guard = lifecycle.begin().unwrap();

    // The host disposes mid-wait, aborting the fused lifetime signal.
    let host = lifecycle.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        host.dispose();
    });

    let waited = fused_wait(
        &TokioDelay,
        plan.delay_ms,
        &turn_cancel,
        &lifecycle.lifetime(),
    )
    .await;

    // The wait was abandoned: no `llm/retry-started`, no retried request.
    assert!(!waited);
    assert!(waited.then(|| plan.started_event(1, 1)).is_none());

    // The drain still completes once the abandoned recovery settles.
    drop(guard);
    tokio::time::timeout(Duration::from_secs(1), lifecycle.dispose())
        .await
        .expect("the abandoned recovery releases the drain");
}

/// The completion path: a wait that outlives the disposal race writes
/// `llm/retry-started` and releases the drain.
#[tokio::test]
async fn a_wait_completed_before_disposal_writes_retry_started() {
    let policy = always_policy(1.0, 1.0);
    let failure = failure("AUTH");
    let plan = plan_retry("mock", &failure, &policy, &RetryState::new(), || 0.5).unwrap();

    let lifecycle = RetryLifecycle::new();
    let turn_cancel = WaitCancel::new();
    let guard = lifecycle.begin().unwrap();

    let waited = fused_wait(
        &TokioDelay,
        plan.delay_ms,
        &turn_cancel,
        &lifecycle.lifetime(),
    )
    .await;
    assert!(waited);

    let started = waited.then(|| encode_retry_started(&plan.started_event(1, 1)).unwrap());
    assert!(started.is_some());

    drop(guard);
    tokio::time::timeout(Duration::from_secs(1), lifecycle.dispose())
        .await
        .expect("the settled recovery releases the drain");
}

#[tokio::test]
async fn a_cancelled_active_recovery_still_drains() {
    // `Promise.allSettled` semantics: a recovery ended by cancellation is as
    // settled as one that completed, so the drain must not hang on it.
    let lifecycle = RetryLifecycle::new();
    let guard = lifecycle.begin().unwrap();

    let task = tokio::spawn(async move {
        let _guard = guard;
        std::future::pending::<()>().await;
    });
    tokio::task::yield_now().await;
    task.abort();

    tokio::time::timeout(Duration::from_secs(1), lifecycle.dispose())
        .await
        .expect("a cancelled recovery is settled for the drain");
}

#[tokio::test]
async fn drain_is_repeatable_and_disposal_is_idempotent() {
    let lifecycle = RetryLifecycle::new();
    drop(lifecycle.begin().unwrap());

    lifecycle.dispose().await;
    tokio::time::timeout(Duration::from_millis(50), lifecycle.dispose())
        .await
        .expect("a second disposal drains the same idle lifetime at once");
    assert!(lifecycle.begin().is_none());
}

// ---------------------------------------------------- always negotiation ---

#[tokio::test]
async fn always_mode_adopts_a_downstream_retry_decision() {
    // The TS `lets downstream specialized recovery run before always
    // fallback`: the downstream decision is returned verbatim and nothing is
    // scheduled locally (no `llm/retry` event, no local wait).
    let cancel = WaitCancel::new();
    match negotiate_always(&cancel, &AllowRetry).await {
        AlwaysNegotiation::Adopted(RequestErrorAction::Retry) => {}
        other => panic!("the downstream decision must be adopted, got {other:?}"),
    }
}

#[tokio::test]
async fn always_mode_falls_through_when_downstream_has_no_opinion() {
    // A downstream without an opinion hands the decision back to the local
    // chain: plan_retry runs as usual and schedules the retry.
    let cancel = WaitCancel::new();
    let policy = always_policy(1.0, 1.0);
    let failure = failure("AUTH");

    match negotiate_always(&cancel, &NoOpinion).await {
        AlwaysNegotiation::Local => {}
        other => panic!("a no-opinion downstream must fall through, got {other:?}"),
    }
    let plan = plan_retry("mock", &failure, &policy, &RetryState::new(), || 0.5).unwrap();
    assert_eq!(plan.retry, 1);
    assert_eq!(plan.mode, RetryMode::Always);
    assert_eq!(plan.delay_ms, 1.0);
}

#[tokio::test]
async fn always_mode_ignores_a_downstream_failure_and_falls_through() {
    // The TS `falls back to always retry when downstream recovery throws
    // synchronously/asynchronously`: the failure is captured, warned about,
    // and ignored — the local always backoff still schedules.
    let cancel = WaitCancel::new();
    let policy = always_policy(1.0, 1.0);
    let failure = failure("AUTH");

    match negotiate_always(&cancel, &Fails).await {
        AlwaysNegotiation::Ignored(error) => {
            assert!(error.to_string().contains("downstream recovery failed"));
        }
        other => panic!("a failing downstream must be ignored, got {other:?}"),
    }
    let plan = plan_retry("mock", &failure, &policy, &RetryState::new(), || 0.5).unwrap();
    assert_eq!(plan.retry, 1);
}

#[tokio::test]
async fn an_abort_before_negotiation_skips_the_downstream() {
    // The TS always branch checks the fused signal before consulting `next`.
    let cancel = WaitCancel::new();
    cancel.cancel();
    let downstream = Counting(AtomicUsize::new(0));

    match negotiate_always(&cancel, &downstream).await {
        AlwaysNegotiation::Aborted => {}
        other => panic!("an abort must win before negotiation, got {other:?}"),
    }
    assert_eq!(downstream.0.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn an_abort_after_the_downstream_settled_wins_over_its_decision() {
    // The TS comment: the loop and plugin lifetime stay open until delegated
    // recovery settles, and an abort then wins before the decision or the
    // fallback can mutate later state.
    let cancel = WaitCancel::new();
    let gate = Arc::new(Notify::new());
    let downstream = SettleOnNotify {
        gate: Arc::clone(&gate),
    };

    let negotiation = tokio::spawn({
        let cancel = cancel.clone();
        async move { negotiate_always(&cancel, &downstream).await }
    });
    tokio::time::sleep(Duration::from_millis(10)).await; // the downstream is being consulted
    cancel.cancel();
    gate.notify_one(); // the downstream answers `retry` after the abort

    match negotiation.await.unwrap() {
        AlwaysNegotiation::Aborted => {}
        other => panic!("the abort must win over the late decision, got {other:?}"),
    }
}

#[test]
fn the_ignored_downstream_warning_matches_the_typescript_text() {
    assert_eq!(
        ignored_downstream_warning("mock"),
        "llm-retry: provider \"mock\" always policy ignored a downstream recovery failure"
    );
}

/// Normal mode never negotiates: an uncovered code delegates (the plan is
/// `None`) without consulting any downstream, and a covered code schedules
/// through the pure decision alone.
#[tokio::test]
async fn normal_mode_delegates_uncovered_codes_without_negotiation() {
    let policy = normal_policy();
    assert!(plan_retry(
        "mock",
        &failure("AUTH"),
        &policy,
        &RetryState::new(),
        || 0.5
    )
    .is_none());
    let plan = plan_retry(
        "mock",
        &failure("SERVER"),
        &policy,
        &RetryState::new(),
        || 0.5,
    )
    .unwrap();
    assert_eq!(plan.retry, 1);
    assert_eq!(plan.mode, RetryMode::Normal);
}

/// Both cancellation sources end a fused wait: a turn cancel abandons the
/// wait exactly as the disposal abort does, and the lifetime stays usable.
#[tokio::test]
async fn a_turn_cancel_during_a_fused_wait_abandons_it_too() {
    let lifecycle = RetryLifecycle::new();
    let turn_cancel = WaitCancel::new();
    turn_cancel.cancel();

    let waited = fused_wait(&TokioDelay, 50.0, &turn_cancel, &lifecycle.lifetime()).await;
    assert!(!waited);

    drop(lifecycle.begin().unwrap());
    tokio::time::timeout(Duration::from_secs(1), lifecycle.dispose())
        .await
        .expect("the turn-cancelled wait does not hold the drain");
}
