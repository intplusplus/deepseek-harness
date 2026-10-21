//! The recovery lifecycle: downstream negotiation and the unload drain.
//!
//! Two behaviours from the TypeScript plugin in
//! `packages/llm/llm-retry/src/index.ts` are lifetime concerns rather than
//! parts of the pure decision, and this module ports them:
//!
//! ## Downstream negotiation in always mode
//!
//! The TypeScript `recover()` awaits the rest of the `agent/request-error`
//! waterfall *first* when the provider policy is `always`: a specialized
//! recovery that answered `retry` wins and no local retry is scheduled; a
//! downstream *failure* is logged and ignored, falling through to the local
//! unbounded backoff; and an abort wins over both the decision and the
//! fallback, checked before and after the downstream settles.
//!
//! [`settle_downstream`] is the `settleDownstream` bridge and
//! [`negotiate_always`] is the always branch itself. It sits on the decision
//! chain *before* [`decide::plan_retry`]: a `Adopted` result replaces local
//! scheduling entirely (the TypeScript listener returns the downstream
//! decision without writing `llm/retry`), while every other fall-through
//! result hands the decision to [`decide::plan_retry`] as before. Normal
//! mode never negotiates.
//!
//! ## Unload drain
//!
//! On disposal the TypeScript plugin removes its listener, aborts a plugin
//! lifetime signal fused into every scheduled wait, and awaits
//! `Promise.allSettled` over the still-active recovery operations before
//! disposal completes. A wait abandoned by that abort resolves without
//! writing `llm/retry-started`, so a scheduled retry never dangles: the log
//! either shows the whole chain (`llm/retry` then `llm/retry-started`) or,
//! for an abandoned wait, a scheduled record whose chain was cut by the
//! documented plugin disposal. A stale callback captured before disposal is
//! rejected instead of entering downstream recovery.
//!
//! [`RetryLifecycle`] is that lifetime: one abort source plus a drain barrier.
//! [`begin`] registers an in-flight recovery and is rejected once disposal
//! started; [`RetryLifecycle::dispose`] raises the abort and yields a future
//! that settles only when every registered recovery has settled — by
//! completing, by cancellation, or by panic unwind.

#![forbid(unsafe_code)]

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use async_trait::async_trait;
use tokio::sync::watch;

use super::decide::RequestErrorAction;
use super::runtime::{RetryDelay, WaitCancel};

// ------------------------------------------------------- negotiation ------

/// Why a downstream recovery could not answer.
#[derive(Debug, thiserror::Error)]
#[error("downstream recovery failed: {0}")]
pub struct DownstreamError(#[source] pub Box<dyn std::error::Error + Send + Sync>);

/// One downstream recovery failure, or its decision.
///
/// The TypeScript bridge is `DownstreamOutcome =
/// { type: 'decision', decision } | { type: 'error', error }`; the `None`
/// decision is the TypeScript `undefined` — a recovery with no opinion.
#[derive(Debug)]
pub enum DownstreamOutcome {
    /// The downstream recovery answered; `None` means it had no opinion.
    Decision(Option<RequestErrorAction>),
    /// The downstream recovery failed and must be ignored, never propagated.
    Error(DownstreamError),
}

/// The next recovery listener the host drives.
///
/// The TypeScript listener awaits `next()`, the rest of the
/// `agent/request-error` waterfall. There is no waterfall here: the host
/// supplies this trait instead, and [`settle_downstream`] guarantees a
/// thrown (erroring) recovery is captured rather than propagated, exactly as
/// the TypeScript `settleDownstream` does.
#[async_trait]
pub trait DownstreamRecovery: Send + Sync {
    /// Ask the downstream recovery for its decision.
    async fn recover(&self) -> Result<Option<RequestErrorAction>, DownstreamError>;
}

/// Await one downstream recovery, capturing its failure.
///
/// The TypeScript `settleDownstream`: a throwing downstream becomes an
/// outcome the caller logs and ignores, not a rejected recovery.
pub async fn settle_downstream(next: &dyn DownstreamRecovery) -> DownstreamOutcome {
    match next.recover().await {
        Ok(decision) => DownstreamOutcome::Decision(decision),
        Err(error) => DownstreamOutcome::Error(error),
    }
}

/// The always-mode negotiation outcome.
#[derive(Debug)]
pub enum AlwaysNegotiation {
    /// The downstream recovery answered `retry`; adopt its decision and
    /// schedule nothing locally.
    Adopted(RequestErrorAction),
    /// The downstream recovery failed and was ignored; fall through to local
    /// scheduling. The caller is expected to log
    /// [`ignored_downstream_warning`].
    Ignored(DownstreamError),
    /// The downstream recovery had no retry to offer; fall through to local
    /// scheduling.
    Local,
    /// An abort won before the downstream was consulted or right after it
    /// settled; neither decision nor fallback may mutate later state.
    Aborted,
}

/// Negotiate with the downstream recovery before an always-mode retry.
///
/// The TypeScript always branch of `recover()`: consult the downstream
/// first, let an abort win over both the decision and the fallback, adopt a
/// downstream `retry` verbatim, and treat a downstream failure as ignored.
/// The `cancel` source must fuse the turn's cancellation with the plugin
/// lifetime, mirroring `AbortSignal.any([signal, lifetime.signal])`.
///
/// On every fall-through result ([`AlwaysNegotiation::Local`] and
/// [`AlwaysNegotiation::Ignored`]) the caller proceeds to
/// [`decide::plan_retry`]; on [`AlwaysNegotiation::Adopted`] the downstream
/// decision is returned to the driver and no `llm/retry` event is written.
pub async fn negotiate_always(
    cancel: &WaitCancel,
    downstream: &dyn DownstreamRecovery,
) -> AlwaysNegotiation {
    if cancel.is_cancelled() {
        return AlwaysNegotiation::Aborted;
    }
    let outcome = settle_downstream(downstream).await;
    // The loop and the plugin lifetime stay open until the delegated
    // recovery settles; an abort then wins before the decision or the
    // fallback can mutate later state.
    if cancel.is_cancelled() {
        return AlwaysNegotiation::Aborted;
    }
    match outcome {
        DownstreamOutcome::Decision(Some(action)) => AlwaysNegotiation::Adopted(action),
        DownstreamOutcome::Decision(None) => AlwaysNegotiation::Local,
        DownstreamOutcome::Error(error) => AlwaysNegotiation::Ignored(error),
    }
}

/// The TypeScript warn text for an ignored downstream recovery failure.
///
/// The TypeScript listener logs
/// `` `llm-retry: provider "${provider}" always policy ignored a downstream
/// recovery failure: %o` `` with the error appended; this is the prefix the
/// caller should log before the error detail.
pub fn ignored_downstream_warning(provider: &str) -> String {
    format!(
        "llm-retry: provider \"{provider}\" always policy ignored a downstream recovery failure"
    )
}

// ---------------------------------------------------------- the wait ------

/// Spend one scheduled retry delay fused to both cancellation sources.
///
/// The TypeScript `backoff()` waits on `AbortSignal.any([signal,
/// lifetime.signal])`; the driver here owns the wait and races it against a
/// [`WaitCancel`]. This helper is that race: the delay wins by completing,
/// either cancellation source ends it early with `false`, and `false` — from
/// either source — means the caller must stop the step without writing
/// `llm/retry-started`.
pub async fn fused_wait(
    delay: &dyn RetryDelay,
    delay_ms: f64,
    turn_cancel: &WaitCancel,
    lifetime: &WaitCancel,
) -> bool {
    let waited = delay.wait(delay_ms);
    tokio::pin!(waited);
    let turn = turn_cancel.cancelled();
    tokio::pin!(turn);
    let life = lifetime.cancelled();
    tokio::pin!(life);
    tokio::select! {
        result = &mut waited => result,
        _ = &mut turn => false,
        _ = &mut life => false,
    }
}

// ------------------------------------------------------- the lifetime -----

/// The in-flight recovery count shared by the lifecycle and its guards.
#[derive(Debug, Default)]
struct LifecycleState {
    draining: bool,
    active: usize,
}

/// The plugin-side lifetime: an abort source plus a drain barrier.
///
/// A host that unloads the retry facility mirrors the TypeScript disposal:
/// [`dispose`] rejects new recoveries, aborts every scheduled wait fused to
/// [`RetryLifecycle::lifetime`], and drains the still-active recoveries
/// before the unload completes. Cloneable, so the host, the driver, and the
/// active recoveries can each hold a handle.
#[derive(Clone)]
pub struct RetryLifecycle {
    state: Arc<Mutex<LifecycleState>>,
    active_tx: watch::Sender<usize>,
    lifetime: WaitCancel,
}

impl RetryLifecycle {
    /// A lifetime that is not yet disposing.
    pub fn new() -> Self {
        // The count channel doubles as the drain wake-up: a guard's drop
        // publishes the new count, and the drain future observes it through
        // `wait_for`, which re-checks the current value before sleeping — so
        // a settle landing between two polls is never lost.
        let (active_tx, _) = watch::channel(0);
        Self {
            state: Arc::new(Mutex::new(LifecycleState::default())),
            active_tx,
            lifetime: WaitCancel::new(),
        }
    }

    /// The abort source fused into every scheduled wait.
    pub fn lifetime(&self) -> WaitCancel {
        self.lifetime.clone()
    }

    /// Whether disposal has started.
    pub fn aborted(&self) -> bool {
        self.lifetime.is_cancelled()
    }

    /// Register one in-flight recovery.
    ///
    /// Returns [`None`] once disposal started — a stale callback captured
    /// before unload must fail without entering downstream recovery, exactly
    /// as the TypeScript listener returns early when the lifetime signal is
    /// aborted. Drop the returned guard (by settling, cancelling, or
    /// unwinding from a panic) to mark the recovery settled.
    pub fn begin(&self) -> Option<ActiveRecovery> {
        let mut state = self.lock();
        if state.draining {
            return None;
        }
        state.active += 1;
        // `send_replace` instead of `send`: the value must update even when
        // no receiver exists yet, or a drain started before any subscriber
        // would observe a stale zero.
        self.active_tx.send_replace(state.active);
        Some(ActiveRecovery {
            state: Arc::clone(&self.state),
            active_tx: self.active_tx.clone(),
        })
    }

    /// Abort the lifetime and drain the active recoveries.
    ///
    /// Awaiting the returned future suspends the unload until every
    /// registered recovery has settled — the `await Promise.allSettled`
    /// barrier. Rejections cannot escape a drain: what is awaited is the
    /// settle, not the result.
    pub fn dispose(&self) -> Drain {
        {
            let mut state = self.lock();
            state.draining = true;
        }
        self.lifetime.cancel();
        let mut active_rx = self.active_tx.subscribe();
        Drain {
            waiting: Box::pin(async move {
                let _ = active_rx.wait_for(|count| *count == 0).await;
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LifecycleState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl Default for RetryLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RetryLifecycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.lock();
        f.debug_struct("RetryLifecycle")
            .field("draining", &state.draining)
            .field("active", &state.active)
            .field("aborted", &self.lifetime.is_cancelled())
            .finish()
    }
}

/// The guard for one registered in-flight recovery.
///
/// Dropping it marks the recovery settled for the drain — whether the
/// recovery completed, was cancelled, or panicked. Guarded by the same
/// mutex as [`RetryLifecycle::begin`], so a registration cannot race a
/// concurrent disposal into a lost count.
pub struct ActiveRecovery {
    state: Arc<Mutex<LifecycleState>>,
    active_tx: watch::Sender<usize>,
}

impl Drop for ActiveRecovery {
    fn drop(&mut self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.active = state.active.saturating_sub(1);
        self.active_tx.send_replace(state.active);
    }
}

/// The unload drain: resolves once every active recovery has settled.
pub struct Drain {
    waiting: Pin<Box<dyn Future<Output = ()> + Send>>,
}

impl Future for Drain {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        self.waiting.as_mut().poll(cx)
    }
}
