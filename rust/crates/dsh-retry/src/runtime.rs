//! How a scheduled delay is actually spent.
//!
//! [`decide::plan_retry`] answers *whether* to retry and *for how long*; this
//! module is where the waiting happens. The TypeScript plugin waits inside
//! `recover()` with a cancellable `setTimeout` fused to the plugin lifetime;
//! here the wait is an asynchronous [`RetryDelay`] raced against a
//! [`WaitCancel`] by the driver, so a cancellation ends the wait instead of
//! outliving it.
//!
//! Splitting it this way keeps the rules testable without fake timers, and
//! lets the same driver run against a real sleep in production and an instant
//! no-op in tests.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::watch;

use super::policy::ResolvedRetryPolicy;

/// Spends one scheduled retry delay.
///
/// Returning `false` means the wait was cancelled: the caller must then stop
/// the step rather than open the retried request, exactly as the TypeScript
/// `cancellableDelay` resolves to `false` on abort.
///
/// The driver also races the returned future against [`RetryConfig::cancel`],
/// so an implementation only describes how long to wait; it does not need to
/// know where cancellation comes from. The returned future must be cancel-safe:
/// dropping it must not leave a trace, because that is what an abandoned wait
/// does.
#[async_trait]
pub trait RetryDelay: Send + Sync {
    /// Wait out `delay_ms` milliseconds.
    async fn wait(&self, delay_ms: f64) -> bool;
}

/// A real asynchronous wait, for production wiring.
pub struct TokioDelay;

#[async_trait]
impl RetryDelay for TokioDelay {
    async fn wait(&self, delay_ms: f64) -> bool {
        // A negative, NaN, or infinite delay has no meaningful duration; a
        // misresolved plan must not panic the driver.
        if let Ok(duration) = Duration::try_from_secs_f64(delay_ms / 1000.0) {
            tokio::time::sleep(duration).await;
        }
        true
    }
}

/// A wait that costs nothing.
///
/// Used by tests and by offline drivers that care about the *decision* the
/// policy produces, not about the wall clock it implies.
pub struct NoDelay;

#[async_trait]
impl RetryDelay for NoDelay {
    async fn wait(&self, _delay_ms: f64) -> bool {
        true
    }
}

/// A wait that is always cancelled.
///
/// Exists to prove the driver honours a cancelled wait instead of opening the
/// retried request anyway.
pub struct CancelledDelay;

#[async_trait]
impl RetryDelay for CancelledDelay {
    async fn wait(&self, _delay_ms: f64) -> bool {
        false
    }
}

/// A sticky, cloneable cancellation source for scheduled waits.
///
/// One wait may be racing against events the loop does not own (a user abort, a
/// shutdown), so the flag is shared rather than passed by value. Cancellation
/// is one-way: once raised it is never lowered, so a wait that observes it can
/// only stop. The flag is a `watch` value rather than a notification, so a
/// cancellation that lands between a caller's check and its wait is still
/// observed instead of being lost.
#[derive(Clone, Debug)]
pub struct WaitCancel {
    sender: watch::Sender<bool>,
    receiver: watch::Receiver<bool>,
}

impl Default for WaitCancel {
    fn default() -> Self {
        Self::new()
    }
}

impl WaitCancel {
    /// A source that is not yet cancelled.
    pub fn new() -> Self {
        let (sender, receiver) = watch::channel(false);
        Self { sender, receiver }
    }

    /// Raise cancellation. Idempotent.
    pub fn cancel(&self) {
        let _ = self.sender.send(true);
    }

    /// Whether cancellation has been raised.
    pub fn is_cancelled(&self) -> bool {
        *self.receiver.borrow()
    }

    /// Resolve once cancellation has been raised, including when it was raised
    /// before this call.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let mut receiver = self.receiver.clone();
        // `wait_for` re-checks the current value, so a cancellation between the
        // check above and this await cannot be missed.
        let _ = receiver.wait_for(|cancelled| *cancelled).await;
    }
}

/// Source of the jitter sample in `[0, 1)`.
pub type RandomSource = Arc<dyn Fn() -> f64 + Send + Sync>;

/// A sample source that always answers the same number.
pub fn constant_random(sample: f64) -> RandomSource {
    Arc::new(move || sample)
}

/// A sample source driven by a self-contained xorshift64 generator.
///
/// `seed` must be non-zero; a zero seed is nudged to one because xorshift
/// cannot escape a zero state. Deterministic for a given seed, which is what
/// makes a recorded retry chain reproducible.
pub fn seeded_random(seed: u64) -> RandomSource {
    let state = Mutex::new(seed | 1);
    Arc::new(move || {
        let mut state = state.lock().unwrap_or_else(|poison| poison.into_inner());
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        // 53 bits of the 64-bit state, so the result stays exact as f64.
        (x >> 11) as f64 / ((1u64 << 53) as f64)
    })
}

/// Default safety bound on model-request attempts within one step.
///
/// A `normal` policy already bounds itself by `maxRetries`, but an `always`
/// policy is unbounded by design, and an offline driver must not hang. This is
/// a Rust-side bound with no TypeScript counterpart, in the same spirit as
/// `dsh_agent_loop::DEFAULT_MAX_STEPS`.
pub const DEFAULT_MAX_ATTEMPTS: usize = 8;

/// How one loop spends a retry decision.
#[derive(Clone)]
pub struct RetryConfig {
    /// Retry policy per provider route.
    pub policies: BTreeMap<String, ResolvedRetryPolicy>,
    /// How to spend a scheduled delay.
    pub delay: Arc<dyn RetryDelay>,
    /// What ends a scheduled wait early.
    pub cancel: WaitCancel,
    /// Jitter sample source.
    pub random: RandomSource,
    /// Attempts allowed within one step, including the first.
    pub max_attempts: usize,
}

impl RetryConfig {
    /// A configuration with a real sleep and a seeded generator.
    pub fn new(policies: BTreeMap<String, ResolvedRetryPolicy>) -> Self {
        Self {
            policies,
            delay: Arc::new(TokioDelay),
            cancel: WaitCancel::new(),
            random: seeded_random(0x2545_F491_4F6C_DD1D),
            max_attempts: DEFAULT_MAX_ATTEMPTS,
        }
    }

    /// Override how a scheduled delay is spent.
    pub fn with_delay(mut self, delay: Arc<dyn RetryDelay>) -> Self {
        self.delay = delay;
        self
    }

    /// Override what ends a scheduled wait early.
    pub fn with_cancel(mut self, cancel: WaitCancel) -> Self {
        self.cancel = cancel;
        self
    }

    /// Override the jitter sample source.
    pub fn with_random(mut self, random: RandomSource) -> Self {
        self.random = random;
        self
    }

    /// Override the per-step attempt bound.
    pub fn with_max_attempts(mut self, max_attempts: usize) -> Self {
        self.max_attempts = max_attempts;
        self
    }
}

impl std::fmt::Debug for RetryConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryConfig")
            .field("policies", &self.policies.keys().collect::<Vec<_>>())
            .field("cancelled", &self.cancel.is_cancelled())
            .field("max_attempts", &self.max_attempts)
            .finish_non_exhaustive()
    }
}
