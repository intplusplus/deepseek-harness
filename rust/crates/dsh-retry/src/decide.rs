//! The recovery decision.
//!
//! Ported from `recover` in `packages/llm/llm-retry/src/index.ts`, minus the
//! two behaviours that are lifetime concerns rather than decisions:
//!
//! - **Downstream delegation in always mode.** The TypeScript listener awaits
//!   the rest of the `agent/request-error` waterfall first and yields to a
//!   specialized recovery that answered `retry`. That needs the async
//!   waterfall, so [`plan_retry`] assumes the caller has already given
//!   downstream recovery its chance.
//! - **Cancellation and disposal.** The wait is cancellable and the plugin
//!   drains in-flight waits before unload. [`plan_retry`] only says *how long*
//!   to wait; the caller owns the timer and the abort signal.
//!
//! What is left is the part with the actual rules, and it is pure: the same
//! failure, policy, state, and random sample always produce the same plan.
//!
//! A `None` return means "delegate" — this policy has no recovery to offer,
//! and whatever comes next in the waterfall decides.

#![forbid(unsafe_code)]

use dsh_llm::LlmFailure;

use super::backoff::local_delay;
use super::brand::RetryId;
use super::events::{LlmRetryEvent, LlmRetryStartedEvent};
use super::policy::{retry_policy_key, ResolvedRetryPolicy, RetryMode};
use super::state::{retry_state_key, RetryState};

/// The action a recovery listener can return.
///
/// The TypeScript union is `{ kind: 'retry' } | undefined`; in Rust the
/// "nothing to say" case is `Option::None`, so the variant itself carries no
/// data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestErrorAction {
    /// Retry the failed request.
    Retry,
}

/// One scheduled retry: everything needed to record it and to wait for it.
#[derive(Clone, Debug, PartialEq)]
pub struct RetryPlan {
    /// The provider route that served the failed request.
    pub provider: String,
    /// Which policy mode scheduled this retry.
    pub mode: RetryMode,
    /// Stable identity of the policy that scheduled this retry.
    pub policy_key: String,
    /// One-based index of this retry within its chain.
    pub retry: u64,
    /// Identity of this retry chain.
    pub retry_id: RetryId,
    /// How long to wait before the retried request, in milliseconds.
    pub delay_ms: f64,
    /// The budget this retry is drawn from; `None` under an always policy.
    pub max_retries: Option<u64>,
}

impl RetryPlan {
    /// The durable event written *before* the wait.
    pub fn retry_event(&self, turn: u64, step: u64, failure: LlmFailure) -> LlmRetryEvent {
        let failure = Box::new(failure);
        match self.mode {
            RetryMode::Normal => LlmRetryEvent::Normal {
                retry_id: self.retry_id.clone(),
                turn,
                step,
                provider: self.provider.clone(),
                policy_key: self.policy_key.clone(),
                retry: self.retry,
                max_retries: self.max_retries.unwrap_or_default(),
                delay_ms: self.delay_ms,
                failure,
            },
            RetryMode::Always => LlmRetryEvent::Always {
                retry_id: self.retry_id.clone(),
                turn,
                step,
                provider: self.provider.clone(),
                policy_key: self.policy_key.clone(),
                retry: self.retry,
                delay_ms: self.delay_ms,
                failure,
            },
        }
    }

    /// The durable transition written after the wait succeeds.
    pub fn started_event(&self, turn: u64, step: u64) -> LlmRetryStartedEvent {
        LlmRetryStartedEvent {
            retry_id: self.retry_id.clone(),
            turn,
            step,
            retry: self.retry,
        }
    }
}

/// Decide whether this policy recovers from `failure`, and if so how.
///
/// `state` is the retry state folded out of the log for the current step;
/// pass an empty map to start a fresh chain. `random` is the jitter sample in
/// `[0, 1]`.
///
/// Returns `None` when the policy has nothing to offer: the code is not
/// retryable, a normal budget is spent, or a normal policy was told to wait
/// longer than it is willing to.
pub fn plan_retry(
    provider: &str,
    failure: &LlmFailure,
    policy: &ResolvedRetryPolicy,
    state: &RetryState,
    random: impl FnOnce() -> f64,
) -> Option<RetryPlan> {
    if !policy.covers(&failure.code) {
        return None;
    }

    let policy_key = retry_policy_key(policy);
    let previous = state.get(&retry_state_key(provider, &policy_key));
    let previous_retry = previous.map(|entry| entry.retry).unwrap_or(0);

    if let (RetryMode::Normal, Some(max_retries)) = (policy.mode(), policy.max_retries()) {
        if previous_retry >= max_retries {
            return None;
        }
    }

    let retry = previous_retry + 1;
    let retry_id = previous
        .map(|entry| entry.retry_id.clone())
        .unwrap_or_else(RetryId::random);
    let backoff = policy.backoff();

    let delay_ms = match usable_retry_after(failure) {
        Some(after) if after > backoff.max_delay_ms => match policy.mode() {
            // A normal policy will not wait longer than it is configured to;
            // it delegates instead of silently clamping the provider's ask.
            RetryMode::Normal => return None,
            RetryMode::Always => local_delay(&backoff, retry, random),
        },
        Some(after) => after,
        None => local_delay(&backoff, retry, random),
    };

    Some(RetryPlan {
        provider: provider.to_string(),
        mode: policy.mode(),
        policy_key,
        retry,
        retry_id,
        delay_ms,
        max_retries: policy.max_retries(),
    })
}

/// The provider's own retry delay, when it is a usable positive duration.
fn usable_retry_after(failure: &LlmFailure) -> Option<f64> {
    let after = failure.provider_retry_after_ms? as f64;
    (after.is_finite() && after > 0.0).then_some(after)
}
