//! `dsh-retry` — provider-routed model-request recovery.
//!
//! Ported from `packages/llm/llm-retry` and `packages/llm/llm/src/retry-policy.ts`.
//! A provider owns a [`ResolvedRetryPolicy`]; this crate turns one
//! [`LlmFailure`] plus the retry state folded out of the session log into a
//! decision: retry after a delay, or delegate to whatever recovery listener
//! comes next.
//!
//! ## What is here and what is not
//!
//! The TypeScript plugin is a Cordis waterfall listener: it awaits downstream
//! recovery, waits a cancellable delay, and drains active waits on disposal.
//! Downstream delegation and the disposal drain need the plugin host; the wait
//! itself is here, as an asynchronous [`RetryDelay`] raced against a
//! [`WaitCancel`] by the driver. Everything with a decision or a wire format in
//! it is ported:
//!
//! - [`policy`] — configuration, validation, and the resolved policy.
//! - [`backoff`] — the bounded exponential delay with symmetric jitter.
//! - [`events`] — the two durable events, as typed plugin events over the
//!   session log's extension slot.
//! - [`state`] — the projection fold that turns the log back into retry state.
//! - [`decide`] — the pure decision, with time and randomness injected so it
//!   is deterministic under test.
//! - [`runtime`] — the wait itself, behind a [`RetryDelay`] trait, plus the
//!   per-provider policy table a loop consults.
//! - [`lifecycle`] — the two host-side lifetime behaviours: the always-mode
//!   downstream negotiation, a pure decision point before [`decide::
//!   plan_retry`], and the unload drain that settles or abandons every
//!   in-flight wait on disposal.
//!
//! Splitting the decision out of the wait is what makes the recovery testable
//! without fake timers: [`decide::plan_retry`] returns a plan, and the caller
//! decides how to spend the delay.

#![forbid(unsafe_code)]

pub mod backoff;
pub mod brand;
pub mod decide;
pub mod events;
pub mod json;
pub mod lifecycle;
pub mod policy;
pub mod runtime;
pub mod state;

pub use backoff::{local_delay, MAX_BACKOFF_EXPONENT};
pub use brand::RetryId;
pub use decide::{plan_retry, RequestErrorAction, RetryPlan};
pub use events::{
    decode_retry, decode_retry_started, encode_retry, encode_retry_started, LlmRetryEvent,
    LlmRetryStartedEvent, LLM_RETRY_STARTED_TAG, LLM_RETRY_TAG,
};
pub use lifecycle::{
    fused_wait, ignored_downstream_warning, negotiate_always, settle_downstream, ActiveRecovery,
    AlwaysNegotiation, DownstreamError, DownstreamOutcome, DownstreamRecovery, Drain,
    RetryLifecycle,
};
pub use policy::{
    resolve_retry_policy, retry_policy_key, AlwaysRetryPolicyConfig, BackoffConfig,
    NormalRetryPolicyConfig, PolicyError, ResolvedRetryBackoff, ResolvedRetryPolicy, RetryMode,
    RetryPolicyConfig, DEFAULT_INITIAL_DELAY_MS, DEFAULT_JITTER_RATIO, DEFAULT_MAX_DELAY_MS,
    DEFAULT_MAX_RETRIES, DEFAULT_RETRYABLE_CODES, MAX_TIMER_DELAY_MS,
};
pub use runtime::{
    constant_random, seeded_random, CancelledDelay, NoDelay, RandomSource, RetryConfig, RetryDelay,
    TokioDelay, WaitCancel, DEFAULT_MAX_ATTEMPTS,
};
pub use state::{
    apply_retry_state, fold_retry_state, retry_state_key, RetryState, RetryStateEntry,
    RetryStateError,
};
