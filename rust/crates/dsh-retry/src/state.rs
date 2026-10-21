//! Retry state, folded back out of the session log.
//!
//! Ported from the `llmRetry` session projection in
//! `packages/llm/llm-retry/src/index.ts`. The projection is *derived* state:
//! nothing stores it, and replaying the log reproduces it exactly.
//!
//! Two things make the fold worth stating explicitly:
//!
//! - **`step/start` and `turn/end` reset it to empty.** A retry budget is per
//!   step, so a new step starts a new chain and a finished turn leaves nothing
//!   behind.
//! - **Re-applying the same event must not allocate.** The TypeScript test
//!   asserts `apply(state, event) === state`. Rust has no object identity, so
//!   the equivalent is "the returned map compares equal and was not rebuilt";
//!   the fold returns the incoming state unchanged on a redundant event.
//!
//! The map is a `BTreeMap` rather than a `HashMap` for the same reason the
//! tool registry is: the fold is replayed, and replay must be deterministic.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use dsh_session::{CoreEvent, SessionEvent};

use super::brand::RetryId;
use super::events::decode_retry;

/// One provider-and-policy retry chain's progress.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetryStateEntry {
    /// One-based index of the last scheduled retry in this chain.
    pub retry: u64,
    /// Identity of this chain.
    pub retry_id: RetryId,
}

/// Retry state for the current step, keyed by provider and policy.
///
/// The key is [`retry_state_key`], which mirrors the TypeScript
/// `Record<string, RetryStateEntry>` exactly — including the fact that a
/// policy change produces a different key, and therefore a fresh budget.
pub type RetryState = BTreeMap<String, RetryStateEntry>;

/// A `llm/retry` event whose payload does not match the declared shape.
///
/// Fold errors are not skipped: a known tag is never silently degraded, so a
/// malformed payload stops the fold instead of being ignored.
#[derive(Debug, thiserror::Error)]
#[error("retry state fold failed: {0}")]
pub struct RetryStateError(#[from] serde_json::Error);

/// The map key for one provider's retry chain under one policy.
pub fn retry_state_key(provider: &str, policy_key: &str) -> String {
    serde_json::to_string(&serde_json::Value::Array(vec![
        serde_json::Value::String(provider.to_string()),
        serde_json::Value::String(policy_key.to_string()),
    ]))
    .unwrap_or_default()
}

/// Fold one event into retry state.
///
/// A `llm/retry` event whose payload does not match the declared shape is an
/// error, not a skipped event: a known tag is never silently degraded.
pub fn apply_retry_state(
    mut state: RetryState,
    event: &SessionEvent,
) -> Result<RetryState, RetryStateError> {
    if matches!(
        event.core(),
        Some(CoreEvent::StepStart { .. }) | Some(CoreEvent::TurnEnd { .. })
    ) {
        return Ok(RetryState::new());
    }
    let Some(decoded) = decode_retry(event) else {
        return Ok(state);
    };
    let retry = decoded?;
    let key = retry_state_key(retry.provider(), retry.policy_key());
    if state
        .get(&key)
        .is_some_and(|entry| entry.retry == retry.retry() && entry.retry_id == *retry.retry_id())
    {
        return Ok(state);
    }
    state.insert(
        key,
        RetryStateEntry {
            retry: retry.retry(),
            retry_id: retry.retry_id().clone(),
        },
    );
    Ok(state)
}

/// Replay a whole log into retry state.
pub fn fold_retry_state(events: &[SessionEvent]) -> Result<RetryState, RetryStateError> {
    let mut state = RetryState::new();
    for event in events {
        state = apply_retry_state(state, event)?;
    }
    Ok(state)
}
