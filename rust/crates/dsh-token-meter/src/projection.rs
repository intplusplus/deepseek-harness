//! Pure client-safe token projections over the session log.
//!
//! Transcribed from `packages/llm/token-meter/src/projection.ts` and
//! `usage-projection.ts`. The `tokenUsage` unit accumulates every committed
//! assistant settlement's usage sample, replacing the previous sample of the
//! same attempt.
//!
//! # Not ported, and why
//!
//! The `contextPressure` and `contextBreakdown` units are not ported. Both
//! fold every surface event's positional operation (`surfaceOp`), which the
//! Rust core log does not carry — see [`crate::surface_fold`] for the gap.
//! Their state machines are exact only while the shadow-price protocol holds,
//! and reconstructing it here would silently drift on every historical
//! replay. Port them together with a Rust surface-op vocabulary.

use dsh_llm::TokenUsage;
use dsh_session::{CoreEvent, SessionEvent, MAX_SAFE_INTEGER};
use dsh_session_projection::{ProjectionDefinition, ProjectionState};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::turn_usage::{stream_usage, LLM_RETRY_STARTED_TAG};

/// Durable cumulative provider usage for a complete session log.
///
/// The four buckets are disjoint. In particular, reasoning tokens are already
/// included in `outputTokens` and are not accumulated again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TokenUsageProjection {
    /// Uncached prompt input across the whole log.
    pub uncached_input_tokens: u64,
    /// Completion output across the whole log.
    pub output_tokens: u64,
    /// Cached prompt tokens read across the whole log.
    pub cache_read_tokens: u64,
    /// Cached prompt tokens written across the whole log.
    pub cache_write_tokens: u64,
}

impl TokenUsageProjection {
    /// Reject a bucket outside the safe-integer bound.
    fn validate_buckets(&self) -> Result<(), String> {
        for (name, value) in [
            ("uncachedInputTokens", self.uncached_input_tokens),
            ("outputTokens", self.output_tokens),
            ("cacheReadTokens", self.cache_read_tokens),
            ("cacheWriteTokens", self.cache_write_tokens),
        ] {
            if value > MAX_SAFE_INTEGER {
                return Err(format!("{name} exceeds the safe-integer bound"));
            }
        }
        Ok(())
    }
}

impl ProjectionState for TokenUsageProjection {
    fn validate(&self) -> Result<(), String> {
        self.validate_buckets()
    }
}

/// The last attempt whose usage replaced into the totals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LastSample {
    turn: u64,
    step: u64,
    buckets: TokenUsageProjection,
}

/// The `tokenUsage` unit's persisted state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TokenUsageState {
    /// Cumulative usage across the whole log.
    pub totals: TokenUsageProjection,
    /// The attempt whose sample is currently counted, if any.
    last: Option<LastSample>,
}

impl ProjectionState for TokenUsageState {
    fn validate(&self) -> Result<(), String> {
        self.totals.validate_buckets()?;
        if let Some(last) = &self.last {
            last.buckets.validate_buckets()?;
        }
        Ok(())
    }
}

fn buckets_from(usage: &TokenUsage) -> TokenUsageProjection {
    TokenUsageProjection {
        uncached_input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_read_tokens: usage.cache_read_tokens.unwrap_or(0),
        cache_write_tokens: usage.cache_write_tokens.unwrap_or(0),
    }
}

fn add_replacing(
    totals: TokenUsageProjection,
    previous: Option<TokenUsageProjection>,
    next: TokenUsageProjection,
) -> TokenUsageProjection {
    let subtract = |total: u64, previous: Option<u64>| total.saturating_sub(previous.unwrap_or(0));
    TokenUsageProjection {
        uncached_input_tokens: subtract(
            totals.uncached_input_tokens,
            previous.map(|p| p.uncached_input_tokens),
        )
        .saturating_add(next.uncached_input_tokens),
        output_tokens: subtract(totals.output_tokens, previous.map(|p| p.output_tokens))
            .saturating_add(next.output_tokens),
        cache_read_tokens: subtract(
            totals.cache_read_tokens,
            previous.map(|p| p.cache_read_tokens),
        )
        .saturating_add(next.cache_read_tokens),
        cache_write_tokens: subtract(
            totals.cache_write_tokens,
            previous.map(|p| p.cache_write_tokens),
        )
        .saturating_add(next.cache_write_tokens),
    }
}

/// The usage one durable assistant settlement reports for its attempt, if any.
fn usage_of(event: &SessionEvent) -> Option<TokenUsage> {
    match event {
        SessionEvent::Core(CoreEvent::AssistantMessage { stream, usage, .. }) => {
            usage.as_deref().or_else(|| stream_usage(stream)).cloned()
        }
        SessionEvent::Core(CoreEvent::AssistantAttempt { stream, .. }) => {
            stream_usage(stream).cloned()
        }
        _ => None,
    }
}

fn turn_step_of(event: &SessionEvent) -> Option<(u64, u64)> {
    match event {
        SessionEvent::Core(CoreEvent::AssistantMessage { turn, step, .. })
        | SessionEvent::Core(CoreEvent::AssistantAttempt { turn, step, .. }) => {
            Some((*turn, *step))
        }
        SessionEvent::Extension(extension) => {
            if extension.kind() != LLM_RETRY_STARTED_TAG {
                return None;
            }
            let payload = extension.payload();
            Some((
                payload.get("turn")?.as_u64()?,
                payload.get("step")?.as_u64()?,
            ))
        }
        _ => None,
    }
}

/// The projection key owned by this unit.
pub const KEY: &str = "tokenUsage";

/// The persisted-cache version.
///
/// Matches the TypeScript unit: the state shape is the same four-bucket
/// cumulative total plus the last replaced attempt, so a cache row written by
/// either implementation applies to both.
pub const STATE_VERSION: u64 = 2;

/// The token-usage projection unit.
///
/// Each committed assistant settlement contributes the last usage sample
/// embedded in its stream. `llm/retry-started` closes the replacement slot so
/// the retried attempt adds to the total.
pub fn token_usage_projection_definition() -> ProjectionDefinition<TokenUsageState> {
    ProjectionDefinition::host_only(
        KEY,
        STATE_VERSION,
        |_header, _offset| TokenUsageState::default(),
        |state, event, _seq| {
            if let SessionEvent::Extension(extension) = event {
                if extension.kind() != LLM_RETRY_STARTED_TAG {
                    return None;
                }
                let (turn, step) = turn_step_of(event)?;
                let last = state.last?;
                return if last.turn == turn && last.step == step {
                    Some(TokenUsageState {
                        totals: state.totals,
                        last: None,
                    })
                } else {
                    None
                };
            }
            if !matches!(
                event,
                SessionEvent::Core(CoreEvent::AssistantMessage { .. })
                    | SessionEvent::Core(CoreEvent::AssistantAttempt { .. })
            ) {
                return None;
            }
            let sample = usage_of(event)?;
            let (turn, step) = turn_step_of(event)?;
            let buckets = buckets_from(&sample);
            let previous = match state.last {
                Some(last) if last.turn == turn && last.step == step => Some(last.buckets),
                _ => None,
            };
            if previous == Some(buckets) {
                return None;
            }
            Some(TokenUsageState {
                totals: add_replacing(state.totals, previous, buckets),
                last: Some(LastSample {
                    turn,
                    step,
                    buckets,
                }),
            })
        },
    )
    .with_wire(
        |state: &TokenUsageState| Arc::new(state.totals),
        |view: &TokenUsageProjection| view.validate_buckets(),
    )
}
