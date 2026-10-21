//! The local exponential delay.
//!
//! Ported from `localDelay` in `packages/llm/llm-retry/src/index.ts`. The
//! random sample is a parameter rather than an ambient `Math.random()`, which
//! is what makes the delay deterministic under test without fake timers.

#![forbid(unsafe_code)]

use super::policy::ResolvedRetryBackoff;

/// Cap on the backoff exponent before `2 ** exponent` overflows to infinity.
///
/// The TypeScript `Math.min(retry - 1, 1024)` is really "clamp the exponent
/// before the power overflows"; beyond this point the exponential term is
/// already at or above any representable cap, so `max_delay_ms` wins anyway.
pub const MAX_BACKOFF_EXPONENT: u64 = 1024;

/// The delay before retry `retry`, in milliseconds.
///
/// `retry` is one-based: the first retry of a chain is `1`. The result is
/// clamped to `max_delay_ms` after jitter, so jitter can never push a delay
/// past the ceiling.
pub fn local_delay(config: &ResolvedRetryBackoff, retry: u64, random: impl FnOnce() -> f64) -> f64 {
    let exponent = retry.saturating_sub(1).min(MAX_BACKOFF_EXPONENT);
    let exponential =
        (config.initial_delay_ms * 2f64.powi(exponent as i32)).min(config.max_delay_ms);
    let jitter = 1.0 - config.jitter_ratio + 2.0 * config.jitter_ratio * random();
    (exponential * jitter).min(config.max_delay_ms)
}
