//! Token and turn accounting over the session log, ported from
//! `packages/session/session-stats`.
//!
//! The crate owns one projection unit, [`stats_projection`], keyed
//! `sessionStats`. It folds:
//!
//! * distinct turns carrying at least one `step/end` and the closed-step count;
//! * tool calls, results, failures, and orphan results, with pending calls
//!   pruned at `turn/end`;
//! * input, output, cache-read, cache-write, and reasoning tokens from every
//!   committed `assistant/message` that reports usage;
//! * the same token figures grouped by provider and model.
//!
//! Missing usage is a first-class outcome: the message still counts, its route
//! still counts, and `missingUsageMessages` records the gap instead of folding
//! a zero that looks like a real report.
//!
//! # Differences from the TypeScript unit
//!
//! The TypeScript projection counts model/tool/first-token/decode wall times
//! from the event envelope's `time` and embedded stream timestamps. Rust core
//! events carry no timestamp, so those fields are not modeled rather than
//! reported as zero. Token and provider/model accounting replaces them; the
//! TypeScript package has no token figures, so the client view is a Rust
//! extension, not a byte-for-byte port. `stateVersion` is 2 for that reason.
//!
//! State and view values are plain JSON with camelCase keys. All counters use
//! [`dsh_session::MAX_SAFE_INTEGER`] as their ceiling and saturate rather than
//! overflow.

#![forbid(unsafe_code)]

pub mod projection;

pub use projection::{
    stats_projection, RouteStats, SessionStatsState, SessionStatsView, KEY, STATE_VERSION,
};
