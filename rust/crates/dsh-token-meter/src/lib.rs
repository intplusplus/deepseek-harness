//! `dsh-token-meter` — token estimation, route pricing, surface folding, and
//! usage projections.
//!
//! Transcribed from `packages/llm/token-meter`. The TypeScript package is a
//! Cordis service that measures how much context a request occupies and how
//! much a completed turn actually cost; the Rust side keeps the parts that
//! have a deterministic answer without a host:
//!
//! - [`estimate`] — the fixed-density heuristic that prices content, messages,
//!   and the tool-schema part of a request envelope.
//! - [`surface_fold`] — the positional pricing fold over a session's visible
//!   surface, split into a read-only plan phase and an infallible commit.
//! - [`route_pricing`] — the request-projected surface pricing that replaces
//!   attachment heuristics with the image and file representations the routed
//!   model actually receives.
//! - [`turn_usage`] — the exact provider-reported token accounting for one
//!   completed turn, derived from its durable attempt lifecycle.
//! - [`projection`] — the session projections that publish the usage totals
//!   and the context occupancy.
//!
//! # Deliberate omissions
//!
//! - The Cordis service (`TokenMeter`) has no Rust counterpart: the Rust
//!   functions are pure and are composed by the caller.
//! - The client (`client.ts`) speaks the DeepSeek usage API over HTTP; the
//!   Rust side consumes the reported usage through [`turn_usage`] instead.
//!
//! Nothing here reads the clock, the filesystem, the environment, or a random
//! source: every value that could vary between runs is an argument.

#![forbid(unsafe_code)]

pub mod estimate;
pub mod projection;
pub mod route_pricing;
pub mod surface_fold;
pub mod turn_usage;

pub use estimate::{
    estimate_content, estimate_message, estimate_structural_block, estimate_system_message,
    estimate_tools_tokens, utf16_len, BLOCK_OVERHEAD, CHARS_PER_TOKEN, ROLE_OVERHEAD,
};
pub use projection::{
    token_usage_projection_definition, TokenUsageProjection, TokenUsageState, KEY, STATE_VERSION,
};
pub use route_pricing::{
    price_surface, ImagePrice, ImageRequestPricing, PricedSurface, RoutePricingError,
    TokenSurfaceNode,
};
pub use surface_fold::{
    commit_surface_tokens, derive_event_message, plan_surface_tokens, MeterImage, MeterSurfaceNode,
    SurfaceEvent, SurfaceOp, SurfaceTarget, SurfaceTokenError, SurfaceTokenPlan,
};
pub use turn_usage::{
    derive_turn_token_usage, TurnTokenUsage, TurnTokenUsageRoute, TurnUsageError,
    LLM_RETRY_STARTED_TAG, LLM_RETRY_TAG,
};
