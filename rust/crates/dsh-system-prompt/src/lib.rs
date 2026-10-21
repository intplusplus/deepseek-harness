//! `dsh-system-prompt` — the system-prompt registry.
//!
//! Ported from `packages/core/system-prompt`. The TypeScript package is a
//! Cordis service: sections, contexts, tool providers, and variables
//! register through `ctx.effect()` disposers, every registration and
//! disposal emits `system-prompt/change`, and `assemble()` runs a
//! cooperative waterfall. Those are **lifetime** concerns and need the
//! plugin host, so they stay on the "not ported" list. What is ported is
//! everything with data, rules, or a wire format in it:
//!
//! - [`orders`] — the centrally allocated section/context placement tables.
//! - [`types`] — registry inputs, the assembled model input, and config.
//! - [`render`] — strict `{{variable}}` interpolation and the snapshot join.
//! - [`registry`] — registration with duplicate/order validation, canonical
//!   ordering, tool ordering, suppression, and `refreshContext`.
//!
//! Registrations are RAII: a [`Registration`] removes what it added when
//! dropped, the Rust shape of the TypeScript effect disposer — keep it
//! alive to keep the contribution, drop it to dispose. Providers are
//! injected closures (`Fn(&AssembleContext) -> ..`), so no clock,
//! filesystem, or randomness is ever called from inside the registry; the
//! same "inject the environment" rule `dsh-retry` uses for its random
//! sample, which is what makes every rule testable without a host.
//!
//! ## What is not here, and why
//!
//! - **`system-prompt/change` emits and the P1-1 rollback** (a listener
//!   throw undoes the registration): both need the Cordis event bus.
//! - **The `system-prompt/assemble` waterfall**: composition,
//!   short-circuiting, and scope-filtered dispatch need the plugin host.
//!   [`SystemPrompt::assemble`] returns the pre-waterfall assembly, which
//!   is where every rule the tests pin lives. The two post-waterfall
//!   restorations — a complete section replacing whatever the waterfall
//!   produced, and required-only contexts under suppression — are folded
//!   in, because with no waterfall the registry's own output already
//!   satisfies them.
//! - **Scope chains**: `ScopedLayers.chainLayers` walks parent scopes, a
//!   rule owned by `@deepseek-ai/dsh-scope`. One global layer plus one
//!   named scope layer is modeled; the parent walk is not. A scope key is
//!   a plain `String` here rather than a `ScopeKey`, for the same reason.
//! - **Whole-layer disposal** (an HMR fiber dropping every registration it
//!   owns at once): drop the [`Registration`]s — or the owning
//!   [`SystemPrompt`] — instead.
//! - **The deferred scoped-variable replacement** (a provider disposing
//!   and re-registering itself mid-evaluation) needs re-entrant
//!   registration through a live shared borrow. The live-iteration rule it
//!   relies on *is* ported and tested; only the self-replacement dance is
//!   not.
//! - **`AbortSignal`** on [`AssembleContext`], and the schemastery config
//!   schema: the Rust [`Config`] struct plus constructor-time validation
//!   covers the same contract.
//!
//! There are no durable session events to bridge: unlike `dsh-retry`,
//! this package contributes no `SessionEventMap` members — its two Cordis
//! events (`system-prompt/assemble`, `system-prompt/change`) are volatile
//! plugin events, so no `SessionEvent::Extension` encode/decode pair
//! exists here.

#![forbid(unsafe_code)]

pub mod error;
pub mod orders;
pub mod registry;
pub mod render;
pub mod types;

pub use error::{InterpolationKind, PromptError};
pub use orders::{
    PromptContextOrderName, PromptSectionOrderName, HARNESS_IDENTITY_SECTION,
    HARNESS_IDENTITY_TEXT, PERSONA_PREFIX_SECTION, PERSONA_SUFFIX_SECTION, TOOL_ORDER_REST,
};
pub use registry::{Registration, SystemPrompt, ToolProvider, VariableProvider};
pub use render::{
    join_context_sections, render_context_sections, render_context_snapshot, render_prompt,
};
pub use types::{
    AssembleContext, AssembledContext, AssembledSection, Config, ContextSnapshotSection,
    PromptAssembly, PromptContext, PromptSection, PromptText, PromptVariables, ToolProviderResult,
};
