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
//!   ordering, tool ordering, suppression, scope chains, and
//!   `refreshContext`.
//! - [`events`] — the `system-prompt/change` notification and its bridge
//!   into the durable session log.
//! - [`waterfall`] — the cooperative `system-prompt/assemble` chain.
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
//! - **The P1-1 rollback** (a `system-prompt/change` listener throwing
//!   undoes the registration): the TypeScript bus rolls the mutation back;
//!   Rust listeners are infallible `Fn()`s.
//! - **The deferred scoped-variable replacement** (a provider disposing
//!   and re-registering itself mid-evaluation) needs re-entrant
//!   registration through a live shared borrow. The live-iteration rule it
//!   relies on *is* ported and tested; only the self-replacement dance is
//!   not.
//! - **`AbortSignal`** on [`AssembleContext`], and the schemastery config
//!   schema: the Rust [`Config`] struct plus constructor-time validation
//!   covers the same contract.
//!
//! ## The two Cordis events
//!
//! TypeScript declares both events on the Cordis bus and neither is in
//! `SessionEventMap`. Rust has no declaration merging, so each is a real
//! type here: the waterfall is a listener list with the bus's admission
//! rule, and the change notification reaches the durable log through
//! [`encode_change`] — the same bridge `dsh-retry` writes for `llm/retry`.

#![forbid(unsafe_code)]

pub mod error;
pub mod events;
pub mod orders;
pub mod registry;
pub mod render;
pub mod types;
pub mod waterfall;

pub use error::{InterpolationKind, PromptError};
pub use events::{decode_change, encode_change, SystemPromptChangeEvent, SYSTEM_PROMPT_CHANGE_TAG};
pub use orders::{
    PromptContextOrderName, PromptSectionOrderName, HARNESS_IDENTITY_SECTION,
    HARNESS_IDENTITY_TEXT, PERSONA_PREFIX_SECTION, PERSONA_SUFFIX_SECTION, TOOL_ORDER_REST,
};
pub use registry::ScopeParentBinding;
pub use registry::{ChangeListener, Registration, SystemPrompt, ToolProvider, VariableProvider};
pub use render::{
    join_context_sections, render_context_sections, render_context_snapshot, render_prompt,
};
pub use types::{
    AssembleContext, AssembledContext, AssembledSection, Config, ContextSnapshotSection,
    PromptAssembly, PromptContext, PromptSection, PromptText, PromptVariables, ToolProviderResult,
};
pub use waterfall::AssembleListener;
