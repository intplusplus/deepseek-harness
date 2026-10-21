//! Skill catalog vocabulary and merge rules, ported from
//! `packages/skill/skill`.
//!
//! The TypeScript package is a Cordis service: providers register
//! asynchronously, catalogs are discovered per workspace, results are cached,
//! and a scope chain decides which layer owns a name. This crate ports the
//! part that has a deterministic answer without a host:
//!
//! - [`name`] — the kebab-case skill-name grammar as a validated newtype.
//! - [`types`] — the catalog vocabulary, including the exact summary wire
//!   shape, invocation policy, and rendering metadata.
//! - [`validate`] — the checks a provider must pass before a candidate is
//!   admitted to a catalog.
//! - [`merge`] — within-layer ranking (rank, provider order, provider-local
//!   order) and cross-layer shadowing (the nearest layer wins outright).
//! - [`render`] — the canonical `<skill_content>` block shared by the `skill`
//!   tool result and user-explicit invocation injection.
//!
//! ## What stays in the host
//!
//! Provider discovery, loading, cancellation, the collect cache, and the
//! Cordis `skills/change` event are lifetime concerns. They are intentionally
//! absent: every function here is synchronous, pure, and testable without a
//! clock, a filesystem, or a runtime.
//!
//! The TypeScript registry validates scalars that a typed Rust provider cannot
//! produce (non-string names, non-boolean invocation flags). Those checks are
//! replaced by the type system: a [`SkillName`] exists only after grammar
//! validation, and [`SkillInvocationPolicy`] carries two `bool` fields. What
//! remains checkable — empty descriptions, non-finite ranks, and provider
//! ownership — is still validated explicitly, because a provider is a
//! separately authored component.

#![forbid(unsafe_code)]

pub mod merge;
pub mod name;
pub mod render;
pub mod types;
pub mod validate;

pub use merge::{
    merge_layers, resolve_layer, runtime_layer, sort_summaries, IndexedCandidate,
    BUNDLED_SKILL_RANK, RUNTIME_PROVIDER, RUNTIME_RANK,
};
pub use name::{is_skill_name, SkillName};
pub use render::{escape_text, render_skill_content};
pub use types::{
    SkillCandidate, SkillDefinition, SkillInvocationForm, SkillInvocationPolicy,
    SkillInvocationSource, SkillInvocationSourceKind, SkillRank, SkillRegistration,
    SkillResourceBase, SkillSource, SkillSummary,
};
pub use validate::{validate_candidate, validate_definition, validate_registration, SkillError};
