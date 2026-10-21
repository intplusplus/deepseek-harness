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
//! - [`host`] — the host layer on top of those pure functions: the provider
//!   registry with RAII registration guards, scope chains, the collect
//!   cache, and the filesystem provider, ported from the Cordis
//!   `SkillRegistry` service and `skill-filesystem` package.
//!
//! ## What stays in the host
//!
//! Provider discovery, loading, and the collect cache live in [`host`] as a
//! synchronous, single-threaded registry; the TypeScript abort-signal race
//! points and the Chokidar platform watchers are intentionally absent — the
//! watcher port keeps only the path-relevance and invalidation-notification
//! logic. Every function outside [`host`] remains synchronous, pure, and
//! testable without a clock, a filesystem, or a runtime.
//!
//! The TypeScript registry validates scalars that a typed Rust provider cannot
//! produce (non-string names, non-boolean invocation flags). Those checks are
//! replaced by the type system: a [`SkillName`] exists only after grammar
//! validation, and [`SkillInvocationPolicy`] carries two `bool` fields. What
//! remains checkable — empty descriptions, non-finite ranks, and provider
//! ownership — is still validated explicitly, because a provider is a
//! separately authored component.

#![forbid(unsafe_code)]

pub mod host;
pub mod merge;
pub mod name;
pub mod render;
pub mod types;
pub mod validate;

pub use host::{
    is_potential_skill_path, is_relevant_watch_event, CatalogSnapshot, ChangeRegistration, FsEntry,
    FsEntryKind, FsProviderConfig, FsSkillProvider, HostConfig, HostError, LocalLocator,
    ProviderControl, ProviderError, ProviderObservation, ProviderRegistration, RuntimeRegistration,
    SkillFiles, SkillHost, SkillLookupOptions, SkillProvider, SkillRoot, StdFiles, WatchEvent,
};

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
