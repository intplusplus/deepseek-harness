//! Session projection registry and drive, ported from
//! `packages/session/session-projection`.
//!
//! A projection is a pure synchronous fold with a per-session watermark. The
//! TypeScript package owns the same computation behind a Cordis service; this
//! crate keeps only the rules and the cache contract:
//!
//! * [`ProjectionDefinition`] registers a `&'static str` key, an
//!   [`ProjectionState`] type, `init(header, inheritedEventCount)`, `apply`,
//!   an optional client view, and a persisted-cache `stateVersion`.
//! * [`ProjectionRegistry`] drives every unit eagerly over committed events,
//!   keeps one watermark per session and key, notifies change listeners when a
//!   raw view allocation changes, checkpoints states, and restores from a
//!   persisted row plus a log suffix.
//! * [`Registration`] and [`ChangeSubscription`] are RAII equivalents of the
//!   Cordis effect disposers.
//!
//! # Type erasure and its safety boundary
//!
//! Rust has no declaration merging, so the TypeScript `SessionProjectionMap`
//! and `SessionProjectionStateMap` become a `&'static str` key plus a generic
//! `S: ProjectionState`. The registry stores `Arc<dyn Any>` cells and knows the
//! concrete type only through the erased unit it registered. Every transition
//! and read downcasts and reports a [`ProjectionError`] on mismatch; no
//! downcast is unchecked. The boundary the compiler does **not** prove is the
//! key-to-type association: two crates can use the same key with different
//! state types as long as their `stateVersion` matches, after which
//! [`ProjectionRegistry::state_of`] returns a type-mismatch error. Keys are
//! therefore product-owned constants (`sessionStats`, `turnOutline`) and a
//! second registration of one key is a reference count, not a replacement.
//!
//! # Identity and zero work
//!
//! `apply` returns `Option<S>`: `None` means “not my event” and the registry
//! reuses the exact cached `Arc`, so the state reference is unchanged and no
//! view is computed. `Some` is always a new reference, even when the value is
//! equal. A client view returns `Arc<W>`; two consecutive allocations are
//! compared with [`Arc::ptr_eq`], so a unit whose view reuses a sub-value
//! suppresses publication across internal-only state changes, while a view
//! that allocates fresh on every call publishes every state change.
//!
//! # Persisted cache
//!
//! A row is `(ver, seq, val)`; `seq` uses the TypeScript `-1` sentinel for the
//! empty log. Seeding parses `val` with serde and then calls
//! [`ProjectionState::validate`]. Unlike Zod, serde does not run semantic
//! checks, so state structs use `#[serde(deny_unknown_fields)]` and explicit
//! validators for ordering and budgets. A row is a fold shortcut: a version
//! mismatch, a malformed value, or a watermark beyond the supplied log end is
//! discarded on a full read and refused on a tail read.
//!
//! # Deliberate omissions
//!
//! The Cordis service lifecycle, the Session append hook, and the client
//! package are not part of this crate. Everything here is synchronous and
//! single-threaded; the host calls [`ProjectionRegistry::drive_event`] after
//! a commit and owns the durable cache I/O.

#![forbid(unsafe_code)]

mod erased;
mod registry;
mod types;

pub use registry::{ChangeSubscription, ProjectionRegistry, Registration};
pub use types::{
    ProjectionChange, ProjectionCheckpoint, ProjectionCheckpointRow, ProjectionCursor,
    ProjectionDefinition, ProjectionError, ProjectionSnapshot, ProjectionState, RestoreOutcome,
};
