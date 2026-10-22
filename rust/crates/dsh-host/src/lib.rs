//! `dsh-host` — the unified plugin-host assembly layer.
//!
//! The TypeScript harness composes every capability onto a Cordis `Context`
//! whose [`ctx.effect()`](https://cordis.js.org) registrations unwind in LIFO
//! order on unload. This crate is the Rust counterpart: a single
//! [`SessionScope`] mounts the now-available host components —
//! [`dsh_retry::RetryLifecycle`], [`dsh_ptc_host::PtcHost`] /
//! [`dsh_ptc_host::PersistentHost`], [`dsh_skill::host::SkillHost`], and the
//! interaction services from `dsh-interaction` — onto one [`dsh_plugin::Scope`]
//! and drains them in LIFO order when the scope drops.
//!
//! The composition never reimplements host logic: each `mount_*` wraps the
//! host's own teardown (`dispose` / `destroy_all` / drop) in a guard. See
//! [`scope`](crate::scope) for the lifetime contract and the `SkillHost` `Send`
//! seam.

#![forbid(unsafe_code)]

pub mod error;
pub mod scope;

pub use error::HostAssemblyError;
pub use scope::SessionScope;

// Re-export the kernel and the composed host types so a deployment needs only
// `dsh-host` in scope to assemble a full session.
pub use dsh_plugin::{Context, PluginHost, Scope, Service, ServiceGuard};
pub use dsh_ptc_host::{HostConfig, LaunchConfig, PersistentHost, PtcHost};
pub use dsh_retry::RetryLifecycle;
pub use dsh_skill::host::SkillHost;
