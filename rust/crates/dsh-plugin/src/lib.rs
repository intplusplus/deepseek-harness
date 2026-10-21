//! `dsh-plugin` — the Cordis-like kernel of the Rust harness.
//!
//! Everything is a plugin. Plugins contribute services and typed-event
//! registrations to a shared [`Context`], and every registration is a
//! reversible effect that unwinds when its [`Scope`] drops. There is no
//! privileged core to patch: you extend the harness by mounting a plugin
//! beside the others, exactly as the TypeScript harness does with Cordis.
//!
//! Three seams, mirroring the Cordis capability model:
//!
//! - **Service definition** — the [`Service`] marker; a swappable capability type.
//! - **Service provider** — the `Arc<T>` registered through [`Scope::set`].
//! - **Consumer** — [`ServiceRegistry::get`] / [`ServiceRegistry::require`].
//!
//! Events carry the extension points: observer events (`emit`) whose failures
//! are contained, and waterfall events (`dispatch`) whose listeners must call
//! `next()` to delegate.

#![forbid(unsafe_code)]

pub mod context;
pub mod event;
pub mod plugin;

pub use context::{Context, ContextError, Service, ServiceGuard, ServiceRegistry};
pub use event::{Event, EventBus, EventError, ObserverGuard, WaterfallGuard};
pub use plugin::{Plugin, PluginError, PluginHost, Scope};
