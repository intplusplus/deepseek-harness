//! `dsh-ptc-host` — the PTC sandbox host: `dsh-ptc`'s frame protocol wired
//! to real child processes.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node` (the Node provider)
//! onto `tokio::process`. `dsh-ptc` owns every decision and every byte of
//! the wire format; this crate owns the action side:
//!
//! | module | TypeScript source | responsibility |
//! | --- | --- | --- |
//! | [`config`] | `config.ts`, `launch.ts` | limits, validation, child argv |
//! | [`env`] | `environment.ts` | startup-only child environment |
//! | [`channel`] | `channel.ts` (`JsonChannel`) | framed stdin/stdout pipes |
//! | [`process`] | `output-stream.ts`, spawn seams | child lifecycle, stderr, exit codes |
//! | [`evaluate`] | `process.ts` (`execute`), `persistent.ts` (`Kernel`) | the shared evaluation pump |
//! | [`host`] | `index.ts` (`NodePtcRuntime`, per-call) | one-shot runs |
//! | [`persistent`] | `persistent.ts`, `policy-processes.ts` | retained pool, reaping, invalidation |
//! | [`bindings`] | `ptc-runtime/src/types.ts` seam | host functions behind `call` frames |
//!
//! ## Where the Rust host diverges from the TypeScript provider
//!
//! * **Pipe layout.** TypeScript runs a third "control" pipe beside raw
//!   stdout/stderr, because its bootstrap emits raw program output. This
//!   host runs the frame protocol over the child's stdin/stdout, so program
//!   output arrives through `log` frames and stderr is diagnostics only.
//!   The wire format itself is unchanged.
//! * **One protocol for both modes.** The one-shot path speaks the persistent
//!   envelope protocol (`run {id, cwd, data}` / `{id, payload}`) instead of
//!   the bare `boot` handshake, so a sandbox implements one vocabulary.
//! * **Ownership instead of `WeakMap`.** A retained environment is keyed by
//!   an owner string supplied by the caller.
//! * **Idle reaping** (`HostConfig::idle_timeout_ms`) and **exit-code
//!   collection** (`PersistentHost::exit_info`) are additions; `None`
//!   restores the TypeScript retain-forever behavior.
//! * **No confinement backend.** Sandbox policy reaches the host only as a
//!   [`dsh_ptc::PolicyKey`] for reuse decisions and as the reported mode;
//!   `sandbox-unavailable` and denial signatures have no producer here.

#![forbid(unsafe_code)]

mod bindings;
mod channel;
pub mod config;
pub mod env;
mod evaluate;
pub mod host;
pub mod persistent;
pub mod process;

pub use bindings::{BindingFn, FnBinding, HostNamespace, PtcBinding};
pub use channel::{Channel, ChannelFailure};
pub use config::{ConfigError, HostConfig, LaunchConfig};
pub use env::{startup_environment, STARTUP_ENVIRONMENT_NAMES};
pub use host::{HostError, PtcHost, RunRequest};
pub use persistent::PersistentHost;
pub use process::{ExitInfo, SandboxProcess};
