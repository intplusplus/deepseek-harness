//! Execution-mode limits and the sandbox invocation, ported from
//! `packages/ptc-runtime/ptc-runtime-node/src/config.ts` and `launch.ts`.
//!
//! The TypeScript provider validates the same five knobs (budgets, output and
//! message caps, pending-call ceiling, grace) and refuses a deployment that
//! would make the frame format lie: a message cap above the 32-bit frame
//! length field cannot be honored, and an output cap below the empty logs
//! array cannot admit anything.

#![forbid(unsafe_code)]

use std::time::Duration;

/// Why a configuration was refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// A numeric knob was zero.
    #[error("ptc-host: {field} must be positive")]
    Positive {
        /// The offending field name.
        field: &'static str,
    },
    /// The output cap cannot admit even the empty logs array.
    #[error("ptc-host: maxOutputBytes must be an integer of at least 4")]
    OutputTooSmall,
    /// The message cap does not fit the frame format's 32-bit length field.
    #[error("ptc-host: maxMessageBytes must fit an unsigned 32-bit frame length")]
    MessageTooLarge,
    /// The sandbox invocation names no executable.
    #[error("ptc-host: launch.executable must be non-empty")]
    EmptyExecutable,
    /// `idle_timeout_ms` was `Some(0)`.
    #[error("ptc-host: idleTimeoutMs must be positive when set")]
    IdleTimeout,
}

/// How to start one sandbox child.
///
/// Ported from `launch.ts`, reduced to what a Rust host needs: the child is
/// whatever executable speaks the frame protocol over stdin/stdout. The host
/// appends the Node heap flag (`--max-old-space-size`) when configured, so the
/// args here name the bootstrap and its mode, not interpreter tuning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Executable name or path, resolved against the host's `PATH`.
    pub executable: String,
    /// Arguments following the executable.
    pub args: Vec<String>,
}

impl LaunchConfig {
    /// One sandbox invocation.
    pub fn new(executable: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            executable: executable.into(),
            args,
        }
    }
}

/// Runtime bounds for the sandbox host, ported from `config.ts`.
///
/// `idle_timeout_ms` is a Rust-side addition: the TypeScript provider retains
/// a persistent process until release, invalidation, or disposal, with no
/// wall-clock reaping. A long-lived Rust host wants an idle ceiling, so the
/// pool may reap a retained process that has not run for this long. `None`
/// restores the TypeScript behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostConfig {
    /// Default elapsed budget in milliseconds when a request names none.
    pub timeout_ms: u64,
    /// Largest elapsed budget accepted from a request.
    pub max_timeout_ms: u64,
    /// Combined serialized logs plus completion value or failure diagnostic.
    pub max_output_bytes: usize,
    /// Maximum control frame, queued-write and outstanding-argument bytes.
    pub max_message_bytes: usize,
    /// Maximum simultaneous host binding calls accepted from one program.
    pub max_pending_calls: usize,
    /// Managed process termination and output-drain grace in milliseconds.
    pub grace_ms: u64,
    /// Reap a retained process idle for this long; `None` retains forever.
    pub idle_timeout_ms: Option<u64>,
    /// Node old-generation heap flag appended to the child argv, in MiB.
    pub max_old_generation_size_mb: Option<u32>,
}

impl Default for HostConfig {
    fn default() -> Self {
        // The defaults mirror `config.ts`.
        Self {
            timeout_ms: 120_000,
            max_timeout_ms: 600_000,
            max_output_bytes: 67_108_864,
            max_message_bytes: 134_217_728,
            max_pending_calls: 128,
            grace_ms: 3_000,
            idle_timeout_ms: None,
            max_old_generation_size_mb: Some(512),
        }
    }
}

impl HostConfig {
    /// Validate the resolved bounds, mirroring `validateConfig`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (field, value) in [
            ("timeoutMs", self.timeout_ms),
            ("maxTimeoutMs", self.max_timeout_ms),
            ("graceMs", self.grace_ms),
        ] {
            if value == 0 {
                return Err(ConfigError::Positive { field });
            }
        }
        if self.max_pending_calls == 0 {
            return Err(ConfigError::Positive {
                field: "maxPendingCalls",
            });
        }
        if self.max_output_bytes < 4 {
            return Err(ConfigError::OutputTooSmall);
        }
        if self.max_message_bytes == 0 || self.max_message_bytes > u32::MAX as usize {
            return Err(ConfigError::MessageTooLarge);
        }
        if self.idle_timeout_ms == Some(0) {
            return Err(ConfigError::IdleTimeout);
        }
        Ok(())
    }

    /// The drain window after a managed kill, as a duration.
    pub fn grace(&self) -> Duration {
        Duration::from_millis(self.grace_ms)
    }

    /// The child argv with the heap flag prepended when configured.
    pub fn child_args(&self, launch: &LaunchConfig) -> Vec<String> {
        let mut args = Vec::with_capacity(launch.args.len() + 1);
        if let Some(mb) = self.max_old_generation_size_mb {
            args.push(format!("--max-old-space-size={mb}"));
        }
        args.extend(launch.args.iter().cloned());
        args
    }
}
