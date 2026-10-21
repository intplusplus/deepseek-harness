//! The run-level vocabulary: file policy, failure taxonomy, and the result.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime/src/types.ts`. The TypeScript
//! seam types the outcome as a *resolved* result with an optional `error`
//! field — a failed program is data the caller reports, never a rejected
//! promise. [`PtcRunResult`] keeps that shape.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// File-effect confinement applied to a program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxMode {
    /// Reads only.
    ReadOnly,
    /// Writes inside the workspace root.
    WorkspaceWrite,
    /// No confinement.
    DangerFullAccess,
}

/// How completely the selected backend enforces the file policy.
///
/// Absent for full access: a provider that does not confine reports no
/// completeness claim rather than a vacuous `full`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxEnforcement {
    /// Every path the policy restricts is restricted.
    Full,
    /// The backend confines what it can observe; see the provider's caveats.
    Partial,
}

/// The applied file policy and the observed denial, when one occurred.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PtcRunSandbox {
    /// The mode this execution ran under.
    pub mode: SandboxMode,
    /// Whether the program's failure text matched a backend denial signature.
    /// Evidence of a denial, not proof that every restriction held.
    pub denied: bool,
    /// Completeness reported by the selected confining backend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enforcement: Option<SandboxEnforcement>,
}

impl PtcRunSandbox {
    /// A sandbox report for `mode` with no observed denial.
    pub fn new(mode: SandboxMode) -> Self {
        Self {
            mode,
            denied: false,
            enforcement: None,
        }
    }
}

/// Why a run failed.
///
/// The kinds are orthogonal outcomes: a budget expiry is not an exception, an
/// abort is not a timeout, and a substrate death is neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PtcFailureKind {
    /// The program threw or failed to parse/transform.
    Exception,
    /// An implementation-owned elapsed budget expired.
    Timeout,
    /// The caller's cancellation fired.
    Abort,
    /// The execution substrate died without settling.
    WorkerExit,
    /// The completion value was not lossless JSON.
    InvalidOutput,
    /// Serialized logs/value/diagnostic exceeded the configured cap.
    OutputLimit,
    /// The program sent invalid or over-budget control traffic.
    Protocol,
    /// Required confinement could not be established.
    SandboxUnavailable,
}

/// One failure diagnostic, suitable for feeding back to a model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PtcRunFailure {
    /// The failure class.
    pub kind: PtcFailureKind,
    /// Human-readable detail.
    pub message: String,
}

impl PtcRunFailure {
    /// Create one diagnostic.
    pub fn new(kind: PtcFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// The outcome of one run.
///
/// Serialization order is `logs`, `value`, `error`, `sandbox`: the persistent
/// provider spreads its evaluation outcome (`{ logs, value? , error? }`) and
/// then attaches `sandbox` (`packages/ptc-runtime/ptc-runtime-node/src/persistent.ts`,
/// `Kernel.run`), so a byte-exact comparison sees the policy report last.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PtcRunResult {
    /// Captured text, in emission order per source channel.
    pub logs: Vec<String>,
    /// The program's completion value, when it ran to completion and the value
    /// crossed the lossless-JSON boundary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    /// Present iff the run failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<PtcRunFailure>,
    /// Applied file policy, when the provider enforces one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<PtcRunSandbox>,
}

impl PtcRunResult {
    /// A successful run with captured logs and no completion value.
    pub fn success(logs: Vec<String>) -> Self {
        Self {
            logs,
            value: None,
            error: None,
            sandbox: None,
        }
    }

    /// A failed run with captured logs.
    pub fn failure(logs: Vec<String>, error: PtcRunFailure) -> Self {
        Self {
            logs,
            value: None,
            error: Some(error),
            sandbox: None,
        }
    }

    /// Whether this result reports a failure.
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// The failure class, when the run failed.
    pub fn failure_kind(&self) -> Option<PtcFailureKind> {
        self.error.as_ref().map(|error| error.kind)
    }

    /// Attach the completion value.
    pub fn with_value(mut self, value: Value) -> Self {
        self.value = Some(value);
        self
    }

    /// Attach the file-policy report.
    pub fn with_sandbox(mut self, sandbox: PtcRunSandbox) -> Self {
        self.sandbox = Some(sandbox);
        self
    }
}
