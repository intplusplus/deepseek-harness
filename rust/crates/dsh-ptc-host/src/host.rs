//! The one-shot sandbox host, ported from `NodePtcRuntime`'s per-call path.
//!
//! Each run spawns a fresh child, drives one evaluation on it, tears the
//! child down, and resolves with a [`PtcRunResult`] whose `error` field
//! carries every host-side failure. A failed program is data, never a
//! rejected future — only contract misuse (an invalid budget, a non-absolute
//! cwd, a refused binding name) rejects.

#![forbid(unsafe_code)]

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use dsh_ptc::protocol::{NamespaceDeclaration, ProgramBootData, ProgramFailureKind};
use dsh_ptc::reserved::{validate_bindings, ReservedError};
use dsh_ptc::types::{PtcFailureKind, PtcRunFailure, PtcRunResult, PtcRunSandbox, SandboxMode};
use dsh_ptc::{clamp_timeout, OutputLedger, Timeout, TimeoutRequest};
use thiserror::Error;

use crate::bindings::{build_table, NamespaceTable};
use crate::config::{ConfigError, HostConfig, LaunchConfig};
use crate::evaluate::{drive, Params, Terminal};
use crate::process::SandboxProcess;

/// One execution request, mirroring `PtcRunSpec` after `resolve`.
#[derive(Clone)]
pub struct RunRequest {
    /// The program source, in the sandbox's language.
    pub program: String,
    /// Absolute working directory for this evaluation.
    pub cwd: String,
    /// The caller's elapsed-budget choice, clamped against the config.
    pub timeout: TimeoutRequest,
    /// Host functions exposed to the program, one namespace per global.
    pub bindings: Vec<crate::bindings::HostNamespace>,
    /// The file-policy identity this run is admitted under.
    pub policy: dsh_ptc::PolicyKey,
    /// Some when the caller names a retained state owner; the one-shot host
    /// ignores the value, the persistent pool keys on it.
    pub state_owner: Option<String>,
}

/// Why a host operation rejected before or around execution.
#[derive(Debug, Error)]
pub enum HostError {
    /// The configuration was refused.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The run was refused by the retained-process policy.
    #[error(transparent)]
    Policy(#[from] dsh_ptc::PolicyError),
    /// A binding declaration used a reserved or unusable name.
    #[error(transparent)]
    Bindings(#[from] ReservedError),
    /// A request input was malformed.
    #[error("ptc-host: {0}")]
    Resolve(String),
    /// The child could not be spawned.
    #[error("ptc-host: sandbox spawn failed: {0}")]
    Spawn(#[from] std::io::Error),
}

/// The one-shot PTC host: fresh child per run.
pub struct PtcHost {
    config: HostConfig,
    launch: LaunchConfig,
}

impl PtcHost {
    /// A host over one sandbox invocation.
    pub fn new(config: HostConfig, launch: LaunchConfig) -> Result<Self, HostError> {
        config.validate()?;
        if launch.executable.is_empty() {
            return Err(ConfigError::EmptyExecutable.into());
        }
        Ok(Self { config, launch })
    }

    /// The validated configuration.
    pub fn config(&self) -> &HostConfig {
        &self.config
    }

    /// Resolve one request's elapsed budget exactly as `run` would.
    pub fn resolve_timeout(
        &self,
        requested: TimeoutRequest,
    ) -> Result<Timeout, dsh_ptc::PolicyError> {
        clamp_timeout(
            requested,
            self.config.timeout_ms,
            self.config.max_timeout_ms,
            "timeoutMs",
        )
    }

    /// Run one program on a fresh child and tear the child down.
    pub async fn run(&self, request: RunRequest) -> Result<PtcRunResult, HostError> {
        let (timeout, table, boot) = resolve_inputs(&self.config, &request)?;
        let (process, mut inbound) = SandboxProcess::spawn(
            &self.launch,
            &self.config.child_args(&self.launch),
            &request.cwd,
            self.config.max_message_bytes,
            self.config.max_output_bytes,
        )?;
        let mut ledger = OutputLedger::new(self.config.max_output_bytes);
        let params = Params {
            eval_id: 1,
            cwd: request.cwd.clone(),
            boot,
            timeout,
            max_message_bytes: self.config.max_message_bytes,
            max_pending_calls: self.config.max_pending_calls,
            grace_ms: self.config.grace_ms,
            already_ready: false,
        };
        let report = drive(
            &process,
            &table,
            params,
            Arc::new(AtomicBool::new(false)),
            &mut ledger,
            &mut inbound,
        )
        .await;

        // A per-call child never outlives its run: end stdin so a healthy
        // child exits by itself, then kill and drain within the grace.
        process.seal_stdin();
        let drained = process.terminate(self.config.grace()).await;
        let mut result = compose(report.terminal, report.logs, &ledger, "");
        if !drained && result.error.is_none() {
            result = PtcRunResult::failure(
                result.logs.clone(),
                PtcRunFailure::new(
                    PtcFailureKind::WorkerExit,
                    "Node process output did not close cleanly",
                ),
            );
        }
        Ok(result.with_sandbox(PtcRunSandbox::new(request.policy.mode)))
    }
}

/// Validate one request and build its boot payload, shared by both hosts.
pub(crate) fn resolve_inputs(
    config: &HostConfig,
    request: &RunRequest,
) -> Result<(Timeout, NamespaceTable, ProgramBootData), HostError> {
    if !Path::new(&request.cwd).is_absolute() {
        return Err(HostError::Resolve("cwd must be absolute".to_owned()));
    }
    let timeout = clamp_timeout(
        request.timeout,
        config.timeout_ms,
        config.max_timeout_ms,
        "timeoutMs",
    )?;
    let (table, declarations) = build_table(request.bindings.clone());
    let validated = validate_bindings(declarations)?;
    let namespaces = validated
        .into_namespaces()
        .into_iter()
        .map(|namespace| NamespaceDeclaration {
            global: namespace.global,
            names: namespace.functions,
            error_class: namespace.error_class,
        })
        .collect();
    let boot = ProgramBootData {
        code: request.program.clone(),
        namespaces,
        max_output_bytes: config.max_output_bytes as u64,
    };
    Ok((timeout, table, boot))
}

/// Compose one terminal into a ledger-bounded result.
///
/// `suffix` is appended to the host-owned diagnostics; the persistent pool
/// passes the TypeScript provider's "retained variables were cleared" note.
pub(crate) fn compose(
    terminal: Terminal,
    logs: Vec<String>,
    ledger: &OutputLedger,
    suffix: &str,
) -> PtcRunResult {
    match terminal {
        Terminal::Completed(value) => ledger.success(logs, value.as_ref()),
        Terminal::ProgramFailure(failure) => ledger.failure(
            logs,
            PtcRunFailure::new(failure_kind(failure.kind), failure.message),
        ),
        Terminal::Timeout(ms) => ledger.failure(
            logs,
            PtcRunFailure::new(
                PtcFailureKind::Timeout,
                format!("execution deadline reached ({ms}ms){suffix}"),
            ),
        ),
        Terminal::WorkerExit(message) => ledger.failure(
            logs,
            PtcRunFailure::new(PtcFailureKind::WorkerExit, format!("{message}{suffix}")),
        ),
        Terminal::Protocol(message) => ledger.failure(
            logs,
            PtcRunFailure::new(PtcFailureKind::Protocol, format!("{message}{suffix}")),
        ),
        Terminal::OutputLimit => ledger.limit(logs),
        Terminal::Disposed => ledger.failure(
            logs,
            PtcRunFailure::new(
                PtcFailureKind::Abort,
                "PTC environment disposed; retained variables were cleared",
            ),
        ),
    }
}

fn failure_kind(kind: ProgramFailureKind) -> PtcFailureKind {
    match kind {
        ProgramFailureKind::Exception => PtcFailureKind::Exception,
        ProgramFailureKind::InvalidOutput => PtcFailureKind::InvalidOutput,
        ProgramFailureKind::OutputLimit => PtcFailureKind::OutputLimit,
    }
}

/// The sandbox mode this host reports; confinement itself is not ported.
///
/// The Rust host has no sandboxing backend to consult, so every result
/// reports the requested mode with no enforcement claim and no observed
/// denial — the same shape `danger-full-access` produces in TypeScript.
pub(crate) fn sandbox_report(mode: SandboxMode) -> PtcRunSandbox {
    PtcRunSandbox::new(mode)
}
