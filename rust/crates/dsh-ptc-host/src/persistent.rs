//! The persistent-process pool, ported from `InMemoryExecution` + `Kernel`
//! and the invalidation side of `policy-processes.ts`.
//!
//! One retained environment per state owner, identified by an opaque owner
//! string standing in for the TypeScript WeakMap's object identity. Whether a
//! run reuses, replaces, or is refused is decided by `dsh-ptc`'s pure
//! `decide_process`; this module owns only the action side — spawning,
//! killing, and bookkeeping.
//!
//! Rust-side additions over the TypeScript provider, both documented on
//! [`HostConfig`]:
//!
//! * **Idle reaping** — a background task closes a retained process that has
//!   not run for `idle_timeout_ms` (the TypeScript provider retains forever
//!   until release, invalidation, or disposal).
//! * **Exit-code collection** — every closed kernel records its
//!   [`ExitInfo`], queryable via [`PersistentHost::exit_info`].

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dsh_ptc::policy::{
    decide_process, invalidated_processes, ModeChange, Owner, ProcessDecision, RetainedProcess,
};
use dsh_ptc::protocol::ProgramFailureKind;
use dsh_ptc::types::{PtcFailureKind, PtcRunFailure, PtcRunResult};
use dsh_ptc::OutputLedger;
use tokio::task::JoinHandle;

use crate::channel::InboundStream;
use crate::config::{ConfigError, HostConfig, LaunchConfig};
use crate::evaluate::{drive, Params, Terminal};
use crate::host::{compose, resolve_inputs, sandbox_report, HostError, RunRequest};
use crate::process::{ExitInfo, SandboxProcess};

struct Inner {
    key: dsh_ptc::PolicyKey,
    running: bool,
    cleanup_pending: bool,
    closed: bool,
    /// Whether the child completed its once-per-process readiness handshake.
    ready: bool,
    sequence: u64,
    exit_info: Option<ExitInfo>,
    last_active: Instant,
}

struct Kernel {
    inner: Mutex<Inner>,
    process: Mutex<Option<Arc<SandboxProcess>>>,
    inbound: Mutex<Option<InboundStream>>,
    disposed: Arc<AtomicBool>,
}

impl Kernel {
    fn new(key: dsh_ptc::PolicyKey) -> Self {
        Self {
            inner: Mutex::new(Inner {
                key,
                running: false,
                cleanup_pending: false,
                closed: false,
                ready: false,
                sequence: 0,
                exit_info: None,
                last_active: Instant::now(),
            }),
            process: Mutex::new(None),
            inbound: Mutex::new(None),
            disposed: Arc::new(AtomicBool::new(false)),
        }
    }

    fn snapshot(&self) -> RetainedProcess {
        let inner = self.inner.lock().expect("kernel lock");
        RetainedProcess {
            key: inner.key.clone(),
            running: inner.running,
            cleanup_pending: inner.cleanup_pending,
            closed: inner.closed,
        }
    }
}

struct Shared {
    config: HostConfig,
    launch: LaunchConfig,
    kernels: Mutex<HashMap<String, Arc<Kernel>>>,
}

/// The persistent PTC host: one retained process per state owner.
pub struct PersistentHost {
    shared: Arc<Shared>,
    reaper: Mutex<Option<JoinHandle<()>>>,
}

impl PersistentHost {
    /// A pool over one sandbox invocation.
    pub fn new(config: HostConfig, launch: LaunchConfig) -> Result<Self, HostError> {
        config.validate()?;
        if launch.executable.is_empty() {
            return Err(ConfigError::EmptyExecutable.into());
        }
        Ok(Self {
            shared: Arc::new(Shared {
                config,
                launch,
                kernels: Mutex::new(HashMap::new()),
            }),
            reaper: Mutex::new(None),
        })
    }

    /// The validated configuration.
    pub fn config(&self) -> &HostConfig {
        &self.shared.config
    }

    /// Evaluate one program on the owner's retained environment.
    ///
    /// Reuse, replacement, and refusal follow `dsh-ptc`'s
    /// [`decide_process`]: a busy owner is refused, a closed one replaced, a
    /// policy mismatch refused. Timeouts, output overflow, protocol
    /// failures, and substrate death close the environment — the retained
    /// variables are gone, and the diagnostics say so.
    pub async fn run(&self, request: RunRequest) -> Result<PtcRunResult, HostError> {
        self.ensure_reaper();
        let owner = request.state_owner.clone().ok_or_else(|| {
            HostError::Resolve("persistent run requires a state owner".to_owned())
        })?;
        let (timeout, table, boot) = resolve_inputs(&self.shared.config, &request)?;

        // Admission and the running flag flip in one critical section, so a
        // concurrent run for the same owner observes `Busy`, not a race.
        let kernel = {
            let mut kernels = self.shared.kernels.lock().expect("kernels lock");
            match kernels.get(&owner) {
                None => {
                    let kernel = Arc::new(Kernel::new(request.policy.clone()));
                    kernel.inner.lock().expect("kernel lock").running = true;
                    kernels.insert(owner, kernel.clone());
                    kernel
                }
                Some(existing) => {
                    let snapshot = existing.snapshot();
                    match decide_process(Owner::Retained, Some(&snapshot), &request.policy) {
                        ProcessDecision::Reuse => {
                            existing.inner.lock().expect("kernel lock").running = true;
                            existing.clone()
                        }
                        ProcessDecision::Spawn => {
                            let kernel = Arc::new(Kernel::new(request.policy.clone()));
                            kernel.inner.lock().expect("kernel lock").running = true;
                            kernels.insert(owner, kernel.clone());
                            kernel
                        }
                        ProcessDecision::Reject(error) => return Err(HostError::Policy(error)),
                    }
                }
            }
        };

        // The child and its inbound stream live across evaluations; the pump
        // borrows both and returns the stream on the keep-alive path.
        let (process, mut inbound) = {
            let mut process_guard = kernel.process.lock().expect("kernel process lock");
            match process_guard.clone() {
                Some(process) => {
                    let inbound = kernel
                        .inbound
                        .lock()
                        .expect("kernel inbound lock")
                        .take()
                        .expect("retained inbound stream");
                    (process, inbound)
                }
                None => {
                    let spawned = SandboxProcess::spawn(
                        &self.shared.launch,
                        &self.shared.config.child_args(&self.shared.launch),
                        &request.cwd,
                        self.shared.config.max_message_bytes,
                        self.shared.config.max_output_bytes,
                    );
                    match spawned {
                        Ok((process, inbound)) => {
                            let process = Arc::new(process);
                            *process_guard = Some(process.clone());
                            (process, inbound)
                        }
                        Err(error) => {
                            let mut inner = kernel.inner.lock().expect("kernel lock");
                            inner.running = false;
                            inner.closed = true;
                            return Err(HostError::Spawn(error));
                        }
                    }
                }
            }
        };

        let (sequence, already_ready) = {
            let mut inner = kernel.inner.lock().expect("kernel lock");
            inner.sequence += 1;
            inner.last_active = Instant::now();
            (inner.sequence, inner.ready)
        };
        let mut ledger = OutputLedger::new(self.shared.config.max_output_bytes)
            .with_overflow_message(format!(
                "retained variables were cleared; outer output exceeded {} bytes",
                self.shared.config.max_output_bytes
            ));
        let params = Params {
            eval_id: sequence,
            cwd: request.cwd.clone(),
            boot,
            timeout,
            max_message_bytes: self.shared.config.max_message_bytes,
            max_pending_calls: self.shared.config.max_pending_calls,
            grace_ms: self.shared.config.grace_ms,
            already_ready,
        };
        let report = drive(
            &process,
            &table,
            params,
            kernel.disposed.clone(),
            &mut ledger,
            &mut inbound,
        )
        .await;

        // A clean completion or a program exception keeps the environment;
        // everything else clears it (the TypeScript `discard`/`stop` split).
        let keep = match &report.terminal {
            Terminal::Completed(_) => true,
            Terminal::ProgramFailure(failure) => failure.kind != ProgramFailureKind::OutputLimit,
            _ => false,
        };
        let mut result = compose(
            report.terminal,
            report.logs,
            &ledger,
            "; retained variables were cleared",
        );
        if keep {
            *kernel.inbound.lock().expect("kernel inbound lock") = Some(inbound);
            let mut inner = kernel.inner.lock().expect("kernel lock");
            inner.running = false;
            inner.ready = true;
            inner.last_active = Instant::now();
        } else {
            drop(inbound);
            *kernel.inbound.lock().expect("kernel inbound lock") = None;
            process.seal_stdin();
            let drained = process.terminate(self.shared.config.grace()).await;
            let exit_info = if drained {
                Some(process.wait_exit().await)
            } else {
                None
            };
            {
                let mut inner = kernel.inner.lock().expect("kernel lock");
                inner.running = false;
                inner.closed = true;
                if let Some(info) = exit_info {
                    inner.exit_info = Some(info);
                }
            }
            if !drained {
                result = PtcRunResult::failure(
                    result.logs.clone(),
                    PtcRunFailure::new(
                        PtcFailureKind::WorkerExit,
                        "managed process exited but its output did not drain cleanly",
                    ),
                );
            }
        }
        Ok(result.with_sandbox(sandbox_report(request.policy.mode)))
    }

    /// Destroy one owner's environment and await its cleanup.
    ///
    /// An active run is aborted: the child dies and the in-flight evaluation
    /// settles as `abort` (the TypeScript `Kernel::dispose`).
    pub async fn destroy(&self, owner: &str) -> Result<(), HostError> {
        let kernel = self
            .shared
            .kernels
            .lock()
            .expect("kernels lock")
            .get(owner)
            .cloned();
        if let Some(kernel) = kernel {
            close_kernel(&kernel, self.shared.config.grace()).await;
        }
        Ok(())
    }

    /// Destroy every retained environment.
    pub async fn destroy_all(&self) -> Result<(), HostError> {
        let all: Vec<Arc<Kernel>> = self
            .shared
            .kernels
            .lock()
            .expect("kernels lock")
            .values()
            .cloned()
            .collect();
        for kernel in all {
            close_kernel(&kernel, self.shared.config.grace()).await;
        }
        Ok(())
    }

    /// Invalidate retained environments after a committed `sandbox/mode`
    /// change, using `dsh-ptc`'s shared invalidation decision.
    pub async fn invalidate(
        &self,
        session_id: &str,
        mode: dsh_ptc::SandboxMode,
    ) -> Result<(), HostError> {
        let victims: Vec<Arc<Kernel>> = {
            let kernels = self.shared.kernels.lock().expect("kernels lock");
            let owners: Vec<String> = kernels.keys().cloned().collect();
            let snapshots: Vec<RetainedProcess> = owners
                .iter()
                .map(|owner| kernels[owner].snapshot())
                .collect();
            invalidated_processes(&snapshots, ModeChange { session_id, mode })
                .into_iter()
                .map(|index| kernels[&owners[index]].clone())
                .collect()
        };
        for kernel in victims {
            close_kernel(&kernel, self.shared.config.grace()).await;
        }
        Ok(())
    }

    /// Every tracked environment's owner and current decision state.
    pub fn status(&self) -> Vec<(String, RetainedProcess)> {
        self.shared
            .kernels
            .lock()
            .expect("kernels lock")
            .iter()
            .map(|(owner, kernel)| (owner.clone(), kernel.snapshot()))
            .collect()
    }

    /// The exit outcome collected when an environment was last closed.
    pub async fn exit_info(&self, owner: &str) -> Option<ExitInfo> {
        let kernel = self
            .shared
            .kernels
            .lock()
            .expect("kernels lock")
            .get(owner)
            .cloned();
        let kernel = kernel?;
        let info = kernel.inner.lock().expect("kernel lock").exit_info.clone();
        info
    }

    /// Start the idle reaper once, inside a runtime, when configured.
    fn ensure_reaper(&self) {
        let mut reaper = self.reaper.lock().expect("reaper lock");
        if reaper.is_some() || self.shared.config.idle_timeout_ms.is_none() {
            return;
        }
        let idle_ms = self.shared.config.idle_timeout_ms.unwrap();
        let interval = Duration::from_millis((idle_ms / 4).clamp(20, 5_000));
        let idle = Duration::from_millis(idle_ms);
        let shared = self.shared.clone();
        *reaper = Some(tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let now = Instant::now();
                let victims: Vec<Arc<Kernel>> = {
                    let kernels = shared.kernels.lock().expect("kernels lock");
                    kernels
                        .values()
                        .filter(|kernel| {
                            let inner = kernel.inner.lock().expect("kernel lock");
                            !inner.closed
                                && !inner.running
                                && !inner.cleanup_pending
                                && now.duration_since(inner.last_active) > idle
                        })
                        .cloned()
                        .collect()
                };
                for kernel in victims {
                    close_kernel(&kernel, shared.config.grace()).await;
                }
            }
        }));
    }
}

impl Drop for PersistentHost {
    fn drop(&mut self) {
        if let Some(reaper) = self.reaper.lock().expect("reaper lock").take() {
            reaper.abort();
        }
    }
}

/// Close one kernel: abort any run, kill the child, collect the exit.
async fn close_kernel(kernel: &Kernel, grace: Duration) {
    {
        let mut inner = kernel.inner.lock().expect("kernel lock");
        if inner.closed {
            return;
        }
        inner.cleanup_pending = true;
    }
    kernel.disposed.store(true, Ordering::SeqCst);
    *kernel.inbound.lock().expect("kernel inbound lock") = None;
    let process = kernel.process.lock().expect("kernel process lock").clone();
    let exit_info = match process {
        Some(process) => {
            process.seal_stdin();
            process.kill();
            tokio::time::timeout(grace, process.wait_exit()).await.ok()
        }
        None => None,
    };
    let mut inner = kernel.inner.lock().expect("kernel lock");
    inner.closed = true;
    inner.running = false;
    inner.cleanup_pending = false;
    if let Some(info) = exit_info {
        inner.exit_info = Some(info);
    }
}
