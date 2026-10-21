//! Retained-process selection, mode invalidation, and deadline clamping.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node/src/persistent.ts`
//! (`InMemoryExecution::run` / `resolve`), `policy-processes.ts`, and
//! `packages/util/timeout` (`clampTimeout`).
//!
//! All of it is decision, none of it is action: the caller owns the process,
//! the timers, and the cancellation. What is decided here:
//!
//! * whether a run may reuse a retained environment, must start a fresh one,
//!   or must be refused outright;
//! * which tracked processes a committed `sandbox/mode` change invalidates;
//! * the effective elapsed budget for one run.
//!
//! ## Why refusal is a first-class outcome
//!
//! A retained environment holds the program's variables. Two runs against one
//! owner cannot overlap, and an environment cannot change its sandbox policy
//! under a program that already holds a descriptor. Both are *caller* errors —
//! the deployment asked for something incoherent — so they are
//! [`PolicyError`], not a failed run with a program-facing diagnostic.

#![forbid(unsafe_code)]

use super::types::SandboxMode;

/// The identity of one retained environment.
///
/// A change to any field makes the environment unusable for a later run: the
/// process was started with a mode, a workspace root, and a session, and
/// none of them can be renegotiated in place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyKey {
    /// The file policy the process was admitted under.
    pub mode: SandboxMode,
    /// The workspace root the process was confined to.
    pub workspace_root: String,
    /// The session that owns the process, when one does.
    pub session_id: Option<String>,
}

impl PolicyKey {
    /// Create one environment identity.
    pub fn new(
        mode: SandboxMode,
        workspace_root: impl Into<String>,
        session_id: Option<String>,
    ) -> Self {
        Self {
            mode,
            workspace_root: workspace_root.into(),
            session_id,
        }
    }
}

/// Whether the run names a retained state owner or is one-shot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    /// The request carries a state owner: state may be retained across runs.
    Retained,
    /// No state owner: the run gets a fresh, non-retained environment and the
    /// environment dies with the run.
    OneShot,
}

/// What is known about one retained environment, without holding it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedProcess {
    /// The identity it was admitted under.
    pub key: PolicyKey,
    /// Whether an evaluation is currently running.
    pub running: bool,
    /// Whether a previous cleanup has not yet completed.
    pub cleanup_pending: bool,
    /// Whether the process has been closed (killed, exited, or invalidated).
    pub closed: bool,
}

/// Why a run was refused before execution.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PolicyError {
    /// The owner's retained environment is busy with another run.
    #[error("persistent PTC does not accept concurrent runs for one state owner")]
    Busy,
    /// A previous environment's cleanup has not completed.
    #[error(
        "persistent PTC environment cleanup is incomplete; release the owner before running another program"
    )]
    CleanupPending,
    /// The requested policy differs from the retained environment's.
    #[error("persistent PTC: release the retained environment before changing its sandbox policy")]
    PolicyChange,
    /// A requested budget was zero.
    #[error("{name} must be a positive finite number")]
    InvalidTimeout {
        /// The field name, so the caller sees which input was refused.
        name: String,
    },
}

/// What the provider should do about an environment for one run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProcessDecision {
    /// Start a fresh environment that the run will tear down (one-shot, first
    /// run for a retained owner, or a replacement for a closed one).
    Spawn,
    /// Reuse the owner's live environment.
    Reuse,
    /// Refuse the run; see [`PolicyError`].
    Reject(PolicyError),
}

/// Decide whether a run reuses, replaces, or may not use an environment.
///
/// The branches and their order mirror the TypeScript `run`:
///
/// 1. A one-shot run never consults the retained map.
/// 2. A retained run with no environment starts one.
/// 3. `assertIdle` runs *before* any policy comparison, so a busy owner is
///    reported as busy even when the policy also changed.
/// 4. A closed environment is replaced rather than refused: it holds nothing,
///    so there is nothing to release.
/// 5. A policy mismatch is refused: the live process holds variables under an
///    authority the new run does not have.
pub fn decide_process(
    owner: Owner,
    retained: Option<&RetainedProcess>,
    requested: &PolicyKey,
) -> ProcessDecision {
    if owner == Owner::OneShot {
        return ProcessDecision::Spawn;
    }
    let Some(retained) = retained else {
        return ProcessDecision::Spawn;
    };
    if retained.running {
        return ProcessDecision::Reject(PolicyError::Busy);
    }
    if retained.cleanup_pending {
        return ProcessDecision::Reject(PolicyError::CleanupPending);
    }
    if retained.closed {
        return ProcessDecision::Spawn;
    }
    if retained.key != *requested {
        return ProcessDecision::Reject(PolicyError::PolicyChange);
    }
    ProcessDecision::Reuse
}

/// Whether a run may pass its state owner and therefore retain its variables.
///
/// An escalated run is temporary by design: `packages/core/tools/src/ptc.ts`
/// passes `stateOwner` only when the run's policy mode equals the session's
/// standing mode, so an approved escalation runs in a fresh environment and
/// discards its declarations.
///
/// Both modes are optional because a runtime that does not enforce file policy
/// has neither; two absent modes are equal, so such a runtime retains state.
pub fn retains_state(
    owner_present: bool,
    run_mode: Option<SandboxMode>,
    standing_mode: Option<SandboxMode>,
) -> bool {
    owner_present && run_mode == standing_mode
}

/// A committed `sandbox/mode` change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModeChange<'a> {
    /// The session whose policy changed.
    pub session_id: &'a str,
    /// The mode it changed to.
    pub mode: SandboxMode,
}

/// Indices of the tracked processes a mode change invalidates.
///
/// Shared by both trackers in the TypeScript provider: `PolicyProcesses`
/// cancels retained processes and `PtcCalls` aborts live calls. Both compare
/// the same two fields — the session and the mode the entry was admitted
/// under — so a change that leaves a process's own mode untouched does not
/// disturb it.
pub fn invalidated_processes(tracked: &[RetainedProcess], change: ModeChange<'_>) -> Vec<usize> {
    tracked
        .iter()
        .enumerate()
        .filter(|(_, process)| {
            process.key.session_id.as_deref() == Some(change.session_id)
                && process.key.mode != change.mode
        })
        .map(|(index, _)| index)
        .collect()
}

/// A caller's elapsed-budget request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeoutRequest {
    /// Use the provider's default.
    Default,
    /// No elapsed deadline (`timeoutMs: null`).
    NoDeadline,
    /// An explicit positive budget in milliseconds.
    Budget(u64),
}

/// The effective elapsed budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Timeout {
    /// No deadline.
    NoDeadline,
    /// A budget in milliseconds.
    Budget(u64),
}

/// Resolve one run's elapsed budget.
///
/// A supplied budget must be positive — zero is not a public
/// disable-deadline sentinel, `NoDeadline` is — and is capped at the
/// provider's maximum. The default is capped the same way, so a
/// misconfigured default cannot exceed the deployment's ceiling.
pub fn clamp_timeout(
    requested: TimeoutRequest,
    default_ms: u64,
    max_ms: u64,
    name: &str,
) -> Result<Timeout, PolicyError> {
    match requested {
        TimeoutRequest::Default => Ok(Timeout::Budget(default_ms.min(max_ms))),
        TimeoutRequest::NoDeadline => Ok(Timeout::NoDeadline),
        TimeoutRequest::Budget(0) => Err(PolicyError::InvalidTimeout {
            name: name.to_owned(),
        }),
        TimeoutRequest::Budget(budget) => Ok(Timeout::Budget(budget.min(max_ms))),
    }
}
