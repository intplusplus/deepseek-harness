//! Retained-process selection, mode invalidation, and deadline clamping.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node/src/persistent.ts`,
//! `policy-processes.ts`, and `packages/util/timeout/src/index.ts`.

use dsh_ptc::{
    clamp_timeout, decide_process, invalidated_processes, retains_state, ModeChange, Owner,
    PolicyError, PolicyKey, ProcessDecision, RetainedProcess, SandboxMode, Timeout, TimeoutRequest,
};

fn key(mode: SandboxMode, root: &str, session: Option<&str>) -> PolicyKey {
    PolicyKey::new(mode, root, session.map(str::to_owned))
}

fn retained(key: PolicyKey, running: bool, cleanup_pending: bool, closed: bool) -> RetainedProcess {
    RetainedProcess {
        key,
        running,
        cleanup_pending,
        closed,
    }
}

#[test]
fn a_one_shot_run_never_consults_retained_state() {
    let requested = key(SandboxMode::WorkspaceWrite, "/work", Some("s1"));
    let busy = retained(requested.clone(), true, true, true);
    assert_eq!(
        decide_process(Owner::OneShot, Some(&busy), &requested),
        ProcessDecision::Spawn
    );
}

#[test]
fn a_retained_owner_starts_one_environment_then_reuses_it() {
    let requested = key(SandboxMode::ReadOnly, "/work", Some("s1"));
    assert_eq!(
        decide_process(Owner::Retained, None, &requested),
        ProcessDecision::Spawn
    );
    let live = retained(requested.clone(), false, false, false);
    assert_eq!(
        decide_process(Owner::Retained, Some(&live), &requested),
        ProcessDecision::Reuse
    );
}

#[test]
fn a_busy_or_cleaning_owner_is_refused_before_any_policy_comparison() {
    let requested = key(SandboxMode::ReadOnly, "/work", Some("s1"));
    // The retained environment was admitted under a different policy *and* is
    // busy: the idleness check owns the answer, exactly as `assertIdle` runs
    // before the comparison in the TypeScript `run`.
    let busy = retained(
        key(SandboxMode::DangerFullAccess, "/other", None),
        true,
        false,
        false,
    );
    assert_eq!(
        decide_process(Owner::Retained, Some(&busy), &requested),
        ProcessDecision::Reject(PolicyError::Busy)
    );

    let cleaning = retained(requested.clone(), false, true, false);
    assert_eq!(
        decide_process(Owner::Retained, Some(&cleaning), &requested),
        ProcessDecision::Reject(PolicyError::CleanupPending)
    );
}

#[test]
fn a_closed_environment_is_replaced_rather_than_refused() {
    let requested = key(SandboxMode::ReadOnly, "/work", Some("s1"));
    // A closed process holds no variables, so there is nothing to release even
    // though its policy differs.
    let closed = retained(
        key(SandboxMode::DangerFullAccess, "/other", None),
        false,
        false,
        true,
    );
    assert_eq!(
        decide_process(Owner::Retained, Some(&closed), &requested),
        ProcessDecision::Spawn
    );
}

#[test]
fn a_policy_change_must_be_released_first() {
    let base = key(SandboxMode::ReadOnly, "/work", Some("s1"));
    for changed in [
        key(SandboxMode::WorkspaceWrite, "/work", Some("s1")),
        key(SandboxMode::ReadOnly, "/other", Some("s1")),
        key(SandboxMode::ReadOnly, "/work", Some("s2")),
        key(SandboxMode::ReadOnly, "/work", None),
    ] {
        let live = retained(base.clone(), false, false, false);
        assert_eq!(
            decide_process(Owner::Retained, Some(&live), &changed),
            ProcessDecision::Reject(PolicyError::PolicyChange),
            "expected a refusal for {changed:?}"
        );
    }
}

#[test]
fn refusals_say_what_to_do_next() {
    assert_eq!(
        PolicyError::PolicyChange.to_string(),
        "persistent PTC: release the retained environment before changing its sandbox policy"
    );
    assert_eq!(
        PolicyError::Busy.to_string(),
        "persistent PTC does not accept concurrent runs for one state owner"
    );
}

#[test]
fn an_escalated_run_is_temporary() {
    // Same mode: the run keeps the owner's retained variables.
    assert!(retains_state(
        true,
        Some(SandboxMode::WorkspaceWrite),
        Some(SandboxMode::WorkspaceWrite)
    ));
    // An approved escalation runs in a fresh environment and discards its
    // declarations.
    assert!(!retains_state(
        true,
        Some(SandboxMode::DangerFullAccess),
        Some(SandboxMode::WorkspaceWrite)
    ));
    // No owner, no retention, whatever the policy says.
    assert!(!retains_state(
        false,
        Some(SandboxMode::ReadOnly),
        Some(SandboxMode::ReadOnly)
    ));
    // A runtime without file policy has neither mode on either side.
    assert!(retains_state(true, None, None));
}

#[test]
fn a_mode_change_invalidates_only_that_sessions_differing_processes() {
    let tracked = vec![
        retained(
            key(SandboxMode::ReadOnly, "/w", Some("s1")),
            false,
            false,
            false,
        ),
        retained(
            key(SandboxMode::ReadOnly, "/w", Some("s2")),
            false,
            false,
            false,
        ),
        // Already at the new mode: a re-commit of the same mode changes nothing.
        retained(
            key(SandboxMode::WorkspaceWrite, "/w", Some("s1")),
            false,
            false,
            false,
        ),
        // No session: never invalidated by a session's policy change.
        retained(key(SandboxMode::ReadOnly, "/w", None), false, false, false),
    ];
    assert_eq!(
        invalidated_processes(
            &tracked,
            ModeChange {
                session_id: "s1",
                mode: SandboxMode::WorkspaceWrite
            }
        ),
        vec![0]
    );
    assert_eq!(
        invalidated_processes(
            &tracked,
            ModeChange {
                session_id: "s3",
                mode: SandboxMode::WorkspaceWrite
            }
        ),
        Vec::<usize>::new()
    );
}

#[test]
fn the_effective_deadline_is_the_default_or_the_capped_request() {
    assert_eq!(
        clamp_timeout(TimeoutRequest::Default, 30_000, 120_000, "timeoutMs"),
        Ok(Timeout::Budget(30_000))
    );
    // A misconfigured default still cannot exceed the ceiling.
    assert_eq!(
        clamp_timeout(TimeoutRequest::Default, 300_000, 120_000, "timeoutMs"),
        Ok(Timeout::Budget(120_000))
    );
    assert_eq!(
        clamp_timeout(TimeoutRequest::Budget(5_000), 30_000, 120_000, "timeoutMs"),
        Ok(Timeout::Budget(5_000))
    );
    assert_eq!(
        clamp_timeout(
            TimeoutRequest::Budget(300_000),
            30_000,
            120_000,
            "timeoutMs"
        ),
        Ok(Timeout::Budget(120_000))
    );
    assert_eq!(
        clamp_timeout(TimeoutRequest::NoDeadline, 30_000, 120_000, "timeoutMs"),
        Ok(Timeout::NoDeadline)
    );
}

#[test]
fn zero_is_not_a_disable_sentinel() {
    let error = clamp_timeout(
        TimeoutRequest::Budget(0),
        30_000,
        120_000,
        "persistent PTC timeoutMs",
    )
    .expect_err("zero is refused");
    assert_eq!(
        error,
        PolicyError::InvalidTimeout {
            name: "persistent PTC timeoutMs".into()
        }
    );
    assert_eq!(
        error.to_string(),
        "persistent PTC timeoutMs must be a positive finite number"
    );
}
