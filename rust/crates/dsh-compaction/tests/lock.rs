//! Durable lock lifecycle integration: acquire, release, and lease-expiry are
//! observable through the public `compaction/start` and `compaction/end` events.

use dsh_compaction::{
    acquire_lock, inspect_lock, release_lock, CompactionEvent, CompactionId, CompactionStart,
    CompactionTurn, DEFAULT_LEASE_MS, LockState,
};
use dsh_session::{CoreEvent, SessionEvent};

/// Build a `compaction/start` extension event stamped with an acquisition time.
fn start_event(id: &str, acquired_at: Option<i64>) -> SessionEvent {
    let start = match acquired_at {
        Some(at) => CompactionStart::with_acquired_at(
            CompactionId::new(id),
            None,
            CompactionTurn::standalone(),
            at,
        ),
        None => CompactionStart::new(CompactionId::new(id), None, CompactionTurn::standalone()),
    };
    CompactionEvent::Start(start).encode().unwrap()
}

#[test]
fn lock_is_free_on_an_empty_log() {
    assert_eq!(inspect_lock(&[], 0, DEFAULT_LEASE_MS), LockState::Free);
}

#[test]
fn acquire_then_release_is_observable_as_free() {
    let start = acquire_lock(&[], true, DEFAULT_LEASE_MS, 1_000, "manual compaction").unwrap();
    assert!(start.turn.is_standalone());
    assert_eq!(start.acquired_at, Some(1_000));

    let end = release_lock(&start, None);
    assert_eq!(end.compaction_id, start.compaction_id);

    let events = vec![
        CompactionEvent::Start(start).encode().unwrap(),
        CompactionEvent::End(end).encode().unwrap(),
    ];
    assert_eq!(inspect_lock(&events, 2_000, DEFAULT_LEASE_MS), LockState::Free);
}

#[test]
fn held_lock_is_visible_while_lease_is_valid() {
    let events = vec![start_event("c1", Some(1_000))];
    let state = inspect_lock(&events, 1_000 + DEFAULT_LEASE_MS / 2, DEFAULT_LEASE_MS);
    match state {
        LockState::Held {
            compaction_id,
            expires_at,
            ..
        } => {
            assert_eq!(compaction_id, CompactionId::new("c1"));
            assert_eq!(expires_at, 1_000 + DEFAULT_LEASE_MS);
        }
        other => panic!("expected held lock, got {other:?}"),
    }
}

#[test]
fn lease_expiry_frees_a_crashed_lock() {
    let events = vec![start_event("c1", Some(1_000))];
    // Past the lease: observable as free again, so a new transaction may start.
    assert_eq!(
        inspect_lock(&events, 1_000 + DEFAULT_LEASE_MS, DEFAULT_LEASE_MS),
        LockState::Free
    );
}

#[test]
fn end_seed_after_unmatched_start_releases_the_lock() {
    let events = vec![
        start_event("c1", Some(1_000)),
        SessionEvent::Core(CoreEvent::EndSeed { inherited: false }),
    ];
    assert_eq!(inspect_lock(&events, 1_500, DEFAULT_LEASE_MS), LockState::Free);
}

#[test]
fn acquire_is_rejected_while_another_lock_is_held() {
    let events = vec![start_event("c1", Some(1_000))];
    let error = acquire_lock(&events, true, DEFAULT_LEASE_MS, 1_200, "manual compaction").unwrap_err();
    assert!(matches!(
        error,
        dsh_compaction::CompactionError::CompactionBusy { .. }
    ));
}

#[test]
fn lease_expiry_allows_a_fresh_acquire_after_crash() {
    let events = vec![start_event("c1", Some(0))];
    // Lease expired at t = lease; a new acquire is permitted.
    let start = acquire_lock(
        &events,
        true,
        DEFAULT_LEASE_MS,
        DEFAULT_LEASE_MS + 1,
        "manual compaction",
    )
    .unwrap();
    assert!(start.turn.is_standalone());
    assert_eq!(start.acquired_at, Some(DEFAULT_LEASE_MS + 1));
}
