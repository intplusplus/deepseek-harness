//! Durable compaction lock lifecycle, persisted through the existing
//! `compaction/start` and `compaction/end` extension events.
//!
//! The lock is a small state machine: `acquire` writes `compaction/start`
//! (holding the durable lock), `release` writes `compaction/end`, and the
//! current state is a pure fold over the log. Because a crash can leave an
//! unmatched `compaction/start` with no `compaction/end`, the lock carries a
//! lease timestamp (`CompactionStart::acquired_at`); once the lease expires the
//! lock is observable as free again, so a later transaction may proceed without
//! a manual recovery step.
//!
//! This follows the `dsh-retry` extension-slot pattern: no core variant is
//! added, the lock markers are ordinary payload types reaching the log through
//! `SessionEvent::Extension`, and their decode rejects unknown fields.

#![forbid(unsafe_code)]

use dsh_session::{CoreEvent, SessionEvent};

use super::brand::CompactionId;
use super::error::CompactionError;
use super::events::{CompactionEnd, CompactionEvent, CompactionStart, CompactionTurn};

/// Default lease before a held-but-unreleased lock is observable as free, in
/// milliseconds. Thirty minutes covers a process restart without letting a
/// stuck lock block compaction indefinitely.
pub const DEFAULT_LEASE_MS: i64 = 30 * 60 * 1000;

/// The observable state of the durable compaction lock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LockState {
    /// No active compaction transaction.
    Free,
    /// A transaction holds the lock.
    Held {
        /// Identity of the holding transaction.
        compaction_id: CompactionId,
        /// Owning turn, or `None` for a standalone manual transaction.
        turn: Option<u64>,
        /// Epoch milliseconds at which the lock was acquired.
        acquired_at: i64,
        /// Epoch milliseconds at which the lease expires.
        expires_at: i64,
    },
}

/// Inspect the current lock state from the durable log.
///
/// The scan is backward: the first `compaction/start` decides. If a later
/// `compaction/end` or `session/end-seed` boundary follows it, the transaction
/// is matched (or belongs to an earlier session lifecycle) and the lock is
/// free. Otherwise an unmatched start holds the lock, unless its lease has
/// expired — then it is observable as free, so a new transaction may acquire.
pub fn inspect_lock(events: &[SessionEvent], now: i64, lease_ms: i64) -> LockState {
    let mut saw_end_or_seed = false;
    for event in events.iter().rev() {
        if matches!(event, SessionEvent::Core(CoreEvent::EndSeed { .. })) {
            saw_end_or_seed = true;
            continue;
        }
        if let Some(Ok(CompactionEvent::End(_))) = CompactionEvent::decode(event) {
            saw_end_or_seed = true;
            continue;
        }
        if let Some(Ok(CompactionEvent::Start(start))) = CompactionEvent::decode(event) {
            if saw_end_or_seed {
                return LockState::Free;
            }
            return match start.acquired_at {
                Some(acquired_at) => {
                    let expires_at = acquired_at.saturating_add(lease_ms);
                    if now >= expires_at {
                        LockState::Free
                    } else {
                        LockState::Held {
                            compaction_id: start.compaction_id,
                            turn: start.turn.get(),
                            acquired_at,
                            expires_at,
                        }
                    }
                }
                // No lease recorded: treat the lock as held without auto-expiry.
                None => LockState::Held {
                    compaction_id: start.compaction_id,
                    turn: start.turn.get(),
                    acquired_at: 0,
                    expires_at: i64::MAX,
                },
            };
        }
    }
    LockState::Free
}

/// Acquire the durable lock, returning the `compaction/start` event to append.
///
/// The durable-lock check runs first (matching the TypeScript transaction entry
/// order), then the owner is resolved: a manual transaction must run between
/// turns (`turn: null`); an automatic transaction must be enclosed by the open
/// turn. On success the returned event is stamped with the acquisition time so
/// lease-expiry is observable on replay.
pub fn acquire_lock(
    events: &[SessionEvent],
    manual: bool,
    lease_ms: i64,
    now: i64,
    stage: &str,
) -> Result<CompactionStart, CompactionError> {
    let mut state = super::state::inspect_compaction_entry_state(events)?;
    // A stale marker whose lease has expired (`now >= acquiredAt + lease`) no
    // longer blocks: that is what lets a crashed transaction be superseded
    // without a manual recovery step. `assert_compaction_inactive` stays the
    // authority for live markers and for `session/end-seed` boundaries.
    if state.unmatched_compaction_start.is_some() {
        if let LockState::Free = inspect_lock(events, now, lease_ms) {
            state.unmatched_compaction_start = None;
        }
    }
    let turn = super::state::plan_compaction_owner_from_state(&state, manual, stage)?;
    Ok(CompactionStart::with_acquired_at(
        CompactionId::random(),
        None,
        CompactionTurn::new(turn),
        now,
    ))
}

/// Release the durable lock, returning the `compaction/end` event to append.
///
/// The end event echoes the start's identity and owning turn so a pure replay
/// can pair it; `error` records a failed attempt when one occurred.
pub fn release_lock(start: &CompactionStart, error: Option<String>) -> CompactionEnd {
    CompactionEnd::new(
        start.compaction_id.clone(),
        start.source_command_id.clone(),
        start.turn,
        error,
    )
}

#[cfg(test)]
mod tests {
    use super::super::events::CompactionTurn;
    use super::*;

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

    fn end_event(id: &str) -> SessionEvent {
        CompactionEvent::End(CompactionEnd::new(
            CompactionId::new(id),
            None,
            CompactionTurn::standalone(),
            None,
        ))
        .encode()
        .unwrap()
    }

    #[test]
    fn free_when_no_markers() {
        assert_eq!(inspect_lock(&[], 1_000, DEFAULT_LEASE_MS), LockState::Free);
    }

    #[test]
    fn held_while_lease_is_valid() {
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
            other => panic!("expected held, got {other:?}"),
        }
    }

    #[test]
    fn free_once_lease_expires() {
        let events = vec![start_event("c1", Some(1_000))];
        assert_eq!(
            inspect_lock(&events, 1_000 + DEFAULT_LEASE_MS, DEFAULT_LEASE_MS),
            LockState::Free
        );
    }

    #[test]
    fn free_when_start_is_matched_by_end() {
        let events = vec![start_event("c1", Some(1_000)), end_event("c1")];
        assert_eq!(
            inspect_lock(&events, 1_500, DEFAULT_LEASE_MS),
            LockState::Free
        );
    }

    #[test]
    fn free_when_end_seed_follows_unmatched_start() {
        let events = vec![
            start_event("c1", Some(1_000)),
            SessionEvent::Core(CoreEvent::EndSeed { inherited: false }),
        ];
        assert_eq!(
            inspect_lock(&events, 1_100, DEFAULT_LEASE_MS),
            LockState::Free
        );
    }

    #[test]
    fn acquire_and_release_round_trip() {
        let start = acquire_lock(&[], true, DEFAULT_LEASE_MS, 5_000, "manual compaction").unwrap();
        assert_eq!(start.acquired_at, Some(5_000));
        assert!(start.turn.is_standalone());
        let end = release_lock(&start, None);
        assert_eq!(end.compaction_id, start.compaction_id);

        let events = vec![
            CompactionEvent::Start(start).encode().unwrap(),
            CompactionEvent::End(end).encode().unwrap(),
        ];
        assert_eq!(
            inspect_lock(&events, 9_000, DEFAULT_LEASE_MS),
            LockState::Free
        );
    }

    #[test]
    fn acquire_rejects_an_active_lock() {
        let events = vec![start_event("c1", Some(1_000))];
        let error =
            acquire_lock(&events, true, DEFAULT_LEASE_MS, 1_200, "manual compaction").unwrap_err();
        assert!(matches!(error, CompactionError::CompactionBusy { .. }));
    }

    #[test]
    fn acquire_rejects_manual_during_open_turn() {
        let events = vec![
            SessionEvent::turn_start(3),
            start_event("c1", Some(1_000)),
            end_event("c1"),
        ];
        // `compaction/README.md`: a manual bracket requires **no** open turn and
        // stamps `turn: null`, so an open turn is rejected even though the
        // previous transaction's lock is already matched.
        let error =
            acquire_lock(&events, true, DEFAULT_LEASE_MS, 2_000, "manual compaction").unwrap_err();
        assert!(matches!(error, CompactionError::ManualTurnOpen));
    }
}
