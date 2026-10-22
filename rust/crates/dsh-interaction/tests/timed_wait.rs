//! The foreground wait: its deadline, its claims, and its cancellation.

use std::sync::Arc;
use std::time::Duration;

use dsh_interaction::questions::UserQuestionError;
use dsh_interaction::timed_wait::*;
use dsh_tools::cancel::CancelSignal;

fn clock() -> Arc<ManualClock> {
    Arc::new(ManualClock::new())
}

#[test]
fn a_wait_expires_at_its_deadline() {
    let clock = clock();
    let wait = TimedQuestionWait::new(clock.clone(), Duration::from_secs(5), None);

    clock.advance(Duration::from_millis(4_999));
    assert!(wait.advance(clock.now()).is_none());
    assert!(!wait.is_closed());

    clock.advance(Duration::from_millis(1));
    let reason = wait.advance(clock.now()).expect("the wait expires");
    assert_eq!(
        reason,
        WaitCloseReason::TimedOut(UserQuestionError::AskTimedOut)
    );
    assert!(wait.is_closed());
    assert!(wait.signal().is_cancelled());
    // The wait is not expired twice.
    assert!(wait.advance(clock.now()).is_none());
}

#[test]
fn a_claimed_wait_stays_open_past_the_deadline() {
    let clock = clock();
    let wait = TimedQuestionWait::new(clock.clone(), Duration::from_secs(5), None);
    let stream = CancelSignal::new();

    let first = wait.attach(&stream).expect("the wait may be claimed");
    let second = wait.attach(&stream).expect("a second claim is allowed");
    assert_eq!(wait.claim_count(), 2);

    clock.advance(Duration::from_secs(10));
    assert!(wait.advance(clock.now()).is_none());
    assert!(!wait.is_closed());

    // The last claim leaving resumes the original deadline.
    first.release();
    assert!(wait.advance(clock.now()).is_none());
    second.release();
    let reason = wait.advance(clock.now()).expect("the wait expires");
    assert_eq!(
        reason,
        WaitCloseReason::TimedOut(UserQuestionError::AskTimedOut)
    );
}

#[test]
fn the_original_remaining_duration_resumes_after_a_claim_leaves() {
    let clock = clock();
    let wait = TimedQuestionWait::new(clock.clone(), Duration::from_secs(5), None);
    let stream = CancelSignal::new();

    clock.advance(Duration::from_secs(1));
    assert_eq!(wait.remaining(), Duration::from_secs(4));

    let claim = wait.attach(&stream).expect("the wait may be claimed");
    clock.advance(Duration::from_secs(2));
    claim.release();

    clock.advance(Duration::from_millis(1_999));
    assert!(wait.advance(clock.now()).is_none());
    clock.advance(Duration::from_millis(1));
    assert!(wait.advance(clock.now()).is_some());
    assert!(wait.is_closed());
}

#[test]
fn a_claim_is_released_when_its_consumer_returns() {
    let clock = clock();
    let wait = TimedQuestionWait::new(clock.clone(), Duration::from_secs(5), None);
    let stream = CancelSignal::new();

    {
        let _claim = wait.attach(&stream).expect("the wait may be claimed");
        assert_eq!(wait.claim_count(), 1);
    }
    assert_eq!(wait.claim_count(), 0);

    clock.advance(Duration::from_secs(5));
    assert!(wait.advance(clock.now()).is_some());
}

#[test]
fn the_parent_cancellation_ends_the_claim_without_expiring_the_parent() {
    let clock = clock();
    let parent = CancelSignal::new();
    let wait = TimedQuestionWait::new(clock.clone(), Duration::from_secs(5), Some(parent.clone()));
    let stream = CancelSignal::new();
    let claim = wait.attach(&stream).expect("the wait may be claimed");

    parent.cancel();
    assert!(wait.is_closed());
    assert_eq!(wait.close_reason(), Some(WaitCloseReason::ParentCancelled));
    // The parent itself is not expired by the wait it owns.
    assert!(parent.is_cancelled());
    assert_eq!(wait.claim_count(), 0);
    drop(claim);
}

#[test]
fn a_wait_whose_parent_already_ended_is_closed_at_once() {
    let clock = clock();
    let parent = CancelSignal::new();
    parent.cancel();
    let wait = TimedQuestionWait::new(clock.clone(), Duration::from_secs(5), Some(parent));
    assert!(wait.is_closed());
    assert_eq!(wait.close_reason(), Some(WaitCloseReason::ParentCancelled));
}

#[test]
fn a_cancelled_or_expired_wait_cannot_be_claimed() {
    let clock = clock();
    let parent = CancelSignal::new();
    parent.cancel();
    let cancelled = TimedQuestionWait::new(clock.clone(), Duration::from_secs(5), Some(parent));
    let stream = CancelSignal::new();
    assert!(cancelled.attach(&stream).is_none());

    let live = TimedQuestionWait::new(clock.clone(), Duration::from_secs(5), None);
    let cancelled_stream = CancelSignal::new();
    cancelled_stream.cancel();
    assert!(live.attach(&cancelled_stream).is_none());

    clock.advance(Duration::from_secs(6));
    assert!(live.attach(&stream).is_none());
    assert!(live.is_closed());
}

#[test]
fn the_remaining_duration_is_reported_to_the_answer_ui() {
    let clock = clock();
    let wait = TimedQuestionWait::new(clock.clone(), Duration::from_secs(5), None);
    let stream = CancelSignal::new();

    clock.advance(Duration::from_secs(1));
    let claim = wait.attach(&stream).expect("the wait may be claimed");
    assert_eq!(claim.remaining(), Duration::from_secs(4));
    assert_eq!(
        wait.remaining_at(Duration::from_secs(3)),
        Duration::from_secs(2)
    );
    // The remaining duration never goes negative.
    assert_eq!(wait.remaining_at(Duration::from_secs(9)), Duration::ZERO);
}

#[test]
fn a_claim_whose_stream_ended_is_released() {
    let clock = clock();
    let wait = TimedQuestionWait::new(clock.clone(), Duration::from_secs(5), None);
    let stream = CancelSignal::new();
    let _claim = wait.attach(&stream).expect("the wait may be claimed");

    stream.cancel();
    let released = wait.release_cancelled_claims();
    assert_eq!(released.len(), 1);
    assert_eq!(wait.claim_count(), 0);
}

#[tokio::test]
async fn the_wait_resolves_when_the_parent_turn_is_cancelled() {
    let clock = clock();
    let parent = CancelSignal::new();
    let wait = TimedQuestionWait::new(
        clock.clone(),
        Duration::from_secs(3_600),
        Some(parent.clone()),
    );
    let handle = tokio::spawn(async move { wait.expired().await });
    parent.cancel();
    let reason = handle.await.unwrap();
    assert_eq!(reason, WaitCloseReason::ParentCancelled);
}
