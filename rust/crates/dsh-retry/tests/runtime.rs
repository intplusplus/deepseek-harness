//! The asynchronous retry wait and its cancellation source.

use std::time::Duration;

use dsh_retry::{CancelledDelay, RetryDelay, TokioDelay, WaitCancel};

#[tokio::test]
async fn tokio_delay_waits_and_reports_completion() {
    assert!(TokioDelay.wait(1.0).await);
}

#[tokio::test]
async fn a_cancelled_delay_reports_that_it_did_not_complete() {
    assert!(!CancelledDelay.wait(0.0).await);
}

/// A cancellation raised before the wait is observed immediately: the source
/// is a value, not a one-shot notification, so no wakeup is lost.
#[tokio::test]
async fn cancellation_raised_before_the_wait_is_observed() {
    let cancel = WaitCancel::new();
    cancel.cancel();
    assert!(cancel.is_cancelled());
    tokio::time::timeout(Duration::from_millis(50), cancel.cancelled())
        .await
        .expect("an already-cancelled source resolves at once");
}

#[tokio::test]
async fn cancellation_raised_while_waiting_wakes_the_wait() {
    let cancel = WaitCancel::new();
    let trigger = cancel.clone();
    let canceller = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        trigger.cancel();
    });
    tokio::time::timeout(Duration::from_secs(1), cancel.cancelled())
        .await
        .expect("the wait is woken by the cancellation");
    canceller.await.unwrap();
}
