//! The cancellation signal shared by one step's tool calls.

#![forbid(unsafe_code)]

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use tokio::sync::Notify;

/// A shared, cloneable cancellation flag.
///
/// One step hands the same signal to every tool it dispatches. Cancellation is
/// one-way and sticky: once raised it is never lowered, so a scheduler that
/// observes it may stop dispatching without racing a resume.
#[derive(Clone, Debug, Default)]
pub struct CancelSignal {
    cancelled: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl CancelSignal {
    /// A signal that is not yet cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Raise cancellation. Idempotent.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    /// Whether cancellation has been raised.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Resolve once cancellation has been raised.
    ///
    /// If the signal is already cancelled this returns immediately; otherwise it
    /// awaits the next `cancel()`. It is safe to `await` from several tasks and
    /// to drop the returned future at any point. An in-flight approval can race
    /// this future so that raising the signal reconciles the pending request as
    /// cancelled (see `crate::approval`).
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let notified = self.notify.notified();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}
