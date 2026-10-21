//! The cancellation signal shared by one step's tool calls.

#![forbid(unsafe_code)]

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// A shared, cloneable cancellation flag.
///
/// One step hands the same signal to every tool it dispatches. Cancellation is
/// one-way and sticky: once raised it is never lowered, so a scheduler that
/// observes it may stop dispatching without racing a resume.
#[derive(Clone, Debug, Default)]
pub struct CancelSignal {
    cancelled: Arc<AtomicBool>,
}

impl CancelSignal {
    /// A signal that is not yet cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Raise cancellation. Idempotent.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been raised.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}
