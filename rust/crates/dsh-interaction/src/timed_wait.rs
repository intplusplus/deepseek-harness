//! The foreground question wait: its lifetime, its deadline, and the Client
//! claims that suspend it.
//!
//! Ported from `packages/interaction/user-questions/src/timed-wait.ts`.
//!
//! ## The clock is injected
//!
//! The TypeScript wait reads `Date.now()` and arms a `setTimeout`. Rust reads
//! neither the process clock nor the environment: the deadline is compared
//! against an injected [`Clock`], so a test can advance time deterministically
//! and assert the exact remaining duration.
//!
//! ## The rules are pure; the waiting is the async half
//!
//! The wait itself is a **state machine**: claiming, releasing, expiring, and
//! closing are synchronous transitions over the injected clock. The
//! asynchronous parts are only the two waits — until the deadline
//! ([`TimedQuestionWait::expired`]) and until a claim ends
//! ([`WaitClaim::wait_until_released`]).
//!
//! ## The rules the port preserves
//!
//! - A wait expires at its deadline **only while no claim holds it**: an
//!   answer UI that holds the wait keeps it open past the Host deadline, and
//!   the original remaining duration resumes when the last claim leaves.
//! - A claim on a cancelled or expired wait yields nothing.
//! - The parent turn's cancellation closes the wait with the parent's reason
//!   and ends every claim stream, without expiring the parent itself.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::Poll;
use std::time::Duration;

use dsh_tools::cancel::CancelSignal;

use crate::questions::UserQuestionError;

/// The monotonic clock the timed wait reads.
///
/// Injected so tests can advance time deterministically; the production
/// implementation reads the system monotonic clock.
pub trait Clock: Send + Sync {
    /// The current reading of the clock, as a duration since an arbitrary
    /// origin. Only differences between readings are ever used.
    fn now(&self) -> Duration;
}

/// The production clock: the system monotonic clock.
///
/// The origin is fixed once per process, so two readings are comparable.
#[derive(Clone, Copy, Debug, Default)]
pub struct MonotonicClock;

impl MonotonicClock {
    /// The process-wide origin every reading is measured from.
    fn origin() -> std::time::Instant {
        static ORIGIN: OnceLock<std::time::Instant> = OnceLock::new();
        *ORIGIN.get_or_init(std::time::Instant::now)
    }
}

impl Clock for MonotonicClock {
    fn now(&self) -> Duration {
        std::time::Instant::now().duration_since(Self::origin())
    }
}

/// A clock a test drives by hand.
#[derive(Debug)]
pub struct ManualClock {
    reading: Mutex<Duration>,
}

impl ManualClock {
    /// A clock reading zero.
    pub fn new() -> Self {
        Self {
            reading: Mutex::new(Duration::ZERO),
        }
    }

    /// Move the clock forward by `delta`.
    pub fn advance(&self, delta: Duration) {
        let mut reading = self.lock();
        *reading += delta;
    }

    /// Move the clock to an absolute reading.
    pub fn set(&self, value: Duration) {
        let mut reading = self.lock();
        *reading = value;
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Duration> {
        self.reading
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

impl Default for ManualClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Duration {
        *self.lock()
    }
}

/// Why a foreground wait ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WaitCloseReason {
    /// The deadline passed with no claim holding the wait.
    TimedOut(UserQuestionError),
    /// The calling turn was cancelled.
    ParentCancelled,
    /// The wait was disposed.
    Disposed,
}

/// One live foreground wait, counted by the Host only while no answer UI holds
/// it.
pub struct TimedQuestionWait {
    clock: Arc<dyn Clock>,
    deadline: Duration,
    parent: Option<CancelSignal>,
    timeout: UserQuestionError,
    signal: CancelSignal,
    claims: Mutex<HashMap<u64, ClaimEntry>>,
    next_claim: AtomicU64,
    closed: Mutex<Option<WaitCloseReason>>,
}

struct ClaimEntry {
    /// The consumer's own cancellation lifetime; when it fires, the claim is
    /// released.
    stream: CancelSignal,
}

impl TimedQuestionWait {
    /// A wait that expires `timeout` after `clock.now()`.
    ///
    /// `parent` is the calling turn's cancellation signal; when it fires, the
    /// wait closes with [`WaitCloseReason::ParentCancelled`] and every claim
    /// stream ends.
    pub fn new(
        clock: Arc<dyn Clock>,
        timeout: Duration,
        parent: Option<CancelSignal>,
    ) -> Arc<Self> {
        Self::with_timeout_error(clock, timeout, parent, UserQuestionError::AskTimedOut)
    }

    /// A wait that expires with a caller-supplied business error.
    pub fn with_timeout_error(
        clock: Arc<dyn Clock>,
        timeout: Duration,
        parent: Option<CancelSignal>,
        timeout_error: UserQuestionError,
    ) -> Arc<Self> {
        let deadline = clock.now() + timeout;
        let wait = Arc::new(Self {
            clock,
            deadline,
            parent,
            timeout: timeout_error,
            signal: CancelSignal::new(),
            claims: Mutex::new(HashMap::new()),
            next_claim: AtomicU64::new(0),
            closed: Mutex::new(None),
        });
        if let Some(parent) = &wait.parent {
            if parent.is_cancelled() {
                wait.close(WaitCloseReason::ParentCancelled);
            }
        }
        wait
    }

    /// The cancellation shared with the foreground answerer.
    pub fn signal(&self) -> &CancelSignal {
        &self.signal
    }

    /// The deadline, as a clock reading.
    pub fn deadline(&self) -> Duration {
        self.deadline
    }

    /// The current clock reading.
    pub fn now(&self) -> Duration {
        self.clock.now()
    }

    /// How much of the wait is left, never negative.
    pub fn remaining(&self) -> Duration {
        self.remaining_at(self.clock.now())
    }

    /// How much of the wait is left at `now`, never negative.
    pub fn remaining_at(&self, now: Duration) -> Duration {
        self.deadline.saturating_sub(now)
    }

    /// Whether the deadline passed.
    pub fn is_expired(&self) -> bool {
        self.is_expired_at(self.clock.now())
    }

    /// Whether the deadline passed at `now`.
    pub fn is_expired_at(&self, now: Duration) -> bool {
        now >= self.deadline
    }

    /// Whether the wait has ended, for any reason.
    pub fn is_closed(&self) -> bool {
        self.close_reason().is_some()
    }

    /// Why the wait ended, when it did.
    pub fn close_reason(&self) -> Option<WaitCloseReason> {
        self.sync_parent();
        self.lock_closed().clone()
    }

    /// How many claims currently hold the wait.
    pub fn claim_count(&self) -> usize {
        self.sync_parent();
        self.lock_claims().len()
    }

    /// Close the wait when the calling turn was cancelled.
    ///
    /// The TypeScript wait registers an `abort` listener on the parent signal
    /// in its constructor, so the parent cancellation closes the wait at once.
    /// `dsh_tools::cancel::CancelSignal` offers no listener registration, so
    /// the port observes the parent lazily: every read of the wait's state
    /// reconciles the parent first. The effect is the same — the wait is
    /// closed with [`WaitCloseReason::ParentCancelled`], its claims are
    /// released, and its own signal is raised — but the reconciliation happens
    /// when the state is read rather than when the parent is cancelled.
    fn sync_parent(&self) {
        if self.lock_closed().is_some() {
            return;
        }
        let parent_cancelled = self
            .parent
            .as_ref()
            .is_some_and(|parent| parent.is_cancelled());
        if parent_cancelled {
            self.close(WaitCloseReason::ParentCancelled);
        }
    }

    /// Advance the wait to `now`, expiring it when the deadline passed and no
    /// claim holds it.
    ///
    /// Returns the reason when the wait expired on this call. A wait that
    /// already ended is not expired twice.
    pub fn advance(&self, now: Duration) -> Option<WaitCloseReason> {
        if self.is_closed() {
            return None;
        }
        if !self.is_expired_at(now) {
            return None;
        }
        if self.claim_count() > 0 {
            // An answer UI holds the wait: it stays open past the Host
            // deadline, and the original remaining duration resumes when the
            // last claim leaves.
            return None;
        }
        self.close(WaitCloseReason::TimedOut(self.timeout.clone()));
        self.close_reason()
    }

    /// Hold the wait for one answer UI until its stream closes or the question
    /// settles.
    ///
    /// Returns the claim when the wait may be claimed, and `None` when the
    /// wait is already cancelled, already expired, or the consumer's own
    /// signal is cancelled.
    pub fn attach(self: &Arc<Self>, stream: &CancelSignal) -> Option<WaitClaim> {
        self.sync_parent();
        let now = self.clock.now();
        if stream.is_cancelled() || self.signal.is_cancelled() {
            return None;
        }
        if self.claim_count() == 0 && self.is_expired_at(now) {
            self.close(WaitCloseReason::TimedOut(self.timeout.clone()));
            return None;
        }
        let id = self.next_claim.fetch_add(1, Ordering::SeqCst);
        self.lock_claims().insert(
            id,
            ClaimEntry {
                stream: stream.clone(),
            },
        );
        Some(WaitClaim {
            id,
            wait: Arc::clone(self),
        })
    }

    /// Release one claim.
    pub fn release(&self, id: u64) {
        self.lock_claims().remove(&id);
    }

    /// Release the claims whose consumer stream was cancelled.
    ///
    /// Returns the released claim ids.
    pub fn release_cancelled_claims(&self) -> Vec<u64> {
        let mut claims = self.lock_claims();
        let cancelled = claims
            .iter()
            .filter(|(_, entry)| entry.stream.is_cancelled())
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        for id in &cancelled {
            claims.remove(id);
        }
        cancelled
    }

    /// Release the timers, the parent cancellation, and every claim.
    pub fn close(&self, reason: WaitCloseReason) {
        {
            let mut closed = self.lock_closed();
            if closed.is_some() {
                return;
            }
            *closed = Some(reason);
        }
        self.lock_claims().clear();
        self.signal.cancel();
    }

    /// The async half of the deadline: resolve when the deadline passes or
    /// the calling turn is cancelled, whichever comes first.
    ///
    /// The deadline is measured against real time, so this helper pairs with
    /// [`MonotonicClock`]; a [`ManualClock`] drives the pure state machine
    /// instead.
    pub async fn expired(&self) -> WaitCloseReason {
        let remaining = self.remaining();
        match tokio::time::timeout(remaining, self.parent_cancelled()).await {
            Ok(()) => WaitCloseReason::ParentCancelled,
            Err(_) => WaitCloseReason::TimedOut(self.timeout.clone()),
        }
    }

    /// The async half of the wait's end: resolve when the wait closes.
    pub async fn closed(&self) -> WaitCloseReason {
        self.signal.cancelled().await;
        self.close_reason().unwrap_or(WaitCloseReason::Disposed)
    }

    async fn parent_cancelled(&self) {
        match &self.parent {
            Some(parent) => parent.cancelled().await,
            None => std::future::pending().await,
        }
    }

    fn lock_claims(&self) -> std::sync::MutexGuard<'_, HashMap<u64, ClaimEntry>> {
        self.claims
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn lock_closed(&self) -> std::sync::MutexGuard<'_, Option<WaitCloseReason>> {
        self.closed
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

/// One answer UI's hold on a foreground wait.
///
/// Dropping the claim releases it, exactly like the TypeScript generator's
/// `finally { release() }`.
pub struct WaitClaim {
    id: u64,
    wait: Arc<TimedQuestionWait>,
}

impl WaitClaim {
    /// The claim's identity inside its wait.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The wait this claim holds.
    pub fn wait(&self) -> &Arc<TimedQuestionWait> {
        &self.wait
    }

    /// How much of the wait is left, never negative.
    pub fn remaining(&self) -> Duration {
        self.wait.remaining()
    }

    /// Release the claim; the wait's deadline resumes.
    pub fn release(&self) {
        self.wait.release(self.id);
    }

    /// The async half of the claim: resolve when the consumer's stream ends or
    /// the wait itself ends, whichever comes first.
    ///
    /// The claim is released either way, so the wait's deadline resumes as
    /// soon as the last claim leaves.
    pub async fn wait_until_released(&self, stream: &CancelSignal) -> ClaimEnd {
        let stream_end = stream.cancelled();
        let wait_end = self.wait.signal().cancelled();
        let outcome = race(stream_end, wait_end).await;
        self.release();
        match outcome {
            RaceOutcome::Left => ClaimEnd::StreamEnded,
            RaceOutcome::Right => ClaimEnd::WaitEnded(
                self.wait
                    .close_reason()
                    .unwrap_or(WaitCloseReason::Disposed),
            ),
        }
    }
}

impl Drop for WaitClaim {
    fn drop(&mut self) {
        self.release();
    }
}

/// Why a claim ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClaimEnd {
    /// The consumer's stream closed.
    StreamEnded,
    /// The wait itself ended.
    WaitEnded(WaitCloseReason),
}

/// Which side of a [`race`] finished first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RaceOutcome {
    /// The left future finished.
    Left,
    /// The right future finished.
    Right,
}

/// Resolve whichever of two futures finishes first.
///
/// A hand-rolled race, so the crate needs no `select!` macro and no futures
/// crate: `Pin<Box<_>>` is always `Unpin`, so polling both sides needs no
/// unsafe code.
async fn race<A: Future<Output = ()>, B: Future<Output = ()>>(left: A, right: B) -> RaceOutcome {
    let mut left = Box::pin(left);
    let mut right = Box::pin(right);
    std::future::poll_fn(move |context| {
        if let Poll::Ready(()) = left.as_mut().poll(context) {
            return Poll::Ready(RaceOutcome::Left);
        }
        if let Poll::Ready(()) = right.as_mut().poll(context) {
            return Poll::Ready(RaceOutcome::Right);
        }
        Poll::Pending
    })
    .await
}
