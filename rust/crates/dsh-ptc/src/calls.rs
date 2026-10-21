//! Nested-call identity and the ordered commit lane.
//!
//! Ported from the per-run scheduler inside
//! `packages/core/tools/src/ptc.ts` (`createRunCodeTool`'s `execute`), and the
//! cancellation bookkeeping of `ptc-calls.ts`.
//!
//! A PTC program can call several tools at once, in any order, and the tools
//! settle in whatever order they finish. Two things must still be true of the
//! durable log:
//!
//! 1. **Sub-call ids are minted in submission order** — `<parent>:ptc:<n>`,
//!    numbered from one per run — so a reader can reconstruct the program's
//!    own call order without timestamps.
//! 2. **Settle records commit in submission order.** A fast call submitted
//!    later waits behind a slow one submitted earlier. Only *execution*
//!    overlaps; every ordered stage (the start append, prepare, the settle
//!    append) is serialized through one lane.
//!
//! [`DispatchLane`] is that lane as an explicit state machine. It owns no
//! futures: [`DispatchLane::poll`] returns the next thing the driver should
//! do, and the driver reports back with [`DispatchLane::settle`] and
//! [`DispatchLane::committed`]. That makes the ordering rules — the head-of-line
//! cursor, the parallel width, the exclusive barrier, and abandonment of queued
//! calls after an abort — testable without a runtime.
//!
//! ## Language check
//!
//! The TypeScript lane is driven by a `for (;;)` loop that awaits a wakeup
//! promise; this port is the same loop with the await removed. `poll` advances
//! exactly one step per call, which is what the driver's `continue` did.

#![forbid(unsafe_code)]

use std::collections::VecDeque;

use dsh_llm::ToolCallId;

/// The tool name the PTC presentation reserves for its own transport.
pub const RUN_CODE_NAME: &str = "run_code";

/// Whether a sub-call may overlap its siblings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Classification {
    /// May run beside sibling parallel calls, up to the run's width.
    Parallel,
    /// Runs alone and forms an ordering barrier.
    Exclusive,
}

/// Per-run sub-call id minting.
///
/// One counter per PTC run, so `<parent>:ptc:3` names the third call the
/// program made, matching the TypeScript `++dispatches`. The counter is not
/// shared between runs: two runs of the same outer call id therefore reuse
/// ids, which is why the durable record carries the enclosing call's range
/// rather than relying on global uniqueness.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubCallSequence {
    issued: u64,
}

impl SubCallSequence {
    /// A sequence that has not minted anything yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint the next id for `parent`.
    pub fn mint(&mut self, parent: &ToolCallId) -> ToolCallId {
        self.issued += 1;
        sub_call_id(parent, self.issued)
    }

    /// How many ids this sequence has minted.
    pub fn issued(&self) -> u64 {
        self.issued
    }
}

/// Format one sub-call id: `<parent>:ptc:<ordinal>`.
pub fn sub_call_id(parent: &ToolCallId, ordinal: u64) -> ToolCallId {
    ToolCallId::new(format!("{}:ptc:{}", parent.as_str(), ordinal))
}

/// Which phase of a run a synthetic "the run is over" message describes.
///
/// The three phases are distinct in the TypeScript source because the program
/// can observe them at three points: before a call is admitted, while a call
/// is queued behind the lane, and as a settled value is handed back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RunOverPhase {
    /// A call was refused before it was admitted.
    NotDispatched,
    /// A queued call was abandoned unstarted, so it never logged a start.
    Abandoned,
    /// A call settled, but its value was discarded because the run ended.
    Discarded,
}

/// The message a binding call sees when the enclosing run is already over.
///
/// Reproduces the TypeScript text exactly, including the parenthesized reason:
/// `run_code run is over (<reason>); <name> tool call abandoned`.
pub fn run_over_message(name: &str, reason: &str, phase: RunOverPhase) -> String {
    let suffix = match phase {
        RunOverPhase::NotDispatched => format!("{name} not dispatched"),
        RunOverPhase::Abandoned => format!("{name} tool call abandoned"),
        RunOverPhase::Discarded => format!("{name} result discarded"),
    };
    format!("{RUN_CODE_NAME} run is over ({reason}); {suffix}")
}

/// One queued or in-flight nested call.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    id: ToolCallId,
    name: String,
    /// The classification this entry was submitted with.
    submitted: Classification,
    /// The classification the lane started it under; `None` until it starts.
    started: Option<Classification>,
    /// Whether the dispatch body parked its outcome.
    settled: bool,
}

impl Entry {
    fn classification(&self) -> Classification {
        self.started.unwrap_or(self.submitted)
    }
}

/// What the driver should do next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaneStep {
    /// Start this call's dispatch stage now. The lane has already charged the
    /// slot, so the driver must eventually [`DispatchLane::settle`] it.
    Start {
        /// The sub-call id.
        id: ToolCallId,
        /// The classification it started under.
        classification: Classification,
    },
    /// Commit this call's settle record now: run post-execute, append
    /// `tool/ptc-dispatch`, then call [`DispatchLane::committed`].
    Commit {
        /// The sub-call id.
        id: ToolCallId,
        /// Whether the commit releases an exclusive barrier.
        classification: Classification,
    },
    /// Abandon this queued call: the run is over, so it never starts and
    /// logs nothing. Nothing to report back.
    Abandon {
        /// The sub-call id that was never started.
        id: ToolCallId,
        /// The program-facing reason the call was dropped.
        message: String,
    },
    /// Nothing can advance until a body settles or a call is submitted.
    Wait,
    /// Every queue and the pool are empty: the run has drained.
    Quiescent,
}

/// The ordered dispatch lane for one PTC run.
#[derive(Clone, Debug)]
pub struct DispatchLane {
    pending: VecDeque<Entry>,
    committing: VecDeque<Entry>,
    in_flight: usize,
    exclusive_active: bool,
    max_parallel: usize,
    abort_reason: Option<String>,
}

impl DispatchLane {
    /// A lane admitting at most `max_parallel` overlapping parallel calls.
    ///
    /// A width below one is treated as one, matching the TypeScript
    /// `Math.max(1, maxParallel)` boundary rather than admitting nothing.
    pub fn new(max_parallel: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            committing: VecDeque::new(),
            in_flight: 0,
            exclusive_active: false,
            max_parallel: max_parallel.max(1),
            abort_reason: None,
        }
    }

    /// The configured width.
    pub fn max_parallel(&self) -> usize {
        self.max_parallel
    }

    /// Queue one submitted call.
    pub fn submit(
        &mut self,
        id: ToolCallId,
        name: impl Into<String>,
        classification: Classification,
    ) {
        self.pending.push_back(Entry {
            id,
            name: name.into(),
            submitted: classification,
            started: None,
            settled: false,
        });
    }

    /// Re-classify a queued call before it starts.
    ///
    /// The TypeScript driver re-reads the registry's mode immediately before
    /// each start, so a registry change while a call waits can flip it to
    /// exclusive. A call that already started keeps the mode it started under.
    pub fn reclassify(&mut self, id: &ToolCallId, classification: Classification) -> bool {
        for entry in &mut self.pending {
            if entry.id == *id {
                entry.submitted = classification;
                return true;
            }
        }
        false
    }

    /// Mark the run over; queued calls are abandoned from here on.
    pub fn abort(&mut self, reason: impl Into<String>) {
        self.abort_reason = Some(reason.into());
    }

    /// The abort reason, when the run is over.
    pub fn abort_reason(&self) -> Option<&str> {
        self.abort_reason.as_deref()
    }

    /// How many bodies are running.
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// How many calls are waiting to start.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// How many settled calls are waiting to commit.
    pub fn committing(&self) -> usize {
        self.committing.len()
    }

    /// Advance the lane by one step.
    pub fn poll(&mut self) -> LaneStep {
        if let Some(front) = self.committing.front() {
            if front.settled {
                return LaneStep::Commit {
                    id: front.id.clone(),
                    classification: front.classification(),
                };
            }
        }

        if let Some(head) = self.pending.front() {
            if let Some(reason) = self.abort_reason.clone() {
                let entry = self.pending.pop_front().expect("checked above");
                return LaneStep::Abandon {
                    message: run_over_message(&entry.name, &reason, RunOverPhase::Abandoned),
                    id: entry.id,
                };
            }
            let classification = head.classification();
            let capacity = !self.exclusive_active
                && match classification {
                    Classification::Exclusive => self.in_flight == 0,
                    Classification::Parallel => self.in_flight < self.max_parallel,
                };
            if capacity {
                let mut entry = self.pending.pop_front().expect("checked above");
                entry.started = Some(classification);
                if classification == Classification::Exclusive {
                    self.exclusive_active = true;
                }
                self.in_flight += 1;
                let id = entry.id.clone();
                // Joined before the start returns, so the commit cursor sees
                // submission order; nothing commits it until `settled` flips.
                self.committing.push_back(entry);
                return LaneStep::Start { id, classification };
            }
        }

        if self.pending.is_empty() && self.committing.is_empty() && self.in_flight == 0 {
            return LaneStep::Quiescent;
        }
        LaneStep::Wait
    }

    /// Record that a started call parked its outcome.
    ///
    /// Returns `false` when `id` is not an in-flight call — a settle for a
    /// call the lane never started, which the host refuses as a protocol
    /// error rather than ignoring.
    pub fn settle(&mut self, id: &ToolCallId) -> bool {
        for entry in &mut self.committing {
            if entry.id == *id && !entry.settled {
                entry.settled = true;
                self.in_flight -= 1;
                return true;
            }
        }
        false
    }

    /// Record that the head-of-line settle record was appended.
    ///
    /// Returns `false` when `id` is not the settled head, which means the
    /// driver committed out of order — the invariant this lane exists to
    /// preserve, so the caller should treat it as a bug rather than retry.
    pub fn committed(&mut self, id: &ToolCallId) -> bool {
        match self.committing.front() {
            Some(entry) if entry.id == *id && entry.settled => {
                let entry = self.committing.pop_front().expect("checked above");
                if entry.classification() == Classification::Exclusive {
                    self.exclusive_active = false;
                }
                true
            }
            _ => false,
        }
    }
}
