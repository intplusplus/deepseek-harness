//! The plan-mode state machine: selection, boundary flush, narration, and the
//! policy text.
//!
//! Ported from `PlanModeController` in `packages/plan/plan-mode/src/index.ts`,
//! minus the Cordis service and the agent-loop wiring. The machine owns two
//! kinds of state that the TypeScript controller keeps apart:
//!
//! - **Logged state** ([`PlanUnitState`]) — folded from the session log, so
//!   resume and fork restore it.
//! - **Pending selections** — process-local, not derived from the log. A
//!   selection made during an open turn waits for the next accepted in-turn
//!   pre-step; one made between turns commits immediately.
//!
//! ## Two-phase decisions
//!
//! The TypeScript code appends `plan/mode` inside `set` and `onBoundary`, then
//! deletes the pending intent only after the append returns. A failed durable
//! write therefore leaves the selection retryable instead of dropping it. Rust
//! has no session to append to here, so the machine splits each durable step:
//!
//! 1. [`PlanMachine::set`] / [`PlanMachine::decide_flush`] return what should be
//!    appended and what narration should follow.
//! 2. The caller appends the event.
//! 3. [`PlanMachine::commit_set`] / [`PlanMachine::commit_flush`] clear the
//!    pending intent and apply the logged mode locally.
//!
//! Skipping step 2 and step 3 leaves the intent pending, which is exactly the
//! retry shape the TypeScript tests pin for a failed append.
//!
//! The open-turn test is injected: [`PlanMachine::from_events`] reads it from
//! [`dsh_session::fold_turn_boundary`], while a caller with a live turn-boundary
//! projection can pass it to [`PlanMachine::new`].

#![forbid(unsafe_code)]

use serde_json::{Map, Value};

use dsh_llm::{ContentBlock, ExtensionSource, MessageId, MessageSource, UserMessage};
use dsh_session::{fold_turn_boundary, CoreEvent, SessionEvent};

use super::fold::{apply_plan_state, fold_plan_state, PlanFoldError};
use super::types::{PendingIntent, PlanProjection, PlanStatus, PlanUnitState};

/// Message-source kind of the plan-mode narration.
pub const PLAN_MODE_SOURCE_KIND: &str = "plan-mode";

/// Notice text when plan mode becomes active.
pub const NARRATION_PLAN_ON: &str = "The user switched this session to plan mode.";

/// Notice text when plan mode becomes inactive.
pub const NARRATION_PLAN_OFF: &str = "The user switched this session back to the default mode.";

/// A user-switch notice the caller injects as a user message.
///
/// The narration exists only when the last logged request header told the model
/// the other mode; before the first header the prompt section is the state
/// statement, so no notice is needed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Narration {
    /// The mode being switched to.
    pub active: bool,
    /// The one-sentence notice.
    pub text: String,
}

impl Narration {
    /// Build the model-visible user message for this notice.
    ///
    /// The source is `{ kind: 'plan-mode', form: 'notice', summary: text }`; the
    /// narration is already one sentence, so it is its own summary. `kind` is
    /// seeded first because [`ExtensionSource::new`] preserves an existing key's
    /// position, which keeps the wire order the TypeScript declaration uses.
    pub fn message(&self, id: MessageId) -> UserMessage {
        let mut fields = Map::new();
        fields.insert(
            "kind".to_string(),
            Value::String(PLAN_MODE_SOURCE_KIND.to_string()),
        );
        fields.insert("form".to_string(), Value::String("notice".to_string()));
        fields.insert("summary".to_string(), Value::String(self.text.clone()));
        UserMessage::with_source(
            id,
            vec![ContentBlock::text(self.text.clone())],
            MessageSource::Extension(ExtensionSource::new(PLAN_MODE_SOURCE_KIND, fields)),
        )
    }
}

/// What one `set` call decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanSetDecision {
    /// The requested mode already matches the current or pending target.
    Noop,
    /// The selection waits for the next accepted in-turn pre-step.
    Queued,
    /// The selection returned to the logged mode; any pending intent is cleared.
    Cancelled,
    /// The caller must append `plan/mode { active }`, then call
    /// [`PlanMachine::commit_set`].
    Commit {
        /// The mode to append.
        active: bool,
        /// The user-switch notice to inject after the append, if any.
        narration: Option<Narration>,
    },
}

impl PlanSetDecision {
    /// The outcome the command handler renders.
    pub fn outcome(&self) -> PlanSetOutcome {
        match self {
            Self::Noop => PlanSetOutcome::Noop,
            Self::Queued => PlanSetOutcome::Queued,
            Self::Cancelled => PlanSetOutcome::Cancelled,
            Self::Commit { .. } => PlanSetOutcome::Committed,
        }
    }

    /// The narration to inject after a successful commit.
    pub fn narration(&self) -> Option<&Narration> {
        match self {
            Self::Commit { narration, .. } => narration.as_ref(),
            Self::Noop | Self::Queued | Self::Cancelled => None,
        }
    }
}

/// What one `set` call did, without its narration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanSetOutcome {
    /// Logged now.
    Committed,
    /// Awaiting the next accepted in-turn pre-step.
    Queued,
    /// An opposite pending selection was cleared; the logged state already
    /// matches.
    Cancelled,
    /// Already in that state.
    Noop,
}

/// What one boundary flush decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlushDecision {
    /// The pending target already equals the logged mode: drop the intent.
    ClearPending {
        /// A notice to inject, when the pending selection asked for narration.
        narration: Option<Narration>,
    },
    /// Append `plan/mode { active }`, then call [`PlanMachine::commit_flush`].
    Append {
        /// The mode to append.
        active: bool,
        /// A notice to inject after the append, when the selection asked for it.
        narration: Option<Narration>,
    },
}

impl FlushDecision {
    /// The mode to append, or `None` when the flush only clears the intent.
    pub fn active(&self) -> Option<bool> {
        match self {
            Self::ClearPending { .. } => None,
            Self::Append { active, .. } => Some(*active),
        }
    }

    /// The notice to inject after the flush succeeds.
    pub fn narration(&self) -> Option<&Narration> {
        match self {
            Self::ClearPending { narration } | Self::Append { narration, .. } => narration.as_ref(),
        }
    }
}

/// The plan-mode state machine for one agent session.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanMachine {
    state: PlanUnitState,
    open_turn: bool,
    pending: Option<PendingIntent>,
}

impl PlanMachine {
    /// Build a machine from folded state and the caller's open-turn signal.
    pub fn new(state: PlanUnitState, open_turn: bool) -> Self {
        Self {
            state,
            open_turn,
            pending: None,
        }
    }

    /// Rebuild the machine from a session log.
    ///
    /// The pending selection is process-local, so a machine rebuilt from the log
    /// has none; the logged mode and the open-turn signal are exact.
    pub fn from_events(events: &[SessionEvent]) -> Result<Self, PlanFoldError> {
        let state = fold_plan_state(events)?;
        let open_turn = fold_turn_boundary(events).open_turn_start_seq.is_some();
        Ok(Self::new(state, open_turn))
    }

    /// Restore a process-local pending selection (a live controller reload).
    pub fn with_pending(mut self, pending: Option<PendingIntent>) -> Self {
        self.pending = pending;
        self
    }

    /// The folded unit state.
    pub fn state(&self) -> &PlanUnitState {
        &self.state
    }

    /// The logged mode.
    pub fn active(&self) -> bool {
        self.state.active
    }

    /// The mode recorded by the latest request header, if any.
    pub fn active_at_last_header(&self) -> Option<bool> {
        self.state.active_at_last_header
    }

    /// Whether a turn is open.
    pub fn is_turn_open(&self) -> bool {
        self.open_turn
    }

    /// The pending selection, if one is waiting.
    pub fn pending(&self) -> Option<PendingIntent> {
        self.pending
    }

    /// Logged mode plus the process-local pending target.
    pub fn status(&self) -> PlanStatus {
        PlanStatus {
            active: self.state.active,
            pending: self.pending.map(|pending| pending.active),
        }
    }

    /// The wire projection derived from the folded state alone.
    pub fn projection(&self) -> PlanProjection {
        super::fold::plan_projection(&self.state)
    }

    /// The `plan:policy` text for this agent.
    ///
    /// A pending selection wins over the logged mode, so the proposed next
    /// assembly already reflects the picker's optimistic state.
    pub fn policy_text<'a>(&self, section: &'a str) -> &'a str {
        policy_text(
            section,
            self.pending.map(|pending| pending.active),
            self.state.active,
        )
    }

    /// Select whether plan mode should be active.
    ///
    /// Between turns the decision is [`PlanSetDecision::Commit`]: no in-turn
    /// pre-step will run until another prompt starts a turn, so the caller
    /// appends immediately. During an open turn the selection is parked with
    /// narration and waits for the next accepted in-turn pre-step.
    ///
    /// A repeated selection of the current or already-pending state is a no-op.
    /// A selection that returns to the logged mode during an open turn parks a
    /// pending intent anyway and reports [`PlanSetOutcome::Cancelled`]; the next
    /// boundary clears it without appending. Between turns that same reversal
    /// clears the intent immediately.
    pub fn set(&mut self, active: bool) -> PlanSetDecision {
        let target = self
            .pending
            .map(|pending| pending.active)
            .unwrap_or(self.state.active);
        if active == target {
            return PlanSetDecision::Noop;
        }
        if self.open_turn {
            self.pending = Some(PendingIntent {
                active,
                narrate: true,
            });
            return if self.state.active == active {
                PlanSetDecision::Cancelled
            } else {
                PlanSetDecision::Queued
            };
        }
        if self.state.active == active {
            self.pending = None;
            return PlanSetDecision::Cancelled;
        }
        PlanSetDecision::Commit {
            active,
            narration: self.narration(active),
        }
    }

    /// Complete a [`PlanSetDecision::Commit`] after its append landed.
    ///
    /// Clears the pending intent and applies the appended `plan/mode` to the
    /// local fold, so [`PlanMachine::active`] reflects the commit immediately.
    /// Call this only after the durable append succeeded; a failed append leaves
    /// the previous intent in place for a retry.
    pub fn commit_set(&mut self, decision: &PlanSetDecision) {
        if let PlanSetDecision::Commit { active, .. } = decision {
            self.pending = None;
            self.state.active = *active;
            self.state.wanted = None;
        }
    }

    /// Decide the next accepted in-turn pre-step flush.
    ///
    /// `None` means nothing is pending. [`FlushDecision::ClearPending`] means the
    /// selection netted out to the logged mode, so no event is written.
    /// [`FlushDecision::Append`] means the caller appends `plan/mode { active }`
    /// and then calls [`PlanMachine::commit_flush`]; a failed append keeps the
    /// intent parked for the next boundary.
    pub fn decide_flush(&self) -> Option<FlushDecision> {
        let pending = self.pending?;
        let narration = self.narration(pending.active);
        let narration = if pending.narrate { narration } else { None };
        if pending.active == self.state.active {
            return Some(FlushDecision::ClearPending { narration });
        }
        Some(FlushDecision::Append {
            active: pending.active,
            narration,
        })
    }

    /// Complete a flush after its append (if any) landed.
    pub fn commit_flush(&mut self, decision: &FlushDecision) {
        self.pending = None;
        if let FlushDecision::Append { active, .. } = decision {
            self.state.active = *active;
            self.state.wanted = None;
        }
    }

    /// Fold one newly appended event into the machine's state.
    ///
    /// Pending selections are untouched: they are process-local and outlive the
    /// events that happen while they wait.
    pub fn observe(&mut self, event: &SessionEvent) -> Result<(), PlanFoldError> {
        let state = apply_plan_state(self.state.clone(), event)?;
        self.state = state;
        match event.core() {
            Some(CoreEvent::TurnStart { .. }) => self.open_turn = true,
            Some(CoreEvent::TurnEnd { .. }) => self.open_turn = false,
            _ => {}
        }
        Ok(())
    }

    /// Build a notice when the last logged header described the other mode.
    fn narration(&self, target: bool) -> Option<Narration> {
        let told = self.state.active_at_last_header?;
        if told == target {
            return None;
        }
        Some(Narration {
            active: target,
            text: if target {
                NARRATION_PLAN_ON
            } else {
                NARRATION_PLAN_OFF
            }
            .to_string(),
        })
    }
}

/// The `plan:policy` section text for one pending/logged pair.
///
/// A pending target wins over the logged mode; inactive agents and agent-less
/// assemblies render the empty string.
pub fn policy_text(section: &str, pending: Option<bool>, logged_active: bool) -> &str {
    if pending.unwrap_or(logged_active) {
        section
    } else {
        ""
    }
}
