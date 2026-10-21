//! The `/plan` command: metadata, copy, and the pure handler decision.
//!
//! Ported from the `ctx.inject(['commands'], ...)` child in
//! `packages/plan/plan-mode/src/index.ts`. The command is a user-facing switch:
//! `/plan off` leaves plan mode, `/plan <message>` enters it and steers the
//! message into the next turn, and `/plan` alone enters silently.
//!
//! The handler is pure. It reads the trimmed raw input and whether attachments
//! accompany the command, mutates the machine's process-local selection through
//! [`PlanMachine::set`], and returns the copy plus the optional steer text. When
//! [`PlanSetDecision::Commit`] comes back, the caller appends `plan/mode` and
//! then calls [`PlanMachine::commit_set`]; the copy is already the committed
//! one.

#![forbid(unsafe_code)]

use serde::Serialize;

use super::machine::{PlanMachine, PlanSetDecision, PlanSetOutcome};
use super::text::js_trim;

/// The command's definition identity.
pub const PLAN_COMMAND_DEFINITION_ID: &str = "@deepseek-ai/dsh-plan-mode";

/// The command's name, without the slash.
pub const PLAN_COMMAND_NAME: &str = "plan";

/// The command's one-line description.
pub const PLAN_COMMAND_DESCRIPTION: &str = "Enter or leave plan mode";

/// The command's input hint.
pub const PLAN_COMMAND_HINT: &str = "[off|message]";

/// Copy when an idle session enters plan mode.
pub const PLAN_ON_TEXT: &str = "Plan mode on. Use /plan off to leave.";

/// Copy when an open-turn session queues plan-mode entry.
pub const PLAN_ENTERING_TEXT: &str =
    "Entering plan mode (applies from the next step). Use /plan off to leave.";

/// Copy when an idle session leaves plan mode.
pub const PLAN_OFF_TEXT: &str = "Plan mode off.";

/// Copy when an open-turn session queues plan-mode exit.
pub const PLAN_LEAVING_TEXT: &str = "Leaving plan mode (applies from the next step).";

/// Copy when a queued entry is cancelled by a later `/plan off`.
pub const PLAN_ENTRY_CANCELLED_TEXT: &str = "Plan mode entry cancelled.";

/// Copy when `/plan off` finds no mode to leave.
pub const PLAN_ALREADY_INACTIVE_TEXT: &str = "Plan mode is already inactive.";

/// Copy when attachments accompany `/plan off`.
pub const PLAN_ATTACHMENTS_OFF_TEXT: &str = "Attachments cannot accompany /plan off.";

/// The command's input declaration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanCommandInput {
    /// The raw-input hint shown by a UI.
    pub hint: String,
    /// Whether the command accepts attachments.
    pub attachments: bool,
}

/// The command as the registry lists it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanCommandMetadata {
    /// The definition identity.
    pub definition_id: String,
    /// The command name.
    pub name: String,
    /// The one-line description.
    pub description: String,
    /// The input declaration.
    pub input: PlanCommandInput,
}

/// Build the command's registry metadata.
pub fn command_metadata() -> PlanCommandMetadata {
    PlanCommandMetadata {
        definition_id: PLAN_COMMAND_DEFINITION_ID.to_string(),
        name: PLAN_COMMAND_NAME.to_string(),
        description: PLAN_COMMAND_DESCRIPTION.to_string(),
        input: PlanCommandInput {
            hint: PLAN_COMMAND_HINT.to_string(),
            attachments: true,
        },
    }
}

/// Whether a command handler succeeded or failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PlanCommandKind {
    /// The handler returned a success.
    Success,
    /// The handler refused the input.
    Error,
}

/// The handler's settled result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanCommandResult {
    /// Success or error.
    pub kind: PlanCommandKind,
    /// The copy shown to the user.
    pub text: String,
}

impl PlanCommandResult {
    /// A success carrying `text`.
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            kind: PlanCommandKind::Success,
            text: text.into(),
        }
    }

    /// An error carrying `text`.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            kind: PlanCommandKind::Error,
            text: text.into(),
        }
    }
}

/// One `/plan` handler decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanCommandOutcome {
    /// The copy the command executor returns.
    pub result: PlanCommandResult,
    /// Text to steer into the next turn.
    ///
    /// `None` means do not steer. `Some("")` means attachments alone accompany
    /// the command: the caller builds the user message from its attachment
    /// blocks and appends no text block.
    pub steer: Option<String>,
    /// The selection decision; a [`PlanSetDecision::Commit`] still needs its
    /// append and [`PlanMachine::commit_set`].
    pub set: PlanSetDecision,
}

/// Run the pure `/plan` handler against one machine.
///
/// `has_attachments` stands in for the TypeScript `attachments.length > 0`:
/// attachments ride along with the message, and `/plan off` refuses them
/// outright.
pub fn run_plan_command(
    machine: &mut PlanMachine,
    raw_input: &str,
    has_attachments: bool,
) -> PlanCommandOutcome {
    let message = js_trim(raw_input);
    if message == "off" && has_attachments {
        return PlanCommandOutcome {
            result: PlanCommandResult::error(PLAN_ATTACHMENTS_OFF_TEXT),
            steer: None,
            set: PlanSetDecision::Noop,
        };
    }
    if message == "off" {
        let set = machine.set(false);
        let text = match set.outcome() {
            PlanSetOutcome::Committed => PLAN_OFF_TEXT,
            PlanSetOutcome::Queued => PLAN_LEAVING_TEXT,
            PlanSetOutcome::Cancelled => PLAN_ENTRY_CANCELLED_TEXT,
            // Repeat the queued wording while an exit still awaits the next
            // accepted pre-step; only a truly inactive session reads
            // idempotent.
            PlanSetOutcome::Noop => {
                if machine.active() {
                    PLAN_LEAVING_TEXT
                } else {
                    PLAN_ALREADY_INACTIVE_TEXT
                }
            }
        };
        return PlanCommandOutcome {
            result: PlanCommandResult::success(text),
            steer: None,
            set,
        };
    }
    let set = machine.set(true);
    let steer = if message.is_empty() && !has_attachments {
        None
    } else {
        Some(message.to_string())
    };
    let text = if set.outcome() == PlanSetOutcome::Committed {
        PLAN_ON_TEXT
    } else {
        PLAN_ENTERING_TEXT
    };
    PlanCommandOutcome {
        result: PlanCommandResult::success(text),
        steer,
        set,
    }
}
