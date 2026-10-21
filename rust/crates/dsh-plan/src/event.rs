//! The durable plan events and the narrow command views the fold reads.
//!
//! TypeScript declares `plan/mode` by merging into `SessionEventMap`, and the
//! command executor declares `command/run` / `command/done`. Rust's core log has
//! no command events, so this port reads all three through the extension slot:
//!
//! - `plan/mode` is the crate's own durable event, encoded and decoded here.
//! - `command/run` and `command/done` are decoded through narrow views carrying
//!   only the fields plan mode needs — the command id, the name, the optional
//!   raw input, and the settlement kind. A malformed payload is a fold error,
//!   not a skipped event, because a known tag is never silently degraded.
//!
//! Only `plan/mode` has an encoder: the command lifecycle belongs to the
//! command executor, and plan mode never writes it.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use dsh_session::SessionEvent;

use super::brand::CommandId;

/// Tag of the durable plan-mode switch.
pub const PLAN_MODE_TAG: &str = "plan/mode";

/// Tag of one resolved slash command entering its handler.
pub const COMMAND_RUN_TAG: &str = "command/run";

/// Tag of the paired command settlement.
pub const COMMAND_DONE_TAG: &str = "command/done";

/// Whether plan mode is in force from this point on.
///
/// Log-only, non-surface, whole-value replace: the last `plan/mode` wins, and a
/// log with none folds to inactive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanModeEvent {
    /// The committed mode.
    pub active: bool,
}

/// Encode one mode switch as the extension event the log stores.
///
/// The extension slot stamps `type` first; `active` follows.
pub fn encode_plan_mode(event: &PlanModeEvent) -> SessionEvent {
    let mut payload = Map::new();
    payload.insert("active".to_string(), Value::Bool(event.active));
    SessionEvent::extension_event(PLAN_MODE_TAG, payload)
}

/// Decode one mode switch from the log.
///
/// `None` means the event carries a different tag; `Some(Err(..))` means it is a
/// `plan/mode` event whose payload does not match the declared shape.
pub fn decode_plan_mode(event: &SessionEvent) -> Option<Result<PlanModeEvent, serde_json::Error>> {
    decode(event, PLAN_MODE_TAG)
}

/// The fields of one `command/run` record that plan mode reads.
///
/// `args` is the command's verbatim raw input, absent when the definition sets
/// `recordInput: false`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandRunView {
    /// Identity pairing this run with its `command/done`.
    pub command_id: CommandId,
    /// The resolved command name.
    pub name: String,
    /// Verbatim raw input, when the command records it.
    #[serde(default)]
    pub args: Option<String>,
}

/// The fields of one `command/done` record that plan mode reads.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandDoneView {
    /// Identity of the run this settles.
    pub command_id: CommandId,
    /// The handler's outcome.
    pub kind: CommandDoneKind,
}

/// A command handler's settled outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CommandDoneKind {
    /// The handler returned a success.
    Success,
    /// The handler threw or was aborted.
    Error,
}

/// Decode the plan-visible fields of one `command/run` record.
pub fn decode_command_run(
    event: &SessionEvent,
) -> Option<Result<CommandRunView, serde_json::Error>> {
    decode(event, COMMAND_RUN_TAG)
}

/// Decode the plan-visible fields of one `command/done` record.
pub fn decode_command_done(
    event: &SessionEvent,
) -> Option<Result<CommandDoneView, serde_json::Error>> {
    decode(event, COMMAND_DONE_TAG)
}

fn decode<T: for<'de> Deserialize<'de>>(
    event: &SessionEvent,
    tag: &str,
) -> Option<Result<T, serde_json::Error>> {
    let extension = event.extension()?;
    if extension.kind() != tag {
        return None;
    }
    Some(extension.decode::<T>())
}
