//! The `plan` projection, folded back out of the session log.
//!
//! Ported from `planProjectionDefinition` in
//! `packages/plan/plan-mode/src/index.ts`. The projection is derived state:
//! nothing stores it, and replaying the log reproduces it exactly, which is what
//! makes resume and fork restore plan mode.
//!
//! The fold pairs the command lifecycle with the committed switch:
//!
//! - A `/plan` run with raw input records `running = { commandId, wanted }`,
//!   where `wanted` is `args.trim() != "off"`. A run with no recorded input is
//!   not a plan selection and leaves the state alone.
//! - Its paired `command/done` turns a successful, still-different selection
//!   into `wanted`; a failed settlement, or one that targets the committed mode,
//!   clears it. The pairing is by command id, so another command's settlement
//!   never touches plan state.
//! - `plan/mode` commits a mode and clears `wanted`.
//! - `request/header` freezes the current `active` into `activeAtLastHeader`,
//!   the anchor the user-switch narration compares against.
//!
//! The wire projection is [`plan_projection`]: `pending` is the effective
//! selection — a running command's target, else the settled `wanted` — when it
//! differs from the committed mode.

#![forbid(unsafe_code)]

use dsh_session::{CoreEvent, SessionEvent};

use super::command::PLAN_COMMAND_NAME;
use super::event::{decode_command_done, decode_command_run, decode_plan_mode, CommandDoneKind};
use super::text::js_trim;
use super::types::{PlanProjection, PlanRunning, PlanUnitState};

/// A known plan, command, or header event whose payload does not match the
/// declared shape.
///
/// Fold errors are not skipped: a known tag is never silently degraded, so a
/// malformed payload stops the fold instead of being ignored.
#[derive(Debug, thiserror::Error)]
#[error("plan state fold failed: {0}")]
pub struct PlanFoldError(#[from] serde_json::Error);

/// Fold one event into the plan unit state.
pub fn apply_plan_state(
    state: PlanUnitState,
    event: &SessionEvent,
) -> Result<PlanUnitState, PlanFoldError> {
    if let Some(decoded) = decode_command_run(event) {
        let run = decoded?;
        if run.name != PLAN_COMMAND_NAME {
            return Ok(state);
        }
        let Some(args) = run.args else {
            return Ok(state);
        };
        return Ok(PlanUnitState {
            running: Some(PlanRunning {
                command_id: run.command_id,
                wanted: js_trim(&args) != "off",
            }),
            ..state
        });
    }
    if let Some(decoded) = decode_command_done(event) {
        let done = decoded?;
        let Some(running) = state.running.as_ref() else {
            return Ok(state);
        };
        if running.command_id != done.command_id {
            return Ok(state);
        }
        let wanted = if done.kind == CommandDoneKind::Success && running.wanted != state.active {
            Some(running.wanted)
        } else {
            None
        };
        return Ok(PlanUnitState {
            wanted,
            running: None,
            ..state
        });
    }
    if let Some(decoded) = decode_plan_mode(event) {
        let mode = decoded?;
        return Ok(PlanUnitState {
            active: mode.active,
            wanted: None,
            ..state
        });
    }
    if matches!(event.core(), Some(CoreEvent::RequestHeader { .. })) {
        return Ok(PlanUnitState {
            active_at_last_header: Some(state.active),
            ..state
        });
    }
    Ok(state)
}

/// Replay a whole log into the plan unit state.
pub fn fold_plan_state(events: &[SessionEvent]) -> Result<PlanUnitState, PlanFoldError> {
    let mut state = PlanUnitState::default();
    for event in events {
        state = apply_plan_state(state, event)?;
    }
    Ok(state)
}

/// Derive the client-facing projection from the host fold state.
///
/// A running command's target takes precedence over the last settled selection,
/// exactly as `state.running?.wanted ?? state.wanted` does. A selection equal to
/// the committed mode is not pending — the switch is already in force.
pub fn plan_projection(state: &PlanUnitState) -> PlanProjection {
    let wanted = state
        .running
        .as_ref()
        .map(|running| running.wanted)
        .or(state.wanted);
    PlanProjection {
        active: state.active,
        pending: wanted.is_some_and(|wanted| wanted != state.active),
    }
}
