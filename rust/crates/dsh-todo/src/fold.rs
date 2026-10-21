//! The `todos` projection, folded back out of the session log.
//!
//! Ported from the `todos` unit in `packages/todo/tool-todo/src/index.ts`. The
//! projection is derived state: nothing stores it, and replaying the log
//! reproduces it exactly.
//!
//! The whole-list rule makes the fold trivial: every `todo/write` carries the
//! complete replacement list, so the latest one wins. The two boundaries around
//! it are the interesting part:
//!
//! - **`turn/start` clears the list.** The checklist belongs to the turn that
//!   wrote it; a new turn starts with no list.
//! - **`turn/end` keeps it.** The finished checklist stays visible between
//!   turns, so a UI can show what the last turn did.
//!
//! Every other event leaves the state untouched, and `todo/write` adds no
//! derived model message — this is log-only UI state.

#![forbid(unsafe_code)]

use dsh_session::{CoreEvent, SessionEvent};

use super::event::decode_todo_write;
use super::types::TodoItem;

/// The agent's current whole todo list, or `None` before the first write or
/// after a later turn begins.
pub type TodoState = Option<Vec<TodoItem>>;

/// A `todo/write` event whose payload does not match the declared shape.
///
/// Fold errors are not skipped: a known tag is never silently degraded, so a
/// malformed payload stops the fold instead of being ignored.
#[derive(Debug, thiserror::Error)]
#[error("todo state fold failed: {0}")]
pub struct TodoFoldError(#[from] serde_json::Error);

/// Fold one event into todo state.
///
/// A `todo/write` event whose payload does not match the declared shape is an
/// error, not a skipped event: a known tag is never silently degraded.
pub fn apply_todo_state(
    state: TodoState,
    event: &SessionEvent,
) -> Result<TodoState, TodoFoldError> {
    if matches!(event.core(), Some(CoreEvent::TurnStart { .. })) {
        return Ok(None);
    }
    let Some(decoded) = decode_todo_write(event) else {
        return Ok(state);
    };
    Ok(Some(decoded?.todos))
}

/// Replay a whole log into todo state.
pub fn fold_todo_state(events: &[SessionEvent]) -> Result<TodoState, TodoFoldError> {
    let mut state: TodoState = None;
    for event in events {
        state = apply_todo_state(state, event)?;
    }
    Ok(state)
}
