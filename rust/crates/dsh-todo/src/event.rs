//! The durable `todo/write` snapshot.
//!
//! TypeScript declares it by merging into `SessionEventMap`:
//!
//! ```ts
//! declare module '@deepseek-ai/dsh-session/types' {
//!   interface SessionEventMap {
//!     'todo/write': { todos: TodoItem[] }
//!   }
//! }
//! ```
//!
//! Rust has no declaration merging, so the event reaches the log through
//! [`SessionEvent::Extension`]. The payload is the whole replacement list —
//! last write wins on replay — and carries no derived history.
//!
//! [`encode_todo_write`] is infallible because the payload is assembled field by
//! field: `type` first (the extension slot's invariant), then `todos`, then each
//! item's `content` and `status`. [`decode_todo_write`] keeps the extension
//! slot's other invariant: a `todo/write` event whose payload does not match the
//! declared shape returns `Some(Err(..))`, never a silently degraded payload.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use dsh_session::SessionEvent;

use super::types::TodoItem;

/// Tag of the whole-list `todo/write` snapshot.
pub const TODO_WRITE_TAG: &str = "todo/write";

/// The complete replacement list recorded by one `todo_write` call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TodoWriteEvent {
    /// The new whole list, in the order the model supplied.
    pub todos: Vec<TodoItem>,
}

/// Encode one snapshot as the extension event the log stores.
///
/// Field order is fixed at the assembly site: `type`, `todos`, then each item's
/// `content` and `status`.
pub fn encode_todo_write(event: &TodoWriteEvent) -> SessionEvent {
    let mut payload = Map::new();
    payload.insert(
        "todos".to_string(),
        Value::Array(event.todos.iter().map(TodoItem::to_value).collect()),
    );
    SessionEvent::extension_event(TODO_WRITE_TAG, payload)
}

/// Decode one snapshot from the log.
///
/// `None` means the event carries a different tag; `Some(Err(..))` means it is a
/// `todo/write` event whose payload does not match the declared shape.
pub fn decode_todo_write(
    event: &SessionEvent,
) -> Option<Result<TodoWriteEvent, serde_json::Error>> {
    let extension = event.extension()?;
    if extension.kind() != TODO_WRITE_TAG {
        return None;
    }
    Some(extension.decode::<TodoWriteEvent>())
}
