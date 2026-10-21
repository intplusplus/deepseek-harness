//! The todo domain: one list entry, its three-state lifecycle, the canonical
//! whole-list value the tool returns, and the errors a write can raise.
//!
//! Ported from `packages/todo/tool-todo/src/types.ts` and the value half of
//! `src/index.ts`. The list carries no identity: every write is a complete
//! replacement, so an entry is only its trimmed `content` line and its
//! `status`. That is also why duplicate content is rejected — content is the
//! only key a caller could mistake for identity.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One entry's lifecycle state.
///
/// The three names are the wire values (`pending`, `in_progress`,
/// `completed`); `in_progress` marks work happening now, and a deployment that
/// allows parallel work may mark several entries at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// Not started.
    Pending,
    /// Being worked on now.
    InProgress,
    /// Done.
    Completed,
}

impl TodoStatus {
    /// Every status, in the declaration order the tool schema's `enum` uses.
    pub const ALL: &'static [Self] = &[Self::Pending, Self::InProgress, Self::Completed];

    /// The status's wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
        }
    }

    /// Parse one wire value, or `None` when it is not a status.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "in_progress" => Some(Self::InProgress),
            "completed" => Some(Self::Completed),
            _ => None,
        }
    }
}

/// One entry in an agent's todo list.
///
/// Deliberately minimal — a human-readable line and a status — because the
/// list is replaced wholesale on every write and entries need no identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TodoItem {
    /// What this task is: a short imperative line.
    pub content: String,
    /// Lifecycle state.
    pub status: TodoStatus,
}

impl TodoItem {
    /// One entry with the given trimmed content and status.
    pub fn new(content: impl Into<String>, status: TodoStatus) -> Self {
        Self {
            content: content.into(),
            status,
        }
    }

    /// This entry as the JSON object the tool output and the event carry.
    ///
    /// Built explicitly rather than through `serde_json::to_value` so field
    /// order (`content`, then `status`) is fixed at the declaration site.
    pub fn to_value(&self) -> Value {
        let mut item = Map::new();
        item.insert("content".to_string(), Value::String(self.content.clone()));
        item.insert(
            "status".to_string(),
            Value::String(self.status.as_str().to_string()),
        );
        Value::Object(item)
    }
}

/// How many entries carry each status.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TodoCounts {
    /// Entries not started.
    pub pending: u64,
    /// Entries in progress now.
    pub in_progress: u64,
    /// Finished entries.
    pub completed: u64,
}

impl TodoCounts {
    /// Count one list by status.
    pub fn count(todos: &[TodoItem]) -> Self {
        let mut counts = Self::default();
        for todo in todos {
            match todo.status {
                TodoStatus::Pending => counts.pending += 1,
                TodoStatus::InProgress => counts.in_progress += 1,
                TodoStatus::Completed => counts.completed += 1,
            }
        }
        counts
    }

    /// Every entry, whatever its status.
    pub fn total(self) -> u64 {
        self.pending + self.in_progress + self.completed
    }

    /// The model-facing confirmation sentence.
    pub fn confirmation_text(self) -> String {
        format!(
            "Updated todo list: {} pending, {} in progress, {} completed.",
            self.pending, self.in_progress, self.completed
        )
    }
}

/// The canonical value of a successful `todo_write`: the replacement list and
/// its counts.
///
/// This is the tool's output value, not its rendering; the model sees
/// [`TodoCounts::confirmation_text`]. Field order (`todos`, then `counts`;
/// `pending`, `inProgress`, `completed`) is the TypeScript declaration order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TodoWriteOutput {
    /// The complete replacement list, canonically trimmed.
    pub todos: Vec<TodoItem>,
    /// Counts over [`TodoWriteOutput::todos`].
    pub counts: TodoCounts,
}

impl TodoWriteOutput {
    /// Build the value for one canonical list.
    pub fn new(todos: Vec<TodoItem>) -> Self {
        let counts = TodoCounts::count(&todos);
        Self { todos, counts }
    }

    /// The model-facing confirmation sentence.
    pub fn confirmation_text(&self) -> String {
        self.counts.confirmation_text()
    }

    /// This value as JSON, with the TypeScript declaration's field order.
    pub fn to_value(&self) -> Value {
        let mut counts = Map::new();
        counts.insert("pending".to_string(), Value::from(self.counts.pending));
        counts.insert(
            "inProgress".to_string(),
            Value::from(self.counts.in_progress),
        );
        counts.insert("completed".to_string(), Value::from(self.counts.completed));

        let mut value = Map::new();
        value.insert(
            "todos".to_string(),
            Value::Array(self.todos.iter().map(TodoItem::to_value).collect()),
        );
        value.insert("counts".to_string(), Value::Object(counts));
        Value::Object(value)
    }
}

/// The deployment's explicit todo policy.
///
/// `allowParallelInProgress` is required: a composition that omits it fails at
/// load rather than silently choosing a concurrency discipline. The flag also
/// selects the active-status clause of the model-facing tool description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TodoWriteConfig {
    /// Whether several entries may be `in_progress` at once.
    pub allow_parallel_in_progress: bool,
}

impl TodoWriteConfig {
    /// A policy with the given parallel-work choice.
    pub fn new(allow_parallel_in_progress: bool) -> Self {
        Self {
            allow_parallel_in_progress,
        }
    }

    /// Whether several entries may be `in_progress` at once.
    pub fn allows_parallel(self) -> bool {
        self.allow_parallel_in_progress
    }
}

/// Schema-level violations of the `todo_write` arguments.
///
/// The registry's JSON-schema walk rejects arguments before the tool body
/// runs; the messages are the same path-qualified violations, joined the way
/// TypeScript's `ToolArgsError` joins them.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid arguments: {}", .violations.join("; "))]
pub struct TodoInputError {
    /// Individual violations in schema-walk order.
    pub violations: Vec<String>,
}

impl TodoInputError {
    /// One violation.
    pub fn new(violations: Vec<String>) -> Self {
        Self { violations }
    }

    /// A single-violation error.
    pub fn single(violation: impl Into<String>) -> Self {
        Self {
            violations: vec![violation.into()],
        }
    }
}

/// Why one `todo_write` call did not produce a snapshot.
///
/// A rejected call never reaches the session log: the caller receives this
/// error, and its text is what the model sees after the `Error: ` envelope.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TodoWriteError {
    /// The arguments do not satisfy the tool's parameter schema.
    #[error("{0}")]
    InvalidArguments(#[from] TodoInputError),
    /// A content line was empty after trimming.
    #[error("invalid todo: `content` must be a non-empty string")]
    EmptyContent,
    /// Two entries share content after trimming; the value is the JSON-quoted
    /// duplicate, exactly as `JSON.stringify(content)` renders it.
    #[error("invalid todos: duplicate content {0}")]
    DuplicateContent(String),
    /// A single-active deployment received more than one `in_progress` entry.
    #[error("invalid todos: at most one task may be in_progress (got {0})")]
    AtMostOneInProgress(u64),
    /// A caller without an owning agent session has nowhere to write the list.
    #[error("todo_write requires an owning agent session")]
    NoOwningSession,
}
