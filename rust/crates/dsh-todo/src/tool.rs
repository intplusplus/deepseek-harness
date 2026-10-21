//! The `todo_write` tool: model-facing description, schemas, argument
//! validation, the pure decision, and presentation.
//!
//! Ported from `packages/todo/tool-todo/src/index.ts`. The TypeScript tool is
//! registered with `defineTool`, which compiles the author-facing parameter DSL
//! into the raw JSON Schema subset and validates every call against it before
//! the body runs. Rust has no such registry in this crate, so the compiled
//! schema and the same path-qualified violations are produced here, and
//! [`decide_todo_write`] is the body: validate, canonicalize, check for an
//! owning session, and return the value plus the durable event.
//!
//! The decision is pure. It never appends to a session and never renders a
//! model request; the caller owns both. [`execute_todo_write`] packages the
//! decision as the [`ToolExecutionResult`] and the event a caller commits, which
//! is the same split `dsh-retry` uses between deciding and waiting.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use serde::Serialize;
use serde_json::{json, Value};

use dsh_llm::ContentBlock;
use dsh_session::SessionEvent;
use dsh_tools::ToolExecutionResult;

use super::event::{encode_todo_write, TodoWriteEvent};
use super::types::{TodoInputError, TodoItem, TodoStatus, TodoWriteError, TodoWriteOutput};

/// The tool's model-facing name.
pub const TODO_WRITE_TOOL: &str = "todo_write";

/// The presentation card's title.
pub const TODO_CALL_TITLE: &str = "Update todo list";

/// First, policy-independent sentence of the tool description.
pub const DESCRIPTION_HEAD: &str = "Record and update a task list to plan multi-step work and show progress; skip it for trivial single-step tasks. Add one todo per concrete step before you start. ";

/// Active-status clause for a deployment that allows parallel work.
pub const DESCRIPTION_PARALLEL: &str = "While work remains, keep the todos being worked on `in_progress`, several only when work runs in parallel. ";

/// Active-status clause for a single-active deployment.
pub const DESCRIPTION_SINGLE: &str = "While work remains, keep exactly one todo `in_progress`. ";

/// Final, policy-independent sentence of the tool description.
pub const DESCRIPTION_TAIL: &str = "Mark each todo `completed` as soon as it is done.";

/// The model-facing description for one activation.
///
/// The active-status clause is the only part the parallel policy changes,
/// because it is the only instruction that policy alters.
pub fn describe(allow_parallel: bool) -> String {
    let active = if allow_parallel {
        DESCRIPTION_PARALLEL
    } else {
        DESCRIPTION_SINGLE
    };
    format!("{DESCRIPTION_HEAD}{active}{DESCRIPTION_TAIL}")
}

/// The compiled raw JSON Schema of `todo_write`'s arguments.
///
/// The root is the implicit open parameter object: `todos` is required and
/// extra root keys are allowed. Each item is closed
/// (`additionalProperties: false`), so the logged snapshot equals what the
/// model believes it wrote.
pub fn parameter_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The COMPLETE task list, replacing any previous list.",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "What the task is — a short imperative line."
                        },
                        "status": {
                            "type": "string",
                            "description": "pending (not started) | in_progress (now) | completed (done).",
                            "enum": ["pending", "in_progress", "completed"]
                        }
                    },
                    "required": ["content", "status"]
                }
            }
        },
        "required": ["todos"]
    })
}

/// The compiled raw JSON Schema of `todo_write`'s canonical value.
pub fn output_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "content": { "type": "string" },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"]
                        }
                    },
                    "required": ["content", "status"]
                }
            },
            "counts": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "pending": { "type": "integer" },
                    "inProgress": { "type": "integer" },
                    "completed": { "type": "integer" }
                },
                "required": ["pending", "inProgress", "completed"]
            }
        },
        "required": ["todos", "counts"]
    })
}

/// Validate model-generated arguments against [`parameter_schema`].
///
/// Returns path-qualified violations in the schema walk's order; empty means
/// valid. The messages are the registry's own, because the TypeScript tool is
/// validated there before its body runs: missing required properties first,
/// then each declared property in declaration order, then the closed object's
/// undeclared keys in the caller's key order.
pub fn validate_arguments(arguments: &Value) -> Vec<String> {
    let mut violations = Vec::new();
    let Value::Object(fields) = arguments else {
        violations.push("\"arguments\" must be an object".to_string());
        return violations;
    };
    match fields.get("todos") {
        None => violations.push("missing required property \"todos\"".to_string()),
        Some(Value::Array(items)) => {
            for (index, item) in items.iter().enumerate() {
                validate_item(item, index, &mut violations);
            }
        }
        Some(_) => violations.push("\"todos\" must be an array".to_string()),
    }
    violations
}

/// Validate one array item at `todos[index]`.
fn validate_item(item: &Value, index: usize, violations: &mut Vec<String>) {
    let path = format!("todos[{index}]");
    let Value::Object(fields) = item else {
        violations.push(format!("\"{path}\" must be an object"));
        return;
    };
    for key in ["content", "status"] {
        if !fields.contains_key(key) {
            violations.push(format!("missing required property \"{path}.{key}\""));
        }
    }
    match fields.get("content") {
        Some(Value::String(_)) | None => {}
        Some(_) => violations.push(format!("\"{path}.content\" must be a string")),
    }
    match fields.get("status") {
        Some(Value::String(status)) if TodoStatus::parse(status).is_some() => {}
        Some(Value::String(_)) => violations.push(format!(
            "\"{path}.status\" must be one of [\"pending\",\"in_progress\",\"completed\"]"
        )),
        Some(_) => violations.push(format!("\"{path}.status\" must be a string")),
        None => {}
    }
    for key in fields.keys() {
        if key != "content" && key != "status" {
            violations.push(format!(
                "\"{path}.{key}\" is not a declared property (additionalProperties: false)"
            ));
        }
    }
}

/// The raw `(content, status)` pairs of schema-valid arguments.
struct RawTodo<'a> {
    content: &'a str,
    status: TodoStatus,
}

/// Extract the raw pairs. Validation has already rejected every other shape, so
/// a mismatch can only mean this crate's schema and its reader disagree.
fn raw_todos(arguments: &Value) -> Result<Vec<RawTodo<'_>>, TodoInputError> {
    let Some(Value::Array(items)) = arguments.get("todos") else {
        return Err(TodoInputError::single(
            "missing required property \"todos\"",
        ));
    };
    let mut raw = Vec::with_capacity(items.len());
    for item in items {
        let Some(fields) = item.as_object() else {
            return Err(TodoInputError::single(
                "\"todos\" contains a non-object item",
            ));
        };
        let Some(content) = fields.get("content").and_then(Value::as_str) else {
            return Err(TodoInputError::single(
                "\"todos\" contains an item without string content",
            ));
        };
        let Some(status) = fields
            .get("status")
            .and_then(Value::as_str)
            .and_then(TodoStatus::parse)
        else {
            return Err(TodoInputError::single(
                "\"todos\" contains an item without a valid status",
            ));
        };
        raw.push(RawTodo { content, status });
    }
    Ok(raw)
}

/// JSON-quote one string the way `JSON.stringify` does, for the duplicate error.
fn json_quote(value: &str) -> String {
    Value::String(value.to_string()).to_string()
}

/// Validate the arguments and build the canonical replacement list.
///
/// Schema violations come back as [`TodoInputError`]; the value constraints the
/// schema cannot express — trimmed non-empty unique content, and at most one
/// `in_progress` item unless the deployment allows parallel work — come back as
/// their own variants with the TypeScript messages.
pub fn canonical_todos(
    arguments: &Value,
    allow_parallel: bool,
) -> Result<Vec<TodoItem>, TodoWriteError> {
    let violations = validate_arguments(arguments);
    if !violations.is_empty() {
        return Err(TodoInputError::new(violations).into());
    }
    let raw = raw_todos(arguments)?;
    let mut todos = Vec::with_capacity(raw.len());
    let mut seen = BTreeSet::new();
    let mut active = 0_u64;
    for todo in raw {
        let content = todo.content.trim();
        if content.is_empty() {
            return Err(TodoWriteError::EmptyContent);
        }
        if !seen.insert(content.to_string()) {
            return Err(TodoWriteError::DuplicateContent(json_quote(content)));
        }
        if todo.status == TodoStatus::InProgress {
            active += 1;
        }
        todos.push(TodoItem::new(content, todo.status));
    }
    if !allow_parallel && active > 1 {
        return Err(TodoWriteError::AtMostOneInProgress(active));
    }
    Ok(todos)
}

/// One accepted `todo_write` call: the canonical value and the event to append.
#[derive(Clone, Debug, PartialEq)]
pub struct TodoWriteDecision {
    /// The tool's locked output value: the replacement list and its counts.
    pub output: TodoWriteOutput,
    /// The durable `todo/write` snapshot the caller appends.
    pub event: SessionEvent,
}

/// Decide one `todo_write` call.
///
/// `has_owning_session` is the caller's `exec.agent !== undefined`: the list is
/// per-agent-session state, so a caller with no owning session is rejected
/// rather than silently dropped. Validation and canonicalization run first, in
/// the TypeScript order, so their errors win over the missing-session error.
pub fn decide_todo_write(
    arguments: &Value,
    allow_parallel: bool,
    has_owning_session: bool,
) -> Result<TodoWriteDecision, TodoWriteError> {
    let todos = canonical_todos(arguments, allow_parallel)?;
    if !has_owning_session {
        return Err(TodoWriteError::NoOwningSession);
    }
    let output = TodoWriteOutput::new(todos);
    let event = encode_todo_write(&TodoWriteEvent {
        todos: output.todos.clone(),
    });
    Ok(TodoWriteDecision { output, event })
}

/// The model-facing result of one accepted call.
pub fn todo_write_result(decision: &TodoWriteDecision) -> ToolExecutionResult {
    ToolExecutionResult::success(vec![ContentBlock::text(
        decision.output.confirmation_text(),
    )])
    .with_value(decision.output.to_value())
}

/// One executed `todo_write` call, still awaiting its durable append.
pub struct TodoWriteCall {
    /// The model-facing result.
    pub result: ToolExecutionResult,
    /// The event to append before the result is committed; `None` on failure.
    pub event: Option<SessionEvent>,
}

/// Execute one `todo_write` call as the tool body would.
///
/// On success the caller appends [`TodoWriteCall::event`] and returns
/// [`TodoWriteCall::result`]. A failed call never reaches the log: its result is
/// the failure the model sees.
pub fn execute_todo_write(
    arguments: &Value,
    allow_parallel: bool,
    has_owning_session: bool,
) -> TodoWriteCall {
    match decide_todo_write(arguments, allow_parallel, has_owning_session) {
        Ok(decision) => TodoWriteCall {
            result: todo_write_result(&decision),
            event: Some(decision.event),
        },
        Err(error) => TodoWriteCall {
            result: ToolExecutionResult::failure(error.to_string()),
            event: None,
        },
    }
}

/// The pending-state presentation of one `todo_write` call.
///
/// Field order (`card`, `title`, `kind`, `rawInput`) matches the TypeScript
/// presenter. Presentation is display-only and may replay logged arguments, so
/// it never validates; a missing `todos` renders as an empty list.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TodoCallPresentation {
    /// The generic card kind.
    pub card: String,
    /// The card title.
    pub title: String,
    /// The tool category.
    pub kind: String,
    /// The raw model-supplied list.
    pub raw_input: Value,
}

/// Build the pending-state presentation for one call.
pub fn present_call(arguments: &Value) -> TodoCallPresentation {
    let raw_input = arguments
        .get("todos")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    TodoCallPresentation {
        card: "generic".to_string(),
        title: TODO_CALL_TITLE.to_string(),
        kind: "other".to_string(),
        raw_input,
    }
}
