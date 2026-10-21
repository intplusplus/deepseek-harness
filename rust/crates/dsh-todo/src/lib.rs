//! `dsh-todo` — the todo tool state model.
//!
//! Ported from `packages/todo/tool-todo`. One tool, `todo_write`, replaces an
//! agent's whole task list and logs the replacement as a `todo/write` snapshot;
//! replay is last-write-wins, and UIs render from session events. A non-agent
//! caller has no owning list and is rejected.
//!
//! ## What is here and what is not
//!
//! Everything with a rule or a wire format is ported:
//!
//! - [`types`] — the entry, its lifecycle, counts, the locked output value, the
//!   deployment policy, and the errors.
//! - [`tool`] — the model-facing description, the compiled parameter and output
//!   schemas, the registry's argument violations, canonicalization, the pure
//!   decision, the tool result, and presentation.
//! - [`event`] — the durable `todo/write` snapshot over the session log's
//!   extension slot.
//! - [`fold`] — the `todos` projection: latest write wins, `turn/start` clears,
//!   `turn/end` keeps.
//!
//! Deliberately **not** ported, because they are host concerns:
//!
//! - **Tool registration and the Cordis service.** The TypeScript plugin
//!   registers through `ctx.tools.register` and `ctx.sessionProjections`;
//!   [`tool::execute_todo_write`] returns the result and the event a host
//!   commits instead of reaching into a session.
//! - **`presentCall`'s registry wrapper.** Presentation is display-only and may
//!   replay logged arguments, so the TypeScript registry validates softly and
//!   falls back to a generic card. [`tool::present_call`] is the presenter's own
//!   output; the soft-validation wrapper belongs to the host.
//! - **The `SessionProjectionMap` type merge.** Rust consumers read
//!   [`fold::fold_todo_state`] directly.
//!
//! There are no model calls and no tool execution in this crate: the decision
//! functions are pure, and the caller owns the session append, the model
//! request, and the error envelope.

#![forbid(unsafe_code)]

pub mod event;
pub mod fold;
pub mod tool;
pub mod types;

pub use event::{decode_todo_write, encode_todo_write, TodoWriteEvent, TODO_WRITE_TAG};
pub use fold::{apply_todo_state, fold_todo_state, TodoFoldError, TodoState};
pub use tool::{
    canonical_todos, decide_todo_write, describe, execute_todo_write, output_schema,
    parameter_schema, present_call, todo_write_result, validate_arguments, TodoCallPresentation,
    TodoWriteCall, TodoWriteDecision, DESCRIPTION_HEAD, DESCRIPTION_PARALLEL, DESCRIPTION_SINGLE,
    DESCRIPTION_TAIL, TODO_CALL_TITLE, TODO_WRITE_TOOL,
};
pub use types::{
    TodoCounts, TodoInputError, TodoItem, TodoStatus, TodoWriteConfig, TodoWriteError,
    TodoWriteOutput,
};
