//! `dsh-plan` — plan mode as logged per-agent collaboration state.
//!
//! Ported from `packages/plan/plan-mode`. While plan mode is active, a
//! deployment-owned guidance section is included in each model request and
//! `exit_plan_mode` presents the completed plan for user review; the `/plan off`
//! command lets a user leave directly.
//!
//! ## What is here and what is not
//!
//! Everything with a rule or a wire format is ported:
//!
//! - [`types`] — the validated config, the folded unit state, the wire
//!   projection, and the process-local pending selection.
//! - [`event`] — the durable `plan/mode` switch and the narrow `command/run` /
//!   `command/done` views the fold reads.
//! - [`fold`] — the projection fold pairing a `/plan` selection with its
//!   settlement and the committed switch.
//! - [`machine`] — the two-phase selection machine: `set`/`commit_set` and
//!   `decide_flush`/`commit_flush`, the narration, and the policy text.
//! - [`tool`] — the stable `exit_plan_mode` tool: description, schemas, pre-ask
//!   checks, review question, answer classification, post-review decision, and
//!   presentation.
//! - [`command`] — the `/plan` metadata, copy, and pure handler.
//!
//! Deliberately **not** ported, because they are host or lifetime concerns:
//!
//! - **The Cordis service and its registrations.** `ctx.systemPrompt.section`,
//!   `ctx.sessionProjections.register`, `ctx.tools.register`, and the
//!   `ctx.inject(['commands'], ...)` child are composition. The pure values they
//!   register are here.
//! - **The `agent/pre-step` waterfall and the agent loop.** [`PlanMachine`]
//!   returns the flush decision; the caller runs it inside the accepted pre-step
//!   and appends the event before request assembly.
//! - **The user-questions channel.** Asking, awaiting, cancellation, and the
//!   `ASK_CANCELLED` distinction belong to the interaction service.
//!   [`classify_review`] and [`decide_exit`] take its answer and error outcome.
//! - **The turn-boundary projection as a service.** [`PlanMachine::from_events`]
//!   derives the open-turn signal from the log; a live caller can pass its own
//!   signal to [`PlanMachine::new`].
//! - **Tool restriction.** Plan mode has none — see
//!   [`tool::tool_restriction`] — because the TypeScript plugin restrains
//!   through guidance alone and keeps the catalog stable.
//!
//! There are no model calls, no tool execution, and no session appends in this
//! crate: every decision is a pure function of the log, the config, and the
//! caller-supplied selection.

#![forbid(unsafe_code)]

mod text;

pub mod brand;
pub mod command;
pub mod event;
pub mod fold;
pub mod machine;
pub mod tool;
pub mod types;

pub use brand::CommandId;
pub use command::{
    command_metadata, run_plan_command, PlanCommandInput, PlanCommandKind, PlanCommandMetadata,
    PlanCommandOutcome, PlanCommandResult, PLAN_ALREADY_INACTIVE_TEXT, PLAN_ATTACHMENTS_OFF_TEXT,
    PLAN_COMMAND_DEFINITION_ID, PLAN_COMMAND_DESCRIPTION, PLAN_COMMAND_HINT, PLAN_COMMAND_NAME,
    PLAN_ENTERING_TEXT, PLAN_ENTRY_CANCELLED_TEXT, PLAN_LEAVING_TEXT, PLAN_OFF_TEXT, PLAN_ON_TEXT,
};
pub use event::{
    decode_command_done, decode_command_run, decode_plan_mode, encode_plan_mode, CommandDoneKind,
    CommandDoneView, CommandRunView, PlanModeEvent, COMMAND_DONE_TAG, COMMAND_RUN_TAG,
    PLAN_MODE_TAG,
};
pub use fold::{apply_plan_state, fold_plan_state, plan_projection, PlanFoldError};
pub use machine::{
    policy_text, FlushDecision, Narration, PlanMachine, PlanSetDecision, PlanSetOutcome,
    NARRATION_PLAN_OFF, NARRATION_PLAN_ON, PLAN_MODE_SOURCE_KIND,
};
pub use tool::{
    classify_review, decide_exit, first_heading, output_schema, parameter_schema,
    plan_review_question, present_call, present_result, require_review_channel, tool_restriction,
    valid_plan_heading, validate_exit_plan, ExitCallPresentation, ExitDecision, ExitPlanModeError,
    ExitPlanModeOutput, ExitResultPresentation, ExitReview, PlanReviewIntent, PlanReviewQuestion,
    ReviewAnswer, ReviewAnswerItem, ReviewOption, APPROVE_LABEL, APPROVE_OPTION_DESCRIPTION,
    EXIT_APPROVED_TEXT, EXIT_DESCRIPTION, EXIT_DISMISSED_TEXT, EXIT_KEEP_PLANNING_TEXT,
    EXIT_PLAN_ARGUMENT_DESCRIPTION, EXIT_PLAN_MODE, KEEP_PLANNING_LABEL,
    KEEP_PLANNING_OPTION_DESCRIPTION, PLAN_REVIEW_INTENT, REVIEW_HEADER, REVIEW_ID,
    REVIEW_QUESTION,
};
pub use types::{
    PendingIntent, PlanModeConfig, PlanModeConfigError, PlanProjection, PlanRunning, PlanStatus,
    PlanUnitState, PLAN_POLICY_ORDER, PLAN_POLICY_SECTION,
};
