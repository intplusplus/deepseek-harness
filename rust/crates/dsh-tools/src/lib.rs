//! `dsh-tools` — the tool contract, the registry, and model-ordered scheduling.
//!
//! Ported from `packages/core/tools` and `packages/core/agent-loop/tool-calls`.
//! The result union keeps "a failure never carries a value" and "only a success
//! concludes a turn" in the type system rather than in runtime checks, and the
//! scheduler guarantees that every model-requested call receives exactly one
//! durable `tool/result` — including a call cancelled before it ever started.
//!
//! [`gate`] adds the permission layer ported from the same package: the
//! pre/post decision unions, the pure rules that converge an ordered decision
//! chain, and the compiled allow/deny restriction filter. The scheduler
//! consults a gate before every tool body; see [`schedule`] for how the two
//! ordering invariants hold for gated calls too.

#![forbid(unsafe_code)]

pub mod cancel;
pub mod gate;
pub mod registry;
pub mod schedule;
pub mod types;

pub use cancel::CancelSignal;
pub use gate::{
    aborted_before_dispatch_result, admits_all, apply_guard, apply_post_tool, ask_denial_reason,
    converge_pre_tool, evaluate_pre_dispatch, evaluate_pre_tool, failure_message_from_content,
    resolve_ask, validate_restriction, AcceptAllPostTool, AcceptPayload, AllowAllGate,
    ApprovalOutcome, DisplayReason, GateOutcome, PostToolDecision, PostToolError, PostToolGate,
    PreToolDecision, PreToolGate, PreToolVote, ResolvedAsk, RestrictionError, ToolRestriction,
};
pub use registry::{ToolHandler, ToolRegistry};
pub use schedule::{
    execute_tool_calls, execute_tool_calls_gated, execute_tool_calls_with_post_execute,
    ToolCallOutcome, ToolSchedule, ToolScheduleConfig, DEFAULT_MAX_PARALLEL_TOOL_CALLS,
};
pub use types::{
    argument, parse_arguments, ToolCallInput, ToolErrorInfo, ToolExecutionMode,
    ToolExecutionResult, ToolFailure, TOOL_ABORTED_BEFORE_DISPATCH, TOOL_UNKNOWN_TOOL,
};
