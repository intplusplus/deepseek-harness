//! `dsh-interaction` — the user interaction channel: user questions, the
//! approval policy layer, and the model-facing `ask_user_question` tool.
//!
//! Ported from three TypeScript packages:
//!
//! - `origin/packages/interaction/user-approval` — the approval policy layer
//!   and the `approval/asked` / `approval/decided` audit pair.
//! - `origin/packages/interaction/user-questions` — the question vocabulary,
//!   its validation rules, the question service, the timed foreground wait,
//!   and the `userQuestions` session projection.
//! - `origin/packages/interaction/tool-ask-user` — the model-facing
//!   `ask_user_question` tool, in its legacy and timed forms.
//!
//! ## Ported items
//!
//! - [`approval`]: [`ApprovalPolicy`], [`APPROVAL_POLICIES`],
//!   [`set_approval_policy`], the policy fold, the audit payloads, and
//!   [`AuditedApprovalService`].
//! - [`questions`]: the question and answer vocabulary, [`UserQuestionError`]
//!   and its stable codes, the validation rules, the answerer / live-agent /
//!   reply-sink seams, and [`UserQuestionService`].
//! - [`timed_wait`]: [`TimedQuestionWait`] with its injected [`Clock`], the
//!   claim bookkeeping, and the two asynchronous waits.
//! - [`projection`]: the pure fold of the `ask_user_question` events and the
//!   `userQuestions` [`ProjectionDefinition`].
//! - [`ask_user`]: [`AskUserQuestionTool`], a [`ToolHandler`].
//!
//! ## Not ported, and why
//!
//! - **The Cordis service lifecycle.** TypeScript mounts these packages as
//!   plugins on a context; Rust composes plain values, so the crate exposes
//!   the services as constructors instead.
//! - **The scope system.** The TypeScript dispatch is agent-scoped
//!   (`scopeTarget(agent, agent)`); the Rust seams carry the agent in the
//!   request and leave the filtering to the deployment.
//! - **The live agent registry.** `dsh-agent` exposes only `Agent { id }` —
//!   no registry, no ownership, no inbox mutation — so `CALLER_NOT_LIVE` and
//!   `DELEGATED_CALLER` are enforced through the injected
//!   [`questions::LiveAgentRegistry`], and the late reply is steered through
//!   the injected [`questions::ReplySink`]. A deployment with no registry
//!   mounted fails closed.
//! - **The async answerer waterfall.** [`dsh_tools::ToolHandler::call`] is
//!   synchronous, so the answerer seam is synchronous too; a deployment
//!   bridges its UI thread to it.
//! - **The `agent` field of the approval request.** `dsh-tools` notes that
//!   `ToolCallInput` carries no agent identity, so [`approval::AuditedApprovalService`]
//!   takes the session explicitly instead.
//! - **The system-prompt contribution.** The TypeScript approval service
//!   injects an `approval:policy` sentence into the assembled system prompt;
//!   the crate exposes the two sentences as constants
//!   ([`approval::ASK_SENTENCE`], [`approval::NEVER_SENTENCE`]) instead of
//!   owning a prompt plugin.
//!
//! ## Known deviations
//!
//! - **The approval outcome vocabulary is reused, not redeclared.** The
//!   TypeScript `ApprovalOutcome` is exactly
//!   [`dsh_tools::gate::ApprovalOutcome`], so this crate reuses that type
//!   rather than declaring a second one.
//! - **`ApprovalResult::Timeout` has no TypeScript counterpart.** A Rust
//!   approval that times out degrades to the fail-closed `unavailable`
//!   outcome.
//! - **An approval request without a tool call is dispatched with an empty
//!   call id.** The TypeScript request has `callId?`, while
//!   [`dsh_tools::approval::ApprovalRequest`] requires one; the optionality is
//!   expressed as an empty call id rather than fabricating one.
//! - **The approval request id is validated.** The TypeScript
//!   `ApprovalRequestId(id)` brands without validating; this port rejects an
//!   empty id.
//! - **The timeout of the timed question is enforced by the answerer.** The
//!   TypeScript service arms its own timer and aborts the answerer's signal;
//!   the Rust port hands the wait key and its budget to the answerer and
//!   recognises the resulting `ASK_TIMED_OUT`.
//!
//! ## Relationship to `dsh-tools::approval`
//!
//! `dsh_tools::approval` owns the **async answerer seam**: the
//! [`ApprovalService`](dsh_tools::approval::ApprovalService) trait, the
//! waterfall of listeners, and the cancel/timeout reconciliation of an
//! in-flight approval. This crate does **not** redefine that trait — it
//! consumes it. What the TypeScript `user-approval` package adds on top of
//! the raw answerer seam is exactly what this crate ports:
//!
//! 1. the **policy layer** — [`approval::ApprovalPolicy`],
//!    [`approval::APPROVAL_POLICIES`], [`approval::set_approval_policy`], and
//!    the `never`-before-dispatch gate that no listener registration order can
//!    bypass; and
//! 2. the **audit events** — `approval/asked` and `approval/decided`, the
//!    durable pair that records what was asked and how it settled.
//!
//! [`approval::AuditedApprovalService`] composes the two and is itself a
//! `dsh_tools::approval::ApprovalService`, so it can be mounted behind
//! [`dsh_tools::approval::await_approval`].

#![forbid(unsafe_code)]

pub mod approval;
pub mod ask_user;
pub mod projection;
pub mod questions;
pub mod timed_wait;

pub use approval::{
    approval_policy_override, effective_approval_policy, has_open_turn, set_approval_policy,
    set_approval_policy_str, ApprovalAsked, ApprovalDecided, ApprovalError, ApprovalIdGenerator,
    ApprovalPolicy, ApprovalPolicyEvent, ApprovalPolicySource, ApprovalRequestId,
    AuditedApprovalService, CountingApprovalIds, RandomApprovalIds, APPROVAL_ASKED_EVENT,
    APPROVAL_DECIDED_EVENT, APPROVAL_POLICIES, APPROVAL_POLICY_EVENT, ASK_SENTENCE, NEVER_SENTENCE,
};
pub use ask_user::{
    AskUserQuestionError, AskUserQuestionMode, AskUserQuestionTool, DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS, PENDING_NOTICE,
};
pub use projection::{
    apply_user_question_event, fold_user_questions, is_timed_ask_user_question_schema,
    questions_of, user_question_projection, UserQuestionFold, UserQuestionProjectionState,
    ASK_USER_QUESTION_TOOL, TIMED_WAIT_PARAMETER, TOOL_OUTCOME_UNKNOWN,
};
pub use questions::{
    validate_answer_batch, validate_questions, AskUserQuestionAnswer, AskUserQuestionAnswerItem,
    AskUserQuestionIntent, AskUserQuestionIntentKind, AskUserQuestionItem, AskUserQuestionOption,
    AskUserQuestionRequest, CountingMessageIds, LiveAgentRegistry, MessageIdSource,
    PendingUserQuestion, QuestionWait, RandomMessageIds, ReplySink, SettledUserQuestion,
    TimedUserQuestionResult, UserQuestionAnswerer, UserQuestionError, UserQuestionProjectionView,
    UserQuestionService, UserQuestionState,
};
pub use timed_wait::{
    ClaimEnd, Clock, ManualClock, MonotonicClock, TimedQuestionWait, WaitClaim, WaitCloseReason,
};
