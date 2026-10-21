//! The `exit_plan_mode` tool: description, schemas, call validation, the review
//! question, the pure post-review decision, and presentation.
//!
//! Ported from the `EXIT_PLAN_MODE` registration in
//! `packages/plan/plan-mode/src/index.ts`. The tool is deliberately still
//! registered while plan mode is inactive, so entering or leaving plan mode
//! changes only the prompt section, never the request tool catalog; the
//! execution-time check is what rejects a call outside plan mode.
//!
//! ## What is pure and what is not
//!
//! The review itself is a human interaction: asking, awaiting an answer, and
//! honouring cancellation belong to the host's user-questions channel. This
//! crate owns everything around it:
//!
//! - [`validate_exit_plan`] and [`require_review_channel`] — the pre-ask checks,
//!   in the TypeScript order and with its exact messages.
//! - [`plan_review_question`] — the review question, options, and presentation
//!   intent the host asks.
//! - [`classify_review`] — the answer rule: exactly one `Approve` selection and
//!   no custom text is consent; anything else is feedback.
//! - [`decide_exit`] — the post-review decision, including the dismissed-review
//!   success that concludes the turn and the reloaded-service failure.
//!
//! The approved decision returns a pending `false` selection instead of
//! appending: the switch is boundary-applied, so plan guidance covers any
//! remaining call of the same assistant batch.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use dsh_llm::{ContentBlock, ToolCallId};
use dsh_tools::{ToolExecutionResult, ToolRestriction};

use super::text::{is_js_whitespace, js_trim};

/// The model-facing exit tool's name.
pub const EXIT_PLAN_MODE: &str = "exit_plan_mode";

/// The model-facing tool description.
pub const EXIT_DESCRIPTION: &str = "Use only in plan mode. Present your plan for the user's review and, on approval, leave plan mode. The user may approve (carry out the plan from your next step) or keep planning — their feedback comes back in the tool result; revise and present again.";

/// The plan argument's description.
pub const EXIT_PLAN_ARGUMENT_DESCRIPTION: &str =
    "The complete plan, as markdown, starting with a # heading that names it.";

/// The review question's id, echoed in the answer this tool reads.
pub const REVIEW_ID: &str = "plan-review";

/// The review question's header.
pub const REVIEW_HEADER: &str = "Plan review";

/// The review question's prompt.
pub const REVIEW_QUESTION: &str = "Approve this plan and leave plan mode?";

/// The review question's approve option label.
pub const APPROVE_LABEL: &str = "Approve";

/// The review question's keep-planning option label.
pub const KEEP_PLANNING_LABEL: &str = "Keep planning";

/// Description of the approve option.
pub const APPROVE_OPTION_DESCRIPTION: &str =
    "Leave plan mode; the plan is carried out from the next step.";

/// Description of the keep-planning option.
pub const KEEP_PLANNING_OPTION_DESCRIPTION: &str =
    "Stay in plan mode; feedback goes back to the model.";

/// Presentation-intent kind that marks the question as a plan review.
pub const PLAN_REVIEW_INTENT: &str = "plan-review";

/// Model-facing result when the plan is approved.
pub const EXIT_APPROVED_TEXT: &str =
    "Plan approved — plan mode exited; carry out the plan starting with your next step.";

/// Model-facing result when the user dismissed the review.
pub const EXIT_DISMISSED_TEXT: &str =
    "The user dismissed the plan review to reply in their own words; plan mode remains active.";

/// Model-facing error when the user keeps planning without feedback.
pub const EXIT_KEEP_PLANNING_TEXT: &str =
    "The user chose to keep planning; revise the plan and present it again.";

/// The compiled raw JSON Schema of `exit_plan_mode`'s arguments.
pub fn parameter_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": EXIT_PLAN_ARGUMENT_DESCRIPTION
            }
        },
        "required": ["plan"]
    })
}

/// The compiled raw JSON Schema of `exit_plan_mode`'s canonical value.
pub fn output_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "approved": { "type": "boolean" }
        },
        "required": ["approved"]
    })
}

/// The canonical value of one `exit_plan_mode` success.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExitPlanModeOutput {
    /// Whether the user approved the plan.
    pub approved: bool,
}

impl ExitPlanModeOutput {
    /// This value as the JSON object the tool returns.
    pub fn to_value(self) -> Value {
        json!({ "approved": self.approved })
    }
}

/// Why an `exit_plan_mode` call cannot present a plan.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ExitPlanModeError {
    /// A caller without an agent has no session to switch.
    #[error("exit_plan_mode requires a calling agent (no session to switch)")]
    NoCallingAgent,
    /// The tool is advertised outside plan mode but only usable inside it.
    #[error("exit_plan_mode is only available in plan mode")]
    NotInPlanMode,
    /// The plan is empty or does not start with a `#` heading.
    #[error("exit_plan_mode requires a non-empty markdown plan starting with a # heading")]
    InvalidPlan,
    /// No user-questions channel is composed to review the plan.
    #[error(
        "no user-questions channel is available to review the plan; ask the user to switch the session mode instead"
    )]
    NoReviewChannel,
    /// The plugin was unloaded while the review awaited; the flush listener is
    /// gone, so a success would claim an exit that can never land.
    #[error(
        "the plan-mode service was reloaded while the plan was under review; present the plan again"
    )]
    Reloaded,
    /// The user chose to keep planning without feedback.
    #[error("The user chose to keep planning; revise the plan and present it again.")]
    KeepPlanning,
    /// The user chose to keep planning and left feedback.
    #[error("The user chose to keep planning; their feedback: {feedback}")]
    KeepPlanningWithFeedback {
        /// The feedback verbatim.
        feedback: String,
    },
}

impl ExitPlanModeError {
    /// The keep-planning error for one feedback string, generic when empty.
    pub fn keep_planning(feedback: &str) -> Self {
        if feedback.is_empty() {
            Self::KeepPlanning
        } else {
            Self::KeepPlanningWithFeedback {
                feedback: feedback.to_string(),
            }
        }
    }
}

/// Whether one plan starts with a `#` heading, as the exit tool requires.
///
/// The TypeScript rule is `/^#\s+\S/` on the trimmed plan: exactly one `#`,
/// then at least one whitespace character, then a non-whitespace character.
/// A `##` heading is therefore rejected even though [`first_heading`] accepts
/// it for presentation.
pub fn valid_plan_heading(plan: &str) -> bool {
    let Some(rest) = js_trim(plan).strip_prefix('#') else {
        return false;
    };
    let mut chars = rest.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !is_js_whitespace(first) {
        return false;
    }
    chars.any(|ch| !is_js_whitespace(ch))
}

/// The pre-ask checks, in the TypeScript order.
///
/// The calling agent is checked first, then the logged mode, then the plan. A
/// rejected call never asks the reviewer and never changes the mode.
pub fn validate_exit_plan(
    has_agent: bool,
    logged_active: bool,
    plan: &str,
) -> Result<(), ExitPlanModeError> {
    if !has_agent {
        return Err(ExitPlanModeError::NoCallingAgent);
    }
    if !logged_active {
        return Err(ExitPlanModeError::NotInPlanMode);
    }
    if !valid_plan_heading(plan) {
        return Err(ExitPlanModeError::InvalidPlan);
    }
    Ok(())
}

/// Reject a review when no user-questions channel is composed.
pub fn require_review_channel(has_channel: bool) -> Result<(), ExitPlanModeError> {
    if has_channel {
        Ok(())
    } else {
        Err(ExitPlanModeError::NoReviewChannel)
    }
}

/// One option offered by the review question.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewOption {
    /// The label the answer echoes.
    pub label: String,
    /// What choosing the option does.
    pub description: String,
}

/// Presentation-only metadata that marks a question as a plan review.
///
/// A capable UI renders the plan as a review decision and answers with one of
/// the question's own labels either way.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanReviewIntent {
    /// Always [`PLAN_REVIEW_INTENT`].
    pub kind: String,
    /// The label that means consent; one of this question's options.
    pub approve: String,
    /// The tool call awaiting the answer.
    pub call_id: ToolCallId,
}

/// The review question the exit tool asks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanReviewQuestion {
    /// Always [`REVIEW_ID`].
    pub id: String,
    /// Always [`REVIEW_HEADER`].
    pub header: String,
    /// Always [`REVIEW_QUESTION`].
    pub question: String,
    /// The complete plan, verbatim.
    pub detail: String,
    /// Approve, then keep planning.
    pub options: Vec<ReviewOption>,
    /// Presentation intent naming the approve option.
    pub intent: PlanReviewIntent,
}

/// Build the review question for one plan and call.
pub fn plan_review_question(plan: &str, call_id: &ToolCallId) -> PlanReviewQuestion {
    PlanReviewQuestion {
        id: REVIEW_ID.to_string(),
        header: REVIEW_HEADER.to_string(),
        question: REVIEW_QUESTION.to_string(),
        detail: plan.to_string(),
        options: vec![
            ReviewOption {
                label: APPROVE_LABEL.to_string(),
                description: APPROVE_OPTION_DESCRIPTION.to_string(),
            },
            ReviewOption {
                label: KEEP_PLANNING_LABEL.to_string(),
                description: KEEP_PLANNING_OPTION_DESCRIPTION.to_string(),
            },
        ],
        intent: PlanReviewIntent {
            kind: PLAN_REVIEW_INTENT.to_string(),
            approve: APPROVE_LABEL.to_string(),
            call_id: call_id.clone(),
        },
    }
}

/// One answer item from the user-questions channel.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewAnswerItem {
    /// The question id the answer belongs to.
    pub id: String,
    /// The selected option labels.
    pub selected: Vec<String>,
    /// Free-form text the user typed instead of, or beside, the options.
    #[serde(default)]
    pub custom: Option<String>,
}

/// One review answer.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewAnswer {
    /// The per-question answers.
    pub answers: Vec<ReviewAnswerItem>,
}

/// The post-review input to [`decide_exit`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExitReview {
    /// The user approved the plan.
    Approved,
    /// The user kept planning, with feedback that may be empty.
    KeepPlanning {
        /// The user's free-form feedback.
        feedback: String,
    },
    /// The user dismissed the review to reply in their own words.
    Dismissed,
    /// The review failed for a reason other than dismissal.
    Failed {
        /// The failure message the model sees after the `Error: ` envelope.
        message: String,
    },
}

/// Classify one review answer.
///
/// Consent is exactly one item with id [`REVIEW_ID`], exactly one selection,
/// that selection is [`APPROVE_LABEL`], and no custom text. Everything else is
/// keep-planning: duplicate items, a missing item, extra selections, and custom
/// text beside `Approve` are all feedback, never consent.
pub fn classify_review(answer: &ReviewAnswer) -> ExitReview {
    let items: Vec<&ReviewAnswerItem> = answer
        .answers
        .iter()
        .filter(|item| item.id == REVIEW_ID)
        .collect();
    let item = if items.len() == 1 {
        items.first().copied()
    } else {
        None
    };
    if let Some(item) = item {
        if item.selected.len() == 1
            && item
                .selected
                .first()
                .is_some_and(|label| label == APPROVE_LABEL)
            && item.custom.is_none()
        {
            return ExitReview::Approved;
        }
    }
    ExitReview::KeepPlanning {
        feedback: item
            .and_then(|item| item.custom.clone())
            .unwrap_or_default(),
    }
}

/// What one reviewed exit call produced.
pub struct ExitDecision {
    /// The model-facing result.
    pub result: ToolExecutionResult,
    /// The boundary-applied selection to park; `Some(false)` on approval.
    pub pending: Option<bool>,
}

/// Decide one exit call after the review settled.
///
/// An approval parks `{ active: false, narrate: false }` instead of appending,
/// so the mode flips at the next accepted in-turn pre-step and the result
/// already narrates the transition. A dismissed review is a success that
/// concludes the turn: the user took the turn back to say something the two
/// options do not cover, so waiting for another model request would be wrong.
/// Every other failure leaves plan mode active and keeps its own message.
pub fn decide_exit(review: &ExitReview, disposed: bool) -> ExitDecision {
    match review {
        ExitReview::Approved => {
            if disposed {
                return failure(ExitPlanModeError::Reloaded);
            }
            ExitDecision {
                result: ToolExecutionResult::success(vec![ContentBlock::text(EXIT_APPROVED_TEXT)])
                    .with_value(ExitPlanModeOutput { approved: true }.to_value()),
                pending: Some(false),
            }
        }
        ExitReview::Dismissed => ExitDecision {
            result: ToolExecutionResult::success(vec![ContentBlock::text(EXIT_DISMISSED_TEXT)])
                .with_value(ExitPlanModeOutput { approved: false }.to_value())
                .concluding_turn(),
            pending: None,
        },
        ExitReview::KeepPlanning { feedback } => {
            failure(ExitPlanModeError::keep_planning(feedback))
        }
        ExitReview::Failed { message } => ExitDecision {
            result: ToolExecutionResult::failure(message.clone()),
            pending: None,
        },
    }
}

/// A failed decision with no pending selection.
fn failure(error: ExitPlanModeError) -> ExitDecision {
    ExitDecision {
        result: ToolExecutionResult::failure(error.to_string()),
        pending: None,
    }
}

/// The plan's first markdown heading, any level, or `None` when it has none.
///
/// Ported from the TypeScript `/^#{1,6}\s+(.+?)\s*$/` line scan. Presentation
/// only: the exit call itself requires a single `#` heading.
pub fn first_heading(plan: &str) -> Option<String> {
    for line in plan.split('\n') {
        let mut count = 0;
        let mut rest_start = 0;
        for (index, ch) in line.char_indices() {
            if ch == '#' && count < 6 {
                count += 1;
                rest_start = index + ch.len_utf8();
            } else {
                break;
            }
        }
        if count == 0 {
            continue;
        }
        let rest = &line[rest_start..];
        let Some(first) = rest.chars().next() else {
            continue;
        };
        if !is_js_whitespace(first) {
            continue;
        }
        if rest.chars().all(is_js_whitespace) {
            // `\s+(.+?)\s*$` backtracks one whitespace character into the group,
            // which needs at least one character beyond `\s+` to exist.
            let mut chars = rest.chars();
            chars.next()?;
            return chars.last().map(|last| last.to_string());
        }
        return Some(rest.trim_matches(is_js_whitespace).to_string());
    }
    None
}

/// The plan-mode tool-visibility restriction.
///
/// Plan mode has none: the TypeScript plugin keeps the full request tool
/// catalog across both modes and restrains through guidance alone. The exit
/// tool is registered in both modes for the same reason — the catalog must not
/// change when the mode does. Returning `None` makes that explicit for callers
/// that compose restriction chains, instead of leaving them to guess whether an
/// absent filter means "no opinion" or "not wired yet".
pub fn tool_restriction() -> Option<ToolRestriction> {
    None
}

/// The pending-state presentation of one exit call.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExitCallPresentation {
    /// The generic card kind.
    pub card: String,
    /// The plan's first heading, or `Plan`.
    pub title: String,
    /// The tool category.
    pub kind: String,
    /// The complete plan as one text block.
    pub content: Vec<ContentBlock>,
}

/// Build the pending-state presentation for one exit call.
pub fn present_call(plan: &str) -> ExitCallPresentation {
    ExitCallPresentation {
        card: "generic".to_string(),
        title: first_heading(plan).unwrap_or_else(|| "Plan".to_string()),
        kind: "other".to_string(),
        content: vec![ContentBlock::text(plan)],
    }
}

/// The completed-state presentation of one exit call.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExitResultPresentation {
    /// The generic card kind.
    pub card: String,
    /// Always `Plan review`.
    pub title: String,
    /// The result's model-facing content.
    pub content: Vec<ContentBlock>,
}

/// Build the completed-state presentation from a result's content.
pub fn present_result(content: &[ContentBlock]) -> ExitResultPresentation {
    ExitResultPresentation {
        card: "generic".to_string(),
        title: REVIEW_HEADER.to_string(),
        content: content.to_vec(),
    }
}
