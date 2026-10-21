//! Tool permission and approval: the pre/post decision unions, the pure rules
//! that converge them, and the visibility restriction filter.
//!
//! Ported from `PreToolDecision` / `PostToolDecision`, the `tools/pre-execute`
//! waterfall, the monotonic guard stage, `serviceAsk`, `postExecute`, and
//! `ToolRestriction` of `packages/core/tools/src/index.ts`.
//!
//! ## What is here and what is not
//!
//! The TypeScript pipeline is an async waterfall: listeners await, an approval
//! service can interleave a human decision, and the registry rechecks caller
//! cancellation after every stage settles. Those are **lifetime** concerns —
//! they need an async runtime and a mounted `ApprovalService` — so they stay on
//! the not-ported list. What is ported is every rule with a deterministic
//! answer:
//!
//! - [`converge_pre_tool`] — an ordered chain of votes collapses the way the
//!   waterfall does: the first listener that does not delegate wins, and an
//!   all-delegating chain converges to the registry's `allow` default.
//! - [`apply_guard`] — the monotonic guard stage: it can deny an allowed call
//!   but can never re-allow one the waterfall denied.
//! - [`resolve_ask`] — `ask` without an approval seam degrades to a denial,
//!   and a seam's four outcomes map one-to-one onto allow/deny.
//! - [`evaluate_pre_tool`] — `deny` materializes its reason as a failure,
//!   `cancel` selects the canonical before-dispatch abort.
//! - [`apply_post_tool`] — `accept` keeps or replaces one projection, `block`
//!   turns corrective feedback into a valueless failure.
//! - [`ToolRestriction`] — the compiled allow/deny filter over global names.
//!
//! Deliberately **not** ported, with the reason:
//!
//! - **The pre-execute waterfall itself.** [`converge_pre_tool`] consumes votes
//!   that the caller has already collected; it does not run listeners or await
//!   them. [`PreToolGate`] is therefore the converged, synchronous decision
//!   surface, and a deployment composes the chain behind it. The post-execute
//!   stage *is* run by the scheduler, through the asynchronous
//!   [`PostToolGate`].
//! - **The approval UI round trip.** [`resolve_ask`] maps an
//!   [`ApprovalOutcome`]; mounting the service that produces one is the
//!   caller's job, exactly as `ctx.get('approval')` is the deployment's.
//! - **Caller cancellation overtaking an in-flight approval.** The TypeScript
//!   registry rechecks the caller signal after `serviceAsk` settles and swaps a
//!   denial for the canonical abort when the approval was itself cancelled;
//!   that interleaving needs the async gate, so the scheduler checks
//!   cancellation before the gate instead.
//! - **The agent-less `ask` degradation** ("no agent to route it through"):
//!   the Rust [`ToolCallInput`] carries no agent identity.
//! - **The scope system and the reserved PTC transport.** `restrict()` fails
//!   on an unscoped call and on the reserved `run_code` name; neither concept
//!   exists in the Rust registry yet, so [`validate_restriction`] ports only
//!   the empty-filter and unknown-name rejections.
//! - **Re-rendering a replaced `value`.** The TypeScript post-execute pipeline
//!   re-validates and re-renders a replacement value through the tool's output
//!   declaration; the Rust contract has no output declaration (a handler
//!   renders its own content), so [`apply_post_tool`] replaces the canonical
//!   value and preserves the dispatched content.
//!
//! Known divergences, all low-risk and all in the "Rust is stricter or
//! semantically equal" direction:
//!
//! - **`additionalContexts: []` loses its key.** TypeScript's `JSON.stringify`
//!   keeps an explicitly empty array; the Rust decision drops it via
//!   `skip_serializing_if`. Semantically equal (empty ≡ absent) and TypeScript's
//!   own result building omits empty arrays too (`index.ts:1820,1826`), so a
//!   typed gate would have to deliberately pass `[]` to observe the difference.
//! - **Ill-typed shapes are rejected, not tolerated.** `{"kind":"accept",
//!   "content":null}`, `{"kind":1}`, and a bare array all fail to deserialize
//!   here, where the TypeScript runtime would carry them into the durable log.
//!   A typed producer cannot emit these.
//! - **`DisplayReason` with `"en"` inside `locales` serializes a duplicate
//!   key.** `en` is a named field plus a flattened map; a caller who also puts
//!   `en` in the map gets `{"en":...,"en":...}`. Last-one-wins on parse, which
//!   matches `JSON.parse`, but the wire form is dirty.
//!
//! The scheduler runs every settled result — including a gate denial or
//! cancellation — through [`PostToolGate`] before it becomes durable, so the
//! `tools/post-execute` waterfall and the finalize step have their place in the
//! ordering. A call that cancellation never started is recorded with the
//! canonical abort directly, without post-execute, matching the TypeScript
//! `appendSkippedToolCall`. As with the pre-execute chain, a gate implementation
//! composes the listener chain and answers through [`apply_post_tool`];
//! [`AcceptAllPostTool`] is the all-delegating default.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use dsh_llm::{ContentBlock, CoreContentBlock, UserMessage};
use dsh_session::ToolErrorInfo;

use super::types::{ToolCallInput, ToolExecutionResult, ToolFailure, TOOL_ABORTED_BEFORE_DISPATCH};

/// Pre-dispatch decision: allow, deny, cancel, or ask before the body runs.
///
/// Ported from `PreToolDecision` (index.ts:608). `deny` must carry a `reason`;
/// `cancel` carries no payload and selects the canonical cancellation result
/// without presenting a policy denial; `ask` runs only after an approval
/// service answers, and degrades to a denial when none is mounted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum PreToolDecision {
    /// The call may run its body.
    Allow,
    /// The call is denied; the reason is materialized as the failure the model sees.
    Deny {
        /// Human-readable denial reason, rendered as `Error: {reason}`.
        reason: String,
        /// Structured routing identity, persisted beside the model-facing text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        info: Option<ToolErrorInfo>,
    },
    /// The call is cancelled before dispatch, without a policy denial.
    Cancel,
    /// The call needs approval; its `reason` is the audited approval reason.
    #[serde(rename_all = "camelCase")]
    Ask {
        /// Why approval is required, and the denial reason if it is not granted.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        /// Localized prompt text for an approval UI, `en` plus other locales.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display_reason: Option<DisplayReason>,
    },
}

impl PreToolDecision {
    /// The default waterfall answer: the call may run.
    pub fn allow() -> Self {
        Self::Allow
    }

    /// A denial with a reason and no structured identity.
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny {
            reason: reason.into(),
            info: None,
        }
    }

    /// A denial with a reason and a structured identity.
    pub fn deny_with_info(reason: impl Into<String>, info: ToolErrorInfo) -> Self {
        Self::Deny {
            reason: reason.into(),
            info: Some(info),
        }
    }

    /// The canonical pre-dispatch cancellation.
    pub fn cancel() -> Self {
        Self::Cancel
    }

    /// An approval request with an optional audited reason.
    pub fn ask(reason: Option<&str>) -> Self {
        Self::Ask {
            reason: reason.map(str::to_owned),
            display_reason: None,
        }
    }
}

/// Localized prompt text shown by an approval UI (TS: `displayReason`).
///
/// The English entry is required; every other locale rides along verbatim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisplayReason {
    /// The English prompt; always present.
    pub en: String,
    /// Other locales by BCP-47 tag, serialized inline beside `en`.
    #[serde(flatten)]
    pub locales: BTreeMap<String, String>,
}

/// One listener's answer in an ordered decision chain.
///
/// Ported from the waterfall contract: a listener either returns a decision
/// (short-circuit) or calls `next()` (delegate).
#[derive(Clone, Debug, PartialEq)]
pub enum PreToolVote {
    /// The listener delegated: the chain continues to the next vote.
    Delegate,
    /// The listener short-circuited with this decision.
    Decide(PreToolDecision),
}

impl PreToolVote {
    /// A vote that short-circuits with `decision`.
    pub fn decide(decision: PreToolDecision) -> Self {
        Self::Decide(decision)
    }
}

/// The answer an approval channel gave for one `ask` (TS: `ApprovalOutcome`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApprovalOutcome {
    /// The user allowed this one call.
    AllowedOnce,
    /// The user rejected the call.
    Rejected,
    /// The approval was cancelled before it was answered.
    Cancelled,
    /// No approval channel is available.
    Unavailable,
}

/// An `ask` resolved through the approval seam (TS: `ToolAskResolution`).
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedAsk {
    /// The decision the approval produced: allow, or a denial with its reason.
    pub decision: PreToolDecision,
    /// Whether the approval channel itself reported cancellation.
    pub approval_cancelled: bool,
}

/// The denial reason an `ask` degrades to with no approval seam mounted
/// (TS: the `ctx.get('approval') === undefined` branch of `serviceAsk`).
pub fn ask_denial_reason(tool_name: &str, ask_reason: Option<&str>) -> String {
    ask_reason.map_or_else(
        || format!("tool \"{tool_name}\" requires approval (not yet supported)"),
        ToString::to_string,
    )
}

/// Resolve an `ask` decision through an approval answer (TS: `serviceAsk`).
///
/// `outcome` is `None` when no approval channel answered — the deployment
/// composed no ApprovalService — which degrades to a denial carrying the ask's
/// own reason, or [`ask_denial_reason`] when it gave none. The four outcomes
/// map one-to-one; `Cancelled` is the only one that reports
/// [`ResolvedAsk::approval_cancelled`].
pub fn resolve_ask(
    tool_name: &str,
    ask_reason: Option<&str>,
    outcome: Option<ApprovalOutcome>,
) -> ResolvedAsk {
    let (decision, approval_cancelled) = match outcome {
        None => (
            PreToolDecision::deny(ask_denial_reason(tool_name, ask_reason)),
            false,
        ),
        Some(ApprovalOutcome::AllowedOnce) => (PreToolDecision::Allow, false),
        Some(ApprovalOutcome::Rejected) => (
            PreToolDecision::deny(format!("the user rejected tool \"{tool_name}\"")),
            false,
        ),
        Some(ApprovalOutcome::Cancelled) => (
            PreToolDecision::deny(format!("approval for tool \"{tool_name}\" was cancelled")),
            true,
        ),
        Some(ApprovalOutcome::Unavailable) => (
            PreToolDecision::deny(format!(
                "tool \"{tool_name}\" requires approval, but no approval channel is available"
            )),
            false,
        ),
    };
    ResolvedAsk {
        decision,
        approval_cancelled,
    }
}

/// Collapse one ordered pre-execute chain (TS: `ctx.waterfall(..., allow)`).
///
/// The first vote that does not delegate wins; every later vote is never
/// consulted, exactly as a short-circuited waterfall skips its tail. An
/// all-delegating chain — including an empty one — converges to `default`,
/// which for the registry is [`PreToolDecision::Allow`].
pub fn converge_pre_tool(votes: &[PreToolVote], default: PreToolDecision) -> PreToolDecision {
    votes
        .iter()
        .find_map(|vote| match vote {
            PreToolVote::Delegate => None,
            PreToolVote::Decide(decision) => Some(decision.clone()),
        })
        .unwrap_or(default)
}

/// The monotonic guard stage (TS: `guardReason`, consulted only after an
/// allowing waterfall).
///
/// A guard denial replaces an allow with a reason-carrying denial; a
/// non-allow decision is never re-examined, so no guard can force-allow a call
/// the waterfall denied. A guard that abstains leaves the decision unchanged.
pub fn apply_guard(decision: PreToolDecision, guard_reason: Option<String>) -> PreToolDecision {
    match decision {
        PreToolDecision::Allow => match guard_reason {
            Some(reason) => PreToolDecision::deny(reason),
            None => PreToolDecision::Allow,
        },
        decided => decided,
    }
}

/// What the pre-dispatch gate concluded for one call.
#[derive(Clone, Debug, PartialEq)]
pub enum GateOutcome {
    /// The tool body may run.
    Run,
    /// The body must not run; the call settles as this synthesized failure.
    Denied(ToolExecutionResult),
    /// The body must not run; the call settles as the canonical cancellation.
    Cancelled(ToolExecutionResult),
}

/// The canonical result when a gate cancels a call before its body runs
/// (TS: `toolAbortedBeforeDispatchResult`).
pub fn aborted_before_dispatch_result() -> ToolExecutionResult {
    ToolExecutionResult::failure("tool call aborted before dispatch").with_error_info(
        ToolErrorInfo {
            name: "AbortError".to_string(),
            code: TOOL_ABORTED_BEFORE_DISPATCH.to_string(),
            reason: None,
        },
    )
}

/// Materialize a denial as the failure the model sees (TS: the
/// `materializeFinalResult` call in `prepareExecution`): the reason rendered
/// into `Error: {reason}` content, and the same reason as the failure message.
fn denial_result(reason: &str, info: Option<&ToolErrorInfo>) -> ToolExecutionResult {
    match info {
        Some(info) => ToolExecutionResult::failure(reason).with_error_info(info.clone()),
        None => ToolExecutionResult::failure(reason),
    }
}

/// Map one converged pre-dispatch decision onto what the scheduler does.
///
/// `deny` materializes its reason and optional structured identity as a
/// failure; `cancel` selects [`aborted_before_dispatch_result`]; `ask` degrades
/// to a denial because no approval seam is wired into the scheduler; only
/// `allow` runs the body.
pub fn evaluate_pre_tool(tool_name: &str, decision: &PreToolDecision) -> GateOutcome {
    match decision {
        PreToolDecision::Allow => GateOutcome::Run,
        PreToolDecision::Deny { reason, info } => {
            GateOutcome::Denied(denial_result(reason, info.as_ref()))
        }
        PreToolDecision::Cancel => GateOutcome::Cancelled(aborted_before_dispatch_result()),
        PreToolDecision::Ask { reason, .. } => GateOutcome::Denied(ToolExecutionResult::failure(
            ask_denial_reason(tool_name, reason.as_deref()),
        )),
    }
}

/// The pre-dispatch approval seam the scheduler consults before every body.
///
/// Ported from the `tools/pre-execute` waterfall plus the monotonic
/// `tools.guard()` stage of `ToolRuntime`. The TypeScript pipeline is async;
/// this seam is the synchronous decision surface — a gate answers with a
/// converged [`PreToolDecision`] and the scheduler acts on it immediately.
/// [`converge_pre_tool`], [`apply_guard`], and [`resolve_ask`] are the pure
/// rules a gate implementation composes to produce that answer.
pub trait PreToolGate: Send + Sync {
    /// The converged pre-execute decision for one pending call.
    fn decide(&self, call: &ToolCallInput) -> PreToolDecision;

    /// The first monotonic guard denial, consulted only when `decide` allowed.
    /// Returning `None` leaves an allowed call allowed.
    fn guard(&self, _call: &ToolCallInput) -> Option<String> {
        None
    }
}

/// The default gate: every call is allowed, exactly as a waterfall whose
/// listeners all delegate converges to the registry's `allow` default.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllowAllGate;

impl PreToolGate for AllowAllGate {
    fn decide(&self, _call: &ToolCallInput) -> PreToolDecision {
        PreToolDecision::Allow
    }
}

/// Consult the gate for one call: the waterfall decision first, then — only
/// when it allowed — the monotonic guard stage (TS: `prepareExecution`).
pub fn evaluate_pre_dispatch(gate: &dyn PreToolGate, call: &ToolCallInput) -> GateOutcome {
    let decision = gate.decide(call);
    let decision = if matches!(decision, PreToolDecision::Allow) {
        apply_guard(decision, gate.guard(call))
    } else {
        decision
    };
    evaluate_pre_tool(&call.name, &decision)
}

/// Post-dispatch decision: accept, replace one projection, or block by turning
/// corrective feedback into an error result.
///
/// Ported from `PostToolDecision` (index.ts:618). The two `accept` projections
/// are mutually exclusive — the TypeScript union forbids `content` and `value`
/// together with `value?: never` / `content?: never`, and the runtime rejects a
/// decision carrying both. Rust keeps them apart with [`AcceptPayload`]
/// variants, so "both" is not representable in memory; the wire decoder still
/// rejects it, exactly as the TypeScript runtime check does.
#[derive(Clone, Debug, PartialEq)]
pub enum PostToolDecision {
    /// Accept the dispatched result, optionally replacing one projection.
    Accept {
        /// Which projection the decision replaces; [`AcceptPayload::Plain`]
        /// accepts the result unchanged.
        payload: AcceptPayload,
        /// Contexts to ferry to the next request, after the result's own.
        additional_contexts: Vec<UserMessage>,
    },
    /// Block: the call becomes a valueless failure carrying the feedback.
    Block {
        /// Corrective feedback; becomes the model-facing content verbatim.
        feedback: Vec<ContentBlock>,
        /// The only contexts a block exposes; body-deferred ones are dropped.
        additional_contexts: Vec<UserMessage>,
    },
}

/// The projection an `accept` replaces. Mutually exclusive by construction.
#[derive(Clone, Debug, PartialEq)]
pub enum AcceptPayload {
    /// Accept the result unchanged.
    Plain,
    /// Replace the model-facing content.
    Content(Vec<ContentBlock>),
    /// Replace the canonical value.
    Value(Value),
}

impl PostToolDecision {
    /// Accept the result unchanged.
    pub fn accept() -> Self {
        Self::Accept {
            payload: AcceptPayload::Plain,
            additional_contexts: Vec::new(),
        }
    }

    /// Accept, replacing the model-facing content.
    pub fn accept_content(content: Vec<ContentBlock>) -> Self {
        Self::Accept {
            payload: AcceptPayload::Content(content),
            additional_contexts: Vec::new(),
        }
    }

    /// Accept, replacing the canonical value.
    pub fn accept_value(value: Value) -> Self {
        Self::Accept {
            payload: AcceptPayload::Value(value),
            additional_contexts: Vec::new(),
        }
    }

    /// Block, turning corrective feedback into an error result.
    pub fn block(feedback: Vec<ContentBlock>) -> Self {
        Self::Block {
            feedback,
            additional_contexts: Vec::new(),
        }
    }

    /// Attach contexts for the next request.
    pub fn with_contexts(mut self, contexts: Vec<UserMessage>) -> Self {
        match &mut self {
            Self::Accept {
                additional_contexts,
                ..
            }
            | Self::Block {
                additional_contexts,
                ..
            } => *additional_contexts = contexts,
        }
        self
    }
}

/// The TypeErrors `postExecute` raises, as a typed error.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PostToolError {
    /// An accept cannot replace the value of a failed result.
    #[error("tools/post-execute cannot replace the value of a failed result")]
    ValueReplacementOfFailure,
}

/// Apply one post-execute decision to a dispatch result (TS: `postExecute`).
///
/// `accept` keeps the call successful, replacing at most one projection, and
/// appends the decision's contexts after the result's own. `block` turns the
/// call into a valueless failure whose content is the corrective feedback and
/// whose only contexts are the ones the decision supplied — a block discards
/// the contexts the body deferred.
///
/// A value replacement on a success preserves the dispatched content: the
/// TypeScript pipeline re-renders through the tool's output declaration, which
/// the Rust contract does not model (a handler renders its own content).
pub fn apply_post_tool(
    decision: &PostToolDecision,
    result: &ToolExecutionResult,
) -> Result<ToolExecutionResult, PostToolError> {
    match decision {
        PostToolDecision::Accept {
            payload,
            additional_contexts,
        } => {
            let mut contexts = result.additional_contexts().to_vec();
            contexts.extend(additional_contexts.iter().cloned());
            match payload {
                AcceptPayload::Plain => Ok(result.clone().with_contexts(contexts)),
                AcceptPayload::Content(content) => Ok(match result {
                    ToolExecutionResult::Success {
                        value,
                        meta,
                        concludes_turn,
                        ..
                    } => ToolExecutionResult::Success {
                        value: value.clone(),
                        content: content.clone(),
                        meta: meta.clone(),
                        additional_contexts: contexts,
                        concludes_turn: *concludes_turn,
                    },
                    ToolExecutionResult::Failure { error, meta, .. } => {
                        ToolExecutionResult::Failure {
                            error: error.clone(),
                            content: content.clone(),
                            meta: meta.clone(),
                            additional_contexts: contexts,
                        }
                    }
                }),
                AcceptPayload::Value(value) => {
                    let ToolExecutionResult::Success {
                        content,
                        meta,
                        concludes_turn,
                        ..
                    } = result
                    else {
                        return Err(PostToolError::ValueReplacementOfFailure);
                    };
                    Ok(ToolExecutionResult::Success {
                        value: value.clone(),
                        content: content.clone(),
                        meta: meta.clone(),
                        additional_contexts: contexts,
                        concludes_turn: *concludes_turn,
                    })
                }
            }
        }
        PostToolDecision::Block {
            feedback,
            additional_contexts,
        } => Ok(ToolExecutionResult::Failure {
            error: ToolFailure::new(failure_message_from_content(feedback)),
            content: feedback.clone(),
            meta: None,
            additional_contexts: additional_contexts.clone(),
        }),
    }
}

/// Derive one failure message from policy feedback without changing its
/// rendered blocks (TS: `failureMessageFromContent`): text blocks join with
/// newlines, other blocks render as `[{type} content]`, and empty feedback
/// falls back to the generic blocked message.
pub fn failure_message_from_content(content: &[ContentBlock]) -> String {
    let message = content
        .iter()
        .map(|block| match block {
            ContentBlock::Core(CoreContentBlock::Text { text }) => text.clone(),
            other => format!("[{} content]", other.block_type()),
        })
        .collect::<Vec<_>>()
        .join("\n");
    if message.is_empty() {
        "tool result blocked by post-execute policy".to_string()
    } else {
        message
    }
}

/// The post-execute policy seam the scheduler consults for every settled
/// result.
///
/// Ported from the `tools/post-execute` waterfall plus the registry's
/// `finalize` step. The scheduler calls [`PostToolGate::decide`] on every
/// result — a body's own result, an unknown-tool failure, and a gate denial or
/// cancellation — and then [`PostToolGate::finalize`] before committing it, in
/// model order. A call skipped by cancellation never reaches this seam; the
/// scheduler records its canonical abort directly. A deployment composes its
/// listener chain behind this seam and produces the converged
/// [`PostToolDecision`]; the pure rules are [`apply_post_tool`] and
/// [`failure_message_from_content`].
///
/// The decision is async because a listener may await an external service,
/// while finalization is a synchronous content transformation: it never
/// changes whether the call failed, only what the model reads.
#[async_trait]
pub trait PostToolGate: Send + Sync {
    /// The converged post-execute decision for one settled result.
    ///
    /// Returning [`PostToolDecision::accept`] keeps the result as dispatched;
    /// the other variants replace a projection or turn the result into a
    /// valueless failure.
    async fn decide(&self, call: &ToolCallInput, result: &ToolExecutionResult) -> PostToolDecision;

    /// Replace the model-facing content after the decision applied, or `None`
    /// to keep the content the decision produced.
    fn finalize(
        &self,
        _call: &ToolCallInput,
        _result: &ToolExecutionResult,
    ) -> Option<Vec<ContentBlock>> {
        None
    }
}

/// The default post-execute policy: accept every result unchanged, exactly as
/// a waterfall whose listeners all delegate converges to the registry default.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptAllPostTool;

#[async_trait]
impl PostToolGate for AcceptAllPostTool {
    async fn decide(
        &self,
        _call: &ToolCallInput,
        _result: &ToolExecutionResult,
    ) -> PostToolDecision {
        PostToolDecision::accept()
    }
}

impl Serialize for PostToolDecision {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Accept {
                payload,
                additional_contexts,
            } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("kind", "accept")?;
                match payload {
                    AcceptPayload::Plain => {}
                    AcceptPayload::Content(content) => map.serialize_entry("content", content)?,
                    AcceptPayload::Value(value) => map.serialize_entry("value", value)?,
                }
                if !additional_contexts.is_empty() {
                    map.serialize_entry("additionalContexts", additional_contexts)?;
                }
                map.end()
            }
            Self::Block {
                feedback,
                additional_contexts,
            } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("kind", "block")?;
                map.serialize_entry("feedback", feedback)?;
                if !additional_contexts.is_empty() {
                    map.serialize_entry("additionalContexts", additional_contexts)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for PostToolDecision {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let mut map = value
            .as_object()
            .cloned()
            .ok_or_else(|| serde::de::Error::custom("post-tool decision must be a JSON object"))?;
        let kind = map
            .remove("kind")
            .and_then(|kind| kind.as_str().map(str::to_owned))
            .ok_or_else(|| {
                serde::de::Error::custom("post-tool decision requires a string `kind`")
            })?;
        let contexts = map
            .remove("additionalContexts")
            .map(|value| {
                serde_json::from_value::<Vec<UserMessage>>(value).map_err(serde::de::Error::custom)
            })
            .transpose()?
            .unwrap_or_default();
        match kind.as_str() {
            "accept" => {
                let has_value = map.contains_key("value");
                let has_content = map.contains_key("content");
                if has_value && has_content {
                    return Err(serde::de::Error::custom(
                        "tools/post-execute accept decision cannot replace both value and content",
                    ));
                }
                let payload = if has_value {
                    AcceptPayload::Value(
                        serde_json::from_value(map.remove("value").unwrap_or(Value::Null))
                            .map_err(serde::de::Error::custom)?,
                    )
                } else if has_content {
                    AcceptPayload::Content(
                        serde_json::from_value(map.remove("content").unwrap_or(Value::Null))
                            .map_err(serde::de::Error::custom)?,
                    )
                } else {
                    AcceptPayload::Plain
                };
                Ok(Self::Accept {
                    payload,
                    additional_contexts: contexts,
                })
            }
            "block" => {
                let feedback = map.remove("feedback").ok_or_else(|| {
                    serde::de::Error::custom("a block decision requires `feedback`")
                })?;
                let feedback =
                    serde_json::from_value(feedback).map_err(serde::de::Error::custom)?;
                Ok(Self::Block {
                    feedback,
                    additional_contexts: contexts,
                })
            }
            other => Err(serde::de::Error::custom(format!(
                "unknown post-tool decision kind `{other}`"
            ))),
        }
    }
}

/// One compiled per-scope filter over global tool names.
///
/// Ported from `ToolRestriction` / `CompiledToolRestriction`: the name sets are
/// snapshotted at registration, so a later mutation of the caller's arrays is
/// not observed, and filters compose by intersection — a name stays visible
/// only when every filter on the chain admits it. Scoped registrations and the
/// reserved presentation transport sit outside the filter on the TypeScript
/// side; the Rust registry has no scope layers yet, so this type is the pure
/// `admits` rule only.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolRestriction {
    allow: Option<BTreeSet<String>>,
    deny: Option<BTreeSet<String>>,
}

impl ToolRestriction {
    /// Compile a filter from optional name lists. `None` means "no opinion":
    /// an absent allow-list admits every name, an absent deny-list removes
    /// none. An empty list is a real answer — an empty allow-list admits
    /// nothing, an empty deny-list removes nothing.
    pub fn new(allow: Option<Vec<String>>, deny: Option<Vec<String>>) -> Self {
        Self {
            allow: allow.map(|names| names.into_iter().collect()),
            deny: deny.map(|names| names.into_iter().collect()),
        }
    }

    /// Whether this filter admits `name` (TS: `ToolLayer::admits`).
    pub fn admits(&self, name: &str) -> bool {
        if self
            .allow
            .as_ref()
            .is_some_and(|allow| !allow.contains(name))
        {
            return false;
        }
        !self.deny.as_ref().is_some_and(|deny| deny.contains(name))
    }
}

/// Whether every filter admits `name`: restrictions intersect across the
/// chain, so any one scope's filter may mask an inherited name.
pub fn admits_all(filters: &[ToolRestriction], name: &str) -> bool {
    filters.iter().all(|filter| filter.admits(name))
}

/// Registration-time rejection of a filter (TS: `restrict()` fails loud
/// rather than silently masking nothing).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RestrictionError {
    /// A filter naming neither `allow` nor `deny` is almost always a
    /// materialized-empty-config bug.
    #[error("tools.restrict({{}}) is a no-op: pass `allow` and/or `deny` (an empty filter is almost always a materialized-empty-config bug)")]
    EmptyFilter,
    /// The filter names a tool the scope does not inherit.
    #[error(
        "tools.restrict() names unknown global tool{plural} {names}; known global tools: {known}"
    )]
    UnknownTools {
        plural: &'static str,
        names: String,
        known: String,
    },
}

/// Validate one raw filter against the global names it may reference
/// (TS: the validation half of `restrict()`).
///
/// The unknown-name list keeps caller order — allow entries first, then deny,
/// duplicates included — and the known list is sorted, exactly as the
/// TypeScript message renders them. The unscoped-call and reserved-transport
/// rejections are not ported: the Rust registry has no scope layers and no PTC
/// transport yet.
pub fn validate_restriction(
    allow: Option<&[String]>,
    deny: Option<&[String]>,
    known: &BTreeSet<String>,
) -> Result<(), RestrictionError> {
    if allow.is_none() && deny.is_none() {
        return Err(RestrictionError::EmptyFilter);
    }
    let unknown: Vec<&String> = allow
        .unwrap_or_default()
        .iter()
        .chain(deny.unwrap_or_default())
        .filter(|name| !known.contains(*name))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    let plural = if unknown.len() > 1 { "s" } else { "" };
    let names = unknown
        .iter()
        .map(|name| format!("{name:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    let known = if known.is_empty() {
        "(none)".to_string()
    } else {
        known.iter().cloned().collect::<Vec<_>>().join(", ")
    };
    Err(RestrictionError::UnknownTools {
        plural,
        names,
        known,
    })
}
