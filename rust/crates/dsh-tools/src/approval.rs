//! Async approval seam: the mounted `ApprovalService`, the approval waterfall
//! (listeners voting allow/deny), and cancel/timeout reconciliation of an
//! in-flight approval.
//!
//! This is the async half of `crate::gate` that §3.6 / §5 left unported: the
//! pure rules (`resolve_ask`, `converge_pre_tool`, `apply_guard`,
//! `evaluate_pre_tool`, `apply_post_tool`) stay the synchronous decision core,
//! and this module is the seam that produces the [`ApprovalOutcome`] those rules
//! consume.
//!
//! ## Model
//!
//! The TypeScript pipeline calls `ctx.get('approval').request({ ... signal })`
//! when a pre-execute decision is `ask`, then maps the four outcomes
//! (`allowed-once` / `rejected` / `cancelled` / `unavailable`) one-to-one onto
//! allow/deny. The request carries an `AbortSignal`; when the surrounding
//! request is cancelled the pending approval reconciles as `cancelled` and the
//! registry swaps a denial for the canonical before-dispatch abort.
//!
//! [`ApprovalService`] is the object-safe async trait (like `LlmProvider`) that
//! a deployment mounts. [`WaterfallApprovalService`] composes a chain of
//! [`ApprovalListener`]s — each may [`Delegate`](ApprovalVote::Delegate),
//! [`Allow`](ApprovalVote::Allow), or [`Deny`](ApprovalVote::Deny) — and a
//! fallback [`ApprovalService`] backend (the human UI). The first
//! non-delegating listener wins, exactly like
//! [`crate::gate::converge_pre_tool`]; when every listener delegates the backend
//! is consulted. The whole round-trip races the caller [`CancelSignal`]
//! (→ [`Cancelled`](ApprovalResult::Cancelled)) and an optional timeout
//! (→ [`Timeout`](ApprovalResult::Timeout)), and dropping the winning future
//! cancels the in-flight approval.
//!
//! ## Determinism without a UI
//!
//! Tests use scripted backends — [`ConstApprovalService`], [`FnApprovalService`],
//! and [`PendingApprovalService`] — so the waterfall, the ask→allow/deny
//! mapping, the timeout, and the cancel reconciliation all run with no real UI.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dsh_llm::ToolCallId;

use crate::cancel::CancelSignal;
use crate::gate::{resolve_ask, ApprovalOutcome, DisplayReason, PreToolDecision, ResolvedAsk};
use crate::types::ToolCallInput;

/// What an approval round-trip produced, richer than the wire
/// [`ApprovalOutcome`] because the service also observes timeouts.
///
/// `Timeout` is a Rust-only outcome: the TypeScript `AbortSignal` models only
/// cancellation, but a mounted service may also give up on an unanswered
/// request. It degrades to a denial, exactly like `unavailable`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApprovalResult {
    /// The user allowed this one call.
    AllowedOnce,
    /// The call was denied, optionally with a listener-supplied reason.
    Rejected { reason: Option<String> },
    /// The in-flight approval was cancelled (the caller signal fired).
    Cancelled,
    /// No approval channel could answer.
    Unavailable,
    /// The approval timed out before it was answered.
    Timeout,
}

/// One approval request handed to a mounted [`ApprovalService`].
///
/// Ported from the argument object of `approval.request({ ... })` in
/// `packages/core/tools/src/index.ts`; `agent` is omitted because
/// [`ToolCallInput`] carries no agent identity.
#[derive(Clone)]
pub struct ApprovalRequest<'a> {
    /// The requested tool name.
    pub tool_name: &'a str,
    /// The call identity pairing the call with its result.
    pub call_id: &'a ToolCallId,
    /// Why approval is required, and the denial reason if it is not granted.
    pub reason: Option<&'a str>,
    /// Localized prompt text for an approval UI.
    pub display_reason: Option<&'a DisplayReason>,
    /// The caller's cancellation signal; firing it reconciles this request as
    /// [`Cancelled`](ApprovalResult::Cancelled).
    pub signal: Option<&'a CancelSignal>,
}

/// The mounted approval seam (TS: `ctx.get('approval')`).
///
/// Object-safe and asynchronous, like `LlmProvider`: a deployment mounts one
/// implementation (a human UI, a scripted backend, a waterfall) and the gate
/// awaits [`request`](ApprovalService::request) whenever a pre-execute decision
/// is `ask`.
#[async_trait]
pub trait ApprovalService: Send + Sync {
    /// Decide one approval request, resolving to an [`ApprovalResult`].
    async fn request(&self, req: ApprovalRequest<'_>) -> ApprovalResult;
}

/// A single listener's vote in the approval waterfall.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApprovalVote {
    /// Delegate to the next listener (or the UI backend if all delegate).
    Delegate,
    /// Allow this one call, short-circuiting the backend.
    Allow,
    /// Deny this call with a reason the model will see.
    Deny { reason: String },
}

/// One listener in an [`ApprovalService`] waterfall.
///
/// Listeners run in registration order; the first non-`Delegate` vote wins, so
/// a `Deny` hard-stops the call and an `Allow` bypasses the UI. A listener may
/// itself await an external service.
#[async_trait]
pub trait ApprovalListener: Send + Sync {
    /// Vote on one approval request.
    async fn vote(&self, req: &ApprovalRequest<'_>) -> ApprovalVote;
}

/// Resolve once cancellation is raised, or never when there is no signal.
async fn wait_cancel(signal: Option<&CancelSignal>) {
    match signal {
        Some(signal) => signal.cancelled().await,
        None => std::future::pending().await,
    }
}

/// Resolve after `timeout`, or never when there is none.
async fn wait_timeout(timeout: Option<Duration>) {
    match timeout {
        Some(timeout) => tokio::time::sleep(timeout).await,
        None => std::future::pending().await,
    }
}

/// An [`ApprovalService`] that runs a waterfall of [`ApprovalListener`]s and
/// falls back to a backend, racing cancellation and an optional timeout.
#[derive(Clone)]
pub struct WaterfallApprovalService {
    listeners: Vec<Arc<dyn ApprovalListener>>,
    backend: Arc<dyn ApprovalService>,
    timeout: Option<Duration>,
}

impl WaterfallApprovalService {
    /// A waterfall with no listeners that defers entirely to `backend`.
    pub fn new(backend: Arc<dyn ApprovalService>) -> Self {
        Self {
            listeners: Vec::new(),
            backend,
            timeout: None,
        }
    }

    /// Register a listener at the tail of the waterfall.
    pub fn register(mut self, listener: Arc<dyn ApprovalListener>) -> Self {
        self.listeners.push(listener);
        self
    }

    /// Resolve an unanswered request as [`Timeout`](ApprovalResult::Timeout)
    /// after `timeout` elapses.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Run the listener waterfall, then the backend; the first non-delegating
    /// listener wins, mirroring [`crate::gate::converge_pre_tool`].
    async fn run_decision(&self, req: &ApprovalRequest<'_>) -> ApprovalResult {
        for listener in &self.listeners {
            match listener.vote(req).await {
                ApprovalVote::Delegate => continue,
                ApprovalVote::Allow => return ApprovalResult::AllowedOnce,
                ApprovalVote::Deny { reason } => {
                    return ApprovalResult::Rejected {
                        reason: Some(reason),
                    }
                }
            }
        }
        self.backend.request(req.clone()).await
    }
}

#[async_trait]
impl ApprovalService for WaterfallApprovalService {
    async fn request(&self, req: ApprovalRequest<'_>) -> ApprovalResult {
        let decision = self.run_decision(&req);
        tokio::pin!(decision);
        // `biased` so a completed decision is preferred over a simultaneously
        // ready cancel/timeout; cancel and timeout can only overtake a still
        // pending approval, dropping `decision` so the in-flight request is
        // reconciled as cancelled.
        tokio::select! {
            biased;
            result = decision => result,
            _ = wait_cancel(req.signal) => ApprovalResult::Cancelled,
            _ = wait_timeout(self.timeout) => ApprovalResult::Timeout,
        }
    }
}

/// An approval backend that always returns the same result.
pub struct ConstApprovalService {
    result: ApprovalResult,
}

impl ConstApprovalService {
    /// A backend that always answers `result`.
    pub fn new(result: ApprovalResult) -> Self {
        Self { result }
    }
}

#[async_trait]
impl ApprovalService for ConstApprovalService {
    async fn request(&self, _req: ApprovalRequest<'_>) -> ApprovalResult {
        self.result.clone()
    }
}

/// An approval backend driven by a closure, for scripted tests.
pub struct FnApprovalService<F> {
    f: F,
}

impl<F> FnApprovalService<F>
where
    F: Fn(&ApprovalRequest<'_>) -> ApprovalResult + Send + Sync + 'static,
{
    /// A backend that answers by calling `f` with the request.
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

#[async_trait]
impl<F> ApprovalService for FnApprovalService<F>
where
    F: Fn(&ApprovalRequest<'_>) -> ApprovalResult + Send + Sync + 'static,
{
    async fn request(&self, req: ApprovalRequest<'_>) -> ApprovalResult {
        (self.f)(&req)
    }
}

/// An approval backend that never resolves, so a caller can race it against
/// cancellation or a timeout (see the reconciliation tests).
pub struct PendingApprovalService;

#[async_trait]
impl ApprovalService for PendingApprovalService {
    async fn request(&self, _req: ApprovalRequest<'_>) -> ApprovalResult {
        std::future::pending().await
    }
}

/// An [`ApprovalListener`] that always allows the call.
pub struct AllowListener;

#[async_trait]
impl ApprovalListener for AllowListener {
    async fn vote(&self, _req: &ApprovalRequest<'_>) -> ApprovalVote {
        ApprovalVote::Allow
    }
}

/// An [`ApprovalListener`] that always denies with `reason`.
pub struct DenyListener {
    /// The denial reason the model will see.
    pub reason: String,
}

impl DenyListener {
    /// A listener that denies with `reason`.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

#[async_trait]
impl ApprovalListener for DenyListener {
    async fn vote(&self, _req: &ApprovalRequest<'_>) -> ApprovalVote {
        ApprovalVote::Deny {
            reason: self.reason.clone(),
        }
    }
}

/// An [`ApprovalListener`] that always delegates.
pub struct DelegateListener;

#[async_trait]
impl ApprovalListener for DelegateListener {
    async fn vote(&self, _req: &ApprovalRequest<'_>) -> ApprovalVote {
        ApprovalVote::Delegate
    }
}

/// An [`ApprovalListener`] driven by a closure, for scripted tests.
pub struct FnApprovalListener<F> {
    f: F,
}

impl<F> FnApprovalListener<F>
where
    F: Fn(&ApprovalRequest<'_>) -> ApprovalVote + Send + Sync + 'static,
{
    /// A listener that votes by calling `f`.
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

#[async_trait]
impl<F> ApprovalListener for FnApprovalListener<F>
where
    F: Fn(&ApprovalRequest<'_>) -> ApprovalVote + Send + Sync + 'static,
{
    async fn vote(&self, req: &ApprovalRequest<'_>) -> ApprovalVote {
        (self.f)(req)
    }
}

/// Map an approval round-trip result (or its absence) onto a [`ResolvedAsk`].
///
/// `None` degrades to a denial — the deployment mounted no [`ApprovalService`],
/// matching the `ctx.get('approval') === undefined` branch of `serviceAsk`.
/// `Timeout` and a listener-supplied `Deny` reason are not part of the wire
/// [`ApprovalOutcome`], so they are mapped here directly; the four wire outcomes
/// delegate to [`resolve_ask`].
pub fn resolve_approval(
    tool_name: &str,
    ask_reason: Option<&str>,
    result: Option<ApprovalResult>,
) -> ResolvedAsk {
    match result {
        None => resolve_ask(tool_name, ask_reason, None),
        Some(ApprovalResult::AllowedOnce) => {
            resolve_ask(tool_name, ask_reason, Some(ApprovalOutcome::AllowedOnce))
        }
        Some(ApprovalResult::Rejected {
            reason: Some(reason),
        }) => ResolvedAsk {
            decision: PreToolDecision::deny(reason),
            approval_cancelled: false,
        },
        Some(ApprovalResult::Rejected { reason: None }) => {
            resolve_ask(tool_name, ask_reason, Some(ApprovalOutcome::Rejected))
        }
        Some(ApprovalResult::Cancelled) => {
            resolve_ask(tool_name, ask_reason, Some(ApprovalOutcome::Cancelled))
        }
        Some(ApprovalResult::Unavailable) => {
            resolve_ask(tool_name, ask_reason, Some(ApprovalOutcome::Unavailable))
        }
        Some(ApprovalResult::Timeout) => ResolvedAsk {
            decision: PreToolDecision::deny(format!("approval for tool \"{tool_name}\" timed out")),
            approval_cancelled: false,
        },
    }
}

/// Await approval for an `ask` decision through a mounted [`ApprovalService`],
/// mapping the result back through [`resolve_approval`]. This is the async seam
/// that wraps the pure `gate` rules: a pure decision never reaches here, and
/// with no service mounted an `ask` degrades to a denial.
pub async fn await_approval(
    call: &ToolCallInput,
    ask_reason: Option<&str>,
    service: Option<&dyn ApprovalService>,
    signal: Option<&CancelSignal>,
) -> ResolvedAsk {
    let Some(service) = service else {
        return resolve_ask(&call.name, ask_reason, None);
    };
    let req = ApprovalRequest {
        tool_name: &call.name,
        call_id: &call.call_id,
        reason: ask_reason,
        display_reason: None,
        signal,
    };
    let result = service.request(req).await;
    resolve_approval(&call.name, ask_reason, Some(result))
}
