//! The approval policy layer and the approval audit events.
//!
//! Ported from `packages/interaction/user-approval` — `src/types.ts` (the wire
//! vocabulary) and `src/index.ts` (the policy layer and the audited request
//! path).
//!
//! ## What this crate adds on top of `dsh-tools`
//!
//! `dsh_tools::approval` already owns the **async answerer seam**: the
//! [`ApprovalService`](dsh_tools::approval::ApprovalService) trait, the
//! waterfall of listeners, the cancel/timeout reconciliation, and the
//! [`ApprovalResult`](dsh_tools::approval::ApprovalResult) it produces. This
//! module does **not** redefine that trait; it consumes it. The two things the
//! TypeScript `user-approval` package contributes on top of the raw answerer
//! seam are exactly the two things ported here:
//!
//! 1. the **policy layer** — [`ApprovalPolicy`], [`APPROVAL_POLICIES`],
//!    [`set_approval_policy`], and the `never`-before-dispatch gate that no
//!    listener registration order can bypass; and
//! 2. the **audit events** — `approval/asked` and `approval/decided`, the
//!    durable pair that records what was asked and how it settled.
//!
//! [`AuditedApprovalService`] composes the two: it applies the session policy
//! before the answerer chain runs, and it appends the audit pair to the
//! session log around the answer.
//!
//! ## Environment injection
//!
//! The TypeScript service mints its request id with `randomUUID()`. Rust's
//! `uuid` crate reads the system random source, so the id source is an
//! injected seam instead: [`ApprovalIdGenerator`]. [`RandomApprovalIds`] is the
//! production implementation (the same `randomUUID()` source the TypeScript
//! uses), while [`CountingApprovalIds`] makes the ids deterministic for tests.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use dsh_llm::ToolCallId;
use dsh_session::{CoreEvent, ExtensionEvent, Session, SessionError, SessionEvent, SessionSeq};
use dsh_tools::approval::{ApprovalRequest, ApprovalResult, ApprovalService};
use dsh_tools::cancel::CancelSignal;
use dsh_tools::gate::{ApprovalOutcome, DisplayReason};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The `type` tag of the durable policy-override event.
pub const APPROVAL_POLICY_EVENT: &str = "approval/policy";

/// The `type` tag of the durable ask audit event.
pub const APPROVAL_ASKED_EVENT: &str = "approval/asked";

/// The `type` tag of the durable outcome audit event.
pub const APPROVAL_DECIDED_EVENT: &str = "approval/decided";

/// The message-source kind the policy-switch notice carries.
pub const USER_APPROVAL_SOURCE_KIND: &str = "user-approval";

/// Model-facing statement for the deterministic `never` policy.
pub const NEVER_SENTENCE: &str = "Approval prompts are disabled in this session: actions that require approval are rejected automatically — do not request sandbox escalation (do not set `sandbox_permissions`).";

/// Model-facing statement for an interactive policy that may still fail closed.
pub const ASK_SENTENCE: &str = "Approval policy: ask. Operations that require approval may ask through the configured answerers; without an available answerer, the request fails closed.";

/// A session's approval policy — what happens to an approval ask BEFORE any
/// interactive answerer sees it.
///
/// - [`Ask`](ApprovalPolicy::Ask) (the default) delegates to the composed
///   answerers; with none composed the chain falls through to the fail-closed
///   `unavailable`.
/// - [`Never`](ApprovalPolicy::Never) never prompts anyone: every ask resolves
///   `rejected` deterministically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApprovalPolicy {
    /// Delegate to the composed answerers.
    Ask,
    /// Never prompt anyone; every ask is rejected deterministically.
    Never,
}

impl ApprovalPolicy {
    /// The policy's wire value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Never => "never",
        }
    }

    /// Parse a policy from its wire value, rejecting anything outside the
    /// closed vocabulary.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "ask" => Some(Self::Ask),
            "never" => Some(Self::Never),
            _ => None,
        }
    }

    /// The model-facing statement for this policy.
    pub const fn sentence(self) -> &'static str {
        match self {
            Self::Ask => ASK_SENTENCE,
            Self::Never => NEVER_SENTENCE,
        }
    }
}

/// Every [`ApprovalPolicy`], for option advertisement and runtime validation of
/// untrusted policy strings.
pub const APPROVAL_POLICIES: [ApprovalPolicy; 2] = [ApprovalPolicy::Ask, ApprovalPolicy::Never];

/// Where a durable policy override came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApprovalPolicySource {
    /// An override seeded into a child session at delegation.
    Delegation,
}

/// Payload of the durable `approval/policy` event.
///
/// The event is log-only: it never reaches the model transcript. The LAST such
/// event in a session log is that session's override.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApprovalPolicyEvent {
    /// The policy in effect until the next switch.
    pub policy: ApprovalPolicy,
    /// Marks an override seeded into a child at delegation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ApprovalPolicySource>,
}

impl ApprovalPolicyEvent {
    /// A runtime switch, with no delegation source.
    pub fn new(policy: ApprovalPolicy) -> Self {
        Self {
            policy,
            source: None,
        }
    }
}

/// Pairs one `approval/asked` audit event with its `approval/decided`.
///
/// Service-issued: one fresh id per audited request.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ApprovalRequestId(String);

impl ApprovalRequestId {
    /// Brand a raw id, rejecting an empty one.
    ///
    /// The TypeScript `ApprovalRequestId(id)` brands without validating; this
    /// port rejects an empty id because a typed producer cannot mint one, and
    /// an empty id would make the audit pair ambiguous.
    pub fn parse(id: &str) -> Result<Self, ApprovalError> {
        if id.is_empty() {
            return Err(ApprovalError::EmptyRequestId);
        }
        Ok(Self(id.to_owned()))
    }

    /// The branded id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Payload of the durable `approval/asked` audit event.
///
/// Log-only, like `hook/*`: it carries no surface operation and never enters
/// the model transcript.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApprovalAsked {
    /// The id pairing this event with its `approval/decided`.
    pub id: ApprovalRequestId,
    /// The tool the question is about.
    pub tool_name: String,
    /// The exact tool call being decided, when the asker had one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<ToolCallId>,
    /// The asker's human-readable explanation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Payload of the durable `approval/decided` audit event.
///
/// Exactly one per ask, appended when the outcome is known: a decision, a
/// cancellation, or the fail-closed `unavailable`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApprovalDecided {
    /// The id of the `approval/asked` this event answers.
    pub id: ApprovalRequestId,
    /// The closed outcome.
    pub outcome: ApprovalOutcome,
}

/// Errors raised by the approval policy layer and the audited request path.
#[derive(Debug, thiserror::Error)]
pub enum ApprovalError {
    /// A policy outside the closed vocabulary was requested.
    #[error("approval policy must be one of \"ask\" or \"never\"")]
    UnknownPolicy,
    /// The audit pair must be turn-enclosed; an idle ask appends nothing.
    #[error(
        "approval.request() outside an open turn: the approval/asked + approval/decided audit pair \
         must be turn-enclosed (a bare event between turns is crash-tail garbage on reload). \
         Ask from inside the turn that needs the decision."
    )]
    NoOpenTurn,
    /// A request id must not be empty.
    #[error("approval request id must not be empty")]
    EmptyRequestId,
    /// The session refused the audit append.
    #[error("session refused the approval audit append: {0}")]
    Session(#[from] SessionError),
    /// The audit payload did not encode as a JSON object.
    #[error("approval audit payload is not a JSON object: {0}")]
    Payload(String),
}

/// The identity source for approval request ids.
///
/// Injected so a test can assert exact ids; the production implementation
/// mirrors the TypeScript `randomUUID()` source.
pub trait ApprovalIdGenerator: Send + Sync {
    /// Mint the next request id.
    fn next_request_id(&self) -> ApprovalRequestId;
}

/// An id generator that issues `prefix-1`, `prefix-2`, … from a counter.
///
/// Deterministic: the same sequence of requests always produces the same ids.
#[derive(Debug)]
pub struct CountingApprovalIds {
    prefix: String,
    counter: AtomicU64,
}

impl CountingApprovalIds {
    /// A generator whose ids are `prefix-<n>`, starting at `prefix-1`.
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            counter: AtomicU64::new(0),
        }
    }
}

impl ApprovalIdGenerator for CountingApprovalIds {
    fn next_request_id(&self) -> ApprovalRequestId {
        let index = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        // The prefix is non-empty by construction only if the caller supplies
        // one; an empty prefix still yields a non-empty id because of the
        // trailing counter.
        ApprovalRequestId(self.prefix.clone() + "-" + &index.to_string())
    }
}

/// The production id generator: a fresh UUID per request.
///
/// This is the one place in the crate that reads the system random source,
/// exactly like the TypeScript `randomUUID()` it replaces. Tests inject
/// [`CountingApprovalIds`] instead.
#[derive(Clone, Copy, Debug, Default)]
pub struct RandomApprovalIds;

impl ApprovalIdGenerator for RandomApprovalIds {
    fn next_request_id(&self) -> ApprovalRequestId {
        ApprovalRequestId(uuid::Uuid::new_v4().to_string())
    }
}

/// One approval request handed to the audited service.
#[derive(Clone, Copy, Debug)]
pub struct AuditedApprovalRequest<'a> {
    /// The tool whose operation requires a decision.
    pub tool_name: &'a str,
    /// The exact tool call being decided, when the asker had one.
    pub call_id: Option<&'a ToolCallId>,
    /// The asker's human-readable explanation of WHY it is asking.
    pub reason: Option<&'a str>,
    /// Localized prompt text for an approval UI.
    pub display_reason: Option<&'a DisplayReason>,
    /// The caller's cancellation signal; firing it settles the request as
    /// `cancelled` and discards a late answer.
    pub signal: Option<&'a CancelSignal>,
}

impl<'a> AuditedApprovalRequest<'a> {
    /// A request for `tool_name` with no call id, reason, or signal.
    pub fn new(tool_name: &'a str) -> Self {
        Self {
            tool_name,
            call_id: None,
            reason: None,
            display_reason: None,
            signal: None,
        }
    }

    /// Attach the exact tool call being decided.
    #[must_use]
    pub fn with_call_id(mut self, call_id: &'a ToolCallId) -> Self {
        self.call_id = Some(call_id);
        self
    }

    /// Attach the asker's explanation.
    #[must_use]
    pub fn with_reason(mut self, reason: &'a str) -> Self {
        self.reason = Some(reason);
        self
    }

    /// Attach the caller's cancellation signal.
    #[must_use]
    pub fn with_signal(mut self, signal: &'a CancelSignal) -> Self {
        self.signal = Some(signal);
        self
    }
}

/// Whether the log currently sits inside an open turn (a `turn/start` not yet
/// closed by a `turn/end`) — the audited request's precondition.
///
/// The audit pair must be turn-enclosed: the turn is the durable log's
/// commit/replay boundary, so a bare event appended between turns is
/// indistinguishable from a crash tail and is silently dropped on reload.
pub fn has_open_turn(events: &[SessionEvent]) -> bool {
    for event in events.iter().rev() {
        match event.core() {
            Some(CoreEvent::TurnStart { .. }) => return true,
            Some(CoreEvent::TurnEnd { .. }) => return false,
            _ => {}
        }
    }
    false
}

/// Append the sole durable representation of a session policy override.
///
/// The policy is validated before the log changes, so an unknown policy
/// appends nothing.
pub fn set_approval_policy(
    session: &mut Session,
    policy: ApprovalPolicy,
) -> Result<SessionSeq, ApprovalError> {
    append_event(
        session,
        APPROVAL_POLICY_EVENT,
        &ApprovalPolicyEvent::new(policy),
    )
}

/// Append a policy override named by an untrusted string.
///
/// This is the runtime-validation entry point: a value outside
/// [`APPROVAL_POLICIES`] is rejected **before** the log changes, exactly like
/// the TypeScript `TypeError` the original throws.
pub fn set_approval_policy_str(
    session: &mut Session,
    policy: &str,
) -> Result<SessionSeq, ApprovalError> {
    let policy = ApprovalPolicy::parse(policy).ok_or(ApprovalError::UnknownPolicy)?;
    set_approval_policy(session, policy)
}

/// Read the session's policy override without applying the configured default.
///
/// Returns the last logged policy, or `None` without one.
pub fn approval_policy_override(events: &[SessionEvent]) -> Option<ApprovalPolicy> {
    for event in events.iter().rev() {
        let Some(extension) = event.extension() else {
            continue;
        };
        if extension.kind() != APPROVAL_POLICY_EVENT {
            continue;
        }
        let Ok(payload) = decode_payload::<ApprovalPolicyEvent>(extension) else {
            continue;
        };
        return Some(payload.policy);
    }
    None
}

/// The session's effective policy: its own override, else the configured
/// default.
pub fn effective_approval_policy(
    events: &[SessionEvent],
    configured: ApprovalPolicy,
) -> ApprovalPolicy {
    approval_policy_override(events).unwrap_or(configured)
}

/// The model-facing notice a policy switch queues for the next model step.
pub fn policy_change_notice(previous: ApprovalPolicy, next: ApprovalPolicy) -> String {
    format!(
        "The approval policy changed from \"{}\" to \"{}\" (changed by the user).",
        previous.as_str(),
        next.as_str()
    )
}

/// An approval service that applies the session policy before the answerer
/// chain and records the audit pair on the session log.
///
/// ## Relationship to `dsh-tools::approval`
///
/// The answerer chain itself is [`dsh_tools::approval::ApprovalService`]; this
/// service *wraps* one instead of redefining the trait. The wrapper adds the
/// two things the TypeScript `user-approval` package contributes:
///
/// - the **policy layer**: a `never` policy is decided **before** the answerer
///   chain is dispatched, so no listener registration order — not even a
///   listener prepended after this service mounted — can turn it into an ask;
/// - the **audit pair**: `approval/asked` is appended before the decision and
///   `approval/decided` after it, both carrying the same request id.
///
/// A throwing answerer is contained: it settles the question closed
/// (`unavailable`), never the caller's tool call open.
pub struct AuditedApprovalService {
    session: Arc<Mutex<Session>>,
    configured_policy: ApprovalPolicy,
    answerer: Arc<dyn ApprovalService>,
    ids: Arc<dyn ApprovalIdGenerator>,
}

impl AuditedApprovalService {
    /// A service over `session`, answering through `answerer`.
    pub fn new(
        session: Arc<Mutex<Session>>,
        answerer: Arc<dyn ApprovalService>,
        configured_policy: ApprovalPolicy,
    ) -> Self {
        Self {
            session,
            configured_policy,
            answerer,
            ids: Arc::new(RandomApprovalIds),
        }
    }

    /// Replace the request-id source; the default is [`RandomApprovalIds`].
    #[must_use]
    pub fn with_id_generator(mut self, ids: Arc<dyn ApprovalIdGenerator>) -> Self {
        self.ids = ids;
        self
    }

    /// The session this service audits.
    pub fn session(&self) -> &Arc<Mutex<Session>> {
        &self.session
    }

    /// The configured default policy, before any session override.
    pub fn configured_policy(&self) -> ApprovalPolicy {
        self.configured_policy
    }

    /// Ask the composed answerers to decide one request, auditing the pair.
    ///
    /// The request requires an open turn: the audit pair must be enclosed by
    /// the durable log's commit/replay boundary, so an idle ask is rejected
    /// before anything is appended. The answerer phase always produces an
    /// outcome — an aborted signal yields `cancelled`, the `never` policy
    /// yields `rejected` without consulting anybody, a missing or throwing
    /// answerer yields the fail-closed `unavailable`.
    pub async fn request(
        &self,
        request: AuditedApprovalRequest<'_>,
    ) -> Result<ApprovalOutcome, ApprovalError> {
        {
            let session = self.lock();
            if !has_open_turn(session.events()) {
                return Err(ApprovalError::NoOpenTurn);
            }
        }

        let id = self.ids.next_request_id();
        {
            let mut session = self.lock();
            let asked = ApprovalAsked {
                id: id.clone(),
                tool_name: request.tool_name.to_owned(),
                call_id: request.call_id.cloned(),
                reason: request.reason.map(str::to_owned),
            };
            append_event(&mut session, APPROVAL_ASKED_EVENT, &asked)?;
        }

        let outcome = self.decide(&request).await;

        {
            let mut session = self.lock();
            let decided = ApprovalDecided { id, outcome };
            append_event(&mut session, APPROVAL_DECIDED_EVENT, &decided)?;
        }
        Ok(outcome)
    }

    /// Dispatch the answerer chain, contained and raced against the request
    /// signal.
    async fn decide(&self, request: &AuditedApprovalRequest<'_>) -> ApprovalOutcome {
        if let Some(signal) = request.signal {
            if signal.is_cancelled() {
                return ApprovalOutcome::Cancelled;
            }
        }
        {
            let session = self.lock();
            if effective_approval_policy(session.events(), self.configured_policy)
                == ApprovalPolicy::Never
            {
                // Decided HERE, before any dispatch: a listener registered with
                // `prepend` after this service mounted would sit ahead of any
                // gate listener, so only the service's own request path can
                // keep the promise that `never` rejects deterministically.
                return ApprovalOutcome::Rejected;
            }
        }

        let call_id;
        let call_id_ref = match request.call_id {
            Some(call_id) => call_id,
            None => {
                // `dsh_tools::approval::ApprovalRequest` requires a call id
                // while the TypeScript request has `callId?`. The optionality
                // is expressed as an empty call id rather than fabricating
                // one, so an answerer that routes by call id sees "no exact
                // tool call".
                call_id = ToolCallId::new(String::new());
                &call_id
            }
        };
        let answerer_request = ApprovalRequest {
            tool_name: request.tool_name,
            call_id: call_id_ref,
            reason: request.reason,
            display_reason: request.display_reason,
            signal: request.signal,
        };

        match contain_panic(self.answerer.request(answerer_request)).await {
            Ok(ApprovalResult::AllowedOnce) => ApprovalOutcome::AllowedOnce,
            Ok(ApprovalResult::Rejected { .. }) => ApprovalOutcome::Rejected,
            Ok(ApprovalResult::Cancelled) => ApprovalOutcome::Cancelled,
            Ok(ApprovalResult::Unavailable) => ApprovalOutcome::Unavailable,
            // `Timeout` has no TypeScript counterpart. It degrades to the
            // fail-closed outcome, exactly like `unavailable`.
            Ok(ApprovalResult::Timeout) => ApprovalOutcome::Unavailable,
            // A throwing answerer must fail the QUESTION closed, not the
            // caller's tool call open — the seam contains its callbacks.
            Err(()) => ApprovalOutcome::Unavailable,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Session> {
        // A panic inside a critical section must not wedge the seam: the
        // session's own state stays consistent because every mutation is an
        // append that either committed or did not.
        self.session
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

impl ApprovalService for AuditedApprovalService {
    /// Decide one request through the audited path.
    ///
    /// This is the [`dsh_tools::approval::ApprovalService`] view of the same
    /// seam, so the audited service can be mounted directly behind
    /// [`dsh_tools::approval::await_approval`]. The audit pair is written
    /// either way; only the returned vocabulary differs.
    ///
    /// The signature is written out by hand instead of through
    /// `#[async_trait]`: this crate does not depend on `async-trait`, and the
    /// desugared form the macro produces for this trait is exactly the one
    /// below.
    fn request<'life0, 'life1, 'async_trait>(
        &'life0 self,
        req: ApprovalRequest<'life1>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ApprovalResult> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            let audited = AuditedApprovalRequest {
                tool_name: req.tool_name,
                call_id: Some(req.call_id),
                reason: req.reason,
                display_reason: req.display_reason,
                signal: req.signal,
            };
            match AuditedApprovalService::request(self, audited).await {
                Ok(ApprovalOutcome::AllowedOnce) => ApprovalResult::AllowedOnce,
                Ok(ApprovalOutcome::Rejected) => ApprovalResult::Rejected { reason: None },
                Ok(ApprovalOutcome::Cancelled) => ApprovalResult::Cancelled,
                Ok(ApprovalOutcome::Unavailable) => ApprovalResult::Unavailable,
                Err(ApprovalError::NoOpenTurn) => ApprovalResult::Unavailable,
                Err(_) => ApprovalResult::Unavailable,
            }
        })
    }
}

/// Append one typed payload as a durable session extension event.
fn append_event<T: Serialize>(
    session: &mut Session,
    kind: &str,
    payload: &T,
) -> Result<SessionSeq, ApprovalError> {
    let value =
        serde_json::to_value(payload).map_err(|error| ApprovalError::Payload(error.to_string()))?;
    let Value::Object(map) = value else {
        return Err(ApprovalError::Payload(
            "an approval audit payload must be a JSON object".to_string(),
        ));
    };
    session
        .append(SessionEvent::extension_event(kind, map))
        .map_err(ApprovalError::Session)
}

/// Decode an extension payload as `T` with its `type` tag removed.
///
/// The stamped `type` discriminator is envelope metadata, not payload.
fn decode_payload<T: for<'de> Deserialize<'de>>(
    extension: &ExtensionEvent,
) -> Result<T, serde_json::Error> {
    let mut payload = extension.payload().clone();
    payload.remove("type");
    serde_json::from_value(Value::Object(payload))
}

/// Poll `future` with every panic contained.
///
/// A throwing answerer must fail the question closed, not the caller's tool
/// call open; the seam contains its callbacks.
async fn contain_panic<F: std::future::Future>(future: F) -> Result<F::Output, ()> {
    let mut future = Box::pin(future);
    std::future::poll_fn(|context| {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            future.as_mut().poll(context)
        })) {
            Ok(std::task::Poll::Pending) => std::task::Poll::Pending,
            Ok(std::task::Poll::Ready(value)) => std::task::Poll::Ready(Ok(value)),
            Err(_) => std::task::Poll::Ready(Err(())),
        }
    })
    .await
}

/// The durable payload map of an approval event, for callers that build the
/// event themselves.
pub fn approval_event_payload<T: Serialize>(
    payload: &T,
) -> Result<Map<String, Value>, ApprovalError> {
    let value =
        serde_json::to_value(payload).map_err(|error| ApprovalError::Payload(error.to_string()))?;
    match value {
        Value::Object(map) => Ok(map),
        other => Err(ApprovalError::Payload(format!(
            "an approval audit payload must be a JSON object, got {other}"
        ))),
    }
}
