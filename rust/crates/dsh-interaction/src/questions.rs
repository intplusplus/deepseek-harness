//! The user-question vocabulary, its validation rules, and the question
//! service.
//!
//! Ported from `packages/interaction/user-questions` — `src/types.ts` (the
//! client-safe vocabulary), `src/index.ts` (the service: validation, the
//! answerer waterfall, the timed wait, and the late reply).
//!
//! ## Pure rules and the async half
//!
//! Everything in this module that has a deterministic answer is a **pure
//! synchronous function**: [`validate_questions`], [`validate_answer_batch`],
//! and the answer-batch reader the projection uses. The service's own methods
//! are synchronous too, because the Rust tool contract
//! ([`dsh_tools::ToolHandler::call`]) is synchronous — a deployment bridges
//! its UI thread to the answerer seam itself.
//!
//! The genuinely time-dependent half — the foreground wait that suspends its
//! deadline while an answer UI holds it — lives in [`crate::timed_wait`].
//!
//! ## What is not ported
//!
//! - **The answerer waterfall's scoping.** The TypeScript dispatch is scoped
//!   by agent (`scopeTarget(agent, agent)`), so an agent-scoped listener sees
//!   only that agent. The Rust [`UserQuestionAnswerer`] seam receives the
//!   request — including the agent — and the deployment filters.
//! - **The live agent registry.** The TypeScript service asks
//!   `ctx.get('agents')` whether the caller is the registry's exact live
//!   instance and whether it is a runtime root. `dsh-agent` exposes only
//!   `Agent { id }` — no registry, no ownership — so the check is an injected
//!   [`LiveAgentRegistry`] seam, and a deployment with no registry mounted
//!   fails closed with `CALLER_NOT_LIVE`.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use dsh_agent::Agent;
use dsh_llm::{ContentBlock, ExtensionSource, MessageId, MessageSource, ToolCallId, UserMessage};
use dsh_tools::cancel::CancelSignal;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::approval::USER_APPROVAL_SOURCE_KIND;

/// The message-source kind a late reply carries.
pub const USER_QUESTION_REPLY_SOURCE_KIND: &str = "user-question-reply";

/// One selectable answer offered to the user.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AskUserQuestionOption {
    /// User-facing label.
    pub label: String,
    /// Optional extra context rendered by capable UIs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// The presentation intents a question may declare.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AskUserQuestionIntentKind {
    /// A plan submitted for review: `detail` is the plan markdown `ask()`
    /// requires, and the decision approves or declines it.
    PlanReview,
}

impl std::fmt::Display for AskUserQuestionIntentKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PlanReview => formatter.write_str("plan-review"),
        }
    }
}

/// A caller-declared presentation intent.
///
/// The question IS this kind of decision, so a UI that recognises the tag may
/// present it as such instead of as a generic option list. An intent changes
/// presentation only, never the protocol.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AskUserQuestionIntent {
    /// Which decision this question is.
    pub kind: AskUserQuestionIntentKind,
    /// The option label that approves the plan; every other option declines
    /// it. Named rather than positional so no UI infers the verdict from
    /// option order.
    pub approve: String,
    /// The logged tool invocation whose arguments contain the reviewed plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<ToolCallId>,
}

/// One question in a user-questions request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AskUserQuestionItem {
    /// Stable caller-provided question id, echoed in the answer.
    pub id: String,
    /// The question to display.
    pub question: String,
    /// Optional supporting detail rendered with the question but kept out of
    /// option labels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Optional short heading/group label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    /// Optional choices the UI can render as a menu.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<AskUserQuestionOption>>,
    /// Whether more than one option may be selected. Absent means
    /// single-select.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_select: Option<bool>,
    /// Optional presentation intent for capable UIs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<AskUserQuestionIntent>,
}

/// Answer to one question.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AskUserQuestionAnswerItem {
    /// The answered question id.
    pub id: String,
    /// Selected option labels. May accompany custom text for a multi-select
    /// question.
    pub selected: Vec<String>,
    /// Optional free-text "Other" answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom: Option<String>,
}

/// The human's answer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AskUserQuestionAnswer {
    /// Structured answers keyed by question id.
    pub answers: Vec<AskUserQuestionAnswerItem>,
}

impl AskUserQuestionAnswer {
    /// An answer carrying no items.
    pub fn empty() -> Self {
        Self {
            answers: Vec::new(),
        }
    }

    /// Whether the batch names every question of `questions` exactly once.
    ///
    /// This is the model-facing contract the answer batch must satisfy: one
    /// item per question of the call, no question named twice.
    pub fn names_each_question_once(&self, questions: &[AskUserQuestionItem]) -> bool {
        let answered: std::collections::HashSet<&str> =
            self.answers.iter().map(|item| item.id.as_str()).collect();
        answered.len() == self.answers.len()
            && questions.len() == self.answers.len()
            && questions
                .iter()
                .all(|question| answered.contains(question.id.as_str()))
    }
}

/// `open` while the tool call may still return the answer; `continued` once the
/// answer can only arrive as a new turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum UserQuestionState {
    /// The call may still return its answer.
    Open,
    /// The answer can only arrive as a later turn.
    Continued,
}

/// One unanswered timed `ask_user_question` call reconstructed from the
/// session log.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PendingUserQuestion {
    /// The tool call identity.
    pub call_id: ToolCallId,
    /// The questions the call asked.
    pub questions: Vec<AskUserQuestionItem>,
    /// Whether the call can still take an answer.
    pub state: UserQuestionState,
}

/// One timed `ask_user_question` call and the answers it settled with.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SettledUserQuestion {
    /// The settled call.
    pub call_id: ToolCallId,
    /// The recorded batch, one entry per question; empty when the late reply
    /// carried none.
    pub answers: Vec<AskUserQuestionAnswerItem>,
}

/// Both halves of one session's timed `ask_user_question` state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UserQuestionProjectionView {
    /// Calls that can still take an answer, in ask order.
    pub active: Vec<PendingUserQuestion>,
    /// Calls an answer settled, in settlement order.
    pub settled: Vec<SettledUserQuestion>,
}

impl UserQuestionProjectionView {
    /// The empty view: no active call, no settled call.
    pub fn empty() -> Self {
        Self {
            active: Vec::new(),
            settled: Vec::new(),
        }
    }

    /// The active calls still continued.
    pub fn continued(&self) -> Vec<&PendingUserQuestion> {
        self.active
            .iter()
            .filter(|question| question.state == UserQuestionState::Continued)
            .collect()
    }
}

/// The stable error taxonomy for user-question failures.
///
/// The `code()` of each variant is the machine-routing identity the
/// TypeScript `UserQuestionError.code` carried.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UserQuestionError {
    /// The request's signal was already or became aborted.
    #[error("ask_user_question was aborted before the user answered")]
    Aborted,
    /// A request carried no question.
    #[error("ask_user_question requires at least one question")]
    EmptyQuestions,
    /// A question's intent names an approve label that is not one of its own
    /// options.
    #[error(
        "question {question} declares intent {intent} whose approve label {approve:?} names none of its options"
    )]
    BadIntentApproveLabel {
        /// The offending question id.
        question: String,
        /// The declared intent kind.
        intent: AskUserQuestionIntentKind,
        /// The approve label that names no option.
        approve: String,
    },
    /// A question's intent declares no detail to review.
    #[error("question {question} declares intent {intent} without the detail it reviews")]
    BadIntentMissingDetail {
        /// The offending question id.
        question: String,
        /// The declared intent kind.
        intent: AskUserQuestionIntentKind,
    },
    /// No answerer accepted the request.
    #[error("no user-questions answerer accepted the request")]
    NoProvider,
    /// The supplied agent is not the registry's exact live instance.
    #[error("human interaction requires the exact live calling agent when an agent is supplied")]
    CallerNotLive,
    /// The supplied agent is owned by another live agent.
    #[error(
        "human interaction is unavailable while the calling agent is owned by another live agent; \
         include the unresolved question or decision in the child agent's final result"
    )]
    DelegatedCaller,
    /// A reply is already waiting for admission.
    #[error("a reply is already queued for this question")]
    ReplyQueued,
    /// The answer batch does not name each question of the call exactly once.
    #[error(
        "the answer batch for {call_id} must name each of its {expected} questions exactly once"
    )]
    BadAnswer {
        /// The call the batch answers.
        call_id: String,
        /// How many questions the call asked.
        expected: usize,
    },
    /// The foreground wait expired before the user answered.
    #[error("ask_user_question timed out before the user answered")]
    AskTimedOut,
    /// The call already owns a foreground wait.
    #[error("the question call already has a foreground wait")]
    DuplicateWait,
    /// The wait budget is not a positive platform timer.
    #[error("timeout must fit a positive platform timer")]
    BadTimeout,
}

impl UserQuestionError {
    /// The stable machine-routing code of this failure.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Aborted => "ASK_ABORTED",
            Self::EmptyQuestions => "EMPTY_QUESTIONS",
            Self::BadIntentApproveLabel { .. } | Self::BadIntentMissingDetail { .. } => {
                "BAD_INTENT"
            }
            Self::NoProvider => "NO_PROVIDER",
            Self::CallerNotLive => "CALLER_NOT_LIVE",
            Self::DelegatedCaller => "DELEGATED_CALLER",
            Self::ReplyQueued => "REPLY_QUEUED",
            Self::BadAnswer { .. } => "BAD_ANSWER",
            Self::AskTimedOut => "ASK_TIMED_OUT",
            Self::DuplicateWait => "DUPLICATE_WAIT",
            Self::BadTimeout => "BAD_TIMEOUT",
        }
    }

    /// The error family name, matching the TypeScript `UserQuestionError.name`.
    pub fn name(&self) -> &'static str {
        "UserQuestionError"
    }
}

/// The tool call a question card is keyed by.
///
/// The TypeScript shape is `{ callId, timed? }`. `timeout_ms` is a Rust-only
/// extension: the foreground wait budget the answerer has to honour, so the
/// answerer never has to learn it from a second channel.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct QuestionWait {
    /// The tool call the Client card is keyed by.
    pub call_id: ToolCallId,
    /// True for a foreground wait that requires a Client claim; absent for
    /// indefinite waits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timed: Option<bool>,
    /// The foreground wait budget in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

impl QuestionWait {
    /// An indefinite wait keyed by `call_id`.
    pub fn keyed(call_id: ToolCallId) -> Self {
        Self {
            call_id,
            timed: None,
            timeout_ms: None,
        }
    }

    /// A foreground wait keyed by `call_id`.
    pub fn timed(call_id: ToolCallId, timeout_ms: u64) -> Self {
        Self {
            call_id,
            timed: Some(true),
            timeout_ms: Some(timeout_ms),
        }
    }
}

/// A request for a human answer.
///
/// This is the process-local seam payload; it is never persisted, so the
/// agent and the cancellation signal are plain fields rather than wire data.
#[derive(Clone, Debug)]
pub struct AskUserQuestionRequest {
    /// The questions to display.
    pub questions: Vec<AskUserQuestionItem>,
    /// The agent on whose behalf the question is asked, when one is known.
    pub agent: Option<Agent>,
    /// The cancellation lifetime of the pending request.
    pub signal: Option<CancelSignal>,
    /// The tool call the Client card is keyed by.
    pub wait: Option<QuestionWait>,
}

impl AskUserQuestionRequest {
    /// A request for `questions` with no agent, signal, or wait key.
    pub fn new(questions: Vec<AskUserQuestionItem>) -> Self {
        Self {
            questions,
            agent: None,
            signal: None,
            wait: None,
        }
    }

    /// Attach the owning agent.
    #[must_use]
    pub fn with_agent(mut self, agent: Agent) -> Self {
        self.agent = Some(agent);
        self
    }

    /// Attach the cancellation signal.
    #[must_use]
    pub fn with_signal(mut self, signal: CancelSignal) -> Self {
        self.signal = Some(signal);
        self
    }

    /// Attach the wait key.
    #[must_use]
    pub fn with_wait(mut self, wait: QuestionWait) -> Self {
        self.wait = Some(wait);
        self
    }
}

/// The answerer waterfall seam (TS `user-questions/request`).
///
/// An implementation either returns an answer — claiming the request — or
/// reports that it cannot answer, so the chain moves on. A deployment
/// composes its listeners behind this trait; the crate owns the validation
/// around the seam, not the UI behind it.
pub trait UserQuestionAnswerer: Send + Sync {
    /// Answer one request.
    fn ask(
        &self,
        request: &AskUserQuestionRequest,
    ) -> Result<AskUserQuestionAnswer, UserQuestionError>;
}

/// The live-agent registry the question service consults.
///
/// The TypeScript service asks `ctx.get('agents')` whether the caller is the
/// registry's exact live instance and whether it is a runtime root.
/// `dsh-agent` exposes only `Agent { id }`, so the check is this seam: a
/// deployment mounts the registry it has.
pub trait LiveAgentRegistry: Send + Sync {
    /// Whether `agent` is the registry's exact live instance.
    fn is_live(&self, agent: &Agent) -> bool;

    /// Whether `agent` is a runtime root — not owned by another live agent.
    fn is_root(&self, agent: &Agent) -> bool;
}

/// The inbox a late reply is steered into (TS `agent.steer`).
///
/// `dsh-agent` has no inbox mutation API, so the steering is this seam.
pub trait ReplySink: Send + Sync {
    /// Steer `message` into `agent`.
    fn steer(&self, agent: &Agent, message: UserMessage) -> Result<(), UserQuestionError>;
}

/// The identity source for the reply messages the channel mints.
pub trait MessageIdSource: Send + Sync {
    /// Mint the next message id.
    fn next_message_id(&self) -> MessageId;
}

/// The production message-id source: a fresh UUID per reply.
///
/// This is the one place that reads the system random source, exactly like the
/// TypeScript `createUserMessage` it replaces. Tests inject
/// [`CountingMessageIds`] instead.
#[derive(Clone, Copy, Debug, Default)]
pub struct RandomMessageIds;

impl MessageIdSource for RandomMessageIds {
    fn next_message_id(&self) -> MessageId {
        MessageId::random()
    }
}

/// A deterministic message-id source: `prefix-1`, `prefix-2`, …
#[derive(Debug)]
pub struct CountingMessageIds {
    prefix: String,
    counter: AtomicU64,
}

impl CountingMessageIds {
    /// A source whose ids are `prefix-<n>`, starting at `prefix-1`.
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            counter: AtomicU64::new(0),
        }
    }
}

impl MessageIdSource for CountingMessageIds {
    fn next_message_id(&self) -> MessageId {
        let index = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        MessageId::new(self.prefix.clone() + "-" + &index.to_string())
    }
}

/// What a timed ask produced: the answer, or a call that is still pending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TimedUserQuestionResult {
    /// The user answered inside the window.
    Answer(AskUserQuestionAnswer),
    /// The window closed with no answer batch; the call stays answerable.
    Pending {
        /// The tool call whose questions remain pending.
        call_id: ToolCallId,
    },
}

/// Validate the question batch of one request.
///
/// The rules the TypeScript `ask()` enforces before it consults any answerer:
///
/// - the batch is not empty (`EMPTY_QUESTIONS`);
/// - an intent's approve label names one of the question's own options, and
///   the question carries the detail the intent reviews (`BAD_INTENT`).
///
/// A presentation intent asserts two things the types cannot: that the named
/// approve label is one of this question's own options, and that a plan review
/// carries the plan it is a review of. Either gap would put a choice the asker
/// never offered — or an approval of something invisible — in front of the
/// user, so the mistake is caught at the asker.
pub fn validate_questions(questions: &[AskUserQuestionItem]) -> Result<(), UserQuestionError> {
    if questions.is_empty() {
        return Err(UserQuestionError::EmptyQuestions);
    }
    for question in questions {
        let Some(intent) = &question.intent else {
            continue;
        };
        let approves = question
            .options
            .as_deref()
            .unwrap_or_default()
            .iter()
            .any(|option| option.label == intent.approve);
        if !approves {
            return Err(UserQuestionError::BadIntentApproveLabel {
                question: question.id.clone(),
                intent: intent.kind,
                approve: intent.approve.clone(),
            });
        }
        if question.detail.is_none() {
            return Err(UserQuestionError::BadIntentMissingDetail {
                question: question.id.clone(),
                intent: intent.kind,
            });
        }
    }
    Ok(())
}

/// Validate that `answer` names each of `questions` exactly once.
///
/// The gateway validated the batch's shape from the type; the model-facing
/// contract also promises one item per question, which only the owner of the
/// asked questions can check before the batch reaches the model.
pub fn validate_answer_batch(
    call_id: &ToolCallId,
    questions: &[AskUserQuestionItem],
    answer: &AskUserQuestionAnswer,
) -> Result<(), UserQuestionError> {
    if answer.names_each_question_once(questions) {
        return Ok(());
    }
    Err(UserQuestionError::BadAnswer {
        call_id: call_id.as_str().to_owned(),
        expected: questions.len(),
    })
}

/// Build the reply message a late answer is steered in as.
///
/// The message text is the answer batch in the shape the model reads, and its
/// source names the call — the `userQuestions` projection reads that source to
/// close the question and record its answers.
pub fn answer_reply_message(
    id: MessageId,
    call_id: &ToolCallId,
    questions: &[AskUserQuestionItem],
    answer: &AskUserQuestionAnswer,
) -> UserMessage {
    let mut fields = Map::new();
    fields.insert(
        "kind".to_string(),
        Value::String(USER_QUESTION_REPLY_SOURCE_KIND.to_string()),
    );
    fields.insert(
        "callId".to_string(),
        Value::String(call_id.as_str().to_string()),
    );
    fields.insert("outcome".to_string(), Value::String("answered".to_string()));
    let source = MessageSource::Extension(ExtensionSource::new(
        USER_QUESTION_REPLY_SOURCE_KIND,
        fields,
    ));

    let mut payload = Map::new();
    payload.insert(
        "kind".to_string(),
        Value::String("answer_to_pending_question".to_string()),
    );
    payload.insert(
        "tool".to_string(),
        Value::String(crate::projection::ASK_USER_QUESTION_TOOL.to_string()),
    );
    payload.insert(
        "callId".to_string(),
        Value::String(call_id.as_str().to_string()),
    );
    payload.insert(
        "questions".to_string(),
        serde_json::to_value(questions).unwrap_or(Value::Array(Vec::new())),
    );
    payload.insert(
        "answers".to_string(),
        serde_json::to_value(&answer.answers).unwrap_or(Value::Array(Vec::new())),
    );
    let text = Value::Object(payload).to_string();

    UserMessage::with_source(id, vec![ContentBlock::text(text)], source)
}

/// Build the policy-switch notice the approval layer queues for the next model
/// step.
pub fn policy_notice_message(id: MessageId, text: String) -> UserMessage {
    let mut fields = Map::new();
    fields.insert(
        "kind".to_string(),
        Value::String(USER_APPROVAL_SOURCE_KIND.to_string()),
    );
    let source = MessageSource::Extension(ExtensionSource::new(USER_APPROVAL_SOURCE_KIND, fields));
    UserMessage::with_source(id, vec![ContentBlock::text(text)], source)
}

/// `ctx.userQuestions`: validation plus the scoped answerer waterfall.
pub struct UserQuestionService {
    answerer: Option<Arc<dyn UserQuestionAnswerer>>,
    agents: Option<Arc<dyn LiveAgentRegistry>>,
    replies: Option<Arc<dyn ReplySink>>,
    message_ids: Arc<dyn MessageIdSource>,
    waits: Mutex<HashMap<WaitKey, AskUserQuestionRequest>>,
    queued_replies: Mutex<HashMap<WaitKey, MessageId>>,
}

/// The identity of one foreground wait: the owning session and the call.
type WaitKey = (dsh_session::SessionId, ToolCallId);

impl UserQuestionService {
    /// A service with no answerer mounted: every ask fails closed with
    /// `NO_PROVIDER`.
    pub fn new() -> Self {
        Self {
            answerer: None,
            agents: None,
            replies: None,
            message_ids: Arc::new(RandomMessageIds),
            waits: Mutex::new(HashMap::new()),
            queued_replies: Mutex::new(HashMap::new()),
        }
    }

    /// Mount the answerer chain.
    #[must_use]
    pub fn with_answerer(mut self, answerer: Arc<dyn UserQuestionAnswerer>) -> Self {
        self.answerer = Some(answerer);
        self
    }

    /// Mount the live-agent registry.
    #[must_use]
    pub fn with_live_agents(mut self, agents: Arc<dyn LiveAgentRegistry>) -> Self {
        self.agents = Some(agents);
        self
    }

    /// Mount the sink late replies are steered into.
    #[must_use]
    pub fn with_reply_sink(mut self, replies: Arc<dyn ReplySink>) -> Self {
        self.replies = Some(replies);
        self
    }

    /// Replace the message-id source; the default is [`RandomMessageIds`].
    #[must_use]
    pub fn with_message_ids(mut self, message_ids: Arc<dyn MessageIdSource>) -> Self {
        self.message_ids = message_ids;
        self
    }

    /// Ask the answerer chain and wait for the user's answer.
    ///
    /// The request is validated before any answerer sees it: an aborted signal
    /// is `ASK_ABORTED`, an empty batch is `EMPTY_QUESTIONS`, and a malformed
    /// intent is `BAD_INTENT`. A supplied agent must be the registry's exact
    /// live root, otherwise the ask is `CALLER_NOT_LIVE` or `DELEGATED_CALLER`.
    pub fn ask(
        &self,
        request: &AskUserQuestionRequest,
    ) -> Result<AskUserQuestionAnswer, UserQuestionError> {
        if let Some(signal) = &request.signal {
            if signal.is_cancelled() {
                return Err(UserQuestionError::Aborted);
            }
        }
        validate_questions(&request.questions)?;
        if let Some(agent) = &request.agent {
            self.assert_live_root(agent)?;
        }
        let Some(answerer) = &self.answerer else {
            return Err(UserQuestionError::NoProvider);
        };
        match answerer.ask(request) {
            Ok(answer) => Ok(answer),
            Err(error) => {
                if let Some(signal) = &request.signal {
                    if signal.is_cancelled() {
                        return Err(UserQuestionError::Aborted);
                    }
                }
                Err(error)
            }
        }
    }

    /// Ask with a foreground wait: the answer when it arrives inside the
    /// window, otherwise a pending result that keeps the call answerable.
    ///
    /// The window is enforced by the answerer, which receives the wait key and
    /// its budget; the service recognises the resulting `ASK_TIMED_OUT` and
    /// turns it into [`TimedUserQuestionResult::Pending`].
    pub fn ask_timed(
        &self,
        request: &AskUserQuestionRequest,
        call_id: &ToolCallId,
        timeout_ms: u64,
    ) -> Result<TimedUserQuestionResult, UserQuestionError> {
        if timeout_ms == 0 || timeout_ms > MAX_TIMEOUT_MS {
            return Err(UserQuestionError::BadTimeout);
        }
        if request.agent.is_none() {
            // A foreground wait is routed through the live agent; without one
            // there is nothing to route to.
            return Err(UserQuestionError::CallerNotLive);
        }
        let agent = request.agent.as_ref().expect("checked above");
        self.assert_live_root(agent)?;

        let key = (agent.id.clone(), call_id.clone());
        {
            let mut waits = self.lock_waits();
            if waits.contains_key(&key) {
                return Err(UserQuestionError::DuplicateWait);
            }
            waits.insert(key.clone(), request.clone());
        }

        let timed_request = request
            .clone()
            .with_wait(QuestionWait::timed(call_id.clone(), timeout_ms));
        let result = self.ask(&timed_request);

        {
            let mut waits = self.lock_waits();
            waits.remove(&key);
        }

        match result {
            Ok(answer) => Ok(TimedUserQuestionResult::Answer(answer)),
            Err(error) if error.code() == UserQuestionError::AskTimedOut.code() => {
                Ok(TimedUserQuestionResult::Pending {
                    call_id: call_id.clone(),
                })
            }
            Err(error) => Err(error),
        }
    }

    /// Whether the call already owns a foreground wait.
    pub fn has_foreground_wait(&self, agent: &Agent, call_id: &ToolCallId) -> bool {
        let key = (agent.id.clone(), call_id.clone());
        self.lock_waits().contains_key(&key)
    }

    /// Answer a continued question.
    ///
    /// The reply is steered into the agent as a user message whose source names
    /// the call; that message is also the record that closes the question in
    /// the projection.
    ///
    /// `view` is the caller's current `userQuestions` projection view: the
    /// projection registry is not `Send`/`Sync`, so the caller reads the state
    /// and hands it over rather than the service holding a registry.
    ///
    /// Returns whether the question is still continued. An accepted reply stays
    /// queued until the agent admits its user message.
    pub fn answer(
        &self,
        view: &UserQuestionProjectionView,
        agent: &Agent,
        call_id: &ToolCallId,
        answer: &AskUserQuestionAnswer,
    ) -> Result<bool, UserQuestionError> {
        self.assert_live_root(agent)?;
        let Some(question) = view
            .continued()
            .into_iter()
            .find(|question| question.call_id == *call_id)
        else {
            return Ok(false);
        };
        if self.reply_is_queued(agent, call_id) {
            return Err(UserQuestionError::ReplyQueued);
        }
        validate_answer_batch(call_id, &question.questions, answer)?;

        let message = answer_reply_message(
            self.message_ids.next_message_id(),
            call_id,
            &question.questions,
            answer,
        );
        let message_id = message.id.clone();
        self.queue_reply(agent, call_id, &message_id);
        let Some(replies) = &self.replies else {
            self.release_reply(agent, call_id, &message_id);
            return Err(UserQuestionError::NoProvider);
        };
        if let Err(error) = replies.steer(agent, message) {
            self.release_reply(agent, call_id, &message_id);
            return Err(error);
        }
        Ok(true)
    }

    /// Whether a reply for this call is already waiting for admission.
    pub fn reply_is_queued(&self, agent: &Agent, call_id: &ToolCallId) -> bool {
        self.lock_queued_replies()
            .contains_key(&(agent.id.clone(), call_id.clone()))
    }

    fn queue_reply(&self, agent: &Agent, call_id: &ToolCallId, message_id: &MessageId) {
        let mut queued = self.lock_queued_replies();
        queued.insert((agent.id.clone(), call_id.clone()), message_id.clone());
    }

    fn release_reply(&self, agent: &Agent, call_id: &ToolCallId, message_id: &MessageId) {
        let mut queued = self.lock_queued_replies();
        let key = (agent.id.clone(), call_id.clone());
        if queued.get(&key) == Some(message_id) {
            queued.remove(&key);
        }
    }

    fn assert_live_root(&self, agent: &Agent) -> Result<(), UserQuestionError> {
        let Some(agents) = &self.agents else {
            // Without a registry the caller cannot be proven to be the live
            // root; the check fails closed.
            return Err(UserQuestionError::CallerNotLive);
        };
        if !agents.is_live(agent) {
            return Err(UserQuestionError::CallerNotLive);
        }
        if !agents.is_root(agent) {
            return Err(UserQuestionError::DelegatedCaller);
        }
        Ok(())
    }

    fn lock_waits(&self) -> std::sync::MutexGuard<'_, HashMap<WaitKey, AskUserQuestionRequest>> {
        self.waits.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn lock_queued_replies(&self) -> std::sync::MutexGuard<'_, HashMap<WaitKey, MessageId>> {
        self.queued_replies
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

impl Default for UserQuestionService {
    fn default() -> Self {
        Self::new()
    }
}

/// The largest foreground wait the platform timer admits, in milliseconds.
pub const MAX_TIMEOUT_MS: u64 = 2_147_483_647;
