//! The fold of `ask_user_question` tool events into the answerable question
//! set, and its session projection.
//!
//! Ported from `packages/interaction/user-questions/src/projection.ts`.
//!
//! The fold is a **pure synchronous function** over the session log: it never
//! reads a clock, a file, or an environment variable, and it returns the very
//! same state when an event is unrelated to it. That is what makes the
//! ordering conclusions assertable directly by tests.
//!
//! ## What the fold decides
//!
//! A `request/header` decides, from the assembled tool schemas it records,
//! whether the `ask_user_question` calls that follow are **timed**: the timed
//! schema declares the [`TIMED_WAIT_PARAMETER`], the blocking legacy schema
//! never does. A call made under the legacy schema is never tracked, so a
//! session that only ever used that tool folds to the empty view.
//!
//! A tracked question stays answerable as `continued` in exactly two cases:
//! the tool returned the pending payload, or session resume repair appended
//! the synthetic [`TOOL_OUTCOME_UNKNOWN`] result for a call the process never
//! finished. An answer batch settles it with that batch; any failure drops it.
//! A PTC sub-call enters the fold when its recorded result is pending. A late
//! reply settles only when the agent admits its user message — an inbox splice
//! alone never settles anything.

#![forbid(unsafe_code)]

use std::sync::Arc;

use dsh_llm::{ContentBlock, ToolCallId};
use dsh_session::{
    CoreEvent, ExtensionEvent, SessionEvent, SessionHeader, SessionLogOffset, SessionSeq,
};
use dsh_session_projection::{ProjectionDefinition, ProjectionState};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::questions::{
    AskUserQuestionAnswerItem, AskUserQuestionItem, AskUserQuestionOption, PendingUserQuestion,
    SettledUserQuestion, UserQuestionProjectionView, UserQuestionState,
    USER_QUESTION_REPLY_SOURCE_KIND,
};

/// Tool whose calls this projection tracks.
pub const ASK_USER_QUESTION_TOOL: &str = "ask_user_question";

/// Model-facing parameter only the timed `ask_user_question` schema declares.
///
/// Its presence in the logged request header is what tells the fold that the
/// `ask_user_question` calls that follow can be continued past a timeout; the
/// blocking legacy schema never declares it.
pub const TIMED_WAIT_PARAMETER: &str = "timeout";

/// The recovery code of a recorded tool call whose completed outcome was not
/// durably recorded (TS `TOOL_OUTCOME_UNKNOWN`).
///
/// The TypeScript constant lives in `@deepseek-ai/dsh-session`; `dsh-session`
/// does not export it, so this port declares the same wire value locally.
pub const TOOL_OUTCOME_UNKNOWN: &str = "TOOL_OUTCOME_UNKNOWN";

/// The `type` tag of the PTC sub-dispatch settle record.
pub const PTC_DISPATCH_TAG: &str = "tool/ptc-dispatch";

/// The projection key owned by this crate.
pub const KEY: &str = "userQuestions";

/// The persisted-cache version.
///
/// The TypeScript unit is version 2; this port keeps the same version because
/// the state and client-view shapes are identical.
pub const STATE_VERSION: u64 = 2;

/// The pure fold state: whether the request header in effect declares the
/// timed `ask_user_question` schema, and the question view that schema's
/// calls build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UserQuestionFold {
    /// Whether the request header in effect declares the timed schema.
    pub timed: bool,
    /// The question view, behind one allocation so the client view can keep
    /// its identity across a change that did not touch it.
    pub questions: Arc<UserQuestionProjectionView>,
}

impl UserQuestionFold {
    /// The fold of a session with no timed call yet.
    pub fn initial() -> Self {
        Self {
            timed: false,
            questions: Arc::new(UserQuestionProjectionView::empty()),
        }
    }
}

/// Host projection state; `questions` is the wire value, reused whenever an
/// event changes nothing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UserQuestionProjectionState {
    /// Events below this sequence are inherited history and are not folded.
    pub inherited_event_count: SessionLogOffset,
    /// Whether the request header in effect declares the timed schema.
    pub timed: bool,
    /// The question view, behind one allocation.
    pub questions: Arc<UserQuestionProjectionView>,
}

impl ProjectionState for UserQuestionProjectionState {
    fn validate(&self) -> Result<(), String> {
        validate_projection_view(&self.questions)
    }
}

/// Whether one logged tool schema is the timed `ask_user_question` tool's.
pub fn is_timed_ask_user_question_schema(name: &str, parameters: &Map<String, Value>) -> bool {
    if name != ASK_USER_QUESTION_TOOL {
        return false;
    }
    let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
        return false;
    };
    properties.contains_key(TIMED_WAIT_PARAMETER)
}

/// Read the question batch out of logged `ask_user_question` arguments.
///
/// The arguments are in the **tool** vocabulary (`multi_select`); the result is
/// in the **service** vocabulary (`multiSelect`). Unreadable arguments yield
/// `None`: malformed model arguments never reached the tool, so there is no
/// card to show.
pub fn questions_of(arguments_text: &str) -> Option<Vec<AskUserQuestionItem>> {
    let parsed: Value = serde_json::from_str(arguments_text).ok()?;
    let questions = parsed.as_object()?.get("questions")?.as_array()?;
    if questions.is_empty() {
        return None;
    }
    let mut items = Vec::with_capacity(questions.len());
    for question in questions {
        items.push(tool_question_of(question)?);
    }
    Some(items)
}

/// Read one tool-vocabulary question.
fn tool_question_of(value: &Value) -> Option<AskUserQuestionItem> {
    let object = value.as_object()?;
    let id = object.get("id")?.as_str()?.to_owned();
    let question = object.get("question")?.as_str()?.to_owned();
    let header = optional_string(object, "header")?;
    let options = match object.get("options") {
        None => None,
        Some(Value::Array(options)) => {
            let mut items = Vec::with_capacity(options.len());
            for option in options {
                items.push(tool_option_of(option)?);
            }
            Some(items)
        }
        Some(_) => return None,
    };
    let multi_select = match object.get("multi_select") {
        None => None,
        Some(Value::Bool(flag)) => Some(*flag),
        Some(_) => return None,
    };
    Some(AskUserQuestionItem {
        id,
        question,
        detail: None,
        header,
        options,
        multi_select,
        intent: None,
    })
}

/// Read one tool-vocabulary option.
fn tool_option_of(value: &Value) -> Option<AskUserQuestionOption> {
    let object = value.as_object()?;
    let label = object.get("label")?.as_str()?.to_owned();
    let description = optional_string(object, "description")?;
    Some(AskUserQuestionOption { label, description })
}

/// Read an optional string field: absent is `None`, a string is its value, and
/// anything else makes the whole record unreadable.
fn optional_string(object: &Map<String, Value>, key: &str) -> Option<Option<String>> {
    match object.get(key) {
        None => Some(None),
        Some(Value::String(text)) => Some(Some(text.clone())),
        Some(_) => None,
    }
}

/// Whether a recorded result carries the pending payload.
pub fn is_pending_result(content: &[ContentBlock]) -> bool {
    let Some(text) = content.iter().find_map(ContentBlock::as_text) else {
        return false;
    };
    let Ok(parsed) = serde_json::from_str::<Value>(text) else {
        // A non-JSON result text is a failure message, never the pending
        // payload.
        return false;
    };
    parsed
        .as_object()
        .and_then(|object| object.get("pending"))
        .and_then(Value::as_bool)
        == Some(true)
}

/// Read the answer batch out of one recorded text: a tool result or a late
/// reply.
///
/// The batch is the `answers` array; each item is strict — an unknown field
/// makes the batch unreadable, exactly like the TypeScript strict schema.
pub fn answer_batch_of(content: &[ContentBlock]) -> Option<Vec<AskUserQuestionAnswerItem>> {
    let text = content.iter().find_map(ContentBlock::as_text)?;
    let parsed: Value = serde_json::from_str(text).ok()?;
    let answers = parsed.as_object()?.get("answers")?.as_array()?;
    let mut items = Vec::with_capacity(answers.len());
    for answer in answers {
        items.push(answer_item_of(answer)?);
    }
    Some(items)
}

/// Read one answer item.
fn answer_item_of(value: &Value) -> Option<AskUserQuestionAnswerItem> {
    let object = value.as_object()?;
    if object
        .keys()
        .any(|key| !matches!(key.as_str(), "id" | "selected" | "custom"))
    {
        return None;
    }
    let id = object.get("id")?.as_str()?.to_owned();
    let selected = object.get("selected")?.as_array()?;
    let mut labels = Vec::with_capacity(selected.len());
    for label in selected {
        labels.push(label.as_str()?.to_owned());
    }
    let custom = optional_string(object, "custom")?;
    Some(AskUserQuestionAnswerItem {
        id,
        selected: labels,
        custom,
    })
}

/// Close one answerable question and keep the answers it settled with.
///
/// Returns `None` when the call is no longer answerable: a reply whose call is
/// gone changes nothing.
fn settle_question(
    view: &UserQuestionProjectionView,
    call_id: &ToolCallId,
    answers: Vec<AskUserQuestionAnswerItem>,
) -> Option<UserQuestionProjectionView> {
    let question = view.active.iter().find(|item| item.call_id == *call_id)?;
    let active = view
        .active
        .iter()
        .filter(|item| item.call_id != *call_id)
        .cloned()
        .collect::<Vec<_>>();
    let mut settled = view.settled.clone();
    settled.push(SettledUserQuestion {
        call_id: question.call_id.clone(),
        answers,
    });
    Some(UserQuestionProjectionView { active, settled })
}

/// Apply one session event to this session's question fold.
///
/// Returns `None` when the event changes nothing — the caller keeps the
/// previous state reference, exactly like the TypeScript fold returning the
/// same object.
pub fn apply_user_question_event(
    fold: &UserQuestionFold,
    event: &SessionEvent,
) -> Option<UserQuestionFold> {
    match event {
        SessionEvent::Core(core) => apply_core_event(fold, core),
        SessionEvent::Extension(extension) => apply_extension_event(fold, extension),
    }
}

/// Apply one core event.
fn apply_core_event(fold: &UserQuestionFold, core: &CoreEvent) -> Option<UserQuestionFold> {
    match core {
        CoreEvent::RequestHeader { header, .. } => {
            // The log records the exact schema list the model saw; a seeded or
            // tool-less header carries none, which reads as the legacy tool.
            let timed = header.tools.as_ref().is_some_and(|tools| {
                tools
                    .iter()
                    .any(|tool| is_timed_ask_user_question_schema(&tool.name, &tool.parameters))
            });
            if timed == fold.timed {
                return None;
            }
            Some(UserQuestionFold {
                timed,
                questions: Arc::clone(&fold.questions),
            })
        }
        CoreEvent::ToolCall {
            call_id,
            name,
            arguments,
            ..
        } => {
            if !fold.timed || name != ASK_USER_QUESTION_TOOL {
                return None;
            }
            let questions = questions_of(arguments)?;
            let active = fold
                .questions
                .active
                .iter()
                .filter(|item| item.call_id != *call_id)
                .cloned()
                .collect::<Vec<_>>();
            let mut next = active;
            next.push(PendingUserQuestion {
                call_id: call_id.clone(),
                questions,
                state: UserQuestionState::Open,
            });
            Some(UserQuestionFold {
                timed: fold.timed,
                questions: Arc::new(UserQuestionProjectionView {
                    active: next,
                    settled: fold.questions.settled.clone(),
                }),
            })
        }
        CoreEvent::ToolResult { message, error, .. } => {
            let call_id = &message.tool_call_id;
            if !fold
                .questions
                .active
                .iter()
                .any(|item| item.call_id == *call_id)
            {
                return None;
            }
            let continued = is_pending_result(&message.content)
                || error
                    .as_ref()
                    .is_some_and(|error| error.code == TOOL_OUTCOME_UNKNOWN);
            if continued {
                let active = fold
                    .questions
                    .active
                    .iter()
                    .map(|item| {
                        if item.call_id == *call_id {
                            let mut item = item.clone();
                            item.state = UserQuestionState::Continued;
                            item
                        } else {
                            item.clone()
                        }
                    })
                    .collect::<Vec<_>>();
                return Some(UserQuestionFold {
                    timed: fold.timed,
                    questions: Arc::new(UserQuestionProjectionView {
                        active,
                        settled: fold.questions.settled.clone(),
                    }),
                });
            }
            let answers = if error.is_none() && !message.is_error {
                answer_batch_of(&message.content)
            } else {
                None
            };
            match answers {
                None => {
                    // Any failure drops the question: it can never be
                    // answered, and it was never settled either.
                    let active = fold
                        .questions
                        .active
                        .iter()
                        .filter(|item| item.call_id != *call_id)
                        .cloned()
                        .collect::<Vec<_>>();
                    Some(UserQuestionFold {
                        timed: fold.timed,
                        questions: Arc::new(UserQuestionProjectionView {
                            active,
                            settled: fold.questions.settled.clone(),
                        }),
                    })
                }
                Some(answers) => {
                    let settled = settle_question(&fold.questions, call_id, answers)?;
                    Some(UserQuestionFold {
                        timed: fold.timed,
                        questions: Arc::new(settled),
                    })
                }
            }
        }
        CoreEvent::UserMessage { message } => {
            if message.source.kind() != USER_QUESTION_REPLY_SOURCE_KIND {
                return None;
            }
            let call_id = reply_call_id(message)?;
            let answers = answer_batch_of(&message.content).unwrap_or_default();
            let settled = settle_question(&fold.questions, &call_id, answers)?;
            Some(UserQuestionFold {
                timed: fold.timed,
                questions: Arc::new(settled),
            })
        }
        _ => None,
    }
}

/// Apply one plugin-declared event.
fn apply_extension_event(
    fold: &UserQuestionFold,
    extension: &ExtensionEvent,
) -> Option<UserQuestionFold> {
    if extension.kind() != PTC_DISPATCH_TAG {
        return None;
    }
    let dispatch = decode_ptc_dispatch(extension)?;
    if dispatch.name != ASK_USER_QUESTION_TOOL || dispatch.is_error {
        return None;
    }
    if !is_pending_result(&dispatch.content) {
        return None;
    }
    let arguments = serde_json::to_string(&dispatch.arguments).ok()?;
    let questions = questions_of(&arguments)?;
    let active = fold
        .questions
        .active
        .iter()
        .filter(|item| item.call_id != dispatch.sub_call_id)
        .cloned()
        .collect::<Vec<_>>();
    let mut next = active;
    next.push(PendingUserQuestion {
        call_id: dispatch.sub_call_id,
        questions,
        state: UserQuestionState::Continued,
    });
    Some(UserQuestionFold {
        timed: fold.timed,
        questions: Arc::new(UserQuestionProjectionView {
            active: next,
            settled: fold.questions.settled.clone(),
        }),
    })
}

/// The payload of one PTC sub-dispatch settle record.
///
/// `dsh-ptc` owns this shape, but it is not a dependency of this crate, so the
/// fields this fold reads are declared locally.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PtcDispatchPayload {
    /// The opaque sub-call id the record pairs on.
    sub_call_id: ToolCallId,
    /// The dispatched tool name.
    name: String,
    /// The JSON-normalized arguments the tool received.
    arguments: Value,
    /// Whether the sub-call settled as an error.
    is_error: bool,
    /// The sub-call's complete model-facing content.
    content: Vec<ContentBlock>,
}

/// Decode a PTC dispatch payload with its `type` tag removed.
fn decode_ptc_dispatch(extension: &ExtensionEvent) -> Option<PtcDispatchPayload> {
    let mut payload = extension.payload().clone();
    payload.remove("type");
    serde_json::from_value(Value::Object(payload)).ok()
}

/// The call a late reply names.
fn reply_call_id(message: &dsh_llm::UserMessage) -> Option<ToolCallId> {
    let dsh_llm::MessageSource::Extension(source) = &message.source else {
        return None;
    };
    let call_id = source.fields().get("callId")?.as_str()?;
    Some(ToolCallId::new(call_id))
}

/// Fold a whole session log into its question state.
pub fn fold_user_questions(events: &[SessionEvent]) -> UserQuestionProjectionView {
    let mut fold = UserQuestionFold::initial();
    for event in events {
        if let Some(next) = apply_user_question_event(&fold, event) {
            fold = next;
        }
    }
    fold.questions.as_ref().clone()
}

/// The `userQuestions` projection: open, continued, and settled timed
/// questions for every client.
pub fn user_question_projection() -> ProjectionDefinition<UserQuestionProjectionState> {
    ProjectionDefinition::host_only(
        KEY,
        STATE_VERSION,
        |_header: &SessionHeader, inherited: SessionLogOffset| UserQuestionProjectionState {
            inherited_event_count: inherited,
            timed: false,
            questions: Arc::new(UserQuestionProjectionView::empty()),
        },
        apply_state,
    )
    .with_wire(
        |state| Arc::clone(&state.questions),
        |view: &UserQuestionProjectionView| validate_projection_view(view),
    )
}

/// Apply one committed event to the projection state.
///
/// `None` keeps the previous state reference; inherited events below the
/// inherited offset are never folded.
fn apply_state(
    state: &UserQuestionProjectionState,
    event: &SessionEvent,
    seq: SessionSeq,
) -> Option<UserQuestionProjectionState> {
    if seq.get() < state.inherited_event_count.get() {
        return None;
    }
    let fold = UserQuestionFold {
        timed: state.timed,
        questions: Arc::clone(&state.questions),
    };
    let next = apply_user_question_event(&fold, event)?;
    Some(UserQuestionProjectionState {
        inherited_event_count: state.inherited_event_count,
        timed: next.timed,
        questions: next.questions,
    })
}

/// Validate a client view: active call ids are unique and non-empty.
pub fn validate_projection_view(view: &UserQuestionProjectionView) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    for question in &view.active {
        if question.call_id.as_str().is_empty() {
            return Err("a pending question needs a non-empty call id".to_string());
        }
        if !seen.insert(question.call_id.as_str()) {
            return Err("callIds must be unique".to_string());
        }
    }
    Ok(())
}
