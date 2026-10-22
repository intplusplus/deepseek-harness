//! The model-facing `ask_user_question` tool.
//!
//! Ported from `packages/interaction/tool-ask-user` — `src/index.ts` (the
//! blocking legacy tool) and `src/timed.ts` (the opt-in timed tool).
//!
//! The tool pauses until a UI provider returns a human answer, then feeds that
//! answer back into the agent loop as an ordinary tool result.
//!
//! ## Two tools, one name
//!
//! The TypeScript package registers one of two schemas under the same
//! `ask_user_question` name, selected by the deployment's configuration:
//!
//! - the **legacy** tool blocks indefinitely and its request is *unkeyed* —
//!   the Client card has no tool call row to reopen from;
//! - the **timed** tool waits for a bounded foreground window and may settle
//!   as *pending*, which keeps the call answerable. Its schema declares the
//!   [`TIMED_WAIT_PARAMETER`], and that declaration is exactly what the
//!   `userQuestions` projection reads to tell a timed call from a legacy one.
//!
//! ## The synchronous seam
//!
//! [`ToolHandler::call`] is synchronous in Rust, so the tool calls the
//! question service synchronously. A deployment bridges its UI thread to the
//! [`UserQuestionAnswerer`] seam; the timeout of the timed path is enforced by
//! the answerer, which receives the wait key and its budget.

#![forbid(unsafe_code)]

use std::sync::Arc;

use dsh_agent::Agent;
use dsh_llm::{ContentBlock, ToolCallId, ToolSchema};
use dsh_session::ToolErrorInfo;
use dsh_tools::{CancelSignal, ToolCallInput, ToolExecutionResult, ToolHandler};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::projection::{ASK_USER_QUESTION_TOOL, TIMED_WAIT_PARAMETER};
use crate::questions::{
    AskUserQuestionAnswer, AskUserQuestionItem, AskUserQuestionOption, AskUserQuestionRequest,
    QuestionWait, TimedUserQuestionResult, UserQuestionError, UserQuestionService,
};

/// The default foreground wait of the timed tool, in seconds.
pub const DEFAULT_TIMEOUT_SECONDS: i64 = 120;

/// The largest foreground wait the timed tool accepts, in seconds.
pub const MAX_TIMEOUT_SECONDS: i64 = 2_147_483;

/// The instruction the pending result carries in its `message` field.
///
/// It is a field of the result value rather than prose beside it because the
/// recorded result text is read back as one JSON object: the `userQuestions`
/// projection decides from it that the call stays answerable, and the Client
/// question row decides from it that the recorded result is the timeout, not
/// an answer batch.
pub const PENDING_NOTICE: &str = "No answer batch arrived before the timeout. This is pending, not a skipped answer. Continue useful independent work. The user can still answer; their reply will be a user message identified as answer_to_pending_question with this callId and the original questions. Do not treat this as permission.";

/// Which of the two `ask_user_question` tools this handler is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AskUserQuestionMode {
    /// The blocking legacy tool: no wait parameter, an unkeyed request.
    Legacy,
    /// The opt-in timed tool: a foreground wait and a possible pending result.
    Timed,
}

/// The model-facing question, in the tool schema's own vocabulary.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolQuestion {
    /// Stable id for this question; echoed in the answer.
    pub id: String,
    /// The question to display.
    pub question: String,
    /// Optional short heading for the question.
    #[serde(default)]
    pub header: Option<String>,
    /// Optional choices the UI can render as a menu.
    #[serde(default)]
    pub options: Option<Vec<ToolOption>>,
    /// Whether the user may select more than one option.
    #[serde(default)]
    pub multi_select: Option<bool>,
}

/// The model-facing option, in the tool schema's own vocabulary.
#[derive(Clone, Debug, Deserialize)]
pub struct ToolOption {
    /// Short user-facing option label.
    pub label: String,
    /// One sentence explaining the tradeoff or impact.
    #[serde(default)]
    pub description: Option<String>,
}

/// The parsed model arguments.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ToolArguments {
    questions: Vec<ToolQuestion>,
    #[serde(default)]
    timeout: Option<f64>,
}

impl ToolQuestion {
    /// Translate the model schema into the service request without changing
    /// ownership.
    fn to_service(&self) -> AskUserQuestionItem {
        AskUserQuestionItem {
            id: self.id.clone(),
            question: self.question.clone(),
            detail: None,
            header: self.header.clone(),
            options: self.options.as_ref().map(|options| {
                options
                    .iter()
                    .map(|option| AskUserQuestionOption {
                        label: option.label.clone(),
                        description: option.description.clone(),
                    })
                    .collect()
            }),
            multi_select: self.multi_select,
            intent: None,
        }
    }
}

/// The model-facing `ask_user_question` tool.
pub struct AskUserQuestionTool {
    mode: AskUserQuestionMode,
    timeout_seconds: i64,
    service: Arc<UserQuestionService>,
    agent: Option<Agent>,
}

impl AskUserQuestionTool {
    /// The blocking legacy tool.
    pub fn legacy(service: Arc<UserQuestionService>) -> Self {
        Self {
            mode: AskUserQuestionMode::Legacy,
            timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
            service,
            agent: None,
        }
    }

    /// The opt-in timed tool with an explicit foreground wait.
    ///
    /// `timeout_seconds` is `-1` for an indefinite wait, or a positive integer
    /// up to [`MAX_TIMEOUT_SECONDS`].
    pub fn timed(
        service: Arc<UserQuestionService>,
        timeout_seconds: i64,
    ) -> Result<Self, AskUserQuestionError> {
        validate_timeout_seconds(timeout_seconds)?;
        Ok(Self {
            mode: AskUserQuestionMode::Timed,
            timeout_seconds,
            service,
            agent: None,
        })
    }

    /// Attach the agent the tool asks on behalf of.
    ///
    /// The timed path requires one: a foreground wait is routed through the
    /// live agent, and without it there is nothing to route to.
    #[must_use]
    pub fn with_agent(mut self, agent: Agent) -> Self {
        self.agent = Some(agent);
        self
    }

    /// Which of the two tools this handler is.
    pub fn mode(&self) -> AskUserQuestionMode {
        self.mode
    }

    /// The configured foreground wait, in seconds; `-1` for an indefinite wait.
    pub fn timeout_seconds(&self) -> i64 {
        self.timeout_seconds
    }

    /// The schema the model sees.
    pub fn tool_schema(&self) -> ToolSchema {
        let description = match self.mode {
            AskUserQuestionMode::Legacy => {
                "Ask the user a concise question when you need confirmation, a choice, or missing information before proceeding."
            }
            AskUserQuestionMode::Timed => {
                "Ask brief, direct, self-contained questions about missing information, preferences, or decisions. Use user-facing terms; assume no knowledge of background work or internal names. Use distinct stable question IDs. A submitted skipped question is an answer item with empty selected and no custom; pending instead means no answer batch arrived before the timeout and the user can still answer."
            }
        };
        let mut properties = Map::new();
        properties.insert(
            "questions".to_string(),
            json!({
                "type": "array",
                "required": true,
                "description": match self.mode {
                    AskUserQuestionMode::Legacy => "Questions to ask the user before continuing.",
                    AskUserQuestionMode::Timed => "Questions to ask the user.",
                },
                "items": {
                    "type": "object",
                    "additionalProperties": true,
                    "properties": {
                        "id": {
                            "type": "string",
                            "required": true,
                            "description": "Stable id for this question; echoed in the answer.",
                        },
                        "question": {
                            "type": "string",
                            "required": true,
                            "description": "The specific question to ask the user.",
                        },
                        "header": {
                            "type": "string",
                            "description": "Optional short heading for the question, such as \"Confirm\" or \"Choose Mode\".",
                        },
                        "options": {
                            "type": "array",
                            "description": "Optional choices to show the user. If you recommend one, put it first and append \"(Recommended)\" to that label.",
                            "items": {
                                "type": "object",
                                "additionalProperties": true,
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "required": true,
                                        "description": "Short user-facing option label.",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "One sentence explaining the tradeoff or impact.",
                                    },
                                },
                            },
                        },
                        "multi_select": {
                            "type": "boolean",
                            "description": "Whether the user may select more than one option. Defaults to false.",
                        },
                    },
                },
            }),
        );
        if self.mode == AskUserQuestionMode::Timed {
            properties.insert(
                TIMED_WAIT_PARAMETER.to_string(),
                json!({
                    "type": "integer",
                    "description": format!(
                        "Wait seconds for the entire batch (default {}); omit unless the user specifies a duration. Use -1 only when an answer is required before proceeding.",
                        self.timeout_seconds
                    ),
                }),
            );
        }
        let mut parameters = Map::new();
        parameters.insert("type".to_string(), Value::String("object".to_string()));
        // `defineTool` wraps the declared parameter map in a JSON-schema
        // `properties` object; the model calls with named arguments under it.
        parameters.insert("properties".to_string(), Value::Object(properties));
        parameters.insert(
            "required".to_string(),
            Value::Array(vec![Value::String("questions".to_string())]),
        );
        ToolSchema::new(
            ASK_USER_QUESTION_TOOL,
            description,
            Value::Object(parameters),
        )
    }

    /// Parse and validate the model arguments.
    fn parse_arguments(
        &self,
        input: &ToolCallInput,
    ) -> Result<ToolArguments, AskUserQuestionError> {
        let arguments: ToolArguments = serde_json::from_value(input.arguments.clone()).map_err(|error| {
            AskUserQuestionError::Arguments(format!(
                "ask_user_question expects a `questions` array of `{{id, question}}` objects: {error}"
            ))
        })?;
        Ok(arguments)
    }

    /// The effective foreground wait: the model's own value when it supplied
    /// one, the configured default otherwise.
    fn effective_timeout(&self, arguments: &ToolArguments) -> Result<i64, AskUserQuestionError> {
        let Some(timeout) = arguments.timeout else {
            return Ok(self.timeout_seconds);
        };
        if !timeout.is_finite() || timeout.fract() != 0.0 {
            return Err(AskUserQuestionError::Timeout(
                "timeout must be -1 or a positive integer up to 2147483 seconds".to_string(),
            ));
        }
        let timeout = timeout as i64;
        validate_timeout_seconds(timeout)?;
        Ok(timeout)
    }

    /// Run the question and project the answer onto the model-facing result.
    fn run(
        &self,
        input: &ToolCallInput,
        arguments: &ToolArguments,
        cancel: &CancelSignal,
    ) -> ToolExecutionResult {
        let timeout = match self.effective_timeout(arguments) {
            Ok(timeout) => timeout,
            Err(error) => return error.into_result(),
        };
        if let Err(error) = validate_question_ids(&arguments.questions) {
            return error.into_result();
        }

        let questions = arguments
            .questions
            .iter()
            .map(ToolQuestion::to_service)
            .collect::<Vec<_>>();
        let mut request = AskUserQuestionRequest::new(questions).with_signal(cancel.clone());
        if let Some(agent) = &self.agent {
            request = request.with_agent(agent.clone());
        }

        match self.mode {
            AskUserQuestionMode::Legacy => {
                let result = self.service.ask(&request);
                match result {
                    Ok(answer) => answer_result(&answer),
                    Err(error) => error.into_result(),
                }
            }
            AskUserQuestionMode::Timed => {
                if self.agent.is_none() {
                    // The timed path routes the question through the live
                    // agent; without one there is nothing to route to.
                    return ToolExecutionResult::failure("timed questions require a live agent");
                }
                if timeout == -1 {
                    // `wait` without `timed`: this tool's indefinite form still
                    // keys the Client card by call id, so the UI can reopen a
                    // closed panel from the tool call. The legacy tool keeps
                    // its unkeyed request.
                    let request = request.with_wait(QuestionWait::keyed(input.call_id.clone()));
                    match self.service.ask(&request) {
                        Ok(answer) => answer_result(&answer),
                        Err(error) => error.into_result(),
                    }
                } else {
                    let timeout_ms = u64::try_from(timeout)
                        .ok()
                        .and_then(|seconds| seconds.checked_mul(1_000))
                        .unwrap_or(u64::MAX);
                    match self.service.ask_timed(&request, &input.call_id, timeout_ms) {
                        Ok(TimedUserQuestionResult::Answer(answer)) => answer_result(&answer),
                        Ok(TimedUserQuestionResult::Pending { call_id }) => {
                            pending_result(&call_id)
                        }
                        Err(error) => error.into_result(),
                    }
                }
            }
        }
    }
}

impl ToolHandler for AskUserQuestionTool {
    fn name(&self) -> &str {
        ASK_USER_QUESTION_TOOL
    }

    fn schema(&self) -> ToolSchema {
        self.tool_schema()
    }

    fn call(&self, input: &ToolCallInput, cancel: &CancelSignal) -> ToolExecutionResult {
        let arguments = match self.parse_arguments(input) {
            Ok(arguments) => arguments,
            Err(error) => return error.into_result(),
        };
        self.run(input, &arguments, cancel)
    }
}

/// The failure modes of the tool that are not user-question errors.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AskUserQuestionError {
    /// The model arguments did not match the tool schema.
    #[error("{0}")]
    Arguments(String),
    /// The foreground wait was not a valid platform timer.
    #[error("{0}")]
    Timeout(String),
    /// Two questions of one call shared an id.
    #[error("question id {id:?} must be unique within this call")]
    DuplicateQuestionId {
        /// The repeated question id.
        id: String,
    },
}

impl AskUserQuestionError {
    /// The model-facing failure this error becomes.
    fn into_result(self) -> ToolExecutionResult {
        ToolExecutionResult::failure(self.to_string())
    }
}

/// Validate a foreground wait: `-1`, or a positive integer up to
/// [`MAX_TIMEOUT_SECONDS`].
pub fn validate_timeout_seconds(timeout: i64) -> Result<i64, AskUserQuestionError> {
    if timeout == -1 || (1..=MAX_TIMEOUT_SECONDS).contains(&timeout) {
        return Ok(timeout);
    }
    Err(AskUserQuestionError::Timeout(
        "timeout must be -1 or a positive integer up to 2147483 seconds".to_string(),
    ))
}

/// Validate that every question of one call carries its own id.
pub fn validate_question_ids(questions: &[ToolQuestion]) -> Result<(), AskUserQuestionError> {
    let mut ids = std::collections::HashSet::new();
    for question in questions {
        if !ids.insert(question.id.as_str()) {
            return Err(AskUserQuestionError::DuplicateQuestionId {
                id: question.id.clone(),
            });
        }
    }
    Ok(())
}

/// Copy the service-owned answer arrays into the model-facing result.
fn answer_result(answer: &AskUserQuestionAnswer) -> ToolExecutionResult {
    let value = json!({
        "answers": answer
            .answers
            .iter()
            .map(|item| {
                let mut entry = Map::new();
                entry.insert("id".to_string(), Value::String(item.id.clone()));
                entry.insert(
                    "selected".to_string(),
                    Value::Array(
                        item.selected
                            .iter()
                            .map(|label| Value::String(label.clone()))
                            .collect(),
                    ),
                );
                if let Some(custom) = &item.custom {
                    entry.insert("custom".to_string(), Value::String(custom.clone()));
                }
                Value::Object(entry)
            })
            .collect::<Vec<_>>(),
    });
    ToolExecutionResult::success(vec![ContentBlock::text(value.to_string())]).with_value(value)
}

/// The model-facing result of a foreground wait that closed with no answer.
fn pending_result(call_id: &ToolCallId) -> ToolExecutionResult {
    let value = json!({
        "pending": true,
        "callId": call_id.as_str(),
        "message": PENDING_NOTICE,
    });
    ToolExecutionResult::success(vec![ContentBlock::text(value.to_string())]).with_value(value)
}

impl UserQuestionError {
    /// The model-facing failure this error becomes.
    ///
    /// A user-question failure carries its structured identity, so the caller
    /// can route it without parsing the message.
    fn into_result(self) -> ToolExecutionResult {
        ToolExecutionResult::failure(self.to_string()).with_error_info(ToolErrorInfo {
            name: self.name().to_string(),
            code: self.code().to_string(),
            reason: None,
        })
    }
}
