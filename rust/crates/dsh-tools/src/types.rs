//! The tool execution contract: arguments, outcomes, failures, and modes.
//!
//! Ported from `packages/core/tools`: the result is a discriminated union where
//! a failure never carries a successful value and `concludesTurn` exists only
//! on a success — the type system, not a runtime check, keeps those apart.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use dsh_llm::{ContentBlock, ToolCallId, UserMessage};

pub use dsh_session::ToolErrorInfo;

/// Structured error code for a call cancelled after the turn aborted but
/// before the tool body was entered.
pub const TOOL_ABORTED_BEFORE_DISPATCH: &str = "ABORTED_BEFORE_DISPATCH";

/// Structured error code for a call naming a tool the registry does not own.
pub const TOOL_UNKNOWN_TOOL: &str = "UNKNOWN_TOOL";

/// Canonical failure detail; the structured identity is optional.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolFailure {
    /// Human-readable failure message without the `Error: ` envelope.
    pub message: String,
    /// Structured routing identity, persisted beside the model-facing text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<ToolErrorInfo>,
}

impl ToolFailure {
    /// A failure with a message and no structured identity.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            info: None,
        }
    }

    /// Attach the structured identity.
    pub fn with_info(mut self, info: ToolErrorInfo) -> Self {
        self.info = Some(info);
        self
    }
}

/// The execution-local outcome of one tool call.
///
/// A success carries a canonical `value` that is deliberately *not* durable —
/// only `content` reaches the model and the log. A failure carries a
/// [`ToolFailure`] and never a value, and never concludes a turn.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolExecutionResult {
    Success {
        /// Canonical execution-local value, omitted from durable events.
        value: Value,
        content: Vec<ContentBlock>,
        meta: Option<Value>,
        additional_contexts: Vec<UserMessage>,
        concludes_turn: bool,
    },
    Failure {
        error: ToolFailure,
        content: Vec<ContentBlock>,
        meta: Option<Value>,
        additional_contexts: Vec<UserMessage>,
    },
}

impl ToolExecutionResult {
    /// A successful result with `content` and no value, context, or meta.
    pub fn success(content: Vec<ContentBlock>) -> Self {
        Self::Success {
            value: Value::Null,
            content,
            meta: None,
            additional_contexts: Vec::new(),
            concludes_turn: false,
        }
    }

    /// A failed result whose model-facing content is the rendered message.
    pub fn failure(message: impl Into<String>) -> Self {
        let message = message.into();
        Self::Failure {
            error: ToolFailure::new(message.clone()),
            content: vec![ContentBlock::text(render_error(&message))],
            meta: None,
            additional_contexts: Vec::new(),
        }
    }

    /// Whether the call failed.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Failure { .. })
    }

    /// The model-facing content.
    pub fn content(&self) -> &[ContentBlock] {
        match self {
            Self::Success { content, .. } | Self::Failure { content, .. } => content,
        }
    }

    /// The tool-private presentation payload, when the tool declared one.
    pub fn meta(&self) -> Option<&Value> {
        match self {
            Self::Success { meta, .. } | Self::Failure { meta, .. } => meta.as_ref(),
        }
    }

    /// Contexts deferred to the next request, in call order.
    pub fn additional_contexts(&self) -> &[UserMessage] {
        match self {
            Self::Success {
                additional_contexts,
                ..
            }
            | Self::Failure {
                additional_contexts,
                ..
            } => additional_contexts,
        }
    }

    /// Whether the agent loop must stop after committing this result.
    ///
    /// Only a success can conclude a turn, so a failure always returns `false`.
    pub fn concludes_turn(&self) -> bool {
        match self {
            Self::Success { concludes_turn, .. } => *concludes_turn,
            Self::Failure { .. } => false,
        }
    }

    /// The structured error identity, when the call failed with one.
    pub fn error_info(&self) -> Option<&ToolErrorInfo> {
        match self {
            Self::Success { .. } => None,
            Self::Failure { error, .. } => error.info.as_ref(),
        }
    }

    /// Attach the canonical value (success only).
    pub fn with_value(mut self, value: Value) -> Self {
        if let Self::Success { value: slot, .. } = &mut self {
            *slot = value;
        }
        self
    }

    /// Attach the structured failure identity (failure only).
    pub fn with_error_info(mut self, info: ToolErrorInfo) -> Self {
        if let Self::Failure { error, .. } = &mut self {
            error.info = Some(info);
        }
        self
    }

    /// Attach the presentation payload.
    pub fn with_meta(mut self, meta: Value) -> Self {
        match &mut self {
            Self::Success { meta: slot, .. } | Self::Failure { meta: slot, .. } => {
                *slot = Some(meta)
            }
        }
        self
    }

    /// Replace the model-facing content, preserving value, failure identity,
    /// meta, and contexts.
    pub fn with_content(mut self, content: Vec<ContentBlock>) -> Self {
        match &mut self {
            Self::Success { content: slot, .. } | Self::Failure { content: slot, .. } => {
                *slot = content
            }
        }
        self
    }

    /// Defer contexts to the next request.
    pub fn with_contexts(mut self, contexts: Vec<UserMessage>) -> Self {
        match &mut self {
            Self::Success {
                additional_contexts,
                ..
            }
            | Self::Failure {
                additional_contexts,
                ..
            } => *additional_contexts = contexts,
        }
        self
    }

    /// Mark the turn terminal after this result commits (success only).
    pub fn concluding_turn(mut self) -> Self {
        if let Self::Success { concludes_turn, .. } = &mut self {
            *concludes_turn = true;
        }
        self
    }
}

/// Render a failure message into the model-facing text envelope.
fn render_error(message: &str) -> String {
    format!("Error: {message}")
}

/// Parse model arguments, preserving invalid JSON as text and mapping empty
/// input to `{}`.
///
/// The raw string is what the model produced and what the log records; parsing
/// never fails, so a malformed call still reaches the tool as text rather than
/// being dropped before dispatch.
pub fn parse_arguments(raw: &str) -> Value {
    if raw.is_empty() {
        return Value::Object(Map::new());
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(value) => value,
        Err(_) => Value::String(raw.to_owned()),
    }
}

/// How one call may overlap its siblings within a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolExecutionMode {
    /// May run alongside sibling parallel calls in a bounded pool.
    Parallel,
    /// Forms a barrier: it runs alone and later calls wait for it.
    Exclusive,
}

/// One model-requested call, parsed and ready to schedule.
#[derive(Clone, Debug)]
pub struct ToolCallInput {
    /// The call identity that pairs the call with its result.
    pub call_id: ToolCallId,
    /// The requested tool name.
    pub name: String,
    /// Parsed arguments: a JSON value, or the raw text when unparseable.
    pub arguments: Value,
}

/// Convenience: read one field of the parsed arguments.
pub fn argument<'a>(input: &'a ToolCallInput, key: &str) -> Option<&'a Value> {
    input.arguments.as_object().and_then(|map| map.get(key))
}
