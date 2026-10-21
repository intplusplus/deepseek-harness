//! One model request and its response, in canonical vocabulary.

#![forbid(unsafe_code)]

use dsh_llm::{
    AssistantMessage, AssistantStreamRecord, FinishReason, LlmCallConfig, Message, TokenUsage,
    ToolSchema,
};

/// Everything a model call needs, already in canonical form.
///
/// The loop assembles this from the log; the provider translates it. A request
/// carries no provider-specific field, so adding a provider cannot leak into
/// the session vocabulary.
#[derive(Clone, Debug)]
pub struct LlmRequest {
    /// The call configuration to use.
    pub config: LlmCallConfig,
    /// The model-visible history, in order.
    pub messages: Vec<Message>,
    /// The tool schemas to offer.
    pub tools: Vec<ToolSchema>,
}

/// What the model produced for one request.
///
/// `stream` is the exact timed record the log stores, so a replay reproduces
/// the response's timing without re-calling the model.
#[derive(Clone, Debug)]
pub struct LlmResponse {
    /// The assembled assistant message.
    pub message: AssistantMessage,
    /// The exact timed stream, compacted without joining delta boundaries.
    pub stream: Vec<AssistantStreamRecord>,
    /// Token accounting, when the provider reported it.
    pub usage: Option<TokenUsage>,
    /// Why the response stopped.
    pub finish_reason: FinishReason,
}

impl LlmResponse {
    /// A response that stopped on its own, with no stream or usage.
    pub fn new(message: AssistantMessage) -> Self {
        Self {
            message,
            stream: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Stop,
        }
    }

    /// Attach the timed stream.
    pub fn with_stream(mut self, stream: Vec<AssistantStreamRecord>) -> Self {
        self.stream = stream;
        self
    }

    /// Attach token accounting.
    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Attach the finish reason.
    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = reason;
        self
    }

    /// The tool calls this response requests, in model order.
    pub fn tool_calls(&self) -> Vec<dsh_llm::ToolCallRef<'_>> {
        self.message
            .content
            .iter()
            .filter_map(dsh_llm::ContentBlock::as_tool_call)
            .collect()
    }
}
