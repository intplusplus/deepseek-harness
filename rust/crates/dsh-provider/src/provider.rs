//! The provider trait and three reference implementations.

#![forbid(unsafe_code)]

use std::{collections::VecDeque, sync::Mutex};

use async_trait::async_trait;
use dsh_llm::{
    AssistantMessage, ContentBlock, FinishReason, LlmFailure, Message, MessageId, ToolCallId,
    UserMessage,
};

use super::request::{LlmRequest, LlmResponse};

/// The model-call boundary.
///
/// A provider either returns a canonical response or a structured
/// [`LlmFailure`]. It never panics and never returns a provider-native type:
/// translation happens here or not at all.
///
/// The call is asynchronous because a network request cannot be answered
/// without yielding. The loop holds providers as `&dyn LlmProvider`, so the
/// trait must stay object-safe; `#[async_trait]` desugars each method into a
/// boxed, `Send` future, which is the smallest form that keeps object safety
/// and lets an implementation hold borrows across an await. A native
/// `async fn` in a trait is not dyn-compatible, and hand-writing the boxed
/// signature in every implementation would repeat the same expansion.
///
/// The returned future must be polled inside a Tokio runtime; a provider that
/// needs another reactor is responsible for bridging to it.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Run one request to completion.
    async fn generate(&self, request: &LlmRequest) -> Result<LlmResponse, LlmFailure>;
}

/// A provider driven by a closure. Handy in tests and in examples.
///
/// The closure is synchronous; a provider that must await something
/// implements [`LlmProvider`] directly.
pub struct FnProvider<F>(F)
where
    F: Fn(&LlmRequest) -> Result<LlmResponse, LlmFailure> + Send + Sync;

impl<F> FnProvider<F>
where
    F: Fn(&LlmRequest) -> Result<LlmResponse, LlmFailure> + Send + Sync,
{
    /// Wrap a closure as a provider.
    pub fn new(handler: F) -> Self {
        Self(handler)
    }
}

#[async_trait]
impl<F> LlmProvider for FnProvider<F>
where
    F: Fn(&LlmRequest) -> Result<LlmResponse, LlmFailure> + Send + Sync,
{
    async fn generate(&self, request: &LlmRequest) -> Result<LlmResponse, LlmFailure> {
        (self.0)(request)
    }
}

/// A provider that replays a scripted list of responses, then fails.
///
/// Deterministic and network-free: it is what lets a test assert on the exact
/// log a loop produces without a model.
pub struct ScriptedProvider {
    responses: Mutex<VecDeque<LlmResponse>>,
}

impl ScriptedProvider {
    /// A provider with `responses` queued, consumed in order.
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
        }
    }

    /// Queue more responses at the back.
    pub fn push(&self, response: LlmResponse) {
        self.script().push_back(response);
    }

    /// How many scripted responses remain.
    pub fn remaining(&self) -> usize {
        self.script().len()
    }

    /// The script queue. A poisoned lock is recovered rather than propagated:
    /// the queue is still structurally valid, and a provider must not panic.
    fn script(&self) -> std::sync::MutexGuard<'_, VecDeque<LlmResponse>> {
        self.responses
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    async fn generate(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmFailure> {
        self.script().pop_front().ok_or_else(|| LlmFailure {
            message: "scripted provider exhausted".to_string(),
            code: "SCRIPT_EXHAUSTED".to_string(),
            status: None,
            provider_retry_after_ms: None,
            request_id: None,
            offload_images: None,
        })
    }
}

/// A provider that echoes the latest user message back as text.
///
/// Useful as a smoke target: it terminates immediately, asks for no tools, and
/// needs no key.
pub struct EchoProvider;

#[async_trait]
impl LlmProvider for EchoProvider {
    async fn generate(&self, request: &LlmRequest) -> Result<LlmResponse, LlmFailure> {
        let echoed: String = request
            .messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::User(message) => message
                    .content
                    .iter()
                    .filter_map(ContentBlock::as_text)
                    .next()
                    .map(str::to_string),
                _ => None,
            })
            .unwrap_or_default();
        let message = AssistantMessage::new(
            MessageId::random(),
            vec![ContentBlock::text(format!("echo: {echoed}"))],
            &request.config.provider,
            &request.config.model,
        );
        Ok(LlmResponse::new(message).with_finish_reason(FinishReason::Stop))
    }
}

/// Convenience for building a scripted response that asks for one tool call.
pub fn tool_call_response(
    provider: &str,
    model: &str,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> LlmResponse {
    let message = AssistantMessage::new(
        MessageId::random(),
        vec![ContentBlock::tool_call(
            ToolCallId::new(call_id),
            name,
            arguments,
        )],
        provider,
        model,
    );
    LlmResponse::new(message).with_finish_reason(FinishReason::ToolCalls)
}

/// Convenience for building a scripted response that ends the turn with text.
pub fn text_response(provider: &str, model: &str, text: &str) -> LlmResponse {
    let message = AssistantMessage::new(
        MessageId::random(),
        vec![ContentBlock::text(text)],
        provider,
        model,
    );
    LlmResponse::new(message)
}

/// Build a user message, the usual input to a turn.
pub fn user_message(text: &str) -> UserMessage {
    UserMessage::new(MessageId::random(), vec![ContentBlock::text(text)])
}
