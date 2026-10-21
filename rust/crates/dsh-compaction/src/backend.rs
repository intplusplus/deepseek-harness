//! Async summarization backend: turn a replayed conversation region into a
//! durable `compaction/summary` record by calling a model through the provider
//! boundary.
//!
//! The backend is the only part of the compaction layer that performs an
//! asynchronous model call, so it is the only part that needs a runtime. It is
//! object-safe and network-free in tests: [`LlmSummarizationBackend`] wraps any
//! `&dyn LlmProvider`, including [`dsh_provider::ScriptedProvider`] and
//! [`dsh_provider::FnProvider`], so a test asserts on the exact summary it
//! produces without a network.

#![forbid(unsafe_code)]

use std::sync::Arc;

use async_trait::async_trait;
use dsh_llm::{
    ContentBlock, LlmCallConfig, Message, MessageId, TokenUsage, ToolSchema, UserMessage,
};
use dsh_provider::{LlmProvider, LlmRequest};
use serde::Serialize;

use super::error::CompactionError;

/// The replayed conversation prefix handed to the summarizer.
///
/// The system head (when present) leads the region messages so the auxiliary
/// call is a genuine prefix of the last routed request and can reuse the
/// provider's warm prefix cache. The trailing compaction instruction is appended
/// by the backend itself.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SummarizationInput {
    /// The conversation's tool schemas, reused for prefix-cache alignment.
    pub tools: Option<Vec<ToolSchema>>,
    /// The derived system head (when present) followed by the region messages.
    pub messages: Vec<Message>,
}

/// Safe text-only summary plus the exact auxiliary call envelope that recorded
/// it.
#[derive(Clone, Debug, PartialEq)]
pub struct SummaryResult {
    /// Safe, text-only summary blocks.
    pub summary: Vec<ContentBlock>,
    /// Complete provider output before the text-only summary projection.
    pub raw_output: Option<Vec<ContentBlock>>,
    /// True for exactly one call through the context's LLM seam.
    pub llm_stream_call: bool,
    /// Provider route that wrote the summary.
    pub provider: String,
    /// Model that wrote the summary.
    pub model: String,
    /// Generation cap sent by the summarize call, when one applied.
    pub max_tokens: Option<u64>,
    /// Provider-reported usage for the summarization request.
    pub usage: Option<TokenUsage>,
}

/// Produce one summary from a replayed region. Implementations call a model.
///
/// The trait is `async_trait` so it stays object-safe; a caller holds the
/// backend as `&dyn SummarizationBackend` and the same boundary works for a
/// scripted provider in tests and a real HTTP provider in production.
#[async_trait]
pub trait SummarizationBackend: Send + Sync {
    /// Run one summarization call and return its safe summary envelope.
    async fn summarize(
        &self,
        input: &SummarizationInput,
        provider: &str,
        model: &str,
        max_tokens: Option<u64>,
    ) -> Result<SummaryResult, CompactionError>;
}

/// The summarization directive, delivered as the FINAL user message after the
/// replayed conversation so the provider's KV cache is reused instead of
/// invalidated. Mirrors `COMPACTION_INSTRUCTION` in `packages/compaction`.
const COMPACTION_INSTRUCTION: &str = "You are now acting as a compaction engine for this AI coding assistant. Condense the conversation ABOVE into a structured checkpoint that lets another model resume the work with no loss of essential context.\n\nOutput EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write \"(none)\" for an empty section — never drop a section.\n\n## Primary Request and Intent\n## Key Technical Concepts\n## Files and Code\n## Errors and Fixes\n## Pending Jobs\n## Current Work\n## Next Step\n## Critical Context\n\nRules:\n- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.\n- Capture user feedback and explicit instructions faithfully, especially corrections.\n- Do NOT mention this summarization request or that the context was compacted.\n- Output only the checkpoint text: do not call any tool or take any other action.";

/// Tags wrapping the structured summary inside the landed checkpoint node.
const SUMMARY_OPEN_TAG: &str = "<compacted-summary>";
const SUMMARY_CLOSE_TAG: &str = "</compacted-summary>";

/// Framing that makes the replacement user message established context.
const CHECKPOINT_PREAMBLE: &str =
    "This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured context as established background and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.";

/// A summarization backend that calls a real (or scripted) model provider.
///
/// It replays the conversation prefix (system head, tools, region messages) and
/// appends the compaction instruction as the final user message, exactly like
/// the TypeScript `summarizeWithLlm` so the provider's warm prefix cache is
/// reused. The returned summary is text-only — image output is rejected.
pub struct LlmSummarizationBackend {
    provider: Arc<dyn LlmProvider>,
}

impl LlmSummarizationBackend {
    /// Wrap one provider as the summarization backend.
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self { provider }
    }
}

#[async_trait]
impl SummarizationBackend for LlmSummarizationBackend {
    async fn summarize(
        &self,
        input: &SummarizationInput,
        provider: &str,
        model: &str,
        max_tokens: Option<u64>,
    ) -> Result<SummaryResult, CompactionError> {
        let mut messages: Vec<Message> = input.messages.clone();
        messages.push(Message::User(UserMessage::new(
            MessageId::random(),
            vec![ContentBlock::text(COMPACTION_INSTRUCTION)],
        )));
        let request = LlmRequest {
            config: LlmCallConfig {
                provider: provider.to_string(),
                model: model.to_string(),
                reasoning_effort: None,
                temperature: None,
                max_tokens,
                stop: None,
            },
            messages,
            tools: input.tools.clone().unwrap_or_default(),
        };
        let response = self
            .provider
            .generate(&request)
            .await
            .map_err(|failure| CompactionError::SummaryProvider(failure.message))?;

        let raw_output = response.message.content.clone();
        let summary: Vec<ContentBlock> = raw_output
            .iter()
            .filter_map(ContentBlock::as_text)
            .map(|text| ContentBlock::text(text.to_string()))
            .collect();
        if !summary
            .iter()
            .any(|block| !block.as_text().map(str::trim).unwrap_or("").is_empty())
        {
            return Err(CompactionError::SummaryEmpty);
        }

        Ok(SummaryResult {
            summary,
            raw_output: Some(raw_output),
            llm_stream_call: true,
            provider: provider.to_string(),
            model: model.to_string(),
            max_tokens,
            usage: response.usage,
        })
    }
}

/// Wrap safe text-only summary blocks in the durable checkpoint framing.
///
/// The framing turns the replacement user message into established background,
/// matching the TypeScript `frameSummary` output.
pub fn frame_summary(summary: &[ContentBlock]) -> Vec<ContentBlock> {
    let mut blocks = Vec::with_capacity(summary.len() + 2);
    blocks.push(ContentBlock::text(format!(
        "{CHECKPOINT_PREAMBLE}\n\n{SUMMARY_OPEN_TAG}"
    )));
    blocks.extend_from_slice(summary);
    blocks.push(ContentBlock::text(SUMMARY_CLOSE_TAG));
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsh_llm::AssistantMessage;
    use dsh_provider::{text_response, FnProvider, ScriptedProvider};

    fn blocks_of(result: &SummaryResult) -> Vec<String> {
        result
            .summary
            .iter()
            .filter_map(ContentBlock::as_text)
            .map(str::to_string)
            .collect()
    }

    #[tokio::test]
    async fn fn_provider_summarizes_the_replayed_region() {
        let summary_text = "## Current Work\n- editing the loop driver.";
        let backend = LlmSummarizationBackend::new(Arc::new(FnProvider::new(|_request| {
            Ok(text_response("mock", "mock", summary_text))
        })));

        let input = SummarizationInput {
            messages: vec![Message::Assistant(AssistantMessage::new(
                MessageId::random(),
                vec![ContentBlock::text("old answer")],
                "mock",
                "mock",
            ))],
            tools: None,
        };
        let result = backend
            .summarize(&input, "mock", "mock", Some(128))
            .await
            .expect("summarize succeeds");

        assert_eq!(blocks_of(&result), vec![summary_text.to_string()]);
        assert!(result.llm_stream_call);
        assert_eq!(result.provider, "mock");
        assert_eq!(result.model, "mock");
        assert_eq!(result.max_tokens, Some(128));
    }

    #[tokio::test]
    async fn scripted_provider_summarizes_and_frames() {
        let backend =
            LlmSummarizationBackend::new(Arc::new(ScriptedProvider::new(vec![text_response(
                "mock",
                "mock",
                "compacted checkpoint text",
            )])));

        let result = backend
            .summarize(&SummarizationInput::default(), "mock", "mock", None)
            .await
            .expect("summarize succeeds");

        let framed = frame_summary(&result.summary);
        assert!(framed.len() >= 3);
        assert!(framed[0].as_text().unwrap().contains(SUMMARY_OPEN_TAG));
        assert!(framed
            .last()
            .unwrap()
            .as_text()
            .unwrap()
            .contains(SUMMARY_CLOSE_TAG));
    }

    #[tokio::test]
    async fn empty_provider_output_is_rejected() {
        let backend = LlmSummarizationBackend::new(Arc::new(FnProvider::new(|_| {
            Ok(text_response("mock", "mock", ""))
        })));
        let error = backend
            .summarize(&SummarizationInput::default(), "mock", "mock", None)
            .await
            .unwrap_err();
        assert!(matches!(error, CompactionError::SummaryEmpty));
    }
}
