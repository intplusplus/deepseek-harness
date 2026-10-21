//! Async summarization backend integration: the real model call is exercised
//! through a scripted provider so no network is involved.

mod common;

use std::sync::Arc;

use dsh_compaction::{frame_summary, LlmSummarizationBackend, SummarizationBackend, SummarizationInput};
use dsh_llm::{AssistantMessage, ContentBlock, Message, MessageId};
use dsh_provider::{FnProvider, ScriptedProvider, text_response};

#[tokio::test]
async fn scripted_provider_drives_a_real_summarize_call() {
    let provider = Arc::new(ScriptedProvider::new(vec![text_response(
        "mock",
        "mock",
        "## Current Work\n- compacting the agent loop driver.",
    )]));
    let backend = LlmSummarizationBackend::new(provider);

    let input = SummarizationInput {
        messages: vec![Message::Assistant(AssistantMessage::new(
            MessageId::random(),
            vec![ContentBlock::text("old conversation content")],
            "mock",
            "mock",
        ))],
        tools: None,
    };

    let result = backend
        .summarize(&input, "mock", "mock", Some(256))
        .await
        .expect("summarize succeeds");

    // The summary is the text-only projection of the raw output.
    assert_eq!(
        result.summary,
        vec![ContentBlock::text(
            "## Current Work\n- compacting the agent loop driver."
        )]
    );
    assert!(result.llm_stream_call);
    assert_eq!(result.provider, "mock");
    assert_eq!(result.model, "mock");
    assert_eq!(result.max_tokens, Some(256));

    // Framing turns the summary into an established background checkpoint.
    let framed = frame_summary(&result.summary);
    assert!(framed.len() >= 3);
    assert!(framed
        .first()
        .unwrap()
        .as_text()
        .unwrap()
        .contains("<compacted-summary>"));
    assert!(framed
        .last()
        .unwrap()
        .as_text()
        .unwrap()
        .contains("</compacted-summary>"));
}

#[tokio::test]
async fn fn_provider_backend_is_network_free() {
    let provider = Arc::new(FnProvider::new(|_request| {
        Ok(text_response("deepseek", "v3", "a single checkpoint line"))
    }));
    let backend = LlmSummarizationBackend::new(provider);

    let result = backend
        .summarize(&SummarizationInput::default(), "deepseek", "v3", None)
        .await
        .expect("summarize succeeds");
    assert_eq!(result.summary.len(), 1);
    assert!(!result.summary.is_empty());
}
