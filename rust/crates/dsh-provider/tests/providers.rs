//! The provider boundary itself: object safety, asynchronous interleaving, and
//! the three reference implementations.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dsh_llm::{LlmCallConfig, LlmFailure, Message};
use dsh_provider::{
    text_response, user_message, EchoProvider, FnProvider, LlmProvider, LlmRequest, LlmResponse,
    ScriptedProvider,
};
use tokio::sync::Barrier;

fn request() -> LlmRequest {
    LlmRequest {
        config: LlmCallConfig {
            provider: "demo".to_string(),
            model: "demo-1".to_string(),
            reasoning_effort: None,
            temperature: None,
            max_tokens: None,
            stop: None,
        },
        messages: vec![Message::User(user_message("hello"))],
        tools: Vec::new(),
    }
}

/// A provider whose call cannot finish until a sibling call has started.
///
/// On a single-threaded runtime this only completes if `generate` really
/// yields: a blocking call would reach the barrier alone and never return.
struct BarrierProvider {
    barrier: Arc<Barrier>,
}

#[async_trait]
impl LlmProvider for BarrierProvider {
    async fn generate(&self, request: &LlmRequest) -> Result<LlmResponse, LlmFailure> {
        self.barrier.wait().await;
        Ok(text_response(
            &request.config.provider,
            &request.config.model,
            "together",
        ))
    }
}

/// Two model calls run on one thread and overlap.
#[tokio::test]
async fn concurrent_calls_interleave_on_a_single_thread() {
    let provider: Arc<dyn LlmProvider> = Arc::new(BarrierProvider {
        barrier: Arc::new(Barrier::new(2)),
    });
    let request = request();

    let calls = async { tokio::join!(provider.generate(&request), provider.generate(&request)) };
    let (first, second) = tokio::time::timeout(Duration::from_secs(5), calls)
        .await
        .expect("two interleaved calls must not deadlock");

    assert_eq!(
        first.unwrap().message.content,
        text_response("demo", "demo-1", "together").message.content
    );
    assert_eq!(
        second.unwrap().message.content,
        text_response("demo", "demo-1", "together").message.content
    );
}

/// The trait is usable behind `Box<dyn LlmProvider>`, which is what keeps the
/// driver free of provider generics.
#[tokio::test]
async fn a_boxed_provider_answers_through_the_trait_object() {
    let provider: Box<dyn LlmProvider> = Box::new(EchoProvider);
    let response = provider.generate(&request()).await.unwrap();
    assert_eq!(
        response.message.content,
        vec![dsh_llm::ContentBlock::text("echo: hello")]
    );
    assert_eq!(response.finish_reason, dsh_llm::FinishReason::Stop);
}

/// A scripted provider replays in order and reports an exhausted script as a
/// typed failure rather than a panic.
#[tokio::test]
async fn a_scripted_provider_replays_then_reports_exhaustion() {
    let provider = ScriptedProvider::new(vec![text_response("demo", "demo-1", "first")]);
    assert_eq!(provider.remaining(), 1);
    let first = provider.generate(&request()).await.unwrap();
    assert_eq!(
        first.message.content,
        vec![dsh_llm::ContentBlock::text("first")]
    );
    assert_eq!(provider.remaining(), 0);

    let exhausted = provider.generate(&request()).await.unwrap_err();
    assert_eq!(exhausted.code, "SCRIPT_EXHAUSTED");
}

/// A synchronous closure is still enough to build a provider.
#[tokio::test]
async fn a_closure_provider_answers_without_awaiting() {
    let provider = FnProvider::new(|request: &LlmRequest| {
        Ok(text_response(
            &request.config.provider,
            &request.config.model,
            "sync",
        ))
    });
    let response = provider.generate(&request()).await.unwrap();
    assert_eq!(
        response.message.content,
        vec![dsh_llm::ContentBlock::text("sync")]
    );
}
