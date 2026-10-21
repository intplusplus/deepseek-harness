//! Configuration and the provider entry point.

use async_trait::async_trait;
use dsh_llm::LlmFailure;
use dsh_provider::{LlmProvider, LlmRequest, LlmResponse};

/// DeepSeek provider configuration.
#[derive(Clone, Debug)]
pub struct DeepSeekConfig {
    /// API key (never logged).
    pub api_key: String,
    /// Base URL; defaults to the public endpoint.
    pub base_url: String,
}

/// The provider. Skeleton: returns a structured failure until the transport lands.
pub struct DeepSeekProvider {
    #[allow(dead_code)]
    config: DeepSeekConfig,
}

impl DeepSeekProvider {
    /// Build a provider from configuration.
    pub fn new(config: DeepSeekConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl LlmProvider for DeepSeekProvider {
    async fn generate(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmFailure> {
        Err(LlmFailure {
            message: "deepseek transport not implemented yet".to_string(),
            code: "NOT_IMPLEMENTED".to_string(),
            status: None,
            provider_retry_after_ms: None,
            request_id: None,
            offload_images: None,
        })
    }
}
