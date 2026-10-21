//! Configuration and the provider entry point.

#![forbid(unsafe_code)]

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use dsh_llm::LlmFailure;
use dsh_provider::{LlmProvider, LlmRequest, LlmResponse};

use crate::translate;
use crate::transport::Transport;

/// OpenAI-compatible endpoint root. Requests go to
/// `{base_url}/chat/completions`, so this deliberately ends at `/v1`, not at
/// the Messages root the TypeScript adapter uses.
pub const DEFAULT_BASE_URL: &str = "https://api.deepseek.com/v1";

/// Deadline for one non-streaming request, and for response headers of any
/// request. Streaming bodies are bounded per read instead.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Maximum silence tolerated between stream reads. The TypeScript adapter uses
/// the same five-minute default (`DEFAULT_STREAM_IDLE_TIMEOUT_MS`).
pub const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// DeepSeek provider configuration.
#[derive(Clone)]
pub struct DeepSeekConfig {
    /// API key, sent as `Authorization: Bearer`. Never logged.
    pub api_key: String,
    /// Endpoint root; see [`DEFAULT_BASE_URL`].
    pub base_url: String,
    /// Whether to call the streaming endpoint.
    ///
    /// Streaming is the default and the only mode that reports token usage
    /// from DeepSeek, because a usage tail chunk is requested through
    /// `stream_options`. The non-streaming path exists for endpoints that
    /// reject Server-Sent Events.
    pub stream: bool,
    /// Whole-call deadline for a non-streaming request.
    pub request_timeout: Duration,
    /// Per-read silence bound for a streaming request.
    pub stream_idle_timeout: Duration,
}

impl DeepSeekConfig {
    /// Defaults for `api_key`: streaming, the public endpoint, and the standard
    /// deadlines.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            stream: true,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            stream_idle_timeout: DEFAULT_STREAM_IDLE_TIMEOUT,
        }
    }

    /// Override the endpoint root.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Choose the streaming or non-streaming endpoint.
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Override the non-streaming deadline.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Override the streaming idle bound.
    pub fn with_stream_idle_timeout(mut self, timeout: Duration) -> Self {
        self.stream_idle_timeout = timeout;
        self
    }
}

/// Hand-written so the key can never reach a log line, a panic message, or a
/// derived snapshot: a derived `Debug` would print it verbatim.
impl fmt::Debug for DeepSeekConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeepSeekConfig")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("stream", &self.stream)
            .field("request_timeout", &self.request_timeout)
            .field("stream_idle_timeout", &self.stream_idle_timeout)
            .finish()
    }
}

/// The provider: one configured [`Transport`] behind the canonical trait.
pub struct DeepSeekProvider {
    config: DeepSeekConfig,
    transport: Transport,
}

impl DeepSeekProvider {
    /// Build a provider from configuration.
    ///
    /// Fails only when the HTTP client cannot be initialized (an unusable TLS
    /// backend); transport failures are reported per call.
    pub fn new(config: DeepSeekConfig) -> Result<Self, LlmFailure> {
        let transport = Transport::new(&config)?;
        Ok(Self { config, transport })
    }

    /// The configuration this provider was built from, with the key redacted
    /// by its `Debug` implementation.
    pub fn config(&self) -> &DeepSeekConfig {
        &self.config
    }
}

#[async_trait]
impl LlmProvider for DeepSeekProvider {
    async fn generate(&self, request: &LlmRequest) -> Result<LlmResponse, LlmFailure> {
        let body = translate::build_request(request, self.config.stream)?;
        let provider = request.config.provider.as_str();
        let model = request.config.model.as_str();
        if self.config.stream {
            self.transport.stream(&body, provider, model).await
        } else {
            self.transport.completion(&body, provider, model).await
        }
    }
}
