//! HTTP and SSE transport for the OpenAI-compatible DeepSeek endpoint.
//!
//! Everything that touches a socket lives here: header assembly, the
//! `POST {base_url}/chat/completions` call, deadline enforcement, and the
//! streaming read loop. Translation stays in [`crate::translate`], which keeps
//! the mapping decisions testable without I/O.
//!
//! Two timeouts are enforced separately, mirroring the TypeScript adapter's
//! distinction between a request deadline and a stream idle watchdog:
//!
//! - a non-streaming call is bounded as a whole by `request_timeout`;
//! - a streaming call is bounded per read by `stream_idle_timeout`, because a
//!   long generation legitimately holds the connection open for minutes while
//!   never going quiet for that long.

#![forbid(unsafe_code)]

use std::time::Duration;

use dsh_provider::LlmResponse;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Client, Response};

use crate::config::DeepSeekConfig;
use crate::error;
use crate::sse::SseFramer;
use crate::translate::{self, StreamAssembler};
use crate::wire::{WireChatCompletion, WireRequest};

/// Path appended to the configured base URL.
const CHAT_COMPLETIONS: &str = "/chat/completions";

/// A configured HTTP client plus the endpoint it talks to.
pub(crate) struct Transport {
    client: Client,
    endpoint: String,
    api_key: String,
    stream_idle_timeout: Duration,
    request_timeout: Duration,
}

impl Transport {
    /// Build the client. Fails only when the TLS backend cannot initialize.
    pub(crate) fn new(config: &DeepSeekConfig) -> Result<Self, dsh_llm::LlmFailure> {
        let client = Client::builder()
            .user_agent(concat!("dsh-provider-deepseek/", env!("CARGO_PKG_VERSION")))
            // No global timeout: it would cap a legitimate long generation.
            // Each call applies its own deadline instead.
            .build()
            .map_err(error::transport_error)?;
        Ok(Self {
            client,
            endpoint: endpoint(&config.base_url),
            api_key: config.api_key.clone(),
            stream_idle_timeout: config.stream_idle_timeout,
            request_timeout: config.request_timeout,
        })
    }

    /// Run one non-streaming request to completion.
    pub(crate) async fn completion(
        &self,
        body: &WireRequest,
        provider: &str,
        model: &str,
    ) -> Result<LlmResponse, dsh_llm::LlmFailure> {
        let response = self.send(body).await?;
        let deadline = self.request_timeout;
        let text = tokio::time::timeout(deadline, response.text())
            .await
            .map_err(|_| {
                error::timeout(format!(
                    "DeepSeek request timed out after {} ms",
                    deadline.as_millis()
                ))
            })?
            .map_err(error::transport_error)?;
        let completion: WireChatCompletion = serde_json::from_str(&text)
            .map_err(|failure| error::malformed(format!("completion is not JSON: {failure}")))?;
        translate::completion_response(completion, provider, model)
    }

    /// Run one streaming request, accumulating every frame to completion.
    pub(crate) async fn stream(
        &self,
        body: &WireRequest,
        provider: &str,
        model: &str,
    ) -> Result<LlmResponse, dsh_llm::LlmFailure> {
        let response = self.send(body).await?;
        let mut stream = response.bytes_stream();
        let mut framer = SseFramer::default();
        let mut assembler = StreamAssembler::new(provider, model);
        let idle = self.stream_idle_timeout;
        loop {
            let next = tokio::time::timeout(idle, stream.next())
                .await
                .map_err(|_| {
                    error::timeout(format!(
                        "DeepSeek stream idle timeout after {} ms",
                        idle.as_millis()
                    ))
                })?;
            let Some(chunk) = next else {
                break;
            };
            let bytes = chunk.map_err(error::transport_error)?;
            for frame in framer.push(&bytes) {
                assembler.push_frame(&frame)?;
                if assembler.is_done() {
                    // `[DONE]` is terminal: anything after it is noise, and a
                    // gateway that never closes would otherwise hang the turn.
                    return assembler.finish();
                }
            }
        }
        assembler.finish()
    }

    /// POST the body and turn a non-2xx response into a classified failure.
    async fn send(&self, body: &WireRequest) -> Result<Response, dsh_llm::LlmFailure> {
        if self.api_key.is_empty() {
            return Err(error::missing_credential());
        }
        let authorization = HeaderValue::from_str(&format!("Bearer {}", self.api_key))
            .map_err(|_| error::invalid_credential())?;
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            ACCEPT,
            HeaderValue::from_static(if body.stream {
                "text/event-stream"
            } else {
                "application/json"
            }),
        );
        headers.insert(AUTHORIZATION, authorization);
        let request = self.client.post(&self.endpoint).headers(headers).json(body);
        let response = tokio::time::timeout(self.request_timeout, request.send())
            .await
            .map_err(|_| {
                error::timeout(format!(
                    "DeepSeek request timed out after {} ms",
                    self.request_timeout.as_millis()
                ))
            })?
            .map_err(error::transport_error)?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        // The error body is small and bounded; a body read failure must not
        // hide the status, so it degrades to an empty body.
        let headers = response.headers().clone();
        let body = response.text().await.unwrap_or_default();
        Err(error::map_http_error(status.as_u16(), &headers, &body))
    }
}

/// The full endpoint URL: the base URL without its trailing slashes, plus the
/// fixed `chat/completions` path.
fn endpoint(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    format!("{trimmed}{CHAT_COMPLETIONS}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_appends_the_path_once() {
        assert_eq!(
            endpoint("https://api.deepseek.com"),
            "https://api.deepseek.com/chat/completions"
        );
        assert_eq!(
            endpoint("https://api.deepseek.com/v1/"),
            "https://api.deepseek.com/v1/chat/completions"
        );
    }
}
