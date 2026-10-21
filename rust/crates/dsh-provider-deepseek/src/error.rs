//! Mapping DeepSeek HTTP/transport failures into canonical [`LlmFailure`].

#![forbid(unsafe_code)]

use dsh_llm::{LlmFailure, ProviderRequestId};
use serde::Deserialize;

/// Top-level error envelope returned by DeepSeek on a non-2xx response.
#[derive(Debug, Deserialize)]
struct WireErrorBody {
    #[serde(default)]
    error: Option<WireErrorDetail>,
}

/// Detail fields of a DeepSeek error.
#[derive(Debug, Deserialize)]
struct WireErrorDetail {
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

/// Classify an HTTP error into a provider-neutral [`LlmFailure`].
///
/// The DeepSeek `error.code`/`error.type` strings are mapped to stable codes
/// when recognized; otherwise the HTTP status decides. `Retry-After` is read in
/// milliseconds and the provider request id is lifted from `X-Request-Id`.
pub(crate) fn map_http_error(
    status: u16,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> LlmFailure {
    let detail = serde_json::from_str::<WireErrorBody>(body)
        .ok()
        .and_then(|body| body.error);
    let message = detail
        .as_ref()
        .and_then(|error| error.message.clone())
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| body.to_string());
    let code = classify(status, detail.as_ref());
    let provider_retry_after_ms = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|seconds| seconds * 1000);
    let request_id = headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(ProviderRequestId::new);
    LlmFailure {
        message,
        code,
        status: Some(status),
        provider_retry_after_ms,
        request_id,
        offload_images: None,
    }
}

/// Decide the canonical failure code from status plus any parsed detail.
fn classify(status: u16, detail: Option<&WireErrorDetail>) -> String {
    if let Some(detail) = detail {
        if let Some(code) = &detail.code {
            if let Some(mapped) = classify_code(code) {
                return mapped;
            }
        }
        if let Some(kind) = &detail.r#type {
            if !kind.is_empty() {
                return kind.to_ascii_uppercase();
            }
        }
    }
    match status {
        401 => "AUTHENTICATION",
        403 => "FORBIDDEN",
        404 => "NOT_FOUND",
        429 => "RATE_LIMIT",
        s if (500..600).contains(&s) => "SERVER_ERROR",
        400 => "INVALID_REQUEST",
        _ => "PROVIDER_ERROR",
    }
    .to_string()
}

/// Map a DeepSeek machine code to the closest canonical code, or `None`.
fn classify_code(code: &str) -> Option<String> {
    let mapped = match code {
        "invalid_api_key" | "authentication_error" | "auth_error" => "AUTHENTICATION",
        "rate_limit_error" | "rate_limit" => "RATE_LIMIT",
        "invalid_request_error" | "invalid_parameter_error" => "INVALID_REQUEST",
        "server_error" | "api_error" => "SERVER_ERROR",
        other if !other.is_empty() => return Some(other.to_ascii_uppercase()),
        _ => return None,
    };
    Some(mapped.to_string())
}

/// Wrap a transport failure (DNS, TLS, socket, timeout) into [`LlmFailure`].
pub(crate) fn transport_error(error: reqwest::Error) -> LlmFailure {
    LlmFailure {
        message: error.to_string(),
        code: "TRANSPORT".to_string(),
        status: None,
        provider_retry_after_ms: None,
        request_id: None,
        offload_images: None,
    }
}
