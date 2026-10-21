//! Mapping DeepSeek HTTP, in-band stream, and transport failures onto the
//! canonical [`LlmFailure`].
//!
//! The classification order and the phrase matchers are ported from
//! `packages/llm/llm-deepseek/src/transport.ts` and
//! `packages/llm/llm/src/error.ts`, so a phrase that the TypeScript adapter
//! routes to `CONTEXT_WINDOW_EXCEEDED` or `QUOTA` routes the same way here.
//! Codes stay provider-neutral: retry policy (`dsh-retry`) matches on the code,
//! never on the message.

#![forbid(unsafe_code)]

use std::sync::OnceLock;

use dsh_llm::{LlmFailure, ProviderRequestId};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;

/// Longest raw response body retained in a failure message.
const MAX_BODY_CHARS: usize = 512;

/// Credential rejected: the key was supplied and refused.
pub(crate) const CODE_AUTH: &str = "AUTH";
/// Account quota or balance exhausted — terminal, not a request-rate limit.
pub(crate) const CODE_QUOTA: &str = "QUOTA";
/// Transient request-rate limit.
pub(crate) const CODE_RATE_LIMIT: &str = "RATE_LIMIT";
/// The request exceeded the model's context window.
pub(crate) const CODE_CONTEXT_WINDOW: &str = "CONTEXT_WINDOW_EXCEEDED";
/// The request was rejected as malformed.
pub(crate) const CODE_INVALID_REQUEST: &str = "INVALID_REQUEST";
/// Provider-side failure, including every 5xx.
pub(crate) const CODE_SERVER: &str = "SERVER";
/// Socket, DNS, or TLS failure before a response arrived.
pub(crate) const CODE_TRANSPORT: &str = "TRANSPORT";
/// A deadline elapsed.
pub(crate) const CODE_TIMEOUT: &str = "TIMEOUT";
/// The provider's response could not be decoded.
pub(crate) const CODE_MALFORMED: &str = "MALFORMED_RESPONSE";
/// The stream ended before the terminal event.
pub(crate) const CODE_STREAM_CLOSED: &str = "STREAM_CLOSED";
/// A content block cannot be expressed on this wire format.
pub(crate) const CODE_UNSUPPORTED: &str = "UNSUPPORTED_CONTENT";
/// The key was supplied but is not a legal header value.
pub(crate) const CODE_INVALID_CREDENTIAL: &str = "INVALID_CREDENTIAL";
/// No key was configured at all.
pub(crate) const CODE_MISSING_CREDENTIAL: &str = "MISSING_CREDENTIAL";

/// Top-level error envelope returned by DeepSeek on a non-2xx response and as
/// an in-band `data: {"error": ...}` frame.
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

impl WireErrorDetail {
    /// The `code`, `type`, and `message` joined into one string, which is what
    /// the phrase matchers scan (mirrors `providerErrorDetail`).
    fn detail_text(&self) -> String {
        [&self.code, &self.r#type, &self.message]
            .into_iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn kind(&self) -> &str {
        self.r#type.as_deref().unwrap_or_default()
    }
}

/// Build a failure with no status, retry hint, or request id.
pub(crate) fn failure(message: impl Into<String>, code: &str) -> LlmFailure {
    LlmFailure {
        message: message.into(),
        code: code.to_string(),
        status: None,
        provider_retry_after_ms: None,
        request_id: None,
        offload_images: None,
    }
}

/// The provider's payload could not be decoded.
pub(crate) fn malformed(detail: impl AsRef<str>) -> LlmFailure {
    failure(
        format!("DeepSeek response: {}", detail.as_ref()),
        CODE_MALFORMED,
    )
}

/// A completed response carried no content block at all.
///
/// Classified rather than returned, because an empty assistant message silently
/// ends the turn with nothing for the user or the loop to act on. The code is
/// part of the default retryable set.
pub(crate) fn empty_response() -> LlmFailure {
    failure("DeepSeek returned no content", dsh_llm::EMPTY_RESPONSE_CODE)
}

/// The stream ended without a terminal event and without a finish reason.
pub(crate) fn stream_closed() -> LlmFailure {
    failure(
        "DeepSeek stream ended before the terminal frame",
        CODE_STREAM_CLOSED,
    )
}

/// A deadline elapsed. `TIMEOUT` is in the default retryable set; `TRANSPORT`
/// is not the right code for a locally enforced deadline.
pub(crate) fn timeout(message: impl Into<String>) -> LlmFailure {
    failure(message, CODE_TIMEOUT)
}

/// A content block has no representation on this wire format.
pub(crate) fn unsupported(detail: impl AsRef<str>) -> LlmFailure {
    failure(
        format!(
            "DeepSeek chat/completions cannot represent {}",
            detail.as_ref()
        ),
        CODE_UNSUPPORTED,
    )
}

/// The canonical history cannot be expressed without changing its meaning.
pub(crate) fn invalid_request(detail: impl AsRef<str>) -> LlmFailure {
    failure(
        format!("DeepSeek chat/completions request: {}", detail.as_ref()),
        CODE_INVALID_REQUEST,
    )
}

/// No API key was configured.
pub(crate) fn missing_credential() -> LlmFailure {
    failure(
        "DeepSeek provider has no API key configured",
        CODE_MISSING_CREDENTIAL,
    )
}

/// The configured key cannot be sent as an `Authorization` header.
pub(crate) fn invalid_credential() -> LlmFailure {
    failure(
        "DeepSeek API key contains characters that are not legal in an HTTP header",
        CODE_INVALID_CREDENTIAL,
    )
}

/// Wrap a transport failure (DNS, TLS, socket, decode) into [`LlmFailure`].
pub(crate) fn transport_error(error: reqwest::Error) -> LlmFailure {
    let code = if error.is_timeout() {
        CODE_TIMEOUT
    } else {
        CODE_TRANSPORT
    };
    // `reqwest::Error`'s Display never includes request headers, so the bearer
    // token cannot leak through this message.
    failure(error.to_string(), code)
}

/// Classify a non-2xx HTTP response.
pub(crate) fn map_http_error(
    status: u16,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> LlmFailure {
    let detail = serde_json::from_str::<WireErrorBody>(body)
        .ok()
        .and_then(|body| body.error);
    build(status.into(), detail.as_ref(), headers, || truncate(body))
}

/// Classify an in-band `data: {"error": ...}` frame, which carries no status.
pub(crate) fn map_in_band_error(payload: &Value) -> LlmFailure {
    let detail = payload
        .get("error")
        .cloned()
        .and_then(|error| serde_json::from_value::<WireErrorDetail>(error).ok());
    let fallback = detail
        .as_ref()
        .and_then(|detail| detail.message.clone())
        .unwrap_or_else(|| "DeepSeek reported a stream error".to_string());
    build(
        None,
        detail.as_ref(),
        &reqwest::header::HeaderMap::new(),
        || fallback,
    )
}

/// Assemble a failure from a status, a parsed detail, and any headers.
fn build(
    status: Option<u16>,
    detail: Option<&WireErrorDetail>,
    headers: &reqwest::header::HeaderMap,
    fallback_message: impl FnOnce() -> String,
) -> LlmFailure {
    let message = detail
        .and_then(|detail| detail.message.clone())
        .filter(|message| !message.is_empty())
        .unwrap_or_else(fallback_message);
    LlmFailure {
        message,
        code: classify(status, detail),
        status,
        provider_retry_after_ms: retry_after_ms(headers),
        request_id: request_id(headers),
        offload_images: None,
    }
}

/// Decide the canonical failure code. Order matters: a 429 that also mentions
/// an exhausted balance is a terminal `QUOTA`, not a retryable `RATE_LIMIT`.
fn classify(status: Option<u16>, detail: Option<&WireErrorDetail>) -> String {
    let detail_text = detail.map(WireErrorDetail::detail_text).unwrap_or_default();
    let kind = detail.map(WireErrorDetail::kind).unwrap_or_default();
    if status == Some(401)
        || status == Some(403)
        || matches!(kind, "authentication_error" | "permission_error")
    {
        return CODE_AUTH.to_string();
    }
    if is_quota_exceeded(&detail_text) || status == Some(402) {
        return CODE_QUOTA.to_string();
    }
    if status == Some(429) || kind == "rate_limit_error" {
        return CODE_RATE_LIMIT.to_string();
    }
    if is_context_window_exceeded(&detail_text) {
        return CODE_CONTEXT_WINDOW.to_string();
    }
    if matches!(status, Some(400 | 413)) || kind == "invalid_request_error" {
        return CODE_INVALID_REQUEST.to_string();
    }
    if status.is_some_and(|status| status >= 500)
        || matches!(kind, "api_error" | "overloaded_error")
    {
        return CODE_SERVER.to_string();
    }
    match status {
        None => CODE_SERVER.to_string(),
        Some(status) => format!("HTTP_{status}"),
    }
}

/// Read `Retry-After` as milliseconds.
///
/// Only the delay-seconds form is understood, as integer or decimal seconds;
/// the HTTP-date form is not parsed, because nothing in the harness needs it
/// and a mis-parsed date would schedule a retry at an arbitrary time. A value
/// that is absent, zero, negative, or unparsable yields `None`.
fn retry_after_ms(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    let seconds: f64 = raw.trim().parse().ok()?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    Some((seconds * 1000.0).round() as u64)
}

/// Lift the provider request id from the first header that carries one.
fn request_id(headers: &reqwest::header::HeaderMap) -> Option<ProviderRequestId> {
    ["request-id", "x-request-id", "x-deepseek-request-id"]
        .into_iter()
        .find_map(|name| headers.get(name))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(ProviderRequestId::new)
}

/// Keep a non-JSON error body (a gateway's HTML page, a proxy's plain text)
/// short enough to be a diagnostic rather than a payload.
fn truncate(body: &str) -> String {
    let body = body.trim();
    if body.is_empty() {
        return "DeepSeek request failed".to_string();
    }
    match body.char_indices().nth(MAX_BODY_CHARS) {
        Some((end, _)) => format!("{}…", &body[..end]),
        None => body.to_string(),
    }
}

/// Phrases naming a context bound being exceeded (`isContextWindowExceededError`).
fn context_window_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        [
            r"(?:^|[^a-z0-9])context[\s_-](?:length|window)[\s_-](?:exceed(?:ed|s)?|overflow(?:ed)?|limit[\s_-]exceeded)(?:$|[^a-z0-9])",
            r"\b(?:maximum|max)(?:\s+(?:allowed|supported))?\s+context\s+(?:length|window)\b",
            r"\b(?:request|prompt|input|messages?)\s+(?:is\s+|are\s+)?too\s+(?:large|long)\s+for\s+(?:(?:this|the)\s+)?(?:model(?:'s)?\s+)?context(?:\s+window)?\b",
            r"\b(?:input|prompt|request)\s+(?:is\s+)?too\s+(?:long|large)\s+for\s+(?:this|the)\s+model\b",
            r"\b(?:input|prompt|request|messages?)\b.{0,40}\b(?:exceed(?:s|ed)?|overflows?|is\s+larger\s+than)\b.{0,40}\b(?:the\s+)?(?:model(?:'s)?\s+)?context(?:\s+(?:length|window))?\b",
        ]
        .into_iter()
        .map(|pattern| {
            let pattern = format!("(?i){pattern}");
            Regex::new(&pattern).expect("static context-window pattern compiles")
        })
        .collect()
    })
}

/// Phrases naming an exhausted quota rather than a transient rate limit
/// (`isQuotaExceededError`).
fn quota_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        [
            r"\binsufficient[\s_-]+(?:quota|balance|credits?)\b",
            r"\b(?:quota|usage[\s_-]+limit)[\s_-]+(?:exceeded|exhausted|reached)\b",
            r"\bexceed(?:ed|s)?[\s_-]+(?:(?:your|the)[\s_-]+)?(?:current[\s_-]+)?quota\b",
            r"\b(?:balance|credits?)[\s_-]+(?:exhausted|depleted)\b",
            r"\bout[\s_-]+of[\s_-]+(?:credits?|budget)\b",
        ]
        .into_iter()
        .map(|pattern| {
            let pattern = format!("(?i){pattern}");
            Regex::new(&pattern).expect("static quota pattern compiles")
        })
        .collect()
    })
}

/// Whether the provider named a context window being exceeded.
pub(crate) fn is_context_window_exceeded(detail: &str) -> bool {
    context_window_patterns()
        .iter()
        .any(|pattern| pattern.is_match(detail))
}

/// Whether the provider named an exhausted quota or balance.
pub(crate) fn is_quota_exceeded(detail: &str) -> bool {
    quota_patterns()
        .iter()
        .any(|pattern| pattern.is_match(detail))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                *name,
                HeaderValue::from_str(value).expect("legal header value"),
            );
        }
        map
    }

    #[test]
    fn status_codes_map_onto_the_canonical_vocabulary() {
        let empty = HeaderMap::new();
        let cases = [
            (401, CODE_AUTH),
            (403, CODE_AUTH),
            (404, "HTTP_404"),
            (500, CODE_SERVER),
        ];
        for (status, expected) in cases {
            assert_eq!(
                map_http_error(status, &empty, "").code,
                expected,
                "{status}"
            );
        }
    }

    #[test]
    fn quota_wording_outranks_a_rate_limit_status() {
        let failure = map_http_error(
            429,
            &HeaderMap::new(),
            r#"{"error":{"message":"Insufficient balance","type":"rate_limit_error"}}"#,
        );
        assert_eq!(failure.code, CODE_QUOTA);
        assert_eq!(failure.status, Some(429));
        assert_eq!(failure.message, "Insufficient balance");
    }

    #[test]
    fn context_window_wording_is_recognized() {
        for message in [
            "This model's maximum context length is 65536 tokens",
            "the request is too large for this model context window",
            "input exceeds the model's context length",
        ] {
            let body =
                format!(r#"{{"error":{{"message":"{message}","type":"invalid_request_error"}}}}"#);
            assert_eq!(
                map_http_error(400, &HeaderMap::new(), &body).code,
                CODE_CONTEXT_WINDOW,
                "{message}"
            );
        }
    }

    #[test]
    fn retry_after_is_read_in_milliseconds() {
        let failure = map_http_error(429, &headers(&[("retry-after", "2.5")]), "");
        assert_eq!(failure.provider_retry_after_ms, Some(2500));
        let absent = map_http_error(429, &headers(&[("retry-after", "0")]), "");
        assert_eq!(absent.provider_retry_after_ms, None);
        let invalid = map_http_error(429, &headers(&[("retry-after", "soon")]), "");
        assert_eq!(invalid.provider_retry_after_ms, None);
    }

    #[test]
    fn request_id_comes_from_the_first_header_that_has_one() {
        let failure = map_http_error(
            500,
            &headers(&[
                ("x-deepseek-request-id", "req-2"),
                ("x-request-id", "req-1"),
            ]),
            "",
        );
        assert_eq!(
            failure.request_id.as_ref().map(ProviderRequestId::as_str),
            Some("req-1")
        );
    }

    #[test]
    fn in_band_error_frames_classify_without_a_status() {
        let payload: Value = serde_json::from_str(
            r#"{"error":{"message":"Authentication Fails","type":"authentication_error"}}"#,
        )
        .expect("valid json");
        let failure = map_in_band_error(&payload);
        assert_eq!(failure.code, CODE_AUTH);
        assert_eq!(failure.status, None);
    }

    #[test]
    fn non_json_bodies_become_a_truncated_message() {
        let body = "x".repeat(MAX_BODY_CHARS + 10);
        let failure = map_http_error(502, &HeaderMap::new(), &body);
        assert_eq!(failure.code, CODE_SERVER);
        assert!(failure.message.ends_with('…'));
        assert_eq!(failure.message.chars().count(), MAX_BODY_CHARS + 1);
    }
}
