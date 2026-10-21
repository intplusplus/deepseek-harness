//! Failure facts, finish reasons, token accounting, and tool schemas.

#![forbid(unsafe_code)]

use std::ops::Not;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::brand::ProviderRequestId;

/// The failure code adapters assign to a completed response that carried no
/// content at all (`packages/llm/llm/src/error.ts`).
///
/// It is part of the default retryable set, so a policy that overrides
/// `retryableCodes` without listing it stops recovering from empty
/// completions.
pub const EMPTY_RESPONSE_CODE: &str = "EMPTY_RESPONSE";

/// Serializable provider or transport failure facts; policy decides whether
/// they are retryable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmFailure {
    /// Human-readable provider or transport failure.
    pub message: String,
    /// Stable provider-neutral machine-routing code.
    pub code: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_retry_after_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<ProviderRequestId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offload_images: Option<u32>,
}

impl std::fmt::Display for LlmFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} [{}]", self.message, self.code)
    }
}

impl std::error::Error for LlmFailure {}

/// Why a model response stopped. Merge-extensible so adapters can surface
/// provider-specific reasons; consumers fall through unknown kinds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    MaxTokens,
    /// Boxed: `LlmFailure` is ~104 bytes and would otherwise inflate every
    /// `StreamChunk` that merely carries a `Stop`.
    Aborted {
        failure: Box<LlmFailure>,
    },
    Error {
        failure: Box<LlmFailure>,
    },
}

impl FinishReason {
    /// Finish because the call was aborted with `failure`.
    pub fn aborted(failure: LlmFailure) -> Self {
        Self::Aborted {
            failure: Box::new(failure),
        }
    }

    /// Finish because the call failed with `failure`.
    pub fn error(failure: LlmFailure) -> Self {
        Self::Error {
            failure: Box::new(failure),
        }
    }

    /// The structured failure, when this reason carries one.
    pub fn failure(&self) -> Option<&LlmFailure> {
        match self {
            Self::Aborted { failure } | Self::Error { failure } => Some(failure),
            _ => None,
        }
    }
}

/// Token accounting for one model call (cache fields are optional).
///
/// Counts are DISJOINT: `input_tokens` is uncached input only; cached input is
/// reported separately as `cache_read_tokens` / `cache_write_tokens` (billed
/// input = sum of the three).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

/// JSON-schema description of a tool, as sent to the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSchema {
    /// Requests deferred loading of the tool definition into model context;
    /// serialized only when true.
    #[serde(default, skip_serializing_if = "Not::not")]
    pub defer_loading: bool,
    pub name: String,
    pub description: String,
    /// JSON Schema object for the arguments.
    pub parameters: Map<String, Value>,
}

impl ToolSchema {
    /// A tool schema with no deferred loading.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        let parameters = parameters.as_object().cloned().unwrap_or_default();
        Self {
            defer_loading: false,
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Merge-extensible provider model modality vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelModality {
    Text,
    Image,
}
