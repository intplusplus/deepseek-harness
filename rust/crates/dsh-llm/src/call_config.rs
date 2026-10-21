//! Conversation call configuration.
//!
//! Provider routing, model, reasoning effort, and sampling values are
//! request-header state that can affect cache reuse; request waterfalls
//! replace them and the loop logs changed snapshots instead of allowing silent
//! per-call drift.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Provider, model, reasoning effort, and sampling scalars of one
/// conversation's requests.
///
/// Every field maps 1:1 onto the same-named `GenerateOptions` field; the loop
/// builds requests from the logged header rather than accepting these per call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmCallConfig {
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// Effective config fields supplied by exact-model adapter resolution rather
/// than by the caller's request proposal.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmCallConfigAdapterDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<bool>,
}

impl LlmCallConfigAdapterDefaults {
    /// Canonical form omits an all-empty snapshot; the session log rejects one
    /// that is present but empty.
    pub fn is_empty(&self) -> bool {
        self.reasoning_effort.is_none() && self.max_tokens.is_none()
    }
}

/// Field-wise equality over [`LlmCallConfig`] — the comparison a caller runs to
/// decide whether a proposed configuration is a real change (worth a logged
/// header snapshot) or the held one restated.
pub fn call_config_equals(a: &LlmCallConfig, b: &LlmCallConfig) -> bool {
    a == b
}
