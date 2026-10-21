//! Provider-owned retry policy configuration and resolution.
//!
//! Ported from `packages/llm/llm/src/retry-policy.ts`. The TypeScript version
//! validates plain objects with schemastery; the Rust version validates
//! already-deserialized structs and uses `#[serde(deny_unknown_fields)]` for
//! the "reject unknown keys" half, so an unknown key is a deserialization
//! error rather than a runtime check.
//!
//! A resolved policy is **runtime-only**: it is captured per provider route
//! and never reaches the session log. What reaches the log is
//! [`retry_policy_key`], a stable string over the whole policy — change any
//! field and the key changes, which starts a fresh retry chain.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use dsh_llm::EMPTY_RESPONSE_CODE;

use super::json;

/// Largest delay a timer represents without clamping (`dsh-timeout`).
pub const MAX_TIMER_DELAY_MS: f64 = 2_147_483_647.0;

/// Default maximum eligible retries after the first request.
pub const DEFAULT_MAX_RETRIES: u64 = 5;
/// Default initial local exponential-backoff delay in milliseconds.
pub const DEFAULT_INITIAL_DELAY_MS: f64 = 500.0;
/// Default maximum locally scheduled or accepted provider delay.
pub const DEFAULT_MAX_DELAY_MS: f64 = 10_000.0;
/// Default symmetric random multiplier range around one.
pub const DEFAULT_JITTER_RATIO: f64 = 0.1;
/// Default transient failure codes eligible for a normal policy.
pub const DEFAULT_RETRYABLE_CODES: &[&str] = &[
    EMPTY_RESPONSE_CODE,
    "RATE_LIMIT",
    "SERVER",
    "TIMEOUT",
    "TRANSPORT",
];

/// Largest integer exactly representable as a JavaScript number.
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// One retry-policy configuration error.
///
/// The message reproduces the TypeScript diagnostic, path prefix included.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct PolicyError(String);

/// Which retry behaviour a provider route opted into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RetryMode {
    /// Retry only configured transient failure codes, within a budget.
    Normal,
    /// Retry every model-request failure until success or cancellation.
    Always,
}

/// Bounded exponential backoff with symmetric jitter around each local delay.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct BackoffConfig {
    /// Initial local exponential-backoff delay in milliseconds (default 500).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_delay_ms: Option<f64>,
    /// Maximum locally scheduled or accepted provider delay (default 10000).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_delay_ms: Option<f64>,
    /// Symmetric random multiplier range around one (default 0.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jitter_ratio: Option<f64>,
}

/// Bounded transient retry behaviour for one provider route.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct NormalRetryPolicyConfig {
    /// Maximum eligible retries after the first request (default 5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u64>,
    /// Stable failure codes eligible for this policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retryable_codes: Option<Vec<String>>,
    /// Local exponential-backoff and jitter configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff: Option<BackoffConfig>,
}

/// Unbounded retry behaviour for one provider route.
///
/// `maxRetries` and `retryableCodes` are accepted and ignored: layered
/// configuration can retain normal-only fields after a mode switch, and the
/// TypeScript schema keeps accepting them while still rejecting unknown keys.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct AlwaysRetryPolicyConfig {
    /// Local exponential-backoff and jitter configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff: Option<BackoffConfig>,
    /// Inactive in always mode; retained so a layered mode switch still parses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u64>,
    /// Inactive in always mode; retained so a layered mode switch still parses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retryable_codes: Option<Vec<String>>,
}

/// Provider-owned model-request retry policy configuration.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum RetryPolicyConfig {
    /// Retry only configured transient failure codes.
    Normal(NormalRetryPolicyConfig),
    /// Retry every model-request failure.
    Always(AlwaysRetryPolicyConfig),
}

/// Fully resolved backoff shared by both retry modes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedRetryBackoff {
    /// Initial local exponential-backoff delay in milliseconds.
    #[serde(serialize_with = "json::serialize_number")]
    pub initial_delay_ms: f64,
    /// Maximum locally scheduled or accepted provider delay in milliseconds.
    #[serde(serialize_with = "json::serialize_number")]
    pub max_delay_ms: f64,
    /// Symmetric random multiplier range around one.
    pub jitter_ratio: f64,
}

/// Immutable provider policy captured when its adapter route is registered.
///
/// The three backoff fields are inlined into each variant rather than
/// flattened, which keeps the serialized shape identical to the TypeScript
/// object (`{ mode, maxRetries, retryableCodes, initialDelayMs, ... }`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum ResolvedRetryPolicy {
    /// Retry only the listed codes, at most `max_retries` times.
    Normal {
        /// Maximum eligible retries after the first request.
        #[serde(rename = "maxRetries")]
        max_retries: u64,
        /// Stable failure codes eligible for this policy.
        #[serde(rename = "retryableCodes")]
        retryable_codes: Vec<String>,
        /// Initial local exponential-backoff delay in milliseconds.
        #[serde(rename = "initialDelayMs", serialize_with = "json::serialize_number")]
        initial_delay_ms: f64,
        /// Maximum locally scheduled or accepted provider delay.
        #[serde(rename = "maxDelayMs", serialize_with = "json::serialize_number")]
        max_delay_ms: f64,
        /// Symmetric random multiplier range around one.
        #[serde(rename = "jitterRatio")]
        jitter_ratio: f64,
    },
    /// Retry every failure, without a budget.
    Always {
        /// Initial local exponential-backoff delay in milliseconds.
        #[serde(rename = "initialDelayMs", serialize_with = "json::serialize_number")]
        initial_delay_ms: f64,
        /// Maximum locally scheduled or accepted provider delay.
        #[serde(rename = "maxDelayMs", serialize_with = "json::serialize_number")]
        max_delay_ms: f64,
        /// Symmetric random multiplier range around one.
        #[serde(rename = "jitterRatio")]
        jitter_ratio: f64,
    },
}

impl ResolvedRetryPolicy {
    /// Which mode this policy is.
    pub fn mode(&self) -> RetryMode {
        match self {
            Self::Normal { .. } => RetryMode::Normal,
            Self::Always { .. } => RetryMode::Always,
        }
    }

    /// The backoff shared by both modes.
    pub fn backoff(&self) -> ResolvedRetryBackoff {
        match *self {
            Self::Normal {
                initial_delay_ms,
                max_delay_ms,
                jitter_ratio,
                ..
            }
            | Self::Always {
                initial_delay_ms,
                max_delay_ms,
                jitter_ratio,
            } => ResolvedRetryBackoff {
                initial_delay_ms,
                max_delay_ms,
                jitter_ratio,
            },
        }
    }

    /// The retry budget, which only normal mode has.
    pub fn max_retries(&self) -> Option<u64> {
        match self {
            Self::Normal { max_retries, .. } => Some(*max_retries),
            Self::Always { .. } => None,
        }
    }

    /// The eligible failure codes, which only normal mode has.
    pub fn retryable_codes(&self) -> Option<&[String]> {
        match self {
            Self::Normal {
                retryable_codes, ..
            } => Some(retryable_codes),
            Self::Always { .. } => None,
        }
    }

    /// Whether `code` is eligible under this policy. Always mode accepts every
    /// code; a normal policy without a match is not recoverable here and must
    /// be delegated.
    pub fn covers(&self, code: &str) -> bool {
        match self {
            Self::Normal {
                retryable_codes, ..
            } => retryable_codes.iter().any(|c| c == code),
            Self::Always { .. } => true,
        }
    }
}

/// Validate, default, and detach one provider-owned retry policy.
///
/// `path` names the provider config that owns the value, exactly as the
/// TypeScript resolver does, so a diagnostic identifies its source.
pub fn resolve_retry_policy(
    config: Option<&RetryPolicyConfig>,
    path: &str,
) -> Result<ResolvedRetryPolicy, PolicyError> {
    let Some(config) = config else {
        let backoff = resolve_backoff(None, &format!("{path}.backoff"))?;
        return Ok(ResolvedRetryPolicy::Normal {
            max_retries: DEFAULT_MAX_RETRIES,
            retryable_codes: DEFAULT_RETRYABLE_CODES
                .iter()
                .map(|c| (*c).to_string())
                .collect(),
            initial_delay_ms: backoff.initial_delay_ms,
            max_delay_ms: backoff.max_delay_ms,
            jitter_ratio: backoff.jitter_ratio,
        });
    };

    match config {
        RetryPolicyConfig::Normal(config) => {
            let max_retries = config.max_retries.unwrap_or(DEFAULT_MAX_RETRIES);
            let retryable_codes = config.retryable_codes.clone().unwrap_or_else(|| {
                DEFAULT_RETRYABLE_CODES
                    .iter()
                    .map(|c| (*c).to_string())
                    .collect()
            });

            if max_retries > MAX_SAFE_INTEGER {
                return Err(PolicyError(format!(
                    "{path}.maxRetries must be a non-negative safe integer"
                )));
            }
            if retryable_codes.is_empty() {
                return Err(PolicyError(format!(
                    "{path}.retryableCodes must not be empty"
                )));
            }
            if retryable_codes.iter().any(|code| code.is_empty()) {
                return Err(PolicyError(format!(
                    "{path}.retryableCodes must contain only non-empty strings"
                )));
            }
            let mut seen: Vec<&str> = Vec::with_capacity(retryable_codes.len());
            for code in &retryable_codes {
                if seen.contains(&code.as_str()) {
                    return Err(PolicyError(format!(
                        "{path}.retryableCodes must not contain duplicates"
                    )));
                }
                seen.push(code.as_str());
            }

            let backoff = resolve_backoff(config.backoff.as_ref(), &format!("{path}.backoff"))?;
            Ok(ResolvedRetryPolicy::Normal {
                max_retries,
                retryable_codes,
                initial_delay_ms: backoff.initial_delay_ms,
                max_delay_ms: backoff.max_delay_ms,
                jitter_ratio: backoff.jitter_ratio,
            })
        }
        RetryPolicyConfig::Always(config) => {
            let backoff = resolve_backoff(config.backoff.as_ref(), &format!("{path}.backoff"))?;
            Ok(ResolvedRetryPolicy::Always {
                initial_delay_ms: backoff.initial_delay_ms,
                max_delay_ms: backoff.max_delay_ms,
                jitter_ratio: backoff.jitter_ratio,
            })
        }
    }
}

fn resolve_backoff(
    config: Option<&BackoffConfig>,
    path: &str,
) -> Result<ResolvedRetryBackoff, PolicyError> {
    let initial_delay_ms = config
        .and_then(|c| c.initial_delay_ms)
        .unwrap_or(DEFAULT_INITIAL_DELAY_MS);
    let max_delay_ms = config
        .and_then(|c| c.max_delay_ms)
        .unwrap_or(DEFAULT_MAX_DELAY_MS);
    let jitter_ratio = config
        .and_then(|c| c.jitter_ratio)
        .unwrap_or(DEFAULT_JITTER_RATIO);

    if !initial_delay_ms.is_finite()
        || initial_delay_ms <= 0.0
        || initial_delay_ms > MAX_TIMER_DELAY_MS
    {
        return Err(PolicyError(format!(
            "{path}.initialDelayMs must be a positive finite number no greater than {}",
            MAX_TIMER_DELAY_MS as u64
        )));
    }
    if !max_delay_ms.is_finite() || max_delay_ms <= 0.0 || max_delay_ms > MAX_TIMER_DELAY_MS {
        return Err(PolicyError(format!(
            "{path}.maxDelayMs must be a positive finite number no greater than {}",
            MAX_TIMER_DELAY_MS as u64
        )));
    }
    if initial_delay_ms > max_delay_ms {
        return Err(PolicyError(format!(
            "{path}.initialDelayMs must be less than or equal to maxDelayMs"
        )));
    }
    if !jitter_ratio.is_finite() || !(0.0..=1.0).contains(&jitter_ratio) {
        return Err(PolicyError(format!(
            "{path}.jitterRatio must be between 0 and 1"
        )));
    }

    Ok(ResolvedRetryBackoff {
        initial_delay_ms,
        max_delay_ms,
        jitter_ratio,
    })
}

/// The stable identity of one policy's *shape*.
///
/// Two policies with the same key share a retry chain, so editing a running
/// provider's policy starts a fresh budget instead of inheriting the old one.
/// Codes are sorted, so reordering them is not a policy change.
pub fn retry_policy_key(policy: &ResolvedRetryPolicy) -> String {
    let backoff = policy.backoff();
    let mode = match policy.mode() {
        RetryMode::Normal => "normal",
        RetryMode::Always => "always",
    };
    let mut parts: Vec<serde_json::Value> = vec![serde_json::Value::String(mode.to_string())];
    if let Some(max_retries) = policy.max_retries() {
        parts.push(json::number(max_retries as f64));
        let mut codes: Vec<&String> = policy
            .retryable_codes()
            .unwrap_or_default()
            .iter()
            .collect();
        codes.sort();
        parts.push(serde_json::Value::Array(
            codes
                .iter()
                .map(|code| serde_json::Value::String((*code).clone()))
                .collect(),
        ));
    }
    parts.push(json::number(backoff.initial_delay_ms));
    parts.push(json::number(backoff.max_delay_ms));
    parts.push(json::number(backoff.jitter_ratio));
    serde_json::to_string(&serde_json::Value::Array(parts)).unwrap_or_default()
}
