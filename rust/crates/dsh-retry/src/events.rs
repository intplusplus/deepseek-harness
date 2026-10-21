//! The durable retry events.
//!
//! TypeScript declares these by merging into `SessionEventMap`:
//!
//! ```ts
//! declare module '@deepseek-ai/dsh-session/types' {
//!   interface SessionEventMap {
//!     'llm/retry': LlmRetryEventData
//!     'llm/retry-started': LlmRetryStartedEventData
//!   }
//! }
//! ```
//!
//! Rust has no declaration merging, so they live here as ordinary types and
//! reach the log through [`SessionEvent::Extension`]. [`encode_retry`] and
//! [`decode_retry`] are the bridge, and they keep the core invariant of the
//! extension slot: a known tag is **never** silently degraded to an opaque
//! extension payload. [`decode_retry`] returns `None` only for a different
//! tag; a `llm/retry` event whose payload does not match the declared shape
//! comes back as `Some(Err(..))`.
//!
//! Both events are written *around* the wait, not after it: `llm/retry` is
//! durable before the cancellable delay begins, so a crash mid-wait still
//! shows that a retry was scheduled.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize, Serializer};

use dsh_llm::LlmFailure;
use dsh_session::{ExtensionEvent, SessionEvent};

use super::brand::RetryId;
use super::json;
use super::policy::RetryMode;

/// Tag of the durable "one retry scheduled" event.
pub const LLM_RETRY_TAG: &str = "llm/retry";
/// Tag of the durable "the retry wait finished" transition.
pub const LLM_RETRY_STARTED_TAG: &str = "llm/retry-started";

/// One provider-routed retry scheduled after a failed request attempt.
///
/// The `normal` variant carries its budget; the `always` variant has none, so
/// a payload can never claim a budget for a policy that does not have one.
/// The `failure` is boxed: the log is append-only and unbounded, and this is
/// the largest field by a wide margin.
///
/// `Serialize` is hand-written below to fix the field order; `Deserialize` is
/// derived and stays internally tagged, because reading a tag is
/// order-independent.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum LlmRetryEvent {
    /// A bounded transient retry within one policy's budget.
    Normal {
        /// Identity of this retry chain.
        #[serde(rename = "retryId")]
        retry_id: RetryId,
        /// The turn containing the failed request.
        turn: u64,
        /// The step containing the failed request attempt.
        step: u64,
        /// The provider route that served the failed request.
        provider: String,
        /// Stable identity of the policy that scheduled this retry.
        #[serde(rename = "policyKey")]
        policy_key: String,
        /// One-based index of this retry within its chain.
        retry: u64,
        /// The budget this retry is drawn from.
        #[serde(rename = "maxRetries")]
        max_retries: u64,
        /// How long the wait before this retry lasts.
        #[serde(rename = "delayMs", serialize_with = "json::serialize_number")]
        delay_ms: f64,
        /// The failure that scheduled this retry.
        failure: Box<LlmFailure>,
    },
    /// An unbounded retry under an always policy.
    Always {
        /// Identity of this retry chain.
        #[serde(rename = "retryId")]
        retry_id: RetryId,
        /// The turn containing the failed request.
        turn: u64,
        /// The step containing the failed request attempt.
        step: u64,
        /// The provider route that served the failed request.
        provider: String,
        /// Stable identity of the policy that scheduled this retry.
        #[serde(rename = "policyKey")]
        policy_key: String,
        /// One-based index of this retry within its chain.
        retry: u64,
        /// How long the wait before this retry lasts.
        #[serde(rename = "delayMs", serialize_with = "json::serialize_number")]
        delay_ms: f64,
        /// The failure that scheduled this retry.
        failure: Box<LlmFailure>,
    },
}

/// Field order on the wire, matching the TypeScript declaration in
/// `packages/llm/llm-retry/src/types.ts`:
/// `retryId, turn, step, provider, mode, policyKey, retry, maxRetries, delayMs, failure`
/// — `mode` is the **fifth** key, not the first.
///
/// An internally tagged enum (`#[serde(tag = "mode")]`) emits its tag ahead of
/// every field, which would put `mode` first. `serde_json`'s `preserve_order`
/// makes insertion order observable on the wire, and a byte-level comparison
/// against a TypeScript-written log needs the declared order, so serialization
/// is written by hand. Deserialization stays derived: reading a tag is
/// order-independent.
impl Serialize for LlmRetryEvent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;

        let (retry_id, turn, step, provider, policy_key, retry, delay_ms, failure, max_retries) =
            match self {
                Self::Normal {
                    retry_id,
                    turn,
                    step,
                    provider,
                    policy_key,
                    retry,
                    max_retries,
                    delay_ms,
                    failure,
                } => (
                    retry_id,
                    *turn,
                    *step,
                    provider,
                    policy_key,
                    *retry,
                    *delay_ms,
                    failure,
                    Some(*max_retries),
                ),
                Self::Always {
                    retry_id,
                    turn,
                    step,
                    provider,
                    policy_key,
                    retry,
                    delay_ms,
                    failure,
                } => (
                    retry_id, *turn, *step, provider, policy_key, *retry, *delay_ms, failure, None,
                ),
            };

        let mut map = serializer.serialize_map(Some(if max_retries.is_some() { 10 } else { 9 }))?;
        map.serialize_entry("retryId", retry_id)?;
        map.serialize_entry("turn", &turn)?;
        map.serialize_entry("step", &step)?;
        map.serialize_entry("provider", provider)?;
        map.serialize_entry("mode", &self.mode())?;
        map.serialize_entry("policyKey", policy_key)?;
        map.serialize_entry("retry", &retry)?;
        if let Some(max_retries) = max_retries {
            map.serialize_entry("maxRetries", &max_retries)?;
        }
        map.serialize_entry("delayMs", &json::number(delay_ms))?;
        map.serialize_entry("failure", failure)?;
        map.end()
    }
}

impl LlmRetryEvent {
    /// Which policy mode scheduled this retry.
    pub fn mode(&self) -> RetryMode {
        match self {
            Self::Normal { .. } => RetryMode::Normal,
            Self::Always { .. } => RetryMode::Always,
        }
    }

    /// Identity of this retry chain.
    pub fn retry_id(&self) -> &RetryId {
        match self {
            Self::Normal { retry_id, .. } | Self::Always { retry_id, .. } => retry_id,
        }
    }

    /// One-based index of this retry within its chain.
    pub fn retry(&self) -> u64 {
        match self {
            Self::Normal { retry, .. } | Self::Always { retry, .. } => *retry,
        }
    }

    /// The provider route that served the failed request.
    pub fn provider(&self) -> &str {
        match self {
            Self::Normal { provider, .. } | Self::Always { provider, .. } => provider,
        }
    }

    /// Stable identity of the policy that scheduled this retry.
    pub fn policy_key(&self) -> &str {
        match self {
            Self::Normal { policy_key, .. } | Self::Always { policy_key, .. } => policy_key,
        }
    }

    /// How long the wait before this retry lasts.
    pub fn delay_ms(&self) -> f64 {
        match self {
            Self::Normal { delay_ms, .. } | Self::Always { delay_ms, .. } => *delay_ms,
        }
    }

    /// The failure that scheduled this retry.
    pub fn failure(&self) -> &LlmFailure {
        match self {
            Self::Normal { failure, .. } | Self::Always { failure, .. } => failure,
        }
    }

    /// The budget this retry is drawn from, which only normal mode has.
    pub fn max_retries(&self) -> Option<u64> {
        match self {
            Self::Normal { max_retries, .. } => Some(*max_retries),
            Self::Always { .. } => None,
        }
    }
}

/// The transition recorded after one retry delay completes and before the
/// next request attempt starts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmRetryStartedEvent {
    /// Identity of the retry chain this attempt continues.
    #[serde(rename = "retryId")]
    pub retry_id: RetryId,
    /// The turn containing the retried request.
    pub turn: u64,
    /// The step containing the retried request attempt.
    pub step: u64,
    /// One-based index of the retry that just started.
    pub retry: u64,
}

/// Encode one scheduled retry as the extension event the log stores.
pub fn encode_retry(event: &LlmRetryEvent) -> Result<SessionEvent, serde_json::Error> {
    encode(LLM_RETRY_TAG, event)
}

/// Encode one started retry as the extension event the log stores.
pub fn encode_retry_started(
    event: &LlmRetryStartedEvent,
) -> Result<SessionEvent, serde_json::Error> {
    encode(LLM_RETRY_STARTED_TAG, event)
}

/// Decode one scheduled retry from the log.
///
/// `None` means the event carries a different tag; `Some(Err(..))` means it is
/// a `llm/retry` event whose payload does not match the declared shape.
pub fn decode_retry(event: &SessionEvent) -> Option<Result<LlmRetryEvent, serde_json::Error>> {
    decode(event, LLM_RETRY_TAG)
}

/// Decode one started retry from the log.
///
/// `None` means the event carries a different tag; `Some(Err(..))` means it is
/// a `llm/retry-started` event whose payload does not match the declared shape.
pub fn decode_retry_started(
    event: &SessionEvent,
) -> Option<Result<LlmRetryStartedEvent, serde_json::Error>> {
    decode(event, LLM_RETRY_STARTED_TAG)
}

fn encode<T: Serialize>(tag: &str, event: &T) -> Result<SessionEvent, serde_json::Error> {
    match serde_json::to_value(event)? {
        serde_json::Value::Object(payload) => {
            Ok(SessionEvent::Extension(ExtensionEvent::new(tag, payload)))
        }
        other => Err(serde::ser::Error::custom(format!(
            "a session event payload must be a JSON object, got {other}"
        ))),
    }
}

fn decode<T: for<'de> Deserialize<'de>>(
    event: &SessionEvent,
    tag: &str,
) -> Option<Result<T, serde_json::Error>> {
    let extension = event.extension()?;
    if extension.kind() != tag {
        return None;
    }
    Some(extension.decode::<T>())
}
