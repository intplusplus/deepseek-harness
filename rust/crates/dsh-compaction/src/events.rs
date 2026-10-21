//! The durable `compaction/*` event vocabulary.
//!
//! TypeScript declares these by merging into `SessionEventMap`. Rust has no
//! declaration merging, so they are ordinary payload types that reach the log
//! through [`SessionEvent::Extension`]. [`CompactionEvent::encode`] and
//! [`CompactionEvent::decode`] are the bridge and preserve the extension-slot
//! invariant: a known tag is never silently degraded to an opaque payload, so a
//! malformed `compaction/*` record decodes to `Some(Err(..))`.
//!
//! Field order is observable because the workspace enables `serde_json`'s
//! `preserve_order`: every row starts with `type`, and the remaining fields
//! follow the order the TypeScript producer inserted them.

#![forbid(unsafe_code)]

use serde::de::Error as _;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use dsh_llm::{ContentBlock, TokenUsage};
use dsh_session::{ExtensionEvent, SessionEvent, SessionSeq};

use super::brand::{CommandId, CompactionId};

/// Tag of the compaction-lock opening event.
pub const COMPACTION_START_TAG: &str = "compaction/start";
/// Tag of the completed-summary record.
pub const COMPACTION_SUMMARY_TAG: &str = "compaction/summary";
/// Tag of the compaction-lock closing event.
pub const COMPACTION_END_TAG: &str = "compaction/end";
/// Tag of the shadow-price record for one model-free prune replacement.
pub const COMPACTION_PRUNE_TAG: &str = "compaction/prune";

/// Every tag this crate owns.
pub const COMPACTION_EVENT_TAGS: &[&str] = &[
    COMPACTION_START_TAG,
    COMPACTION_SUMMARY_TAG,
    COMPACTION_END_TAG,
    COMPACTION_PRUNE_TAG,
];

/// The turn that owns a compaction transaction.
///
/// TypeScript declares `turn: number | null`: the key is always present, and
/// `null` identifies a standalone manual transaction between turns. Wrapping
/// `Option<u64>` keeps that rule on the wire — a missing key is a decode error,
/// while an explicit `null` serializes and decodes as [`CompactionTurn::standalone`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompactionTurn(Option<u64>);

impl CompactionTurn {
    /// Create an owner from an optional turn number.
    pub fn new(turn: Option<u64>) -> Self {
        Self(turn)
    }

    /// A standalone manual transaction between turns.
    pub fn standalone() -> Self {
        Self(None)
    }

    /// The owning turn number, or `None` for a standalone transaction.
    pub fn get(self) -> Option<u64> {
        self.0
    }

    /// Whether this transaction is standalone.
    pub fn is_standalone(self) -> bool {
        self.0.is_none()
    }
}

impl From<Option<u64>> for CompactionTurn {
    fn from(turn: Option<u64>) -> Self {
        Self(turn)
    }
}

impl Serialize for CompactionTurn {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Some(turn) => serializer.serialize_u64(turn),
            None => serializer.serialize_none(),
        }
    }
}

impl<'de> Deserialize<'de> for CompactionTurn {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Option::<u64>::deserialize(deserializer).map(Self)
    }
}

/// Deserialize a `turn` field whose key must be present.
///
/// `Option<u64>` alone also accepts a missing key, because serde's
/// missing-field path succeeds for `Option`; requiring the field through an
/// explicit `deserialize_with` keeps `null` distinct from absence.
fn deserialize_required_turn<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<CompactionTurn, D::Error> {
    CompactionTurn::deserialize(deserializer)
}

/// One inclusive span of surface-node sequences.
///
/// This is a surface-POSITION span, not a numeric interval: after an earlier
/// replacement lands a high-sequence summary node at an older position,
/// `start` can be greater than `end`. The authoritative set is
/// `shadowedSeqs`; `start`/`end` name its first and last members.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShadowedRange {
    /// First shadowed surface-node sequence.
    pub start: SessionSeq,
    /// Last shadowed surface-node sequence.
    pub end: SessionSeq,
}

impl ShadowedRange {
    /// Create one inclusive surface-position span.
    pub fn new(start: SessionSeq, end: SessionSeq) -> Self {
        Self { start, end }
    }
}

/// Marks the start of one compaction transaction and holds its lock.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompactionStart {
    /// Stable identity shared by the whole start/summary/end transaction.
    pub compaction_id: CompactionId,
    /// Initiating manual command, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_command_id: Option<CommandId>,
    /// Owning turn, or an explicit `null` between turns.
    #[serde(deserialize_with = "deserialize_required_turn")]
    pub turn: CompactionTurn,
    /// Epoch milliseconds at which the durable lock was acquired, used for
    /// lease-expiry recovery. Absent on histories written before lease timing
    /// was recorded; such a lock is treated as held (never auto-expiring).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acquired_at: Option<i64>,
}

impl CompactionStart {
    /// Create one lock-opening event.
    pub fn new(
        compaction_id: CompactionId,
        source_command_id: Option<CommandId>,
        turn: CompactionTurn,
    ) -> Self {
        Self {
            compaction_id,
            source_command_id,
            turn,
            acquired_at: None,
        }
    }

    /// Create one lock-opening event stamped with its acquisition time.
    pub fn with_acquired_at(
        compaction_id: CompactionId,
        source_command_id: Option<CommandId>,
        turn: CompactionTurn,
        acquired_at: i64,
    ) -> Self {
        Self {
            compaction_id,
            source_command_id,
            turn,
            acquired_at: Some(acquired_at),
        }
    }
}

/// A completed summary, its selected input, and its model-call facts.
///
/// The summary is log-only: the replacement `user/message` immediately after
/// this record performs the surface change. The adjacency is contractual
/// because `shadowedTokenCount` is the replacement's shadow price.
///
/// `rawOutput` is the complete provider output before the backend's safe
/// summary projection. [`CompactionSummary::llm_stream_call`] is serialized
/// only when true and true requires `rawOutput`; an unmarked summary may still
/// carry `rawOutput` but never claims an LLM-seam call.
///
/// The hand-written `Serialize` follows the TypeScript producer in
/// `commitCompactionBody`: `rawOutput`/`llmStreamCall` precede `shadowedRange`,
/// while the declaration's intersection places them after the route fields.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionSummary {
    /// Transaction identity shared with `compaction/start`.
    pub compaction_id: CompactionId,
    /// Initiating manual command, when present.
    pub source_command_id: Option<CommandId>,
    /// Safe, text-only summary blocks.
    pub summary: Vec<ContentBlock>,
    /// Complete provider output, when retained.
    pub raw_output: Option<Vec<ContentBlock>>,
    /// True for exactly one call through the context's LLM seam.
    pub llm_stream_call: bool,
    /// Declared replaced surface span.
    pub shadowed_range: ShadowedRange,
    /// All shadowed surface-node sequences, in surface order.
    pub shadowed_seqs: Vec<SessionSeq>,
    /// Heuristic price of the shadowed content.
    pub shadowed_token_count: u64,
    /// Provider route that wrote the summary.
    pub provider: String,
    /// Model that wrote the summary.
    pub model: String,
    /// Generation cap sent by the summarize call, when one applied.
    pub max_tokens: Option<u64>,
    /// Provider-reported usage for the summarization request.
    pub usage: Option<TokenUsage>,
}

impl CompactionSummary {
    /// Validate the `rawOutput`/`llmStreamCall` invariant this type carries.
    ///
    /// The fields are public so callers can assemble a payload without a
    /// positional constructor; encoding still refuses an impossible pair.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.llm_stream_call && self.raw_output.is_none() {
            return Err("compaction/summary: llmStreamCall requires rawOutput");
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CompactionSummaryWire {
    compaction_id: CompactionId,
    #[serde(default)]
    source_command_id: Option<CommandId>,
    summary: Vec<ContentBlock>,
    #[serde(default)]
    raw_output: Option<Vec<ContentBlock>>,
    #[serde(default, deserialize_with = "deserialize_true_only")]
    llm_stream_call: bool,
    shadowed_range: ShadowedRange,
    shadowed_seqs: Vec<SessionSeq>,
    shadowed_token_count: u64,
    provider: String,
    model: String,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(default)]
    usage: Option<TokenUsage>,
}

fn deserialize_true_only<'de, D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    let value = bool::deserialize(deserializer)?;
    if value {
        Ok(true)
    } else {
        Err(D::Error::custom(
            "compaction/summary: llmStreamCall must be true when present",
        ))
    }
}

impl<'de> Deserialize<'de> for CompactionSummary {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = CompactionSummaryWire::deserialize(deserializer)?;
        if wire.llm_stream_call && wire.raw_output.is_none() {
            return Err(D::Error::custom(
                "compaction/summary: llmStreamCall requires rawOutput",
            ));
        }
        Ok(Self {
            compaction_id: wire.compaction_id,
            source_command_id: wire.source_command_id,
            summary: wire.summary,
            raw_output: wire.raw_output,
            llm_stream_call: wire.llm_stream_call,
            shadowed_range: wire.shadowed_range,
            shadowed_seqs: wire.shadowed_seqs,
            shadowed_token_count: wire.shadowed_token_count,
            provider: wire.provider,
            model: wire.model,
            max_tokens: wire.max_tokens,
            usage: wire.usage,
        })
    }
}

impl Serialize for CompactionSummary {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("compactionId", &self.compaction_id)?;
        if let Some(source_command_id) = &self.source_command_id {
            map.serialize_entry("sourceCommandId", source_command_id)?;
        }
        map.serialize_entry("summary", &self.summary)?;
        if let Some(raw_output) = &self.raw_output {
            map.serialize_entry("rawOutput", raw_output)?;
        }
        if self.llm_stream_call {
            map.serialize_entry("llmStreamCall", &true)?;
        }
        map.serialize_entry("shadowedRange", &self.shadowed_range)?;
        map.serialize_entry("shadowedSeqs", &self.shadowed_seqs)?;
        map.serialize_entry("shadowedTokenCount", &self.shadowed_token_count)?;
        map.serialize_entry("provider", &self.provider)?;
        map.serialize_entry("model", &self.model)?;
        if let Some(max_tokens) = self.max_tokens {
            map.serialize_entry("maxTokens", &max_tokens)?;
        }
        if let Some(usage) = &self.usage {
            map.serialize_entry("usage", usage)?;
        }
        map.end()
    }
}

/// Marks the end of a compaction transaction and releases its lock.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompactionEnd {
    /// Transaction identity shared with `compaction/start`.
    pub compaction_id: CompactionId,
    /// Initiating manual command, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_command_id: Option<CommandId>,
    /// Owning turn, or an explicit `null` between turns.
    #[serde(deserialize_with = "deserialize_required_turn")]
    pub turn: CompactionTurn,
    /// Failure diagnostic for an unsuccessful attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl CompactionEnd {
    /// Create one lock-closing event.
    pub fn new(
        compaction_id: CompactionId,
        source_command_id: Option<CommandId>,
        turn: CompactionTurn,
        error: Option<String>,
    ) -> Self {
        Self {
            compaction_id,
            source_command_id,
            turn,
            error,
        }
    }
}

/// Shadow price of one model-free prune replacement.
///
/// The replacement `tool/result` must be appended synchronously right after
/// this event. A pure consumer subtracts `shadowedTokenCount` from its running
/// price without retaining per-node prices.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionPrune {
    /// The replaced surface span.
    pub shadowed_range: ShadowedRange,
    /// All shadowed surface-node sequences, in surface order.
    pub shadowed_seqs: Vec<SessionSeq>,
    /// Heuristic price of the shadowed content.
    pub shadowed_token_count: u64,
}

impl CompactionPrune {
    /// Create one prune pricing record.
    pub fn new(
        shadowed_range: ShadowedRange,
        shadowed_seqs: Vec<SessionSeq>,
        shadowed_token_count: u64,
    ) -> Self {
        Self {
            shadowed_range,
            shadowed_seqs,
            shadowed_token_count,
        }
    }
}

/// A successful compaction operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionResult {
    /// Stable identity shared by the complete durable lifecycle.
    pub compaction_id: CompactionId,
    /// Human command that initiated a manual compaction, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_command_id: Option<CommandId>,
    /// Sequence of the appended `compaction/start` event.
    pub start_seq: SessionSeq,
    /// Sequence of the appended `compaction/summary` event.
    pub summary_seq: SessionSeq,
    /// Sequence of the appended `compaction/end` event.
    pub end_seq: SessionSeq,
    /// Summary content blocks produced by the backend.
    pub summary: Vec<ContentBlock>,
    /// The replaced surface span.
    pub shadowed_range: ShadowedRange,
    /// All shadowed surface-node sequences, in surface order.
    pub shadowed_seqs: Vec<SessionSeq>,
    /// Estimated token count of the shadowed content.
    pub shadowed_token_count: u64,
}

/// One decoded `compaction/*` event.
#[derive(Clone, Debug, PartialEq)]
pub enum CompactionEvent {
    /// The lock-opening marker.
    Start(CompactionStart),
    /// The completed-summary record.
    ///
    /// Boxed to keep this transient decode view small: the summary carries the
    /// full content blocks, raw output, usage, and shadow span, while the other
    /// three variants are markers. The log stores JSON, so the box is invisible
    /// on the wire.
    Summary(Box<CompactionSummary>),
    /// The lock-closing marker.
    End(CompactionEnd),
    /// One prune shadow-price record.
    Prune(CompactionPrune),
}

impl CompactionEvent {
    /// The event's `type` tag.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Start(_) => COMPACTION_START_TAG,
            Self::Summary(_) => COMPACTION_SUMMARY_TAG,
            Self::End(_) => COMPACTION_END_TAG,
            Self::Prune(_) => COMPACTION_PRUNE_TAG,
        }
    }

    /// Encode this event as the extension event the log stores.
    ///
    /// Returns a serialization error when the payload violates an invariant
    /// this type can express only on the wire — currently
    /// `llmStreamCall: true` without `rawOutput`.
    pub fn encode(&self) -> Result<SessionEvent, serde_json::Error> {
        match self {
            Self::Start(event) => encode_payload(COMPACTION_START_TAG, event),
            Self::Summary(event) => {
                event.validate().map_err(serde::ser::Error::custom)?;
                encode_payload(COMPACTION_SUMMARY_TAG, event)
            }
            Self::End(event) => encode_payload(COMPACTION_END_TAG, event),
            Self::Prune(event) => encode_payload(COMPACTION_PRUNE_TAG, event),
        }
    }

    /// Decode one event from the log.
    ///
    /// `None` means the event carries a different tag. `Some(Err(..))` means it
    /// is one of this crate's tags whose payload does not match the declared
    /// type.
    pub fn decode(event: &SessionEvent) -> Option<Result<Self, serde_json::Error>> {
        let extension = event.extension()?;
        // The stamped `type` discriminator is envelope metadata, not payload:
        // the payload types declare `deny_unknown_fields` without a `type`
        // field, so decoding the verbatim payload would always reject them.
        let decoded = match extension.kind() {
            COMPACTION_START_TAG => decode_payload::<CompactionStart>(extension).map(Self::Start),
            COMPACTION_SUMMARY_TAG => decode_payload::<CompactionSummary>(extension)
                .map(|payload| Self::Summary(Box::new(payload))),
            COMPACTION_END_TAG => decode_payload::<CompactionEnd>(extension).map(Self::End),
            COMPACTION_PRUNE_TAG => decode_payload::<CompactionPrune>(extension).map(Self::Prune),
            _ => return None,
        };
        Some(decoded)
    }
}

/// Deserialize an extension payload as `T` with its `type` tag removed.
fn decode_payload<T: for<'de> Deserialize<'de>>(
    extension: &ExtensionEvent,
) -> Result<T, serde_json::Error> {
    let mut payload = extension.payload().clone();
    payload.remove("type");
    serde_json::from_value(Value::Object(payload))
}

fn encode_payload<T: Serialize>(tag: &str, event: &T) -> Result<SessionEvent, serde_json::Error> {
    match serde_json::to_value(event)? {
        Value::Object(payload) => Ok(SessionEvent::Extension(ExtensionEvent::new(tag, payload))),
        other => Err(serde::ser::Error::custom(format!(
            "a session event payload must be a JSON object, got {other}"
        ))),
    }
}
