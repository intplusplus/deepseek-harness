//! The raw streaming protocol emitted by adapters, plus the lossless compact
//! records embedded in durable assistant attempt events.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::brand::ToolCallId;
use super::types::{FinishReason, TokenUsage};

/// Raw streaming protocol emitted by adapters.
///
/// Block indexes correlate interleaved deltas, and `block-end` carries the
/// assembled block. Adapters emit usage before the terminal finish and nothing
/// afterward; tool arguments remain raw JSON strings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum StreamChunk {
    BlockStart {
        index: u64,
        block_type: String,
    },
    TextDelta {
        index: u64,
        text: String,
    },
    ReasoningDelta {
        index: u64,
        text: String,
    },
    ToolCallDelta {
        index: u64,
        id: ToolCallId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        arguments_delta: String,
    },
    BlockEnd {
        index: u64,
        block: super::content::ContentBlock,
    },
    Usage {
        usage: TokenUsage,
    },
    Finish {
        reason: FinishReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        replay_state: Option<ReplayEnvelope>,
    },
}

/// Adapter-private lossless-JSON state for replaying a successful response.
///
/// Both halves stay opaque to the harness; only the split is shared
/// vocabulary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayEnvelope {
    pub response: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocks: Option<Vec<Value>>,
}

/// One model chunk paired with its original session timestamp.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimedStreamChunk {
    pub time: u64,
    pub chunk: StreamChunk,
}

/// Lossless compact records embedded in durable assistant attempt events.
///
/// Every compact record except a raw `chunk` is one packed delta run; the raw
/// `chunk` record preserves chunks the accumulator never packs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum AssistantStreamRecord {
    TextChunks {
        time0: u64,
        index: u64,
        dt: Vec<u64>,
        texts: Vec<String>,
    },
    ReasoningChunks {
        time0: u64,
        index: u64,
        dt: Vec<u64>,
        texts: Vec<String>,
    },
    ToolCallChunks {
        time0: u64,
        index: u64,
        dt: Vec<u64>,
        id: ToolCallId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        args: Vec<String>,
    },
    /// Boxed: a raw `chunk` record otherwise inflates every record in a stream
    /// to the size of the largest `StreamChunk` variant.
    Chunk { time: u64, chunk: Box<StreamChunk> },
}
