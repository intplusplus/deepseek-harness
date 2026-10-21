//! DeepSeek `chat/completions` wire types (OpenAI-compatible).
//!
//! The Rust provider speaks the standard OpenAI-compatible DeepSeek endpoint
//! (`POST {base_url}/chat/completions`) rather than the in-house Messages
//! transport the TypeScript `llm-deepseek` package uses. That keeps the wire
//! format self-describing: request/response JSON plus Server-Sent Events
//! framed as `data: {...}` with a terminating `data: [DONE]`.
//!
//! All optional fields default to `None`/`0` so a partial stream frame never
//! fails to deserialize; the translator decides what a missing field means.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Request body sent to `chat/completions`.
#[derive(Clone, Debug, Serialize, Default)]
pub(crate) struct WireRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// One conversation message on the wire. `content` is a string for text roles
/// and may be `null` when the assistant message carries only tool calls.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WireMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCallRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool the model may call, as sent in the request `tools` array.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct WireTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: WireFunction,
}

/// Tool function schema sent to the model.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct WireFunction {
    pub name: String,
    pub description: String,
    pub parameters: Map<String, Value>,
}

/// A tool call the model made, as echoed back in a request `assistant` message.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct WireToolCallRequest {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: WireToolCallFunction,
}

/// Arguments of a request-side tool call.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct WireToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// A non-streaming chat completion response.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WireChatCompletion {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    pub choices: Vec<WireChoice>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
}

/// One completion choice.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WireChoice {
    #[serde(default)]
    pub index: u64,
    pub message: WireMessageResponse,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// An assistant message inside a completion.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WireMessageResponse {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<WireToolCallResponse>>,
}

/// A tool call produced by the model.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WireToolCallResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub r#type: Option<String>,
    pub function: WireToolCallResponseFunction,
}

/// Function identity and raw JSON arguments of a model tool call.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WireToolCallResponseFunction {
    #[serde(default)]
    pub name: Option<String>,
    pub arguments: String,
}

/// Token accounting, present at the end of a stream or in a full response.
#[derive(Clone, Debug, Deserialize, Default)]
pub(crate) struct WireUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: Option<u64>,
    #[serde(default)]
    pub prompt_tokens_details: Option<WireTokenDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<WireTokenDetails>,
}

/// Optional token detail buckets.
#[derive(Clone, Debug, Deserialize, Default)]
pub(crate) struct WireTokenDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

/// One Server-Sent Event decoded from the streaming endpoint.
#[derive(Clone, Debug, Deserialize, Default)]
pub(crate) struct WireChunk {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<WireChunkChoice>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
}

/// One choice inside a streamed chunk.
#[derive(Clone, Debug, Deserialize, Default)]
pub(crate) struct WireChunkChoice {
    #[serde(default)]
    pub index: u64,
    #[serde(default)]
    pub delta: WireDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// A streamed delta for one choice.
#[derive(Clone, Debug, Deserialize, Default)]
pub(crate) struct WireDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<WireToolCallDelta>>,
}

/// A streamed delta for a tool call; `index` correlates parallel calls.
#[derive(Clone, Debug, Deserialize, Default)]
pub(crate) struct WireToolCallDelta {
    #[serde(default)]
    pub index: Option<u64>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub function: Option<WireToolCallDeltaFunction>,
}

/// Streamed function identity/arguments for one tool call delta.
#[derive(Clone, Debug, Deserialize, Default)]
pub(crate) struct WireToolCallDeltaFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}
