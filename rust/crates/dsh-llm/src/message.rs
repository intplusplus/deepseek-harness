//! Message vocabulary: the five role messages and the merge-extensible
//! `MessageSource` union.
//!
//! `MessageSourceMap` grows by declaration merging in TypeScript; here a
//! producer-declared source kind lands in the `Extension` slot and consumers
//! fall through unknown kinds. Model and tool sources answer their role
//! messages; user and developer messages carry any producer's kind.

#![forbid(unsafe_code)]

use std::ops::Not;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use super::brand::{MessageId, ToolCallId};
use super::content::ContentBlock;

/// The closed core source-kind vocabulary.
pub const CORE_MESSAGE_SOURCE_KINDS: &[&str] = &["user", "model", "tool", "system-prompt"];

/// A producer-declared message source, preserved verbatim.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtensionSource {
    kind: String,
    fields: Map<String, Value>,
}

impl ExtensionSource {
    /// Create an extension source; `kind` is stamped into the fields.
    pub fn new(kind: impl Into<String>, mut fields: Map<String, Value>) -> Self {
        let kind = kind.into();
        fields.insert("kind".to_string(), Value::String(kind.clone()));
        Self { kind, fields }
    }

    /// The source's `kind` tag, as declared by the producer.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The source's verbatim JSON fields, including its `kind` tag.
    pub fn fields(&self) -> &Map<String, Value> {
        &self.fields
    }
}

impl Serialize for ExtensionSource {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.fields.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ExtensionSource {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::Object(fields) => {
                let kind = fields
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                Ok(Self { kind, fields })
            }
            other => Err(serde::de::Error::custom(format!(
                "message source must be a JSON object, got {other}"
            ))),
        }
    }
}

/// Where a message came from, in the harness's own vocabulary.
///
/// `MessageSource.kind` answers *who produced this*; consumers switch on it
/// and fall through unknown kinds.
#[derive(Clone, Debug, PartialEq)]
pub enum MessageSource {
    /// A direct human prompt.
    User,
    /// Provider/model identity and adapter-private replay data.
    Model {
        provider: String,
        model: String,
        /// Lossless-JSON adapter state needed to replay the provider response.
        /// Serialized only when present; see the manual `Serialize` impl.
        replay_state: Option<Value>,
    },
    /// A tool-role message carrying one tool result.
    Tool { call_id: ToolCallId },
    /// A system-role message produced by the system-prompt plugin.
    SystemPrompt,
    /// A producer-declared source kind, preserved verbatim.
    Extension(ExtensionSource),
}

impl MessageSource {
    /// The source's `kind` tag.
    pub fn kind(&self) -> &str {
        match self {
            Self::User => "user",
            Self::Model { .. } => "model",
            Self::Tool { .. } => "tool",
            Self::SystemPrompt => "system-prompt",
            Self::Extension(source) => source.kind(),
        }
    }
}

impl Serialize for MessageSource {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut fields = Map::new();
        match self {
            Self::User => {
                fields.insert("kind".to_string(), Value::String("user".to_string()));
            }
            Self::Model {
                provider,
                model,
                replay_state,
            } => {
                fields.insert("kind".to_string(), Value::String("model".to_string()));
                fields.insert("provider".to_string(), Value::String(provider.clone()));
                fields.insert("model".to_string(), Value::String(model.clone()));
                if let Some(state) = replay_state {
                    fields.insert("replayState".to_string(), state.clone());
                }
            }
            Self::Tool { call_id } => {
                fields.insert("kind".to_string(), Value::String("tool".to_string()));
                fields.insert(
                    "callId".to_string(),
                    Value::String(call_id.as_str().to_string()),
                );
            }
            Self::SystemPrompt => {
                fields.insert(
                    "kind".to_string(),
                    Value::String("system-prompt".to_string()),
                );
            }
            Self::Extension(source) => return source.serialize(serializer),
        }
        fields.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MessageSource {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let kind = value
            .as_object()
            .and_then(|map| map.get("kind"))
            .and_then(Value::as_str);
        match kind {
            Some("user") => Ok(Self::User),
            Some("model") => {
                let fields = value
                    .as_object()
                    .ok_or_else(|| serde::de::Error::custom("model source must be an object"))?;
                let provider =
                    string_field(fields, "provider").map_err(serde::de::Error::custom)?;
                let model = string_field(fields, "model").map_err(serde::de::Error::custom)?;
                let replay_state = fields.get("replayState").cloned();
                Ok(Self::Model {
                    provider,
                    model,
                    replay_state,
                })
            }
            Some("tool") => {
                let fields = value
                    .as_object()
                    .ok_or_else(|| serde::de::Error::custom("tool source must be an object"))?;
                let call_id = string_field(fields, "callId").map_err(serde::de::Error::custom)?;
                Ok(Self::Tool {
                    call_id: ToolCallId::new(call_id),
                })
            }
            Some("system-prompt") => Ok(Self::SystemPrompt),
            _ => serde_json::from_value::<ExtensionSource>(value)
                .map(Self::Extension)
                .map_err(serde::de::Error::custom),
        }
    }
}

fn string_field(fields: &Map<String, Value>, key: &str) -> Result<String, String> {
    fields
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("missing string field `{key}`"))
}

/// A rendered system prompt attributed to the system-prompt producer; empty
/// content sends no prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemMessage {
    pub id: MessageId,
    pub content: Vec<ContentBlock>,
    pub source: MessageSource,
}

impl SystemMessage {
    /// The system-prompt producer is the only legal source for a system message.
    pub fn new(id: MessageId, content: Vec<ContentBlock>) -> Self {
        Self {
            id,
            content,
            source: MessageSource::SystemPrompt,
        }
    }
}

/// Incremental agent session changes in conversation order, currently tool
/// additions and removals.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeveloperMessage {
    pub id: MessageId,
    pub content: Vec<ContentBlock>,
    pub source: MessageSource,
}

impl DeveloperMessage {
    /// Developer messages carry any producer's kind.
    pub fn new(id: MessageId, content: Vec<ContentBlock>, source: MessageSource) -> Self {
        Self {
            id,
            content,
            source,
        }
    }
}

/// A user-role message: a direct human prompt, a synthetic `agent.inject()`
/// context, or an entered goal continuation round.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    pub id: MessageId,
    pub content: Vec<ContentBlock>,
    pub source: MessageSource,
}

impl UserMessage {
    /// A direct human prompt (`kind: 'user'`).
    pub fn new(id: MessageId, content: Vec<ContentBlock>) -> Self {
        Self {
            id,
            content,
            source: MessageSource::User,
        }
    }

    /// A user-role message from another producer's kind.
    pub fn with_source(id: MessageId, content: Vec<ContentBlock>, source: MessageSource) -> Self {
        Self {
            id,
            content,
            source,
        }
    }
}

/// A model-produced assistant message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub id: MessageId,
    pub content: Vec<ContentBlock>,
    pub source: MessageSource,
}

impl AssistantMessage {
    /// A routed model message; the source records provider and model identity.
    pub fn new(
        id: MessageId,
        content: Vec<ContentBlock>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            id,
            content,
            source: MessageSource::Model {
                provider: provider.into(),
                model: model.into(),
                replay_state: None,
            },
        }
    }

    /// Attach adapter-private replay state to the model source.
    pub fn with_replay_state(mut self, state: Value) -> Self {
        if let MessageSource::Model { replay_state, .. } = &mut self.source {
            *replay_state = Some(state);
        }
        self
    }
}

/// A first-class tool-role message carrying the result of one tool invocation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub id: MessageId,
    pub content: Vec<ContentBlock>,
    pub source: MessageSource,
    /// Provider-issued id of the tool call this message answers.
    pub tool_call_id: ToolCallId,
    /// Whether the tool invocation failed; serialized only when true.
    #[serde(default, skip_serializing_if = "Not::not")]
    pub is_error: bool,
}

impl ToolResultMessage {
    /// A tool result; the source is derived from the answered call id.
    pub fn new(
        id: MessageId,
        content: Vec<ContentBlock>,
        call_id: ToolCallId,
        is_error: bool,
    ) -> Self {
        Self {
            id,
            content,
            source: MessageSource::Tool {
                call_id: call_id.clone(),
            },
            tool_call_id: call_id,
            is_error,
        }
    }
}

/// Any conversation message. The `role` tag is the wire discriminator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "kebab-case")]
pub enum Message {
    System(SystemMessage),
    Developer(DeveloperMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolResultMessage),
}

impl Message {
    /// The message's role tag.
    pub fn role(&self) -> &'static str {
        match self {
            Self::System(_) => "system",
            Self::Developer(_) => "developer",
            Self::User(_) => "user",
            Self::Assistant(_) => "assistant",
            Self::Tool(_) => "tool",
        }
    }

    /// The message's exact model-facing blocks.
    pub fn content(&self) -> &[ContentBlock] {
        match self {
            Self::System(message) => &message.content,
            Self::Developer(message) => &message.content,
            Self::User(message) => &message.content,
            Self::Assistant(message) => &message.content,
            Self::Tool(message) => &message.content,
        }
    }

    /// The message's stable identity.
    pub fn id(&self) -> &MessageId {
        match self {
            Self::System(message) => &message.id,
            Self::Developer(message) => &message.id,
            Self::User(message) => &message.id,
            Self::Assistant(message) => &message.id,
            Self::Tool(message) => &message.id,
        }
    }

    /// The message's producer-declared source.
    pub fn source(&self) -> &MessageSource {
        match self {
            Self::System(message) => &message.source,
            Self::Developer(message) => &message.source,
            Self::User(message) => &message.source,
            Self::Assistant(message) => &message.source,
            Self::Tool(message) => &message.source,
        }
    }
}
