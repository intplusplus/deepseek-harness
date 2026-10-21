//! The append-only `SessionEvent` vocabulary.
//!
//! TypeScript grows `SessionEventMap` by declaration merging: the compaction
//! seam adds `compaction/*`, the agent package adds `agent/inbox/spliced`, and
//! the hook bridge adds `hook/*`. Rust has no declaration merging, so the core
//! owns the closed [`CoreEvent`] union and plugins extend the log through the
//! [`SessionEvent::Extension`] slot, which preserves the declared event
//! verbatim. Unknown event types are ignorable, not fatal — the per-event
//! `ignorable` marker of the TypeScript design.

#![forbid(unsafe_code)]

use std::ops::Not;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use dsh_llm::{
    AssistantMessage, AssistantStreamRecord, DeveloperMessage, SystemMessage, TokenUsage,
    ToolCallId, ToolResultMessage, UserMessage,
};

use super::types::{EpochHeader, RequestContext, RequestHeaderReason, SessionSeq, TurnEndReason};

/// The closed core event vocabulary — the `type` tags the core owns.
pub const CORE_EVENT_TAGS: &[&str] = &[
    "turn/start",
    "turn/end",
    "step/start",
    "step/end",
    "user/message",
    "developer/message",
    "system/message",
    "assistant/message",
    "assistant/attempt",
    "tool/call",
    "tool/result",
    "request/header",
    "request/context",
    "session/end-seed",
];

/// One core session event.
///
/// Field names match the TypeScript wire format (`camelCase`); the `type` tag
/// is the event map key.
///
/// ## Layout
///
/// The log is append-only and grows without bound, so an event's *inline* size
/// is paid by every event, including the tiny lifecycle ones: a bare
/// `turn/start` must not cost as much as a full `assistant/message`. Every
/// payload larger than a couple of pointers is therefore boxed, which keeps
/// the enum at roughly the size of its largest small variant. `Box<T>`
/// serializes exactly as `T`, so this is invisible on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CoreEvent {
    /// Opens turn `turn` before the loop claims queued input or runs pre-step.
    #[serde(rename = "turn/start")]
    TurnStart { turn: u64 },
    /// Closes turn `turn` with the reason that ended it.
    #[serde(rename = "turn/end")]
    TurnEnd { turn: u64, reason: TurnEndReason },
    /// Opens step `step` of turn `turn` — one model call plus its tools.
    #[serde(rename = "step/start")]
    StepStart { turn: u64, step: u64 },
    /// Closes step `step` of turn `turn`.
    #[serde(rename = "step/end")]
    StepEnd { turn: u64, step: u64 },
    /// A user-role message on the model-visible surface.
    #[serde(rename = "user/message")]
    UserMessage { message: Box<UserMessage> },
    /// An incremental agent session change admitted at the named turn and step.
    #[serde(rename = "developer/message")]
    DeveloperMessage {
        turn: u64,
        step: u64,
        message: Box<DeveloperMessage>,
        /// Earlier request/header defining every tool addition; required
        /// exactly when additions are present.
        #[serde(default, rename = "headerSeq", skip_serializing_if = "Option::is_none")]
        header_seq: Option<SessionSeq>,
    },
    /// The rendered system prompt on the model-visible surface.
    #[serde(rename = "system/message")]
    SystemMessage {
        turn: u64,
        step: u64,
        message: Box<SystemMessage>,
    },
    /// Assembled assistant message for one step.
    #[serde(rename = "assistant/message")]
    AssistantMessage {
        turn: u64,
        step: u64,
        message: Box<AssistantMessage>,
        /// Exact timed model stream, compacted without joining delta boundaries.
        stream: Vec<AssistantStreamRecord>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Box<TokenUsage>>,
        /// A turn cancelled mid-stream finalizes its delivered prefix as this
        /// event with `interrupted: true`; serialized only when true.
        #[serde(default, skip_serializing_if = "Not::not")]
        interrupted: bool,
    },
    /// One model attempt that committed no surface message.
    #[serde(rename = "assistant/attempt")]
    AssistantAttempt {
        turn: u64,
        step: u64,
        stream: Vec<AssistantStreamRecord>,
    },
    /// The model requested one tool invocation.
    #[serde(rename = "tool/call")]
    ToolCall {
        turn: u64,
        step: u64,
        #[serde(rename = "callId")]
        call_id: ToolCallId,
        name: String,
        /// Raw JSON string exactly as the model produced it (unparsed).
        arguments: String,
    },
    /// A completed tool call's model-facing result.
    #[serde(rename = "tool/result")]
    ToolResult {
        turn: u64,
        step: u64,
        message: Box<ToolResultMessage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<Box<super::types::ToolErrorInfo>>,
        /// Tool-private presentation payload; opaque to the core but MUST be
        /// JSON-serializable.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        meta: Option<Value>,
    },
    /// Full header for the next request, appended inside its step before dispatch.
    #[serde(rename = "request/header")]
    RequestHeader {
        header: Box<EpochHeader>,
        reason: RequestHeaderReason,
        /// This request begins a distinct model-message series; serialized
        /// only when true.
        #[serde(default, rename = "startsSeries", skip_serializing_if = "Not::not")]
        starts_series: bool,
    },
    /// Route metadata for the next request, logged only when the route,
    /// capacity, or system prompt update mode changes.
    #[serde(rename = "request/context")]
    RequestContext(RequestContext),
    /// Separates inherited or restored history from later lifecycle-owned work.
    #[serde(rename = "session/end-seed")]
    EndSeed {
        /// Serialized only when true.
        #[serde(default, skip_serializing_if = "Not::not")]
        inherited: bool,
    },
}

/// A plugin-declared session event.
///
/// `kind` is the event's `type` tag (for example `compaction/start` or
/// `agent/inbox/spliced`); the payload is preserved verbatim, `type` included.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtensionEvent {
    kind: String,
    payload: Map<String, Value>,
}

impl ExtensionEvent {
    /// Create an extension event; `type` is stamped from `kind`.
    ///
    /// `type` is placed **first** in the payload, not appended. The TypeScript
    /// wire format opens every row with its discriminator — every literal in
    /// the format tests is `{ type: '...', ... }` — and `serde_json`'s
    /// `preserve_order` makes insertion order observable on the wire, so
    /// appending the tag would put it last and break byte-level comparison
    /// against a TypeScript-written log. Position is the only reason this is
    /// not a plain `insert`.
    pub fn new(kind: impl Into<String>, mut payload: Map<String, Value>) -> Self {
        let kind = kind.into();
        // Drop any caller-supplied `type` first, so the stamped tag is the only
        // one and it lands at the front rather than being overwritten in place.
        payload.remove("type");
        let mut stamped = Map::new();
        stamped.insert("type".to_string(), Value::String(kind.clone()));
        stamped.extend(payload);
        Self {
            kind,
            payload: stamped,
        }
    }

    /// The event's `type` tag, as declared by the plugin.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The event's verbatim JSON payload, including its `type` tag.
    pub fn payload(&self) -> &Map<String, Value> {
        &self.payload
    }

    /// Interpret the payload as `T`, the plugin-declared shape.
    pub fn decode<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_value(Value::Object(self.payload.clone()))
    }
}

impl Serialize for ExtensionEvent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.payload.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ExtensionEvent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::Object(payload) => {
                let kind = payload
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                Ok(Self { kind, payload })
            }
            other => Err(serde::de::Error::custom(format!(
                "session event must be a JSON object, got {other}"
            ))),
        }
    }
}

/// One append-only session event.
#[derive(Clone, Debug, PartialEq)]
pub enum SessionEvent {
    Core(CoreEvent),
    Extension(ExtensionEvent),
}

impl SessionEvent {
    /// The core event, when this is one.
    pub fn core(&self) -> Option<&CoreEvent> {
        match self {
            Self::Core(event) => Some(event),
            Self::Extension(_) => None,
        }
    }

    /// The extension event, when this is one.
    pub fn extension(&self) -> Option<&ExtensionEvent> {
        match self {
            Self::Core(_) => None,
            Self::Extension(event) => Some(event),
        }
    }

    /// Opens turn `turn`.
    pub fn turn_start(turn: u64) -> Self {
        Self::Core(CoreEvent::TurnStart { turn })
    }

    /// Closes turn `turn` with `reason`.
    pub fn turn_end(turn: u64, reason: TurnEndReason) -> Self {
        Self::Core(CoreEvent::TurnEnd { turn, reason })
    }

    /// Opens step `step` of turn `turn`.
    pub fn step_start(turn: u64, step: u64) -> Self {
        Self::Core(CoreEvent::StepStart { turn, step })
    }

    /// Closes step `step` of turn `turn`.
    pub fn step_end(turn: u64, step: u64) -> Self {
        Self::Core(CoreEvent::StepEnd { turn, step })
    }

    /// A user-role message on the model-visible surface.
    pub fn user_message(message: UserMessage) -> Self {
        Self::Core(CoreEvent::UserMessage {
            message: Box::new(message),
        })
    }

    /// An incremental developer message at the named turn and step.
    pub fn developer_message(
        turn: u64,
        step: u64,
        message: DeveloperMessage,
        header_seq: Option<SessionSeq>,
    ) -> Self {
        Self::Core(CoreEvent::DeveloperMessage {
            turn,
            step,
            message: Box::new(message),
            header_seq,
        })
    }

    /// The rendered system prompt on the model-visible surface.
    pub fn system_message(turn: u64, step: u64, message: SystemMessage) -> Self {
        Self::Core(CoreEvent::SystemMessage {
            turn,
            step,
            message: Box::new(message),
        })
    }

    /// An assembled assistant message for one step.
    pub fn assistant_message(
        turn: u64,
        step: u64,
        message: AssistantMessage,
        stream: Vec<AssistantStreamRecord>,
        usage: Option<TokenUsage>,
        interrupted: bool,
    ) -> Self {
        Self::Core(CoreEvent::AssistantMessage {
            turn,
            step,
            message: Box::new(message),
            stream,
            usage: usage.map(Box::new),
            interrupted,
        })
    }

    /// The model requested one tool invocation.
    pub fn tool_call(
        turn: u64,
        step: u64,
        call_id: ToolCallId,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::Core(CoreEvent::ToolCall {
            turn,
            step,
            call_id,
            name: name.into(),
            arguments: arguments.into(),
        })
    }

    /// A completed tool call's model-facing result.
    pub fn tool_result(turn: u64, step: u64, message: ToolResultMessage) -> Self {
        Self::Core(CoreEvent::ToolResult {
            turn,
            step,
            message: Box::new(message),
            error: None,
            meta: None,
        })
    }

    /// A completed tool call's result with its full durable outcome.
    pub fn tool_result_full(
        turn: u64,
        step: u64,
        message: ToolResultMessage,
        error: Option<super::types::ToolErrorInfo>,
        meta: Option<Value>,
    ) -> Self {
        Self::Core(CoreEvent::ToolResult {
            turn,
            step,
            message: Box::new(message),
            error: error.map(Box::new),
            meta,
        })
    }

    /// A completed tool call that failed; the error is model-facing too.
    pub fn tool_error(
        turn: u64,
        step: u64,
        message: ToolResultMessage,
        error: super::types::ToolErrorInfo,
    ) -> Self {
        Self::tool_result_full(turn, step, message, Some(error), None)
    }

    /// A full request header snapshot.
    pub fn request_header(header: EpochHeader, reason: RequestHeaderReason) -> Self {
        Self::Core(CoreEvent::RequestHeader {
            header: Box::new(header),
            reason,
            starts_series: false,
        })
    }

    /// A full request header snapshot that begins a distinct message series.
    pub fn request_header_series(header: EpochHeader, reason: RequestHeaderReason) -> Self {
        Self::Core(CoreEvent::RequestHeader {
            header: Box::new(header),
            reason,
            starts_series: true,
        })
    }

    /// Route metadata for the next request.
    pub fn request_context(context: RequestContext) -> Self {
        Self::Core(CoreEvent::RequestContext(context))
    }

    /// A plugin-declared event, preserved verbatim.
    pub fn extension_event(kind: impl Into<String>, payload: Map<String, Value>) -> Self {
        Self::Extension(ExtensionEvent::new(kind, payload))
    }
}

impl Serialize for SessionEvent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Core(event) => event.serialize(serializer),
            Self::Extension(event) => event.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for SessionEvent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let tag = value
            .as_object()
            .and_then(|map| map.get("type"))
            .and_then(Value::as_str);
        match tag {
            // Known tags deserialize strictly: a malformed core event is an
            // error, never a silently demoted extension.
            Some(tag) if CORE_EVENT_TAGS.contains(&tag) => {
                serde_json::from_value::<CoreEvent>(value)
                    .map(Self::Core)
                    .map_err(serde::de::Error::custom)
            }
            _ => serde_json::from_value::<ExtensionEvent>(value)
                .map(Self::Extension)
                .map_err(serde::de::Error::custom),
        }
    }
}
