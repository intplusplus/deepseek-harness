//! Merge-extensible content blocks.
//!
//! TypeScript grows `ContentBlockMap` by declaration merging. Rust has no
//! declaration merging, so the same extensibility is modeled as a closed
//! [`CoreContentBlock`] union plus an [`ExtensionBlock`] slot that preserves a
//! plugin-declared block verbatim — consumers fall through unknown block
//! types, exactly as the TS comment requires.

#![forbid(unsafe_code)]

use std::ops::Not;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use super::brand::ToolCallId;

/// The closed core block-type vocabulary.
pub const CORE_CONTENT_BLOCK_TYPES: &[&str] = &[
    "text",
    "reasoning",
    "image",
    "file",
    "tool-call",
    "tool-addition",
    "tool-removal",
];

/// Intrinsic display metadata of a durable image attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageDimensions {
    pub width: u64,
    pub height: u64,
}

/// A durable raster image reference owned by the attachment service.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageAttachmentRef {
    pub attachment_id: String,
    pub media_type: String,
    pub bytes: u64,
    pub width: u64,
    pub height: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_dimensions: Option<ImageDimensions>,
}

/// A durable verbatim file reference owned by the attachment service.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileAttachmentRef {
    pub attachment_id: String,
    pub name: String,
    pub bytes: u64,
}

/// The closed core content-block vocabulary.
///
/// New core blocks must land with adapter, UI, and compaction support, exactly
/// as the TS map requires.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum CoreContentBlock {
    /// Plain text visible to the end user.
    Text { text: String },
    /// Reasoning / thinking content, distinct from visible text.
    Reasoning { text: String },
    /// A durable raster image reference, valid in user or assistant content.
    Image {
        attachment: ImageAttachmentRef,
        /// Derived from a durable image-offload decision or preserved by a
        /// message rewrite; serialized only when true.
        #[serde(default, skip_serializing_if = "Not::not")]
        offloaded: bool,
    },
    /// A durable verbatim file reference, valid in user content.
    File { attachment: FileAttachmentRef },
    /// A tool invocation requested by the model.
    ToolCall {
        id: ToolCallId,
        name: String,
        /// Raw JSON string as produced by the model.
        arguments: String,
    },
    /// Activates a tool definition from the developer event's referenced
    /// request header. The historical header owns the schema.
    ToolAddition { tool_name: String },
    /// Records the dynamic removal of a tool identified by its session-local name.
    ToolRemoval { tool_name: String },
}

/// A plugin-declared content block.
///
/// The `type` tag and payload are preserved verbatim; the core falls through
/// unknown block types instead of rejecting them.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtensionBlock {
    kind: String,
    payload: Map<String, Value>,
}

impl ExtensionBlock {
    /// Create an extension block; `type` is stamped from `kind`.
    pub fn new(kind: impl Into<String>, mut payload: Map<String, Value>) -> Self {
        let kind = kind.into();
        payload.insert("type".to_string(), Value::String(kind.clone()));
        Self { kind, payload }
    }

    /// The block's `type` tag, as declared by the plugin.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The block's verbatim JSON payload, including its `type` tag.
    pub fn payload(&self) -> &Map<String, Value> {
        &self.payload
    }

    /// Interpret the payload as `T`, the plugin-declared shape.
    pub fn decode<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_value(Value::Object(self.payload.clone()))
    }
}

impl Serialize for ExtensionBlock {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.payload.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ExtensionBlock {
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
                "extension block must be a JSON object, got {other}"
            ))),
        }
    }
}

/// A borrowed view of a [`CoreContentBlock::ToolCall`] block.
#[derive(Clone, Copy, Debug)]
pub struct ToolCallRef<'a> {
    /// The call's identity, pairing it with its result.
    pub id: &'a ToolCallId,
    /// The requested tool name.
    pub name: &'a str,
    /// Raw JSON string exactly as the model produced it.
    pub arguments: &'a str,
}

/// One content block: a core block, or a plugin-declared block.
#[derive(Clone, Debug, PartialEq)]
pub enum ContentBlock {
    Core(CoreContentBlock),
    Extension(ExtensionBlock),
}

impl ContentBlock {
    /// Plain text visible to the end user.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Core(CoreContentBlock::Text { text: text.into() })
    }

    /// Reasoning / thinking content.
    pub fn reasoning(text: impl Into<String>) -> Self {
        Self::Core(CoreContentBlock::Reasoning { text: text.into() })
    }

    /// A tool invocation requested by the model.
    pub fn tool_call(
        id: ToolCallId,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::Core(CoreContentBlock::ToolCall {
            id,
            name: name.into(),
            arguments: arguments.into(),
        })
    }

    /// A tool addition recorded on a developer message.
    pub fn tool_addition(tool_name: impl Into<String>) -> Self {
        Self::Core(CoreContentBlock::ToolAddition {
            tool_name: tool_name.into(),
        })
    }

    /// A tool removal recorded on a developer message.
    pub fn tool_removal(tool_name: impl Into<String>) -> Self {
        Self::Core(CoreContentBlock::ToolRemoval {
            tool_name: tool_name.into(),
        })
    }

    /// A plugin-declared block, preserved verbatim.
    pub fn extension(kind: impl Into<String>, payload: Map<String, Value>) -> Self {
        Self::Extension(ExtensionBlock::new(kind, payload))
    }

    /// The text, when this block is a plain text block.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Core(CoreContentBlock::Text { text }) => Some(text),
            _ => None,
        }
    }

    /// The tool invocation, when this block is one.
    pub fn as_tool_call(&self) -> Option<ToolCallRef<'_>> {
        match self {
            Self::Core(CoreContentBlock::ToolCall {
                id,
                name,
                arguments,
            }) => Some(ToolCallRef {
                id,
                name,
                arguments,
            }),
            _ => None,
        }
    }

    /// The block's `type` tag.
    pub fn block_type(&self) -> &str {
        match self {
            Self::Core(CoreContentBlock::Text { .. }) => "text",
            Self::Core(CoreContentBlock::Reasoning { .. }) => "reasoning",
            Self::Core(CoreContentBlock::Image { .. }) => "image",
            Self::Core(CoreContentBlock::File { .. }) => "file",
            Self::Core(CoreContentBlock::ToolCall { .. }) => "tool-call",
            Self::Core(CoreContentBlock::ToolAddition { .. }) => "tool-addition",
            Self::Core(CoreContentBlock::ToolRemoval { .. }) => "tool-removal",
            Self::Extension(block) => block.kind(),
        }
    }
}

impl Serialize for ContentBlock {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Core(block) => block.serialize(serializer),
            Self::Extension(block) => block.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ContentBlock {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let tag = value
            .as_object()
            .and_then(|map| map.get("type"))
            .and_then(Value::as_str);
        match tag {
            // Known tags deserialize strictly: a malformed core block is an
            // error, never a silently demoted extension.
            Some(tag) if CORE_CONTENT_BLOCK_TYPES.contains(&tag) => {
                serde_json::from_value::<CoreContentBlock>(value)
                    .map(Self::Core)
                    .map_err(serde::de::Error::custom)
            }
            _ => serde_json::from_value::<ExtensionBlock>(value)
                .map(Self::Extension)
                .map_err(serde::de::Error::custom),
        }
    }
}
