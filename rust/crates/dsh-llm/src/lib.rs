//! `dsh-llm` — the canonical provider-neutral message and streaming vocabulary
//! for the loop, session log, and plugins.
//!
//! Adapters alone translate provider wire messages; the mapped interfaces here
//! make the content, source, and finish unions extensible. TypeScript
//! declaration merging becomes explicit `Extension` slots on the three
//! extensible unions ([`ContentBlock`], [`MessageSource`]) plus closed core
//! unions for the rest.

#![forbid(unsafe_code)]

pub mod brand;
pub mod call_config;
pub mod content;
pub mod message;
pub mod stream;
pub mod types;

pub use brand::{LlmAttemptId, MessageId, ProviderRequestId, ReasoningEffortId, ToolCallId};
pub use call_config::{call_config_equals, LlmCallConfig, LlmCallConfigAdapterDefaults};
pub use content::{
    ContentBlock, CoreContentBlock, ExtensionBlock, FileAttachmentRef, ImageAttachmentRef,
    ImageDimensions, ToolCallRef, CORE_CONTENT_BLOCK_TYPES,
};
pub use message::{
    AssistantMessage, DeveloperMessage, ExtensionSource, Message, MessageSource, SystemMessage,
    ToolResultMessage, UserMessage, CORE_MESSAGE_SOURCE_KINDS,
};
pub use stream::{AssistantStreamRecord, ReplayEnvelope, StreamChunk, TimedStreamChunk};
pub use types::{
    FinishReason, LlmFailure, ModelModality, TokenUsage, ToolSchema, EMPTY_RESPONSE_CODE,
};
