//! Storage-level session types: identity, sequence branding, the header, and
//! the closed vocabularies the event log references.
//!
//! Branded numbers keep the TypeScript contract visible: every sequence,
//! offset, and watermark is a non-negative safe integer, admitted at the
//! boundary rather than trusted from the wire.

#![forbid(unsafe_code)]

use std::fmt;

use serde::{Deserialize, Serialize};

use dsh_llm::{LlmCallConfig, LlmCallConfigAdapterDefaults, LlmFailure, ToolSchema};

/// `Number.MAX_SAFE_INTEGER` — the bound every branded number admits.
pub const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

/// Current logical session format version, stamped into every newly written
/// [`SessionHeader`].
///
/// A single monotonic integer with no major/minor split. Readers accept only
/// this value; a stored generation below it requires the migration chain, and
/// one above it is refused as a future format.
pub const SESSION_FORMAT_VERSION: u32 = 4;

/// Identifies one session in the store (and its persistence artifacts).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    /// Brand a session id. Empty ids are rejected.
    pub fn new(id: impl Into<String>) -> Result<Self, SessionError> {
        let id = id.into();
        if id.is_empty() {
            return Err(SessionError::EmptySessionId);
        }
        Ok(Self(id))
    }

    /// The branded session id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for SessionId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

macro_rules! branded_number {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Admit a numeric value as an existing position; values beyond the
            /// safe-integer bound are rejected.
            pub fn new(value: u64) -> Result<Self, SessionError> {
                if value > MAX_SAFE_INTEGER {
                    return Err(SessionError::UnsafeInteger { value, max: MAX_SAFE_INTEGER });
                }
                Ok(Self(value))
            }

            /// The branded value.
            pub fn get(self) -> u64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

branded_number!(
    /// Sequence number of one existing event in a session log.
    SessionSeq
);

branded_number!(
    /// A session log gap, prefix length, or read offset, which may equal the
    /// event count.
    SessionLogOffset
);

impl SessionSeq {
    /// Admit a value the caller already guarantees is in range (log positions
    /// derived from a vector length, for example).
    pub(crate) fn unchecked(value: u64) -> Self {
        Self(value)
    }
}

/// Inclusive session event watermark, or `None` before any event exists
/// (the TypeScript `-1` cursor).
pub type SessionSeqCursor = Option<SessionSeq>;

/// One existing session event position, or explicit absence.
pub type OptionalSessionSeq = Option<SessionSeq>;

/// Coarse product classification for a session created as a subagent child.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SessionOrigin {
    Subagent,
}

/// Immutable validated storage metadata, kept outside the conversation event log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    /// Current logical format version, stamped from [`SESSION_FORMAT_VERSION`].
    pub version: u32,
    /// The session's id.
    pub id: SessionId,
    /// Non-negative safe-integer Unix epoch milliseconds of creation.
    pub created_at: u64,
    /// Absolute working directory the session was created in (if any).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// The session this one was forked from (seed lineage), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<SessionId>,
    /// Whether this session contains a fork-inherited event prefix.
    pub is_seeded: bool,
    /// Coarse product classification for a subagent child.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<SessionOrigin>,
    /// Delegation depth: absent (zero) for a top-level session, parent depth
    /// plus one for a subagent child. Persisted so a recursion budget survives
    /// restart and resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delegation_depth: Option<u32>,
    /// Id of the agent preset this session's agent was composed from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_preset: Option<String>,
}

/// Caller-supplied storage fields the store folds into a [`SessionHeader`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMeta {
    pub cwd: Option<String>,
    pub parent_session: Option<SessionId>,
    pub created_at: Option<u64>,
    pub is_seeded: bool,
    pub origin: Option<SessionOrigin>,
    pub delegation_depth: Option<u32>,
    pub agent_preset: Option<String>,
}

/// Why an active agent driver was cancelled.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum AgentCancelCause {
    User,
    Parent,
    Hook { reason: String },
    Disposed,
}

/// Durable cancellation cause, including imports whose original coarse record
/// carried no cause.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum TurnEndCancelCause {
    User,
    Parent,
    Hook { reason: String },
    Disposed,
    Legacy,
}

/// Why a turn ended. Merge-extensible sum type in TypeScript; the core members
/// live here and plugin members land in the event extension slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum TurnEndReason {
    Completed,
    Aborted {
        reason: TurnEndCancelCause,
    },
    Blocked,
    /// The turn failed; `error` is always a structured failure.
    ///
    /// Boxed: `LlmFailure` is ~104 bytes and `turn/end` is one of the most
    /// frequent events, so keeping it inline would tax every one of them.
    Error {
        error: Box<LlmFailure>,
    },
    /// At least one step reached its output-token ceiling.
    MaxTokens,
    /// A crash-orphaned turn was closed after the fact.
    Interrupted,
    /// Fork-seed construction closed a turn that was still open at the fork
    /// boundary in the source session.
    Forked,
}

impl TurnEndReason {
    /// The turn failed with a structured failure.
    pub fn error(error: LlmFailure) -> Self {
        Self::Error {
            error: Box::new(error),
        }
    }
}

/// Why a `request/header` snapshot was logged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RequestHeaderReason {
    Initial,
    Resume,
    Change,
    Series,
}

/// How a route admits changed system prompt text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SystemPromptUpdate {
    InHistory,
    Leading,
}

/// Logged request state outside derived history: call config and tools.
///
/// The system prompt is derived history — a `system/message` event — so this
/// type has no `system` field at all: the canonical form is unrepresentable
/// rather than rejected after the fact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpochHeader {
    /// The conversation's call configuration.
    pub config: LlmCallConfig,
    /// Effective config fields materialized from the exact adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_defaults: Option<LlmCallConfigAdapterDefaults>,
    /// Assembled tool schemas; absent for a tool-less request. An empty list
    /// is non-canonical and rejected by the log.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSchema>>,
}

/// Registration-bound metadata for one resolved model route.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestContext {
    /// Registered provider route the metadata belongs to.
    pub provider: String,
    /// Provider-owned model id the metadata belongs to.
    pub model: String,
    /// Maximum combined request and response context in tokens, when advertised.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// How the route admits changed system prompt text; absent clears an older
    /// route's mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt_update: Option<SystemPromptUpdate>,
}

/// Optional failure identity and raw user-facing reason, outside model content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolErrorInfo {
    pub name: String,
    pub code: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Errors raised by session identity, admission, and persistence.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("session id must not be empty")]
    EmptySessionId,
    #[error("value {value} exceeds the safe-integer bound {max}")]
    UnsafeInteger { value: u64, max: u64 },
    #[error("event data is not lossless JSON: {0}")]
    NotJson(String),
    #[error("non-canonical request header: {0}")]
    NonCanonicalHeader(String),
    #[error("legacy event type `{0}` is rejected; v0 logs must be migrated first")]
    LegacyEvent(String),
    #[error("session format version {found} is newer than the supported version {current}")]
    UnsupportedVersion { found: u32, current: u32 },
    #[error(
        "stored generation v{found} predates the current format v{current}; the migration chain is required"
    )]
    MigrationRequired { found: u32, current: u32 },
    #[error("stored log is empty")]
    EmptyLog,
    #[error("no stored session generation in directory")]
    NoStoredGeneration,
    #[error("io error: {0}")]
    Io(String),
    #[error("json error: {0}")]
    Json(String),
}
