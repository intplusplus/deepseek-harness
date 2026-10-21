//! Session-reference inputs, candidates, rendered data, and durable wire
//! records.
//!
//! Field order follows the TypeScript declarations: `serde` serializes struct
//! members in declaration order, and the workspace enables `preserve_order`
//! for `serde_json` maps, so a source record round-trips with the same key
//! order as a TypeScript-written session log.

#![forbid(unsafe_code)]

use std::fmt;

use dsh_llm::{ContentBlock, ExtensionSource, MessageSource, UserMessage};
use dsh_session::SessionId;
use serde::{Deserialize, Serialize};

use super::config::SessionReferenceError;

/// An opaque source session id.
///
/// The TypeScript `SessionId` accepts every string, including an empty one.
/// The Rust [`SessionId`] rejects empty ids, so mention parsing and source
/// snapshots keep their own unbranded value and convert only where a real
/// session identity is required.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionReferenceId(String);

impl SessionReferenceId {
    /// Brand an opaque session id. No validation, matching the TS constructor.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the id, yielding the raw string.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for SessionReferenceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for SessionReferenceId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<SessionId> for SessionReferenceId {
    fn from(id: SessionId) -> Self {
        Self(id.as_str().to_string())
    }
}

impl TryFrom<SessionReferenceId> for SessionId {
    type Error = dsh_session::SessionError;

    fn try_from(id: SessionReferenceId) -> Result<Self, Self::Error> {
        Self::new(id.0)
    }
}

/// One source session selected by a host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionReferenceInput {
    /// Opaque source session identity.
    pub session_id: SessionReferenceId,
    /// Optional user-facing mention label.
    pub label: Option<String>,
}

impl SessionReferenceInput {
    /// A reference without an explicit label.
    pub fn new(session_id: SessionReferenceId) -> Self {
        Self {
            session_id,
            label: None,
        }
    }

    /// A reference with an explicit display label.
    pub fn with_label(session_id: SessionReferenceId, label: impl Into<String>) -> Self {
        Self {
            session_id,
            label: Some(label.into()),
        }
    }
}

/// A reference after self-checking, deduplication, and label defaulting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedReference {
    /// Source session identity.
    pub session_id: SessionReferenceId,
    /// Display label; defaults to the session id.
    pub label: String,
}

/// One host-facing candidate from exact session metadata.
///
/// The wire order matches TypeScript: `sessionId`, `label`, optional
/// `displayTitle`, optional `cwd`, `sameWorkspace`, `createdAt`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionReferenceCandidate {
    /// Opaque source session identity.
    pub session_id: SessionReferenceId,
    /// Latest log-backed title, falling back to the opaque session id.
    pub label: String,
    /// Presentation title, preferring a subagent's durable creation label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_title: Option<String>,
    /// Source session working directory, when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Whether the source cwd is recorded and equals the requesting workspace.
    pub same_workspace: bool,
    /// Source session creation time in Unix epoch milliseconds.
    pub created_at: u64,
}

/// One discovery candidate carrying its canonical prompt mention.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionReferenceMentionCandidate {
    /// Opaque source session identity.
    pub session_id: SessionReferenceId,
    /// Latest log-backed title, falling back to the opaque session id.
    pub label: String,
    /// Presentation title, preferring a subagent's durable creation label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_title: Option<String>,
    /// Source session working directory, when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Whether the source cwd is recorded and equals the requesting workspace.
    pub same_workspace: bool,
    /// Source session creation time in Unix epoch milliseconds.
    pub created_at: u64,
    /// Canonical `@[label](dsh-session:…)` mention for the prompt draft.
    pub mention: String,
}

impl SessionReferenceMentionCandidate {
    /// Attach a canonical mention to one listed candidate.
    pub fn from_candidate(
        candidate: SessionReferenceCandidate,
        mention: impl Into<String>,
    ) -> Self {
        Self {
            session_id: candidate.session_id,
            label: candidate.label,
            display_title: candidate.display_title,
            cwd: candidate.cwd,
            same_workspace: candidate.same_workspace,
            created_at: candidate.created_at,
            mention: mention.into(),
        }
    }
}

/// Original message role in a projected referenced conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReferencedConversationRole {
    /// A user-role message from a direct prompt or compaction checkpoint.
    User,
    /// An assistant text message.
    Assistant,
}

impl ReferencedConversationRole {
    /// The wire role name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// Text-only projected conversation item.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferencedConversationItem {
    /// Original message role.
    pub role: ReferencedConversationRole,
    /// Visible text retained from that message.
    pub text: String,
}

/// Snapshot data serialized inside the untrusted prompt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReferencedSessionData {
    /// Source session identity.
    pub session_id: SessionReferenceId,
    /// Display label supplied by the host.
    pub label: String,
    /// Source session working directory, serialized as `null` when absent.
    pub cwd: Option<String>,
    /// Inclusive watermark of the captured event, serialized as `null` before
    /// any event exists.
    pub captured_through_seq: Option<u64>,
    /// Projected user and assistant text in source order.
    pub conversation: Vec<ReferencedConversationItem>,
}

/// Retention facts stored beside the durable context.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReferenceRetentionStats {
    /// Whether a compaction checkpoint appeared in the projected source.
    pub compacted: bool,
    /// Messages observed in the source projection.
    pub original_messages: u64,
    /// Messages present in the retained preview.
    pub retained_messages: u64,
    /// Whole messages dropped to fit the budget.
    pub omitted_messages: u64,
    /// UTF-8 text bytes dropped from whole messages or message tails.
    pub omitted_bytes: u64,
    /// Whether any message or byte was omitted.
    pub truncated: bool,
}

/// One projection fitted to the byte budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedReference {
    /// Preview serialized into the prompt.
    pub data: ReferencedSessionData,
    /// Complete projection kept for the retrievable transcript.
    pub full_data: ReferencedSessionData,
    /// Retention facts shared by the preview and transcript.
    pub stats: ReferenceRetentionStats,
}

/// The `kind` discriminator of a rendered reference source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionReferenceSourceKind {
    /// Material lifted out of another session's log.
    #[serde(rename = "session-reference")]
    SessionReference,
}

/// The context form of a rendered reference source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionReferenceForm {
    /// Recalled cross-session snapshot.
    #[serde(rename = "recall")]
    Recall,
}

/// One cited source in the durable reference source record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionReferenceSourceEntry {
    /// Source session identity.
    pub session_id: SessionReferenceId,
    /// Display label serialized with the snapshot.
    pub label: String,
    /// Source Session format generation; absence identifies version 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured_format_version: Option<u32>,
    /// Inclusive source event watermark, serialized as `null` when absent.
    pub captured_through_seq: Option<u64>,
    /// Whether the projection started from a compaction checkpoint.
    pub compacted: bool,
    /// Messages observed in the source projection.
    pub original_messages: u64,
    /// Messages present in the retained preview.
    pub retained_messages: u64,
    /// Whole messages dropped to fit the budget.
    pub omitted_messages: u64,
    /// UTF-8 text bytes omitted from the preview.
    pub omitted_bytes: u64,
    /// Whether the preview is incomplete.
    pub truncated: bool,
    /// Position of this reference in the source message.
    pub input_index: u64,
}

impl SessionReferenceSourceEntry {
    /// Build one entry from a retained preview and its capture version.
    pub fn new(index: usize, captured_format_version: u32, retained: &RetainedReference) -> Self {
        Self {
            session_id: retained.data.session_id.clone(),
            label: retained.data.label.clone(),
            captured_format_version: Some(captured_format_version),
            captured_through_seq: retained.data.captured_through_seq,
            compacted: retained.stats.compacted,
            original_messages: retained.stats.original_messages,
            retained_messages: retained.stats.retained_messages,
            omitted_messages: retained.stats.omitted_messages,
            omitted_bytes: retained.stats.omitted_bytes,
            truncated: retained.stats.truncated,
            input_index: index as u64,
        }
    }
}

/// Durable source session, cited event seqs, and snapshot facts for prepared
/// cross-session context.
///
/// Wire order is `kind`, `form`, `version`, `references`, matching the
/// TypeScript declaration a session log carries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionReferenceSource {
    /// Source discriminator.
    pub kind: SessionReferenceSourceKind,
    /// Context form.
    pub form: SessionReferenceForm,
    /// Record version.
    pub version: u32,
    /// One entry per distinct referenced session, in citation order.
    pub references: Vec<SessionReferenceSourceEntry>,
}

impl SessionReferenceSource {
    /// Build a version-1 recall source from retained references.
    pub fn new(captured: &[(u32, RetainedReference)]) -> Self {
        Self {
            kind: SessionReferenceSourceKind::SessionReference,
            form: SessionReferenceForm::Recall,
            version: 1,
            references: captured
                .iter()
                .enumerate()
                .map(|(index, (version, retained))| {
                    SessionReferenceSourceEntry::new(index, *version, retained)
                })
                .collect(),
        }
    }

    /// Convert the source to the merge-extensible message source slot.
    ///
    /// Serialization fixes the field order; the extension preserves that map
    /// verbatim, including its leading `kind` key.
    ///
    /// # Errors
    ///
    /// Returns [`SessionReferenceError::NotJsonSerializable`] when the source
    /// cannot be represented as JSON.
    pub fn to_extension_source(&self) -> Result<ExtensionSource, SessionReferenceError> {
        let value =
            serde_json::to_value(self).map_err(|_| SessionReferenceError::NotJsonSerializable)?;
        let serde_json::Value::Object(fields) = value else {
            return Err(SessionReferenceError::NotJsonSerializable);
        };
        Ok(ExtensionSource::new("session-reference", fields))
    }

    /// Build the tenant user message that carries the aggregated snapshot.
    ///
    /// `message_id` is injected because this crate never reads randomness.
    ///
    /// # Errors
    ///
    /// Returns [`SessionReferenceError::NotJsonSerializable`] when the source
    /// cannot be represented as JSON.
    pub fn to_user_message(
        &self,
        message_id: dsh_llm::MessageId,
        prompt: impl Into<String>,
    ) -> Result<UserMessage, SessionReferenceError> {
        let source = self.to_extension_source()?;
        Ok(UserMessage::with_source(
            message_id,
            vec![ContentBlock::text(prompt)],
            MessageSource::Extension(source),
        ))
    }
}

/// Direct message content and optional referenced-session context.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedReferencedMessage {
    /// Readable message content after host mention tokens are removed.
    pub content: Vec<ContentBlock>,
    /// Aggregated untrusted snapshot, absent when the message has no
    /// references.
    pub additional_context: Option<UserMessage>,
}
