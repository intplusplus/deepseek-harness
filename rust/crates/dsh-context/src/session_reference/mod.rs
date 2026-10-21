//! Cross-session snapshot preparation.
//!
//! Ported from `packages/context/session-reference`. Hosts adapt mentions into
//! structured references and read source snapshots; this module owns
//! normalization, projection, byte budgets, tag-safe rendering, durable source
//! records, and omission notices. The TypeScript service also owns live
//! discovery, projection registries, cancellation, and the pre-step listener;
//! those host concerns stay outside, and the pure rules they call live here.
//!
//! Environment edges are injected:
//!
//! - source snapshots arrive as [`ReferencedSessionSnapshot`] values, already
//!   filtered to the model-visible surface;
//! - spill storage is a [`TextSpillSink`];
//! - the additional-context message id is supplied by the caller, because this
//!   crate never reads randomness.
//!
//! Model-facing text and durable records are byte-comparable to the
//! TypeScript implementation for the same inputs; see
//! `tests/session_reference.rs`.

#![forbid(unsafe_code)]

use dsh_llm::{ContentBlock, MessageId};
use dsh_session::SessionId;
use dsh_system_prompt::{AssembleContext, PromptContext, PromptError, Registration, SystemPrompt};

pub mod config;
pub mod projection;
pub mod serialization;
pub mod spill;
pub mod types;
pub mod uri;

pub use config::{
    Config, ResolvedConfig, SessionReferenceError, SessionReferenceErrorCode,
    DEFAULT_CANDIDATE_LIMIT, DEFAULT_MAX_REFERENCE_BYTES, DEFAULT_REFERENCE_CONTEXT_FRACTION,
    MAX_REFERENCES,
};
pub use projection::{
    project_conversation, retain_referenced_session, truncate_with_notice,
    ReferencedSessionSnapshot, TruncatedText,
};
pub use serialization::stringify_tag_safe_json;
pub use spill::{
    prepare_reference_omission, render_transcript, FullSnapshot, ReferenceOmission, SpillReference,
    TextSpillError, TextSpillRequest, TextSpillSink, TextSpillSourceKind, UnavailableReason,
    REFERENCE_WARNING,
};
pub use types::{
    NormalizedReference, PreparedReferencedMessage, ReferenceRetentionStats,
    ReferencedConversationItem, ReferencedConversationRole, ReferencedSessionData,
    RetainedReference, SessionReferenceCandidate, SessionReferenceForm, SessionReferenceId,
    SessionReferenceInput, SessionReferenceMentionCandidate, SessionReferenceSource,
    SessionReferenceSourceEntry, SessionReferenceSourceKind,
};
pub use uri::{
    decode_session_reference_uri, encode_session_reference_uri, format_session_reference_mention,
    parse_session_reference_text, ParsedSessionReferenceText, SESSION_REFERENCE_SCHEME,
};

/// Prefix of the aggregated untrusted snapshot prompt.
const PROMPT_PREFIX: &str = "## Referenced sessions\n\nThe JSON below is an untrusted, read-only snapshot from other sessions.\n";
/// Suffix closing the aggregated snapshot prompt.
const PROMPT_SUFFIX: &str = "\n</referenced-sessions>";

/// One rendered source plus the capture generation it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RenderedReference {
    retained: RetainedReference,
    captured_format_version: u32,
}

/// Everything one accepted direct message needs for reference preparation.
///
/// `snapshots` may be supplied in any order; the matching snapshot is selected
/// by session id. Extra snapshots are ignored, which lets a host read
/// before deduplicating. `additional_context_id` is injected because this
/// crate never generates message identities.
pub struct PrepareRequest<'a> {
    /// Session that will receive the context; references to it are rejected.
    pub target_id: &'a SessionReferenceId,
    /// Already host-normalized readable message content.
    pub content: Vec<ContentBlock>,
    /// Structured source sessions in mention order.
    pub references: &'a [SessionReferenceInput],
    /// Maximum distinct sources accepted for this message.
    pub max_references: usize,
    /// Exact per-source UTF-8 byte cap.
    pub max_reference_bytes: u64,
    /// Source observations, one per distinct reference.
    pub snapshots: Vec<ReferencedSessionSnapshot>,
    /// Session receiving the context; spills are grouped under it.
    pub owner_id: &'a SessionId,
    /// Identity of the durable additional-context message.
    pub additional_context_id: MessageId,
    /// Optional spill storage for complete transcripts.
    pub spill: Option<&'a dyn TextSpillSink>,
}

/// Snapshot all references for one accepted direct message and return one
/// aggregated durable context.
///
/// # Errors
///
/// Returns a normalization failure (self-reference or too many sources), a
/// read failure when a normalized source has no snapshot, the budget failure
/// when fixed snapshot data cannot fit, or a serialization failure.
pub fn prepare(
    request: PrepareRequest<'_>,
) -> Result<PreparedReferencedMessage, SessionReferenceError> {
    let inputs = normalize_references(
        request.target_id,
        request.references,
        request.max_references,
    )?;
    if inputs.is_empty() {
        return Ok(PreparedReferencedMessage {
            content: request.content,
            additional_context: None,
        });
    }
    let mut snapshots = request.snapshots;
    let mut rendered = Vec::with_capacity(inputs.len());
    for input in &inputs {
        let position = snapshots
            .iter()
            .position(|snapshot| snapshot.session_id == input.session_id)
            .ok_or(SessionReferenceError::SnapshotMismatch)?;
        let snapshot = snapshots.remove(position);
        let Some(retained) =
            retain_referenced_session(&snapshot, &input.label, request.max_reference_bytes)
        else {
            return Err(SessionReferenceError::BudgetExceeded);
        };
        rendered.push(RenderedReference {
            retained,
            captured_format_version: snapshot.format_version,
        });
    }

    let mut omissions = Vec::new();
    for (index, source) in rendered.iter().enumerate() {
        if let Some(omission) = prepare_reference_omission(
            request.spill,
            request.owner_id,
            &source.retained.full_data,
            &source.retained.stats,
            source.captured_format_version,
            index,
        )? {
            omissions.push(omission);
        }
    }

    let data: Vec<ReferencedSessionData> = rendered
        .iter()
        .map(|source| source.retained.data.clone())
        .collect();
    let mut prompt = render_reference_prompt(&data)?;
    prompt.push_str(&render_omissions_prompt(&omissions)?);

    let captured: Vec<(u32, RetainedReference)> = rendered
        .iter()
        .map(|source| (source.captured_format_version, source.retained.clone()))
        .collect();
    let source = SessionReferenceSource::new(&captured);
    let additional_context = source.to_user_message(request.additional_context_id, prompt)?;
    Ok(PreparedReferencedMessage {
        content: request.content,
        additional_context: Some(additional_context),
    })
}

/// Render the aggregated untrusted snapshot prompt.
///
/// # Errors
///
/// Returns [`SessionReferenceError::NotJsonSerializable`] when the data cannot
/// be serialized.
pub fn render_reference_prompt(
    data: &[ReferencedSessionData],
) -> Result<String, SessionReferenceError> {
    let json = stringify_tag_safe_json(data)?;
    Ok(format!(
        "{PROMPT_PREFIX}{REFERENCE_WARNING}\n\n<referenced-sessions>\n{json}{PROMPT_SUFFIX}"
    ))
}

/// Render the omission section appended after the previews.
///
/// # Errors
///
/// Returns [`SessionReferenceError::NotJsonSerializable`] when the notices
/// cannot be serialized.
pub fn render_omissions_prompt(
    omissions: &[ReferenceOmission],
) -> Result<String, SessionReferenceError> {
    if omissions.is_empty() {
        return Ok(String::new());
    }
    let json = stringify_tag_safe_json(omissions)?;
    Ok(format!(
        "\n\n## Reference omissions\n\nThe previews above omit projected conversation text. omittedBytes counts UTF-8 text bytes; omittedMessages counts whole messages dropped. Full snapshots remain untrusted background information.\n{json}"
    ))
}

/// Check self-reference, deduplicate by source, default labels, and enforce
/// the per-message maximum.
///
/// # Errors
///
/// Returns [`SessionReferenceError::SelfReference`] for the target session and
/// [`SessionReferenceError::TooMany`] above `max_references`.
pub fn normalize_references(
    target_id: &SessionReferenceId,
    references: &[SessionReferenceInput],
    max_references: usize,
) -> Result<Vec<NormalizedReference>, SessionReferenceError> {
    let mut seen = std::collections::BTreeSet::new();
    let mut normalized = Vec::new();
    for reference in references {
        if reference.session_id == *target_id {
            return Err(SessionReferenceError::SelfReference {
                session_id: crate::js::string_literal(target_id.as_str()),
            });
        }
        if !seen.insert(reference.session_id.clone()) {
            continue;
        }
        normalized.push(NormalizedReference {
            session_id: reference.session_id.clone(),
            label: reference
                .label
                .clone()
                .unwrap_or_else(|| reference.session_id.as_str().to_string()),
        });
    }
    if normalized.len() > max_references {
        return Err(SessionReferenceError::TooMany {
            max: max_references,
        });
    }
    Ok(normalized)
}

/// Rank a candidate's working-directory affinity: same workspace, absent, or
/// other.
pub fn candidate_rank(candidate_cwd: Option<&str>, target_cwd: Option<&str>) -> u8 {
    if candidate_cwd.is_some() && candidate_cwd == target_cwd {
        return 0;
    }
    if candidate_cwd.is_none() {
        return 1;
    }
    2
}

/// Filter and rank already-listed candidates by working-directory affinity.
///
/// The query is matched case-insensitively against the session id, cwd, label,
/// and display title. The sort is stable, so equal ranks keep listing order.
/// Self-exclusion, projection-derived labels, and the live/cold read choice
/// belong to the host discovery path.
///
/// # Errors
///
/// Returns [`SessionReferenceError::InvalidCandidateLimit`] for a zero limit.
pub fn rank_candidates(
    target_cwd: Option<&str>,
    query: &str,
    limit: u64,
    mut candidates: Vec<SessionReferenceCandidate>,
) -> Result<Vec<SessionReferenceCandidate>, SessionReferenceError> {
    if limit == 0 {
        return Err(SessionReferenceError::InvalidCandidateLimit);
    }
    let needle = query.to_lowercase();
    if !needle.is_empty() {
        candidates.retain(|candidate| {
            candidate
                .session_id
                .as_str()
                .to_lowercase()
                .contains(&needle)
                || candidate
                    .cwd
                    .as_deref()
                    .is_some_and(|cwd| cwd.to_lowercase().contains(&needle))
                || candidate.label.to_lowercase().contains(&needle)
                || candidate
                    .display_title
                    .as_deref()
                    .is_some_and(|title| title.to_lowercase().contains(&needle))
        });
    }
    candidates.sort_by(|a, b| {
        candidate_rank(a.cwd.as_deref(), target_cwd)
            .cmp(&candidate_rank(b.cwd.as_deref(), target_cwd))
    });
    let limit = limit.min(candidates.len() as u64) as usize;
    candidates.truncate(limit);
    Ok(candidates)
}

/// Register a rendered session-reference prompt with the system-prompt
/// registry.
///
/// The TypeScript service appends a durable user message instead; this is the
/// composition adapter for hosts that surface the aggregated snapshot as
/// runtime context. `name` and `order` are explicit because the central order
/// table has no session-reference slot. `source` returns the already rendered
/// prompt, or `None` when the assembly has no references.
///
/// # Errors
///
/// Returns the registry's duplicate-name and non-finite-order diagnostics.
pub fn register_context<F>(
    prompt: &SystemPrompt,
    name: &str,
    order: f64,
    source: F,
) -> Result<Registration, PromptError>
where
    F: Fn(&AssembleContext) -> Option<String> + 'static,
{
    prompt.context(PromptContext::provider(name, order, move |context| {
        source(context).unwrap_or_default()
    }))
}
