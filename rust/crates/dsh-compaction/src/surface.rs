//! Pure replay of the model-visible session surface.
//!
//! The Rust core log does not carry TypeScript's `surfaceOp` marker, so this
//! fold reconstructs positional replacements from the documented adjacency
//! contract instead:
//!
//! * a `compaction/summary` is effective only when the immediately following
//!   event is a `user/message` whose checkpoint source names the summary's
//!   `compactionId`;
//! * a `compaction/prune` is effective only when the immediately following
//!   event is a `tool/result`.
//!
//! An effective replacement removes its `shadowedSeqs` from the current
//! surface and inserts the replacement node at the removed span's position.
//! Because replacements land fresh high-sequence nodes at older positions, a
//! later `shadowedRange` can have `start > end`; membership is always decided by
//! surface position, never by numeric sequence order.

#![forbid(unsafe_code)]

use dsh_session::{CoreEvent, SessionEvent, SessionSeq};

use super::checkpoint::CompactionCheckpointSource;
use super::error::SurfaceError;
use super::events::{CompactionEvent, ShadowedRange};

/// What one model-visible surface node contributes to pairing and selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceNodeKind {
    /// A rendered system prompt.
    System,
    /// An incremental developer message.
    Developer,
    /// A user-role message.
    User,
    /// An assistant message carrying `tool_calls` open calls.
    Assistant {
        /// Number of `tool-call` blocks in the message.
        tool_calls: usize,
    },
    /// A tool result answering one open call.
    Tool,
}

/// One current surface position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceNode {
    seq: SessionSeq,
    kind: SurfaceNodeKind,
}

impl SurfaceNode {
    /// Create one surface node at its log position.
    pub fn new(seq: SessionSeq, kind: SurfaceNodeKind) -> Self {
        Self { seq, kind }
    }

    /// The event sequence this node names.
    pub fn seq(&self) -> SessionSeq {
        self.seq
    }

    /// What this node contributes to pairing and selection.
    pub fn kind(&self) -> SurfaceNodeKind {
        self.kind
    }
}

/// One applied positional replacement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceReplacement {
    /// The metering event that priced the replacement (`compaction/summary` or
    /// `compaction/prune`).
    pub metering_seq: SessionSeq,
    /// The replacement message event.
    pub seq: SessionSeq,
    /// Declared first shadowed surface sequence.
    pub start: SessionSeq,
    /// Declared last shadowed surface sequence.
    pub end: SessionSeq,
    /// Actual surface entries removed, in surface order.
    pub shadowed_seqs: Vec<SessionSeq>,
    /// Heuristic price recorded by the metering event.
    pub shadowed_token_count: u64,
}

/// Current surface plus the positional replacements that produced it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SurfaceProjection {
    nodes: Vec<SurfaceNode>,
    replacements: Vec<SurfaceReplacement>,
}

impl SurfaceProjection {
    /// Current message-producing sequences, in model-visible order.
    pub fn nodes(&self) -> &[SurfaceNode] {
        &self.nodes
    }

    /// Applied replacements, in log order.
    pub fn replacements(&self) -> &[SurfaceReplacement] {
        &self.replacements
    }

    /// Current surface position of `seq`, or `None` when it is not visible.
    pub fn position_of(&self, seq: SessionSeq) -> Option<usize> {
        self.nodes.iter().position(|node| node.seq() == seq)
    }

    /// Whether `seq` is a current surface node.
    pub fn contains(&self, seq: SessionSeq) -> bool {
        self.position_of(seq).is_some()
    }

    /// Monotonic count of applied positional replacements.
    pub fn replace_generation(&self) -> usize {
        self.replacements.len()
    }

    /// Sum of the heuristic prices recorded by every applied replacement.
    pub fn shadowed_token_count(&self) -> u64 {
        self.replacements.iter().fold(0_u64, |total, replacement| {
            total.saturating_add(replacement.shadowed_token_count)
        })
    }
}

/// Replay a complete session log into its current compaction-aware surface.
///
/// A known `compaction/*` tag with a malformed payload fails the fold. An
/// unrelated tag is ignored. A summary or prune whose contractual replacement
/// is not adjacent remains a log-only record and changes no surface position.
pub fn fold_surface(events: &[SessionEvent]) -> Result<SurfaceProjection, SurfaceError> {
    let mut nodes = Vec::new();
    let mut replacements = Vec::new();
    let mut index = 0;

    while index < events.len() {
        let event = &events[index];
        if let Some(decoded) = CompactionEvent::decode(event) {
            let decoded = decoded?;
            match decoded {
                CompactionEvent::Summary(summary) => {
                    if let Some(replacement_seq) = checkpoint_replacement(events, index, &summary)?
                    {
                        apply_replacement(
                            &mut nodes,
                            &mut replacements,
                            &summary.shadowed_seqs,
                            summary.shadowed_range,
                            summary.shadowed_token_count,
                            sequence(index)?,
                            replacement_seq,
                            SurfaceNodeKind::User,
                        )?;
                        index += 2;
                        continue;
                    }
                }
                CompactionEvent::Prune(prune) => {
                    if matches!(
                        events.get(index + 1),
                        Some(SessionEvent::Core(CoreEvent::ToolResult { .. }))
                    ) {
                        if prune.shadowed_seqs.len() != 1 {
                            return Err(SurfaceError::PruneSpanNotSingle);
                        }
                        let replacement_seq = sequence(index + 1)?;
                        validate_prune_rewrite(events, prune.shadowed_seqs[0], replacement_seq)?;
                        apply_replacement(
                            &mut nodes,
                            &mut replacements,
                            &prune.shadowed_seqs,
                            prune.shadowed_range,
                            prune.shadowed_token_count,
                            sequence(index)?,
                            replacement_seq,
                            SurfaceNodeKind::Tool,
                        )?;
                        index += 2;
                        continue;
                    }
                }
                CompactionEvent::Start(_) | CompactionEvent::End(_) => {}
            }
        }

        if let Some(kind) = node_kind(event) {
            nodes.push(SurfaceNode::new(sequence(index)?, kind));
        }
        index += 1;
    }

    Ok(SurfaceProjection {
        nodes,
        replacements,
    })
}

/// The replacement event sequence when `summary` is immediately followed by
/// its matching checkpoint user message.
fn checkpoint_replacement(
    events: &[SessionEvent],
    summary_index: usize,
    summary: &super::events::CompactionSummary,
) -> Result<Option<SessionSeq>, SurfaceError> {
    let Some(SessionEvent::Core(CoreEvent::UserMessage { message })) =
        events.get(summary_index + 1)
    else {
        return Ok(None);
    };
    let Some(source) = CompactionCheckpointSource::from_message_source(&message.source) else {
        return Ok(None);
    };
    if source.compaction_id == summary.compaction_id {
        Ok(Some(sequence(summary_index + 1)?))
    } else {
        Ok(None)
    }
}

/// Validate that a prune replacement changed only the selected tool result's
/// content, matching the canonical surface replacement rule.
fn validate_prune_rewrite(
    events: &[SessionEvent],
    original_seq: SessionSeq,
    replacement_seq: SessionSeq,
) -> Result<(), SurfaceError> {
    let original = events
        .get(original_seq.get() as usize)
        .ok_or(SurfaceError::MissingSurfaceEvent { seq: original_seq })?;
    let replacement =
        events
            .get(replacement_seq.get() as usize)
            .ok_or(SurfaceError::MissingSurfaceEvent {
                seq: replacement_seq,
            })?;

    let (
        SessionEvent::Core(CoreEvent::ToolResult {
            turn: original_turn,
            step: original_step,
            message: original_message,
            error: original_error,
            meta: original_meta,
        }),
        SessionEvent::Core(CoreEvent::ToolResult {
            turn: replacement_turn,
            step: replacement_step,
            message: replacement_message,
            error: replacement_error,
            meta: replacement_meta,
        }),
    ) = (original, replacement)
    else {
        return Err(SurfaceError::MissingToolResult { seq: original_seq });
    };

    if original_turn != replacement_turn
        || original_step != replacement_step
        || original_error != replacement_error
        || original_meta != replacement_meta
    {
        return Err(SurfaceError::PruneRewriteMismatch);
    }

    let mut original_message = original_message.as_ref().clone();
    let mut replacement_message = replacement_message.as_ref().clone();
    original_message.content.clear();
    replacement_message.content.clear();
    if original_message != replacement_message {
        return Err(SurfaceError::PruneRewriteMismatch);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_replacement(
    nodes: &mut Vec<SurfaceNode>,
    replacements: &mut Vec<SurfaceReplacement>,
    shadowed_seqs: &[SessionSeq],
    range: ShadowedRange,
    shadowed_token_count: u64,
    metering_seq: SessionSeq,
    replacement_seq: SessionSeq,
    replacement_kind: SurfaceNodeKind,
) -> Result<(), SurfaceError> {
    if shadowed_seqs.is_empty() {
        return Err(SurfaceError::EmptyShadowedSeqs);
    }
    if shadowed_seqs.first() != Some(&range.start) || shadowed_seqs.last() != Some(&range.end) {
        return Err(SurfaceError::RangeMismatch {
            start: range.start,
            end: range.end,
        });
    }

    let Some(first) = nodes.iter().position(|node| node.seq() == shadowed_seqs[0]) else {
        return Err(SurfaceError::MissingShadowedSeq {
            seq: shadowed_seqs[0],
        });
    };
    if first == 0 && nodes[0].kind() == SurfaceNodeKind::System {
        return Err(SurfaceError::SystemHeadProtected);
    }
    for (offset, seq) in shadowed_seqs.iter().enumerate() {
        let Some(node) = nodes.get(first + offset) else {
            return Err(SurfaceError::NotContiguous { seq: *seq });
        };
        if node.seq() != *seq {
            return Err(SurfaceError::NotContiguous { seq: *seq });
        }
    }

    nodes.splice(
        first..first + shadowed_seqs.len(),
        [SurfaceNode::new(replacement_seq, replacement_kind)],
    );
    replacements.push(SurfaceReplacement {
        metering_seq,
        seq: replacement_seq,
        start: range.start,
        end: range.end,
        shadowed_seqs: shadowed_seqs.to_vec(),
        shadowed_token_count,
    });
    Ok(())
}

fn node_kind(event: &SessionEvent) -> Option<SurfaceNodeKind> {
    let SessionEvent::Core(event) = event else {
        return None;
    };
    match event {
        CoreEvent::SystemMessage { .. } => Some(SurfaceNodeKind::System),
        CoreEvent::DeveloperMessage { .. } => Some(SurfaceNodeKind::Developer),
        CoreEvent::UserMessage { .. } => Some(SurfaceNodeKind::User),
        CoreEvent::AssistantMessage { message, .. } => {
            let tool_calls = message
                .content
                .iter()
                .filter(|block| block.as_tool_call().is_some())
                .count();
            Some(SurfaceNodeKind::Assistant { tool_calls })
        }
        CoreEvent::ToolResult { .. } => Some(SurfaceNodeKind::Tool),
        _ => None,
    }
}

fn sequence(index: usize) -> Result<SessionSeq, SurfaceError> {
    SessionSeq::new(index as u64).map_err(|_| SurfaceError::IndexOutOfRange { index })
}
