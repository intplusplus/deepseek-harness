//! Deterministic, model-free tool-result pruning.
//!
//! The pruner trims the middle of over-budget text while preserving every rich
//! block in order. Text slicing counts Unicode code points, not UTF-16 code
//! units, so a retained boundary cannot split a surrogate pair. Grapheme
//! clusters may still split, exactly as in the TypeScript implementation.
//!
//! Planning is pure: [`plan_prune_session`] returns the replacement messages
//! and character savings, and the caller appends the `compaction/prune`
//! metering event plus each replacement to the durable log. The shadow token
//! price is deliberately not computed here because the Rust token-meter seam
//! does not exist yet.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use dsh_llm::{ContentBlock, ToolCallId, ToolResultMessage};
use dsh_session::{CoreEvent, SessionEvent, SessionSeq};

use super::error::{CompactionError, PruneError, SurfaceError};
use super::events::{CompactionPrune, ShadowedRange};
use super::surface::{SurfaceNodeKind, SurfaceProjection};

/// Marker inserted once at the first text block that intersects the removed span.
pub const PRUNE_MARKER: &str = "\n\n[... tool result middle pruned ...]\n\n";
/// Default character budget above which a tool result is pruned.
pub const DEFAULT_THRESHOLD_CHARS: u64 = 8_192;
/// Default characters retained from the head of the text.
pub const DEFAULT_HEAD_CHARS: u64 = 4_096;
/// Default characters retained from the tail of the text.
pub const DEFAULT_TAIL_CHARS: u64 = 1_024;

/// Raw tool-result pruning configuration.
///
/// Every field is optional; [`resolve_prune_config`] applies the documented
/// defaults and validates the budgets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct ToolResultPruneConfig {
    /// Prune when the total text exceeds this many code points.
    pub threshold_chars: Option<u64>,
    /// Code points retained from the head.
    pub head_chars: Option<u64>,
    /// Code points retained from the tail.
    pub tail_chars: Option<u64>,
}

/// Validated and immutable character budgets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedPruneConfig {
    /// Prune when the total text exceeds this many code points.
    pub threshold_chars: u64,
    /// Code points retained from the head.
    pub head_chars: u64,
    /// Code points retained from the tail.
    pub tail_chars: u64,
}

/// Resolve optional character budgets, rejecting a non-positive threshold.
pub fn resolve_prune_config(
    config: &ToolResultPruneConfig,
) -> Result<ResolvedPruneConfig, PruneError> {
    let threshold_chars = config.threshold_chars.unwrap_or(DEFAULT_THRESHOLD_CHARS);
    if threshold_chars == 0 {
        return Err(PruneError::Config(
            "ToolResultPruneConfig: thresholdChars (0) must be a positive integer".to_string(),
        ));
    }
    let head_chars = config.head_chars.unwrap_or(DEFAULT_HEAD_CHARS);
    let tail_chars = config.tail_chars.unwrap_or(DEFAULT_TAIL_CHARS);
    let emitted_chars = head_chars
        .saturating_add(PRUNE_MARKER.chars().count() as u64)
        .saturating_add(tail_chars);
    if emitted_chars > threshold_chars {
        return Err(PruneError::Config(format!(
            "ToolResultPruneConfig: headChars + marker + tailChars ({emitted_chars}) must be at most thresholdChars ({threshold_chars})"
        )));
    }
    Ok(ResolvedPruneConfig {
        threshold_chars,
        head_chars,
        tail_chars,
    })
}

/// Measure text content in Unicode code points; non-text blocks cost zero.
pub fn measure_content(blocks: &[ContentBlock]) -> u64 {
    blocks
        .iter()
        .fold(0_u64, |total, block| match block.as_text() {
            Some(text) => total.saturating_add(text.chars().count() as u64),
            None => total,
        })
}

/// Replace an over-budget text middle while retaining rich-block order.
///
/// Returns `None` when the text is already within budget. The replacement is
/// guaranteed to be smaller than the input; a budget combination that cannot
/// produce one fails instead of returning an unchanged payload.
pub fn prune_content(
    blocks: &[ContentBlock],
    config: &ResolvedPruneConfig,
) -> Result<Option<Vec<ContentBlock>>, PruneError> {
    let total_chars = measure_content(blocks);
    if total_chars <= config.threshold_chars {
        return Ok(None);
    }

    let removed_start = i128::from(config.head_chars);
    let removed_end = i128::from(total_chars) - i128::from(config.tail_chars);
    let mut pruned = Vec::with_capacity(blocks.len());
    let mut consumed: i128 = 0;
    let mut marker_inserted = false;

    for block in blocks {
        let Some(text) = block.as_text() else {
            pruned.push(block.clone());
            continue;
        };

        let points: Vec<char> = text.chars().collect();
        let block_start = consumed;
        let block_end = block_start + points.len() as i128;
        let head_end = (removed_start - block_start).clamp(0, points.len() as i128) as usize;
        let tail_start = (removed_end - block_start).clamp(0, points.len() as i128) as usize;
        let intersects_removed = block_start < removed_end && block_end > removed_start;
        let marker = if intersects_removed && !marker_inserted {
            PRUNE_MARKER
        } else {
            ""
        };
        if !marker.is_empty() {
            marker_inserted = true;
        }

        let mut projected =
            String::with_capacity(head_end + (points.len() - tail_start) + marker.len());
        projected.extend(points[..head_end].iter());
        projected.push_str(marker);
        projected.extend(points[tail_start..].iter());
        if !projected.is_empty() {
            pruned.push(ContentBlock::text(projected));
        }
        consumed = block_end;
    }

    if !marker_inserted {
        return Err(PruneError::MissingRemovedSpan);
    }
    let chars_after = measure_content(&pruned);
    if chars_after > config.threshold_chars || chars_after >= total_chars {
        return Err(PruneError::ReplacementNotSmaller);
    }
    Ok(Some(pruned))
}

/// One planned tool-result replacement.
#[derive(Clone, Debug, PartialEq)]
pub struct PruneDecision {
    /// The current surface sequence being replaced.
    pub original_seq: SessionSeq,
    /// The answered tool-call identity.
    pub call_id: ToolCallId,
    /// Text code points before pruning.
    pub chars_before: u64,
    /// Text code points after pruning.
    pub chars_after: u64,
    /// Complete replacement message; identity, source, call id, and error flag
    /// are preserved and only `content` changed.
    pub replacement: ToolResultMessage,
}

impl PruneDecision {
    /// The single-node surface range this decision replaces.
    pub fn shadowed_range(&self) -> ShadowedRange {
        ShadowedRange::new(self.original_seq, self.original_seq)
    }

    /// Build the adjacent `compaction/prune` metering record.
    ///
    /// `shadowed_token_count` stays caller-supplied because the fixed estimator
    /// lives outside this crate.
    pub fn prune_event(&self, shadowed_token_count: u64) -> CompactionPrune {
        CompactionPrune::new(
            self.shadowed_range(),
            vec![self.original_seq],
            shadowed_token_count,
        )
    }
}

/// One complete deterministic prune pass over a stable surface snapshot.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrunePlan {
    /// Replacements in surface order.
    pub decisions: Vec<PruneDecision>,
    /// Aggregate text code points removed.
    pub chars_removed: u64,
}

/// Plan replacements for every over-budget current tool result.
///
/// The surface is read in order and each decision cites its original node, so a
/// caller can append the metering event and replacement synchronously adjacent.
/// Replacements committed by an earlier failed pass remain durable and are
/// simply measured again on the next pass.
pub fn plan_prune_session(
    surface: &SurfaceProjection,
    events: &[SessionEvent],
    config: &ResolvedPruneConfig,
) -> Result<PrunePlan, CompactionError> {
    let mut decisions = Vec::new();
    let mut chars_removed = 0_u64;

    for node in surface.nodes() {
        if node.kind() != SurfaceNodeKind::Tool {
            continue;
        }
        let event = events
            .get(node.seq().get() as usize)
            .ok_or(SurfaceError::MissingSurfaceEvent { seq: node.seq() })?;
        let SessionEvent::Core(CoreEvent::ToolResult { message, .. }) = event else {
            return Err(SurfaceError::MissingToolResult { seq: node.seq() }.into());
        };
        let original = message.as_ref();
        let Some(content) = prune_content(&original.content, config)? else {
            continue;
        };
        let chars_before = measure_content(&original.content);
        let chars_after = measure_content(&content);
        let mut replacement = original.clone();
        replacement.content = content;
        decisions.push(PruneDecision {
            original_seq: node.seq(),
            call_id: replacement.tool_call_id.clone(),
            chars_before,
            chars_after,
            replacement,
        });
        chars_removed = chars_removed.saturating_add(chars_before - chars_after);
    }

    Ok(PrunePlan {
        decisions,
        chars_removed,
    })
}
