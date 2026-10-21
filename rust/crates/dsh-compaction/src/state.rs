//! Durable lock inspection, surface range validation, and the compaction state
//! projection.
//!
//! The projection is a pure fold: replaying the same log always yields the same
//! current surface, the same applied replacements, and the same cumulative
//! shadow price. No caches, clocks, or session handles participate.

#![forbid(unsafe_code)]

use dsh_session::{CoreEvent, SessionEvent, SessionSeq};

use super::error::{CompactionError, SurfaceError};
use super::events::{CompactionEvent, ShadowedRange, COMPACTION_END_TAG, COMPACTION_START_TAG};
use super::pairing::ToolPairing;
use super::surface::{fold_surface, SurfaceNode, SurfaceNodeKind, SurfaceProjection};

/// The latest markers that decide whether a compaction may start.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactionEntryState {
    /// Turn number of the latest still-open turn, or `None` between turns.
    pub open_turn: Option<u64>,
    /// Sequence of the latest unmatched `compaction/start`, if any.
    pub unmatched_compaction_start: Option<SessionSeq>,
    /// Sequence of the newest `session/end-seed` boundary, if any.
    pub latest_end_seed_seq: Option<SessionSeq>,
}

/// Inspect the latest open turn, unmatched compaction start, and seed boundary.
///
/// The scan is backward and stops once all three facts are known. A later
/// `session/end-seed` proves that an unmatched start belongs to an earlier
/// session lifecycle, so it no longer holds the lock.
pub fn inspect_compaction_entry_state(
    events: &[SessionEvent],
) -> Result<CompactionEntryState, CompactionError> {
    let mut state = CompactionEntryState::default();
    let mut compaction_entry_state_known = false;
    let mut open_turn_state_known = false;

    for index in (0..events.len()).rev() {
        let seq = sequence(index)?;
        match &events[index] {
            SessionEvent::Core(CoreEvent::EndSeed { .. }) => {
                if state.latest_end_seed_seq.is_none() {
                    state.latest_end_seed_seq = Some(seq);
                }
            }
            SessionEvent::Core(CoreEvent::TurnStart { turn }) if !open_turn_state_known => {
                state.open_turn = Some(*turn);
                open_turn_state_known = true;
            }
            SessionEvent::Core(CoreEvent::TurnEnd { .. }) if !open_turn_state_known => {
                state.open_turn = None;
                open_turn_state_known = true;
            }
            SessionEvent::Extension(extension) if !compaction_entry_state_known => {
                match extension.kind() {
                    COMPACTION_START_TAG => {
                        let event = CompactionEvent::decode(&events[index])
                            .ok_or_else(|| corrupt_marker(seq))?
                            .map_err(SurfaceError::from)?;
                        if let CompactionEvent::Start(_) = event {
                            state.unmatched_compaction_start = Some(seq);
                        }
                        compaction_entry_state_known = true;
                    }
                    COMPACTION_END_TAG => {
                        let _ = CompactionEvent::decode(&events[index])
                            .ok_or_else(|| corrupt_marker(seq))?
                            .map_err(SurfaceError::from)?;
                        compaction_entry_state_known = true;
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        if open_turn_state_known
            && compaction_entry_state_known
            && state.latest_end_seed_seq.is_some()
        {
            break;
        }
    }

    Ok(state)
}

/// Fail one inspected marker that decodes to the wrong variant.
fn corrupt_marker(seq: SessionSeq) -> CompactionError {
    CompactionError::Surface(SurfaceError::MissingSurfaceEvent { seq })
}

/// Reject a durable unmatched compaction marker unless a later seed boundary
/// proves that its owner belongs to an earlier session lifecycle.
pub fn assert_compaction_inactive(
    state: &CompactionEntryState,
    stage: &str,
) -> Result<(), CompactionError> {
    let Some(start) = state.unmatched_compaction_start else {
        return Ok(());
    };
    if state
        .latest_end_seed_seq
        .is_some_and(|end_seed| end_seed > start)
    {
        return Ok(());
    }
    Err(CompactionError::CompactionBusy {
        stage: stage.to_string(),
    })
}

/// Inspect the log and choose the owner for one compaction transaction.
///
/// A manual transaction must run between turns and writes `turn: null`; an
/// automatic transaction must be enclosed by the open turn. The durable lock is
/// checked first, matching the TypeScript transaction entry order.
pub fn plan_compaction_owner(
    events: &[SessionEvent],
    manual: bool,
    stage: &str,
) -> Result<Option<u64>, CompactionError> {
    let state = inspect_compaction_entry_state(events)?;
    assert_compaction_inactive(&state, stage)?;
    if manual {
        if state.open_turn.is_some() {
            return Err(CompactionError::ManualTurnOpen);
        }
        Ok(None)
    } else {
        state.open_turn.map(Some).ok_or(CompactionError::NoOpenTurn)
    }
}

/// One validated inclusive span of current surface positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceRegion {
    /// First surface-node sequence, inclusive.
    pub start: SessionSeq,
    /// Last surface-node sequence, inclusive.
    pub end: SessionSeq,
    /// Position of `start` in the current surface.
    pub start_idx: usize,
    /// Position of `end` in the current surface.
    pub end_idx: usize,
    /// Every shadowed sequence, in surface order.
    pub shadowed_seqs: Vec<SessionSeq>,
}

/// Validate one requested surface-position span before asynchronous work.
///
/// Both edges must be present and balanced so an assistant tool call remains
/// paired with its result. A reversed range is rejected with both positions.
pub fn validate_surface_region(
    nodes: &[SurfaceNode],
    start: SessionSeq,
    end: SessionSeq,
) -> Result<SurfaceRegion, CompactionError> {
    let Some(start_idx) = nodes.iter().position(|node| node.seq() == start) else {
        return Err(CompactionError::RangeStartNotFound { start });
    };
    let Some(end_idx) = nodes.iter().position(|node| node.seq() == end) else {
        return Err(CompactionError::RangeEndNotFound { end });
    };
    if start_idx > end_idx {
        return Err(CompactionError::RangeReversed {
            start,
            end,
            start_position: start_idx,
            end_position: end_idx,
        });
    }

    let pairing = ToolPairing::new(nodes)?;
    if !pairing.balanced_before(nodes[start_idx].seq())? {
        return Err(CompactionError::RangeStartUnbalanced { start });
    }
    if !pairing.balanced_after(nodes[end_idx].seq())? {
        return Err(CompactionError::RangeEndUnbalanced { end });
    }

    Ok(SurfaceRegion {
        start,
        end,
        start_idx,
        end_idx,
        shadowed_seqs: nodes[start_idx..=end_idx]
            .iter()
            .map(SurfaceNode::seq)
            .collect(),
    })
}

/// One surface node's measured price.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodePrice {
    /// The priced surface node.
    pub seq: SessionSeq,
    /// Tokens attributed to that node by the conversation meter.
    pub tokens: u64,
}

impl NodePrice {
    /// Pair one surface sequence with its measured price.
    pub fn new(seq: SessionSeq, tokens: u64) -> Self {
        Self { seq, tokens }
    }
}

/// Resolve the next compactable range from a stable priced surface.
///
/// The range starts at the first non-system node, retains a recent priced tail,
/// and never splits an assistant tool-call/result pair. A system message at
/// surface position 0 is never inside the range.
pub fn select_compactable_range(
    nodes: &[SurfaceNode],
    prices: &[NodePrice],
    retain_tokens: u64,
) -> Result<Option<ShadowedRange>, CompactionError> {
    if prices.is_empty() {
        return Ok(None);
    }
    if nodes.len() != prices.len()
        || nodes
            .iter()
            .zip(prices)
            .any(|(node, price)| node.seq() != price.seq)
    {
        return Err(CompactionError::TokenMeterSurfaceMismatch);
    }

    let first_idx = match nodes.first().map(SurfaceNode::kind) {
        Some(SurfaceNodeKind::System) => 1,
        _ => 0,
    };

    let mut accumulated = 0_u64;
    let mut keep_from_idx = prices.len();
    for index in (0..prices.len()).rev() {
        accumulated = accumulated.saturating_add(prices[index].tokens);
        keep_from_idx = index;
        if accumulated >= retain_tokens {
            break;
        }
    }
    if keep_from_idx <= first_idx {
        return Ok(None);
    }

    let pairing = ToolPairing::new(nodes)?;
    while keep_from_idx > first_idx {
        if pairing.balanced_before(nodes[keep_from_idx].seq())? {
            break;
        }
        keep_from_idx -= 1;
    }
    if keep_from_idx <= first_idx {
        return Ok(None);
    }

    Ok(Some(ShadowedRange::new(
        nodes[first_idx].seq(),
        nodes[keep_from_idx - 1].seq(),
    )))
}

/// The compaction-aware projection of one complete session log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionState {
    /// Current model-visible surface.
    pub surface: SurfaceProjection,
    /// Latest lock and turn markers.
    pub entry: CompactionEntryState,
}

impl CompactionState {
    /// Cumulative heuristic price of every applied replacement.
    pub fn shadowed_token_count(&self) -> u64 {
        self.surface.shadowed_token_count()
    }

    /// Monotonic count of applied positional replacements.
    pub fn replace_generation(&self) -> usize {
        self.surface.replace_generation()
    }
}

/// Replay one log into its current compaction state.
pub fn fold_compaction_state(events: &[SessionEvent]) -> Result<CompactionState, CompactionError> {
    Ok(CompactionState {
        surface: fold_surface(events)?,
        entry: inspect_compaction_entry_state(events)?,
    })
}

fn sequence(index: usize) -> Result<SessionSeq, CompactionError> {
    SessionSeq::new(index as u64).map_err(|_| SurfaceError::IndexOutOfRange { index }.into())
}
