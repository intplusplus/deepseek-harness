//! Tool-call/result balance over one session surface.
//!
//! Compaction changes surface positions, so safe cuts come from tool-call and
//! tool-result content in current surface order rather than from step markers.
//! A surface of `N` nodes has `N + 1` cuts; entry `i` is the cut before node
//! `i`, and the final entry is the cut after the tail.

#![forbid(unsafe_code)]

use std::collections::HashMap;

use dsh_session::SessionSeq;

use super::error::ToolPairingError;
use super::surface::{SurfaceNode, SurfaceNodeKind};

/// Validated balance state for one surface snapshot.
///
/// The TypeScript service caches this per session generation. This port is a
/// pure value: rebuilding it from a surface is cheap, deterministic, and has no
/// lifecycle to invalidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolPairing {
    cut_balanced: Vec<bool>,
    index_by_seq: HashMap<SessionSeq, usize>,
}

impl ToolPairing {
    /// Derive balance state from a surface in current order.
    ///
    /// The leading cut is trivially balanced. An assistant node opens one call
    /// per `tool-call` block; a tool result closes one. A result with no
    /// preceding open call rejects the surface as corrupt.
    pub fn new(nodes: &[SurfaceNode]) -> Result<Self, ToolPairingError> {
        let mut cut_balanced = Vec::with_capacity(nodes.len() + 1);
        cut_balanced.push(true);
        let mut index_by_seq = HashMap::with_capacity(nodes.len());
        let mut in_progress: usize = 0;

        for (index, node) in nodes.iter().enumerate() {
            match node.kind() {
                SurfaceNodeKind::Assistant { tool_calls } => in_progress += tool_calls,
                SurfaceNodeKind::Tool => {
                    if in_progress == 0 {
                        return Err(ToolPairingError::OrphanResult { seq: node.seq() });
                    }
                    in_progress -= 1;
                }
                SurfaceNodeKind::System | SurfaceNodeKind::Developer | SurfaceNodeKind::User => {}
            }
            index_by_seq.insert(node.seq(), index);
            cut_balanced.push(in_progress == 0);
        }

        Ok(Self {
            cut_balanced,
            index_by_seq,
        })
    }

    /// Balance of every cut, in surface order.
    ///
    /// A surface of `N` nodes yields `N + 1` entries; entry `i` is the cut
    /// before node `i` and entry `N` is the cut after the tail.
    pub fn cuts(&self) -> &[bool] {
        &self.cut_balanced
    }

    /// Whether the cut immediately before `seq` is balanced.
    pub fn balanced_before(&self, seq: SessionSeq) -> Result<bool, ToolPairingError> {
        self.cut_balance(seq, 0)
    }

    /// Whether the cut immediately after `seq` is balanced.
    pub fn balanced_after(&self, seq: SessionSeq) -> Result<bool, ToolPairingError> {
        self.cut_balance(seq, 1)
    }

    /// Balance of one cut relative to a sequence's current position.
    fn cut_balance(&self, seq: SessionSeq, offset: usize) -> Result<bool, ToolPairingError> {
        let Some(index) = self.index_by_seq.get(&seq) else {
            return Err(ToolPairingError::MissingSeq { seq });
        };
        self.cut_balanced
            .get(index + offset)
            .copied()
            .ok_or(ToolPairingError::MissingSeq { seq })
    }
}

/// Whether the cut immediately before a current surface sequence is balanced.
pub fn tool_pairing_balanced_before(
    nodes: &[SurfaceNode],
    seq: SessionSeq,
) -> Result<bool, ToolPairingError> {
    ToolPairing::new(nodes)?.balanced_before(seq)
}

/// Whether the cut immediately after a current surface sequence is balanced.
pub fn tool_pairing_balanced_after(
    nodes: &[SurfaceNode],
    seq: SessionSeq,
) -> Result<bool, ToolPairingError> {
    ToolPairing::new(nodes)?.balanced_after(seq)
}
