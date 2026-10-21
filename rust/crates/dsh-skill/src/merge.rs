//! Deterministic catalog merging.
//!
//! The TypeScript registry collects runtime skills and provider candidates per
//! layer, sorts them by `(rank, providerOrder, localOrder)`, keeps the first
//! entry per name, then overlays layers from the global one to the nearest
//! viewing scope. A nearer layer wins a duplicate name outright; rank only
//! decides duplicates inside one layer. Nothing here performs discovery, so
//! the whole rule set is a pure function over already-collected candidates.

use std::collections::{HashMap, HashSet};

use crate::name::SkillName;
use crate::types::{SkillCandidate, SkillDefinition, SkillRank, SkillSummary};

/// Provider label reserved for in-memory registrations.
pub const RUNTIME_PROVIDER: &str = "runtime";

/// Rank assigned to in-memory registrations.
pub const RUNTIME_RANK: f64 = 250.0;

/// Standard precedence rank for packaged skill providers and bundled roots.
pub const BUNDLED_SKILL_RANK: f64 = 600.0;

/// One candidate plus the two registration orders that break rank ties.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexedCandidate<L> {
    /// The candidate being ranked.
    pub candidate: SkillCandidate<L>,
    /// Service-wide monotonic provider registration order.
    ///
    /// Runtime entries use `-1`, which places them before every registered
    /// provider at the same rank.
    pub provider_order: i64,
    /// Position within the provider's discovery result.
    pub local_order: u64,
}

impl<L> IndexedCandidate<L> {
    /// Pair a candidate with its provider and provider-local order.
    pub fn new(candidate: SkillCandidate<L>, provider_order: i64, local_order: u64) -> Self {
        Self {
            candidate,
            provider_order,
            local_order,
        }
    }
}

/// Resolve one layer's duplicate names.
///
/// Entries are ordered by rank, provider registration order, then
/// provider-local order; the first entry for a name wins and later duplicates
/// are discarded. The returned candidates keep that winning order.
pub fn resolve_layer<L>(entries: Vec<IndexedCandidate<L>>) -> Vec<SkillCandidate<L>> {
    let mut entries = entries;
    entries.sort_by(|left, right| {
        left.candidate
            .rank
            .cmp(&right.candidate.rank)
            .then_with(|| left.provider_order.cmp(&right.provider_order))
            .then_with(|| left.local_order.cmp(&right.local_order))
    });

    let mut seen = HashSet::new();
    let mut winners = Vec::new();
    for entry in entries {
        if seen.insert(entry.candidate.summary.name.clone()) {
            winners.push(entry.candidate);
        }
    }
    winners
}

/// Overlay resolved layers so the nearest layer wins each duplicate name.
///
/// `layers` must arrive in shadowing order: the global layer first, then
/// scope layers from the farthest ancestor to the exact scope, matching
/// `collectFresh` in the TypeScript registry. The output keeps first-seen
/// insertion order; [`sort_summaries`] produces the catalog's name ordering.
pub fn merge_layers<L>(
    layers: impl IntoIterator<Item = Vec<SkillCandidate<L>>>,
) -> Vec<SkillCandidate<L>> {
    let mut positions: HashMap<SkillName, usize> = HashMap::new();
    let mut merged: Vec<SkillCandidate<L>> = Vec::new();
    for layer in layers {
        for candidate in layer {
            match positions.get(&candidate.summary.name) {
                Some(&index) => merged[index] = candidate,
                None => {
                    positions.insert(candidate.summary.name.clone(), merged.len());
                    merged.push(candidate);
                }
            }
        }
    }
    merged
}

/// Build the runtime contribution for one layer.
///
/// Runtime skills are ordered by name before their `localOrder` is assigned,
/// then ranked with [`RUNTIME_RANK`]; the full definition is the locator, so a
/// later load returns the body without provider work.
pub fn runtime_layer(definitions: Vec<SkillDefinition>) -> Vec<IndexedCandidate<SkillDefinition>> {
    let mut definitions = definitions;
    definitions.sort_by(|left, right| {
        compare_skill_names(left.summary.name.as_str(), right.summary.name.as_str())
    });
    let rank = SkillRank::from_finite(RUNTIME_RANK);
    definitions
        .into_iter()
        .enumerate()
        .map(|(local_order, definition)| {
            let metadata = definition.metadata.clone();
            let candidate = SkillCandidate::new(definition.summary.clone(), rank, definition)
                .with_metadata(metadata);
            IndexedCandidate::new(candidate, -1, local_order as u64)
        })
        .collect()
}

/// Sort summaries into the catalog's deterministic order.
///
/// Skill names are ASCII by grammar, so Rust's byte order is the same order as
/// the TypeScript `<` comparison. The function exists so callers do not have to
/// know that fact.
pub fn sort_summaries(summaries: &mut [SkillSummary]) {
    summaries.sort_by(|left, right| left.name.cmp(&right.name));
}

/// Compare two skill names in catalog order.
pub fn compare_skill_names(left: &str, right: &str) -> std::cmp::Ordering {
    left.cmp(right)
}
