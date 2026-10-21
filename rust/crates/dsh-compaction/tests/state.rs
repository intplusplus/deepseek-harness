//! Lock inspection, region validation, range selection, and replay projection.
//!
//! Ported from `packages/compaction/compaction-basic/src/region.ts` and the
//! entry-state rules exercised by `manual-compaction.spec.ts`.

mod common;

use common::{
    assistant_with_calls, checkpoint, end, end_seed, seq, start, summary, system, text,
    tool_result, turn_end, turn_start, user,
};
use dsh_compaction::{
    assert_compaction_inactive, fold_compaction_state, inspect_compaction_entry_state,
    plan_compaction_owner, select_compactable_range, validate_surface_region, CompactionEntryState,
    CompactionError, NodePrice, SurfaceError, SurfaceNode, SurfaceNodeKind,
};

fn node(value: u64, kind: SurfaceNodeKind) -> SurfaceNode {
    SurfaceNode::new(seq(value), kind)
}

fn call_node(value: u64) -> SurfaceNode {
    node(value, SurfaceNodeKind::Assistant { tool_calls: 1 })
}

#[test]
fn inspects_the_open_turn_and_an_unmatched_start() {
    let events = [turn_start(1), start("c1", Some(1))];
    let state = inspect_compaction_entry_state(&events).unwrap();
    assert_eq!(state.open_turn, Some(1));
    assert_eq!(state.unmatched_compaction_start, Some(seq(1)));
    assert_eq!(state.latest_end_seed_seq, None);
}

#[test]
fn a_turn_end_closes_the_open_turn() {
    let events = [turn_start(1), turn_end(1)];
    let state = inspect_compaction_entry_state(&events).unwrap();
    assert_eq!(state.open_turn, None);
}

#[test]
fn a_later_end_seed_invalidates_an_unmatched_start() {
    let events = [start("c1", Some(1)), end_seed()];
    let state = inspect_compaction_entry_state(&events).unwrap();
    assert_eq!(state.unmatched_compaction_start, Some(seq(0)));
    assert_eq!(state.latest_end_seed_seq, Some(seq(1)));
    assert_compaction_inactive(&state, "compaction").unwrap();
}

#[test]
fn an_end_seed_before_the_start_does_not_invalidate_it() {
    let events = [end_seed(), start("c1", Some(1))];
    let state = inspect_compaction_entry_state(&events).unwrap();
    assert_eq!(state.latest_end_seed_seq, Some(seq(0)));
    let error = assert_compaction_inactive(&state, "compactRegion").unwrap_err();
    assert!(matches!(error, CompactionError::CompactionBusy { .. }));
    assert!(error
        .to_string()
        .contains("compactRegion: compaction already in progress"));
}

#[test]
fn a_compaction_end_clears_the_lock() {
    let events = [start("c1", Some(1)), end("c1", Some(1))];
    let state = inspect_compaction_entry_state(&events).unwrap();
    assert_eq!(state.unmatched_compaction_start, None);
    assert_compaction_inactive(&state, "compaction").unwrap();
}

#[test]
fn a_later_start_becomes_the_unmatched_lock() {
    let events = [
        start("c1", Some(1)),
        end("c1", Some(1)),
        start("c2", Some(1)),
    ];
    let state = inspect_compaction_entry_state(&events).unwrap();
    assert_eq!(state.unmatched_compaction_start, Some(seq(2)));
}

#[test]
fn summary_markers_do_not_hold_or_release_the_lock() {
    let events = [
        start("c1", Some(1)),
        summary("c1", &[seq(0)], 3),
        end("c1", Some(1)),
    ];
    let state = inspect_compaction_entry_state(&events).unwrap();
    assert_eq!(state.unmatched_compaction_start, None);
}

#[test]
fn a_busy_stage_is_named_in_the_diagnostic() {
    let state = CompactionEntryState {
        open_turn: None,
        unmatched_compaction_start: Some(seq(0)),
        latest_end_seed_seq: None,
    };
    let error = assert_compaction_inactive(&state, "manual compaction").unwrap_err();
    assert_eq!(
        error.to_string(),
        "manual compaction: compaction already in progress; the session compaction lock is already active"
    );
}

#[test]
fn manual_plan_requires_an_idle_session() {
    let error = plan_compaction_owner(&[turn_start(1)], true, "manual").unwrap_err();
    assert!(matches!(error, CompactionError::ManualTurnOpen));
    assert_eq!(
        plan_compaction_owner(&[turn_start(1), turn_end(1)], true, "manual").unwrap(),
        None
    );
}

#[test]
fn automatic_plan_requires_an_open_turn() {
    let error = plan_compaction_owner(&[], false, "compactRegion").unwrap_err();
    assert!(matches!(error, CompactionError::NoOpenTurn));
    assert_eq!(
        plan_compaction_owner(&[turn_start(3)], false, "compactRegion").unwrap(),
        Some(3)
    );
}

#[test]
fn plan_checks_the_durable_lock_before_the_turn_owner() {
    let error = plan_compaction_owner(&[start("c1", Some(1))], false, "compactRegion").unwrap_err();
    assert!(matches!(error, CompactionError::CompactionBusy { .. }));
}

#[test]
fn validates_a_balanced_surface_region() {
    let nodes = [
        node(0, SurfaceNodeKind::User),
        call_node(1),
        node(2, SurfaceNodeKind::Tool),
        node(3, SurfaceNodeKind::User),
    ];
    let region = validate_surface_region(&nodes, seq(0), seq(3)).unwrap();
    assert_eq!(region.start, seq(0));
    assert_eq!(region.end, seq(3));
    assert_eq!(region.start_idx, 0);
    assert_eq!(region.end_idx, 3);
    assert_eq!(region.shadowed_seqs, vec![seq(0), seq(1), seq(2), seq(3)]);
}

#[test]
fn rejects_missing_region_edges() {
    let nodes = [
        node(0, SurfaceNodeKind::User),
        node(1, SurfaceNodeKind::User),
    ];
    assert!(matches!(
        validate_surface_region(&nodes, seq(9), seq(0)).unwrap_err(),
        CompactionError::RangeStartNotFound { start } if start == seq(9)
    ));
    assert!(matches!(
        validate_surface_region(&nodes, seq(0), seq(9)).unwrap_err(),
        CompactionError::RangeEndNotFound { end } if end == seq(9)
    ));
}

#[test]
fn rejects_a_reversed_region_with_both_positions() {
    let nodes = [
        node(0, SurfaceNodeKind::User),
        node(1, SurfaceNodeKind::User),
    ];
    let error = validate_surface_region(&nodes, seq(1), seq(0)).unwrap_err();
    assert!(matches!(
        error,
        CompactionError::RangeReversed {
            start_position: 1,
            end_position: 0,
            ..
        }
    ));
    assert!(error
        .to_string()
        .contains("is after end seq 0 (position 0)"));
}

#[test]
fn rejects_an_unbalanced_region_start() {
    let nodes = [
        call_node(0),
        node(1, SurfaceNodeKind::Tool),
        node(2, SurfaceNodeKind::User),
    ];
    assert!(matches!(
        validate_surface_region(&nodes, seq(1), seq(2)).unwrap_err(),
        CompactionError::RangeStartUnbalanced { start } if start == seq(1)
    ));
}

#[test]
fn rejects_an_unbalanced_region_end() {
    let nodes = [node(0, SurfaceNodeKind::User), call_node(1)];
    assert!(matches!(
        validate_surface_region(&nodes, seq(0), seq(1)).unwrap_err(),
        CompactionError::RangeEndUnbalanced { end } if end == seq(1)
    ));
}

#[test]
fn selects_a_range_that_retains_the_priced_tail() {
    let nodes = [
        node(0, SurfaceNodeKind::User),
        node(1, SurfaceNodeKind::User),
        node(2, SurfaceNodeKind::User),
    ];
    let prices = [
        NodePrice::new(seq(0), 10),
        NodePrice::new(seq(1), 20),
        NodePrice::new(seq(2), 30),
    ];
    let retained = select_compactable_range(&nodes, &prices, 40)
        .unwrap()
        .unwrap();
    assert_eq!(retained.start, seq(0));
    assert_eq!(retained.end, seq(0));

    let none_retained = select_compactable_range(&nodes, &prices, 0)
        .unwrap()
        .unwrap();
    assert_eq!(none_retained.start, seq(0));
    assert_eq!(none_retained.end, seq(1));
}

#[test]
fn selection_always_retains_the_last_priced_node() {
    let nodes = [node(0, SurfaceNodeKind::User)];
    let prices = [NodePrice::new(seq(0), 5)];
    assert_eq!(select_compactable_range(&nodes, &prices, 0).unwrap(), None);
    assert_eq!(select_compactable_range(&nodes, &[], 0).unwrap(), None);
}

#[test]
fn selection_starts_after_a_system_head() {
    let nodes = [
        node(0, SurfaceNodeKind::System),
        node(1, SurfaceNodeKind::User),
        node(2, SurfaceNodeKind::User),
    ];
    let prices = [
        NodePrice::new(seq(0), 1),
        NodePrice::new(seq(1), 2),
        NodePrice::new(seq(2), 3),
    ];
    let range = select_compactable_range(&nodes, &prices, 0)
        .unwrap()
        .unwrap();
    assert_eq!(range.start, seq(1));
    assert_eq!(range.end, seq(1));
}

#[test]
fn selection_rejects_a_price_surface_mismatch() {
    let nodes = [node(0, SurfaceNodeKind::User)];
    assert!(matches!(
        select_compactable_range(&nodes, &[NodePrice::new(seq(1), 1)], 0).unwrap_err(),
        CompactionError::TokenMeterSurfaceMismatch
    ));
    assert_eq!(select_compactable_range(&nodes, &[], 0).unwrap(), None);
}

#[test]
fn selection_never_splits_a_tool_pair() {
    let nodes = [
        node(0, SurfaceNodeKind::User),
        call_node(1),
        node(2, SurfaceNodeKind::Tool),
        node(3, SurfaceNodeKind::User),
    ];
    let prices = [
        NodePrice::new(seq(0), 1),
        NodePrice::new(seq(1), 1),
        NodePrice::new(seq(2), 1),
        NodePrice::new(seq(3), 1),
    ];
    let outside_pair = select_compactable_range(&nodes, &prices, 1)
        .unwrap()
        .unwrap();
    assert_eq!(outside_pair.end, seq(2));

    let inside_pair = select_compactable_range(&nodes, &prices, 2)
        .unwrap()
        .unwrap();
    assert_eq!(inside_pair.end, seq(0));
}

#[test]
fn fold_replays_the_same_state_deterministically() {
    let events = [
        turn_start(1),
        user("u", vec![text("question")]),
        assistant_with_calls("a", &["c1"]),
        tool_result("r", "c1", vec![text("answer")]),
        start("c1", Some(1)),
        summary("c1", &[seq(1), seq(2), seq(3)], 11),
        checkpoint("c1", "cp"),
        end("c1", Some(1)),
    ];
    let first = fold_compaction_state(&events).unwrap();
    let second = fold_compaction_state(&events).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.surface.nodes().len(), 1);
    assert_eq!(first.surface.nodes()[0].seq(), seq(6));
    assert_eq!(first.replace_generation(), 1);
    assert_eq!(first.shadowed_token_count(), 11);
    assert_eq!(first.entry.open_turn, Some(1));
    assert_eq!(first.entry.unmatched_compaction_start, None);
}

#[test]
fn fold_rejects_a_malformed_compaction_marker() {
    let events = [common::extension(
        dsh_compaction::COMPACTION_START_TAG,
        serde_json::json!({
            "type": dsh_compaction::COMPACTION_START_TAG,
            "compactionId": "c1"
        }),
    )];
    let error = fold_compaction_state(&events).unwrap_err();
    assert!(matches!(
        error,
        CompactionError::Surface(SurfaceError::Event(_))
    ));
}

#[test]
fn a_system_node_after_the_head_is_not_protected() {
    let events = [
        user("u", vec![text("question")]),
        system("s"),
        summary("c1", &[seq(1)], 2),
        checkpoint("c1", "cp"),
    ];
    let surface = dsh_compaction::fold_surface(&events).unwrap();
    assert_eq!(
        surface
            .nodes()
            .iter()
            .map(|node| node.seq())
            .collect::<Vec<_>>(),
        vec![seq(0), seq(3)]
    );
}
