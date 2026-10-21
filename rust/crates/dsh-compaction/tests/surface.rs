//! Pure surface replay: adjacency, positional replacement, and protected head.
//!
//! The TypeScript session records an explicit `surfaceOp`; the Rust log has no
//! such marker, so these tests pin the documented adjacency contract.

mod common;

use common::{
    assistant, assistant_with_calls, checkpoint, extension, prune, prune_many, seq, summary,
    summary_with_range, system, text, tool_result, turn_end, user,
};
use dsh_compaction::{
    fold_surface, ShadowedRange, SurfaceError, SurfaceNodeKind, COMPACTION_START_TAG,
};
use serde_json::json;

#[test]
fn folds_message_events_in_log_order() {
    let events = [
        system("s"),
        user("u", vec![text("hi")]),
        assistant("a"),
        tool_result("r", "c1", vec![text("done")]),
    ];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(
        surface
            .nodes()
            .iter()
            .map(|node| node.seq())
            .collect::<Vec<_>>(),
        vec![seq(0), seq(1), seq(2), seq(3)]
    );
    assert_eq!(
        surface
            .nodes()
            .iter()
            .map(|node| node.kind())
            .collect::<Vec<_>>(),
        vec![
            SurfaceNodeKind::System,
            SurfaceNodeKind::User,
            SurfaceNodeKind::Assistant { tool_calls: 0 },
            SurfaceNodeKind::Tool,
        ]
    );
    assert!(surface.replacements().is_empty());
    assert_eq!(surface.replace_generation(), 0);
    assert_eq!(surface.shadowed_token_count(), 0);
}

#[test]
fn summary_replaces_only_when_followed_by_matching_checkpoint() {
    let events = [
        user("u1", vec![text("one")]),
        user("u2", vec![text("two")]),
        summary("c1", &[seq(0), seq(1)], 12),
        checkpoint("c1", "cp"),
    ];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(surface.nodes().len(), 1);
    assert_eq!(surface.nodes()[0].seq(), seq(3));
    assert_eq!(surface.nodes()[0].kind(), SurfaceNodeKind::User);
    assert_eq!(surface.position_of(seq(3)), Some(0));
    assert!(!surface.contains(seq(0)));
    assert!(!surface.contains(seq(1)));

    let replacement = &surface.replacements()[0];
    assert_eq!(replacement.metering_seq, seq(2));
    assert_eq!(replacement.seq, seq(3));
    assert_eq!(replacement.start, seq(0));
    assert_eq!(replacement.end, seq(1));
    assert_eq!(replacement.shadowed_seqs, vec![seq(0), seq(1)]);
    assert_eq!(replacement.shadowed_token_count, 12);
    assert_eq!(surface.shadowed_token_count(), 12);
}

#[test]
fn summary_without_matching_checkpoint_is_log_only() {
    let events = [
        user("u1", vec![text("one")]),
        summary("c1", &[seq(0)], 4),
        user("tail", vec![text("tail")]),
    ];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(
        surface
            .nodes()
            .iter()
            .map(|node| node.seq())
            .collect::<Vec<_>>(),
        vec![seq(0), seq(2)]
    );
    assert!(surface.replacements().is_empty());
}

#[test]
fn summary_followed_by_checkpoint_with_different_id_is_log_only() {
    let events = [
        user("u1", vec![text("one")]),
        summary("c1", &[seq(0)], 4),
        checkpoint("c2", "cp"),
    ];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(
        surface
            .nodes()
            .iter()
            .map(|node| node.seq())
            .collect::<Vec<_>>(),
        vec![seq(0), seq(2)]
    );
    assert!(surface.replacements().is_empty());
}

#[test]
fn summary_followed_by_non_message_is_log_only() {
    let events = [
        user("u1", vec![text("one")]),
        summary("c1", &[seq(0)], 4),
        turn_end(1),
    ];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(surface.nodes().len(), 1);
    assert!(surface.replacements().is_empty());
}

#[test]
fn summary_rejects_empty_shadowed_seqs() {
    let events = [
        user("u1", vec![text("one")]),
        summary_with_range("c1", ShadowedRange::new(seq(0), seq(0)), &[], 1),
        checkpoint("c1", "cp"),
    ];
    assert!(matches!(
        fold_surface(&events).unwrap_err(),
        SurfaceError::EmptyShadowedSeqs
    ));
}

#[test]
fn summary_rejects_a_range_that_does_not_name_the_span_edges() {
    let events = [
        user("u1", vec![text("one")]),
        summary_with_range("c1", ShadowedRange::new(seq(0), seq(1)), &[seq(0)], 1),
        checkpoint("c1", "cp"),
    ];
    assert!(matches!(
        fold_surface(&events).unwrap_err(),
        SurfaceError::RangeMismatch { start, end } if start == seq(0) && end == seq(1)
    ));
}

#[test]
fn summary_rejects_a_non_contiguous_span() {
    let events = [
        user("u1", vec![text("one")]),
        user("u2", vec![text("two")]),
        user("u3", vec![text("three")]),
        summary("c1", &[seq(0), seq(2)], 4),
        checkpoint("c1", "cp"),
    ];
    assert!(matches!(
        fold_surface(&events).unwrap_err(),
        SurfaceError::NotContiguous { seq: at } if at == seq(2)
    ));
}

#[test]
fn summary_rejects_a_shadowed_sequence_missing_from_the_surface() {
    let events = [
        user("u1", vec![text("one")]),
        user("u2", vec![text("two")]),
        summary("c1", &[seq(5)], 4),
        checkpoint("c1", "cp"),
    ];
    assert!(matches!(
        fold_surface(&events).unwrap_err(),
        SurfaceError::MissingShadowedSeq { seq: at } if at == seq(5)
    ));
}

#[test]
fn summary_protects_a_system_head() {
    let events = [
        system("s"),
        summary("c1", &[seq(0)], 4),
        checkpoint("c1", "cp"),
    ];
    assert!(matches!(
        fold_surface(&events).unwrap_err(),
        SurfaceError::SystemHeadProtected
    ));
}

#[test]
fn summary_may_replace_from_the_first_non_system_node() {
    let events = [
        system("s"),
        user("u1", vec![text("one")]),
        user("u2", vec![text("two")]),
        summary("c1", &[seq(1), seq(2)], 4),
        checkpoint("c1", "cp"),
    ];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(
        surface
            .nodes()
            .iter()
            .map(|node| node.seq())
            .collect::<Vec<_>>(),
        vec![seq(0), seq(4)]
    );
    assert_eq!(surface.replacements()[0].start, seq(1));
}

#[test]
fn nested_replacement_uses_surface_positions_not_numeric_order() {
    let events = [
        user("u1", vec![text("one")]),
        user("u2", vec![text("two")]),
        summary("c1", &[seq(0)], 3),
        checkpoint("c1", "cp1"),
        summary_with_range(
            "c2",
            ShadowedRange::new(seq(3), seq(1)),
            &[seq(3), seq(1)],
            9,
        ),
        checkpoint("c2", "cp2"),
    ];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(surface.nodes().len(), 1);
    assert_eq!(surface.nodes()[0].seq(), seq(5));
    let nested = &surface.replacements()[1];
    assert!(nested.start > nested.end);
    assert_eq!(nested.shadowed_seqs, vec![seq(3), seq(1)]);
    assert_eq!(surface.shadowed_token_count(), 12);
}

#[test]
fn prune_replaces_the_adjacent_tool_result() {
    let events = [
        tool_result("r", "c1", vec![text("original")]),
        prune(seq(0), 7),
        tool_result("r", "c1", vec![text("pruned")]),
    ];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(surface.nodes().len(), 1);
    assert_eq!(surface.nodes()[0].seq(), seq(2));
    assert_eq!(surface.nodes()[0].kind(), SurfaceNodeKind::Tool);
    let replacement = &surface.replacements()[0];
    assert_eq!(replacement.metering_seq, seq(1));
    assert_eq!(replacement.seq, seq(2));
    assert_eq!(replacement.start, seq(0));
    assert_eq!(replacement.end, seq(0));
    assert_eq!(replacement.shadowed_seqs, vec![seq(0)]);
    assert_eq!(replacement.shadowed_token_count, 7);
}

#[test]
fn prune_without_an_adjacent_tool_result_is_log_only() {
    let events = [
        tool_result("r", "c1", vec![text("original")]),
        prune(seq(0), 7),
        user("tail", vec![text("tail")]),
    ];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(
        surface
            .nodes()
            .iter()
            .map(|node| node.seq())
            .collect::<Vec<_>>(),
        vec![seq(0), seq(2)]
    );
    assert!(surface.replacements().is_empty());
}

#[test]
fn prune_rejects_a_multi_node_shadow_span() {
    let events = [
        tool_result("r1", "c1", vec![text("one")]),
        tool_result("r2", "c2", vec![text("two")]),
        prune_many(&[seq(0), seq(1)], 8),
        tool_result("r1", "c1", vec![text("one")]),
    ];
    assert!(matches!(
        fold_surface(&events).unwrap_err(),
        SurfaceError::PruneSpanNotSingle
    ));
}

#[test]
fn prune_rejects_a_rewrite_that_changes_identity() {
    let events = [
        tool_result("r", "c1", vec![text("original")]),
        prune(seq(0), 7),
        tool_result("r", "c2", vec![text("pruned")]),
    ];
    assert!(matches!(
        fold_surface(&events).unwrap_err(),
        SurfaceError::PruneRewriteMismatch
    ));
}

#[test]
fn prune_rejects_an_original_that_is_not_a_tool_result() {
    let events = [
        user("u", vec![text("not a result")]),
        prune(seq(0), 7),
        tool_result("r", "c1", vec![text("pruned")]),
    ];
    assert!(matches!(
        fold_surface(&events).unwrap_err(),
        SurfaceError::MissingToolResult { seq: at } if at == seq(0)
    ));
}

#[test]
fn a_known_compaction_tag_with_a_bad_payload_fails_the_fold() {
    let events = [extension(
        COMPACTION_START_TAG,
        json!({ "type": COMPACTION_START_TAG, "compactionId": "c1" }),
    )];
    let error = fold_surface(&events).unwrap_err();
    assert!(matches!(error, SurfaceError::Event(_)));
    assert!(error.to_string().contains("missing field `turn`"));
}

#[test]
fn unrelated_extension_events_are_ignored() {
    let events = [
        user("u1", vec![text("one")]),
        extension("other/event", json!({ "type": "other/event" })),
        user("u2", vec![text("two")]),
    ];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(
        surface
            .nodes()
            .iter()
            .map(|node| node.seq())
            .collect::<Vec<_>>(),
        vec![seq(0), seq(2)]
    );
}

#[test]
fn shadowed_token_count_sums_every_applied_replacement() {
    let events = [
        user("u1", vec![text("one")]),
        summary("c1", &[seq(0)], 5),
        checkpoint("c1", "cp1"),
        user("u2", vec![text("two")]),
        summary("c2", &[seq(3)], 7),
        checkpoint("c2", "cp2"),
    ];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(
        surface
            .nodes()
            .iter()
            .map(|node| node.seq())
            .collect::<Vec<_>>(),
        vec![seq(2), seq(5)]
    );
    assert_eq!(surface.replace_generation(), 2);
    assert_eq!(surface.shadowed_token_count(), 12);
}

#[test]
fn a_tool_calling_assistant_contributes_one_open_call_per_block() {
    let events = [assistant_with_calls("a", &["c1", "c2"])];
    let surface = fold_surface(&events).unwrap();
    assert_eq!(
        surface.nodes()[0].kind(),
        SurfaceNodeKind::Assistant { tool_calls: 2 }
    );
}
