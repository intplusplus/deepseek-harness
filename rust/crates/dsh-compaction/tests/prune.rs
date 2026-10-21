//! Deterministic tool-result pruning: budgets, code-point slicing, and plans.
//!
//! Ported from `packages/compaction/compaction-tool-result-pruner/tests/tool-result-pruner.spec.ts`.

mod common;

use common::{assistant_with_calls, prune, seq, text, tool_result, tool_result_with_details};
use dsh_compaction::{
    fold_surface, measure_content, plan_prune_session, prune_content, resolve_prune_config,
    CompactionError, PruneError, ResolvedPruneConfig, SurfaceError, ToolResultPruneConfig,
    DEFAULT_HEAD_CHARS, DEFAULT_TAIL_CHARS, DEFAULT_THRESHOLD_CHARS, PRUNE_MARKER,
};
use dsh_llm::{ContentBlock, ToolCallId};

fn config(threshold_chars: u64, head_chars: u64, tail_chars: u64) -> ResolvedPruneConfig {
    ResolvedPruneConfig {
        threshold_chars,
        head_chars,
        tail_chars,
    }
}

#[test]
fn resolves_detached_defaults_and_partial_overrides() {
    assert_eq!(
        resolve_prune_config(&ToolResultPruneConfig::default()).unwrap(),
        config(
            DEFAULT_THRESHOLD_CHARS,
            DEFAULT_HEAD_CHARS,
            DEFAULT_TAIL_CHARS
        )
    );
    assert_eq!(
        resolve_prune_config(&ToolResultPruneConfig {
            threshold_chars: Some(100),
            head_chars: Some(20),
            tail_chars: Some(10),
        })
        .unwrap(),
        config(100, 20, 10)
    );
}

#[test]
fn rejects_a_zero_threshold() {
    let error = resolve_prune_config(&ToolResultPruneConfig {
        threshold_chars: Some(0),
        ..ToolResultPruneConfig::default()
    })
    .unwrap_err();
    assert!(matches!(error, PruneError::Config(_)));
    assert!(error
        .to_string()
        .contains("ToolResultPruneConfig: thresholdChars (0) must be a positive integer"));
}

#[test]
fn rejects_an_output_budget_above_the_threshold() {
    let error = resolve_prune_config(&ToolResultPruneConfig {
        threshold_chars: Some(50),
        head_chars: Some(20),
        tail_chars: Some(20),
    })
    .unwrap_err();
    assert!(error.to_string().contains("headChars + marker + tailChars"));
    assert!(error
        .to_string()
        .contains("must be at most thresholdChars (50)"));
}

#[test]
fn rejects_unknown_config_keys_at_the_deserialization_boundary() {
    let error =
        serde_json::from_value::<ToolResultPruneConfig>(serde_json::json!({ "threshold": 10 }))
            .unwrap_err();
    assert!(error.to_string().contains("unknown field `threshold`"));
}

#[test]
fn measures_text_code_points_only() {
    let blocks = [text("a😀b"), ContentBlock::reasoning("not measured")];
    assert_eq!(measure_content(&blocks), 3);
}

#[test]
fn keeps_configured_head_and_tail_without_splitting_surrogate_pairs() {
    let pruned = prune_content(&[text("😀".repeat(60))], &config(50, 4, 3))
        .unwrap()
        .unwrap();
    assert_eq!(pruned.len(), 1);
    assert_eq!(
        pruned[0].as_text(),
        Some(format!("{}{PRUNE_MARKER}{}", "😀".repeat(4), "😀".repeat(3)).as_str())
    );
    assert!(measure_content(&pruned) <= 50);
    assert!(!pruned[0].as_text().unwrap().contains('\u{FFFD}'));
}

#[test]
fn preserves_non_text_blocks_and_their_relative_order() {
    let reasoning = ContentBlock::reasoning("private-rich-block");
    let call = ContentBlock::tool_call(ToolCallId::new("nested"), "nested", "{}");
    let blocks = [
        text("A".repeat(40)),
        reasoning.clone(),
        text("B".repeat(30)),
        call.clone(),
        text("C".repeat(30)),
    ];
    let pruned = prune_content(&blocks, &config(50, 4, 3)).unwrap().unwrap();
    assert_eq!(
        pruned,
        vec![
            text(format!("AAAA{PRUNE_MARKER}")),
            reasoning,
            call,
            text("CCC"),
        ]
    );
    assert!(measure_content(&pruned) <= 50);
}

#[test]
fn supports_zero_sized_head_and_tail_while_still_shrinking() {
    let marker_chars = PRUNE_MARKER.chars().count() as u64;
    let pruned = prune_content(&[text("x".repeat(100))], &config(marker_chars, 0, 0))
        .unwrap()
        .unwrap();
    assert_eq!(pruned, vec![text(PRUNE_MARKER)]);
    assert_eq!(measure_content(&pruned), marker_chars);
}

#[test]
fn skips_content_at_or_below_the_threshold() {
    assert_eq!(
        prune_content(&[text("x".repeat(50))], &config(50, 4, 3)).unwrap(),
        None
    );
    assert_eq!(
        prune_content(&[text("short")], &config(50, 4, 3)).unwrap(),
        None
    );
}

#[test]
fn rejects_an_invalid_budget_pair_that_cannot_shrink() {
    let error = prune_content(&[text("x".repeat(300))], &config(40, 100, 100)).unwrap_err();
    assert!(matches!(error, PruneError::ReplacementNotSmaller));
}

#[test]
fn plans_replacements_for_over_budget_results() {
    let events = [
        assistant_with_calls("a", &["c1"]),
        tool_result_with_details("r", "c1", vec![text("x".repeat(100))]),
    ];
    let surface = fold_surface(&events).unwrap();
    let plan = plan_prune_session(&surface, &events, &config(50, 4, 3)).unwrap();
    assert_eq!(plan.decisions.len(), 1);
    let decision = &plan.decisions[0];
    assert_eq!(decision.original_seq, seq(1));
    assert_eq!(decision.call_id, ToolCallId::new("c1"));
    assert_eq!(decision.chars_before, 100);
    assert!(decision.chars_after <= 50);
    assert_eq!(
        plan.chars_removed,
        decision.chars_before - decision.chars_after
    );
    assert_eq!(decision.replacement.tool_call_id, ToolCallId::new("c1"));
    assert!(decision.replacement.is_error);
    assert_eq!(decision.replacement.id.as_str(), "r");
    assert_eq!(decision.replacement.source.kind(), "tool");
    let content = decision.replacement.content[0].as_text().unwrap();
    assert!(content.starts_with("xxxx"));
    assert!(content.contains(PRUNE_MARKER));
    assert!(content.ends_with("xxx"));
}

#[test]
fn plan_prune_session_skips_short_results() {
    let events = [
        assistant_with_calls("a", &["c1", "c2"]),
        tool_result("r1", "c1", vec![text("A".repeat(100))]),
        tool_result("r2", "c2", vec![text("short")]),
    ];
    let surface = fold_surface(&events).unwrap();
    let plan = plan_prune_session(&surface, &events, &config(50, 4, 3)).unwrap();
    assert_eq!(plan.decisions.len(), 1);
    assert_eq!(plan.decisions[0].original_seq, seq(1));
}

#[test]
fn a_second_pass_over_the_replacement_surface_is_a_no_op() {
    let original_content = vec![text("A".repeat(100))];
    let pruned_content = prune_content(&original_content, &config(50, 4, 3))
        .unwrap()
        .unwrap();
    let events = [
        assistant_with_calls("a", &["c1"]),
        tool_result("r", "c1", original_content),
        prune(seq(1), 7),
        tool_result("r", "c1", pruned_content),
    ];
    let surface = fold_surface(&events).unwrap();
    assert!(!surface.contains(seq(1)));
    assert!(surface.contains(seq(3)));
    let plan = plan_prune_session(&surface, &events, &config(50, 4, 3)).unwrap();
    assert!(plan.decisions.is_empty());
    assert_eq!(plan.chars_removed, 0);
}

#[test]
fn a_decision_builds_the_adjacent_metering_record() {
    let events = [
        assistant_with_calls("a", &["c1"]),
        tool_result("r", "c1", vec![text("A".repeat(100))]),
    ];
    let surface = fold_surface(&events).unwrap();
    let plan = plan_prune_session(&surface, &events, &config(50, 4, 3)).unwrap();
    let decision = &plan.decisions[0];
    let range = decision.shadowed_range();
    assert_eq!(range.start, seq(1));
    assert_eq!(range.end, seq(1));
    let event = decision.prune_event(9);
    assert_eq!(event.shadowed_range, range);
    assert_eq!(event.shadowed_seqs, vec![seq(1)]);
    assert_eq!(event.shadowed_token_count, 9);
}

#[test]
fn plan_errors_when_a_surface_node_has_no_event() {
    let events = [
        assistant_with_calls("a", &["c1"]),
        tool_result("r", "c1", vec![text("A".repeat(100))]),
    ];
    let surface = fold_surface(&events).unwrap();
    let error = plan_prune_session(&surface, &events[..1], &config(50, 4, 3)).unwrap_err();
    assert!(matches!(
        error,
        CompactionError::Surface(SurfaceError::MissingSurfaceEvent { seq: at }) if at == seq(1)
    ));
}
