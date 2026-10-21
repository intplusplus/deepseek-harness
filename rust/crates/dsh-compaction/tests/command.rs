//! Manual compaction command integration: the full lifecycle produces the
//! correct durable events and reduces the replayed surface, and the pure trigger
//! decision plus `CompactionService` are wired end to end.

mod common;

use std::sync::Arc;

use common::{assistant, seq, system, user};
use dsh_compaction::{
    compact, should_compact, CompactionEvent, CompactionService, CompactOptions, LlmSummarizationBackend,
    ResolvedCompactSpec, DEFAULT_LEASE_MS,
};
use dsh_llm::ContentBlock;
use dsh_provider::{FnProvider, ScriptedProvider, text_response};
use dsh_session::{fold_surface, SessionEvent};

/// A small balanced surface: system head + two user/assistant pairs.
fn session_with_old_history() -> Vec<SessionEvent> {
    vec![
        system("s0"),
        user("u1", vec![ContentBlock::text("first question")]),
        assistant("a1"),
        user("u2", vec![ContentBlock::text("second question")]),
        assistant("a2"),
    ]
}

#[tokio::test]
async fn manual_compact_emits_the_full_event_chain() {
    let mut events = session_with_old_history();
    let base_len = events.len();

    let backend = LlmSummarizationBackend::new(Arc::new(FnProvider::new(|_| {
        Ok(text_response("mock", "mock", "## Current Work\n- consolidated checkpoint"))
    })));

    let options = CompactOptions {
        manual: true,
        source_command_id: None,
        now: 1_000,
        lease_ms: DEFAULT_LEASE_MS,
        retain_tokens: 0,
        provider: "mock".to_string(),
        model: "mock".to_string(),
        max_tokens: Some(128),
        prune_config: None,
    };

    let plan = compact(&events, &backend, config(), &options)
        .await
        .unwrap()
        .expect("a compactable range exists");

    // Order: start, summary, checkpoint, end (no prune phase).
    assert_eq!(plan.prune_events.len(), 0);
    assert!(matches!(
        CompactionEvent::decode(&plan.start),
        Some(Ok(CompactionEvent::Start(_)))
    ));
    let summary = match CompactionEvent::decode(&plan.summary) {
        Some(Ok(CompactionEvent::Summary(s))) => s,
        other => panic!("expected a summary event, got {other:?}"),
    };
    assert!(matches!(
        CompactionEvent::decode(&plan.end),
        Some(Ok(CompactionEvent::End(_)))
    ));

    // The summary shadows the three old content nodes (user1, assistant1, user2).
    assert_eq!(summary.shadowed_seqs, vec![seq(1), seq(2), seq(3)]);

    // The produced events append after the original log in this order.
    events.push(plan.start);
    events.push(plan.summary);
    events.push(plan.checkpoint);
    events.push(plan.end);
    assert_eq!(events.len(), base_len + 4);

    // Folding the combined log shows the replacement collapsed the surface.
    let folded = fold_surface(&events).unwrap();
    // Original surface had 5 nodes; the 3 shadowed nodes became 1 checkpoint.
    assert_eq!(folded.nodes().len(), 3);
    // The resolution records the position-relative sequence numbers.
    assert_eq!(plan.result.start_seq, seq(base_len as u64));
}

#[tokio::test]
async fn manual_compact_with_prune_phase_emits_prune_events() {
    // assistant opens a call, tool result is huge -> pruned; plus a trailing
    // user/assistant pair so a summary range still exists after pruning.
    let mut events = vec![
        system("s0"),
        common::assistant_with_calls("a1", &["c1"]),
        common::tool_result(
            "r1",
            "c1",
            vec![ContentBlock::text("x".repeat(20_000))],
        ),
        user("u1", vec![ContentBlock::text("follow-up")]),
        assistant("a2"),
    ];

    let backend = LlmSummarizationBackend::new(Arc::new(ScriptedProvider::new(vec![
        text_response("mock", "mock", "## Current Work\n- pruned and summarized"),
    ])));

    let options = CompactOptions {
        manual: true,
        source_command_id: None,
        now: 1_000,
        lease_ms: DEFAULT_LEASE_MS,
        retain_tokens: 0,
        provider: "mock".to_string(),
        model: "mock".to_string(),
        max_tokens: Some(128),
        prune_config: Some(dsh_compaction::ResolvedPruneConfig {
            threshold_chars: 8_192,
            head_chars: 4_096,
            tail_chars: 1_024,
        }),
    };

    let plan = compact(&events, &backend, config(), &options)
        .await
        .unwrap()
        .expect("a compactable range exists");

    // Prune phase emits one compaction/prune + one rewritten tool/result.
    assert_eq!(plan.prune_events.len(), 2);
    assert!(matches!(
        CompactionEvent::decode(&plan.prune_events[0]),
        Some(Ok(CompactionEvent::Prune(_)))
    ));
    assert!(matches!(
        &plan.prune_events[1],
        SessionEvent::Core(dsh_session::CoreEvent::ToolResult { .. })
    ));

    events.push(plan.start);
    events.extend(plan.prune_events);
    events.push(plan.summary);
    events.push(plan.checkpoint);
    events.push(plan.end);

    let folded = fold_surface(&events).unwrap();
    // 5 original nodes -> 1 pruned replacement + 3 summary-shadowed nodes -> fewer.
    assert!(folded.nodes().len() < 5);
}

#[test]
fn should_compact_is_a_pure_threshold_decision() {
    let spec = ResolvedCompactSpec {
        target: dsh_compaction::ModelTarget::new("mock", "mock"),
        context_window: 100_000,
        threshold_ratio: 0.8,
        threshold_tokens: 80_000,
        retain_tokens: 16_000,
        summarization_provider: String::new(),
        summarization_model: String::new(),
        max_tokens: 65_536,
        compaction_retries: 1,
        max_overflow_retries: 1,
    };
    assert!(!should_compact(79_999, &spec));
    assert!(should_compact(80_000, &spec));
    assert!(should_compact(90_000, &spec));
}

#[tokio::test]
async fn compaction_service_runs_a_manual_transaction() {
    let backend = LlmSummarizationBackend::new(Arc::new(ScriptedProvider::new(vec![
        text_response("mock", "mock", "## Current Work\n- service-driven checkpoint"),
    ])));
    let service = CompactionService::new(
        Arc::new(backend),
        config(),
        "mock",
        "mock",
    );

    let events = session_with_old_history();
    let plan = service
        .compact_now(None, None, 1_000, DEFAULT_LEASE_MS, None)
        .await
        .unwrap()
        .expect("a compactable range exists");

    assert!(matches!(
        CompactionEvent::decode(&plan.start),
        Some(Ok(CompactionEvent::Start(_)))
    ));
    // The service reports the pure trigger decision consistently.
    let spec = ResolvedCompactSpec {
        target: dsh_compaction::ModelTarget::new("mock", "mock"),
        context_window: 100_000,
        threshold_ratio: 0.8,
        threshold_tokens: 80_000,
        retain_tokens: 16_000,
        summarization_provider: String::new(),
        summarization_model: String::new(),
        max_tokens: 65_536,
        compaction_retries: 1,
        max_overflow_retries: 1,
    };
    assert!(!service.should_compact(10, &spec));
    assert!(service.should_compact(90_000, &spec));
}

#[tokio::test]
async fn compact_returns_none_with_nothing_to_compact() {
    // A surface of only a system head yields no compactable range.
    let events = vec![system("s0")];
    let backend = LlmSummarizationBackend::new(Arc::new(ScriptedProvider::new(vec![
        text_response("mock", "mock", "summary"),
    ])));
    let options = CompactOptions {
        manual: true,
        source_command_id: None,
        now: 1_000,
        lease_ms: DEFAULT_LEASE_MS,
        retain_tokens: 0,
        provider: "mock".to_string(),
        model: "mock".to_string(),
        max_tokens: Some(128),
        prune_config: None,
    };
    let plan = compact(&events, &backend, config(), &options).await.unwrap();
    assert!(plan.is_none());
}

/// A resolved default policy, mirroring `resolveConfig({})` in TypeScript.
fn config() -> dsh_compaction::ResolvedConfig {
    dsh_compaction::resolve_config(&dsh_compaction::BasicCompactionConfig::default()).unwrap()
}
