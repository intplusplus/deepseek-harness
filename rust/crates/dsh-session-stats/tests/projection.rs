//! `sessionStats` fold behavior: step/turn authority, tool pairing and
//! orphans, token and route accounting, missing usage, saturation, replay
//! equivalence, checkpoint seeding, and lossless JSON state.

use std::sync::Arc;

use dsh_llm::{
    AssistantMessage, ContentBlock, MessageId, MessageSource, TokenUsage, ToolCallId,
    ToolResultMessage,
};
use dsh_session::{
    Session, SessionEvent, SessionId, SessionLogOffset, SessionMeta, SessionSeq, ToolErrorInfo,
    TurnEndReason, MAX_SAFE_INTEGER,
};
use dsh_session_projection::{
    ProjectionCheckpoint, ProjectionCheckpointRow, ProjectionCursor, ProjectionError,
    ProjectionRegistry,
};
use dsh_session_stats::{
    stats_projection, SessionStatsState, SessionStatsView, KEY, STATE_VERSION,
};
use serde_json::{json, Map, Value};

fn session(id: &str) -> Session {
    Session::create(SessionId::new(id).unwrap(), SessionMeta::default(), 0)
}

fn offset(value: u64) -> SessionLogOffset {
    SessionLogOffset::new(value).unwrap()
}

fn seq(value: u64) -> SessionSeq {
    SessionSeq::new(value).unwrap()
}

fn append(session: &mut Session, event: SessionEvent) -> SessionSeq {
    session.append(event).unwrap()
}

fn usage(
    input: u64,
    output: u64,
    cache_read: Option<u64>,
    cache_write: Option<u64>,
    reasoning: Option<u64>,
) -> TokenUsage {
    TokenUsage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: None,
        cache_read_tokens: cache_read,
        cache_write_tokens: cache_write,
        reasoning_tokens: reasoning,
    }
}

fn assistant(
    turn: u64,
    step: u64,
    provider: &str,
    model: &str,
    usage: Option<TokenUsage>,
) -> SessionEvent {
    SessionEvent::assistant_message(
        turn,
        step,
        AssistantMessage::new(
            MessageId::new(format!("assistant-{turn}-{step}")),
            vec![ContentBlock::text("answer")],
            provider,
            model,
        ),
        vec![],
        usage,
        false,
    )
}

fn tool_call(turn: u64, step: u64, id: &str) -> SessionEvent {
    SessionEvent::tool_call(turn, step, ToolCallId::new(id), "echo", "{}")
}

fn tool_result(turn: u64, step: u64, id: &str, is_error: bool) -> SessionEvent {
    SessionEvent::tool_result(
        turn,
        step,
        ToolResultMessage::new(
            MessageId::new(format!("result-{id}")),
            vec![ContentBlock::text("done")],
            ToolCallId::new(id),
            is_error,
        ),
    )
}

fn tool_failure(turn: u64, step: u64, id: &str) -> SessionEvent {
    SessionEvent::tool_error(
        turn,
        step,
        ToolResultMessage::new(
            MessageId::new(format!("failure-{id}")),
            vec![ContentBlock::text("failed")],
            ToolCallId::new(id),
            false,
        ),
        ToolErrorInfo {
            name: "echo".to_string(),
            code: "FAILED".to_string(),
            reason: None,
        },
    )
}

fn stats(registry: &ProjectionRegistry, session: &Session) -> Value {
    registry
        .snapshot(session, offset(0))
        .unwrap()
        .values
        .get(KEY)
        .unwrap()
        .clone()
}

fn state(registry: &ProjectionRegistry, session: &Session) -> Arc<SessionStatsState> {
    registry
        .state_of::<SessionStatsState>(session, offset(0), KEY)
        .unwrap()
        .unwrap()
}

fn row(value: Value) -> ProjectionCheckpointRow {
    ProjectionCheckpointRow {
        ver: STATE_VERSION,
        seq: ProjectionCursor::BEFORE_LOG,
        val: value,
    }
}

fn checkpoint(value: Value) -> ProjectionCheckpoint {
    [(KEY.to_string(), row(value))].into_iter().collect()
}

#[test]
fn an_empty_log_serves_the_zeroed_view_with_typescript_field_order() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let session = session("stats-empty");
    assert_eq!(KEY, "sessionStats");
    assert_eq!(STATE_VERSION, 2);

    let value = stats(&registry, &session);
    assert_eq!(
        serde_json::to_string(&value).unwrap(),
        concat!(
            r#"{"turns":0,"steps":0,"toolCalls":0,"toolResults":0,"toolErrors":0,"#,
            r#""orphanToolResults":0,"assistantMessages":0,"usageMessages":0,"#,
            r#""missingUsageMessages":0,"inputTokens":0,"outputTokens":0,"#,
            r#""cacheReadTokens":0,"cacheWriteTokens":0,"reasoningTokens":0,"#,
            r#""totalTokens":0,"byProvider":{}}"#
        )
    );

    let rows = registry.checkpoint(&session, offset(0)).unwrap();
    assert_eq!(
        rows.get(KEY).unwrap().val,
        serde_json::to_value(SessionStatsState::default()).unwrap()
    );
}

#[test]
fn step_end_counts_distinct_turns_and_every_closed_step() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let mut session = session("stats-steps");
    append(&mut session, SessionEvent::step_start(1, 1));
    append(&mut session, assistant(1, 1, "alpha", "m1", None));
    append(&mut session, SessionEvent::step_end(1, 1));
    append(&mut session, SessionEvent::step_end(1, 2));
    append(&mut session, SessionEvent::step_end(2, 1));
    append(&mut session, SessionEvent::step_end(2, 2));
    append(&mut session, SessionEvent::step_end(3, 1));

    let view = stats(&registry, &session);
    assert_eq!(view["turns"], json!(3));
    assert_eq!(
        view["steps"],
        json!(5),
        "one step/end per closed step, not one per message"
    );
    let state = state(&registry, &session);
    assert_eq!(state.last_turn, Some(3));
}

#[test]
fn tool_calls_results_failures_and_orphans_are_counted() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let mut session = session("stats-tools");
    append(&mut session, tool_call(1, 1, "a"));
    append(&mut session, tool_result(1, 1, "a", false));
    append(&mut session, tool_call(1, 2, "b"));
    append(&mut session, tool_result(1, 2, "b", true));
    append(&mut session, tool_result(1, 2, "orphan", false));
    append(&mut session, tool_call(1, 3, "c"));
    append(&mut session, tool_failure(1, 3, "c"));
    append(&mut session, tool_call(1, 4, "pending"));
    append(
        &mut session,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );

    let view = stats(&registry, &session);
    assert_eq!(view["toolCalls"], json!(4));
    assert_eq!(view["toolResults"], json!(4));
    assert_eq!(
        view["toolErrors"],
        json!(2),
        "both the isError flag and error info count"
    );
    assert_eq!(view["orphanToolResults"], json!(1));
    assert!(state(&registry, &session).pending_calls.is_empty());
}

#[test]
fn turn_end_without_pending_calls_keeps_the_state_allocation() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let mut session = session("stats-pending-identity");
    append(
        &mut session,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );
    append(
        &mut session,
        SessionEvent::turn_end(2, TurnEndReason::Completed),
    );
    let changes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&changes);
    let _subscription = registry.on_changed(move |_change| {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });

    registry.drive_event(&session, offset(0), seq(0)).unwrap();
    let before = state(&registry, &session);
    registry.drive_event(&session, offset(0), seq(1)).unwrap();
    let after = state(&registry, &session);

    assert!(Arc::ptr_eq(&before, &after));
    assert_eq!(changes.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn a_turn_end_drops_unsettled_calls_and_a_late_result_becomes_an_orphan() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let mut session = session("stats-cancelled-tool");
    append(&mut session, tool_call(1, 1, "a"));
    append(
        &mut session,
        SessionEvent::turn_end(
            1,
            TurnEndReason::Aborted {
                reason: dsh_session::TurnEndCancelCause::User,
            },
        ),
    );
    append(&mut session, tool_result(1, 1, "a", false));

    let view = stats(&registry, &session);
    assert_eq!(view["toolCalls"], json!(1));
    assert_eq!(view["toolResults"], json!(1));
    assert_eq!(view["orphanToolResults"], json!(1));
    assert!(state(&registry, &session).pending_calls.is_empty());
}

#[test]
fn assistant_messages_aggregate_disjoint_tokens_and_route_totals() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let mut session = session("stats-tokens");
    append(
        &mut session,
        assistant(
            1,
            1,
            "alpha",
            "m1",
            Some(usage(10, 4, Some(3), Some(1), Some(2))),
        ),
    );
    append(
        &mut session,
        assistant(1, 2, "alpha", "m1", Some(usage(5, 2, None, None, None))),
    );
    append(
        &mut session,
        assistant(
            1,
            3,
            "beta",
            "m2",
            Some(TokenUsage {
                total_tokens: Some(999),
                ..usage(1, 1, Some(0), Some(0), None)
            }),
        ),
    );
    append(&mut session, assistant(2, 1, "beta", "m2", None));
    append(
        &mut session,
        SessionEvent::assistant_message(
            2,
            2,
            AssistantMessage {
                id: MessageId::new("unrouted"),
                content: vec![ContentBlock::text("synthetic")],
                source: MessageSource::SystemPrompt,
            },
            vec![],
            Some(usage(2, 2, None, None, None)),
            false,
        ),
    );

    let view = stats(&registry, &session);
    assert_eq!(view["assistantMessages"], json!(5));
    assert_eq!(view["usageMessages"], json!(4));
    assert_eq!(view["missingUsageMessages"], json!(1));
    assert_eq!(view["inputTokens"], json!(18));
    assert_eq!(view["outputTokens"], json!(9));
    assert_eq!(view["cacheReadTokens"], json!(3));
    assert_eq!(view["cacheWriteTokens"], json!(1));
    assert_eq!(view["reasoningTokens"], json!(2));
    assert_eq!(
        view["totalTokens"],
        json!(31),
        "the total is disjoint components, not the provider-reported totalTokens"
    );

    let routes = view["byProvider"].as_object().unwrap();
    let providers: Vec<&str> = routes.keys().map(String::as_str).collect();
    assert_eq!(
        providers,
        vec!["alpha", "beta"],
        "providers and models are sorted"
    );
    assert_eq!(
        routes["alpha"]["m1"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec![
            "messages",
            "usageMessages",
            "missingUsageMessages",
            "inputTokens",
            "outputTokens",
            "cacheReadTokens",
            "cacheWriteTokens",
            "reasoningTokens",
            "totalTokens"
        ]
    );
    assert_eq!(
        routes["alpha"]["m1"],
        json!({
            "messages": 2,
            "usageMessages": 2,
            "missingUsageMessages": 0,
            "inputTokens": 15,
            "outputTokens": 6,
            "cacheReadTokens": 3,
            "cacheWriteTokens": 1,
            "reasoningTokens": 2,
            "totalTokens": 25
        })
    );
    assert_eq!(
        routes["beta"]["m2"],
        json!({
            "messages": 2,
            "usageMessages": 1,
            "missingUsageMessages": 1,
            "inputTokens": 1,
            "outputTokens": 1,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
            "reasoningTokens": 0,
            "totalTokens": 2
        })
    );
    assert_eq!(
        state(&registry, &session)
            .by_provider
            .get("alpha")
            .unwrap()
            .get("m1")
            .unwrap()
            .messages,
        2,
        "an unrouted message contributes no provider/model row"
    );
}

#[test]
fn missing_usage_is_a_reported_gap_not_a_zero_usage_record() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let mut session = session("stats-missing-usage");
    append(&mut session, assistant(1, 1, "gamma", "m3", None));

    let view = stats(&registry, &session);
    assert_eq!(view["assistantMessages"], json!(1));
    assert_eq!(view["usageMessages"], json!(0));
    assert_eq!(view["missingUsageMessages"], json!(1));
    assert_eq!(view["totalTokens"], json!(0));
    assert_eq!(view["byProvider"]["gamma"]["m3"]["messages"], json!(1));
    assert_eq!(view["byProvider"]["gamma"]["m3"]["usageMessages"], json!(0));
    assert_eq!(
        view["byProvider"]["gamma"]["m3"]["missingUsageMessages"],
        json!(1)
    );
}

#[test]
fn unrelated_events_keep_the_state_allocation_and_stay_quiet() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let mut session = session("stats-unrelated");
    let changes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&changes);
    let _subscription = registry.on_changed(move |_change| {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });

    append(&mut session, SessionEvent::turn_start(1));
    registry.drive_event(&session, offset(0), seq(0)).unwrap();
    let before = state(&registry, &session);

    let unrelated = vec![
        SessionEvent::step_start(1, 1),
        SessionEvent::user_message(dsh_llm::UserMessage::new(
            MessageId::new("user"),
            vec![ContentBlock::text("prompt")],
        )),
        SessionEvent::extension_event("compaction/start", Map::new()),
    ];
    for (index, event) in unrelated.into_iter().enumerate() {
        let position = append(&mut session, event);
        assert_eq!(position, seq(index as u64 + 1));
        registry.drive_event(&session, offset(0), position).unwrap();
        let current = state(&registry, &session);
        assert!(
            Arc::ptr_eq(&before, &current),
            "seq {position} is not a stats event"
        );
    }
    assert_eq!(changes.load(std::sync::atomic::Ordering::Relaxed), 0);

    append(&mut session, SessionEvent::step_end(1, 1));
    registry.drive_event(&session, offset(0), seq(4)).unwrap();
    assert_eq!(changes.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn replay_and_incremental_drive_produce_the_same_view_and_checkpoint() {
    let events = vec![
        SessionEvent::turn_start(1),
        SessionEvent::step_start(1, 1),
        tool_call(1, 1, "a"),
        tool_result(1, 1, "a", false),
        assistant(
            1,
            1,
            "alpha",
            "m1",
            Some(usage(10, 4, Some(3), Some(1), Some(2))),
        ),
        SessionEvent::step_end(1, 1),
        SessionEvent::turn_end(1, TurnEndReason::Completed),
        SessionEvent::turn_start(2),
        assistant(2, 1, "beta", "m2", None),
        SessionEvent::step_end(2, 1),
        SessionEvent::turn_end(2, TurnEndReason::Completed),
    ];
    let replayed_registry = ProjectionRegistry::new();
    let _replayed = replayed_registry.register(stats_projection()).unwrap();
    let mut replayed = session("stats-replay");
    for event in events.clone() {
        append(&mut replayed, event);
    }
    let replayed_view = stats(&replayed_registry, &replayed);
    let replayed_rows = replayed_registry.checkpoint(&replayed, offset(0)).unwrap();

    let driven_registry = ProjectionRegistry::new();
    let _driven = driven_registry.register(stats_projection()).unwrap();
    let mut driven = session("stats-driven");
    for event in events {
        let position = append(&mut driven, event);
        driven_registry
            .drive_event(&driven, offset(0), position)
            .unwrap();
    }
    let driven_view = stats(&driven_registry, &driven);
    let driven_rows = driven_registry.checkpoint(&driven, offset(0)).unwrap();

    assert_eq!(driven_view, replayed_view);
    assert_eq!(driven_rows, replayed_rows);
    assert_eq!(replayed_view["turns"], json!(2));
    assert_eq!(replayed_view["steps"], json!(2));
}

#[test]
fn restore_seeds_stats_and_folds_only_the_tail() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let mut session = session("stats-restore");
    append(&mut session, SessionEvent::step_end(1, 1));
    append(&mut session, tool_call(1, 2, "a"));
    let rows = registry.checkpoint(&session, offset(0)).unwrap();
    append(&mut session, SessionEvent::step_end(1, 3));
    append(&mut session, tool_result(1, 3, "a", false));
    let expected = stats(&registry, &session);

    let restored = registry
        .restore(
            &rows,
            &session.events()[2..],
            offset(2),
            session.header(),
            offset(0),
        )
        .unwrap();

    assert_eq!(restored.snapshot.values.get(KEY), Some(&expected));
    assert_eq!(restored.snapshot.as_of_seq, ProjectionCursor::at(seq(3)));
    assert_eq!(
        restored.checkpoint.get(KEY).unwrap().seq,
        ProjectionCursor::at(seq(3))
    );
    assert_eq!(restored.checkpoint.get(KEY).unwrap().val["steps"], json!(2));
    assert_eq!(
        restored.checkpoint.get(KEY).unwrap().val["toolCalls"],
        json!(1)
    );
    assert_eq!(
        restored.checkpoint.get(KEY).unwrap().val["toolResults"],
        json!(1)
    );
    assert_eq!(restored.checkpoint.get(KEY).unwrap().val["turns"], json!(1));
}

#[test]
fn counters_saturate_at_the_safe_integer_bound() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let seeded = SessionStatsState {
        tool_calls: MAX_SAFE_INTEGER - 1,
        input_tokens: MAX_SAFE_INTEGER - 1,
        total_tokens: MAX_SAFE_INTEGER - 1,
        ..SessionStatsState::default()
    };
    let rows = checkpoint(serde_json::to_value(&seeded).unwrap());
    let events = vec![
        tool_call(1, 1, "a"),
        tool_call(1, 1, "b"),
        assistant(1, 1, "alpha", "m1", Some(usage(10, 0, None, None, None))),
    ];

    let restored = registry
        .restore(
            &rows,
            &events,
            offset(0),
            session("stats-saturated").header(),
            offset(0),
        )
        .unwrap();
    let value = &restored.checkpoint.get(KEY).unwrap().val;

    assert_eq!(value["toolCalls"], json!(MAX_SAFE_INTEGER));
    assert_eq!(value["inputTokens"], json!(MAX_SAFE_INTEGER));
    assert_eq!(value["totalTokens"], json!(MAX_SAFE_INTEGER));
    assert_eq!(value["byProvider"]["alpha"]["m1"]["totalTokens"], json!(10));
}

#[test]
fn a_persisted_row_that_is_not_lossless_json_state_is_rejected() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let mut value = serde_json::to_value(SessionStatsState::default()).unwrap();
    value["turns"] = json!(1.5);
    let rows = checkpoint(value);

    assert!(registry.view_checkpoint(&rows, None).unwrap().is_empty());
    let error = registry
        .restore(
            &rows,
            &[],
            offset(0),
            session("stats-invalid-row").header(),
            offset(0),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::InvalidState {
            key: "sessionStats",
            ..
        }
    ));
}

#[test]
fn the_state_round_trips_through_json_after_a_real_fold() {
    let registry = ProjectionRegistry::new();
    let _unit = registry.register(stats_projection()).unwrap();
    let mut session = session("stats-json");
    append(&mut session, tool_call(1, 1, "a"));
    append(&mut session, tool_result(1, 1, "a", false));
    append(
        &mut session,
        assistant(
            1,
            1,
            "alpha",
            "m1",
            Some(usage(10, 4, Some(3), Some(1), Some(2))),
        ),
    );
    append(&mut session, SessionEvent::step_end(1, 1));
    append(
        &mut session,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );

    let state = state(&registry, &session);
    let encoded = serde_json::to_value(&*state).unwrap();
    let decoded: SessionStatsState = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, *state);

    let view: SessionStatsView = serde_json::from_value(stats(&registry, &session)).unwrap();
    assert_eq!(view.turns, 1);
    assert_eq!(view.steps, 1);
    assert_eq!(view.total_tokens, 18);
}
