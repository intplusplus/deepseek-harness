//! Plan-mode state, projection, narration, and policy behavior, ported from
//! `packages/plan/plan-mode/tests/plan-mode.spec.ts` and
//! `tests/projection.spec.ts`.
//!
//! The TypeScript suite mounts the Cordis service and drives the real agent
//! loop. The rules with observable consequences live in the fold and the state
//! machine, so these tests replay the same events through the pure entry points
//! and assert the decisions, wire values, and exact bytes a caller appends.

use dsh_llm::{
    ContentBlock, CoreContentBlock, LlmCallConfig, MessageId, MessageSource, UserMessage,
};
use dsh_plan::{
    apply_plan_state, decode_plan_mode, encode_plan_mode, fold_plan_state, plan_projection,
    policy_text, tool_restriction, FlushDecision, PendingIntent, PlanMachine, PlanModeConfig,
    PlanModeConfigError, PlanModeEvent, PlanProjection, PlanSetDecision, PlanSetOutcome,
    PlanStatus, PlanUnitState, NARRATION_PLAN_OFF, NARRATION_PLAN_ON, PLAN_MODE_TAG,
    PLAN_POLICY_ORDER, PLAN_POLICY_SECTION,
};
use dsh_session::{EpochHeader, RequestHeaderReason, SessionEvent, TurnEndReason};
use dsh_tools::admits_all;
use serde_json::{json, Map};

fn command_run(id: &str, name: &str, args: Option<&str>) -> SessionEvent {
    let mut payload = Map::new();
    payload.insert("commandId".to_string(), json!(id));
    payload.insert("name".to_string(), json!(name));
    if let Some(args) = args {
        payload.insert("args".to_string(), json!(args));
    }
    payload.insert("source".to_string(), json!({ "kind": "user" }));
    SessionEvent::extension_event("command/run", payload)
}

fn command_done(id: &str, kind: &str) -> SessionEvent {
    let mut payload = Map::new();
    payload.insert("commandId".to_string(), json!(id));
    payload.insert("kind".to_string(), json!(kind));
    SessionEvent::extension_event("command/done", payload)
}

fn plan_mode(active: bool) -> SessionEvent {
    encode_plan_mode(&PlanModeEvent { active })
}

fn header(model: &str) -> SessionEvent {
    SessionEvent::request_header(
        EpochHeader {
            config: LlmCallConfig {
                provider: "test".to_string(),
                model: model.to_string(),
                reasoning_effort: None,
                temperature: None,
                max_tokens: None,
                stop: None,
            },
            adapter_defaults: None,
            tools: None,
        },
        RequestHeaderReason::Initial,
    )
}

fn text_of(message: &UserMessage) -> String {
    match message.content.as_slice() {
        [ContentBlock::Core(CoreContentBlock::Text { text })] => text.clone(),
        other => format!("{other:?}"),
    }
}

#[test]
fn the_config_requires_a_non_empty_section_and_names_unknown_keys() {
    assert_eq!(
        PlanModeConfig::resolve(&json!({})).unwrap_err(),
        PlanModeConfigError::MissingSection
    );
    assert_eq!(
        PlanModeConfig::resolve(&json!({ "section": 5 })).unwrap_err(),
        PlanModeConfigError::MissingSection
    );
    assert_eq!(
        PlanModeConfig::resolve(&json!("policy")).unwrap_err(),
        PlanModeConfigError::MissingSection
    );
    assert_eq!(
        PlanModeConfig::resolve(&json!({ "section": "  \t " })).unwrap_err(),
        PlanModeConfigError::EmptySection
    );
    assert_eq!(
        PlanModeConfig::resolve(&json!({ "section": "\u{FEFF}" })).unwrap_err(),
        PlanModeConfigError::EmptySection
    );
    assert_eq!(
        PlanModeConfigError::MissingSection.to_string(),
        "PlanModeConfig needs a string `section`"
    );
    assert_eq!(
        PlanModeConfigError::EmptySection.to_string(),
        "PlanModeConfig needs a non-empty `section`"
    );

    let error = PlanModeConfig::resolve(&json!({
        "section": "policy",
        "tools": ["read"],
        "extra": true
    }))
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "PlanModeConfig has unknown key(s) tools, extra — config is { section }"
    );

    let raw = json!({ "section": " keep the spacing " });
    let config = PlanModeConfig::resolve(&raw).expect("a string section resolves");
    assert_eq!(config.section, " keep the spacing ");
    assert_eq!(raw, json!({ "section": " keep the spacing " }));
    assert_eq!(PLAN_POLICY_SECTION, "plan:policy");
    assert_eq!(PLAN_POLICY_ORDER, 500.0);
}

#[test]
fn the_empty_log_is_inactive_and_the_last_mode_switch_wins() {
    assert_eq!(
        fold_plan_state(&[]).expect("an empty log folds"),
        PlanUnitState::default()
    );

    let events = [plan_mode(true), plan_mode(false), plan_mode(true)];
    let state = fold_plan_state(&events).expect("mode switches fold");
    assert!(state.active);
    assert_eq!(state.wanted, None);
    assert_eq!(state.running, None);
    assert_eq!(state.active_at_last_header, None);

    let prefix = fold_plan_state(&events[..1]).expect("a prefix folds");
    assert!(prefix.active);
    assert!(!fold_plan_state(&[]).expect("an empty prefix folds").active);
}

#[test]
fn the_projection_pairs_a_plan_selection_with_its_settlement() {
    let mut state = PlanUnitState::default();
    assert_eq!(
        plan_projection(&state),
        PlanProjection {
            active: false,
            pending: false
        }
    );

    let run = command_run("plan-1", "plan", Some(""));
    state = apply_plan_state(state, &run).expect("the run folds");
    assert_eq!(
        plan_projection(&state),
        PlanProjection {
            active: false,
            pending: true
        }
    );

    let settled = command_done("plan-1", "success");
    state = apply_plan_state(state, &settled).expect("the settlement folds");
    assert_eq!(
        plan_projection(&state),
        PlanProjection {
            active: false,
            pending: true
        }
    );

    state = apply_plan_state(state, &plan_mode(true)).expect("the switch folds");
    assert_eq!(
        plan_projection(&state),
        PlanProjection {
            active: true,
            pending: false
        }
    );
}

#[test]
fn a_failed_settlement_or_a_selection_already_in_force_is_not_pending() {
    let mut state =
        apply_plan_state(PlanUnitState::default(), &plan_mode(true)).expect("the switch folds");
    state =
        apply_plan_state(state, &command_run("off-1", "plan", Some("off"))).expect("the run folds");
    assert_eq!(
        plan_projection(&state),
        PlanProjection {
            active: true,
            pending: true
        }
    );

    state = apply_plan_state(state, &command_done("off-1", "error")).expect("the failure folds");
    assert_eq!(
        plan_projection(&state),
        PlanProjection {
            active: true,
            pending: false
        }
    );

    state = apply_plan_state(state, &plan_mode(false)).expect("the switch folds");
    state = apply_plan_state(state, &command_run("off-2", "plan", Some(" off ")))
        .expect("the run folds");
    assert_eq!(
        plan_projection(&state),
        PlanProjection {
            active: false,
            pending: false
        }
    );
}

#[test]
fn only_a_plan_command_with_recorded_input_is_a_selection() {
    let state = PlanUnitState::default();
    let other = command_run("compact-1", "compact", Some(""));
    assert_eq!(
        apply_plan_state(state.clone(), &other).expect("folds"),
        state
    );

    let no_input = command_run("plan-no-input", "plan", None);
    assert_eq!(
        apply_plan_state(state.clone(), &no_input).expect("folds"),
        state
    );

    let message = command_run("plan-message", "plan", Some(" sketch the refactor first "));
    let selected = apply_plan_state(state.clone(), &message).expect("the selection folds");
    assert!(
        selected
            .running
            .as_ref()
            .expect("the run is recorded")
            .wanted
    );
    assert_eq!(
        plan_projection(&selected),
        PlanProjection {
            active: false,
            pending: true
        }
    );

    // NEL is not ECMAScript whitespace, so it is not trimmed away from `off`.
    let nel = command_run("plan-nel", "plan", Some("\u{0085}off"));
    let selected = apply_plan_state(state, &nel).expect("the selection folds");
    assert!(selected.running.expect("the run is recorded").wanted);
}

#[test]
fn the_command_trims_only_ecmascript_whitespace() {
    let mut machine = PlanMachine::new(PlanUnitState::default(), false);
    let outcome = dsh_plan::run_plan_command(&mut machine, "\u{0085}off", false);
    assert_eq!(outcome.result.text, dsh_plan::PLAN_ON_TEXT);
    assert_eq!(outcome.steer, Some("\u{0085}off".to_string()));
}

#[test]
fn a_settlement_only_pairs_with_its_own_command() {
    let state = apply_plan_state(
        PlanUnitState::default(),
        &command_run("plan-1", "plan", Some("")),
    )
    .expect("the run folds");
    let foreign = apply_plan_state(state.clone(), &command_done("other-1", "success"))
        .expect("a foreign settlement folds");
    assert_eq!(foreign, state);

    let stale = apply_plan_state(state.clone(), &command_done("plan-2", "error"))
        .expect("a stale settlement folds");
    assert_eq!(stale, state);
}

#[test]
fn a_request_header_freezes_the_logged_mode() {
    let mut state = PlanUnitState::default();
    state = apply_plan_state(state, &header("test-model")).expect("the header folds");
    assert_eq!(state.active_at_last_header, Some(false));

    state = apply_plan_state(state, &plan_mode(true)).expect("the switch folds");
    state = apply_plan_state(state, &header("test-model")).expect("the header folds");
    assert_eq!(state.active_at_last_header, Some(true));

    state = apply_plan_state(state, &plan_mode(false)).expect("the switch folds");
    assert_eq!(state.active_at_last_header, Some(true));
    state = apply_plan_state(state, &header("test-model")).expect("the header folds");
    assert_eq!(state.active_at_last_header, Some(false));
}

#[test]
fn a_malformed_known_event_stops_the_fold_and_an_unknown_tag_is_ignored() {
    let malformed = SessionEvent::extension_event(
        PLAN_MODE_TAG,
        Map::from_iter([("active".to_string(), json!("yes"))]),
    );
    let error = fold_plan_state(&[malformed]).expect_err("a bad payload is a fold error");
    assert!(error.to_string().starts_with("plan state fold failed: "));

    let unknown = SessionEvent::extension_event(
        "plan/other",
        Map::from_iter([("active".to_string(), json!("yes"))]),
    );
    assert_eq!(
        fold_plan_state(&[unknown]).expect("an unknown tag is ignored"),
        PlanUnitState::default()
    );

    let missing_id = SessionEvent::extension_event(
        "command/run",
        Map::from_iter([("name".to_string(), json!("plan"))]),
    );
    assert!(fold_plan_state(&[missing_id]).is_err());

    let bad_kind = command_done("plan-1", "exploded");
    assert!(fold_plan_state(&[bad_kind]).is_err());
}

#[test]
fn a_mode_switch_serializes_as_the_typescript_row() {
    assert_eq!(
        serde_json::to_string(&plan_mode(true)).expect("the event serializes"),
        r#"{"type":"plan/mode","active":true}"#
    );
    assert_eq!(
        serde_json::to_string(&plan_mode(false)).expect("the event serializes"),
        r#"{"type":"plan/mode","active":false}"#
    );
    let event = plan_mode(true);
    let decoded = decode_plan_mode(&event).expect("the tag matches");
    assert_eq!(
        decoded.expect("the payload matches"),
        PlanModeEvent { active: true }
    );
    assert!(decode_plan_mode(&SessionEvent::turn_start(1)).is_none());

    let bad = SessionEvent::extension_event(
        PLAN_MODE_TAG,
        Map::from_iter([("active".to_string(), json!(1))]),
    );
    assert!(matches!(decode_plan_mode(&bad), Some(Err(_))));
}

#[test]
fn a_selection_between_turns_commits_immediately() {
    let mut machine = PlanMachine::new(PlanUnitState::default(), false);
    let decision = machine.set(true);
    assert_eq!(
        decision,
        PlanSetDecision::Commit {
            active: true,
            narration: None
        }
    );
    assert_eq!(decision.outcome(), PlanSetOutcome::Committed);
    assert_eq!(
        machine.status(),
        PlanStatus {
            active: false,
            pending: None
        }
    );

    machine.commit_set(&decision);
    assert!(machine.active());
    assert_eq!(
        machine.status(),
        PlanStatus {
            active: true,
            pending: None
        }
    );
    assert_eq!(
        machine.projection(),
        PlanProjection {
            active: true,
            pending: false
        }
    );

    let decision = machine.set(false);
    assert_eq!(
        decision,
        PlanSetDecision::Commit {
            active: false,
            narration: None
        }
    );
    machine.commit_set(&decision);
    assert!(!machine.active());
}

#[test]
fn a_repeated_selection_is_a_noop() {
    let mut open = PlanMachine::new(PlanUnitState::default(), true);
    assert_eq!(open.set(false), PlanSetDecision::Noop);
    assert_eq!(open.set(true), PlanSetDecision::Queued);
    assert_eq!(open.set(true), PlanSetDecision::Noop);
    assert_eq!(
        open.status(),
        PlanStatus {
            active: false,
            pending: Some(true)
        }
    );

    let mut idle = PlanMachine::new(PlanUnitState::default(), false);
    assert_eq!(idle.set(false), PlanSetDecision::Noop);
    assert_eq!(idle.pending(), None);
}

#[test]
fn an_open_turn_queues_the_selection_until_the_next_boundary() {
    let mut machine = PlanMachine::new(PlanUnitState::default(), true);
    assert_eq!(machine.set(true), PlanSetDecision::Queued);
    assert_eq!(
        machine.status(),
        PlanStatus {
            active: false,
            pending: Some(true)
        }
    );
    assert_eq!(
        machine.projection(),
        PlanProjection {
            active: false,
            pending: false
        }
    );
    assert_eq!(machine.policy_text("policy"), "policy");

    let flush = machine.decide_flush().expect("the selection is pending");
    assert_eq!(flush.active(), Some(true));
    assert!(!machine.active());

    machine.commit_flush(&flush);
    assert_eq!(
        machine.status(),
        PlanStatus {
            active: true,
            pending: None
        }
    );
    assert_eq!(machine.decide_flush(), None);
}

#[test]
fn a_reversal_during_an_open_turn_cancels_at_the_boundary_without_an_append() {
    let mut machine = PlanMachine::new(PlanUnitState::default(), true);
    assert_eq!(machine.set(true), PlanSetDecision::Queued);
    assert_eq!(machine.set(false), PlanSetDecision::Cancelled);
    assert_eq!(
        machine.status(),
        PlanStatus {
            active: false,
            pending: Some(false)
        }
    );

    let flush = machine
        .decide_flush()
        .expect("the cancelled selection still waits");
    assert_eq!(flush, FlushDecision::ClearPending { narration: None });
    assert_eq!(flush.active(), None);
    machine.commit_flush(&flush);
    assert_eq!(machine.pending(), None);
    assert!(!machine.active());
}

#[test]
fn a_between_turns_reversal_clears_a_parked_intent_without_logging() {
    let state = PlanUnitState::default();
    let mut machine = PlanMachine::new(state, false).with_pending(Some(PendingIntent {
        active: true,
        narrate: true,
    }));
    assert_eq!(machine.set(false), PlanSetDecision::Cancelled);
    assert_eq!(machine.pending(), None);
    assert_eq!(
        machine.status(),
        PlanStatus {
            active: false,
            pending: None
        }
    );
}

#[test]
fn a_commit_narrates_only_when_the_last_header_told_the_other_mode() {
    let told_default = PlanUnitState {
        active_at_last_header: Some(false),
        ..PlanUnitState::default()
    };
    let mut machine = PlanMachine::new(told_default, false);
    let decision = machine.set(true);
    let narration = decision
        .narration()
        .expect("the header told the default mode");
    assert!(narration.active);
    assert_eq!(narration.text, NARRATION_PLAN_ON);

    let told_plan = PlanUnitState {
        active_at_last_header: Some(true),
        ..PlanUnitState::default()
    };
    assert_eq!(
        PlanMachine::new(told_plan, false).set(true),
        PlanSetDecision::Commit {
            active: true,
            narration: None
        }
    );
    assert_eq!(
        PlanMachine::new(PlanUnitState::default(), false).set(true),
        PlanSetDecision::Commit {
            active: true,
            narration: None
        }
    );

    let active_with_default_header = PlanUnitState {
        active: true,
        active_at_last_header: Some(true),
        ..PlanUnitState::default()
    };
    let mut machine = PlanMachine::new(active_with_default_header, false);
    let decision = machine.set(false);
    let narration = decision.narration().expect("the header told plan mode");
    assert!(!narration.active);
    assert_eq!(narration.text, NARRATION_PLAN_OFF);
}

#[test]
fn a_flush_narrates_once_for_a_user_selection() {
    let told_default = PlanUnitState {
        active_at_last_header: Some(false),
        ..PlanUnitState::default()
    };
    let mut machine = PlanMachine::new(told_default, true);
    assert_eq!(machine.set(true), PlanSetDecision::Queued);
    let flush = machine.decide_flush().expect("the selection is pending");
    assert_eq!(
        flush,
        FlushDecision::Append {
            active: true,
            narration: Some(dsh_plan::Narration {
                active: true,
                text: NARRATION_PLAN_ON.to_string(),
            }),
        }
    );
    machine.commit_flush(&flush);
    assert_eq!(machine.decide_flush(), None);
}

#[test]
fn a_net_zero_flip_narrates_nothing_only_when_the_header_agrees() {
    let told_plan = PlanUnitState {
        active_at_last_header: Some(true),
        ..PlanUnitState::default()
    };
    let mut machine = PlanMachine::new(told_plan, true);
    assert_eq!(machine.set(true), PlanSetDecision::Queued);
    assert_eq!(machine.set(false), PlanSetDecision::Cancelled);
    let flush = machine
        .decide_flush()
        .expect("the cancelled selection still waits");
    assert_eq!(
        flush,
        FlushDecision::ClearPending {
            narration: Some(dsh_plan::Narration {
                active: false,
                text: NARRATION_PLAN_OFF.to_string(),
            }),
        }
    );
    machine.commit_flush(&flush);
    assert_eq!(machine.pending(), None);

    let mut quiet = PlanMachine::new(PlanUnitState::default(), true);
    assert_eq!(quiet.set(true), PlanSetDecision::Queued);
    assert_eq!(quiet.set(false), PlanSetDecision::Cancelled);
    assert_eq!(
        quiet.decide_flush(),
        Some(FlushDecision::ClearPending { narration: None })
    );
}

#[test]
fn an_exit_selection_flushes_silently() {
    let told_plan = PlanUnitState {
        active: true,
        active_at_last_header: Some(true),
        ..PlanUnitState::default()
    };
    let mut machine = PlanMachine::new(told_plan, true).with_pending(Some(PendingIntent {
        active: false,
        narrate: false,
    }));
    let flush = machine
        .decide_flush()
        .expect("the exit selection is pending");
    assert_eq!(
        flush,
        FlushDecision::Append {
            active: false,
            narration: None
        }
    );
    machine.commit_flush(&flush);
    assert!(!machine.active());
}

#[test]
fn a_failed_append_leaves_the_intent_parked_for_the_next_boundary() {
    let mut machine = PlanMachine::new(PlanUnitState::default(), true);
    assert_eq!(machine.set(true), PlanSetDecision::Queued);

    let first = machine.decide_flush().expect("the selection is pending");
    assert_eq!(first.active(), Some(true));
    // The caller's append failed, so it never called `commit_flush`.
    assert_eq!(
        machine.status(),
        PlanStatus {
            active: false,
            pending: Some(true)
        }
    );

    let retry = machine.decide_flush().expect("the intent is still parked");
    assert_eq!(retry, first);
    machine.commit_flush(&retry);
    assert_eq!(
        machine.status(),
        PlanStatus {
            active: true,
            pending: None
        }
    );
}

#[test]
fn the_policy_text_follows_the_pending_selection() {
    assert_eq!(policy_text("policy", None, false), "");
    assert_eq!(policy_text("policy", None, true), "policy");
    assert_eq!(policy_text("policy", Some(false), true), "");
    assert_eq!(policy_text("policy", Some(true), false), "policy");

    let machine =
        PlanMachine::new(PlanUnitState::default(), true).with_pending(Some(PendingIntent {
            active: true,
            narrate: true,
        }));
    assert_eq!(machine.policy_text("policy"), "policy");
    assert_eq!(
        PlanMachine::new(PlanUnitState::default(), false).policy_text("policy"),
        ""
    );
}

#[test]
fn plan_mode_does_not_restrict_the_tool_catalog() {
    assert_eq!(tool_restriction(), None);
    assert!(admits_all(&[], "write"));
    assert!(admits_all(&[], "bash"));
}

#[test]
fn from_events_derives_the_open_turn_and_leaves_pending_process_local() {
    let open = [SessionEvent::turn_start(1), plan_mode(true)];
    let machine = PlanMachine::from_events(&open).expect("the log folds");
    assert!(machine.active());
    assert!(machine.is_turn_open());
    assert_eq!(machine.pending(), None);

    let closed = [
        SessionEvent::turn_start(1),
        plan_mode(true),
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    ];
    let machine = PlanMachine::from_events(&closed).expect("the log folds");
    assert!(!machine.is_turn_open());
    assert_eq!(machine.pending(), None);
}

#[test]
fn observe_tracks_turn_boundaries_and_mode_switches() {
    let mut machine = PlanMachine::new(PlanUnitState::default(), false);
    machine.observe(&plan_mode(true)).expect("the switch folds");
    assert!(machine.active());
    machine
        .observe(&SessionEvent::turn_start(2))
        .expect("the turn folds");
    assert!(machine.is_turn_open());
    machine
        .observe(&header("test-model"))
        .expect("the header folds");
    assert_eq!(machine.active_at_last_header(), Some(true));
    machine
        .observe(&SessionEvent::turn_end(2, TurnEndReason::Completed))
        .expect("the turn folds");
    assert!(!machine.is_turn_open());

    let malformed = SessionEvent::extension_event(
        PLAN_MODE_TAG,
        Map::from_iter([("active".to_string(), json!("yes"))]),
    );
    assert!(machine.observe(&malformed).is_err());
    assert!(machine.active());
}

#[test]
fn the_status_omits_pending_and_the_projection_always_carries_it() {
    assert_eq!(
        serde_json::to_string(&PlanStatus {
            active: false,
            pending: None
        })
        .expect("the status serializes"),
        r#"{"active":false}"#
    );
    assert_eq!(
        serde_json::to_string(&PlanStatus {
            active: false,
            pending: Some(true)
        })
        .expect("the status serializes"),
        r#"{"active":false,"pending":true}"#
    );
    assert_eq!(
        serde_json::to_string(&PlanProjection {
            active: true,
            pending: false
        })
        .expect("the projection serializes"),
        r#"{"active":true,"pending":false}"#
    );
}

#[test]
fn a_narration_message_is_the_typescript_notice() {
    let narration = dsh_plan::Narration {
        active: true,
        text: NARRATION_PLAN_ON.to_string(),
    };
    let message = narration.message(MessageId::new("m1"));
    assert_eq!(text_of(&message), NARRATION_PLAN_ON);
    assert_eq!(
        serde_json::to_string(&message).expect("the message serializes"),
        concat!(
            r#"{"id":"m1","content":[{"type":"text","text":"The user switched this session to plan mode."}],"#,
            r#""source":{"kind":"plan-mode","form":"notice","summary":"The user switched this session to plan mode."}}"#
        )
    );
    match &message.source {
        MessageSource::Extension(source) => {
            assert_eq!(source.kind(), "plan-mode");
            let keys: Vec<&str> = source.fields().keys().map(String::as_str).collect();
            assert_eq!(keys, ["kind", "form", "summary"]);
        }
        other => panic!("expected the plan-mode source, got {other:?}"),
    }

    let off = dsh_plan::Narration {
        active: false,
        text: NARRATION_PLAN_OFF.to_string(),
    };
    assert_eq!(
        text_of(&off.message(MessageId::new("m2"))),
        "The user switched this session back to the default mode."
    );
}
