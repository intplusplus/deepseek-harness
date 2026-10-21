//! Ported from `packages/llm/llm-retry/tests/retry.spec.ts`.
//!
//! The TypeScript suite drives a whole Cordis context with fake timers. The
//! decision half needs none of that: `plan_retry` is pure, so the delay
//! arithmetic, budget exhaustion, provider `Retry-After` handling, and
//! delegation are asserted directly. The lifecycle behaviours (cancellable
//! waits, disposal drain, downstream delegation in always mode) are not
//! ported; they need the async waterfall.

use std::collections::BTreeMap;

use dsh_llm::{LlmFailure, EMPTY_RESPONSE_CODE};
use dsh_retry::{
    apply_retry_state, decode_retry, encode_retry, encode_retry_started, fold_retry_state,
    local_delay, plan_retry, resolve_retry_policy, retry_state_key, AlwaysRetryPolicyConfig,
    BackoffConfig, LlmRetryEvent, LlmRetryStartedEvent, NormalRetryPolicyConfig,
    ResolvedRetryBackoff, ResolvedRetryPolicy, RetryId, RetryPolicyConfig, RetryState,
    RetryStateEntry,
};
use dsh_session::SessionEvent;

/// The TS `normalConfig()`: jitter off, so delays are exact.
fn normal_policy() -> ResolvedRetryPolicy {
    resolve_retry_policy(
        Some(&RetryPolicyConfig::Normal(NormalRetryPolicyConfig {
            max_retries: Some(2),
            retryable_codes: Some(vec!["SERVER".to_string(), "RATE_LIMIT".to_string()]),
            backoff: Some(BackoffConfig {
                initial_delay_ms: Some(500.0),
                max_delay_ms: Some(10_000.0),
                jitter_ratio: Some(0.0),
            }),
        })),
        "test",
    )
    .unwrap()
}

fn always_policy(backoff: BackoffConfig) -> ResolvedRetryPolicy {
    resolve_retry_policy(
        Some(&RetryPolicyConfig::Always(AlwaysRetryPolicyConfig {
            backoff: Some(backoff),
            max_retries: None,
            retryable_codes: None,
        })),
        "test",
    )
    .unwrap()
}

fn no_jitter(initial: f64, max: f64) -> BackoffConfig {
    BackoffConfig {
        initial_delay_ms: Some(initial),
        max_delay_ms: Some(max),
        jitter_ratio: Some(0.0),
    }
}

fn failure(code: &str) -> LlmFailure {
    LlmFailure {
        message: "busy".to_string(),
        code: code.to_string(),
        status: None,
        provider_retry_after_ms: None,
        request_id: None,
        offload_images: None,
    }
}

// ---------------------------------------------------------------- policy ---

#[test]
fn an_omitted_policy_resolves_to_the_documented_defaults() {
    let policy = resolve_retry_policy(None, "provider.retryPolicy").unwrap();
    assert_eq!(policy.mode(), dsh_retry::RetryMode::Normal);
    assert_eq!(policy.max_retries(), Some(5));
    assert_eq!(policy.backoff().initial_delay_ms, 500.0);
    assert_eq!(policy.backoff().max_delay_ms, 10_000.0);
    assert_eq!(policy.backoff().jitter_ratio, 0.1);
    assert!(policy.covers(EMPTY_RESPONSE_CODE));
}

#[test]
fn the_default_retryable_codes_cover_an_empty_completion() {
    // The TS suite proves this end to end through a finish-chunk error; here
    // it is the code the adapters assign to an empty completion.
    let policy = resolve_retry_policy(None, "provider.retryPolicy").unwrap();
    assert!(policy.covers(EMPTY_RESPONSE_CODE));
    assert!(policy.covers("RATE_LIMIT"));
    assert!(!policy.covers("AUTH"));
}

#[test]
fn an_overridden_code_set_replaces_the_defaults() {
    let policy = normal_policy();
    assert!(!policy.covers(EMPTY_RESPONSE_CODE));
    assert!(policy.covers("SERVER"));
}

#[test]
fn policy_keys_are_stable_under_code_reordering() {
    let a = resolve_retry_policy(
        Some(&RetryPolicyConfig::Normal(NormalRetryPolicyConfig {
            retryable_codes: Some(vec!["SERVER".to_string(), "RATE_LIMIT".to_string()]),
            ..normal_config_only()
        })),
        "test",
    )
    .unwrap();
    let b = resolve_retry_policy(
        Some(&RetryPolicyConfig::Normal(NormalRetryPolicyConfig {
            retryable_codes: Some(vec!["RATE_LIMIT".to_string(), "SERVER".to_string()]),
            ..normal_config_only()
        })),
        "test",
    )
    .unwrap();
    assert_eq!(
        dsh_retry::retry_policy_key(&a),
        dsh_retry::retry_policy_key(&b)
    );
}

fn normal_config_only() -> NormalRetryPolicyConfig {
    NormalRetryPolicyConfig {
        max_retries: None,
        retryable_codes: None,
        backoff: None,
    }
}

#[test]
fn the_policy_key_matches_the_typescript_bytes() {
    let policy = normal_policy();
    // `JSON.stringify(['normal', 2, ['RATE_LIMIT','SERVER'], 500, 10000, 0])`
    assert_eq!(
        dsh_retry::retry_policy_key(&policy),
        "[\"normal\",2,[\"RATE_LIMIT\",\"SERVER\"],500,10000,0]"
    );
}

#[test]
fn an_always_policy_key_carries_no_budget() {
    let policy = always_policy(no_jitter(1.0, 1.0));
    assert_eq!(dsh_retry::retry_policy_key(&policy), "[\"always\",1,1,0]");
}

#[test]
fn a_policy_change_starts_a_different_key() {
    let before = always_policy(no_jitter(1.0, 1.0));
    let after = always_policy(no_jitter(3.0, 3.0));
    assert_ne!(
        dsh_retry::retry_policy_key(&before),
        dsh_retry::retry_policy_key(&after)
    );
}

#[test]
fn an_out_of_range_backoff_is_rejected() {
    let config = RetryPolicyConfig::Normal(NormalRetryPolicyConfig {
        backoff: Some(BackoffConfig {
            initial_delay_ms: Some(0.0),
            max_delay_ms: None,
            jitter_ratio: None,
        }),
        ..normal_config_only()
    });
    let error = resolve_retry_policy(Some(&config), "provider.retryPolicy").unwrap_err();
    assert!(error
        .to_string()
        .contains("initialDelayMs must be a positive finite number"));
}

#[test]
fn an_initial_delay_above_the_cap_is_rejected() {
    let config = RetryPolicyConfig::Normal(NormalRetryPolicyConfig {
        backoff: Some(BackoffConfig {
            initial_delay_ms: Some(2_000.0),
            max_delay_ms: Some(1_000.0),
            jitter_ratio: None,
        }),
        ..normal_config_only()
    });
    let error = resolve_retry_policy(Some(&config), "provider.retryPolicy").unwrap_err();
    assert!(error
        .to_string()
        .contains("initialDelayMs must be less than or equal to maxDelayMs"));
}

#[test]
fn duplicate_retryable_codes_are_rejected() {
    let config = RetryPolicyConfig::Normal(NormalRetryPolicyConfig {
        retryable_codes: Some(vec!["SERVER".to_string(), "SERVER".to_string()]),
        ..normal_config_only()
    });
    let error = resolve_retry_policy(Some(&config), "provider.retryPolicy").unwrap_err();
    assert!(error.to_string().contains("must not contain duplicates"));
}

#[test]
fn empty_retryable_codes_are_rejected() {
    let config = RetryPolicyConfig::Normal(NormalRetryPolicyConfig {
        retryable_codes: Some(vec![]),
        ..normal_config_only()
    });
    let error = resolve_retry_policy(Some(&config), "provider.retryPolicy").unwrap_err();
    assert!(error.to_string().contains("must not be empty"));
}

#[test]
fn an_unknown_key_is_rejected_at_deserialization() {
    let json = r#"{"mode":"normal","retryPolciy":{}}"#;
    let error = serde_json::from_str::<RetryPolicyConfig>(json).unwrap_err();
    assert!(error.to_string().contains("retryPolciy"));
}

#[test]
fn always_mode_tolerates_retained_normal_fields() {
    // Layered config can keep normal-only fields after a mode switch.
    let config: RetryPolicyConfig =
        serde_json::from_str(r#"{"mode":"always","maxRetries":3}"#).unwrap();
    let policy = resolve_retry_policy(Some(&config), "test").unwrap();
    assert_eq!(policy.mode(), dsh_retry::RetryMode::Always);
    assert_eq!(policy.max_retries(), None);
}

// --------------------------------------------------------------- backoff ---

fn backoff(initial: f64, max: f64, jitter: f64) -> ResolvedRetryBackoff {
    ResolvedRetryBackoff {
        initial_delay_ms: initial,
        max_delay_ms: max,
        jitter_ratio: jitter,
    }
}

#[test]
fn the_first_delay_is_the_initial_delay_when_jitter_is_off() {
    assert_eq!(
        local_delay(&backoff(500.0, 10_000.0, 0.0), 1, || 0.5),
        500.0
    );
}

#[test]
fn delays_double_and_stay_capped() {
    let config = backoff(1.0, 4.0, 0.0);
    assert_eq!(local_delay(&config, 1, || 0.5), 1.0);
    assert_eq!(local_delay(&config, 2, || 0.5), 2.0);
    assert_eq!(local_delay(&config, 3, || 0.5), 4.0);
    assert_eq!(local_delay(&config, 4, || 0.5), 4.0);
}

#[test]
fn jitter_is_symmetric_around_one() {
    // jitterRatio 0.1: the multiplier spans [0.9, 1.1].
    let config = backoff(500.0, 10_000.0, 0.1);
    assert_eq!(local_delay(&config, 1, || 0.0), 450.0);
    assert_eq!(local_delay(&config, 1, || 1.0), 550.0);
    assert_eq!(local_delay(&config, 1, || 0.5), 500.0);
}

#[test]
fn the_zero_jitter_lower_bound_is_accepted() {
    // initialDelayMs 1, maxDelayMs 1, jitterRatio 1, sample 0 -> 0.
    assert_eq!(local_delay(&backoff(1.0, 1.0, 1.0), 1, || 0.0), 0.0);
}

#[test]
fn a_huge_retry_index_does_not_overflow_the_exponent() {
    // `Math.min(retry - 1, 1024)` clamps before `2 ** exponent` overflows.
    assert_eq!(local_delay(&backoff(1.0, 4.0, 0.0), 5_000, || 0.5), 4.0);
}

#[test]
fn always_mode_backoff_grows_without_a_budget() {
    let config = backoff(1.0, 4.0, 0.1);
    let delays: Vec<f64> = (1..=4)
        .map(|retry| local_delay(&config, retry, || 1.0))
        .collect::<Vec<_>>();
    assert_eq!(delays, vec![1.1, 2.2, 4.0, 4.0]);
}

// ---------------------------------------------------------------- events ---

#[test]
fn a_normal_retry_event_round_trips_through_the_log() {
    let event = LlmRetryEvent::Normal {
        retry_id: RetryId::new("retry-1"),
        turn: 1,
        step: 1,
        provider: "mock".to_string(),
        policy_key: "[\"normal\",2,[\"RATE_LIMIT\",\"SERVER\"],500,10000,0]".to_string(),
        retry: 1,
        max_retries: 2,
        delay_ms: 500.0,
        failure: Box::new(failure("RATE_LIMIT")),
    };
    let logged = encode_retry(&event).unwrap();
    let decoded = decode_retry(&logged).unwrap().unwrap();
    assert_eq!(decoded, event);
}

#[test]
fn an_always_retry_event_carries_no_budget() {
    let event = LlmRetryEvent::Always {
        retry_id: RetryId::new("retry-1"),
        turn: 3,
        step: 2,
        provider: "other".to_string(),
        policy_key: "[\"always\",1,1,0]".to_string(),
        retry: 4,
        delay_ms: 4.0,
        failure: Box::new(failure("AUTH")),
    };
    let logged = encode_retry(&event).unwrap();
    assert!(serde_json::to_string(&logged)
        .unwrap()
        .contains("\"llm/retry\""));
    assert!(!serde_json::to_string(&logged)
        .unwrap()
        .contains("maxRetries"));
    assert_eq!(decode_retry(&logged).unwrap().unwrap().max_retries(), None);
}

/// The wire bytes, not just the decoded value.
///
/// `serde_json`'s `preserve_order` makes field insertion order observable, so
/// the TypeScript declaration order in
/// `packages/llm/llm-retry/src/types.ts` is now a testable property:
/// `retryId, turn, step, provider, mode, policyKey, retry, maxRetries, delayMs,
/// failure` — with `mode` fifth, and the `type` discriminator first.
///
/// An internally tagged enum would emit `mode` ahead of every field, and a
/// plain `insert` of the extension tag would append it last; both would pass
/// every round-trip test and still fail a byte-level comparison.
#[test]
fn a_normal_retry_event_serializes_in_the_typescript_field_order() {
    let event = LlmRetryEvent::Normal {
        retry_id: RetryId::new("retry-1"),
        turn: 1,
        step: 1,
        provider: "mock".to_string(),
        policy_key: "policy".to_string(),
        retry: 1,
        max_retries: 2,
        delay_ms: 500.0,
        failure: Box::new(failure("RATE_LIMIT")),
    };
    let json = serde_json::to_string(&encode_retry(&event).unwrap()).unwrap();
    assert_eq!(
        json,
        concat!(
            r#"{"type":"llm/retry","retryId":"retry-1","turn":1,"step":1,"#,
            r#""provider":"mock","mode":"normal","policyKey":"policy","retry":1,"#,
            r#""maxRetries":2,"delayMs":500,"failure":{"message":"busy","code":"RATE_LIMIT"}}"#,
        )
    );
}

#[test]
fn an_always_retry_event_serializes_in_the_typescript_field_order() {
    let event = LlmRetryEvent::Always {
        retry_id: RetryId::new("retry-1"),
        turn: 3,
        step: 2,
        provider: "other".to_string(),
        policy_key: "policy".to_string(),
        retry: 4,
        delay_ms: 4.0,
        failure: Box::new(failure("AUTH")),
    };
    let json = serde_json::to_string(&encode_retry(&event).unwrap()).unwrap();
    assert_eq!(
        json,
        concat!(
            r#"{"type":"llm/retry","retryId":"retry-1","turn":3,"step":2,"#,
            r#""provider":"other","mode":"always","policyKey":"policy","retry":4,"#,
            r#""delayMs":4,"failure":{"message":"busy","code":"AUTH"}}"#,
        )
    );
}

#[test]
fn a_retry_started_event_serializes_in_the_typescript_field_order() {
    let event = LlmRetryStartedEvent {
        retry_id: RetryId::new("retry-1"),
        turn: 1,
        step: 1,
        retry: 1,
    };
    let json = serde_json::to_string(&encode_retry_started(&event).unwrap()).unwrap();
    assert_eq!(
        json,
        r#"{"type":"llm/retry-started","retryId":"retry-1","turn":1,"step":1,"retry":1}"#
    );
}

#[test]
fn a_whole_delay_serializes_as_a_javascript_number() {
    let event = LlmRetryEvent::Normal {
        retry_id: RetryId::new("retry-1"),
        turn: 1,
        step: 1,
        provider: "mock".to_string(),
        policy_key: "policy".to_string(),
        retry: 1,
        max_retries: 2,
        delay_ms: 500.0,
        failure: Box::new(failure("SERVER")),
    };
    let json = serde_json::to_string(&encode_retry(&event).unwrap()).unwrap();
    assert!(json.contains("\"delayMs\":500"), "{json}");
    assert!(!json.contains("500.0"), "{json}");
}

#[test]
fn fractional_delays_keep_their_fraction() {
    let event = LlmRetryEvent::Always {
        retry_id: RetryId::new("retry-1"),
        turn: 1,
        step: 1,
        provider: "mock".to_string(),
        policy_key: "policy".to_string(),
        retry: 1,
        delay_ms: 1.1,
        failure: Box::new(failure("SERVER")),
    };
    let json = serde_json::to_string(&encode_retry(&event).unwrap()).unwrap();
    assert!(json.contains("\"delayMs\":1.1"), "{json}");
}

#[test]
fn decoding_a_different_tag_yields_nothing() {
    let other = encode_retry_started(&LlmRetryStartedEvent {
        retry_id: RetryId::new("retry-1"),
        turn: 1,
        step: 1,
        retry: 1,
    })
    .unwrap();
    assert!(decode_retry(&other).is_none());
}

#[test]
fn a_known_tag_with_a_broken_payload_is_an_error_not_a_skip() {
    let broken = SessionEvent::Extension(dsh_session::ExtensionEvent::new(
        "llm/retry",
        serde_json::from_value(serde_json::json!({"type": "llm/retry", "mode": "normal"})).unwrap(),
    ));
    assert!(decode_retry(&broken).unwrap().is_err());
}

#[test]
fn a_core_event_is_not_a_retry_event() {
    assert!(decode_retry(&SessionEvent::turn_start(1)).is_none());
}

// ----------------------------------------------------------------- state ---

fn retry_logged(provider: &str, policy_key: &str, retry: u64, retry_id: &str) -> SessionEvent {
    let event = LlmRetryEvent::Normal {
        retry_id: RetryId::new(retry_id),
        turn: 1,
        step: 1,
        provider: provider.to_string(),
        policy_key: policy_key.to_string(),
        retry,
        max_retries: 2,
        delay_ms: 500.0,
        failure: Box::new(failure("SERVER")),
    };
    encode_retry(&event).unwrap()
}

#[test]
fn the_state_key_matches_the_typescript_bytes() {
    assert_eq!(retry_state_key("mock", "policy"), "[\"mock\",\"policy\"]");
}

#[test]
fn replaying_a_recorded_retry_keeps_the_same_state() {
    let event = retry_logged("mock", "policy", 1, "retry-1");
    let state = apply_retry_state(RetryState::new(), &event).unwrap();
    assert_eq!(apply_retry_state(state.clone(), &event).unwrap(), state);
}

#[test]
fn a_new_step_resets_the_state() {
    let mut state =
        apply_retry_state(RetryState::new(), &retry_logged("mock", "p", 1, "r")).unwrap();
    assert_eq!(state.len(), 1);
    state = apply_retry_state(state, &SessionEvent::step_start(1, 2)).unwrap();
    assert!(state.is_empty());
}

#[test]
fn a_closed_turn_resets_the_state() {
    let mut state =
        apply_retry_state(RetryState::new(), &retry_logged("mock", "p", 1, "r")).unwrap();
    state = apply_retry_state(
        state,
        &SessionEvent::turn_end(1, dsh_session::TurnEndReason::Completed),
    )
    .unwrap();
    assert!(state.is_empty());
}

#[test]
fn a_step_that_already_ran_does_not_reset_the_state() {
    let state = apply_retry_state(RetryState::new(), &retry_logged("mock", "p", 1, "r")).unwrap();
    let kept = apply_retry_state(state.clone(), &SessionEvent::step_end(1, 1)).unwrap();
    assert_eq!(kept, state);
}

#[test]
fn budgets_are_scoped_to_provider_and_policy() {
    let mut state = RetryState::new();
    state = apply_retry_state(state, &retry_logged("mock", "p", 1, "a")).unwrap();
    state = apply_retry_state(state, &retry_logged("other", "p", 1, "b")).unwrap();
    state = apply_retry_state(state, &retry_logged("mock", "q", 1, "c")).unwrap();
    assert_eq!(state.len(), 3);
}

#[test]
fn folding_a_log_reproduces_the_state() {
    let events = vec![
        SessionEvent::turn_start(1),
        SessionEvent::step_start(1, 1),
        retry_logged("mock", "p", 1, "a"),
        retry_logged("mock", "p", 2, "a"),
        SessionEvent::step_end(1, 1),
    ];
    let state = fold_retry_state(&events).unwrap();
    assert_eq!(
        state.get(&retry_state_key("mock", "p")),
        Some(&RetryStateEntry {
            retry: 2,
            retry_id: RetryId::new("a")
        })
    );
}

#[test]
fn a_retry_chain_keeps_its_identity_across_retries() {
    let events = vec![
        retry_logged("mock", "p", 1, "a"),
        retry_logged("mock", "p", 2, "a"),
    ];
    let state = fold_retry_state(&events).unwrap();
    assert_eq!(state.len(), 1);
    assert_eq!(
        state[&retry_state_key("mock", "p")].retry_id,
        RetryId::new("a")
    );
}

// ---------------------------------------------------------------- decide ---

#[test]
fn a_retryable_failure_schedules_one_retry() {
    let plan = plan_retry(
        "mock",
        &failure("RATE_LIMIT"),
        &normal_policy(),
        &RetryState::new(),
        || 0.5,
    )
    .unwrap();
    assert_eq!(plan.retry, 1);
    assert_eq!(plan.delay_ms, 500.0);
    assert_eq!(plan.max_retries, Some(2));
    assert_eq!(plan.mode, dsh_retry::RetryMode::Normal);
}

#[test]
fn a_non_retryable_code_is_delegated() {
    assert!(plan_retry(
        "mock",
        &failure("AUTH"),
        &normal_policy(),
        &RetryState::new(),
        || 0.5
    )
    .is_none());
}

#[test]
fn always_mode_covers_a_code_the_normal_policy_rejects() {
    let policy = always_policy(no_jitter(1.0, 1.0));
    let plan = plan_retry(
        "mock",
        &failure("AUTH"),
        &policy,
        &RetryState::new(),
        || 0.5,
    )
    .unwrap();
    assert_eq!(plan.retry, 1);
    assert_eq!(plan.max_retries, None);
}

#[test]
fn a_spent_normal_budget_is_delegated() {
    let policy = normal_policy();
    let mut state = RetryState::new();
    state.insert(
        retry_state_key("mock", &dsh_retry::retry_policy_key(&policy)),
        RetryStateEntry {
            retry: 2,
            retry_id: RetryId::new("a"),
        },
    );
    assert!(plan_retry("mock", &failure("SERVER"), &policy, &state, || 0.5).is_none());
}

#[test]
fn always_mode_has_no_budget_to_spend() {
    let policy = always_policy(no_jitter(1.0, 4.0));
    let mut state = RetryState::new();
    state.insert(
        retry_state_key("mock", &dsh_retry::retry_policy_key(&policy)),
        RetryStateEntry {
            retry: 40,
            retry_id: RetryId::new("a"),
        },
    );
    let plan = plan_retry("mock", &failure("AUTH"), &policy, &state, || 0.5).unwrap();
    assert_eq!(plan.retry, 41);
}

#[test]
fn a_retry_chain_reuses_its_identity_until_the_policy_changes() {
    let policy = normal_policy();
    let key = retry_state_key("mock", &dsh_retry::retry_policy_key(&policy));
    let first = plan_retry(
        "mock",
        &failure("SERVER"),
        &policy,
        &RetryState::new(),
        || 0.5,
    )
    .unwrap();
    let mut state = RetryState::new();
    state.insert(
        key.clone(),
        RetryStateEntry {
            retry: first.retry,
            retry_id: first.retry_id.clone(),
        },
    );
    let second = plan_retry("mock", &failure("SERVER"), &policy, &state, || 0.5).unwrap();
    assert_eq!(second.retry_id, first.retry_id);
    assert_eq!(second.retry, 2);

    let replaced = always_policy(no_jitter(1.0, 1.0));
    let after_switch = plan_retry("mock", &failure("SERVER"), &replaced, &state, || 0.5).unwrap();
    assert_ne!(after_switch.retry_id, first.retry_id);
    assert_eq!(after_switch.retry, 1);
}

#[test]
fn a_provider_retry_after_within_the_cap_is_used_verbatim() {
    let failure = LlmFailure {
        provider_retry_after_ms: Some(2_000),
        ..failure("RATE_LIMIT")
    };
    let plan = plan_retry(
        "mock",
        &failure,
        &normal_policy(),
        &RetryState::new(),
        || 0.0,
    )
    .unwrap();
    assert_eq!(plan.delay_ms, 2_000.0);
}

#[test]
fn a_normal_policy_delegates_an_over_cap_retry_after() {
    let failure = LlmFailure {
        provider_retry_after_ms: Some(10_001),
        ..failure("RATE_LIMIT")
    };
    assert!(plan_retry(
        "mock",
        &failure,
        &normal_policy(),
        &RetryState::new(),
        || 0.0
    )
    .is_none());
}

#[test]
fn always_mode_falls_back_to_local_backoff_on_an_over_cap_retry_after() {
    let failure = LlmFailure {
        provider_retry_after_ms: Some(10),
        ..failure("AUTH")
    };
    let policy = always_policy(BackoffConfig {
        initial_delay_ms: Some(2.0),
        max_delay_ms: Some(4.0),
        jitter_ratio: Some(0.5),
    });
    // Local backoff, not the provider's 10ms: cap is 4ms.
    assert_eq!(
        plan_retry("mock", &failure, &policy, &RetryState::new(), || 1.0)
            .unwrap()
            .delay_ms,
        3.0
    );
}

#[test]
fn a_zero_retry_after_is_ignored_in_favour_of_local_backoff() {
    let failure = LlmFailure {
        provider_retry_after_ms: Some(0),
        ..failure("RATE_LIMIT")
    };
    let plan = plan_retry(
        "mock",
        &failure,
        &normal_policy(),
        &RetryState::new(),
        || 0.5,
    )
    .unwrap();
    assert_eq!(plan.delay_ms, 500.0);
}

#[test]
fn the_plan_produces_the_events_the_log_stores() {
    let policy = normal_policy();
    let failure = failure("RATE_LIMIT");
    let plan = plan_retry("mock", &failure, &policy, &RetryState::new(), || 0.5).unwrap();

    let scheduled = plan.retry_event(7, 3, failure.clone());
    assert_eq!(scheduled.retry(), 1);
    assert_eq!(scheduled.delay_ms(), 500.0);
    assert_eq!(scheduled.failure(), &failure);
    assert_eq!(scheduled.max_retries(), Some(2));

    let started = plan.started_event(7, 3);
    assert_eq!(
        started,
        LlmRetryStartedEvent {
            retry_id: plan.retry_id.clone(),
            turn: 7,
            step: 3,
            retry: 1,
        }
    );
}

/// The TS `keeps always mode unbounded while preserving cancellable jittered
/// backoff`: five attempts, four scheduled retries, delays `[1.1, 2.2, 4, 4]`.
#[test]
fn an_always_chain_schedules_four_retries_before_recovering() {
    let policy = always_policy(BackoffConfig {
        initial_delay_ms: Some(1.0),
        max_delay_ms: Some(4.0),
        jitter_ratio: Some(0.1),
    });
    let mut state = RetryState::new();
    let mut events: Vec<SessionEvent> = Vec::new();
    let mut delays: Vec<f64> = Vec::new();

    for _ in 0..4 {
        let plan = plan_retry("mock", &failure("AUTH"), &policy, &state, || 1.0).unwrap();
        delays.push(plan.delay_ms);
        events.push(encode_retry(&plan.retry_event(1, 1, failure("AUTH"))).unwrap());
        state = fold_retry_state(&events).unwrap();
    }

    assert_eq!(delays, vec![1.1, 2.2, 4.0, 4.0]);
    let logged: Vec<LlmRetryEvent> = events
        .iter()
        .map(|e| decode_retry(e).unwrap().unwrap())
        .collect();
    assert_eq!(
        logged
            .iter()
            .map(|e| (e.mode(), e.retry(), e.max_retries()))
            .collect::<Vec<_>>(),
        vec![
            (dsh_retry::RetryMode::Always, 1, None),
            (dsh_retry::RetryMode::Always, 2, None),
            (dsh_retry::RetryMode::Always, 3, None),
            (dsh_retry::RetryMode::Always, 4, None),
        ]
    );
    // One chain: every scheduled retry shares its identity.
    let ids: Vec<&RetryId> = logged.iter().map(LlmRetryEvent::retry_id).collect();
    assert!(ids.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn the_scheduled_event_feeds_the_state_the_next_decision_reads() {
    let policy = normal_policy();
    let mut state = RetryState::new();
    let mut events: Vec<SessionEvent> = Vec::new();

    // Three failures in a row: the budget is 2, so the third delegates.
    for attempt in 1..=3 {
        let failure = failure("SERVER");
        match plan_retry("mock", &failure, &policy, &state, || 0.5) {
            Some(plan) => {
                let event = plan.retry_event(1, 1, failure);
                events.push(encode_retry(&event).unwrap());
                state = fold_retry_state(&events).unwrap();
                assert_eq!(plan.retry, attempt as u64);
            }
            None => assert_eq!(attempt, 3),
        }
    }
    assert_eq!(events.len(), 2);
}

#[test]
fn an_empty_state_map_behaves_like_a_fresh_step() {
    let empty: BTreeMap<String, RetryStateEntry> = BTreeMap::new();
    let policy = normal_policy();
    let plan = plan_retry("mock", &failure("SERVER"), &policy, &empty, || 0.5).unwrap();
    assert_eq!(plan.retry, 1);
}
