//! Basic-compaction configuration, capacity budgets, and trigger decisions.
//!
//! Ported from `packages/compaction/compaction-basic/tests/compaction-basic.spec.ts`
//! and `packages/compaction/compaction-basic/src/config.ts`.

use dsh_compaction::{
    compaction_attempt_limit, conversation_target, overflow_compaction_allowed,
    plan_overflow_recovery, plan_pressure_after_prune, plan_pressure_step, pressure_cleared,
    reserved_completion_tokens, resolve_compact_spec, resolve_config, resolve_target_policy,
    routed_target, should_recover_overflow, still_above_threshold, BasicCompactionConfig,
    CompactionError, CompactionPolicyConfig, CompactionTrigger, ModelCompactPolicyConfig,
    ModelTarget, OverflowRecovery, PressurePlan, ResolvedRetention, TargetPressureConfigError,
    CONTEXT_WINDOW_EXCEEDED_CODE, DEFAULT_COMPACTION_RETRIES, DEFAULT_HEADROOM_TOKENS,
    DEFAULT_MAX_OVERFLOW_RETRIES, DEFAULT_RETAIN_RATIO, DEFAULT_THRESHOLD_RATIO,
};
use dsh_llm::LlmCallConfig;
use dsh_session::EpochHeader;
use serde_json::json;

fn target() -> ModelTarget {
    ModelTarget::new("p", "m")
}

fn resolved_defaults() -> dsh_compaction::ResolvedConfig {
    resolve_config(&BasicCompactionConfig::default()).unwrap()
}

fn resolved_policy() -> dsh_compaction::ResolvedTargetPolicy {
    resolve_target_policy(&resolved_defaults(), &target())
}

fn spec(context_window: u64, reserved: u64) -> dsh_compaction::ResolvedCompactSpec {
    resolve_compact_spec(&resolved_policy(), context_window, reserved).unwrap()
}

fn header(provider: &str, model: &str, max_tokens: Option<u64>) -> EpochHeader {
    EpochHeader {
        config: LlmCallConfig {
            provider: provider.to_string(),
            model: model.to_string(),
            reasoning_effort: None,
            temperature: None,
            max_tokens,
            stop: None,
        },
        adapter_defaults: None,
        tools: None,
    }
}

#[test]
fn resolves_the_documented_defaults() {
    let config = resolved_defaults();
    assert_eq!(config.threshold_ratio, DEFAULT_THRESHOLD_RATIO);
    assert_eq!(config.headroom_tokens, DEFAULT_HEADROOM_TOKENS);
    assert_eq!(
        config.retention,
        ResolvedRetention::Ratio(DEFAULT_RETAIN_RATIO)
    );
    assert_eq!(config.max_tokens, DEFAULT_HEADROOM_TOKENS);
    assert_eq!(config.compaction_retries, DEFAULT_COMPACTION_RETRIES);
    assert_eq!(config.max_overflow_retries, DEFAULT_MAX_OVERFLOW_RETRIES);
    assert!(config.auto);
    assert_eq!(config.summarization_provider, "");
    assert_eq!(config.summarization_model, "");
    assert!(config.model_policies.is_empty());
}

#[test]
fn max_tokens_defaults_to_the_resolved_headroom() {
    let config = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            headroom_tokens: Some(1_000),
            ..CompactionPolicyConfig::default()
        },
        ..BasicCompactionConfig::default()
    })
    .unwrap();
    assert_eq!(config.headroom_tokens, 1_000);
    assert_eq!(config.max_tokens, 1_000);
}

#[test]
fn a_model_headroom_becomes_max_tokens_only_without_a_top_level_cap() {
    let model_policy = ModelCompactPolicyConfig {
        provider: "p".to_string(),
        model: "m".to_string(),
        policy: CompactionPolicyConfig {
            headroom_tokens: Some(200),
            ..CompactionPolicyConfig::default()
        },
    };
    let without_top_level = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            headroom_tokens: Some(500),
            ..CompactionPolicyConfig::default()
        },
        model_policies: Some(vec![model_policy.clone()]),
        ..BasicCompactionConfig::default()
    })
    .unwrap();
    assert_eq!(
        resolve_target_policy(&without_top_level, &target()).max_tokens,
        200
    );

    let with_top_level = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            max_tokens: Some(900),
            headroom_tokens: Some(500),
            ..CompactionPolicyConfig::default()
        },
        model_policies: Some(vec![model_policy]),
        ..BasicCompactionConfig::default()
    })
    .unwrap();
    assert_eq!(
        resolve_target_policy(&with_top_level, &target()).max_tokens,
        900
    );
}

#[test]
fn rejects_ratios_outside_the_open_unit_interval() {
    for ratio in [0.0, 1.1, f64::NAN] {
        let error = resolve_config(&BasicCompactionConfig {
            policy: CompactionPolicyConfig {
                threshold_ratio: Some(ratio),
                ..CompactionPolicyConfig::default()
            },
            ..BasicCompactionConfig::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("must be a number in (0, 1]"));
    }
}

#[test]
fn rejects_ratio_retention_not_below_the_threshold() {
    let error = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            threshold_ratio: Some(0.8),
            retain_ratio: Some(0.8),
            ..CompactionPolicyConfig::default()
        },
        ..BasicCompactionConfig::default()
    })
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("retainRatio (0.8) must be less than the resolved thresholdRatio (0.8)"));
}

#[test]
fn rejects_mutually_exclusive_retention_forms() {
    let error = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            retain_ratio: Some(0.1),
            retain_tokens: Some(100),
            ..CompactionPolicyConfig::default()
        },
        ..BasicCompactionConfig::default()
    })
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("retainRatio and retainTokens are mutually exclusive"));
}

#[test]
fn rejects_a_half_summarization_pair() {
    let error = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            summarization_provider: Some("p".to_string()),
            ..CompactionPolicyConfig::default()
        },
        ..BasicCompactionConfig::default()
    })
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("must be set together as an empty or non-empty pair"));

    let error = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            summarization_provider: Some(String::new()),
            summarization_model: Some("m".to_string()),
            ..CompactionPolicyConfig::default()
        },
        ..BasicCompactionConfig::default()
    })
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("must be set together as an empty or non-empty pair"));
}

#[test]
fn rejects_duplicate_model_policies() {
    let policy = ModelCompactPolicyConfig {
        provider: "p".to_string(),
        model: "m".to_string(),
        policy: CompactionPolicyConfig::default(),
    };
    let error = resolve_config(&BasicCompactionConfig {
        model_policies: Some(vec![policy.clone(), policy]),
        ..BasicCompactionConfig::default()
    })
    .unwrap_err();
    assert!(error.to_string().contains("duplicate model policy for p/m"));
}

#[test]
fn rejects_empty_provider_or_model_targets() {
    let empty_provider = resolve_config(&BasicCompactionConfig {
        model_policies: Some(vec![ModelCompactPolicyConfig {
            provider: String::new(),
            model: "m".to_string(),
            policy: CompactionPolicyConfig::default(),
        }]),
        ..BasicCompactionConfig::default()
    })
    .unwrap_err();
    assert!(empty_provider
        .to_string()
        .contains(".provider must be a non-empty string"));

    let empty_model = resolve_config(&BasicCompactionConfig {
        model_policies: Some(vec![ModelCompactPolicyConfig {
            provider: "p".to_string(),
            model: String::new(),
            policy: CompactionPolicyConfig::default(),
        }]),
        ..BasicCompactionConfig::default()
    })
    .unwrap_err();
    assert!(empty_model
        .to_string()
        .contains(".model must be a non-empty string"));
}

#[test]
fn rejects_a_zero_max_tokens() {
    let error = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            max_tokens: Some(0),
            ..CompactionPolicyConfig::default()
        },
        ..BasicCompactionConfig::default()
    })
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("maxTokens (0) must be a positive integer"));
}

#[test]
fn merges_an_exact_target_override_over_the_defaults() {
    let config = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            threshold_ratio: Some(0.7),
            headroom_tokens: Some(1_000),
            summarization_provider: Some("default-provider".to_string()),
            summarization_model: Some("default-model".to_string()),
            compaction_retries: Some(2),
            max_overflow_retries: Some(3),
            ..CompactionPolicyConfig::default()
        },
        model_policies: Some(vec![ModelCompactPolicyConfig {
            provider: "p".to_string(),
            model: "m".to_string(),
            policy: CompactionPolicyConfig {
                threshold_ratio: Some(0.5),
                headroom_tokens: Some(7),
                retain_tokens: Some(100),
                summarization_provider: Some("override-provider".to_string()),
                summarization_model: Some("override-model".to_string()),
                max_tokens: Some(42),
                compaction_retries: Some(4),
                max_overflow_retries: Some(0),
                ..CompactionPolicyConfig::default()
            },
        }]),
        ..BasicCompactionConfig::default()
    })
    .unwrap();

    let overridden = resolve_target_policy(&config, &target());
    assert_eq!(overridden.threshold_ratio, 0.5);
    assert_eq!(overridden.headroom_tokens, 7);
    assert_eq!(overridden.retention, ResolvedRetention::Tokens(100));
    assert_eq!(overridden.summarization_provider, "override-provider");
    assert_eq!(overridden.summarization_model, "override-model");
    assert_eq!(overridden.max_tokens, 42);
    assert_eq!(overridden.compaction_retries, 4);
    assert_eq!(overridden.max_overflow_retries, 0);

    let inherited = resolve_target_policy(&config, &ModelTarget::new("p", "other"));
    assert_eq!(inherited.threshold_ratio, 0.7);
    assert_eq!(inherited.headroom_tokens, 1_000);
    assert_eq!(
        inherited.retention,
        ResolvedRetention::Ratio(DEFAULT_RETAIN_RATIO)
    );
    assert_eq!(inherited.summarization_provider, "default-provider");
    assert_eq!(inherited.max_tokens, 1_000);
    assert_eq!(inherited.compaction_retries, 2);
    assert_eq!(inherited.max_overflow_retries, 3);
}

#[test]
fn resolves_the_capacity_budget_table() {
    for (window, reserved, threshold, retain) in [
        (1_048_576, 256_000, 727_040, 126_812),
        (1_000_000, 0, 800_000, 160_000),
        (1_000_000, 100_000, 800_000, 144_000),
        (1_000_000, 134_464, 800_000, 138_485),
        (1_000_000, 256_000, 678_464, 119_040),
    ] {
        let resolved = spec(window, reserved);
        assert_eq!(resolved.threshold_tokens, threshold);
        assert_eq!(resolved.retain_tokens, retain);
        assert_eq!(resolved.context_window, window);
    }
}

#[test]
fn caps_pressure_by_the_remaining_capacity() {
    let config = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            headroom_tokens: Some(100_000),
            ..CompactionPolicyConfig::default()
        },
        ..BasicCompactionConfig::default()
    })
    .unwrap();
    let policy = resolve_target_policy(&config, &target());
    let resolved = resolve_compact_spec(&policy, 1_000_000, 300_000).unwrap();
    assert_eq!(resolved.threshold_tokens, 600_000);
    assert_eq!(resolved.retain_tokens, 112_000);
}

#[test]
fn rejects_a_non_positive_context_window() {
    let error = resolve_compact_spec(&resolved_policy(), 0, 0).unwrap_err();
    assert!(matches!(error, TargetPressureConfigError { .. }));
    assert_eq!(error.target_key, "p/m");
    assert!(error
        .to_string()
        .contains("contextWindow (0) must be a positive integer"));
}

#[test]
fn rejects_a_window_with_no_message_budget() {
    let error = resolve_compact_spec(&resolved_policy(), 1_000, 1_000).unwrap_err();
    assert!(error.to_string().contains("leaving no message budget"));
}

#[test]
fn rejects_a_window_with_no_pressure_budget() {
    let error = resolve_compact_spec(&resolved_policy(), 100, 20).unwrap_err();
    assert!(error.to_string().contains("leaving no pressure budget"));
}

#[test]
fn rejects_retention_at_or_above_the_threshold() {
    let config = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            retain_tokens: Some(800_000),
            ..CompactionPolicyConfig::default()
        },
        ..BasicCompactionConfig::default()
    })
    .unwrap();
    let error =
        resolve_compact_spec(&resolve_target_policy(&config, &target()), 1_000_000, 0).unwrap_err();
    assert!(error
        .to_string()
        .contains("retainTokens (800000) must be less than threshold tokens 800000"));
}

#[test]
fn routed_target_requires_both_route_components() {
    assert_eq!(routed_target(None), None);
    assert_eq!(routed_target(Some(&header("", "m", None))), None);
    assert_eq!(routed_target(Some(&header("p", "", None))), None);
    assert_eq!(
        routed_target(Some(&header("p", "m", None))),
        Some(ModelTarget::new("p", "m"))
    );
}

#[test]
fn reserved_completion_tokens_prefers_the_header_then_the_default() {
    assert_eq!(
        reserved_completion_tokens(Some(&header("p", "m", Some(100))), Some(200)),
        100
    );
    assert_eq!(
        reserved_completion_tokens(Some(&header("p", "m", None)), Some(200)),
        200
    );
    assert_eq!(reserved_completion_tokens(None, Some(200)), 200);
    assert_eq!(reserved_completion_tokens(None, None), 0);
}

#[test]
fn conversation_target_prefers_the_routed_target_then_a_complete_fallback() {
    let routed = ModelTarget::new("p", "m");
    let fallback = ModelTarget::new("f", "m");
    assert_eq!(
        conversation_target(Some(routed.clone()), Some(fallback.clone())),
        Some(routed.clone())
    );
    assert_eq!(
        conversation_target(None, Some(fallback.clone())),
        Some(fallback)
    );
    assert_eq!(
        conversation_target(None, Some(ModelTarget::new("", "m"))),
        None
    );
    assert_eq!(conversation_target(None, None), None);
}

#[test]
fn pressure_plans_follow_the_threshold_and_pruner_availability() {
    let spec = spec(1_000_000, 0);
    assert_eq!(
        plan_pressure_step(799_999, &spec, false),
        PressurePlan::BelowThreshold
    );
    assert_eq!(
        plan_pressure_step(800_000, &spec, true),
        PressurePlan::PruneThenRecheck
    );
    assert_eq!(
        plan_pressure_step(800_000, &spec, false),
        PressurePlan::Compact
    );
    assert_eq!(
        plan_pressure_after_prune(799_999, &spec),
        PressurePlan::BelowThreshold
    );
    assert_eq!(
        plan_pressure_after_prune(800_000, &spec),
        PressurePlan::Compact
    );
    assert!(pressure_cleared(799_999, &spec));
    assert!(!pressure_cleared(800_000, &spec));
}

#[test]
fn compaction_attempt_limit_adds_the_first_attempt() {
    assert_eq!(compaction_attempt_limit(&spec(1_000_000, 0)), 2);

    let config = resolve_config(&BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            compaction_retries: Some(3),
            ..CompactionPolicyConfig::default()
        },
        ..BasicCompactionConfig::default()
    })
    .unwrap();
    let spec =
        resolve_compact_spec(&resolve_target_policy(&config, &target()), 1_000_000, 0).unwrap();
    assert_eq!(compaction_attempt_limit(&spec), 4);
}

#[test]
fn still_above_threshold_reports_the_attempt_count() {
    let error = still_above_threshold(900_000, &spec(1_000_000, 0));
    assert!(matches!(
        error,
        CompactionError::StillAboveThreshold {
            attempts: 2,
            total_tokens: 900_000,
            threshold_tokens: 800_000
        }
    ));
    assert!(error
        .to_string()
        .contains("after 2 compaction attempts (900000 estimated tokens >= threshold 800000)"));
}

#[test]
fn overflow_recovery_requires_the_canonical_failure_code() {
    assert!(should_recover_overflow(CONTEXT_WINDOW_EXCEEDED_CODE, false));
    assert!(!should_recover_overflow(CONTEXT_WINDOW_EXCEEDED_CODE, true));
    assert!(!should_recover_overflow("OTHER", false));
}

#[test]
fn overflow_compaction_respects_the_retry_budget_and_route() {
    assert!(overflow_compaction_allowed(0, 1, true));
    assert!(!overflow_compaction_allowed(1, 1, true));
    assert!(!overflow_compaction_allowed(0, 1, false));
}

#[test]
fn overflow_recovery_retries_only_after_durable_progress() {
    assert_eq!(
        plan_overflow_recovery(0, false, 1, 2),
        OverflowRecovery::Retry { retries: 1 }
    );
    assert_eq!(
        plan_overflow_recovery(1, false, 1, 2),
        OverflowRecovery::Retry { retries: 2 }
    );
    assert_eq!(
        plan_overflow_recovery(1, false, 2, 2),
        OverflowRecovery::Delegate
    );
    assert_eq!(
        plan_overflow_recovery(1, true, 1, 2),
        OverflowRecovery::Delegate
    );
}

#[test]
fn trigger_strings_match_the_typescript_vocabulary() {
    assert_eq!(CompactionTrigger::Pressure.as_str(), "pressure");
    assert_eq!(
        CompactionTrigger::ContextOverflow.as_str(),
        "context-overflow"
    );
}

#[test]
fn config_serializes_camel_case_and_flattens_policy_fields() {
    let value = serde_json::to_value(BasicCompactionConfig {
        policy: CompactionPolicyConfig {
            threshold_ratio: Some(0.5),
            max_tokens: Some(10),
            ..CompactionPolicyConfig::default()
        },
        model_policies: Some(vec![ModelCompactPolicyConfig {
            provider: "p".to_string(),
            model: "m".to_string(),
            policy: CompactionPolicyConfig {
                retain_tokens: Some(5),
                ..CompactionPolicyConfig::default()
            },
        }]),
        auto: Some(false),
    })
    .unwrap();
    assert_eq!(value["thresholdRatio"], json!(0.5));
    assert_eq!(value["maxTokens"], json!(10));
    assert_eq!(value["auto"], json!(false));
    assert_eq!(value["modelPolicies"][0]["provider"], json!("p"));
    assert_eq!(value["modelPolicies"][0]["model"], json!("m"));
    assert_eq!(value["modelPolicies"][0]["retainTokens"], json!(5));
    assert!(value.get("policy").is_none());
}
