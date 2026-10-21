//! Deterministic basic-compaction configuration and trigger rules.
//!
//! Capacity scaling, retention validation, exact-target overrides, pressure
//! qualification, bounded retries, and overflow recovery are pure decisions.
//! The asynchronous backend, LLM resolution, token metering, and durable append
//! stay outside this crate; this module is the part a replay or a unit test can
//! evaluate without a context.

#![forbid(unsafe_code)]

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use dsh_session::EpochHeader;

use super::error::{CompactionError, ConfigError, TargetPressureConfigError};

/// Provider-neutral code a provider reports when one request exceeds its
/// context window.
pub const CONTEXT_WINDOW_EXCEEDED_CODE: &str = "CONTEXT_WINDOW_EXCEEDED";

/// Default request-pressure fraction.
pub const DEFAULT_THRESHOLD_RATIO: f64 = 0.8;
/// Default verbatim-tail fraction.
pub const DEFAULT_RETAIN_RATIO: f64 = 0.16;
/// Default pressure headroom beyond the routed output reservation.
pub const DEFAULT_HEADROOM_TOKENS: u64 = 65_536;
/// Default extra attempts after the first compaction.
pub const DEFAULT_COMPACTION_RETRIES: u64 = 1;
/// Default retries after canonical context overflow.
pub const DEFAULT_MAX_OVERFLOW_RETRIES: u64 = 1;

/// Policy fields shared by the default policy and exact model overrides.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CompactionPolicyConfig {
    /// Window fraction for pressure, capped by the capacity remaining after the
    /// output reservation and headroom.
    pub threshold_ratio: Option<f64>,
    /// Additional pressure headroom beyond the routed output reservation.
    pub headroom_tokens: Option<u64>,
    /// Recent context retained as a fraction of the message budget.
    pub retain_ratio: Option<f64>,
    /// Absolute recent-context budget; mutually exclusive with `retain_ratio`.
    pub retain_tokens: Option<u64>,
    /// Summary provider; set together with `summarization_model`.
    pub summarization_provider: Option<String>,
    /// Summary model; set together with `summarization_provider`.
    pub summarization_model: Option<String>,
    /// Provider generation cap for summarization.
    pub max_tokens: Option<u64>,
    /// Extra attempts after the first compaction.
    pub compaction_retries: Option<u64>,
    /// Maximum retries after canonical context overflow.
    pub max_overflow_retries: Option<u64>,
}

/// Exact provider/model override merged over the default compaction policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCompactPolicyConfig {
    /// Registered provider route to match.
    pub provider: String,
    /// Exact routed model id to match within `provider`.
    pub model: String,
    /// Partial policy merged over the defaults.
    #[serde(flatten)]
    pub policy: CompactionPolicyConfig,
}

/// Basic compaction configuration with an optional exact-target policy table.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct BasicCompactionConfig {
    /// Shared defaults.
    #[serde(flatten)]
    pub policy: CompactionPolicyConfig,
    /// Exact provider/model overrides; duplicate targets fail resolution.
    pub model_policies: Option<Vec<ModelCompactPolicyConfig>>,
    /// Enable automatic pressure and overflow-recovery listeners.
    pub auto: Option<bool>,
}

/// Exactly one validated retention form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResolvedRetention {
    /// Retain a fraction of the message budget.
    Ratio(f64),
    /// Retain an absolute token budget.
    Tokens(u64),
}

impl ResolvedRetention {
    /// The ratio, when this form is ratio-based.
    pub fn retain_ratio(self) -> Option<f64> {
        match self {
            Self::Ratio(ratio) => Some(ratio),
            Self::Tokens(_) => None,
        }
    }

    /// The token budget, when this form is absolute.
    pub fn retain_tokens(self) -> Option<u64> {
        match self {
            Self::Ratio(_) => None,
            Self::Tokens(tokens) => Some(tokens),
        }
    }
}

/// Validated immutable defaults whose target-specific values remain unresolved.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedConfig {
    /// Resolved pressure fraction.
    pub threshold_ratio: f64,
    /// Resolved pressure headroom.
    pub headroom_tokens: u64,
    /// Resolved retention form.
    pub retention: ResolvedRetention,
    /// Summary provider, empty when inherited.
    pub summarization_provider: String,
    /// Summary model, empty when inherited.
    pub summarization_model: String,
    /// Summary generation cap.
    pub max_tokens: u64,
    /// Extra attempts after the first compaction.
    pub compaction_retries: u64,
    /// Maximum retries after canonical context overflow.
    pub max_overflow_retries: u64,
    /// Validated exact-target overrides.
    pub model_policies: Vec<ModelCompactPolicyConfig>,
    /// Whether automatic listeners are enabled.
    pub auto: bool,
}

/// One exact durable provider/model route.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelTarget {
    /// Registered provider route.
    pub provider: String,
    /// Model id within the route.
    pub model: String,
}

impl ModelTarget {
    /// Create one exact route.
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }

    /// Whether both route components are non-empty.
    pub fn is_complete(&self) -> bool {
        !self.provider.is_empty() && !self.model.is_empty()
    }

    /// The `provider/model` warning key.
    pub fn key(&self) -> String {
        format!("{}/{}", self.provider, self.model)
    }
}

/// Fully merged policy for one routed target, before capacity scaling.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedTargetPolicy {
    /// The exact route this policy applies to.
    pub target: ModelTarget,
    /// Resolved pressure fraction.
    pub threshold_ratio: f64,
    /// Resolved pressure headroom.
    pub headroom_tokens: u64,
    /// Resolved retention form.
    pub retention: ResolvedRetention,
    /// Summary provider, empty when inherited.
    pub summarization_provider: String,
    /// Summary model, empty when inherited.
    pub summarization_model: String,
    /// Summary generation cap.
    pub max_tokens: u64,
    /// Extra attempts after the first compaction.
    pub compaction_retries: u64,
    /// Maximum retries after canonical context overflow.
    pub max_overflow_retries: u64,
}

/// One routed model's concrete pressure and retention budget.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedCompactSpec {
    /// The exact route these budgets describe.
    pub target: ModelTarget,
    /// Adapter-declared full window.
    pub context_window: u64,
    /// Pressure fraction used to derive the threshold.
    pub threshold_ratio: f64,
    /// Request pressure that qualifies for compaction.
    pub threshold_tokens: u64,
    /// Recent tail retained verbatim.
    pub retain_tokens: u64,
    /// Summary provider, empty when inherited.
    pub summarization_provider: String,
    /// Summary model, empty when inherited.
    pub summarization_model: String,
    /// Summary generation cap.
    pub max_tokens: u64,
    /// Extra attempts after the first compaction.
    pub compaction_retries: u64,
    /// Maximum retries after canonical context overflow.
    pub max_overflow_retries: u64,
}

/// Why automatic policy is asking a backend to consider compaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompactionTrigger {
    /// Normal step-boundary pressure.
    Pressure,
    /// Provider-confirmed context overflow.
    ContextOverflow,
}

impl CompactionTrigger {
    /// The TypeScript trigger string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pressure => "pressure",
            Self::ContextOverflow => "context-overflow",
        }
    }
}

/// Resolve and validate service defaults plus exact-target partial overrides.
pub fn resolve_config(config: &BasicCompactionConfig) -> Result<ResolvedConfig, ConfigError> {
    validate_policy(&config.policy, "BasicCompactionConfig")?;

    let headroom_tokens = config
        .policy
        .headroom_tokens
        .unwrap_or(DEFAULT_HEADROOM_TOKENS);
    let max_tokens = config.policy.max_tokens.unwrap_or(headroom_tokens);
    assert_positive_integer(
        "BasicCompactionConfig.maxTokens (explicit or from headroomTokens)",
        max_tokens,
    )?;
    let threshold_ratio = config
        .policy
        .threshold_ratio
        .unwrap_or(DEFAULT_THRESHOLD_RATIO);
    let retention = resolve_retention(
        &config.policy,
        ResolvedRetention::Ratio(DEFAULT_RETAIN_RATIO),
    );
    validate_ratio_retention(threshold_ratio, retention, "BasicCompactionConfig")?;

    let mut model_policies = resolve_model_policies(config.model_policies.as_deref())?;
    for (index, policy) in model_policies.iter_mut().enumerate() {
        if policy.policy.max_tokens.is_none() && config.policy.max_tokens.is_none() {
            if let Some(headroom) = policy.policy.headroom_tokens {
                policy.policy.max_tokens = Some(headroom);
            }
        }
        assert_positive_integer(
            &format!(
                "BasicCompactionConfig: modelPolicies[{index}].maxTokens (explicit or from headroomTokens)"
            ),
            policy.policy.max_tokens.unwrap_or(max_tokens),
        )?;
        validate_ratio_retention(
            policy.policy.threshold_ratio.unwrap_or(threshold_ratio),
            resolve_retention(&policy.policy, retention),
            &format!("BasicCompactionConfig: modelPolicies[{index}]"),
        )?;
    }

    Ok(ResolvedConfig {
        threshold_ratio,
        headroom_tokens,
        retention,
        summarization_provider: config
            .policy
            .summarization_provider
            .clone()
            .unwrap_or_default(),
        summarization_model: config
            .policy
            .summarization_model
            .clone()
            .unwrap_or_default(),
        max_tokens,
        compaction_retries: config
            .policy
            .compaction_retries
            .unwrap_or(DEFAULT_COMPACTION_RETRIES),
        max_overflow_retries: config
            .policy
            .max_overflow_retries
            .unwrap_or(DEFAULT_MAX_OVERFLOW_RETRIES),
        model_policies,
        auto: config.auto.unwrap_or(true),
    })
}

/// Merge the exact provider/model override over the validated default policy.
pub fn resolve_target_policy(
    config: &ResolvedConfig,
    target: &ModelTarget,
) -> ResolvedTargetPolicy {
    let override_policy = config
        .model_policies
        .iter()
        .find(|policy| policy.provider == target.provider && policy.model == target.model);

    let threshold_ratio = override_policy
        .and_then(|policy| policy.policy.threshold_ratio)
        .unwrap_or(config.threshold_ratio);
    let headroom_tokens = override_policy
        .and_then(|policy| policy.policy.headroom_tokens)
        .unwrap_or(config.headroom_tokens);
    let retention = match override_policy {
        Some(policy) => resolve_retention(&policy.policy, config.retention),
        None => config.retention,
    };
    let summarization_provider = override_policy
        .and_then(|policy| policy.policy.summarization_provider.clone())
        .unwrap_or_else(|| config.summarization_provider.clone());
    let summarization_model = override_policy
        .and_then(|policy| policy.policy.summarization_model.clone())
        .unwrap_or_else(|| config.summarization_model.clone());
    let max_tokens = override_policy
        .and_then(|policy| policy.policy.max_tokens)
        .unwrap_or(config.max_tokens);
    let compaction_retries = override_policy
        .and_then(|policy| policy.policy.compaction_retries)
        .unwrap_or(config.compaction_retries);
    let max_overflow_retries = override_policy
        .and_then(|policy| policy.policy.max_overflow_retries)
        .unwrap_or(config.max_overflow_retries);

    ResolvedTargetPolicy {
        target: target.clone(),
        threshold_ratio,
        headroom_tokens,
        retention,
        summarization_provider,
        summarization_model,
        max_tokens,
        compaction_retries,
        max_overflow_retries,
    }
}

/// Scale one routed policy into concrete token budgets for its model capacity.
///
/// Pressure is capped by both the window fraction and the capacity remaining
/// after the routed output reservation plus compaction headroom. Retention
/// scales the message budget before headroom is deducted.
pub fn resolve_compact_spec(
    policy: &ResolvedTargetPolicy,
    context_window: u64,
    reserved_completion_tokens: u64,
) -> Result<ResolvedCompactSpec, TargetPressureConfigError> {
    let target_key = policy.target.key();
    if context_window == 0 {
        return Err(TargetPressureConfigError::new(
            target_key,
            format!("BasicCompactionConfig: contextWindow ({context_window}) must be a positive integer"),
        ));
    }

    let message_budget = i128::from(context_window) - i128::from(reserved_completion_tokens);
    if message_budget <= 0 {
        return Err(TargetPressureConfigError::new(
            target_key,
            format!(
                "compaction-basic: {}/{} reserves {} completion tokens of its {}-token context window, leaving no message budget; configure the adapter model's contextWindow above the effective request maxTokens",
                policy.target.provider,
                policy.target.model,
                reserved_completion_tokens,
                context_window,
            ),
        ));
    }
    let pressure_budget = message_budget - i128::from(policy.headroom_tokens);
    if pressure_budget <= 0 {
        return Err(TargetPressureConfigError::new(
            target_key,
            format!(
                "compaction-basic: {}/{} reserves {} completion tokens and {} headroom tokens of its {}-token context window, leaving no pressure budget; reduce the effective request maxTokens or compaction headroomTokens, or configure a larger adapter model contextWindow",
                policy.target.provider,
                policy.target.model,
                reserved_completion_tokens,
                policy.headroom_tokens,
                context_window,
            ),
        ));
    }

    let threshold_tokens = (context_window as f64 * policy.threshold_ratio)
        .floor()
        .min(pressure_budget as f64)
        .max(0.0) as u64;
    let retain_tokens = match policy.retention {
        ResolvedRetention::Tokens(tokens) => tokens,
        ResolvedRetention::Ratio(ratio) => (message_budget as f64 * ratio).floor().max(0.0) as u64,
    };
    if retain_tokens >= threshold_tokens {
        return Err(TargetPressureConfigError::new(
            target_key,
            format!(
                "BasicCompactionConfig: {}/{} retainTokens ({retain_tokens}) must be less than threshold tokens {threshold_tokens}",
                policy.target.provider, policy.target.model,
            ),
        ));
    }

    Ok(ResolvedCompactSpec {
        target: policy.target.clone(),
        context_window,
        threshold_ratio: policy.threshold_ratio,
        threshold_tokens,
        retain_tokens,
        summarization_provider: policy.summarization_provider.clone(),
        summarization_model: policy.summarization_model.clone(),
        max_tokens: policy.max_tokens,
        compaction_retries: policy.compaction_retries,
        max_overflow_retries: policy.max_overflow_retries,
    })
}

/// Resolve the exact provider/model durably routed for the latest request.
pub fn routed_target(header: Option<&EpochHeader>) -> Option<ModelTarget> {
    let config = &header?.config;
    let target = ModelTarget::new(&config.provider, &config.model);
    target.is_complete().then_some(target)
}

/// Output tokens the routed request reserves, which the provider charges to the
/// same window as the prompt.
pub fn reserved_completion_tokens(
    header: Option<&EpochHeader>,
    default_max_tokens: Option<u64>,
) -> u64 {
    header
        .and_then(|header| header.config.max_tokens)
        .or(default_max_tokens)
        .unwrap_or(0)
}

/// Resolve the conversation target used to select an optional policy override.
pub fn conversation_target(
    routed: Option<ModelTarget>,
    fallback: Option<ModelTarget>,
) -> Option<ModelTarget> {
    routed.or_else(|| fallback.filter(ModelTarget::is_complete))
}

/// The next deterministic pressure action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PressurePlan {
    /// The measured request is below the routed threshold.
    BelowThreshold,
    /// Pressure qualifies and an optional pruner should land before a summary
    /// range is chosen; remeasure afterward.
    PruneThenRecheck,
    /// Pressure qualifies; choose a range with the resolved retention budget.
    Compact,
}

/// Decide whether a measured request qualifies for pressure compaction.
pub fn plan_pressure_step(
    total_tokens: u64,
    spec: &ResolvedCompactSpec,
    pruner_available: bool,
) -> PressurePlan {
    if total_tokens < spec.threshold_tokens {
        PressurePlan::BelowThreshold
    } else if pruner_available {
        PressurePlan::PruneThenRecheck
    } else {
        PressurePlan::Compact
    }
}

/// Recheck pressure after the optional model-free prune pass.
pub fn plan_pressure_after_prune(total_tokens: u64, spec: &ResolvedCompactSpec) -> PressurePlan {
    if total_tokens < spec.threshold_tokens {
        PressurePlan::BelowThreshold
    } else {
        PressurePlan::Compact
    }
}

/// Whether a measured request is below the routed threshold.
pub fn pressure_cleared(total_tokens: u64, spec: &ResolvedCompactSpec) -> bool {
    total_tokens < spec.threshold_tokens
}

/// The bounded number of summary attempts for one pressured turn.
pub fn compaction_attempt_limit(spec: &ResolvedCompactSpec) -> u64 {
    spec.compaction_retries.saturating_add(1)
}

/// Build the bounded-retry failure after every attempt stayed above threshold.
pub fn still_above_threshold(total_tokens: u64, spec: &ResolvedCompactSpec) -> CompactionError {
    CompactionError::StillAboveThreshold {
        attempts: compaction_attempt_limit(spec),
        total_tokens,
        threshold_tokens: spec.threshold_tokens,
    }
}

/// The decision after one overflow recovery attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverflowRecovery {
    /// Retry the failed request from the replacement surface.
    Retry {
        /// Updated number of retries used for this agent.
        retries: u64,
    },
    /// Preserve the original request error.
    Delegate,
}

/// Whether one request failure is a canonical context-window overflow that may
/// enter recovery.
pub fn should_recover_overflow(failure_code: &str, aborted: bool) -> bool {
    !aborted && failure_code == CONTEXT_WINDOW_EXCEEDED_CODE
}

/// Whether the retry budget and routed target allow an overflow compaction.
pub fn overflow_compaction_allowed(
    retries_used: u64,
    max_retries: u64,
    routed_target: bool,
) -> bool {
    routed_target && retries_used < max_retries
}

/// Decide whether an overflow attempt may retry the failed request.
///
/// A durable surface replacement is sufficient retry proof even when the
/// optional later summary phase failed; cancellation always wins.
pub fn plan_overflow_recovery(
    retries_used: u64,
    aborted: bool,
    replace_generation_before: u64,
    replace_generation_after: u64,
) -> OverflowRecovery {
    if !aborted && replace_generation_after > replace_generation_before {
        OverflowRecovery::Retry {
            retries: retries_used.saturating_add(1),
        }
    } else {
        OverflowRecovery::Delegate
    }
}

fn resolve_retention(
    config: &CompactionPolicyConfig,
    fallback: ResolvedRetention,
) -> ResolvedRetention {
    if let Some(tokens) = config.retain_tokens {
        ResolvedRetention::Tokens(tokens)
    } else if let Some(ratio) = config.retain_ratio {
        ResolvedRetention::Ratio(ratio)
    } else {
        fallback
    }
}

fn validate_ratio_retention(
    threshold_ratio: f64,
    retention: ResolvedRetention,
    name: &str,
) -> Result<(), ConfigError> {
    if let ResolvedRetention::Ratio(retain_ratio) = retention {
        if retain_ratio >= threshold_ratio {
            return Err(ConfigError::new(format!(
                "{name}: retainRatio ({retain_ratio}) must be less than the resolved thresholdRatio ({threshold_ratio})"
            )));
        }
    }
    Ok(())
}

fn resolve_model_policies(
    configured: Option<&[ModelCompactPolicyConfig]>,
) -> Result<Vec<ModelCompactPolicyConfig>, ConfigError> {
    let Some(configured) = configured else {
        return Ok(Vec::new());
    };
    let mut seen = HashSet::with_capacity(configured.len());
    let mut resolved = Vec::with_capacity(configured.len());
    for (index, policy) in configured.iter().enumerate() {
        let name = format!("BasicCompactionConfig: modelPolicies[{index}]");
        if policy.provider.is_empty() {
            return Err(ConfigError::new(format!(
                "{name}.provider must be a non-empty string"
            )));
        }
        if policy.model.is_empty() {
            return Err(ConfigError::new(format!(
                "{name}.model must be a non-empty string"
            )));
        }
        validate_policy(&policy.policy, &name)?;
        let key = format!("{}\u{0}{}", policy.provider, policy.model);
        if !seen.insert(key) {
            return Err(ConfigError::new(format!(
                "BasicCompactionConfig: duplicate model policy for {}/{}",
                policy.provider, policy.model
            )));
        }
        resolved.push(policy.clone());
    }
    Ok(resolved)
}

fn validate_policy(policy: &CompactionPolicyConfig, name: &str) -> Result<(), ConfigError> {
    if let Some(ratio) = policy.threshold_ratio {
        assert_ratio(&format!("{name}.thresholdRatio"), ratio)?;
    }
    if let Some(ratio) = policy.retain_ratio {
        assert_ratio(&format!("{name}.retainRatio"), ratio)?;
    }
    if policy.retain_ratio.is_some() && policy.retain_tokens.is_some() {
        return Err(ConfigError::new(format!(
            "{name}: retainRatio and retainTokens are mutually exclusive"
        )));
    }
    if let Some(max_tokens) = policy.max_tokens {
        assert_positive_integer(&format!("{name}.maxTokens"), max_tokens)?;
    }
    validate_summarization_pair(policy, name)
}

fn validate_summarization_pair(
    policy: &CompactionPolicyConfig,
    name: &str,
) -> Result<(), ConfigError> {
    let provider = policy.summarization_provider.as_deref();
    let model = policy.summarization_model.as_deref();
    if provider.is_none() && model.is_none() {
        return Ok(());
    }
    let Some(provider) = provider else {
        return Err(ConfigError::new(format!(
            "{name}: summarizationProvider and summarizationModel must be set together as an empty or non-empty pair"
        )));
    };
    let Some(model) = model else {
        return Err(ConfigError::new(format!(
            "{name}: summarizationProvider and summarizationModel must be set together as an empty or non-empty pair"
        )));
    };
    if provider.is_empty() != model.is_empty() {
        return Err(ConfigError::new(format!(
            "{name}: summarizationProvider and summarizationModel must be set together as an empty or non-empty pair"
        )));
    }
    Ok(())
}

fn assert_ratio(name: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(ConfigError::new(format!(
            "{name} ({value}) must be a number in (0, 1]"
        )));
    }
    Ok(())
}

fn assert_positive_integer(name: &str, value: u64) -> Result<(), ConfigError> {
    if value == 0 {
        return Err(ConfigError::new(format!(
            "{name} ({value}) must be a positive integer"
        )));
    }
    Ok(())
}
