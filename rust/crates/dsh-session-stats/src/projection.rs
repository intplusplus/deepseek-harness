//! The `sessionStats` projection unit.
//!
//! The TypeScript fold counts turn/step boundaries and pairs tool calls with
//! their results. This Rust port keeps those rules, adds provider/model token
//! accounting, and omits the wall-time figures that require an event timestamp:
//! the Rust session log stores no `time` field on core events.
//!
//! `step/end` — not `assistant/message` — is the counted step event because it
//! is the step lifecycle authority: completed, failed, cancelled, and
//! max-tokens steps all land exactly one, while a cancelled step assembles no
//! message and an empty usage-host message must not add a step.
//!
//! Token figures come from committed `assistant/message` events only. A
//! message whose `usage` is absent increments `missingUsageMessages`; its
//! provider/model route is still counted. Attempts that never committed a
//! message carry no route identity and are not attributed.
//!
//! State is plain JSON: counters, a `byProvider` tree, the last counted turn,
//! and the set of unsettled tool-call ids. All values stay within the
//! JavaScript safe-integer bound, saturated rather than overflowed.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use dsh_llm::{AssistantMessage, MessageSource, TokenUsage, ToolCallId};
use dsh_session::{
    CoreEvent, SessionEvent, SessionHeader, SessionLogOffset, SessionSeq, MAX_SAFE_INTEGER,
};
use dsh_session_projection::{ProjectionDefinition, ProjectionState};
use serde::{Deserialize, Serialize};

/// The projection key owned by this crate.
pub const KEY: &str = "sessionStats";

/// The persisted-cache version.
///
/// The TypeScript unit is version 1 but persists wall-time fields and boundary
/// times this schema does not have. A distinct version keeps a cache row
/// written by either implementation from being forward-applied into the other.
pub const STATE_VERSION: u64 = 2;

/// Provider/model token and message totals.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RouteStats {
    /// Committed assistant messages attributed to this route.
    pub messages: u64,
    /// Messages that carried a usage record.
    pub usage_messages: u64,
    /// Messages whose usage was absent.
    pub missing_usage_messages: u64,
    /// Uncached prompt tokens.
    pub input_tokens: u64,
    /// Completion tokens.
    pub output_tokens: u64,
    /// Cached prompt tokens read.
    pub cache_read_tokens: u64,
    /// Cached prompt tokens written.
    pub cache_write_tokens: u64,
    /// Reasoning tokens reported alongside the completion.
    pub reasoning_tokens: u64,
    /// Disjoint input + cache + output total, saturated at the safe bound.
    pub total_tokens: u64,
}

impl RouteStats {
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("messages", self.messages),
            ("usageMessages", self.usage_messages),
            ("missingUsageMessages", self.missing_usage_messages),
            ("inputTokens", self.input_tokens),
            ("outputTokens", self.output_tokens),
            ("cacheReadTokens", self.cache_read_tokens),
            ("cacheWriteTokens", self.cache_write_tokens),
            ("reasoningTokens", self.reasoning_tokens),
            ("totalTokens", self.total_tokens),
        ] {
            if value > MAX_SAFE_INTEGER {
                return Err(format!("{name} exceeds the safe-integer bound"));
            }
        }
        if self
            .usage_messages
            .saturating_add(self.missing_usage_messages)
            != self.messages
        {
            return Err(
                "usageMessages + missingUsageMessages must equal messages for every route"
                    .to_string(),
            );
        }
        if self.total_tokens
            != combined_total(
                self.input_tokens,
                self.output_tokens,
                self.cache_read_tokens,
                self.cache_write_tokens,
            )
        {
            return Err("totalTokens must equal the disjoint token components".to_string());
        }
        Ok(())
    }
}

/// One committed client view of the whole-log figures.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SessionStatsView {
    /// Distinct turns carrying at least one closed step.
    pub turns: u64,
    /// Closed steps (`step/end`), including failed and cancelled steps.
    pub steps: u64,
    /// Tool calls requested by the model.
    pub tool_calls: u64,
    /// Tool results committed.
    pub tool_results: u64,
    /// Results carrying a failure marker.
    pub tool_errors: u64,
    /// Results with no preceding `tool/call` in the same turn.
    pub orphan_tool_results: u64,
    /// Committed assistant messages.
    pub assistant_messages: u64,
    /// Assistant messages carrying a usage record.
    pub usage_messages: u64,
    /// Assistant messages without a usage record.
    pub missing_usage_messages: u64,
    /// Uncached prompt tokens across every committed assistant message.
    pub input_tokens: u64,
    /// Completion tokens across every committed assistant message.
    pub output_tokens: u64,
    /// Cached prompt tokens read.
    pub cache_read_tokens: u64,
    /// Cached prompt tokens written.
    pub cache_write_tokens: u64,
    /// Reasoning tokens reported alongside completions.
    pub reasoning_tokens: u64,
    /// Disjoint input + cache + output total.
    pub total_tokens: u64,
    /// Token and message totals per provider and model, both keys sorted.
    pub by_provider: BTreeMap<String, BTreeMap<String, RouteStats>>,
}

impl SessionStatsView {
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("turns", self.turns),
            ("steps", self.steps),
            ("toolCalls", self.tool_calls),
            ("toolResults", self.tool_results),
            ("toolErrors", self.tool_errors),
            ("orphanToolResults", self.orphan_tool_results),
            ("assistantMessages", self.assistant_messages),
            ("usageMessages", self.usage_messages),
            ("missingUsageMessages", self.missing_usage_messages),
            ("inputTokens", self.input_tokens),
            ("outputTokens", self.output_tokens),
            ("cacheReadTokens", self.cache_read_tokens),
            ("cacheWriteTokens", self.cache_write_tokens),
            ("reasoningTokens", self.reasoning_tokens),
            ("totalTokens", self.total_tokens),
        ] {
            if value > MAX_SAFE_INTEGER {
                return Err(format!("{name} exceeds the safe-integer bound"));
            }
        }
        if self.total_tokens
            != combined_total(
                self.input_tokens,
                self.output_tokens,
                self.cache_read_tokens,
                self.cache_write_tokens,
            )
        {
            return Err("totalTokens must equal the disjoint token components".to_string());
        }
        for (provider, models) in &self.by_provider {
            for (model, route) in models {
                route
                    .validate()
                    .map_err(|reason| format!("route {provider}/{model}: {reason}"))?;
            }
        }
        Ok(())
    }
}

/// The persisted fold state: view totals plus in-flight boundaries.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SessionStatsState {
    /// Distinct turns carrying at least one closed step so far.
    pub turns: u64,
    /// Closed steps so far.
    pub steps: u64,
    /// Tool calls requested so far.
    pub tool_calls: u64,
    /// Tool results committed so far.
    pub tool_results: u64,
    /// Failed tool results committed so far.
    pub tool_errors: u64,
    /// Results with no matching preceding call.
    pub orphan_tool_results: u64,
    /// Committed assistant messages so far.
    pub assistant_messages: u64,
    /// Messages carrying usage.
    pub usage_messages: u64,
    /// Messages without usage.
    pub missing_usage_messages: u64,
    /// Uncached prompt tokens.
    pub input_tokens: u64,
    /// Completion tokens.
    pub output_tokens: u64,
    /// Cached prompt tokens read.
    pub cache_read_tokens: u64,
    /// Cached prompt tokens written.
    pub cache_write_tokens: u64,
    /// Reasoning tokens.
    pub reasoning_tokens: u64,
    /// Disjoint token total.
    pub total_tokens: u64,
    /// Per provider/model totals.
    pub by_provider: BTreeMap<String, BTreeMap<String, RouteStats>>,
    /// Turn of the last counted `step/end`; null before the first.
    pub last_turn: Option<u64>,
    /// Tool calls whose result has not landed in the current turn.
    pub pending_calls: BTreeSet<ToolCallId>,
}

impl ProjectionState for SessionStatsState {
    fn validate(&self) -> Result<(), String> {
        SessionStatsView {
            turns: self.turns,
            steps: self.steps,
            tool_calls: self.tool_calls,
            tool_results: self.tool_results,
            tool_errors: self.tool_errors,
            orphan_tool_results: self.orphan_tool_results,
            assistant_messages: self.assistant_messages,
            usage_messages: self.usage_messages,
            missing_usage_messages: self.missing_usage_messages,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cache_read_tokens: self.cache_read_tokens,
            cache_write_tokens: self.cache_write_tokens,
            reasoning_tokens: self.reasoning_tokens,
            total_tokens: self.total_tokens,
            by_provider: self.by_provider.clone(),
        }
        .validate()
    }
}

/// The `sessionStats` unit.
pub fn stats_projection() -> ProjectionDefinition<SessionStatsState> {
    ProjectionDefinition::host_only(
        KEY,
        STATE_VERSION,
        |_header: &SessionHeader, _inherited: SessionLogOffset| SessionStatsState::default(),
        apply,
    )
    .with_wire(
        |state| {
            Arc::new(SessionStatsView {
                turns: state.turns,
                steps: state.steps,
                tool_calls: state.tool_calls,
                tool_results: state.tool_results,
                tool_errors: state.tool_errors,
                orphan_tool_results: state.orphan_tool_results,
                assistant_messages: state.assistant_messages,
                usage_messages: state.usage_messages,
                missing_usage_messages: state.missing_usage_messages,
                input_tokens: state.input_tokens,
                output_tokens: state.output_tokens,
                cache_read_tokens: state.cache_read_tokens,
                cache_write_tokens: state.cache_write_tokens,
                reasoning_tokens: state.reasoning_tokens,
                total_tokens: state.total_tokens,
                by_provider: state.by_provider.clone(),
            })
        },
        |view| view.validate(),
    )
}

/// Apply one committed event; `None` keeps the previous state reference.
fn apply(
    state: &SessionStatsState,
    event: &SessionEvent,
    _seq: SessionSeq,
) -> Option<SessionStatsState> {
    let SessionEvent::Core(core) = event else {
        return None;
    };
    match core {
        CoreEvent::StepEnd { turn, .. } => {
            let mut next = state.clone();
            if next.last_turn != Some(*turn) {
                next.turns = add_count(next.turns, 1);
                next.last_turn = Some(*turn);
            }
            next.steps = add_count(next.steps, 1);
            Some(next)
        }
        CoreEvent::ToolCall { call_id, .. } => {
            let mut next = state.clone();
            next.tool_calls = add_count(next.tool_calls, 1);
            next.pending_calls.insert(call_id.clone());
            Some(next)
        }
        CoreEvent::ToolResult { message, error, .. } => {
            let mut next = state.clone();
            next.tool_results = add_count(next.tool_results, 1);
            if !next.pending_calls.remove(&message.tool_call_id) {
                next.orphan_tool_results = add_count(next.orphan_tool_results, 1);
            }
            if error.is_some() || message.is_error {
                next.tool_errors = add_count(next.tool_errors, 1);
            }
            Some(next)
        }
        CoreEvent::AssistantMessage { message, usage, .. } => {
            Some(count_assistant_message(state, message, usage.as_deref()))
        }
        CoreEvent::TurnEnd { .. } => {
            if state.pending_calls.is_empty() {
                return None;
            }
            let mut next = state.clone();
            next.pending_calls.clear();
            Some(next)
        }
        _ => None,
    }
}

fn count_assistant_message(
    state: &SessionStatsState,
    message: &AssistantMessage,
    usage: Option<&TokenUsage>,
) -> SessionStatsState {
    let mut next = state.clone();
    next.assistant_messages = add_count(next.assistant_messages, 1);
    let route = match &message.source {
        MessageSource::Model {
            provider, model, ..
        } => Some((provider.clone(), model.clone())),
        _ => None,
    };

    match usage {
        Some(usage) => {
            next.usage_messages = add_count(next.usage_messages, 1);
            add_usage(&mut next, usage);
            next.total_tokens = combined_total(
                next.input_tokens,
                next.output_tokens,
                next.cache_read_tokens,
                next.cache_write_tokens,
            );
            if let Some((provider, model)) = route {
                let route = next
                    .by_provider
                    .entry(provider)
                    .or_default()
                    .entry(model)
                    .or_default();
                route.messages = add_count(route.messages, 1);
                route.usage_messages = add_count(route.usage_messages, 1);
                add_route_usage(route, usage);
                route.total_tokens = combined_total(
                    route.input_tokens,
                    route.output_tokens,
                    route.cache_read_tokens,
                    route.cache_write_tokens,
                );
            }
        }
        None => {
            next.missing_usage_messages = add_count(next.missing_usage_messages, 1);
            if let Some((provider, model)) = route {
                let route = next
                    .by_provider
                    .entry(provider)
                    .or_default()
                    .entry(model)
                    .or_default();
                route.messages = add_count(route.messages, 1);
                route.missing_usage_messages = add_count(route.missing_usage_messages, 1);
            }
        }
    }
    next
}

fn add_usage(state: &mut SessionStatsState, usage: &TokenUsage) {
    state.input_tokens = add_count(state.input_tokens, usage.input_tokens);
    state.output_tokens = add_count(state.output_tokens, usage.output_tokens);
    state.cache_read_tokens = add_count(
        state.cache_read_tokens,
        usage.cache_read_tokens.unwrap_or(0),
    );
    state.cache_write_tokens = add_count(
        state.cache_write_tokens,
        usage.cache_write_tokens.unwrap_or(0),
    );
    state.reasoning_tokens = add_count(state.reasoning_tokens, usage.reasoning_tokens.unwrap_or(0));
}

fn add_route_usage(route: &mut RouteStats, usage: &TokenUsage) {
    route.input_tokens = add_count(route.input_tokens, usage.input_tokens);
    route.output_tokens = add_count(route.output_tokens, usage.output_tokens);
    route.cache_read_tokens = add_count(
        route.cache_read_tokens,
        usage.cache_read_tokens.unwrap_or(0),
    );
    route.cache_write_tokens = add_count(
        route.cache_write_tokens,
        usage.cache_write_tokens.unwrap_or(0),
    );
    route.reasoning_tokens = add_count(route.reasoning_tokens, usage.reasoning_tokens.unwrap_or(0));
}

/// Saturating increment capped at the JavaScript safe-integer bound.
fn add_count(current: u64, value: u64) -> u64 {
    current.saturating_add(value).min(MAX_SAFE_INTEGER)
}

/// Disjoint input + cache-read + cache-write + output total.
fn combined_total(input: u64, output: u64, cache_read: u64, cache_write: u64) -> u64 {
    input
        .saturating_add(output)
        .saturating_add(cache_read)
        .saturating_add(cache_write)
        .min(MAX_SAFE_INTEGER)
}
