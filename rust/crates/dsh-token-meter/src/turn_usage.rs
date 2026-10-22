//! Exact provider-reported token accounting for one completed turn.
//!
//! Transcribed from `packages/llm/token-meter/src/turn-usage.ts`. The fold is
//! deliberately strict: no attempt is inferred from a usage sample. Any missing
//! lifecycle boundary, incomplete attempt usage, unsafe count, or
//! contradictory exact total makes the whole disclosure unavailable — a partial
//! number would look like a real report.
//!
//! # The retry seam
//!
//! `llm/retry` and `llm/retry-started` are plugin events. The Rust core reads
//! their `turn`/`step` fields straight off the extension payload, so the fold
//! stays independent of the retry crate while the vocabulary stays identical.

use dsh_llm::{AssistantStreamRecord, MessageSource, StreamChunk, TokenUsage};
use dsh_session::{CoreEvent, SessionEvent, MAX_SAFE_INTEGER};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The `llm/retry-started` plugin-event tag.
pub const LLM_RETRY_STARTED_TAG: &str = "llm/retry-started";
/// The `llm/retry` plugin-event tag.
pub const LLM_RETRY_TAG: &str = "llm/retry";

/// One provider/model route that contributed a billed request attempt.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnTokenUsageRoute {
    /// Registered provider route.
    pub provider: String,
    /// Provider-owned model id.
    pub model: String,
}

/// Exact provider-reported token accounting for every attempt in one completed
/// turn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnTokenUsage {
    /// Sum of uncached prompt input across all attempts.
    pub uncached_input_tokens: u64,
    /// Sum of completion output across all attempts.
    pub output_tokens: u64,
    /// Exact aggregate prompt plus output total across all attempts.
    pub total_tokens: u64,
    /// Present only when every attempt reported the bucket.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    /// Present only when every attempt reported the bucket.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    /// Output subset, present only when every attempt reported it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    /// Present only when every billed attempt has provider/model attribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routes: Option<Vec<TurnTokenUsageRoute>>,
}

/// Why a turn's attempt lifecycle could not be priced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum TurnUsageError {
    /// A sum of reported counts crossed the safe-integer bound.
    #[error("token usage: a token sum is not a safe integer")]
    UnsafeSum,
}

struct NormalizedAttempt {
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    cache_read_tokens: Option<u64>,
    cache_write_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
    route: Option<TurnTokenUsageRoute>,
}

/// One turn's attempt-lifecycle position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttemptState {
    /// Between steps.
    Idle,
    /// A step opened; its attempt is being collected.
    Open { turn: u64, step: u64 },
    /// An attempt settled without committing a message.
    FinishClosed { turn: u64, step: u64 },
    /// An attempt settled by a message or a retry.
    Settled { turn: u64, step: u64, by: SettledBy },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SettledBy {
    Message,
    Retry,
}

fn is_count(value: u64) -> bool {
    value <= MAX_SAFE_INTEGER
}

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        let total = total.checked_add(value)?;
        is_count(total).then_some(total)
    })
}

/// The route of an assistant message, when it names a provider and a model.
fn message_route(source: &MessageSource) -> Option<TurnTokenUsageRoute> {
    let MessageSource::Model {
        provider, model, ..
    } = source
    else {
        return None;
    };
    if provider.is_empty() || model.is_empty() {
        return None;
    }
    Some(TurnTokenUsageRoute {
        provider: provider.clone(),
        model: model.clone(),
    })
}

/// The last usage sample embedded in one attempt's compact stream.
pub(crate) fn stream_usage(stream: &[AssistantStreamRecord]) -> Option<&TokenUsage> {
    stream.iter().rev().find_map(|record| match record {
        AssistantStreamRecord::Chunk { chunk, .. } => match chunk.as_ref() {
            StreamChunk::Usage { usage } => Some(usage),
            _ => None,
        },
        _ => None,
    })
}

/// The `turn`/`step` pair of one retry plugin event, or `None` for any other
/// event.
fn retry_turn_step(event: &SessionEvent, tag: &str) -> Option<(u64, u64)> {
    let SessionEvent::Extension(extension) = event else {
        return None;
    };
    if extension.kind() != tag {
        return None;
    }
    let payload = extension.payload();
    let turn = payload.get("turn")?.as_u64()?;
    let step = payload.get("step")?.as_u64()?;
    Some((turn, step))
}

/// Validate one usage sample into an exact per-attempt accounting.
///
/// `None` means the sample cannot prove an exact total: an unsafe count, a
/// reasoning bucket above output, or an exact total contradicting the buckets
/// it is made of.
fn normalize_usage(
    usage: &TokenUsage,
    route: Option<TurnTokenUsageRoute>,
) -> Option<NormalizedAttempt> {
    let input_tokens = usage.input_tokens;
    let output_tokens = usage.output_tokens;
    if !is_count(input_tokens) || !is_count(output_tokens) {
        return None;
    }
    let cache_read_tokens = usage.cache_read_tokens;
    if cache_read_tokens.is_some_and(|value| !is_count(value)) {
        return None;
    }
    let cache_write_tokens = usage.cache_write_tokens;
    if cache_write_tokens.is_some_and(|value| !is_count(value)) {
        return None;
    }
    let reasoning_tokens = usage.reasoning_tokens;
    if let Some(value) = reasoning_tokens {
        if !is_count(value) || value > output_tokens {
            return None;
        }
    }

    let known_prompt = checked_sum(
        std::iter::once(input_tokens)
            .chain(cache_read_tokens)
            .chain(cache_write_tokens),
    )?;

    let exact_total = match usage.total_tokens {
        Some(total_tokens) => {
            if !is_count(total_tokens) {
                return None;
            }
            let exact_prompt = total_tokens.checked_sub(output_tokens)?;
            if exact_prompt < known_prompt {
                return None;
            }
            if cache_read_tokens.is_some()
                && cache_write_tokens.is_some()
                && exact_prompt != known_prompt
            {
                return None;
            }
            total_tokens
        }
        None => {
            // A sample without an exact total must report both cache buckets,
            // so the disjoint sum proves the total.
            cache_read_tokens?;
            cache_write_tokens?;
            checked_sum([known_prompt, output_tokens])?
        }
    };

    Some(NormalizedAttempt {
        input_tokens,
        output_tokens,
        total_tokens: exact_total,
        cache_read_tokens,
        cache_write_tokens,
        reasoning_tokens,
        route,
    })
}

/// Sum an optional bucket across attempts, present only when every attempt
/// reported it.
fn optional_sum(
    values: impl IntoIterator<Item = Option<u64>>,
) -> Result<Option<u64>, TurnUsageError> {
    let mut total: u64 = 0;
    for value in values {
        let Some(value) = value else {
            return Ok(None);
        };
        total = total.checked_add(value).ok_or(TurnUsageError::UnsafeSum)?;
        if !is_count(total) {
            return Err(TurnUsageError::UnsafeSum);
        }
    }
    Ok(Some(total))
}

/// Aggregate every settled attempt into the turn's exact accounting.
fn aggregate_attempts(
    attempts: &[NormalizedAttempt],
) -> Result<Option<TurnTokenUsage>, TurnUsageError> {
    if attempts.is_empty() {
        return Ok(None);
    }
    let input_tokens =
        checked_sum(attempts.iter().map(|a| a.input_tokens)).ok_or(TurnUsageError::UnsafeSum)?;
    let output_tokens =
        checked_sum(attempts.iter().map(|a| a.output_tokens)).ok_or(TurnUsageError::UnsafeSum)?;
    let total_tokens =
        checked_sum(attempts.iter().map(|a| a.total_tokens)).ok_or(TurnUsageError::UnsafeSum)?;

    let cache_read_tokens = optional_sum(attempts.iter().map(|a| a.cache_read_tokens))?;
    let cache_write_tokens = optional_sum(attempts.iter().map(|a| a.cache_write_tokens))?;
    let reasoning_tokens = optional_sum(attempts.iter().map(|a| a.reasoning_tokens))?;

    // Routes publish only when every billed attempt carries attribution;
    // duplicates collapse to their first-seen order.
    let attributed: Option<Vec<_>> = attempts
        .iter()
        .map(|attempt| attempt.route.clone())
        .collect();
    let routes = attributed.map(|routes| {
        let mut unique: Vec<TurnTokenUsageRoute> = Vec::new();
        for route in routes {
            if !unique.iter().any(|seen| seen == &route) {
                unique.push(route);
            }
        }
        unique
    });

    Ok(Some(TurnTokenUsage {
        uncached_input_tokens: input_tokens,
        output_tokens,
        total_tokens,
        cache_read_tokens,
        cache_write_tokens,
        reasoning_tokens,
        routes,
    }))
}

/// Fold one complete turn's durable attempt lifecycle into exact token
/// accounting.
///
/// `events` are turn-local durable events from `turn/start` through
/// `turn/end`.
///
/// No attempt is inferred from a usage sample. Any missing lifecycle boundary,
/// incomplete attempt usage, unsafe count, or contradictory exact total makes
/// the whole disclosure unavailable.
pub fn derive_turn_token_usage(
    events: &[SessionEvent],
) -> Result<Option<TurnTokenUsage>, TurnUsageError> {
    let mut state = AttemptState::Idle;
    let mut attempts: Vec<NormalizedAttempt> = Vec::new();
    let mut sampled: Option<TokenUsage> = None;
    let mut turn: Option<u64> = None;
    let mut saw_end = false;
    let mut invalid = false;

    // Close the open attempt with its sampled usage, if one is armed. The
    // route attribution rides the message that settles the attempt.
    macro_rules! close_open {
        ($route:expr) => {{
            if let Some(sample) = sampled.take() {
                match normalize_usage(&sample, $route) {
                    Some(normalized) => {
                        attempts.push(normalized);
                        true
                    }
                    None => false,
                }
            } else {
                false
            }
        }};
    }

    for event in events {
        if invalid {
            break;
        }
        let SessionEvent::Core(core) = event else {
            if retry_turn_step(event, LLM_RETRY_STARTED_TAG).is_some()
                || retry_turn_step(event, LLM_RETRY_TAG).is_some()
            {
                // A retry event outside a core turn frame is handled below.
                continue;
            }
            continue;
        };
        match core {
            CoreEvent::TurnStart { turn: opened } => {
                if turn.is_some() || state != AttemptState::Idle {
                    invalid = true;
                } else {
                    turn = Some(*opened);
                }
                continue;
            }
            CoreEvent::TurnEnd { turn: closed, .. } => {
                if turn.is_none() {
                    invalid = true;
                    break;
                }
                if *closed != turn.unwrap() || state != AttemptState::Idle || saw_end {
                    invalid = true;
                } else {
                    saw_end = true;
                }
                continue;
            }
            _ => {}
        }
        let Some(current_turn) = turn else {
            invalid = true;
            break;
        };
        if saw_end {
            invalid = true;
            break;
        }

        match core {
            CoreEvent::StepStart { turn: opened, step } => {
                if *opened != current_turn || state != AttemptState::Idle {
                    invalid = true;
                } else {
                    state = AttemptState::Open {
                        turn: current_turn,
                        step: *step,
                    };
                    sampled = None;
                }
            }
            CoreEvent::AssistantAttempt {
                turn: attempt_turn,
                step,
                stream,
            } => {
                if *attempt_turn != current_turn
                    || state
                        != (AttemptState::Open {
                            turn: current_turn,
                            step: *step,
                        })
                {
                    invalid = true;
                    continue;
                }
                if let Some(usage) = stream_usage(stream) {
                    sampled = Some(usage.clone());
                }
                if !close_open!(None) {
                    invalid = true;
                } else {
                    state = AttemptState::FinishClosed {
                        turn: current_turn,
                        step: *step,
                    };
                }
            }
            CoreEvent::AssistantMessage {
                turn: message_turn,
                step,
                message,
                stream,
                usage,
                ..
            } => {
                if *message_turn != current_turn
                    || state
                        != (AttemptState::Open {
                            turn: current_turn,
                            step: *step,
                        })
                {
                    invalid = true;
                    continue;
                }
                if let Some(usage) = usage.as_deref().or_else(|| stream_usage(stream)) {
                    sampled = Some(usage.clone());
                }
                let route = message_route(&message.source);
                if !close_open!(route) {
                    invalid = true;
                } else {
                    state = AttemptState::Settled {
                        turn: current_turn,
                        step: *step,
                        by: SettledBy::Message,
                    };
                }
            }
            _ => {
                // `llm/retry-started` reopens the replacement slot after a
                // retry settlement; `llm/retry` settles the failed attempt;
                // `step/end` closes the step.
                if let Some((retry_turn, retry_step)) =
                    retry_turn_step(event, LLM_RETRY_STARTED_TAG)
                {
                    if retry_turn != current_turn
                        || !matches!(state, AttemptState::Settled { turn, step, by: SettledBy::Retry }
                            if turn == current_turn && step == retry_step)
                    {
                        invalid = true;
                    } else {
                        state = AttemptState::Open {
                            turn: current_turn,
                            step: retry_step,
                        };
                        sampled = None;
                    }
                    continue;
                }
                if let Some((retry_turn, retry_step)) = retry_turn_step(event, LLM_RETRY_TAG) {
                    if retry_turn != current_turn
                        || state == AttemptState::Idle
                        || !same_attempt(state, current_turn, retry_step)
                    {
                        invalid = true;
                        continue;
                    }
                    match state {
                        AttemptState::Settled { .. } => invalid = true,
                        AttemptState::Open { .. } => {
                            if !close_open!(None) {
                                invalid = true;
                            }
                        }
                        _ => {}
                    }
                    if !invalid {
                        state = AttemptState::Settled {
                            turn: current_turn,
                            step: retry_step,
                            by: SettledBy::Retry,
                        };
                    }
                    continue;
                }
                if let CoreEvent::StepEnd { turn: closed, step } = core {
                    if *closed != current_turn
                        || state == AttemptState::Idle
                        || !same_attempt(state, current_turn, *step)
                    {
                        invalid = true;
                        continue;
                    }
                    if let AttemptState::Open { .. } = state {
                        if !close_open!(None) {
                            invalid = true;
                        }
                    }
                    if !invalid {
                        state = AttemptState::Idle;
                        sampled = None;
                    }
                }
            }
        }
    }

    if invalid || !saw_end || state != AttemptState::Idle {
        return Ok(None);
    }
    aggregate_attempts(&attempts)
}

fn same_attempt(state: AttemptState, turn: u64, step: u64) -> bool {
    match state {
        AttemptState::Open { turn: t, step: s }
        | AttemptState::FinishClosed { turn: t, step: s }
        | AttemptState::Settled {
            turn: t, step: s, ..
        } => t == turn && s == step,
        AttemptState::Idle => false,
    }
}
