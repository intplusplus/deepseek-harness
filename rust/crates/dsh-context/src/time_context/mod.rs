//! Time-context: the durable request-clock reading.
//!
//! Ported from `packages/context/time-context`. The TypeScript plugin samples
//! `Date.now()` in a pre-step listener, folds injection state from session
//! projection events, formats with `Intl.DateTimeFormat`, and appends a durable
//! user message. Rust keeps every rule and injects both environment edges:
//!
//! - the clock is the `now_ms` argument every renderer takes;
//! - the zone database is a [`TimeZoneResolver`], so this crate never reads the
//!   process time zone;
//! - the plugin listener, the projection registry, and the message append
//!   belong to the host and stay out; [`fold_time_context`] and
//!   [`request_messages`] are the pure rules those host paths call.
//!
//! The model-facing text is byte-identical to the TypeScript plugin's for the
//! same instant, zone, and history: see `tests/time_context.rs`.

#![forbid(unsafe_code)]

use dsh_llm::UserMessage;
use dsh_session::{CoreEvent, SessionEvent};
use dsh_system_prompt::{AssembleContext, PromptContext, PromptError, Registration, SystemPrompt};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod browser;
pub mod timestamp;
pub mod zone;

pub use browser::{
    browser_time_zone, derive_browser_time_zone_context, derive_from_sources,
    render_browser_time_zone_context, BrowserTimeZoneContext,
};
pub use timestamp::format_timestamp;
pub use zone::{TimeZoneResolver, ZoneResolution, ZoneTable, ZoneTableError};

use crate::js;

/// The plugin's name, used as the folded projection key and the context name.
pub const TIME_CONTEXT_NAME: &str = "time-context";

/// Default minimum interval between durable injections (ten minutes).
pub const DEFAULT_REFRESH_INTERVAL_MS: u64 = 600_000;

/// Session-projection state version of the folded time-context readings.
pub const TIME_CONTEXT_STATE_VERSION: u32 = 2;

/// The first `f64` value that is not a JavaScript safe integer (`2^53`).
const MAX_SAFE_INTEGER_EXCLUSIVE: f64 = 9_007_199_254_740_992.0;

/// Plugin configuration.
///
/// `refresh_interval_ms` is an `f64` because the deployment supplies a JSON
/// number and the TypeScript validation rejects fractional, non-finite, and
/// out-of-range values by name; keeping the raw number makes that diagnostic
/// reproducible.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Config {
    /// Fallback display zone when the open turn has no unique browser zone.
    /// `None` uses the injected system zone.
    pub time_zone: Option<String>,
    /// Minimum milliseconds between durable injections in one session.
    /// `None` takes [`DEFAULT_REFRESH_INTERVAL_MS`]; `0` injects at every
    /// eligible step.
    pub refresh_interval_ms: Option<f64>,
}

/// Configuration with defaults applied and the fallback zone resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedConfig {
    time_zone: String,
    refresh_interval_ms: u64,
}

impl ResolvedConfig {
    /// The canonical fallback display zone.
    pub fn time_zone(&self) -> &str {
        &self.time_zone
    }

    /// The validated refresh interval in milliseconds.
    pub fn refresh_interval_ms(&self) -> u64 {
        self.refresh_interval_ms
    }

    /// Build one reading for an instant and its request position.
    ///
    /// The display zone is the request's unique browser zone when it has one,
    /// otherwise this configuration's fallback. The offset comes from `zones`
    /// at `now_ms`, so DST changes are reflected at the exact boundary.
    ///
    /// # Errors
    ///
    /// Returns [`TimeContextError::MissingOffset`] when `zones` no longer knows
    /// the selected zone at that instant; both zones were resolved when the
    /// configuration was built, so this signals a caller-supplied resolver
    /// that contradicts its own earlier answer.
    pub fn reading(
        &self,
        zones: &dyn TimeZoneResolver,
        now_ms: i64,
        turn: u64,
        step: u64,
        previous_ms: Option<i64>,
        browser: BrowserTimeZoneContext,
    ) -> Result<TimeReading, TimeContextError> {
        let time_zone = match &browser {
            BrowserTimeZoneContext::Resolved { time_zone } => time_zone.clone(),
            BrowserTimeZoneContext::Mixed { .. } | BrowserTimeZoneContext::Missing => {
                self.time_zone.clone()
            }
        };
        let offset_seconds = zones.offset_seconds(&time_zone, now_ms).ok_or_else(|| {
            TimeContextError::MissingOffset {
                zone: time_zone.clone(),
                epoch_ms: now_ms,
            }
        })?;
        Ok(TimeReading {
            now_ms,
            turn,
            step,
            previous_ms,
            time_zone,
            offset_seconds,
            browser,
        })
    }
}

/// Validate a deployment-authored refresh interval.
///
/// # Errors
///
/// Returns [`TimeContextError::InvalidRefreshInterval`] unless `value` is a
/// non-negative JavaScript safe integer.
pub fn validate_refresh_interval_ms(value: f64) -> Result<u64, TimeContextError> {
    if !value.is_finite()
        || value < 0.0
        || value.fract() != 0.0
        || value >= MAX_SAFE_INTEGER_EXCLUSIVE
    {
        return Err(TimeContextError::InvalidRefreshInterval {
            value: js::number_literal(value),
        });
    }
    // `value` is a non-negative integer below 2^53, so the cast is exact.
    Ok(value as u64)
}

/// Apply defaults and resolve the fallback zone.
///
/// `system_time_zone` is the injected process zone: the TypeScript plugin reads
/// it from the host when no explicit zone is configured. A `None` system zone
/// is the "failed to resolve the system time zone" case.
///
/// # Errors
///
/// Returns an invalid-interval diagnostic, an invalid explicit-zone
/// diagnostic, or the unresolved-system-zone diagnostic.
pub fn resolve_config(
    config: &Config,
    system_time_zone: Option<&str>,
    zones: &dyn TimeZoneResolver,
) -> Result<ResolvedConfig, TimeContextError> {
    let refresh_interval_ms = match config.refresh_interval_ms {
        Some(value) => validate_refresh_interval_ms(value)?,
        None => DEFAULT_REFRESH_INTERVAL_MS,
    };
    let time_zone = match &config.time_zone {
        Some(zone) => zones
            .canonical(zone)
            .ok_or_else(|| TimeContextError::InvalidTimeZone {
                zone: js::string_literal(zone),
            })?,
        None => {
            let zone = system_time_zone.ok_or(TimeContextError::UnresolvedSystemZone)?;
            zones
                .canonical(zone)
                .ok_or(TimeContextError::UnresolvedSystemZone)?
        }
    };
    Ok(ResolvedConfig {
        time_zone,
        refresh_interval_ms,
    })
}

/// Whether the eligibility interval admits a durable injection now.
///
/// Mirrors the TypeScript guard: a non-zero interval suppresses injection while
/// `now` is at or after the previous injection and less than one interval has
/// elapsed. Backward wall-clock movement injects rather than suppressing.
pub fn should_inject(
    refresh_interval_ms: u64,
    now_ms: i64,
    last_injection_ms: Option<i64>,
) -> bool {
    if refresh_interval_ms == 0 {
        return true;
    }
    let Some(last_injection_ms) = last_injection_ms else {
        return true;
    };
    if now_ms < last_injection_ms {
        return true;
    }
    match now_ms.checked_sub(last_injection_ms) {
        Some(elapsed) => (elapsed as u128) >= u128::from(refresh_interval_ms),
        None => true,
    }
}

/// One fully resolved request-clock observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeReading {
    /// Injected wall-clock instant, Unix epoch milliseconds.
    pub now_ms: i64,
    /// Open turn number.
    pub turn: u64,
    /// Step number within the turn.
    pub step: u64,
    /// Time of the preceding model-visible message (step 1) or of the
    /// preceding durable step context (later steps), when one exists.
    pub previous_ms: Option<i64>,
    /// Canonical display zone.
    pub time_zone: String,
    /// UTC offset of `time_zone` at `now_ms`, in seconds.
    pub offset_seconds: i32,
    /// Browser-zone facts for the request, rendered as the policy line.
    pub browser: BrowserTimeZoneContext,
}

impl TimeReading {
    /// Render the durable model-facing text.
    pub fn render(&self) -> String {
        render_time_context(self)
    }
}

/// Render the durable model-facing time-context text.
///
/// Byte-identical to the TypeScript `renderText`: three lines, a
/// `model-visible message` baseline on step one and a `step context` baseline
/// afterwards, and `unavailable` when no baseline exists.
pub fn render_time_context(reading: &TimeReading) -> String {
    let elapsed = match reading.previous_ms {
        Some(previous) => format_duration(reading.now_ms.saturating_sub(previous)),
        None => "unavailable".to_string(),
    };
    let baseline = if reading.step == 1 {
        "model-visible message"
    } else {
        "step context"
    };
    format!(
        "Time sampled while preparing turn {}, step {}: {}\n{}\nElapsed since the preceding {}: {}.",
        reading.turn,
        reading.step,
        format_timestamp(
            reading.now_ms,
            reading.offset_seconds,
            &reading.time_zone
        ),
        render_browser_time_zone_context(&reading.browser),
        baseline,
        elapsed,
    )
}

/// Format a non-negative elapsed millisecond count as compact whole-second units.
///
/// Negative elapsed time clamps to `0s`, matching `Math.max(0, elapsedMs)`.
pub fn format_duration(elapsed_ms: i64) -> String {
    let seconds = elapsed_ms.max(0) / 1_000;
    let days = seconds / 86_400;
    let seconds = seconds % 86_400;
    let hours = seconds / 3_600;
    let seconds = seconds % 3_600;
    let minutes = seconds / 60;
    let seconds = seconds % 60;
    let mut parts: Vec<String> = Vec::new();
    if days > 0 {
        parts.push(format!("{days}d"));
    }
    if hours > 0 {
        parts.push(format!("{hours}h"));
    }
    if minutes > 0 {
        parts.push(format!("{minutes}m"));
    }
    parts.push(format!("{seconds}s"));
    parts.join(" ")
}

/// Folded time-context readings, serialized as the `timeContext` projection.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeContextState {
    /// Time of the latest model-visible event (user/assistant message, tool
    /// result), or `None`.
    pub last_message_time: Option<i64>,
    /// Time of this plugin's latest durable injection, or `None`.
    pub last_injection_time: Option<i64>,
    /// Latest injection time in the open turn, or `None` before that turn
    /// receives one.
    pub last_turn_injection_time: Option<i64>,
}

/// One event's contribution to the time-context projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeContextObservation {
    /// A `turn/start` or `turn/end` clears the open-turn injection time.
    TurnBoundary,
    /// A user message; `injected` marks this plugin's own durable reading.
    UserMessage {
        /// Event time in Unix epoch milliseconds.
        time_ms: i64,
        /// Whether the source kind is [`TIME_CONTEXT_NAME`].
        injected: bool,
    },
    /// An assistant message or tool result.
    OtherMessage {
        /// Event time in Unix epoch milliseconds.
        time_ms: i64,
    },
    /// Any other event leaves the state unchanged.
    Ignored,
}

/// Classify one session event for the time-context projection.
///
/// Session events in the Rust log carry no timestamp, so the host supplies the
/// event's commit time; everything else is the event's own content.
pub fn observe_event(event: &SessionEvent, time_ms: i64) -> TimeContextObservation {
    match event {
        SessionEvent::Core(CoreEvent::TurnStart { .. } | CoreEvent::TurnEnd { .. }) => {
            TimeContextObservation::TurnBoundary
        }
        SessionEvent::Core(CoreEvent::UserMessage { message }) => {
            TimeContextObservation::UserMessage {
                time_ms,
                injected: message.source.kind() == TIME_CONTEXT_NAME,
            }
        }
        SessionEvent::Core(CoreEvent::AssistantMessage { .. } | CoreEvent::ToolResult { .. }) => {
            TimeContextObservation::OtherMessage { time_ms }
        }
        _ => TimeContextObservation::Ignored,
    }
}

/// Apply one observation to the folded state.
pub fn fold_time_context(
    state: &TimeContextState,
    observation: &TimeContextObservation,
) -> TimeContextState {
    match observation {
        TimeContextObservation::TurnBoundary => TimeContextState {
            last_turn_injection_time: None,
            ..state.clone()
        },
        TimeContextObservation::UserMessage { time_ms, injected } => {
            let mut next = TimeContextState {
                last_message_time: Some(*time_ms),
                ..state.clone()
            };
            if *injected {
                next.last_injection_time = Some(*time_ms);
                next.last_turn_injection_time = Some(*time_ms);
            }
            next
        }
        TimeContextObservation::OtherMessage { time_ms } => TimeContextState {
            last_message_time: Some(*time_ms),
            ..state.clone()
        },
        TimeContextObservation::Ignored => state.clone(),
    }
}

/// Collect entered and proposed user messages belonging to one open turn.
///
/// Walks the log backward to the matching `turn/start`; messages found above it
/// are the turn's entered messages, in log order, followed by `proposed`. When
/// the boundary is absent the result is `proposed` alone.
pub fn request_messages<'a>(
    events: &'a [SessionEvent],
    turn: u64,
    proposed: &[&'a UserMessage],
) -> Vec<&'a UserMessage> {
    let mut entered: Vec<&'a UserMessage> = Vec::new();
    for event in events.iter().rev() {
        match event {
            SessionEvent::Core(CoreEvent::TurnStart { turn: started }) if *started == turn => {
                entered.reverse();
                entered.extend(proposed.iter().copied());
                return entered;
            }
            SessionEvent::Core(CoreEvent::UserMessage { message }) => {
                entered.push(message.as_ref());
            }
            _ => {}
        }
    }
    proposed.to_vec()
}

/// Register the time-context contribution with the system-prompt registry.
///
/// `source` runs once per assembly; it supplies the reading built from the
/// injected clock and zone resolver (see [`ResolvedConfig::reading`]). `None`
/// contributes empty text, which the registry drops. `order` is explicit
/// because the central order table has no time-context slot: the TypeScript
/// plugin injects a user message rather than a system-prompt context, so the
/// placement of this adaptation belongs to the composition.
///
/// # Errors
///
/// Returns the registry's duplicate-name and non-finite-order diagnostics.
pub fn register_context<F>(
    prompt: &SystemPrompt,
    name: &str,
    order: f64,
    source: F,
) -> Result<Registration, PromptError>
where
    F: Fn(&AssembleContext) -> Option<TimeReading> + 'static,
{
    prompt.context(PromptContext::provider(name, order, move |context| {
        source(context)
            .map(|reading| reading.render())
            .unwrap_or_default()
    }))
}

/// Why a time-context configuration or reading failed.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TimeContextError {
    /// The refresh interval is not a non-negative safe integer.
    #[error("time-context: refreshIntervalMs must be a non-negative safe integer, got {value}")]
    InvalidRefreshInterval {
        /// The rejected value rendered as `String(value)`.
        value: String,
    },
    /// An explicit zone the resolver does not know.
    #[error("time-context: invalid IANA timeZone {zone}")]
    InvalidTimeZone {
        /// The rejected zone as a JSON string literal.
        zone: String,
    },
    /// No explicit zone was configured and the injected system zone is absent
    /// or unknown.
    #[error("time-context: failed to resolve the system time zone")]
    UnresolvedSystemZone,
    /// A browser zone is neither `UTC` nor IANA `Area/Location` shaped.
    #[error("browser time zone must be canonical UTC or IANA Area/Location: {value}")]
    BrowserZoneSyntax {
        /// The rejected value as a JSON string literal.
        value: String,
    },
    /// The resolver does not know a well-shaped browser zone.
    #[error("browser time zone is unsupported: {value}")]
    BrowserZoneUnsupported {
        /// The rejected value as a JSON string literal.
        value: String,
    },
    /// The resolver canonicalizes the browser zone to another label.
    #[error("browser time zone must be canonical: {value}")]
    BrowserZoneNotCanonical {
        /// The rejected value as a JSON string literal.
        value: String,
    },
    /// A resolved reading named a zone with no offset at its instant.
    #[error("time zone `{zone}` has no offset at {epoch_ms}")]
    MissingOffset {
        /// The zone with no offset.
        zone: String,
        /// The instant that was asked about.
        epoch_ms: i64,
    },
}
