//! The `turnOutline` projection unit.
//!
//! `turn/start` anchors each entry because its seq is the load-through target
//! for a jump: the loop logs the boundary before the turn's prompt and steps,
//! so a window paged back through that seq contains the whole turn.
//!
//! The TypeScript unit serves four fields per turn (`turn`, `seq`, `prompt`,
//! `response`). This port keeps them first and appends the lifecycle facts the
//! Rust reconstruction needs: the closing seq, closed-step count, tool counts,
//! and a cancellation/error status. Unknown extension events are ignored, and
//! entries are strictly increasing by turn, so the array is also its own stable
//! id order.
//!
//! Previews mirror the rail's loaded-turn previews: text blocks are
//! space-joined, whitespace is collapsed, and a clipped preview ends with an
//! ellipsis. Prompt and response budgets are one and three rail-card lines
//! (50 and 120 characters). Rust counts Unicode scalar values, while
//! JavaScript's `String.length` counts UTF-16 code units; the difference is
//! observable only for astral characters, and it keeps every slice on a valid
//! UTF-8 boundary instead of panicking.
//!
//! The response buffers as a draft and commits at `turn/end`. The state holds
//! the turn list behind an `Arc`; the client view returns that same allocation,
//! so draft-only applies keep the view reference and the change feed stays
//! quiet between the boundary, prompt, and settled response.

#![forbid(unsafe_code)]

use std::sync::Arc;

use dsh_llm::ContentBlock;
use dsh_session::{
    CoreEvent, SessionEvent, SessionHeader, SessionLogOffset, SessionSeq, TurnEndReason,
};
use dsh_session_projection::{ProjectionDefinition, ProjectionState};
use serde::{Deserialize, Serialize};

/// The projection key owned by this crate.
pub const KEY: &str = "turnOutline";

/// The persisted-cache version.
///
/// The TypeScript unit is version 2. The added lifecycle fields change both the
/// state and client view, so this Rust schema uses version 3 to keep either
/// implementation's cache rows out of the other's fold.
pub const STATE_VERSION: u64 = 3;

/// Prompt budget: one rail-card line.
pub const PROMPT_PREVIEW_LIMIT: usize = 50;
/// Response budget: three rail-card lines.
pub const RESPONSE_PREVIEW_LIMIT: usize = 120;

/// How a turn ended, or that it is still open.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TurnOutlineStatus {
    /// The turn has started and not ended.
    Open,
    /// The model finished normally.
    Completed,
    /// Pre-step policy blocked the turn.
    Blocked,
    /// The turn was cancelled.
    Aborted,
    /// The turn failed.
    Error,
    /// At least one step reached its output-token ceiling.
    MaxTokens,
    /// A crash-orphaned turn was closed after the fact.
    Interrupted,
    /// Fork-seed construction closed a turn open at the fork boundary.
    Forked,
}

impl TurnOutlineStatus {
    fn from_reason(reason: &TurnEndReason) -> Self {
        match reason {
            TurnEndReason::Completed => Self::Completed,
            TurnEndReason::Aborted { .. } => Self::Aborted,
            TurnEndReason::Blocked => Self::Blocked,
            TurnEndReason::Error { .. } => Self::Error,
            TurnEndReason::MaxTokens => Self::MaxTokens,
            TurnEndReason::Interrupted => Self::Interrupted,
            TurnEndReason::Forked => Self::Forked,
        }
    }
}

/// One started turn's outline facts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TurnOutlineEntry {
    /// Host-assigned turn number; the stable id and sort key.
    pub turn: u64,
    /// The turn's `turn/start` seq — paging a window back through this seq
    /// loads the whole turn.
    pub seq: SessionSeq,
    /// Bounded first-human-prompt preview; empty until an eligible prompt lands.
    pub prompt: String,
    /// Bounded final-response preview; empty until a response settles.
    pub response: String,
    /// The closing `turn/end` seq; absent while the turn is open.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_seq: Option<SessionSeq>,
    /// Closed steps (`step/end`) in this turn.
    pub steps: u64,
    /// Tool calls requested in this turn.
    pub tool_calls: u64,
    /// Tool results committed in this turn.
    pub tool_results: u64,
    /// Failed tool results in this turn.
    pub tool_errors: u64,
    /// Lifecycle state; `open` exactly while `endSeq` is absent.
    pub status: TurnOutlineStatus,
}

/// Fold state: served entries plus the open turn's response draft.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TurnOutlineState {
    /// Started turns in ascending turn order, behind one allocation so the
    /// client-view identity can be reused across draft-only changes.
    pub turns: Arc<Vec<TurnOutlineEntry>>,
    /// Newest text-bearing assistant preview of the open turn.
    pub draft: String,
}

impl Default for TurnOutlineState {
    fn default() -> Self {
        Self {
            turns: Arc::new(Vec::new()),
            draft: String::new(),
        }
    }
}

impl ProjectionState for TurnOutlineState {
    fn validate(&self) -> Result<(), String> {
        validate_entries(&self.turns)?;
        if char_len(&self.draft) > RESPONSE_PREVIEW_LIMIT {
            return Err(format!(
                "draft exceeds the {RESPONSE_PREVIEW_LIMIT}-character response budget"
            ));
        }
        Ok(())
    }
}

/// The `turnOutline` unit.
pub fn outline_projection() -> ProjectionDefinition<TurnOutlineState> {
    ProjectionDefinition::host_only(
        KEY,
        STATE_VERSION,
        |_header: &SessionHeader, _inherited: SessionLogOffset| TurnOutlineState::default(),
        apply,
    )
    .with_wire(
        |state| Arc::clone(&state.turns),
        |turns: &Vec<TurnOutlineEntry>| validate_entries(turns),
    )
}

/// Apply one committed event; `None` keeps the previous state reference.
fn apply(
    state: &TurnOutlineState,
    event: &SessionEvent,
    seq: SessionSeq,
) -> Option<TurnOutlineState> {
    let SessionEvent::Core(core) = event else {
        return None;
    };
    match core {
        CoreEvent::TurnStart { turn } => {
            if state.turns.last().is_some_and(|last| *turn <= last.turn) {
                return None;
            }
            let mut turns = (*state.turns).clone();
            turns.push(TurnOutlineEntry {
                turn: *turn,
                seq,
                prompt: String::new(),
                response: String::new(),
                end_seq: None,
                steps: 0,
                tool_calls: 0,
                tool_results: 0,
                tool_errors: 0,
                status: TurnOutlineStatus::Open,
            });
            Some(TurnOutlineState {
                turns: Arc::new(turns),
                draft: String::new(),
            })
        }
        CoreEvent::UserMessage { message } => {
            if message.source.kind() != "user" {
                return None;
            }
            let last = state.turns.last()?;
            if last.end_seq.is_some() || !last.prompt.is_empty() {
                return None;
            }
            let prompt = preview(&message.content, PROMPT_PREVIEW_LIMIT);
            if prompt.is_empty() {
                return None;
            }
            let mut turns = (*state.turns).clone();
            let updated = turns.last_mut()?;
            updated.prompt = prompt;
            Some(TurnOutlineState {
                turns: Arc::new(turns),
                draft: state.draft.clone(),
            })
        }
        CoreEvent::AssistantMessage { turn, message, .. } => {
            let last = state.turns.last()?;
            if last.turn != *turn || last.end_seq.is_some() {
                return None;
            }
            let draft = preview(&message.content, RESPONSE_PREVIEW_LIMIT);
            if draft.is_empty() || draft == state.draft {
                return None;
            }
            Some(TurnOutlineState {
                turns: Arc::clone(&state.turns),
                draft,
            })
        }
        CoreEvent::StepEnd { turn, .. } => update_open_entry(state, *turn, |entry| {
            entry.steps = entry.steps.saturating_add(1);
        }),
        CoreEvent::ToolCall { turn, .. } => update_open_entry(state, *turn, |entry| {
            entry.tool_calls = entry.tool_calls.saturating_add(1);
        }),
        CoreEvent::ToolResult {
            turn,
            message,
            error,
            ..
        } => update_open_entry(state, *turn, |entry| {
            entry.tool_results = entry.tool_results.saturating_add(1);
            if error.is_some() || message.is_error {
                entry.tool_errors = entry.tool_errors.saturating_add(1);
            }
        }),
        CoreEvent::TurnEnd { turn, reason } => {
            let last = state.turns.last()?;
            if last.turn != *turn || last.end_seq.is_some() {
                return None;
            }
            let mut turns = (*state.turns).clone();
            let updated = turns.last_mut()?;
            if !state.draft.is_empty() {
                updated.response = state.draft.clone();
            }
            updated.end_seq = Some(seq);
            updated.status = TurnOutlineStatus::from_reason(reason);
            Some(TurnOutlineState {
                turns: Arc::new(turns),
                draft: String::new(),
            })
        }
        _ => None,
    }
}

fn update_open_entry(
    state: &TurnOutlineState,
    turn: u64,
    update: impl FnOnce(&mut TurnOutlineEntry),
) -> Option<TurnOutlineState> {
    let last = state.turns.last()?;
    if last.turn != turn || last.end_seq.is_some() {
        return None;
    }
    let mut turns = (*state.turns).clone();
    update(turns.last_mut()?);
    Some(TurnOutlineState {
        turns: Arc::new(turns),
        draft: state.draft.clone(),
    })
}

fn validate_entries(turns: &[TurnOutlineEntry]) -> Result<(), String> {
    let mut previous = None;
    for entry in turns {
        if previous.is_some_and(|turn| entry.turn <= turn) {
            return Err("turn outline entries must be strictly increasing by turn".to_string());
        }
        previous = Some(entry.turn);
        if char_len(&entry.prompt) > PROMPT_PREVIEW_LIMIT {
            return Err(format!(
                "turn {} prompt exceeds the {PROMPT_PREVIEW_LIMIT}-character budget",
                entry.turn
            ));
        }
        if char_len(&entry.response) > RESPONSE_PREVIEW_LIMIT {
            return Err(format!(
                "turn {} response exceeds the {RESPONSE_PREVIEW_LIMIT}-character budget",
                entry.turn
            ));
        }
        match (entry.end_seq, entry.status) {
            (None, TurnOutlineStatus::Open) => {}
            (Some(end), status) if status != TurnOutlineStatus::Open => {
                if end < entry.seq {
                    return Err(format!(
                        "turn {} closes before it starts ({end} < {})",
                        entry.turn, entry.seq
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "turn {} must be open exactly while endSeq is absent",
                    entry.turn
                ));
            }
        }
    }
    Ok(())
}

/// Space-join text blocks, collapse whitespace, and cap at `limit` characters.
fn preview(content: &[ContentBlock], limit: usize) -> String {
    let window = limit.saturating_mul(2);
    let mut text = String::new();
    let mut text_len = 0;
    let mut unread = false;
    for block in content {
        let Some(block_text) = block.as_text() else {
            continue;
        };
        if text_len >= window {
            unread = true;
            break;
        }
        let clipped = char_len(block_text) > window;
        let chunk: String = if clipped {
            block_text.chars().take(window).collect()
        } else {
            block_text.to_string()
        };
        if !text.is_empty() {
            text.push(' ');
            text_len += 1;
        }
        text_len += char_len(&chunk);
        text.push_str(&chunk);
        if clipped {
            unread = true;
            break;
        }
    }
    let normalized = collapse_whitespace(&text);
    let length = char_len(&normalized);
    if length > limit - 1 {
        let cut: String = normalized.chars().take(limit - 1).collect();
        return format!("{}…", cut.trim_end());
    }
    if unread {
        return format!("{normalized}…");
    }
    normalized
}

fn char_len(value: &str) -> usize {
    value.chars().count()
}

/// Replace every whitespace run with one space and trim both ends.
fn collapse_whitespace(value: &str) -> String {
    let mut collapsed = String::new();
    let mut pending_space = false;
    for character in value.chars() {
        if character.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_space && !collapsed.is_empty() {
            collapsed.push(' ');
        }
        pending_space = false;
        collapsed.push(character);
    }
    collapsed
}
