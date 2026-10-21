//! Pure folds over the log: request reconstruction, turn/step boundaries, and
//! the derived model history.
//!
//! These are the Rust counterparts of `foldRequestHeader`, the turn-boundary
//! projection, and `deriveMessages()`: every one is a pure function of the
//! committed events, so replay is re-derivation from the same log.

#![forbid(unsafe_code)]

use dsh_llm::Message;

use super::{
    event::{CoreEvent, SessionEvent},
    types::*,
};

/// The latest full `request/header` snapshot, or `None` before the first.
///
/// The latest snapshot reconstructs the header; the system prompt is not part
/// of it — that is derived history, the `system/message` event.
pub fn fold_request_header(events: &[SessionEvent]) -> Option<EpochHeader> {
    events.iter().rev().find_map(|event| match event {
        SessionEvent::Core(CoreEvent::RequestHeader { header, .. }) => {
            Some(header.as_ref().clone())
        }
        _ => None,
    })
}

/// The latest `request/context` record, or `None` before the first.
pub fn fold_request_context(events: &[SessionEvent]) -> Option<RequestContext> {
    events.iter().rev().find_map(|event| match event {
        SessionEvent::Core(CoreEvent::RequestContext(context)) => Some(context.clone()),
        _ => None,
    })
}

/// Which step boundary a folded position marks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepBoundaryKind {
    Start,
    End,
}

/// One step boundary and its log position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepBoundary {
    pub kind: StepBoundaryKind,
    pub seq: SessionSeq,
}

/// Turn and step boundaries folded from one session log.
///
/// The key is registered by the agent loop and absent otherwise: without it no
/// turn events exist, so readers treat the default as "no open turn / no
/// boundaries" — capability absence, not corrupt state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TurnBoundaryProjection {
    /// Seq of the open turn's `turn/start`, or `None` between turns.
    pub open_turn_start_seq: OptionalSessionSeq,
    /// Seq of the latest `step/start` event, or `None` before the first step.
    pub last_step_start_seq: OptionalSessionSeq,
    /// The latest step boundary and its seq, or `None` before the first.
    pub last_step_boundary: Option<StepBoundary>,
    /// Turn number of the latest `turn/start`; 0 before the first turn.
    pub last_turn: u64,
}

/// Fold the turn/step boundaries from a session log.
pub fn fold_turn_boundary(events: &[SessionEvent]) -> TurnBoundaryProjection {
    let mut projection = TurnBoundaryProjection::default();
    for (index, event) in events.iter().enumerate() {
        let seq = SessionSeq::unchecked(index as u64);
        match event {
            SessionEvent::Core(CoreEvent::TurnStart { turn }) => {
                projection.open_turn_start_seq = Some(seq);
                projection.last_turn = *turn;
            }
            SessionEvent::Core(CoreEvent::TurnEnd { .. }) => {
                projection.open_turn_start_seq = None;
            }
            SessionEvent::Core(CoreEvent::StepStart { .. }) => {
                projection.last_step_start_seq = Some(seq);
                projection.last_step_boundary = Some(StepBoundary {
                    kind: StepBoundaryKind::Start,
                    seq,
                });
            }
            SessionEvent::Core(CoreEvent::StepEnd { .. }) => {
                projection.last_step_boundary = Some(StepBoundary {
                    kind: StepBoundaryKind::End,
                    seq,
                });
            }
            _ => {}
        }
    }
    projection
}

/// Project model-visible history from the log.
///
/// The five message-bearing event types fold in log order; lifecycle, header,
/// context, and attempt events produce no model message. Model-visible means
/// logged: every model request is reconstructable from these events alone.
///
/// The v1 fold does not yet implement the system-node surface rules
/// (normalization to the first system node, clearing on an empty rendering);
/// those rewrite history rather than select it and land with the system-prompt
/// subsystem.
pub fn derive_messages(events: &[SessionEvent]) -> Vec<Message> {
    let mut messages = Vec::new();
    for event in events {
        match event {
            SessionEvent::Core(CoreEvent::SystemMessage { message, .. }) => {
                messages.push(Message::System(message.as_ref().clone()));
            }
            SessionEvent::Core(CoreEvent::UserMessage { message }) => {
                messages.push(Message::User(message.as_ref().clone()));
            }
            SessionEvent::Core(CoreEvent::DeveloperMessage { message, .. }) => {
                messages.push(Message::Developer(message.as_ref().clone()));
            }
            SessionEvent::Core(CoreEvent::AssistantMessage { message, .. }) => {
                messages.push(Message::Assistant(message.as_ref().clone()));
            }
            SessionEvent::Core(CoreEvent::ToolResult { message, .. }) => {
                messages.push(Message::Tool(message.as_ref().clone()));
            }
            _ => {}
        }
    }
    messages
}
