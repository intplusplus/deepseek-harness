//! `dsh-session` — the append-only, event-sourced session log.
//!
//! A session is the single source of truth for an agent's interaction history:
//! an append-only log of typed events from which model history is derived.
//! The vocabulary is merge-extensible — the core owns the closed [`CoreEvent`]
//! union, and plugins extend the log through the [`SessionEvent::Extension`]
//! slot, mirroring the TypeScript declaration-merging design.

#![forbid(unsafe_code)]

pub mod event;
pub mod fold;
pub mod jsonl;
pub mod log;
pub mod types;

pub use event::{CoreEvent, ExtensionEvent, SessionEvent, CORE_EVENT_TAGS};
pub use fold::{
    derive_messages, fold_request_context, fold_request_header, fold_turn_boundary, StepBoundary,
    StepBoundaryKind, TurnBoundaryProjection,
};
pub use jsonl::{
    decode, encode, latest_generation, parse_generation, read_session, session_file_name, verify,
    write_session,
};
pub use log::Session;
pub use types::{
    AgentCancelCause, EpochHeader, OptionalSessionSeq, RequestContext, RequestHeaderReason,
    SessionError, SessionHeader, SessionId, SessionLogOffset, SessionMeta, SessionOrigin,
    SessionSeq, SessionSeqCursor, SystemPromptUpdate, ToolErrorInfo, TurnEndCancelCause,
    TurnEndReason, MAX_SAFE_INTEGER, SESSION_FORMAT_VERSION,
};
