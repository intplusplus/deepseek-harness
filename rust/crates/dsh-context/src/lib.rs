//! Request-time context providers over the system-prompt registry.
//!
//! Ported from the host-independent rules and wire formats of three
//! TypeScript packages:
//!
//! - [`time_context`] — the durable request-clock reading, browser-zone
//!   policy, timestamp formatting, and injection scheduling;
//! - [`file_reference`] — the browser-safe `@path` completion grammar and the
//!   exact model guidance for user-selected paths;
//! - [`session_reference`] — canonical session URIs, mention parsing,
//!   current-surface projection, byte-bounded previews, spill notices, and the
//!   durable `session-reference` source record.
//!
//! Environment edges are injected rather than read: clocks and zones arrive as
//! arguments or through [`time_context::TimeZoneResolver`], source snapshots
//! are supplied by the host, and spill storage is a
//! [`session_reference::TextSpillSink`]. No code here reads the process clock,
//! filesystem, environment, or randomness.

#![forbid(unsafe_code)]

mod js;

pub mod file_reference;
pub mod session_reference;
pub mod time_context;

pub use file_reference::{
    active_at_token, format_file_mention, register_section as register_file_reference_section,
    ActiveAtToken, FileReferenceCandidate, FileReferenceKind, FILE_REFERENCE_PROMPT,
    FILE_REFERENCE_SECTION_NAME,
};
pub use js::{is_line_terminator, is_whitespace, number_literal, string_literal};
pub use time_context::{
    fold_time_context, format_duration, format_timestamp, observe_event,
    register_context as register_time_context, render_time_context, request_messages,
    resolve_config, should_inject, validate_refresh_interval_ms, BrowserTimeZoneContext,
    Config as TimeContextConfig, ResolvedConfig as ResolvedTimeContextConfig, TimeContextError,
    TimeContextObservation, TimeContextState, TimeReading, TimeZoneResolver, ZoneResolution,
    ZoneTable, ZoneTableError, DEFAULT_REFRESH_INTERVAL_MS, TIME_CONTEXT_NAME,
    TIME_CONTEXT_STATE_VERSION,
};
