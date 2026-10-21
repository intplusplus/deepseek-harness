//! Turn outline over the session log, ported from
//! `packages/session/session-turn-outline`.
//!
//! The crate owns one projection unit, [`outline_projection`], keyed
//! `turnOutline`. Every `turn/start` appends a stable entry (turn number and
//! boundary seq), the first human prompt fills its prompt preview, the newest
//! text-bearing assistant message fills an internal draft, and `turn/end`
//! commits the response together with the closing seq, step/tool counts, and
//! cancellation or error status.
//!
//! Entries are strictly increasing by turn, so the array order is the stable
//! id order. Unknown extension events and unrelated core events fold to the
//! same reference. The turn list lives behind an `Arc`; the client view returns
//! that allocation, so draft-only changes do not notify the change feed.
//!
//! # Differences from the TypeScript unit
//!
//! The TypeScript entry carries only `turn`, `seq`, `prompt`, and `response`.
//! This port appends `endSeq`, `steps`, `toolCalls`, `toolResults`,
//! `toolErrors`, and `status`, because the Rust reconstruction has no
//! client-side window fold to derive them. `stateVersion` is 3 for that reason.
//! Preview limits count Unicode scalar values instead of UTF-16 code units so
//! slicing never splits a character.

#![forbid(unsafe_code)]

pub mod projection;

pub use projection::{
    outline_projection, TurnOutlineEntry, TurnOutlineState, TurnOutlineStatus, KEY,
    PROMPT_PREVIEW_LIMIT, RESPONSE_PREVIEW_LIMIT, STATE_VERSION,
};
