//! The durable `tool/ptc-dispatch*` event vocabulary.
//!
//! Ported from the `SessionEventMap` declaration in
//! `packages/core/tools/src/types.ts` and the two producers in
//! `packages/core/tools/src/ptc.ts`.
//!
//! There are exactly two tags. A nested call appends one start record when the
//! scheduler actually starts it, and one settle record when it settles; a call
//! abandoned in the queue while the run was ending logs *neither*, so a start
//! always means the tool pipeline was entered. The run's own boundary is not a
//! PTC event: the enclosing `run_code` call is an ordinary core
//! `tool/call` + `tool/result` pair, which is why `tool/ptc-dispatch`'s docs
//! say the sub-calls "never re-enter model context".
//!
//! Field order is observable (`serde_json`'s `preserve_order`), and the
//! TypeScript producer inserts fields in a specific order that is **not** the
//! declaration order of the interface — `error` and `meta` sit inside the
//! survive `tool/result`'s own vocabulary, between `isError` and `content`:
//!
//! ```text
//! tool/ptc-dispatch:
//!   rootCallId, parentCallId, subCallId, name, arguments,
//!   isError, [error], content, [meta]
//! ```
//!
//! [`PtcDispatch`] declares its fields in exactly that order so the derived
//! `Serialize` is byte-identical to the TypeScript log.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

use dsh_llm::{ContentBlock, ToolCallId};
use dsh_session::{ExtensionEvent, SessionEvent, ToolErrorInfo};

/// Tag of the `run_code` sub-dispatch start record.
pub const PTC_DISPATCH_START_TAG: &str = "tool/ptc-dispatch-start";
/// Tag of the `run_code` sub-dispatch settle record.
pub const PTC_DISPATCH_TAG: &str = "tool/ptc-dispatch";

/// Every tag this crate owns.
pub const PTC_EVENT_TAGS: &[&str] = &[PTC_DISPATCH_START_TAG, PTC_DISPATCH_TAG];

/// Payload of one sub-dispatch start.
///
/// Appended when the scheduler starts the call, not when the program submits
/// it. `arguments` is the JSON-normalized value the dispatch received,
/// normalized *before* dispatch, so this append cannot fail on payload shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PtcDispatchStart {
    /// Root model-requested call owning this execution tree.
    pub root_call_id: ToolCallId,
    /// The outer `run_code` call the program runs inside.
    pub parent_call_id: ToolCallId,
    /// Opaque sub-call id, `<parent>:ptc:<n>`.
    pub sub_call_id: ToolCallId,
    /// The dispatched tool name.
    pub name: String,
    /// The JSON-normalized arguments the tool received.
    pub arguments: Value,
}

impl PtcDispatchStart {
    /// Create one start record.
    pub fn new(
        root_call_id: ToolCallId,
        parent_call_id: ToolCallId,
        sub_call_id: ToolCallId,
        name: impl Into<String>,
        arguments: Value,
    ) -> Self {
        Self {
            root_call_id,
            parent_call_id,
            sub_call_id,
            name: name.into(),
            arguments,
        }
    }
}

/// Payload of one sub-dispatch settle.
///
/// The outcome is expressed in `tool/result`'s own vocabulary — `content` plus
/// `isError`, the optional structured `error`, and the optional presentation
/// `meta` — so a UI or SDK renders a nested call through the exact path it
/// uses for a native one.
///
/// `arguments` is a *sibling parse* of the dispatched value: byte-identical
/// JSON, a separate object, so a tool that mutates its arguments cannot
/// desynchronize this record from what it actually received.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PtcDispatch {
    /// Root model-requested call owning this execution tree.
    pub root_call_id: ToolCallId,
    /// The outer `run_code` call the program runs inside.
    pub parent_call_id: ToolCallId,
    /// Opaque sub-call id, shared with the start record.
    pub sub_call_id: ToolCallId,
    /// The dispatched tool name.
    pub name: String,
    /// The JSON-normalized arguments the tool received.
    pub arguments: Value,
    /// Whether the sub-call settled as an error.
    pub is_error: bool,
    /// Structured failure identity and raw user-facing detail, outside model
    /// content. Present only on a failure that carried one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ToolErrorInfo>,
    /// The sub-call's complete model-facing content.
    pub content: Vec<ContentBlock>,
    /// Replayable presentation metadata, separate from model content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl PtcDispatch {
    /// Create one successful settle record.
    pub fn success(
        root_call_id: ToolCallId,
        parent_call_id: ToolCallId,
        sub_call_id: ToolCallId,
        name: impl Into<String>,
        arguments: Value,
        content: Vec<ContentBlock>,
    ) -> Self {
        Self {
            root_call_id,
            parent_call_id,
            sub_call_id,
            name: name.into(),
            arguments,
            is_error: false,
            error: None,
            content,
            meta: None,
        }
    }

    /// Create one failed settle record.
    pub fn failure(
        root_call_id: ToolCallId,
        parent_call_id: ToolCallId,
        sub_call_id: ToolCallId,
        name: impl Into<String>,
        arguments: Value,
        content: Vec<ContentBlock>,
    ) -> Self {
        Self {
            is_error: true,
            ..Self::success(
                root_call_id,
                parent_call_id,
                sub_call_id,
                name,
                arguments,
                content,
            )
        }
    }

    /// Attach the structured failure identity.
    pub fn with_error(mut self, error: ToolErrorInfo) -> Self {
        self.error = Some(error);
        self
    }

    /// Attach the presentation payload.
    pub fn with_meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// One decoded `tool/ptc-dispatch*` event.
#[derive(Clone, Debug, PartialEq)]
pub enum PtcEvent {
    /// A started sub-dispatch.
    Start(PtcDispatchStart),
    /// A settled sub-dispatch.
    ///
    /// Boxed because the settle record carries the full content blocks while
    /// the start record is a small header; the log stores JSON, so the box is
    /// invisible on the wire.
    Settle(Box<PtcDispatch>),
}

impl PtcEvent {
    /// The event's `type` tag.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Start(_) => PTC_DISPATCH_START_TAG,
            Self::Settle(_) => PTC_DISPATCH_TAG,
        }
    }

    /// The sub-call id the two records pair on.
    pub fn sub_call_id(&self) -> &ToolCallId {
        match self {
            Self::Start(event) => &event.sub_call_id,
            Self::Settle(event) => &event.sub_call_id,
        }
    }

    /// Encode this event as the extension event the log stores.
    pub fn encode(&self) -> Result<SessionEvent, serde_json::Error> {
        match self {
            Self::Start(event) => encode_payload(PTC_DISPATCH_START_TAG, event),
            Self::Settle(event) => encode_payload(PTC_DISPATCH_TAG, event),
        }
    }

    /// Decode one event from the log.
    ///
    /// `None` means the event carries a different tag. `Some(Err(..))` means it
    /// is one of this crate's tags whose payload does not match the declared
    /// type — a known tag is never silently degraded to an opaque payload.
    pub fn decode(event: &SessionEvent) -> Option<Result<Self, serde_json::Error>> {
        let extension = event.extension()?;
        let decoded = match extension.kind() {
            PTC_DISPATCH_START_TAG => {
                decode_payload::<PtcDispatchStart>(extension).map(Self::Start)
            }
            PTC_DISPATCH_TAG => decode_payload::<PtcDispatch>(extension)
                .map(|payload| Self::Settle(Box::new(payload))),
            _ => return None,
        };
        Some(decoded)
    }
}

/// Deserialize an extension payload as `T` with its `type` tag removed.
///
/// The stamped `type` discriminator is envelope metadata, not payload: the
/// payload types declare `deny_unknown_fields` without a `type` field, so
/// decoding the verbatim payload would always reject them.
///
/// `deserialize_with`'s error surface also mentions the missing discriminator
/// when the tag is absent, which [`PtcEvent::decode`] surfaces as a decode
/// error rather than a silent `None`.
fn decode_payload<T: for<'de> Deserialize<'de>>(
    extension: &ExtensionEvent,
) -> Result<T, serde_json::Error> {
    let mut payload = extension.payload().clone();
    payload.remove("type");
    serde_json::from_value(Value::Object(payload))
}

fn encode_payload<T: Serialize>(tag: &str, event: &T) -> Result<SessionEvent, serde_json::Error> {
    match serde_json::to_value(event)? {
        Value::Object(payload) => Ok(SessionEvent::Extension(ExtensionEvent::new(tag, payload))),
        other => Err(serde::ser::Error::custom(format!(
            "a session event payload must be a JSON object, got {other}"
        ))),
    }
}
