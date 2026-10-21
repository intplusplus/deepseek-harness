//! The `system-prompt/change` notification as a durable session event.
//!
//! TypeScript declares the event on the Cordis bus, not in `SessionEventMap`:
//!
//! ```ts
//! declare module '@deepseek-ai/cordis' {
//!   interface Events {
//!     'system-prompt/change'(): void
//!   }
//! }
//! ```
//!
//! It is a volatile `emit` — every registration and every disposal of a
//! prompt provider notifies listeners, and the notification carries no data.
//! Rust has no declaration merging, so the bridge into the durable log is
//! written here by hand, the same way `dsh-retry` bridges `llm/retry`: the
//! event reaches [`SessionEvent::Extension`] through [`encode_change`], and
//! [`decode_change`] keeps the extension slot's invariant — a
//! `system-prompt/change` row whose payload does not match the declared shape
//! comes back as `Some(Err(..))`, never as a silently degraded payload.
//!
//! The deviation from TypeScript is deliberate: there the notification dies
//! with the bus, while here the host layer records prompt churn in the
//! append-only log so a replay can see every registration and disposal.

#![forbid(unsafe_code)]

use serde::Deserialize;
use serde_json::Map;

use dsh_session::SessionEvent;

/// Tag of the prompt-registry change notification.
pub const SYSTEM_PROMPT_CHANGE_TAG: &str = "system-prompt/change";

/// The declared shape of one `system-prompt/change` payload: nothing but the
/// envelope's own `type` tag.
///
/// `type` is declared as a field so that [`serde`]'s `deny_unknown_fields`
/// covers the *whole* payload — a `system-prompt/change` row carrying any
/// other key is a malformed payload, not one whose extras are silently
/// tolerated. The tag itself is validated by [`decode_change`] before the
/// payload is read, so the field only ever holds the tag on the decode path.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemPromptChangeEvent {
    /// The envelope tag this payload was read from.
    #[serde(rename = "type")]
    kind: String,
}

impl SystemPromptChangeEvent {
    /// The envelope tag this payload was read from.
    pub fn kind(&self) -> &str {
        &self.kind
    }
}

/// Encode one change notification as the extension event the log stores.
///
/// Infallible by construction: the payload is empty and the tag is a
/// constant, so there is nothing for `serde_json` to reject. The wire bytes
/// are exactly `{"type":"system-prompt/change"}` — `type` first, as every
/// extension row opens with its discriminator, and nothing else, as the
/// TypeScript declaration takes no arguments.
pub fn encode_change() -> SessionEvent {
    SessionEvent::extension_event(SYSTEM_PROMPT_CHANGE_TAG, Map::new())
}

/// Decode one change notification from the log.
///
/// `None` means the event carries a different tag; `Some(Err(..))` means it
/// is a `system-prompt/change` event whose payload does not match the
/// declared shape.
pub fn decode_change(
    event: &SessionEvent,
) -> Option<Result<SystemPromptChangeEvent, serde_json::Error>> {
    let extension = event.extension()?;
    if extension.kind() != SYSTEM_PROMPT_CHANGE_TAG {
        return None;
    }
    Some(extension.decode::<SystemPromptChangeEvent>())
}
