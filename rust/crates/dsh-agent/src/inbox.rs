//! The durable inbox fold: pending input reconstructed from committed
//! `agent/inbox/spliced` session extension events.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use dsh_llm::UserMessage;
use dsh_session::SessionEvent;

use super::{InboxState, InboxTarget};

/// The `type` tag of the inbox splice event declared by this package.
pub const INBOX_SPLICED: &str = "agent/inbox/spliced";

/// One normalized mutation of an agent's durable pending-message lists.
///
/// The session-projection registry applies the committed event before
/// `Session::append()` returns; this fold is the detached reader's equivalent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InboxSplice {
    /// Which list the mutation applies to.
    pub target: InboxTarget,
    /// Splice start position within the target list.
    pub start: u64,
    /// Number of entries removed at `start`; absent removes none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub removed_count: Option<u64>,
    /// Entries inserted at `start`, in order.
    pub inserted: Vec<UserMessage>,
    /// `canceled` marks a splice that canceled pending input. It is outcome
    /// metadata: the committed mutation itself still applies to the fold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
}

/// Apply one committed splice to the fold state.
pub fn apply_splice(state: &mut InboxState, splice: &InboxSplice) {
    let list = state.target_mut(splice.target);
    let start = (splice.start.min(list.len() as u64)) as usize;
    let removed = splice
        .removed_count
        .unwrap_or(0)
        .min((list.len() - start) as u64) as usize;
    list.splice(start..start + removed, splice.inserted.iter().cloned());
}

/// Reconstruct the pending inbox from a session log.
///
/// Every committed splice applies in order, so the fold is a pure function of
/// the durable events. A splice whose payload no longer decodes is skipped:
/// the fold tolerates a writer that shipped a payload shape this reader does
/// not know, exactly as detached readers supply projection definitions
/// explicitly rather than failing the whole state.
pub fn fold_inbox(events: &[SessionEvent]) -> InboxState {
    let mut state = InboxState::default();
    for event in events {
        let Some(extension) = event.extension() else {
            continue;
        };
        if extension.kind() != INBOX_SPLICED {
            continue;
        }
        let Ok(splice) = extension.decode::<InboxSplice>() else {
            continue;
        };
        apply_splice(&mut state, &splice);
    }
    state
}
