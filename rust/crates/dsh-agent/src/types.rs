//! Agent identity and the pending-inbox value types.
//!
//! The TypeScript package declares `agent/inbox/spliced` by declaration
//! merging into `SessionEventMap`. Rust has no declaration merging, so the
//! event itself lives in the session extension slot and the typed payload
//! lives here, next to the fold that consumes it.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use dsh_llm::UserMessage;
use dsh_session::SessionId;

/// Public agent handle: session-backed identity. The runtime face augments its
/// live capabilities.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Agent {
    /// Session-backed agent identity.
    pub id: SessionId,
}

impl Agent {
    /// An agent over an existing session.
    pub fn new(id: SessionId) -> Self {
        Self { id }
    }
}

/// One of the two ordered pending-message lists owned by an agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InboxTarget {
    NextTurn,
    NextStep,
}

/// Complete pending inbox value reconstructed from durable splices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InboxState {
    /// Input that waits for a waking message before the next turn opens.
    pub next_turn: Vec<UserMessage>,
    /// Input admitted inside the current turn, at the next step claim.
    pub next_step: Vec<UserMessage>,
}

impl InboxState {
    /// The pending list of one target.
    pub fn target(&self, target: InboxTarget) -> &[UserMessage] {
        match target {
            InboxTarget::NextTurn => &self.next_turn,
            InboxTarget::NextStep => &self.next_step,
        }
    }

    /// The pending list of one target, mutably.
    pub fn target_mut(&mut self, target: InboxTarget) -> &mut Vec<UserMessage> {
        match target {
            InboxTarget::NextTurn => &mut self.next_turn,
            InboxTarget::NextStep => &mut self.next_step,
        }
    }

    /// Total pending input across both targets.
    pub fn len(&self) -> usize {
        self.next_turn.len() + self.next_step.len()
    }

    /// Whether both targets are empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
