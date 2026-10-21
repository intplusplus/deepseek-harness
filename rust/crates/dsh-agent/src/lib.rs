//! `dsh-agent` — agent identity and the durable inbox projection.
//!
//! An agent is a session-backed identity plus two ordered pending-message
//! lists. The lists are not stored: they are folded from committed
//! `agent/inbox/spliced` session extension events, which is the Rust answer to
//! the TypeScript declaration-merged event.

#![forbid(unsafe_code)]

pub mod inbox;
pub mod types;

pub use inbox::{apply_splice, fold_inbox, InboxSplice, INBOX_SPLICED};
pub use types::{Agent, InboxState, InboxTarget};
