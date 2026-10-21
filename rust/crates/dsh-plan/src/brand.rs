//! Branded identifiers owned by plan mode.
//!
//! The TypeScript package imports `CommandId` from `@deepseek-ai/dsh-commands`.
//! Rust has no command crate yet, so the identifier is defined here as a
//! nominal newtype: the command lifecycle events this crate folds carry it, and
//! a bare `String` would let a session id or tool call id slip into the same
//! slot.

#![forbid(unsafe_code)]

use std::fmt;

use serde::{Deserialize, Serialize};

/// Identity of one resolved slash-command invocation.
///
/// The constructor performs no validation, matching the TypeScript brand
/// helper; the command executor owns minting.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CommandId(String);

impl CommandId {
    /// Brand a raw command identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the brand, yielding the raw identifier.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for CommandId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for CommandId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<CommandId> for String {
    fn from(id: CommandId) -> Self {
        id.0
    }
}

impl AsRef<str> for CommandId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}
