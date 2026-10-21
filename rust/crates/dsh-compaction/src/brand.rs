//! Branded compaction identities.
//!
//! TypeScript brands plain strings (`Branded<'CompactionId'>`) so a command
//! identity cannot be substituted for a compaction identity. The newtypes here
//! provide the same nominal separation; constructors mirror the TS helpers and
//! perform no validation.

#![forbid(unsafe_code)]

use std::fmt;

use serde::{Deserialize, Serialize};

macro_rules! branded_string {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Brand a raw identifier. No validation, matching the TypeScript
            /// constructor.
            pub fn new(id: impl Into<String>) -> Self {
                Self(id.into())
            }

            /// The branded identifier as a string slice.
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Consume the brand, yielding the raw identifier.
            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl From<String> for $name {
            fn from(id: String) -> Self {
                Self(id)
            }
        }

        impl From<$name> for String {
            fn from(id: $name) -> String {
                id.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

branded_string!(
    /// Stable identity shared by one compaction start/summary/end transaction.
    CompactionId
);

impl CompactionId {
    /// Mint a fresh transaction identity (the TS `randomUUID()` source).
    pub fn random() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

branded_string!(
    /// Identity of the human command that initiated a manual compaction.
    ///
    /// The TypeScript owner is `@deepseek-ai/dsh-commands/brand`; the Rust
    /// commands crate does not exist yet, so the brand lives here with the
    /// same transparent wire representation. A future commands crate can
    /// re-export or replace it without changing the log format.
    CommandId
);
