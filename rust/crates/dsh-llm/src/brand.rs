//! Branded identifiers.
//!
//! The TypeScript owners brand plain strings (`Branded<'MessageId'>`) to get
//! nominal typing across package boundaries. Rust newtypes give the same
//! compile-time distinction; the constructors mirror the TS `*Id(value)`
//! helpers and perform no validation, exactly like them.

#![forbid(unsafe_code)]

use std::fmt;

use serde::{Deserialize, Serialize};

macro_rules! branded_id {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Brand a raw identifier. No validation, matching the TS constructor.
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
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
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

branded_id!(
    /// Stable identity carried by one message across inbox, log, and
    /// model-request boundaries.
    MessageId
);

impl MessageId {
    /// Fresh random message identity (the TS `randomUUID()` source).
    pub fn random() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::random()
    }
}

branded_id!(
    /// Correlates a model-issued tool call with its result. Provider-issued
    /// for real adapters; synthesized by mocks and assembler fallbacks.
    ToolCallId
);

branded_id!(
    /// Provider-issued request identifier retained for diagnostics across
    /// package boundaries.
    ProviderRequestId
);

branded_id!(
    /// Identity of one model streaming attempt, unique within one agent
    /// lifecycle.
    LlmAttemptId
);

branded_id!(
    /// Adapter-owned identifier for one model's selectable reasoning effort.
    ReasoningEffortId
);
