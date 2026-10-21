//! The retry-chain identity.

#![forbid(unsafe_code)]

use std::fmt;

use serde::{Deserialize, Serialize};

/// Stable identity shared by every attempt in one request-step retry chain.
///
/// One retry chain is one `llm/retry` → `llm/retry-started` sequence per
/// provider and policy. A policy switch produces a new chain, which is how a
/// replaced adapter registration gets a fresh budget instead of inheriting the
/// old one's spent retries.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RetryId(String);

impl RetryId {
    /// Brand a raw identifier. No validation, matching the TS constructor.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Mint a fresh chain identity (the TS `randomUUID()` source).
    pub fn random() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
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

impl fmt::Display for RetryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for RetryId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<RetryId> for String {
    fn from(id: RetryId) -> String {
        id.0
    }
}

impl AsRef<str> for RetryId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}
