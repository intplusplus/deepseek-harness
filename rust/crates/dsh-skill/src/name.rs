//! The kebab-case skill-name grammar.
//!
//! TypeScript validates the public name with `^[a-z0-9]+(?:-[a-z0-9]+)*$` at
//! every boundary that accepts provider or user input. Rust makes the grammar
//! a newtype instead: a [`SkillName`] exists only after the check, so catalog
//! keys, summary fields, and render attributes cannot disagree about validity.

use std::fmt;
use std::ops::Deref;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::validate::SkillError;

/// Whether a string is a valid public skill name.
///
/// The grammar is one or more lowercase `[a-z0-9]` segments joined by single
/// hyphens: `a`, `pdf-tools`, and `2026-report` are valid; empty strings,
/// uppercase, underscores, leading/trailing hyphens, and repeated hyphens are
/// not.
pub fn is_skill_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    name.split('-').all(|segment| {
        !segment.is_empty()
            && segment
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
    })
}

/// A skill name that satisfies the public kebab-case grammar.
///
/// Serializes as the bare string, so it can occupy the `name` field of a
/// catalog entry or summary without changing the wire format.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SkillName(String);

impl SkillName {
    /// Validate `name` and wrap it.
    pub fn new(name: impl Into<String>) -> Result<Self, SkillError> {
        let name = name.into();
        if is_skill_name(&name) {
            Ok(Self(name))
        } else {
            Err(SkillError::InvalidName { name })
        }
    }

    /// The name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Deref for SkillName {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for SkillName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SkillName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for SkillName {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SkillName {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::new(raw).map_err(D::Error::custom)
    }
}
