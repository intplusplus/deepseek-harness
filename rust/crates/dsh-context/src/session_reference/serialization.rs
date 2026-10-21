//! Tag-safe JSON serialization for the model-visible reference envelope.
//!
//! Ported from `packages/context/session-reference/src/serialization.ts`.
//! Source text and labels are untrusted: replacing every literal `<` with
//! `\u003c` keeps the parsed value identical while preventing a hostile label
//! from spelling `</referenced-sessions>`.

#![forbid(unsafe_code)]

use serde::Serialize;

use super::config::SessionReferenceError;

/// Serialize JSON while preventing source data from spelling an XML-like tag.
///
/// # Errors
///
/// Returns [`SessionReferenceError::NotJsonSerializable`] when `value` has no
/// JSON representation.
pub fn stringify_tag_safe_json<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, SessionReferenceError> {
    let serialized =
        serde_json::to_string(value).map_err(|_| SessionReferenceError::NotJsonSerializable)?;
    Ok(serialized.replace('<', "\\u003c"))
}
