//! Canonical session URIs and inline mention encoding.
//!
//! Ported from `packages/context/session-reference/src/uri.ts`. A URI is
//! `dsh-session:` followed by the unpadded base64url form of
//! `JSON.stringify(sessionId)`. Decoding re-encodes the parsed id and requires
//! the exact same URI, which rejects padded, non-canonical, and
//! whitespace-bearing spellings.

#![forbid(unsafe_code)]

use serde_json::Value;

use super::config::SessionReferenceError;
use super::types::{SessionReferenceId, SessionReferenceInput};
use crate::js;

/// URI scheme reserved for DeepSeek Harness session snapshots.
pub const SESSION_REFERENCE_SCHEME: &str = "dsh-session:";

const BASE64URL_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Encode any JavaScript session-id string as a canonical lossless URI.
pub fn encode_session_reference_uri(session_id: &SessionReferenceId) -> String {
    let payload = base64url_encode(js::string_literal(session_id.as_str()).as_bytes());
    format!("{SESSION_REFERENCE_SCHEME}{payload}")
}

/// Decode and canonicalize one session-reference URI.
///
/// # Errors
///
/// Returns [`SessionReferenceError::InvalidUri`] unless `uri` is the canonical
/// encoding of a JSON string.
pub fn decode_session_reference_uri(
    uri: &str,
) -> Result<SessionReferenceId, SessionReferenceError> {
    let invalid = || SessionReferenceError::InvalidUri {
        uri: js::string_literal(uri),
    };
    let Some(payload) = uri.strip_prefix(SESSION_REFERENCE_SCHEME) else {
        return Err(invalid());
    };
    if payload.is_empty() || !payload.bytes().all(is_base64url_byte) {
        return Err(invalid());
    }
    let bytes = base64url_decode(payload).ok_or_else(invalid)?;
    let decoded = String::from_utf8(bytes).map_err(|_| invalid())?;
    let parsed: Value = serde_json::from_str(&decoded).map_err(|_| invalid())?;
    let Value::String(session_id) = parsed else {
        return Err(invalid());
    };
    let session_id = SessionReferenceId::new(session_id);
    if encode_session_reference_uri(&session_id) != uri {
        return Err(invalid());
    }
    Ok(session_id)
}

/// Render a host-neutral Markdown mention carrying the canonical URI.
pub fn format_session_reference_mention(reference: &SessionReferenceInput) -> String {
    let label = escape_label(
        reference
            .label
            .as_deref()
            .unwrap_or_else(|| reference.session_id.as_str()),
    );
    format!(
        "@[{label}]({})",
        encode_session_reference_uri(&reference.session_id)
    )
}

/// Result of extracting canonical mentions from plain text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedSessionReferenceText {
    /// Text with opaque tokens replaced by readable `@label` spans.
    pub text: String,
    /// Structured references in first-appearance order, before deduplication.
    pub references: Vec<SessionReferenceInput>,
}

/// Extract Markdown mentions and bare canonical URIs from one text value.
///
/// Explicit Markdown mentions fail on any malformed URI. Bare text is treated
/// as a reference only when it has a non-empty base64url-shaped payload, then
/// still fails if that candidate is not canonical.
///
/// # Errors
///
/// Returns [`SessionReferenceError::InvalidUri`] for the first matched URI
/// that does not decode canonically.
pub fn parse_session_reference_text(
    text: &str,
) -> Result<ParsedSessionReferenceText, SessionReferenceError> {
    let characters: Vec<char> = text.chars().collect();
    let mut rendered = String::with_capacity(text.len());
    let mut references = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        if characters[index] == '@' && characters.get(index + 1) == Some(&'[') {
            if let Some((end, raw_label, uri)) = match_markdown(&characters, index) {
                let session_id = decode_session_reference_uri(&uri)?;
                let label = unescape_label(&raw_label);
                rendered.push('@');
                rendered.push_str(&label);
                references.push(SessionReferenceInput::with_label(session_id, label));
                index = end;
                continue;
            }
        }
        if starts_with_scheme(&characters, index) {
            let payload_start = index + SESSION_REFERENCE_SCHEME.chars().count();
            let mut end = payload_start;
            while end < characters.len() && is_base64url_character(characters[end]) {
                end += 1;
            }
            if end > payload_start {
                let uri: String = characters[index..end].iter().collect();
                let session_id = decode_session_reference_uri(&uri)?;
                rendered.push('@');
                rendered.push_str(session_id.as_str());
                references.push(SessionReferenceInput::new(session_id));
                index = end;
                continue;
            }
        }
        rendered.push(characters[index]);
        index += 1;
    }
    Ok(ParsedSessionReferenceText {
        text: rendered,
        references,
    })
}

/// Match the Markdown alternative at `start`, returning the exclusive end and
/// the raw label and URI.
fn match_markdown(characters: &[char], start: usize) -> Option<(usize, String, String)> {
    let mut cursor = start + 2;
    let mut label = String::new();
    loop {
        let character = *characters.get(cursor)?;
        if character == '\\' {
            let escaped = *characters.get(cursor + 1)?;
            if js::is_line_terminator(escaped) {
                return None;
            }
            label.push('\\');
            label.push(escaped);
            cursor += 2;
        } else if character == ']' {
            if characters.get(cursor + 1) != Some(&'(') {
                return None;
            }
            cursor += 2;
            break;
        } else {
            label.push(character);
            cursor += 1;
        }
    }
    let uri_start = cursor;
    while cursor < characters.len() {
        let character = characters[cursor];
        if character == ')' {
            let uri = characters[uri_start..cursor].iter().collect();
            return Some((cursor + 1, label, uri));
        }
        if js::is_whitespace(character) {
            return None;
        }
        cursor += 1;
    }
    None
}

/// Whether `characters[start..]` begins with the reserved URI scheme.
fn starts_with_scheme(characters: &[char], start: usize) -> bool {
    SESSION_REFERENCE_SCHEME
        .chars()
        .enumerate()
        .all(|(offset, expected)| characters.get(start + offset) == Some(&expected))
}

/// Escape the two characters that would end or break a Markdown label.
fn escape_label(label: &str) -> String {
    let mut escaped = String::with_capacity(label.len());
    for character in label.chars() {
        if matches!(character, '\\' | ']') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

/// Undo `/\\(.)/gu`: a backslash before any non-line-terminator yields that
/// character.
fn unescape_label(label: &str) -> String {
    let characters: Vec<char> = label.chars().collect();
    let mut unescaped = String::with_capacity(label.len());
    let mut index = 0;
    while index < characters.len() {
        if characters[index] == '\\' {
            if let Some(next) = characters.get(index + 1) {
                if !js::is_line_terminator(*next) {
                    unescaped.push(*next);
                    index += 2;
                    continue;
                }
            }
        }
        unescaped.push(characters[index]);
        index += 1;
    }
    unescaped
}

/// Encode bytes as unpadded base64url.
fn base64url_encode(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let first = u32::from(chunk[0]);
        let second = chunk.get(1).copied().map(u32::from);
        let third = chunk.get(2).copied().map(u32::from);
        let combined = (first << 16) | (second.unwrap_or(0) << 8) | third.unwrap_or(0);
        encoded.push(BASE64URL_ALPHABET[(combined >> 18) as usize & 0x3f] as char);
        encoded.push(BASE64URL_ALPHABET[(combined >> 12) as usize & 0x3f] as char);
        if second.is_some() {
            encoded.push(BASE64URL_ALPHABET[(combined >> 6) as usize & 0x3f] as char);
        }
        if third.is_some() {
            encoded.push(BASE64URL_ALPHABET[combined as usize & 0x3f] as char);
        }
    }
    encoded
}

/// Decode unpadded base64url bytes, rejecting impossible lengths and
/// characters.
fn base64url_decode(payload: &str) -> Option<Vec<u8>> {
    if payload.len() % 4 == 1 {
        return None;
    }
    let mut bytes = Vec::with_capacity(payload.len() / 4 * 3 + 2);
    let mut accumulator = 0u32;
    let mut bits = 0u32;
    for character in payload.chars() {
        let value = base64url_value(character)?;
        accumulator = (accumulator << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push((accumulator >> bits) as u8);
            accumulator &= (1 << bits) - 1;
        }
    }
    Some(bytes)
}

/// The six-bit value of one base64url character.
fn base64url_value(character: char) -> Option<u32> {
    match character {
        'A'..='Z' => Some(u32::from(character as u8 - b'A')),
        'a'..='z' => Some(u32::from(character as u8 - b'a') + 26),
        '0'..='9' => Some(u32::from(character as u8 - b'0') + 52),
        '-' => Some(62),
        '_' => Some(63),
        _ => None,
    }
}

/// Whether one ASCII byte belongs to the base64url alphabet.
fn is_base64url_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')
}

/// Whether one character belongs to the base64url alphabet.
fn is_base64url_character(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
}
