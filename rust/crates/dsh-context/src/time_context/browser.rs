//! Browser-zone derivation and the model-facing request-zone policy line.
//!
//! A browser reports its zone on an ordinary user-rpc message; the harness
//! validates that value before it can select the display zone. Everything here
//! is pure: the semantic zone checks go through an injected
//! [`TimeZoneResolver`](super::TimeZoneResolver), never through the host.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use dsh_llm::{MessageSource, UserMessage};

use super::TimeContextError;
use crate::js;

/// Browser-zone facts derived from user-rpc messages in one open turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BrowserTimeZoneContext {
    /// Every user-rpc message in the turn carries the same zone.
    Resolved {
        /// The unique canonical zone.
        time_zone: String,
    },
    /// The turn carries more than one zone, sorted and duplicate-free.
    Mixed {
        /// Every distinct canonical zone.
        time_zones: Vec<String>,
    },
    /// No user-rpc message in the turn carries a zone.
    Missing,
}

/// Read and validate a Host-canonicalized browser zone from one message source.
///
/// The TypeScript check is `source.kind === 'user'` with string `rpcId` and
/// `clientTimeZone` fields. Rust's core `MessageSource::User` carries no extra
/// fields, so the fields are representable only on a producer-declared `user`
/// extension source; that is the shape this function reads.
///
/// # Errors
///
/// Returns [`TimeContextError::BrowserZoneSyntax`] for a value that is neither
/// `UTC` nor IANA `Area/Location` shaped,
/// [`TimeContextError::BrowserZoneUnsupported`] when the resolver does not know
/// it, and [`TimeContextError::BrowserZoneNotCanonical`] when the resolver
/// canonicalizes it to a different label.
pub fn browser_time_zone(
    source: &MessageSource,
    zones: &dyn super::TimeZoneResolver,
) -> Result<Option<String>, TimeContextError> {
    if source.kind() != "user" {
        return Ok(None);
    }
    let MessageSource::Extension(extension) = source else {
        return Ok(None);
    };
    let fields = extension.fields();
    let rpc_id = fields.get("rpcId").and_then(serde_json::Value::as_str);
    let Some(value) = fields
        .get("clientTimeZone")
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(None);
    };
    if rpc_id.is_none() {
        return Ok(None);
    }
    if value != "UTC" && !is_iana_zone_syntax(value) {
        return Err(TimeContextError::BrowserZoneSyntax {
            value: js::string_literal(value),
        });
    }
    let canonical =
        zones
            .canonical(value)
            .ok_or_else(|| TimeContextError::BrowserZoneUnsupported {
                value: js::string_literal(value),
            })?;
    if canonical != value {
        return Err(TimeContextError::BrowserZoneNotCanonical {
            value: js::string_literal(value),
        });
    }
    Ok(Some(value.to_string()))
}

/// Derive the unique, mixed, or missing browser zone for one open turn.
///
/// # Errors
///
/// Returns the validation failures documented on [`browser_time_zone`].
pub fn derive_browser_time_zone_context(
    messages: &[UserMessage],
    zones: &dyn super::TimeZoneResolver,
) -> Result<BrowserTimeZoneContext, TimeContextError> {
    derive_from_sources(messages.iter().map(|message| &message.source), zones)
}

/// Derive browser-zone facts from message sources.
///
/// The result orders zones by their byte order, which for the ASCII-only
/// syntax the grammar admits is also JavaScript's UTF-16 code-unit order.
///
/// # Errors
///
/// Returns the validation failures documented on [`browser_time_zone`].
pub fn derive_from_sources<'a>(
    sources: impl IntoIterator<Item = &'a MessageSource>,
    zones: &dyn super::TimeZoneResolver,
) -> Result<BrowserTimeZoneContext, TimeContextError> {
    let mut time_zones = BTreeSet::new();
    for source in sources {
        if let Some(time_zone) = browser_time_zone(source, zones)? {
            time_zones.insert(time_zone);
        }
    }
    let mut time_zones: Vec<String> = time_zones.into_iter().collect();
    if time_zones.is_empty() {
        return Ok(BrowserTimeZoneContext::Missing);
    }
    if time_zones.len() == 1 {
        let time_zone = time_zones.remove(0);
        return Ok(BrowserTimeZoneContext::Resolved { time_zone });
    }
    Ok(BrowserTimeZoneContext::Mixed { time_zones })
}

/// Render the model instruction for one browser-zone context.
pub fn render_browser_time_zone_context(context: &BrowserTimeZoneContext) -> String {
    match context {
        BrowserTimeZoneContext::Resolved { time_zone } => format!(
            "Browser time zone for this request: {time_zone}. \
             Interpret otherwise-unqualified dates and times in this zone."
        ),
        BrowserTimeZoneContext::Mixed { time_zones } => format!(
            "Browser time zone for this request: mixed {}. \
             Ask the user to clarify otherwise-unqualified dates and times.",
            json_array(time_zones)
        ),
        BrowserTimeZoneContext::Missing => "Browser time zone for this request: unavailable. \
             Ask the user to clarify otherwise-unqualified dates and times."
            .to_string(),
    }
}

/// Render a string array exactly as `JSON.stringify(string[])` does.
fn json_array(values: &[String]) -> String {
    let mut rendered = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            rendered.push(',');
        }
        rendered.push_str(&js::string_literal(value));
    }
    rendered.push(']');
    rendered
}

/// The IANA `Area/Location` shape the TypeScript grammar admits.
fn is_iana_zone_syntax(value: &str) -> bool {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    if !first.is_ascii_alphabetic() {
        return false;
    }
    // The first character is ASCII, so its UTF-8 length is one byte.
    let mut segments = value[1..].split('/');
    let area = segments.next().unwrap_or_default();
    if !area.chars().all(is_zone_character) {
        return false;
    }
    let mut locations = 0;
    for location in segments {
        if location.is_empty() || !location.chars().all(is_zone_character) {
            return false;
        }
        locations += 1;
    }
    locations > 0
}

/// One character the IANA zone grammar admits after the first.
fn is_zone_character(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '_' | '+' | '.' | '-')
}
