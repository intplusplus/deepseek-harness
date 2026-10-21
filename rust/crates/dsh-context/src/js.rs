//! JavaScript-compatible string and number rules used by the ported code.
//!
//! TypeScript compares, sorts, and prints strings with JavaScript semantics.
//! Rust differs in two places this crate cares about: the ECMAScript `\s`
//! character set, and the textual form of a number in a diagnostic. Both live
//! here so the ported rules use one definition.

#![forbid(unsafe_code)]

/// Whether `character` matches ECMAScript's `\s` (WhiteSpace + LineTerminator).
///
/// This is deliberately not [`char::is_whitespace`]: ECMAScript excludes
/// U+0085 NEXT LINE and includes U+FEFF ZERO WIDTH NO-BREAK SPACE.
pub fn is_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0020}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

/// Whether `character` is a JavaScript line terminator.
///
/// The regex `.` atom cannot match these without the `s` flag, which the
/// ported patterns do not use.
pub fn is_line_terminator(character: char) -> bool {
    matches!(character, '\u{000A}' | '\u{000D}' | '\u{2028}' | '\u{2029}')
}

/// Render `value` exactly as `JSON.stringify(string)` renders it.
///
/// Valid UTF-8 strings contain no lone surrogates, so the output is identical:
/// quotes, backslashes, and the five short escapes are used, remaining control
/// characters become lowercase `\u00xx`, and every other character is raw.
pub fn string_literal(value: &str) -> String {
    let mut rendered = String::with_capacity(value.len() + 2);
    rendered.push('"');
    for character in value.chars() {
        match character {
            '"' => rendered.push_str("\\\""),
            '\\' => rendered.push_str("\\\\"),
            '\u{0008}' => rendered.push_str("\\b"),
            '\u{000C}' => rendered.push_str("\\f"),
            '\n' => rendered.push_str("\\n"),
            '\r' => rendered.push_str("\\r"),
            '\t' => rendered.push_str("\\t"),
            control if control < '\u{0020}' => {
                rendered.push_str(&format!("\\u{:04x}", u32::from(control)));
            }
            other => rendered.push(other),
        }
    }
    rendered.push('"');
    rendered
}

/// Render `value` the way `String(value)` does for the values diagnostics name.
///
/// Rust's `f64` display differs for non-finite values and negative zero, which
/// are exactly the values the refresh-interval diagnostic rejects.
pub fn number_literal(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f64::INFINITY {
        return "Infinity".to_string();
    }
    if value == f64::NEG_INFINITY {
        return "-Infinity".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    format!("{value}")
}
