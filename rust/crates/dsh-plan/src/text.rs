//! ECMAScript whitespace, for the `.trim()` calls this port copies verbatim.
//!
//! Rust's `char::is_whitespace` and ECMAScript's `\s` disagree on two code
//! points: Rust omits the BOM (`U+FEFF`) and includes NEL (`U+0085`).
//! TypeScript's `trim()` uses the ECMAScript set, so trimming through Rust's
//! default would accept or reject different input at the plan-mode boundaries.

#![forbid(unsafe_code)]

/// Whether `ch` belongs to the ECMAScript WhiteSpace or LineTerminator set.
pub(crate) fn is_js_whitespace(ch: char) -> bool {
    matches!(
        ch,
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

/// Trim ECMAScript whitespace from both ends, as `String.prototype.trim` does.
pub(crate) fn js_trim(text: &str) -> &str {
    text.trim_matches(is_js_whitespace)
}
