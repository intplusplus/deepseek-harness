//! The `/name` skill-invocation gesture.
//!
//! Transcribed from `packages/skill/tool-skill/src/index.ts`. The TypeScript
//! gesture is the global regex
//! `/(^|\s)\/([a-z0-9]+(?:-[a-z0-9]+)*)(?=\s|$)/g`; no regex crate is available
//! in this offline build, so this module hand-rolls the exact match — the
//! leading `(^|\s)` consumes one character, and the trailing lookahead must
//! hold without consuming.
//!
//! That boundary discipline is what keeps file paths (`/usr/bin`), fractions
//! (`5/8`), and hyphen-terminated fragments (`/abc-`) out: a second `/`, a
//! non-boundary character, or an incomplete kebab segment breaks the match.

use dsh_llm::{CoreContentBlock, MessageSource, UserMessage};

/// The tokens named by the claimed user messages, deduplicated in first-seen
/// order.
///
/// Every text block of direct user input is scanned; no other source can forge
/// a gesture. Names are unvalidated against the registry — unknown names stay
/// plain prose at the injecting boundary.
pub fn invoked_skill_names(messages: &[UserMessage]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for message in messages {
        if message.source.kind() != "user" {
            continue;
        }
        for block in &message.content {
            let text = match block {
                dsh_llm::ContentBlock::Core(CoreContentBlock::Text { text }) => text,
                _ => continue,
            };
            for name in scan_gestures(text) {
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
    }
    names
}

/// Whether `byte` is the whitespace the JavaScript `\s` class admits.
///
/// The set is the Unicode whitespace property plus the byte-order mark; the
/// two agree everywhere the skill grammar can appear in practice.
fn is_js_whitespace(character: char) -> bool {
    character.is_whitespace() || character == '\u{FEFF}'
}

/// Scan one text block for gesture tokens.
pub fn scan_gestures(text: &str) -> Vec<String> {
    let characters: Vec<char> = text.chars().collect();
    let mut names = Vec::new();
    let mut position = 0_usize;

    while position < characters.len() {
        // `(^|\s)`: an empty anchor at the search start, or one consumed
        // whitespace character. Try both alternatives at each position.
        let name_start = if position == 0 && characters[0] == '/' {
            // The `^` alternative, consuming nothing.
            1
        } else if is_js_whitespace(characters[position])
            && characters.get(position + 1) == Some(&'/')
        {
            // The `\s` alternative, consuming the whitespace.
            position + 2
        } else {
            position += 1;
            continue;
        };

        let Some((name_end, segments)) = scan_kebab(&characters, name_start) else {
            position += 1;
            continue;
        };
        // `(?=\s|$)` — the lookahead must hold without consuming.
        let boundary = match characters.get(name_end) {
            None => true,
            Some(&next) => is_js_whitespace(next),
        };
        if segments > 0 && boundary {
            names.push(characters[name_start..name_end].iter().collect());
            // A match consumes through the name; the scan resumes there, so
            // the whitespace before the next gesture stays available.
            position = name_end;
        } else {
            position += 1;
        }
    }
    names
}

/// Scan the greedy kebab grammar `[a-z0-9]+(?:-[a-z0-9]+)*` from `start`.
///
/// Returns the match end and its segment count. JavaScript's backtracking
/// cannot rescue a failed lookahead here: the repetition is maximal, and
/// shortening it only moves the boundary onto a name character, which is
/// never whitespace.
fn scan_kebab(characters: &[char], start: usize) -> Option<(usize, usize)> {
    let mut cursor = start;
    let mut segments = 0;
    let mut end = start;
    loop {
        let segment_start = cursor;
        while cursor < characters.len()
            && (characters[cursor].is_ascii_lowercase() || characters[cursor].is_ascii_digit())
        {
            cursor += 1;
        }
        if cursor == segment_start {
            break;
        }
        segments += 1;
        end = cursor;
        // The optional `-segment` repetition: taken only when a full segment
        // follows the hyphen.
        let takes_hyphen = cursor + 1 < characters.len()
            && characters[cursor] == '-'
            && (characters[cursor + 1].is_ascii_lowercase()
                || characters[cursor + 1].is_ascii_digit());
        if takes_hyphen {
            cursor += 1;
            continue;
        }
        break;
    }
    (segments > 0).then_some((end, segments))
}

/// Whether `source` is a direct human prompt.
pub fn is_user_source(source: &MessageSource) -> bool {
    matches!(source, MessageSource::User)
}
