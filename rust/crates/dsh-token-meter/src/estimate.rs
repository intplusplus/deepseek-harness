//! The fixed-density token heuristic.
//!
//! Transcribed from `packages/llm/token-meter/src/estimate.ts`. The
//! heuristic prices content by its JSON length at a fixed character density,
//! and every priced message carries a role-framing overhead.
//!
//! # Why a heuristic
//!
//! The TypeScript original prices content *before* the exact tokenizer is
//! available: the meter has to answer "how big is this request" while the
//! request is still being assembled. The numbers are therefore an estimate,
//! and the estimate is deliberately crude — a fixed characters-per-token
//! density plus a fixed structural overhead per block.
//!
//! # Length unit
//!
//! The TypeScript original measures `JSON.stringify(x).length`, which counts
//! **UTF-16 code units**. Rust's `String::len()` counts **bytes**. The two
//! disagree on any non-ASCII text: a Chinese character is one UTF-16 code
//! unit but three UTF-8 bytes, and an emoji is two UTF-16 code units but
//! four UTF-8 bytes.
//!
//! This module counts **UTF-16 code units**, so a Rust estimate matches the
//! TypeScript estimate exactly on every input. That is the whole point of the
//! heuristic being comparable: the meter answers the same question on both
//! sides of the rewrite. See [`utf16_len`] for the exact rule.

use dsh_llm::{ContentBlock, CoreContentBlock, Message};
use dsh_session::EpochHeader;
use serde_json::Value;

/// Fixed text-density estimate used until exact tokenization is needed.
pub const CHARS_PER_TOKEN: usize = 4;

/// Per-block structural overhead for JSON framing and type tags.
pub const BLOCK_OVERHEAD: usize = 4;

/// Role-field framing overhead added to every priced message.
pub const ROLE_OVERHEAD: usize = 4;

/// The length of `text` in UTF-16 code units, matching the TypeScript
/// `String.length`.
///
/// Rust strings are UTF-8, so a character outside the Basic Multilingual
/// Plane occupies a surrogate pair in UTF-16 and counts as two units. The
/// count therefore is *not* `text.chars().count()` — that would report one
/// for an emoji where JavaScript reports two.
pub fn utf16_len(text: &str) -> usize {
    text.chars().map(char::len_utf16).sum()
}

/// The structural JSON price of one block outside the typed pricing arms: the
/// fixed heuristic for merge-extended blocks and for image references, whose
/// request price is route-owned rather than fixed.
///
/// Image offload marks do not change this reference-only heuristic; route
/// pricing owns their placeholders (TS `estimate.ts:29-35`).
pub fn estimate_structural_block(block: &ContentBlock) -> usize {
    if let ContentBlock::Core(CoreContentBlock::Image { .. }) = block {
        // The offloaded marker is excluded from the reference shape, exactly
        // as the TypeScript original strips it before measuring.
        let reference = without_offloaded(block);
        return BLOCK_OVERHEAD + ceil_div(utf16_len(&reference), CHARS_PER_TOKEN);
    }
    let encoded = serde_json::to_string(block).unwrap_or_default();
    BLOCK_OVERHEAD + ceil_div(utf16_len(&encoded), CHARS_PER_TOKEN)
}

/// The JSON shape of an image block without its `offloaded` marker.
fn without_offloaded(block: &ContentBlock) -> String {
    let mut value = serde_json::to_value(block).unwrap_or(Value::Null);
    if let Some(map) = value.as_object_mut() {
        map.remove("offloaded");
    }
    serde_json::to_string(&value).unwrap_or_default()
}

/// Price content blocks recursively under the fixed density heuristic
/// (TS `estimate.ts:42-63`).
///
/// A merge-extended block — one whose `type` the core does not own — keeps
/// the conservative structural JSON price, as do image references.
pub fn estimate_content(blocks: &[ContentBlock]) -> usize {
    let mut tokens = 0;
    for block in blocks {
        tokens += match block {
            ContentBlock::Core(
                CoreContentBlock::Text { text } | CoreContentBlock::Reasoning { text },
            ) => ceil_div(utf16_len(text), CHARS_PER_TOKEN) + BLOCK_OVERHEAD,
            ContentBlock::Core(CoreContentBlock::ToolCall {
                name, arguments, ..
            }) => {
                ceil_div(utf16_len(name), CHARS_PER_TOKEN)
                    + ceil_div(utf16_len(arguments), CHARS_PER_TOKEN)
                    + BLOCK_OVERHEAD
            }
            _ => estimate_structural_block(block),
        };
    }
    tokens
}

/// Price the rendered system prompt: the `system/message` surface node's text.
///
/// Adapters serialize the prompt as a plain string — a system-role message or
/// the request's system field — not as a typed content block, so the price is
/// text density plus role framing with no per-block overhead
/// (TS `estimate.ts:73-80`).
///
/// An empty system prompt prices at zero: "no system prompt".
pub fn estimate_system_message(message: &Message) -> usize {
    let blocks = message.content();
    if blocks.is_empty() {
        return 0;
    }
    let characters: usize = blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Core(CoreContentBlock::Text { text }) => utf16_len(text),
            other => utf16_len(&serde_json::to_string(other).unwrap_or_default()),
        })
        .sum();
    ceil_div(characters, CHARS_PER_TOKEN) + ROLE_OVERHEAD
}

/// Heuristically price one model-visible message (TS `estimate.ts:88-91`).
///
/// A system-role message prices as [`estimate_system_message`]; every other
/// message prices its content plus the role framing.
pub fn estimate_message(message: &Message) -> usize {
    if matches!(message, Message::System(_)) {
        return estimate_system_message(message);
    }
    estimate_content(message.content()) + ROLE_OVERHEAD
}

/// Price the tool-schema part of a canonical request envelope — the envelope's
/// only priced field, since the system prompt is a surface node
/// (TS `estimate.ts:99-102`).
///
/// An absent or empty tool list prices at zero.
pub fn estimate_tools_tokens(header: Option<&EpochHeader>) -> usize {
    let Some(header) = header else {
        return 0;
    };
    let Some(tools) = header.tools.as_ref() else {
        return 0;
    };
    if tools.is_empty() {
        return 0;
    }
    let encoded = serde_json::to_string(tools).unwrap_or_default();
    ceil_div(utf16_len(&encoded), CHARS_PER_TOKEN) + BLOCK_OVERHEAD
}

/// Ceiling division; the TypeScript original uses `Math.ceil`.
fn ceil_div(value: usize, divisor: usize) -> usize {
    value.div_ceil(divisor)
}
