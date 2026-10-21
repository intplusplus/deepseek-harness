//! Current-surface projection and byte-bounded reference rendering.
//!
//! Ported from `packages/context/session-reference/src/projection.ts`. The
//! TypeScript function receives a `SessionSurfaceSnapshot` that
//! `session-query` has already filtered to the model-visible surface. Rust has
//! no `session-query` surface operation yet, so [`ReferencedSessionSnapshot`]
//! carries the already surface-filtered events and this module owns the
//! projection rules: direct user text and compaction checkpoints survive,
//! assistant text survives, and reasoning, tools, system, and developer
//! material never do.

#![forbid(unsafe_code)]

use dsh_llm::{ContentBlock, CoreContentBlock};
use dsh_session::{CoreEvent, SessionEvent};

use super::serialization::stringify_tag_safe_json;
use super::types::{
    ReferenceRetentionStats, ReferencedConversationItem, ReferencedConversationRole,
    ReferencedSessionData, RetainedReference, SessionReferenceId,
};

/// The source observation this module projects.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferencedSessionSnapshot {
    /// Source session identity.
    pub session_id: SessionReferenceId,
    /// Source Session format generation.
    pub format_version: u32,
    /// Source session working directory, when recorded.
    pub cwd: Option<String>,
    /// Inclusive watermark of the captured event, or `None` before any event.
    pub captured_through_seq: Option<u64>,
    /// Surface-filtered events in source order.
    pub events: Vec<SessionEvent>,
}

/// One projected message plus retention bookkeeping that is not serialized.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ProjectedItem {
    role: ReferencedConversationRole,
    text: String,
    checkpoint: bool,
    original_text: String,
    omitted_bytes: u64,
}

impl ProjectedItem {
    /// A newly projected, untruncated message.
    fn new(role: ReferencedConversationRole, text: String, checkpoint: bool) -> Self {
        Self {
            role,
            original_text: text.clone(),
            text,
            checkpoint,
            omitted_bytes: 0,
        }
    }
}

/// Project current user/assistant conversation, excluding tools, reasoning,
/// and injected context.
pub fn project_conversation(events: &[SessionEvent]) -> Vec<ReferencedConversationItem> {
    project_items(events)
        .into_iter()
        .map(|item| ReferencedConversationItem {
            role: item.role,
            text: item.text,
        })
        .collect()
}

/// Project one surface snapshot into its retained messages.
fn project_items(events: &[SessionEvent]) -> Vec<ProjectedItem> {
    let mut conversation = Vec::new();
    for event in events {
        match event {
            SessionEvent::Core(CoreEvent::UserMessage { message }) => {
                let checkpoint = message.source.kind() == "compact-checkpoint";
                if !checkpoint && message.source.kind() != "user" {
                    continue;
                }
                let text = text_content(&message.content);
                if !text.is_empty() {
                    conversation.push(ProjectedItem::new(
                        ReferencedConversationRole::User,
                        text,
                        checkpoint,
                    ));
                }
            }
            SessionEvent::Core(CoreEvent::AssistantMessage { message, .. }) => {
                let text = text_content(&message.content);
                if !text.is_empty() {
                    conversation.push(ProjectedItem::new(
                        ReferencedConversationRole::Assistant,
                        text,
                        false,
                    ));
                }
            }
            _ => {}
        }
    }
    conversation
}

/// Join the plain text blocks of one message with a newline.
fn text_content(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Core(CoreContentBlock::Text { text }) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Fit one projected snapshot into an exact rendered JSON-object byte cap.
///
/// Whole messages are dropped first, oldest first and never the newest one or
/// a compaction checkpoint. If fixed data still exceeds the cap, the longest
/// retained message is replaced by a head/tail excerpt with an omission notice,
/// and the search repeats until the serialized preview fits.
///
/// Returns `None` when the fixed data (session id, label, cwd, watermark) or an
/// empty conversation cannot fit.
pub fn retain_referenced_session(
    snapshot: &ReferencedSessionSnapshot,
    label: &str,
    max_bytes: u64,
) -> Option<RetainedReference> {
    let original = project_items(&snapshot.events);
    let mut retained = original.clone();
    let mut omitted_messages = 0u64;
    let mut dropped_omitted_bytes = 0u64;

    let data = |retained: &[ProjectedItem]| ReferencedSessionData {
        session_id: snapshot.session_id.clone(),
        label: label.to_string(),
        cwd: snapshot.cwd.clone(),
        captured_through_seq: snapshot.captured_through_seq,
        conversation: retained
            .iter()
            .map(|item| ReferencedConversationItem {
                role: item.role,
                text: item.text.clone(),
            })
            .collect(),
    };
    let full_data = data(&retained);
    let size = |retained: &[ProjectedItem]| {
        stringify_tag_safe_json(&data(retained))
            .map(|serialized| serialized.len() as u64)
            .unwrap_or(u64::MAX)
    };

    while size(&retained) > max_bytes {
        let Some(newest_index) = retained.len().checked_sub(1) else {
            break;
        };
        let Some(drop_index) = retained
            .iter()
            .enumerate()
            .position(|(index, item)| !item.checkpoint && index != newest_index)
        else {
            break;
        };
        let removed = retained.remove(drop_index);
        omitted_messages += 1;
        dropped_omitted_bytes += removed.original_text.len() as u64;
    }

    while size(&retained) > max_bytes {
        let mut longest_index = None;
        let mut longest_bytes = 0u64;
        for (index, item) in retained.iter().enumerate() {
            let bytes = item.text.len() as u64;
            if bytes > longest_bytes {
                longest_bytes = bytes;
                longest_index = Some(index);
            }
        }
        let longest_index = longest_index?;
        if longest_bytes == 0 {
            return None;
        }
        let overflow = size(&retained).saturating_sub(max_bytes);
        let target = longest_bytes.saturating_sub(overflow);
        let item = retained.get(longest_index)?;
        let shortened = truncate_with_notice(&item.original_text, target);
        if shortened.text == item.text {
            return None;
        }
        retained[longest_index] = ProjectedItem {
            role: item.role,
            text: shortened.text,
            checkpoint: item.checkpoint,
            original_text: item.original_text.clone(),
            omitted_bytes: shortened.omitted_bytes,
        };
    }

    let compacted = original.iter().any(|item| item.checkpoint);
    let retained_omitted_bytes = retained
        .iter()
        .fold(0u64, |sum, item| sum.saturating_add(item.omitted_bytes));
    let omitted_bytes = retained_omitted_bytes.saturating_add(dropped_omitted_bytes);
    Some(RetainedReference {
        data: data(&retained),
        full_data,
        stats: ReferenceRetentionStats {
            compacted,
            original_messages: original.len() as u64,
            retained_messages: retained.len() as u64,
            omitted_messages,
            omitted_bytes,
            truncated: omitted_messages > 0 || omitted_bytes > 0,
        },
    })
}

/// A truncated message text and the exact number of omitted UTF-8 bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TruncatedText {
    /// Retained head and tail with the omission notice.
    pub text: String,
    /// UTF-8 bytes of the source that the excerpt does not carry.
    pub omitted_bytes: u64,
}

/// Truncate `text` to at most `max_output_bytes` using the head/tail rule.
///
/// The retained byte budget is split into `ceil(retained / 2)` head bytes and
/// `floor(retained / 2)` tail bytes. Both cuts are moved to UTF-8 boundaries,
/// and the appended notice counts the bytes actually dropped. A binary search
/// selects the largest retained budget whose excerpt and notice fit.
pub fn truncate_with_notice(text: &str, max_output_bytes: u64) -> TruncatedText {
    let total_bytes = text.len() as u64;
    if total_bytes <= max_output_bytes {
        return TruncatedText {
            text: text.to_string(),
            omitted_bytes: 0,
        };
    }
    let mut low = 0u64;
    let mut high = max_output_bytes;
    let mut best = TruncatedText {
        text: String::new(),
        omitted_bytes: total_bytes,
    };
    while low <= high {
        let retained_bytes = low + (high - low) / 2;
        let head_bytes = retained_bytes.div_ceil(2);
        let tail_bytes = retained_bytes / 2;
        let excerpt = head_tail(text, head_bytes, tail_bytes);
        let candidate = format!(
            "{}\n[… omitted {} UTF-8 bytes …]",
            excerpt.text, excerpt.omitted_bytes
        );
        if candidate.len() as u64 <= max_output_bytes {
            best = TruncatedText {
                text: candidate,
                omitted_bytes: excerpt.omitted_bytes,
            };
            low = retained_bytes + 1;
        } else if retained_bytes == 0 {
            break;
        } else {
            high = retained_bytes - 1;
        }
    }
    best
}

/// One head/tail excerpt before the notice is appended.
struct Excerpt {
    text: String,
    omitted_bytes: u64,
}

/// The `TextRetainer({ kind: 'headTail' })` result for one complete string.
fn head_tail(text: &str, head_bytes: u64, tail_bytes: u64) -> Excerpt {
    let bytes = text.as_bytes();
    let total = bytes.len() as u64;
    let prefix_len = total.min(head_bytes);
    let suffix_len = (total - prefix_len).min(tail_bytes);
    let prefix = &bytes[..prefix_len as usize];
    let suffix = &bytes[(bytes.len() - suffix_len as usize)..];
    let budget_omitted = total - prefix_len - suffix_len;
    if budget_omitted == 0 {
        let mut joined = Vec::with_capacity(bytes.len());
        joined.extend_from_slice(prefix);
        joined.extend_from_slice(suffix);
        return Excerpt {
            text: String::from_utf8_lossy(&joined).into_owned(),
            omitted_bytes: 0,
        };
    }
    let prefix = trim_trailing_partial_utf8(prefix);
    let suffix = trim_leading_continuation_utf8(suffix);
    let mut joined = Vec::with_capacity(prefix.len() + suffix.len());
    joined.extend_from_slice(prefix);
    joined.extend_from_slice(suffix);
    Excerpt {
        text: String::from_utf8_lossy(&joined).into_owned(),
        omitted_bytes: total - prefix.len() as u64 - suffix.len() as u64,
    }
}

/// Drop a trailing incomplete UTF-8 sequence from a prefix cut.
fn trim_trailing_partial_utf8(bytes: &[u8]) -> &[u8] {
    let mut index = bytes.len();
    while index > 0 && bytes[index - 1] & 0xc0 == 0x80 && bytes.len() - (index - 1) <= 3 {
        index -= 1;
    }
    if index == 0 {
        return bytes;
    }
    let lead = bytes[index - 1];
    let expected = if lead < 0x80 {
        1
    } else if lead < 0xe0 {
        2
    } else if lead < 0xf0 {
        3
    } else if lead < 0xf8 {
        4
    } else {
        0
    };
    if expected == 0 {
        return bytes;
    }
    if bytes.len() - (index - 1) < expected {
        &bytes[..index - 1]
    } else {
        bytes
    }
}

/// Drop leading continuation bytes from a suffix cut.
fn trim_leading_continuation_utf8(bytes: &[u8]) -> &[u8] {
    let mut index = 0;
    while index < bytes.len() && bytes[index] & 0xc0 == 0x80 {
        index += 1;
    }
    &bytes[index..]
}
