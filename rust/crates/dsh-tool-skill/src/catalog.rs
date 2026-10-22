//! The durable session skill catalog: entry normalization, the two rendered
//! model-facing frames, and the tolerant read of a published record.
//!
//! Transcribed from `packages/skill/tool-skill/src/index.ts`. The catalog is a
//! `catalog`-form context message, so it records the entries it published
//! beside the model-facing prose: a consumer presenting the list must not
//! re-parse the `<available_skills>` block, whose framing exists for the model.
//!
//! # Description length
//!
//! The TypeScript `catalogDescription` measures `normalized.length`, which
//! counts **UTF-16 code units**, and slices on the same unit. Rust counts
//! bytes, so this module measures and slices in UTF-16 code units to stay
//! byte-comparable with the TypeScript output on non-ASCII descriptions.

use dsh_llm::{ExtensionSource, MessageId, MessageSource, UserMessage};
use serde_json::{json, Map, Value};

use crate::digest::CatalogEntry;

/// The source kind every published catalog message carries.
pub const SKILL_CATALOG_SOURCE_KIND: &str = "skill-catalog";

/// The default maximum normalized description length.
pub const DEFAULT_CATALOG_DESCRIPTION_MAX_LENGTH: usize = 500;

/// The `<system-reminder>` frame of a first publication.
const CATALOG_OPENING: &str = "A skill is a reusable set of task-specific instructions. The following skills are available in this session:";
const CATALOG_CLOSING_FIRST: &str = "If the user names a skill, or the task clearly matches a skill's description, call the `skill` tool with the exact skill name before taking task actions. Load all applicable skills, then follow their full instructions. This catalog contains summaries only; do not infer or follow a skill's instructions until it has been loaded.\nA user may also invoke a skill directly; its <skill_content> block then appears in this conversation. Follow it, and do not call the `skill` tool again for that skill.";
/// The `<system-reminder>` frame of a replacement publication.
const UPDATE_OPENING: &str = "The available skill catalog changed. This complete catalog replaces every earlier available-skills list in this session:";
const UPDATE_CLOSING_EMPTY: &str = "No skills are currently available through the `skill` tool. Do not use names from earlier skill catalogs.\nA user may still invoke a skill directly; its <skill_content> block then appears in this conversation. Follow it, and do not call the `skill` tool for it.";
const UPDATE_CLOSING_NONEMPTY: &str = "Use only names in this replacement catalog. If the user names a listed skill, or the task clearly matches its description, call the `skill` tool with the exact name before acting.\nA user may also invoke a skill directly; its <skill_content> block then appears in this conversation. Follow it, and do not call the `skill` tool again for that skill.";

/// The length of `text` in UTF-16 code units.
fn utf16_len(text: &str) -> usize {
    text.chars().map(char::len_utf16).sum()
}

/// The first `units` UTF-16 code units of `text`.
fn utf16_slice(text: &str, units: usize) -> String {
    let mut taken = 0;
    let mut out = String::new();
    for character in text.chars() {
        let width = character.len_utf16();
        if taken + width > units {
            break;
        }
        out.push(character);
        taken += width;
    }
    out
}

/// Collapse whitespace runs and cap the rendered description length.
///
/// The pseudo-XML escaping belongs to the rendered frame, not to the published
/// fact, so the stored entry keeps the unescaped text.
pub fn catalog_description(value: &str, max_length: usize) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if utf16_len(&normalized) <= max_length {
        return normalized;
    }
    format!("{}...", utf16_slice(&normalized, max_length - 3))
}

/// The durable entry list mirroring the rendered catalog lines.
pub fn catalog_source_entries(
    skills: &[dsh_skill::SkillSummary],
    description_max_length: usize,
) -> Vec<CatalogEntry> {
    skills
        .iter()
        .map(|skill| CatalogEntry {
            name: skill.name.as_str().to_string(),
            description: catalog_description(&skill.description, description_max_length),
        })
        .collect()
}

/// The model-facing catalog lines, escaped for the frame.
fn render_catalog_entries(entries: &[CatalogEntry]) -> Vec<String> {
    entries
        .iter()
        .map(|entry| {
            format!(
                "- `{}`: {}",
                entry.name,
                dsh_skill::escape_text(&entry.description)
            )
        })
        .collect()
}

/// The verbatim frame around the entries: opening, entries, closing.
fn rendered_text(opening: &str, entries: &[CatalogEntry], closing: &str) -> String {
    let mut lines = vec![
        "<system-reminder>".to_string(),
        opening.to_string(),
        String::new(),
        "<available_skills>".to_string(),
    ];
    lines.extend(render_catalog_entries(entries));
    lines.extend([
        "</available_skills>".to_string(),
        String::new(),
        closing.to_string(),
        "</system-reminder>".to_string(),
    ]);
    lines.join("\n")
}

/// The `skill-catalog` source fields of a published message.
fn catalog_source(entries: &[CatalogEntry], update: bool) -> ExtensionSource {
    let mut fields = Map::new();
    fields.insert("kind".to_string(), json!(SKILL_CATALOG_SOURCE_KIND));
    fields.insert("form".to_string(), json!("catalog"));
    if update {
        fields.insert("update".to_string(), json!(true));
    }
    fields.insert(
        "entries".to_string(),
        Value::Array(
            entries
                .iter()
                .map(|entry| {
                    json!({
                        "name": entry.name,
                        "description": entry.description,
                    })
                })
                .collect(),
        ),
    );
    ExtensionSource::new(SKILL_CATALOG_SOURCE_KIND, fields)
}

/// Render this session's first catalog publication.
pub fn render_catalog_message(entries: &[CatalogEntry]) -> UserMessage {
    UserMessage::with_source(
        MessageId::new("skill-catalog"),
        vec![dsh_llm::ContentBlock::text(rendered_text(
            CATALOG_OPENING,
            entries,
            CATALOG_CLOSING_FIRST,
        ))],
        MessageSource::Extension(catalog_source(entries, false)),
    )
}

/// Render a replacement catalog publication.
pub fn render_catalog_update(entries: &[CatalogEntry]) -> UserMessage {
    let closing = if entries.is_empty() {
        UPDATE_CLOSING_EMPTY
    } else {
        UPDATE_CLOSING_NONEMPTY
    };
    UserMessage::with_source(
        MessageId::new("skill-catalog"),
        vec![dsh_llm::ContentBlock::text(rendered_text(
            UPDATE_OPENING,
            entries,
            closing,
        ))],
        MessageSource::Extension(catalog_source(entries, true)),
    )
}

/// The identity of a catalog message inside a step's claimed batch.
pub struct PublishedCatalog {
    /// The claimed message carrying the catalog source.
    pub message_id: String,
    /// Its recorded entries.
    pub entries: Vec<CatalogEntry>,
}

/// Read the entries of one durable catalog message, or `None` when the record
/// is not a usable catalog.
///
/// A seed may carry any source object with a non-empty kind, so an unreadable
/// record is treated as "not this plugin's catalog" rather than an error — the
/// posture the replaced content digest had. Throwing inside the step listener
/// would fail every subsequent turn of the session.
pub fn read_catalog_entries(source: &ExtensionSource) -> Option<Vec<CatalogEntry>> {
    if source.kind() != SKILL_CATALOG_SOURCE_KIND {
        return None;
    }
    let entries = source.fields().get("entries")?.as_array()?;
    let mut readable = Vec::with_capacity(entries.len());
    for entry in entries {
        let map = entry.as_object()?;
        let name = map.get("name")?.as_str()?;
        if name.is_empty() {
            return None;
        }
        let description = map.get("description")?.as_str()?;
        readable.push(CatalogEntry {
            name: name.to_string(),
            description: description.to_string(),
        });
    }
    Some(readable)
}

/// The identity of one published catalog message inside a claimed batch.
///
/// The first usable catalog wins, exactly like the TypeScript scan.
pub fn catalog_message(messages: &[UserMessage]) -> Option<PublishedCatalog> {
    for message in messages {
        let dsh_llm::MessageSource::Extension(source) = &message.source else {
            continue;
        };
        if let Some(entries) = read_catalog_entries(source) {
            return Some(PublishedCatalog {
                message_id: message.id.as_str().to_string(),
                entries,
            });
        }
    }
    None
}

/// The catalog history of a session: whether one was ever published, and the
/// digest of the newest publication still visible on the surface.
///
/// `visible` carries the current surface node seqs — the Rust core log does
/// not retain per-event surface membership, so the caller supplies it.
pub fn catalog_history(
    events: &[dsh_session::SessionEvent],
    visible: &std::collections::BTreeSet<u64>,
) -> CatalogHistory {
    let mut published = false;
    let mut index = events.len();
    while index > 0 {
        index -= 1;
        let event = &events[index];
        let dsh_session::SessionEvent::Core(dsh_session::CoreEvent::UserMessage { message }) =
            event
        else {
            continue;
        };
        let dsh_llm::MessageSource::Extension(source) = &message.source else {
            continue;
        };
        let Some(entries) = read_catalog_entries(source) else {
            continue;
        };
        published = true;
        if visible.contains(&(index as u64)) {
            return CatalogHistory {
                visible_digest: Some(digest_of(&entries)),
                published,
            };
        }
    }
    CatalogHistory {
        visible_digest: None,
        published,
    }
}

/// The digest helper shared with the history fold.
fn digest_of(entries: &[CatalogEntry]) -> String {
    crate::digest::digest_catalog_entries(entries)
}

/// Whether a catalog was published, and which publication is still visible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogHistory {
    /// The digest of the newest publication still on the surface.
    pub visible_digest: Option<String>,
    /// Whether any publication exists at all.
    pub published: bool,
}
