//! The canonical model-facing skill block.
//!
//! Both the `skill` tool result and user-explicit invocation injection render
//! through this module, so the model sees one `<skill_content>` shape on both
//! paths. The skill name rides an escaped attribute; the body is embedded
//! verbatim because skills are trusted local content and user-supplied
//! invocation text stays outside the wrapper.

use crate::types::{SkillDefinition, SkillResourceBase};

/// Render one loaded skill for the model.
///
/// The output is byte-identical to `renderSkillContent` in
/// `packages/skill/skill/src/index.ts`: a wrapper with an escaped `name`
/// attribute, a `<skill_resources>` hint, and the unmodified instruction body.
pub fn render_skill_content(skill: &SkillDefinition) -> String {
    let mut lines = vec![
        format!(
            "<skill_content name=\"{}\">",
            escape_attr(skill.summary.name.as_str())
        ),
        "<skill_resources>".to_string(),
    ];
    lines.extend(render_resource_hint(skill));
    lines.extend([
        "</skill_resources>".to_string(),
        String::new(),
        "<skill_instructions>".to_string(),
        skill.content.clone(),
        "</skill_instructions>".to_string(),
        "</skill_content>".to_string(),
    ]);
    lines.join("\n")
}

/// Escape model-facing prose embedded inside skill markup.
///
/// This prevents provider-supplied text from opening or closing framing tags.
/// The replacement order matters: `&` is escaped first so the ampersands
/// introduced by the other replacements are not doubled, exactly like the
/// TypeScript `replaceAll` chain.
pub fn escape_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Escape a value embedded in a double-quoted markup attribute.
fn escape_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
}

fn render_resource_hint(skill: &SkillDefinition) -> Vec<String> {
    let Some(base) = skill.summary.resource_base.as_ref() else {
        return vec![
            format!(
                "Resources for this skill are managed by provider \"{}\".",
                escape_text(&skill.summary.provider)
            ),
            "Load referenced resources only as needed.".to_string(),
        ];
    };
    match base {
        SkillResourceBase::Directory { path } => vec![
            format!("Base directory for this skill: {}", escape_text(path)),
            "Resolve relative paths mentioned by this skill against the base directory before using them. Load referenced resources only as needed."
                .to_string(),
        ],
        SkillResourceBase::Url { url } => vec![
            format!("Base URL for this skill: {}", escape_text(url)),
            "Resolve relative URLs mentioned by this skill against the base URL before using them. Load referenced resources only as needed."
                .to_string(),
        ],
        SkillResourceBase::Opaque { description } => vec![
            format!("Resources for this skill: {}", escape_text(description)),
            "Load referenced resources only as needed.".to_string(),
        ],
    }
}
