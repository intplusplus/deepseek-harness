//! Rendering: strict `{{variable}}` interpolation and the runtime-context
//! snapshot join.
//!
//! These are the pure halves of the TypeScript module — `renderPrompt`,
//! `renderContextSections`, `joinContextSections`, `renderContextSnapshot`,
//! and the internal `interpolate` — with the assembly passed by reference
//! instead of assembled here. They can throw (an unknown or malformed
//! reference), so every function returns `Result`; TypeScript throws
//! synchronously from the same conditions.
//!
//! Two deliberate deviations, both unobservable to the ported tests:
//!
//! - The malformed-reference excerpt is the first 16 *characters* of the
//!   remaining text, where TypeScript slices 16 UTF-16 code units. They
//!   agree for every non-supplementary-plane input.
//! - The "registered variables" list is sorted (the variables map is a
//!   `BTreeMap`), where TypeScript lists `Object.keys` in insertion order.

#![forbid(unsafe_code)]

use super::error::{reference, InterpolationKind};
use super::types::{ContextSnapshotSection, PromptAssembly, PromptVariables};

/// Header line every non-empty runtime-context snapshot carries.
const SNAPSHOT_HEADER: &str =
    "Current runtime context. This snapshot supersedes earlier runtime-context snapshots.";

/// Interpolate strict `{{variable}}` references, drop empty sections, and
/// join the rest with blank lines.
///
/// Sections with `interpolate: Some(false)` retain literal text. Malformed,
/// unknown, or undefined references in other sections fail; a lone `{{`
/// without any later `}}` is literal prose, and substituted values are not
/// scanned again.
pub fn render_prompt(assembly: &PromptAssembly) -> Result<String, super::error::PromptError> {
    let mut parts: Vec<String> = Vec::with_capacity(assembly.sections.len());
    for section in &assembly.sections {
        let text = resolve_input(
            &section.text,
            &section.name,
            InterpolationKind::Section,
            section.interpolate,
            &assembly.variables,
        )?;
        if !text.is_empty() {
            parts.push(text);
        }
    }
    Ok(parts.join("\n\n"))
}

/// The same snapshot, kept as the named contributions it was assembled
/// from: one entry per contributing context that rendered to non-empty
/// text.
pub fn render_context_sections(
    assembly: &PromptAssembly,
) -> Result<Vec<ContextSnapshotSection>, super::error::PromptError> {
    let mut sections: Vec<ContextSnapshotSection> = Vec::with_capacity(assembly.contexts.len());
    for context in &assembly.contexts {
        let text = resolve_input(
            &context.text,
            &context.name,
            InterpolationKind::Context,
            context.interpolate,
            &assembly.variables,
        )?;
        if !text.is_empty() {
            sections.push(ContextSnapshotSection {
                name: context.name.clone(),
                text,
            });
        }
    }
    Ok(sections)
}

/// The model-facing snapshot text for an already-rendered section list.
pub fn join_context_sections(sections: &[ContextSnapshotSection]) -> String {
    let body = sections
        .iter()
        .map(|section| section.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    if body.is_empty() {
        return String::new();
    }
    format!("{SNAPSHOT_HEADER}\n\n{body}")
}

/// Render the complete dynamic context snapshot, preserving literal
/// contributions.
pub fn render_context_snapshot(
    assembly: &PromptAssembly,
) -> Result<String, super::error::PromptError> {
    Ok(join_context_sections(&render_context_sections(assembly)?))
}

/// Resolve one section or context text, honoring its interpolation flag.
fn resolve_input(
    text: &str,
    name: &str,
    kind: InterpolationKind,
    interpolate: Option<bool>,
    variables: &PromptVariables,
) -> Result<String, super::error::PromptError> {
    match interpolate {
        Some(false) => Ok(text.to_string()),
        _ => interpolate_text(text, name, kind, variables),
    }
}

/// Interpolate one section or context and attribute diagnostics to its
/// owning input.
fn interpolate_text(
    text: &str,
    name: &str,
    kind: InterpolationKind,
    variables: &PromptVariables,
) -> Result<String, super::error::PromptError> {
    let mut result = String::new();
    let mut last = 0;
    let mut open = match find_from(text, "{{", 0) {
        Some(open) => open,
        None => return Ok(result + &text[last..]),
    };
    loop {
        match match_group(text, open) {
            // A later closing brace makes this malformed; otherwise it is
            // literal prose.
            None => {
                if find_from(text, "}}", open + 2).is_some() {
                    let excerpt: String = text[open..].chars().take(16).collect();
                    return Err(super::error::PromptError::MalformedReferenceAt {
                        excerpt,
                        kind,
                        owner: name.to_string(),
                    });
                }
                result.push_str(&text[last..open + 2]);
                last = open + 2;
            }
            Some(end) => {
                // `{{}}` yields an empty name and follows the
                // malformed-reference path.
                let variable = &text[open + 2..end - 2];
                if !is_valid_variable_name(variable) {
                    return Err(super::error::PromptError::MalformedReference {
                        reference: reference(variable),
                        kind,
                        owner: name.to_string(),
                    });
                }
                // Unregistered names never resolve through a prototype.
                match variables.get(variable) {
                    None => {
                        return Err(super::error::PromptError::UnknownVariable {
                            reference: reference(variable),
                            kind,
                            owner: name.to_string(),
                            known: variables.known_list(),
                        });
                    }
                    Some(None) => {
                        return Err(super::error::PromptError::MissingVariableValue {
                            reference: reference(variable),
                            kind,
                            owner: name.to_string(),
                        });
                    }
                    Some(Some(value)) => {
                        result.push_str(&text[last..open]);
                        result.push_str(value);
                        last = end;
                    }
                }
            }
        }
        match find_from(text, "{{", last) {
            Some(next) => open = next,
            None => return Ok(result + &text[last..]),
        }
    }
}

/// Match a complete `{{...}}` reference group at `open`, returning the byte
/// index just past it — the Rust shape of TypeScript's `/^\{\{([^{}]*)\}\}/`.
fn match_group(text: &str, open: usize) -> Option<usize> {
    let rest = &text[open + 2..];
    let mut offset = 0;
    for c in rest.chars() {
        match c {
            // A brace inside the group disqualifies the match; the caller
            // decides whether what follows is malformed or literal prose.
            '{' => return None,
            '}' => {
                let after = &rest[offset + 1..];
                if after.starts_with('}') {
                    return Some(open + 2 + offset + 2);
                }
                return None;
            }
            _ => offset += c.len_utf8(),
        }
    }
    None
}

/// Byte index of the next `needle` at or after `from`, if any.
fn find_from(text: &str, needle: &str, from: usize) -> Option<usize> {
    text.get(from..)
        .and_then(|rest| rest.find(needle))
        .map(|index| from + index)
}

/// Valid variable names: how they are written between the braces —
/// `/^[a-z][a-z0-9_]*$/`.
pub(crate) fn is_valid_variable_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_lowercase() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}
