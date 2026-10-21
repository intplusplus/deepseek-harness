//! Registration, assembly, and interpolation failures.
//!
//! Every message is a port of a TypeScript throw site. The vitest suite
//! asserts on these substrings, so the wording is part of the contract,
//! not decoration — keep them in sync when editing.

#![forbid(unsafe_code)]

use std::fmt;

use super::orders::TOOL_ORDER_REST;

/// A `{{name}}` reference exactly as it appears in prompt text.
///
/// Written character by character on purpose: brace escaping inside
/// `format!`/`thiserror` templates is easy to get wrong by sight.
pub(crate) fn reference(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    out.push('{');
    out.push('{');
    out.push_str(name);
    out.push('}');
    out.push('}');
    out
}

/// Errors raised by prompt registration, assembly, and rendering.
#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    #[error(
        "prompt section \"{name}\" is already registered (for a per-agent override, register through that agent's `agent.ctx` instead)"
    )]
    DuplicateSection { name: String },
    #[error("prompt section \"{name}\" is already registered in this scope")]
    DuplicateScopedSection { name: String },
    #[error(
        "prompt context \"{name}\" is already registered (for a per-agent override, register through that agent's `agent.ctx` instead)"
    )]
    DuplicateContext { name: String },
    #[error("prompt context \"{name}\" is already registered in this scope")]
    DuplicateScopedContext { name: String },
    #[error(
        "prompt variable \"{name}\" is already registered (for a per-agent value, register through that agent's `agent.ctx` instead)"
    )]
    DuplicateVariable { name: String },
    #[error("prompt variable \"{name}\" is already registered in this scope")]
    DuplicateScopedVariable { name: String },
    #[error("prompt section \"{name}\" order must be a finite number")]
    NonFiniteSectionOrder { name: String },
    #[error("prompt context \"{name}\" order must be a finite number")]
    NonFiniteContextOrder { name: String },
    #[error("invalid prompt variable name \"{name}\" (must match /^[a-z][a-z0-9_]*$/)")]
    InvalidVariableName { name: String },
    #[error("toolOrder lists \"{name}\" more than once")]
    DuplicateToolOrderEntry { name: String },
    #[error(
        "toolOrder must contain the \"{rest}\" rest entry (where unlisted tools are inserted)",
        rest = TOOL_ORDER_REST
    )]
    MissingRestEntry,
    #[error(
        "tool provider returned reserved tool name \"{rest}\" (reserved for toolOrder's rest entry)",
        rest = TOOL_ORDER_REST
    )]
    ReservedToolName,
    #[error("toolOrder lists unregistered tool{plural} {names}; known tools: {known}")]
    UnknownToolOrderNames {
        plural: &'static str,
        names: String,
        known: String,
    },
    #[error("multiple complete prompt sections are active: {names}")]
    MultipleCompleteSections { names: String },
    #[error(
        "malformed prompt variable reference at \"{excerpt}...\" in {kind} \"{owner}\" (references are complete simple {{{{name}}}} groups)"
    )]
    MalformedReferenceAt {
        excerpt: String,
        kind: InterpolationKind,
        owner: String,
    },
    #[error(
        "malformed prompt variable reference \"{reference}\" in {kind} \"{owner}\" (variable names match /^[a-z][a-z0-9_]*$/)"
    )]
    MalformedReference {
        reference: String,
        kind: InterpolationKind,
        owner: String,
    },
    #[error("unknown prompt variable \"{reference}\" in {kind} \"{owner}\"; registered variables: {known}")]
    UnknownVariable {
        reference: String,
        kind: InterpolationKind,
        owner: String,
        known: String,
    },
    #[error("prompt variable \"{reference}\" has no value for this assembly ({kind} \"{owner}\")")]
    MissingVariableValue {
        reference: String,
        kind: InterpolationKind,
        owner: String,
    },
}

/// The diagnostics' owning input: a prompt section or a runtime context.
///
/// TypeScript passes the literal `'section' | 'context'` string; here it is
/// a sum type so a call site cannot invent a third kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationKind {
    Section,
    Context,
}

impl InterpolationKind {
    /// The word as it appears in diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Section => "section",
            Self::Context => "context",
        }
    }
}

impl fmt::Display for InterpolationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
