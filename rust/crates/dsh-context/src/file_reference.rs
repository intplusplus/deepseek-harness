//! File-reference completion grammar and model guidance.
//!
//! Ported from `packages/context/file-reference` and the prompt installation
//! in `file-reference-local`. Discovery itself reads the workspace and stays
//! with the host; this module owns the browser-safe token grammar that both
//! terminal and web clients share, plus the exact model-facing guidance.

#![forbid(unsafe_code)]

use dsh_system_prompt::{
    AssembleContext, PromptError, PromptSection, PromptSectionOrderName, Registration, SystemPrompt,
};

use crate::js;

/// The section name used by the composed file-reference provider.
pub const FILE_REFERENCE_SECTION_NAME: &str = "context:file-reference";

/// Model guidance for path-only references selected by a user interface.
pub const FILE_REFERENCE_PROMPT: &str = "Tokens prefixed with @ are paths the user explicitly referenced. Relative paths resolve from the workspace root; absolute paths identify files or directories on the host. A trailing slash marks a directory: list it when its contents matter. Anything else is a file: use the read tool when its contents are needed, and do not claim to have inspected it before reading. @\"...\" quotes a path containing spaces.";

/// Whether a completion candidate is a file or a directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileReferenceKind {
    /// A file; accepting the candidate finishes the mention.
    File,
    /// A directory; accepting the candidate keeps completion open.
    Directory,
}

impl FileReferenceKind {
    /// The wire tag used by the TypeScript `FileReferenceCandidate`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Directory => "directory",
        }
    }
}

/// One path-only completion candidate inside the target session working
/// directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileReferenceCandidate {
    /// User-facing path accepted by normal prompts and filesystem tools.
    pub path: String,
    /// Whether the path names a file or a directory.
    pub kind: FileReferenceKind,
}

/// An active `@` token ending at the editor cursor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveAtToken {
    /// Complete token replaced when the user accepts a completion.
    pub prefix: String,
    /// Path query after `@` or `@"`.
    pub query: String,
    /// Whether the user opened a quoted path.
    pub quoted: bool,
}

/// Extract an `@path` or `@"path with spaces` token at the cursor.
///
/// The cursor column counts Unicode scalar values from the start of `line`.
/// An `@` inside another token, such as an email address, is not a completion
/// trigger. A quoted token takes precedence over the plain token that shares
/// its `@`, matching the two ordered patterns in the TypeScript grammar.
pub fn active_at_token(line: &str, cursor_col: usize) -> Option<ActiveAtToken> {
    let characters: Vec<char> = line.chars().take(cursor_col).collect();
    for quote_index in (0..characters.len()).rev() {
        if characters[quote_index] != '@' {
            continue;
        }
        if quote_index > 0 && !js::is_whitespace(characters[quote_index - 1]) {
            continue;
        }
        if characters.get(quote_index + 1) == Some(&'"')
            && !characters[quote_index + 2..].contains(&'"')
        {
            return Some(ActiveAtToken {
                prefix: characters[quote_index..].iter().collect(),
                query: characters[quote_index + 2..].iter().collect(),
                quoted: true,
            });
        }
        if characters[quote_index + 1..]
            .iter()
            .all(|character| !js::is_whitespace(*character))
        {
            return Some(ActiveAtToken {
                prefix: characters[quote_index..].iter().collect(),
                query: characters[quote_index + 1..].iter().collect(),
                quoted: false,
            });
        }
    }
    None
}

/// Format a selected path as prompt text.
///
/// Whitespace uses the quoted `@"path"` grammar; a quoted directory keeps its
/// quote open after the trailing slash so completion can descend another level.
/// Returns `None` for a path the editor grammar cannot represent safely.
pub fn format_file_mention(
    candidate: &FileReferenceCandidate,
    preserve_quote: bool,
) -> Option<String> {
    let path = match candidate.kind {
        FileReferenceKind::Directory => format!("{}/", candidate.path),
        FileReferenceKind::File => candidate.path.clone(),
    };
    if path.chars().any(|character| {
        matches!(
            character,
            '\u{0000}'..='\u{001f}' | '\u{007f}'..='\u{009f}' | '"'
        )
    }) {
        return None;
    }
    let quoted = preserve_quote || path.chars().any(js::is_whitespace);
    if !quoted {
        return Some(format!("@{path}"));
    }
    match candidate.kind {
        FileReferenceKind::Directory => Some(format!("@\"{path}")),
        FileReferenceKind::File => Some(format!("@\"{path}\"")),
    }
}

/// Register the file-reference guidance while the host can still offer the
/// read tool.
///
/// `available` is evaluated at each assembly, mirroring the TypeScript text
/// provider that returns an empty section once the agent has no `read` tool.
/// An empty section contributes nothing.
///
/// # Errors
///
/// Returns the registry's duplicate-name and non-finite-order diagnostics.
pub fn register_section<F>(prompt: &SystemPrompt, available: F) -> Result<Registration, PromptError>
where
    F: Fn(&AssembleContext) -> bool + 'static,
{
    prompt.section(PromptSection::provider(
        FILE_REFERENCE_SECTION_NAME,
        PromptSectionOrderName::FileReference.order(),
        move |context| {
            if available(context) {
                FILE_REFERENCE_PROMPT.to_string()
            } else {
                String::new()
            }
        },
    ))
}
