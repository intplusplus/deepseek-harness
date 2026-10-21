//! The `@path` completion grammar and the model guidance section.
//!
//! Ported from `packages/context/file-reference/src/grammar.ts` and the
//! prompt installation in `file-reference-local`. Discovery reads the
//! workspace and stays with the host; what is pinned here is the
//! browser-safe token grammar both clients share and the exact text the
//! model sees.

use dsh_context::file_reference::*;
use dsh_context::register_file_reference_section;
use dsh_system_prompt::{AssembleContext, Config, SystemPrompt};

fn token(line: &str, cursor: usize) -> ActiveAtToken {
    active_at_token(line, cursor).unwrap_or_else(|| panic!("no active token in {line:?}"))
}

fn file(path: &str) -> FileReferenceCandidate {
    FileReferenceCandidate {
        path: path.to_string(),
        kind: FileReferenceKind::File,
    }
}

fn directory(path: &str) -> FileReferenceCandidate {
    FileReferenceCandidate {
        path: path.to_string(),
        kind: FileReferenceKind::Directory,
    }
}

// --- the active token -------------------------------------------------------

#[test]
fn an_at_token_at_the_start_of_a_line_completes() {
    assert_eq!(
        token("@src", 4),
        ActiveAtToken {
            prefix: "@src".to_string(),
            query: "src".to_string(),
            quoted: false,
        }
    );
}

#[test]
fn an_at_token_after_whitespace_completes() {
    let active = token("read @src/ma", 12);
    assert_eq!(active.prefix, "@src/ma");
    assert_eq!(active.query, "src/ma");
    assert!(!active.quoted);
}

#[test]
fn an_at_inside_another_token_is_not_a_trigger() {
    // The TS grammar requires `^` or whitespace before the `@`, which is what
    // keeps an email address from opening completion.
    assert!(active_at_token("mail me@example.com", 18).is_none());
    assert!(active_at_token("a@b", 3).is_none());
}

#[test]
fn a_cursor_before_the_at_sees_no_token() {
    assert!(active_at_token("read @src", 4).is_none());
}

#[test]
fn whitespace_after_the_at_ends_the_token() {
    assert!(active_at_token("@src more", 9).is_none());
    assert!(active_at_token("@src more", 5).is_none());
}

#[test]
fn a_quoted_token_takes_precedence_over_the_plain_one() {
    // The TS grammar tests the quoted pattern first: an open quote means the
    // query may contain spaces.
    let active = token(r#"@"my fi"#, 7);
    assert!(active.quoted);
    assert_eq!(active.prefix, r#"@"my fi"#);
    assert_eq!(active.query, "my fi");
}

#[test]
fn a_closed_quote_falls_back_to_the_plain_token() {
    // `@"a"` has no *open* quote, so the quoted pattern cannot match and the
    // plain one carries the quote characters as ordinary path text.
    let active = token(r#"@"a""#, 4);
    assert!(!active.quoted);
    assert_eq!(active.query, r#""a""#);
}

#[test]
fn an_empty_at_token_still_completes_with_an_empty_query() {
    assert_eq!(
        token("@", 1),
        ActiveAtToken {
            prefix: "@".to_string(),
            query: String::new(),
            quoted: false,
        }
    );
}

#[test]
fn the_cursor_column_counts_scalar_values_not_bytes() {
    let line = "@中文";
    assert_eq!(token(line, 3).query, "中文");
}

// --- formatting a selected path --------------------------------------------

#[test]
fn a_plain_path_needs_no_quotes() {
    assert_eq!(
        format_file_mention(&file("src/main.rs"), false).unwrap(),
        "@src/main.rs"
    );
}

#[test]
fn a_directory_carries_a_trailing_slash() {
    assert_eq!(
        format_file_mention(&directory("src"), false).unwrap(),
        "@src/"
    );
}

#[test]
fn whitespace_forces_the_quoted_grammar() {
    assert_eq!(
        format_file_mention(&file("my file.txt"), false).unwrap(),
        r#"@"my file.txt""#
    );
}

#[test]
fn a_quoted_directory_keeps_its_quote_open_after_the_slash() {
    // Completion can descend another level, so the quote stays open.
    assert_eq!(
        format_file_mention(&directory("my dir"), false).unwrap(),
        r#"@"my dir/"#
    );
}

#[test]
fn preserve_quote_quotes_a_path_that_does_not_need_it() {
    assert_eq!(
        format_file_mention(&file("plain"), true).unwrap(),
        r#"@"plain""#
    );
    assert_eq!(
        format_file_mention(&directory("plain"), true).unwrap(),
        r#"@"plain/"#
    );
}

#[test]
fn a_path_the_grammar_cannot_represent_is_rejected() {
    // A `"` would end the quoted token early, and control characters cannot
    // be typed into a prompt safely.
    assert!(format_file_mention(&file(r#"a"b"#), false).is_none());
    assert!(format_file_mention(&file("a\nb"), false).is_none());
    assert!(format_file_mention(&file("a\u{7f}b"), false).is_none());
}

#[test]
fn a_whitespace_only_path_is_still_representable() {
    assert_eq!(format_file_mention(&file(" "), false).unwrap(), r#"@" ""#);
}

// --- the guidance section ---------------------------------------------------

#[test]
fn the_guidance_is_registered_under_the_reserved_section_name() {
    let sp = SystemPrompt::new(Config::default()).unwrap();
    let _section = register_file_reference_section(&sp, |_| true).unwrap();
    let assembly = sp.assemble(&AssembleContext::new()).unwrap();
    let names: Vec<&str> = assembly
        .sections
        .iter()
        .map(|section| section.name.as_str())
        .collect();
    assert!(
        names.contains(&FILE_REFERENCE_SECTION_NAME),
        "the guidance registers under its reserved name: {names:?}"
    );
}

#[test]
fn the_guidance_disappears_without_the_read_tool() {
    let sp = SystemPrompt::new(Config::default()).unwrap();
    let _section = register_file_reference_section(&sp, |context| {
        context
            .field("has_read")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    })
    .unwrap();

    let with = section_text(&sp, true);
    let without = section_text(&sp, false);
    assert_eq!(with, FILE_REFERENCE_PROMPT, "the guidance is byte-exact");
    assert!(
        without.is_empty(),
        "an empty section contributes nothing, so it drops out of the prompt"
    );
    assert!(
        !dsh_system_prompt::render_prompt(&sp.assemble(&AssembleContext::new()).unwrap())
            .unwrap()
            .contains("Tokens prefixed with @"),
        "without the read tool the guidance never reaches the model"
    );
}

/// The assembled text of the file-reference section alone.
fn section_text(sp: &SystemPrompt, has_read: bool) -> String {
    let context = AssembleContext::new().with_field("has_read", has_read);
    let assembly = sp.assemble(&context).unwrap();
    assembly
        .sections
        .iter()
        .find(|section| section.name == FILE_REFERENCE_SECTION_NAME)
        .map(|section| section.text.clone())
        .unwrap_or_default()
}

#[test]
fn registering_the_guidance_twice_fails() {
    let sp = SystemPrompt::new(Config::default()).unwrap();
    let _first = register_file_reference_section(&sp, |_| true).unwrap();
    assert!(
        register_file_reference_section(&sp, |_| true).is_err(),
        "section names are unique per layer"
    );
}
