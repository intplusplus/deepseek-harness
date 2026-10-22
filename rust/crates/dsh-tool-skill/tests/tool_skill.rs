//! Regression locks for the catalog digest, entry normalization, the gesture
//! scan, and the `skill` loader.

use dsh_llm::{ContentBlock, MessageId, MessageSource, UserMessage};
use dsh_skill::{SkillDefinition, SkillInvocationPolicy, SkillName, SkillSource, SkillSummary};
use dsh_tool_skill::catalog::{
    catalog_description, catalog_history, catalog_message, read_catalog_entries,
    render_catalog_message, render_catalog_update,
};
use dsh_tool_skill::digest::{digest_catalog_entries, sha256_hex, CatalogEntry};
use dsh_tool_skill::gesture::{invoked_skill_names, scan_gestures};
use dsh_tool_skill::tool::{fn_skill_catalog, SkillTool, SkillToolError};
use dsh_tools::{ToolCallInput, ToolExecutionResult, ToolHandler};
use std::sync::Arc;

// --- SHA-256 --------------------------------------------------------------

#[test]
fn the_empty_digest_matches_the_nist_vector() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn the_abc_digest_matches_the_nist_vector() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn the_two_block_digest_matches_the_nist_vector() {
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

// --- description normalization -------------------------------------------

#[test]
fn whitespace_runs_collapse_to_single_spaces() {
    assert_eq!(catalog_description("  a   b\t\nc ", 100), "a b c");
}

#[test]
fn a_long_description_truncates_with_an_ellipsis() {
    let description = catalog_description("abcdefghij", 8);
    assert_eq!(description, "abcde...");
}

#[test]
fn the_truncation_counts_utf16_units_not_bytes() {
    // Six CJK characters are six UTF-16 units but eighteen UTF-8 bytes; a cap
    // of five units truncates on unit three, not byte three.
    let description = catalog_description("中文中文中文", 5);
    assert_eq!(description, "中文...");
}

// --- digest ---------------------------------------------------------------

#[test]
fn the_digest_is_stable_and_order_sensitive() {
    let one = vec![
        CatalogEntry {
            name: "alpha".into(),
            description: "First".into(),
        },
        CatalogEntry {
            name: "beta".into(),
            description: "Second".into(),
        },
    ];
    let mut two = one.clone();
    two.reverse();
    assert_eq!(digest_catalog_entries(&one), digest_catalog_entries(&one));
    assert_ne!(digest_catalog_entries(&one), digest_catalog_entries(&two));
}

#[test]
fn the_digest_quotes_entries_so_separators_stay_exact() {
    // A description containing a newline must not split the canonical line.
    let entries = vec![CatalogEntry {
        name: "alpha".into(),
        description: "first\nsecond".into(),
    }];
    let without = vec![CatalogEntry {
        name: "alpha".into(),
        description: "firstsecond".into(),
    }];
    assert_ne!(
        digest_catalog_entries(&entries),
        digest_catalog_entries(&without)
    );
}

// --- catalog rendering ----------------------------------------------------

#[test]
fn the_first_publication_uses_the_introductory_frame() {
    let entries = vec![CatalogEntry {
        name: "pdf-tools".into(),
        description: "Read PDFs <safely>".into(),
    }];
    let message = render_catalog_message(&entries);
    let text = message.content[0].as_text().unwrap();
    assert!(text.contains("The following skills are available in this session:"));
    assert!(text.contains("- `pdf-tools`: Read PDFs &lt;safely&gt;"));
    assert!(!text.contains("replaces every earlier"));
}

#[test]
fn a_replacement_announces_itself_and_the_empty_case() {
    let empty = render_catalog_update(&[]);
    let text = empty.content[0].as_text().unwrap();
    assert!(text.contains("replaces every earlier available-skills list"));
    assert!(text.contains("No skills are currently available"));

    let full = render_catalog_update(&[CatalogEntry {
        name: "beta".into(),
        description: "Second".into(),
    }]);
    let text = full.content[0].as_text().unwrap();
    assert!(text.contains("Use only names in this replacement catalog."));
}

#[test]
fn the_published_source_records_the_entries() {
    let message = render_catalog_message(&[CatalogEntry {
        name: "pdf-tools".into(),
        description: "Read PDFs".into(),
    }]);
    let MessageSource::Extension(source) = &message.source else {
        panic!("catalog source must be an extension");
    };
    let entries = read_catalog_entries(source).expect("readable");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "pdf-tools");
}

#[test]
fn an_unreadable_record_is_not_this_plugins_catalog() {
    let mut fields = serde_json::Map::new();
    fields.insert("kind".into(), serde_json::json!("skill-catalog"));
    fields.insert("entries".into(), serde_json::json!("not-an-array"));
    let source = dsh_llm::ExtensionSource::new("skill-catalog", fields);
    assert_eq!(read_catalog_entries(&source), None);
}

#[test]
fn the_first_usable_catalog_in_a_batch_wins() {
    let noise = UserMessage::with_source(
        MessageId::new("n1"),
        vec![ContentBlock::text("hello")],
        MessageSource::User,
    );
    let published = render_catalog_message(&[CatalogEntry {
        name: "alpha".into(),
        description: "First".into(),
    }]);
    let found = catalog_message(&[noise, published]).expect("found");
    assert_eq!(found.entries[0].name, "alpha");
}

// --- catalog history ------------------------------------------------------

fn published_event(name: &str) -> dsh_session::SessionEvent {
    let message = render_catalog_message(&[CatalogEntry {
        name: name.into(),
        description: "First".into(),
    }]);
    dsh_session::SessionEvent::Core(dsh_session::CoreEvent::UserMessage {
        message: Box::new(message),
    })
}

#[test]
fn the_history_finds_the_newest_visible_publication() {
    let events = vec![
        published_event("alpha"),
        dsh_session::SessionEvent::Core(dsh_session::CoreEvent::TurnStart { turn: 1 }),
        published_event("beta"),
    ];
    // The second publication was shadowed by a compaction; only the first is
    // visible.
    let visible = [0_u64].into_iter().collect();
    let history = catalog_history(&events, &visible);
    assert!(history.published);
    let digest = history.visible_digest.expect("visible");
    assert_eq!(
        digest,
        digest_catalog_entries(&[CatalogEntry {
            name: "alpha".into(),
            description: "First".into(),
        }])
    );
}

#[test]
fn a_history_without_publications_reports_neither() {
    let events = vec![dsh_session::SessionEvent::Core(
        dsh_session::CoreEvent::TurnStart { turn: 1 },
    )];
    let history = catalog_history(&events, &[].into_iter().collect());
    assert!(!history.published);
    assert_eq!(history.visible_digest, None);
}

// --- gesture scan ---------------------------------------------------------

#[test]
fn a_leading_gesture_matches_at_the_start() {
    assert_eq!(scan_gestures("/pdf-tools explain this"), vec!["pdf-tools"]);
}

#[test]
fn a_mid_sentence_gesture_matches_after_whitespace() {
    assert_eq!(
        scan_gestures("please use /pdf-tools here"),
        vec!["pdf-tools"]
    );
}

#[test]
fn file_paths_fractions_and_hyphen_tails_stay_prose() {
    assert!(scan_gestures("look under /usr/bin").is_empty());
    assert!(scan_gestures("the odds are 5/8").is_empty());
    assert!(scan_gestures("check /abc- now").is_empty());
    assert!(scan_gestures("a/b").is_empty());
}

#[test]
fn consecutive_gestures_both_match_and_dedupe_keeps_first_seen_order() {
    assert_eq!(scan_gestures("/alpha /beta"), vec!["alpha", "beta"]);
    let message = UserMessage::new(
        MessageId::new("m1"),
        vec![ContentBlock::text("/alpha then /beta then /alpha")],
    );
    assert_eq!(invoked_skill_names(&[message]), vec!["alpha", "beta"]);
}

#[test]
fn only_direct_user_input_can_forge_a_gesture() {
    let injected = UserMessage::with_source(
        MessageId::new("m2"),
        vec![ContentBlock::text("/injected")],
        dsh_llm::MessageSource::SystemPrompt,
    );
    assert!(invoked_skill_names(&[injected]).is_empty());
}

// --- the skill tool -------------------------------------------------------

fn summary(name: &str, model_invocable: bool) -> SkillSummary {
    SkillSummary {
        name: SkillName::new(name).unwrap(),
        path: None,
        description: "First".into(),
        when_to_use: None,
        invocation: SkillInvocationPolicy {
            model_invocable,
            user_invocable: true,
        },
        source: SkillSource::new(SkillSource::RUNTIME),
        provider: "runtime".into(),
        resource_base: None,
    }
}

fn definition(name: &str, model_invocable: bool) -> SkillDefinition {
    SkillDefinition::new(summary(name, model_invocable), "Do the thing step by step.")
}

fn tool(catalog: Arc<dyn dsh_tool_skill::SkillCatalog>) -> SkillTool {
    SkillTool::new(catalog)
}

fn call(name: &str) -> ToolCallInput {
    ToolCallInput {
        call_id: dsh_llm::ToolCallId::new("call-1"),
        name: "skill".into(),
        arguments: serde_json::json!({ "name": name }),
    }
}

#[test]
fn a_valid_load_renders_the_skill_block() {
    let handler = tool(Arc::new(fn_skill_catalog(|name| {
        Ok(if name == "pdf-tools" {
            Some(definition("pdf-tools", true))
        } else {
            None
        })
    })));
    let result = handler.call(&call("pdf-tools"), &dsh_tools::CancelSignal::new());
    let ToolExecutionResult::Success { value, content, .. } = &result else {
        panic!("expected success");
    };
    assert_eq!(value["name"], "pdf-tools");
    assert_eq!(value["provider"], "runtime");
    let text = content[0].as_text().unwrap();
    assert!(text.starts_with("<skill_content name=\"pdf-tools\">"));
    assert!(text.contains("Do the thing step by step."));
}

#[test]
fn an_invalid_name_fails_before_any_lookup() {
    let looked = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = Arc::clone(&looked);
    let handler = tool(Arc::new(fn_skill_catalog(move |_name| {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(None)
    })));
    let result = handler.call(&call("Not_Valid"), &dsh_tools::CancelSignal::new());
    assert!(result.is_error());
    let text = result.content()[0].as_text().unwrap();
    assert!(text.contains("invalid skill name \"Not_Valid\""));
    assert!(!looked.load(std::sync::atomic::Ordering::Relaxed));
}

#[test]
fn an_unknown_name_reports_availability() {
    let handler = tool(Arc::new(fn_skill_catalog(|_name| Ok(None))));
    let result = handler.call(&call("missing"), &dsh_tools::CancelSignal::new());
    let text = result.content()[0].as_text().unwrap();
    assert!(text.contains("is unknown or no longer available"));
}

#[test]
fn a_user_only_skill_refuses_model_invocation() {
    let handler = tool(Arc::new(fn_skill_catalog(|_name| {
        Ok(Some(definition("user-only", false)))
    })));
    let result = handler.call(&call("user-only"), &dsh_tools::CancelSignal::new());
    let text = result.content()[0].as_text().unwrap();
    assert!(text.contains("is not available for model invocation"));
}

#[test]
fn the_lookup_error_surfaces_as_a_failure() {
    let handler = tool(Arc::new(fn_skill_catalog(|_name| {
        Err(SkillToolError::Lookup("registry poisoned".into()))
    })));
    let result = handler.call(&call("pdf-tools"), &dsh_tools::CancelSignal::new());
    assert!(result.is_error());
    assert!(result.content()[0]
        .as_text()
        .unwrap()
        .contains("registry poisoned"));
}

#[test]
fn the_tool_schema_declares_the_required_name() {
    let handler = tool(Arc::new(fn_skill_catalog(|_name| Ok(None))));
    let schema = handler.schema();
    assert_eq!(schema.name, "skill");
    let properties = schema.parameters.get("properties").unwrap();
    assert!(properties.get("name").is_some());
    assert_eq!(
        schema.parameters.get("required").unwrap(),
        &serde_json::json!(["name"])
    );
}
