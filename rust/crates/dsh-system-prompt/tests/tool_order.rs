//! Ported from `packages/core/system-prompt/tests/tool-order.spec.ts`.
//!
//! Every case drives the registry with plain providers; the one spec that
//! needed a waterfall listener (`canonicalizes BEFORE the assemble
//! waterfall`) is not ported — the canonicalization it observes is pinned
//! by the ordering assertions here, and the listener-appended tool it
//! checks is waterfall behavior.

use dsh_llm::ToolSchema;
use dsh_system_prompt::*;
use serde_json::json;

fn tool(name: &str) -> ToolSchema {
    ToolSchema::new(name, name, json!({ "type": "object", "properties": {} }))
}

fn tool_with(name: &str, description: &str) -> ToolSchema {
    ToolSchema::new(
        name,
        description,
        json!({ "type": "object", "properties": {} }),
    )
}

fn mount(config: Config) -> SystemPrompt {
    SystemPrompt::new(config).unwrap()
}

fn tool_names(assembly: &PromptAssembly) -> Vec<&str> {
    assembly
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect()
}

#[test]
fn exports_the_rest_entry_as_unlisted_tools() {
    // The ONE place the public constant's value is pinned; everything else
    // references TOOL_ORDER_REST.
    assert_eq!(TOOL_ORDER_REST, "<unlisted-tools>");
}

#[test]
fn assembles_tools_in_lexicographic_name_order_when_no_tool_order_is_configured() {
    let sp = mount(Config::default());
    let _first = sp.tools(|_| ToolProviderResult {
        schemas: vec![tool("charlie"), tool("alpha")],
        known_names: None,
    });
    let _second = sp.tools(|_| ToolProviderResult {
        schemas: vec![tool("bravo")],
        known_names: None,
    });
    assert_eq!(
        tool_names(&sp.assemble(&AssembleContext::new()).unwrap()),
        ["alpha", "bravo", "charlie"]
    );
}

#[test]
fn assembles_the_same_order_regardless_of_provider_registration_order() {
    let forward = mount(Config::default());
    let _a = forward.tools(|_| ToolProviderResult {
        schemas: vec![tool("alpha")],
        known_names: None,
    });
    let _z = forward.tools(|_| ToolProviderResult {
        schemas: vec![tool("zulu")],
        known_names: None,
    });
    let backward = mount(Config::default());
    let _z = backward.tools(|_| ToolProviderResult {
        schemas: vec![tool("zulu")],
        known_names: None,
    });
    let _a = backward.tools(|_| ToolProviderResult {
        schemas: vec![tool("alpha")],
        known_names: None,
    });
    assert_eq!(
        tool_names(&forward.assemble(&AssembleContext::new()).unwrap()),
        ["alpha", "zulu"]
    );
    assert_eq!(
        tool_names(&backward.assemble(&AssembleContext::new()).unwrap()),
        ["alpha", "zulu"]
    );
}

#[test]
fn applies_a_configured_tool_order_with_the_rest_entry_lexicographic() {
    let sp = mount(Config {
        tool_order: Some(vec![
            "todo_write".to_string(),
            TOOL_ORDER_REST.to_string(),
            "bash".to_string(),
        ]),
        ..Config::default()
    });
    let _tools = sp.tools(|_| ToolProviderResult {
        schemas: vec![
            tool("bash"),
            tool("echo_b"),
            tool("todo_write"),
            tool("echo_a"),
        ],
        known_names: None,
    });
    assert_eq!(
        tool_names(&sp.assemble(&AssembleContext::new()).unwrap()),
        ["todo_write", "echo_a", "echo_b", "bash"]
    );
}

#[test]
fn rejects_the_assembly_when_tool_order_names_an_unregistered_tool() {
    let sp = mount(Config {
        tool_order: Some(vec![
            "todo_write".to_string(),
            "ghost".to_string(),
            TOOL_ORDER_REST.to_string(),
            "wraith".to_string(),
        ]),
        ..Config::default()
    });
    let _tools = sp.tools(|_| ToolProviderResult {
        schemas: vec![tool("bash"), tool("todo_write")],
        known_names: None,
    });
    let err = sp.assemble(&AssembleContext::new()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "toolOrder lists unregistered tools \"ghost\", \"wraith\"; known tools: bash, todo_write"
    );
}

#[test]
fn names_the_single_unregistered_tool_when_no_tools_are_registered_at_all() {
    let sp = mount(Config {
        tool_order: Some(vec!["ghost".to_string(), TOOL_ORDER_REST.to_string()]),
        ..Config::default()
    });
    let err = sp.assemble(&AssembleContext::new()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "toolOrder lists unregistered tool \"ghost\"; known tools: (none)"
    );
}

#[test]
fn rejects_a_provider_tool_named_like_the_reserved_rest_entry() {
    for tool_order in [None, Some(vec![TOOL_ORDER_REST.to_string()])] {
        let sp = mount(Config {
            tool_order,
            ..Config::default()
        });
        let _tools = sp.tools(|_| ToolProviderResult {
            schemas: vec![tool(TOOL_ORDER_REST)],
            known_names: None,
        });
        let err = sp.assemble(&AssembleContext::new()).unwrap_err();
        assert_eq!(
            err.to_string(),
            "tool provider returned reserved tool name \"<unlisted-tools>\" (reserved for toolOrder's rest entry)"
        );
    }
}

#[test]
fn keeps_collection_order_between_tools_that_share_a_name() {
    let sp = mount(Config::default());
    let _tools = sp.tools(|_| ToolProviderResult {
        schemas: vec![
            tool_with("dup", "first"),
            tool("anchor"),
            tool_with("dup", "second"),
        ],
        known_names: None,
    });
    let assembly = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(
        assembly
            .tools
            .iter()
            .map(|tool| tool.description.as_str())
            .collect::<Vec<_>>(),
        ["anchor", "first", "second"]
    );
}

#[test]
fn rejects_a_tool_order_without_the_rest_entry_at_load() {
    for tool_order in [
        Vec::new(),
        vec!["bash".to_string(), "todo_write".to_string()],
    ] {
        let err = SystemPrompt::new(Config {
            tool_order: Some(tool_order),
            ..Config::default()
        })
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("must contain the \"<unlisted-tools>\" rest entry"),
            "{err}"
        );
    }
}

#[test]
fn rejects_duplicate_tool_order_entries_at_load() {
    for tool_order in [
        vec![
            "bash".to_string(),
            "bash".to_string(),
            TOOL_ORDER_REST.to_string(),
        ],
        vec![
            TOOL_ORDER_REST.to_string(),
            "bash".to_string(),
            TOOL_ORDER_REST.to_string(),
        ],
    ] {
        let err = SystemPrompt::new(Config {
            tool_order: Some(tool_order),
            ..Config::default()
        })
        .unwrap_err();
        assert!(err.to_string().contains("more than once"), "{err}");
    }
}

#[test]
fn treats_a_restricted_tool_order_entry_as_absence_but_a_typo_still_throws() {
    let sp = mount(Config {
        tool_order: Some(vec!["bash".to_string(), TOOL_ORDER_REST.to_string()]),
        ..Config::default()
    });
    // A provider mimicking the registry's restriction split: bash exists
    // (knownNames) but is masked for this assembly (schemas).
    let _tools = sp.tools(|_| ToolProviderResult {
        schemas: vec![tool("read")],
        known_names: Some(vec!["read".to_string(), "bash".to_string()]),
    });
    assert_eq!(
        tool_names(&sp.assemble(&AssembleContext::new()).unwrap()),
        ["read"]
    );

    let bad = mount(Config {
        tool_order: Some(vec!["basj".to_string(), TOOL_ORDER_REST.to_string()]),
        ..Config::default()
    });
    let _tools = bad.tools(|_| ToolProviderResult {
        schemas: vec![tool("read")],
        known_names: Some(vec!["read".to_string(), "bash".to_string()]),
    });
    let err = bad.assemble(&AssembleContext::new()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "toolOrder lists unregistered tool \"basj\"; known tools: bash, read"
    );
}
