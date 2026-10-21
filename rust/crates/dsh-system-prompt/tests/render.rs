//! Ported from `packages/core/system-prompt/tests/system-prompt.spec.ts`
//! (the pure-function half).
//!
//! These assertions drive `renderPrompt` / `renderContextSections` /
//! `joinContextSections` with hand-built assemblies — no registry, no host.
//! The registry-dependent specs from the same file (waterfall
//! composition, change events, HMR disposal) need the Cordis plugin host
//! and are not ported; the assembly-level rules they pin that survive
//! without a host live in `assembly.rs`.

use dsh_system_prompt::*;
use serde_json::json;

fn section(name: &str, text: &str) -> AssembledSection {
    AssembledSection {
        name: name.to_string(),
        text: text.to_string(),
        interpolate: None,
    }
}

fn assembly(sections: Vec<AssembledSection>, variables: PromptVariables) -> PromptAssembly {
    PromptAssembly {
        sections,
        contexts: Vec::new(),
        tools: Vec::new(),
        variables,
    }
}

#[test]
fn filters_empty_sections_and_joins_the_rest_with_blank_lines() {
    let result = render_prompt(&assembly(
        vec![
            AssembledSection {
                name: "empty".to_string(),
                text: String::new(),
                interpolate: None,
            },
            section("real", "content"),
        ],
        PromptVariables::default(),
    ))
    .unwrap();
    assert_eq!(result, "content");
}

#[test]
fn interpolates_registered_variables_in_section_text() {
    let result = render_prompt(&assembly(
        vec![section("s", "on {{model}}")],
        [("model", Some("m"))].into_iter().collect(),
    ))
    .unwrap();
    assert_eq!(result, "on m");
}

#[test]
fn rejects_an_unknown_variable_and_lists_what_is_registered() {
    let err = render_prompt(&assembly(
        vec![section("persona", "on {{modle}}")],
        [("model", Some("m"))].into_iter().collect(),
    ))
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "unknown prompt variable \"{{modle}}\" in section \"persona\"; registered variables: model"
    );
}

#[test]
fn names_none_when_no_variables_are_registered_at_all() {
    let err = render_prompt(&assembly(
        vec![section("s", "{{x}}")],
        PromptVariables::default(),
    ))
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "unknown prompt variable \"{{x}}\" in section \"s\"; registered variables: (none)"
    );
}

#[test]
fn rejects_a_reference_whose_variable_has_no_value_for_this_assembly() {
    let err = render_prompt(&assembly(
        vec![section("persona", "in {{cwd}}")],
        [("cwd", None)].into_iter().collect(),
    ))
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "prompt variable \"{{cwd}}\" has no value for this assembly (section \"persona\")"
    );
}

#[test]
fn rejects_a_malformed_complete_reference_e_g_inner_spaces() {
    let err = render_prompt(&assembly(
        vec![section("s", "on {{ model }}")],
        [("model", Some("m"))].into_iter().collect(),
    ))
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "malformed prompt variable reference \"{{ model }}\" in section \"s\" (variable names match /^[a-z][a-z0-9_]*$/)"
    );
}

#[test]
fn leaves_a_lone_open_brace_verbatim_only_when_no_close_follows_anywhere() {
    let text = render_prompt(&assembly(
        vec![section("s", "shell ${X:-{{fallback} stays")],
        PromptVariables::default(),
    ))
    .unwrap();
    assert_eq!(text, "shell ${X:-{{fallback} stays");
}

#[test]
fn rejects_a_mangled_reference_while_a_close_still_follows() {
    for text in ["{{{model}}}", "x {{a{b}} y {{model}}"] {
        let err = render_prompt(&assembly(
            vec![section("s", text)],
            [("model", Some("m"))].into_iter().collect(),
        ))
        .unwrap_err();
        assert!(
            err.to_string()
                .starts_with("malformed prompt variable reference at"),
            "{text}: {err}"
        );
    }
}

#[test]
fn treats_a_prototype_property_reference_as_unknown() {
    // `in` would find Object.prototype.constructor and splice function
    // source into the prompt; own-property lookup must reject it instead.
    let err = render_prompt(&assembly(
        vec![section("s", "on {{constructor}}")],
        [("model", Some("m"))].into_iter().collect(),
    ))
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "unknown prompt variable \"{{constructor}}\" in section \"s\"; registered variables: model"
    );
}

#[test]
fn resolves_a_variable_named_like_a_prototype_property_once_registered() {
    let result = render_prompt(&assembly(
        vec![section("s", "on {{constructor}}")],
        [("constructor", Some("own-value"))].into_iter().collect(),
    ))
    .unwrap();
    assert_eq!(result, "on own-value");
}

#[test]
fn never_rescans_substituted_values() {
    let result = render_prompt(&assembly(
        vec![section("s", "v = {{model}}!")],
        [("model", Some("literal {{sneaky}} inside"))]
            .into_iter()
            .collect(),
    ))
    .unwrap();
    assert_eq!(result, "v = literal {{sneaky}} inside!");
}

#[test]
fn preserves_literal_context_in_both_the_sections_and_the_snapshot() {
    let literal = "/workspace/{{project}}/{{unregistered}}/{{malformed-name}}";
    let assembly = PromptAssembly {
        sections: Vec::new(),
        contexts: vec![
            AssembledContext {
                name: "literal".to_string(),
                text: literal.to_string(),
                interpolate: Some(false),
            },
            AssembledContext {
                name: "template".to_string(),
                text: "{{project}}".to_string(),
                interpolate: Some(true),
            },
            AssembledContext {
                name: "default".to_string(),
                text: "{{project}}".to_string(),
                interpolate: None,
            },
        ],
        tools: Vec::new(),
        variables: [("project", Some("expanded"))].into_iter().collect(),
    };
    assert_eq!(
        render_context_sections(&assembly).unwrap(),
        vec![
            ContextSnapshotSection {
                name: "literal".to_string(),
                text: literal.to_string(),
            },
            ContextSnapshotSection {
                name: "template".to_string(),
                text: "expanded".to_string(),
            },
            ContextSnapshotSection {
                name: "default".to_string(),
                text: "expanded".to_string(),
            },
        ]
    );
    assert_eq!(
        render_context_snapshot(&assembly).unwrap(),
        format!(
            "Current runtime context. This snapshot supersedes earlier runtime-context snapshots.\n\n{literal}\n\nexpanded\n\nexpanded"
        )
    );
}

#[test]
fn returns_an_empty_snapshot_without_active_context() {
    assert_eq!(
        render_context_snapshot(&PromptAssembly::default()).unwrap(),
        ""
    );
    assert_eq!(join_context_sections(&[]), "");
    let filtered = PromptAssembly {
        contexts: vec![AssembledContext {
            name: "empty".to_string(),
            text: String::new(),
            interpolate: None,
        }],
        ..PromptAssembly::default()
    };
    assert_eq!(render_context_snapshot(&filtered).unwrap(), "");
}

#[test]
fn attributes_context_interpolation_failures_to_the_contributing_context() {
    let assembly = PromptAssembly {
        sections: Vec::new(),
        contexts: vec![AssembledContext {
            name: "policy".to_string(),
            text: "Mode: {{missing}}.".to_string(),
            interpolate: None,
        }],
        tools: Vec::new(),
        variables: PromptVariables::default(),
    };
    let err = render_context_snapshot(&assembly).unwrap_err();
    assert_eq!(
        err.to_string(),
        "unknown prompt variable \"{{missing}}\" in context \"policy\"; registered variables: (none)"
    );
}

#[test]
fn section_placements_stay_unique_integral_and_at_least_ten_apart() {
    let orders: Vec<f64> = PromptSectionOrderName::ALL
        .iter()
        .map(|name| name.order())
        .collect();
    assert!(orders.iter().all(|order| order.fract() == 0.0));
    let mut sorted = orders.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted.dedup();
    assert_eq!(sorted.len(), orders.len());
    assert!(sorted.windows(2).all(|pair| pair[1] - pair[0] >= 10.0));
    assert_eq!(
        SystemPrompt::get_section_order(PromptSectionOrderName::ToolBash),
        1000.0
    );
}

#[test]
fn context_placements_stay_unique_and_integral() {
    let orders: Vec<f64> = PromptContextOrderName::ALL
        .iter()
        .map(|name| name.order())
        .collect();
    assert!(orders.iter().all(|order| order.fract() == 0.0));
    let mut sorted = orders.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted.dedup();
    assert_eq!(sorted.len(), orders.len());
    assert_eq!(
        SystemPrompt::get_context_order(PromptContextOrderName::WorkingDirectory),
        100.0
    );
}

#[test]
fn placement_names_match_the_typescript_table_keys() {
    let sections: Vec<&str> = PromptSectionOrderName::ALL
        .iter()
        .map(|name| name.name())
        .collect();
    assert_eq!(
        sections,
        [
            "HARNESS_IDENTITY",
            "DEPLOYMENT_PERSONA_PREFIX",
            "PLAN_POLICY",
            "TEAM_POLICY",
            "PTC_ONLY",
            "FILE_REFERENCE",
            "TOOL_BASH",
            "TOOL_PWSH",
            "TOOL_READ",
            "TOOL_WRITE",
            "TOOL_EDIT",
            "TOOL_GLOB",
            "TOOL_GREP",
            "TOOL_JOBS",
            "TOOL_PTY",
            "TOOL_WEB_SEARCH",
            "TOOL_WEB_FETCH",
            "TOOL_LSP",
            "TOOL_SESSION_QUERY",
            "TOOL_GOAL",
            "TOOL_WORKFLOW",
            "TOOL_RALPH",
            "TOOL_SUBAGENT",
            "TOOL_REPORT",
            "TOOL_COMPUTER_USE",
            "MCP_SERVERS",
            "TOOLS_SDK",
            "DELIVERABLE_FILE_REFERENCES",
            "STRUCTURED_OUTPUT",
            "HARNESS_SOURCE",
            "WEB_SURFACE",
            "DEPLOYMENT_PERSONA_SUFFIX",
        ]
    );
    let contexts: Vec<&str> = PromptContextOrderName::ALL
        .iter()
        .map(|name| name.name())
        .collect();
    assert_eq!(
        contexts,
        [
            "WORKING_DIRECTORY",
            "SANDBOX_POLICY",
            "APPROVAL_POLICY",
            "SUBAGENT_DELEGATION",
            "TEAM_IDENTITY",
            "HARNESS_SOURCE",
            "WEB_SURFACE",
        ]
    );
}

#[test]
fn assembly_round_trips_through_json_with_camel_case() {
    let assembly = PromptAssembly {
        sections: vec![AssembledSection {
            name: "s".to_string(),
            text: "t".to_string(),
            interpolate: Some(true),
        }],
        contexts: vec![AssembledContext {
            name: "c".to_string(),
            text: "x".to_string(),
            interpolate: None,
        }],
        tools: Vec::new(),
        variables: [("model", Some("m")), ("cwd", None)].into_iter().collect(),
    };
    let value = serde_json::to_value(&assembly).unwrap();
    let variables = value.get("variables").unwrap().as_object().unwrap();
    assert!(variables.contains_key("model"));
    // A registered-but-valueless variable serializes as absent, exactly as
    // JSON.stringify drops an `undefined` value.
    assert!(!variables.contains_key("cwd"));
    let restored: PromptAssembly = serde_json::from_value(value).unwrap();
    assert_eq!(restored.sections, assembly.sections);
    assert_eq!(restored.contexts, assembly.contexts);
    assert_eq!(restored.tools, assembly.tools);
    // The wire format cannot carry "registered but valueless": the round
    // trip leaves it unregistered, so a later reference to it reports an
    // unknown variable rather than a missing value.
    assert_eq!(restored.variables.get("model"), Some(Some("m")));
    assert_eq!(restored.variables.get("cwd"), None);

    // `knownNames` is the one camelCase field this package owns.
    let result = ToolProviderResult {
        schemas: Vec::new(),
        known_names: Some(vec!["a".to_string()]),
    };
    let encoded = serde_json::to_value(&result).unwrap();
    assert_eq!(encoded.get("knownNames").unwrap(), &json!(["a"]));
    assert_eq!(
        serde_json::from_value::<ToolProviderResult>(encoded).unwrap(),
        result
    );
}
