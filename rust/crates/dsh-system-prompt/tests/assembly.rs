//! Ported from `packages/core/system-prompt/tests/system-prompt.spec.ts`
//! (the registry half).
//!
//! Every assembly carries the plugin's own built-ins —
//! `harness:identity` and `deployment:persona-prefix` /
//! `deployment:persona-suffix` (from config). Registry-mechanics tests
//! strip them with [`contributed`] to stay focused on their own sections;
//! the built-ins' behavior is pinned by its own tests.
//!
//! Not ported from the TypeScript file, because each needs the Cordis
//! plugin host: `system-prompt/change` emission and the P1-1 rollback,
//! the `system-prompt/assemble` waterfall (composition, short-circuit,
//! listener-added contexts and variables), scope-filtered dispatch, and
//! whole-fiber HMR disposal. The one self-replacing scoped variable
//! (`defers a scoped variable that replaces the last provider`) needs
//! re-entrant registration through a live borrow; the live-iteration rule
//! it relies on is ported and tested by
//! [`live_iterates_variables_registered_by_an_earlier_provider`].

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use dsh_llm::ToolSchema;
use dsh_system_prompt::*;
use serde_json::{json, Value};

const BUILT_IN: [&str; 3] = [
    "harness:identity",
    "deployment:persona-prefix",
    "deployment:persona-suffix",
];

fn mount(config: Config) -> SystemPrompt {
    SystemPrompt::new(config).unwrap()
}

fn tool(name: &str) -> ToolSchema {
    ToolSchema::new(name, name, json!({ "type": "object", "properties": {} }))
}

fn section_names(assembly: &PromptAssembly) -> Vec<&str> {
    assembly
        .sections
        .iter()
        .map(|section| section.name.as_str())
        .collect()
}

fn contributed(assembly: &PromptAssembly) -> Vec<&AssembledSection> {
    assembly
        .sections
        .iter()
        .filter(|section| !BUILT_IN.contains(&section.name.as_str()))
        .collect()
}

fn contributed_names(assembly: &PromptAssembly) -> Vec<&str> {
    contributed(assembly)
        .iter()
        .map(|section| section.name.as_str())
        .collect()
}

fn tool_names(assembly: &PromptAssembly) -> Vec<&str> {
    assembly
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect()
}

#[test]
fn assembles_sections_in_order_with_context_resolved_text_and_collected_tools() {
    let sp = mount(Config {
        persona_prefix: Some("You are DeepSeek Harness.".to_string()),
        ..Config::default()
    });
    let _cwd = sp
        .section(PromptSection::provider("cwd", 20.0, |_| {
            "cwd: /tmp".to_string()
        }))
        .unwrap();
    let _rules = sp
        .section(PromptSection::static_text("rules", 10.0, "Be precise."))
        .unwrap();
    let _later = sp
        .context(PromptContext::provider("later", 20.0, |_| {
            "context 2".to_string()
        }))
        .unwrap();
    let _earlier = sp
        .context(PromptContext::static_text("earlier", 10.0, "context 1"))
        .unwrap();
    let _tools = sp.tools(|_| ToolProviderResult {
        schemas: vec![tool("echo")],
        known_names: None,
    });

    let assembly = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(
        section_names(&assembly),
        [
            "harness:identity",
            "deployment:persona-prefix",
            "rules",
            "cwd",
            "deployment:persona-suffix",
        ]
    );
    assert_eq!(
        assembly
            .sections
            .iter()
            .map(|section| section.text.as_str())
            .collect::<Vec<_>>(),
        [
            HARNESS_IDENTITY_TEXT,
            "You are DeepSeek Harness.",
            "Be precise.",
            "cwd: /tmp",
            "",
        ]
    );
    assert_eq!(
        assembly.contexts,
        vec![
            AssembledContext {
                name: "earlier".to_string(),
                text: "context 1".to_string(),
                interpolate: None,
            },
            AssembledContext {
                name: "later".to_string(),
                text: "context 2".to_string(),
                interpolate: None,
            },
        ]
    );
    assert_eq!(tool_names(&assembly), ["echo"]);
    assert_eq!(assembly.variables, PromptVariables::default());
    assert_eq!(
        render_prompt(&assembly).unwrap(),
        format!("{HARNESS_IDENTITY_TEXT}\n\nYou are DeepSeek Harness.\n\nBe precise.\n\ncwd: /tmp")
    );
    assert_eq!(
        render_context_snapshot(&assembly).unwrap(),
        "Current runtime context. This snapshot supersedes earlier runtime-context snapshots.\n\ncontext 1\n\ncontext 2"
    );
}

#[test]
fn breaks_equal_section_orders_by_code_unit_name_regardless_of_registration_order() {
    for names in [["äther", "zeta"], ["zeta", "äther"]] {
        let sp = mount(Config::default());
        let _sections: Vec<Registration> = names
            .iter()
            .map(|name| {
                sp.section(PromptSection::static_text(*name, 10.0, *name))
                    .unwrap()
            })
            .collect();
        let assembly = sp.assemble(&AssembleContext::new()).unwrap();
        assert_eq!(contributed_names(&assembly), ["zeta", "äther"]);
    }
}

#[test]
fn resolves_section_text_providers_against_the_assemble_context_at_each_assemble() {
    let sp = mount(Config::default());
    let calls = Rc::new(Cell::new(0usize));
    let counted = Rc::clone(&calls);
    let _dynamic = sp
        .section(PromptSection::provider("dynamic", 0.0, move |context| {
            counted.set(counted.get() + 1);
            let who = context
                .field("who")
                .and_then(Value::as_str)
                .unwrap_or("nobody");
            format!("call {} for {who}", counted.get())
        }))
        .unwrap();

    let with_who = AssembleContext::new().with_field("who", "alice");
    assert_eq!(
        contributed(&sp.assemble(&with_who).unwrap())[0].text,
        "call 1 for alice"
    );
    assert_eq!(
        contributed(&sp.assemble(&AssembleContext::new()).unwrap())[0].text,
        "call 2 for nobody"
    );
}

#[test]
fn rejects_duplicate_and_non_finite_registrations_without_leaking() {
    let sp = mount(Config::default());
    let _dup = sp
        .section(PromptSection::static_text("dup", 0.0, "first"))
        .unwrap();
    let err = sp
        .section(PromptSection::static_text("dup", 1.0, "second"))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("prompt section \"dup\" is already registered"),
        "{err}"
    );
    let err = sp
        .section(PromptSection::static_text("bad-order", f64::NAN, "x"))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("prompt section \"bad-order\" order must be a finite number"),
        "{err}"
    );
    // The failed registrations leaked nothing; the original stays intact.
    assert_eq!(
        contributed(&sp.assemble(&AssembleContext::new()).unwrap())
            .iter()
            .map(|section| section.text.as_str())
            .collect::<Vec<_>>(),
        ["first"]
    );

    let _policy = sp
        .context(PromptContext::static_text("policy", 1.0, "first"))
        .unwrap();
    let err = sp
        .context(PromptContext::static_text("policy", 2.0, "second"))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("prompt context \"policy\" is already registered"),
        "{err}"
    );
    let err = sp
        .context(PromptContext::static_text("bad", f64::NAN, "x"))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("prompt context \"bad\" order must be a finite number"),
        "{err}"
    );
    assert_eq!(
        sp.assemble(&AssembleContext::new()).unwrap().contexts.len(),
        1
    );

    let _model = sp.variable("model", |_| Some("m1".to_string())).unwrap();
    let err = sp
        .variable("model", |_| Some("m2".to_string()))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("prompt variable \"model\" is already registered"),
        "{err}"
    );
    let err = sp.variable("Not Valid", |_| None).unwrap_err();
    assert!(
        err.to_string()
            .contains("invalid prompt variable name \"Not Valid\""),
        "{err}"
    );
    assert_eq!(
        sp.assemble(&AssembleContext::new())
            .unwrap()
            .variables
            .len(),
        1
    );
}

#[test]
fn rejects_multiple_effective_complete_sections() {
    let sp = mount(Config::default());
    let _first = sp
        .section(PromptSection::static_text("first", 10.0, "first").with_complete(true))
        .unwrap();
    let _second = sp
        .section(PromptSection::static_text("second", 20.0, "second").with_complete(true))
        .unwrap();
    let err = sp.assemble(&AssembleContext::new()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "multiple complete prompt sections are active: \"first\", \"second\""
    );
}

#[test]
fn keeps_one_complete_section_as_the_sole_prompt() {
    let sp = mount(Config::default());
    let _complete = sp
        .section(PromptSection::static_text("complete", 10.0, "Exact prompt.").with_complete(true))
        .unwrap();
    let _extra = sp
        .section(PromptSection::static_text("extra", 20.0, "extra"))
        .unwrap();
    let assembly = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(
        assembly.sections,
        vec![AssembledSection {
            name: "complete".to_string(),
            text: "Exact prompt.".to_string(),
            interpolate: None,
        }]
    );
    assert_eq!(render_prompt(&assembly).unwrap(), "Exact prompt.");
}

#[test]
fn does_not_leak_one_step_mutations_into_later_assemblies() {
    let sp = mount(Config::default());
    let _base = sp
        .section(PromptSection::static_text("base", 10.0, "base"))
        .unwrap();
    let _tools = sp.tools(|_| ToolProviderResult {
        schemas: vec![ToolSchema::new(
            "t",
            "tool",
            json!({ "type": "object", "properties": {} }),
        )],
        known_names: None,
    });

    let mut first = sp.assemble(&AssembleContext::new()).unwrap();
    first.sections[0].name = "mutated".to_string();
    first.sections[0].text = "mutated".to_string();
    first.contexts.push(AssembledContext {
        name: "mutated".to_string(),
        text: "mutated".to_string(),
        interpolate: None,
    });
    first.tools[0].description = "mutated".to_string();
    first.tools[0]
        .parameters
        .insert("leak".to_string(), json!({ "type": "string" }));

    let second = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(
        section_names(&second),
        [
            "harness:identity",
            "deployment:persona-prefix",
            "base",
            "deployment:persona-suffix",
        ]
    );
    assert_eq!(second.sections[0].text, HARNESS_IDENTITY_TEXT);
    assert!(second.contexts.is_empty());
    assert_eq!(second.tools[0].description, "tool");
    assert!(!second.tools[0].parameters.contains_key("leak"));
}

#[test]
fn registers_the_harness_identity_and_the_configured_deployment_persona() {
    let sp = mount(Config {
        persona_prefix: Some("You are DeepSeek Harness.".to_string()),
        ..Config::default()
    });
    let assembly = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(
        section_names(&assembly),
        [
            "harness:identity",
            "deployment:persona-prefix",
            "deployment:persona-suffix",
        ]
    );
    assert_eq!(
        render_prompt(&assembly).unwrap(),
        format!("{HARNESS_IDENTITY_TEXT}\n\nYou are DeepSeek Harness.")
    );
    // The names are reserved by the plugin — one owner per section.
    let err = sp
        .section(PromptSection::static_text(
            PERSONA_PREFIX_SECTION,
            0.0,
            "imposter",
        ))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("prompt section \"deployment:persona-prefix\" is already registered"),
        "{err}"
    );
}

#[test]
fn renders_no_persona_section_for_a_persona_less_deployment() {
    let sp = mount(Config::default());
    assert_eq!(
        render_prompt(&sp.assemble(&AssembleContext::new()).unwrap()).unwrap(),
        HARNESS_IDENTITY_TEXT
    );
}

#[test]
fn can_omit_the_harness_identity_for_a_deployment_that_owns_the_persona() {
    let sp = mount(Config {
        include_harness_identity: Some(false),
        persona_prefix: Some("You are a helpful software engineer assistant.".to_string()),
        ..Config::default()
    });
    let assembly = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(
        section_names(&assembly),
        ["deployment:persona-prefix", "deployment:persona-suffix"]
    );
    assert_eq!(
        render_prompt(&assembly).unwrap(),
        "You are a helpful software engineer assistant."
    );
}

#[test]
fn renders_the_environment_after_guidance_and_reports_strict_interpolation_errors() {
    let sp = mount(Config {
        persona_prefix: Some("Model {{model}}.".to_string()),
        persona_suffix: Some("Workspace {{cwd}}.".to_string()),
        ..Config::default()
    });
    let _model = sp.variable("model", |_| Some("m".to_string())).unwrap();
    let _guidance = sp
        .section(PromptSection::static_text("guidance", 100.0, "Use tools."))
        .unwrap();

    let unresolved = sp.assemble(&AssembleContext::new()).unwrap();
    let err = render_prompt(&unresolved).unwrap_err();
    // The TypeScript test matches this message with `toThrow`, which is a
    // substring check; the full message also lists registered variables.
    assert_eq!(
        err.to_string(),
        "unknown prompt variable \"{{cwd}}\" in section \"deployment:persona-suffix\"; registered variables: model"
    );

    let _cwd = sp.variable("cwd", |_| Some("/work".to_string())).unwrap();
    assert_eq!(
        render_prompt(&sp.assemble(&AssembleContext::new()).unwrap()).unwrap(),
        format!("{HARNESS_IDENTITY_TEXT}\n\nModel m.\n\nUse tools.\n\nWorkspace /work.")
    );
}

#[test]
fn keeps_reusable_instructions_identical_across_local_environments() {
    struct Environment {
        model: String,
        cwd: String,
        platform: String,
        source: String,
        url: String,
    }
    let sp = mount(Config {
        persona_prefix: Some("Model {{model}}.".to_string()),
        persona_suffix: Some("In {{cwd}} on {{platform}}.".to_string()),
        ..Config::default()
    });
    let environment = Rc::new(RefCell::new(Environment {
        model: "model-a".to_string(),
        cwd: "/alice/project".to_string(),
        platform: "darwin".to_string(),
        source: "/alice/dsh".to_string(),
        url: "http://127.0.0.1:3080".to_string(),
    }));
    let mut guards: Vec<Registration> = Vec::new();
    for key in ["model", "cwd", "platform"] {
        let environment = Rc::clone(&environment);
        guards.push(
            sp.variable(key, move |_| {
                let environment = environment.borrow();
                Some(match key {
                    "model" => environment.model.clone(),
                    "cwd" => environment.cwd.clone(),
                    _ => environment.platform.clone(),
                })
            })
            .unwrap(),
        );
    }
    // Local paths and endpoints follow the reusable instructions, so they
    // get their own placements below the guidance.
    let reusable: Vec<PromptSectionOrderName> = PromptSectionOrderName::ALL
        .iter()
        .copied()
        .filter(|name| {
            !matches!(
                name,
                PromptSectionOrderName::HarnessIdentity
                    | PromptSectionOrderName::DeploymentPersonaPrefix
                    | PromptSectionOrderName::HarnessSource
                    | PromptSectionOrderName::WebSurface
                    | PromptSectionOrderName::DeploymentPersonaSuffix
            )
        })
        .collect();
    for name in reusable.iter().rev() {
        guards.push(
            sp.section(PromptSection::static_text(
                name.name(),
                name.order(),
                name.name(),
            ))
            .unwrap(),
        );
    }
    let source = Rc::clone(&environment);
    let _source = sp
        .section(PromptSection::provider(
            "source",
            PromptSectionOrderName::HarnessSource.order(),
            move |_| source.borrow().source.clone(),
        ))
        .unwrap();
    let url = Rc::clone(&environment);
    let _web = sp
        .section(PromptSection::provider(
            "web",
            PromptSectionOrderName::WebSurface.order(),
            move |_| url.borrow().url.clone(),
        ))
        .unwrap();

    let reusable_text = reusable
        .iter()
        .map(|name| name.name())
        .collect::<Vec<_>>()
        .join("\n\n");
    let prefix = format!("{HARNESS_IDENTITY_TEXT}\n\nModel model-a.\n\n{reusable_text}\n\n");
    let first = render_prompt(&sp.assemble(&AssembleContext::new()).unwrap()).unwrap();
    assert_eq!(
        first,
        format!("{prefix}/alice/dsh\n\nhttp://127.0.0.1:3080\n\nIn /alice/project on darwin.")
    );

    *environment.borrow_mut() = Environment {
        model: "model-a".to_string(),
        cwd: "C:/bob/project".to_string(),
        platform: "win32".to_string(),
        source: "C:/bob/dsh".to_string(),
        url: "http://127.0.0.1:4080".to_string(),
    };
    let second = render_prompt(&sp.assemble(&AssembleContext::new()).unwrap()).unwrap();
    assert_eq!(
        second,
        format!("{prefix}C:/bob/dsh\n\nhttp://127.0.0.1:4080\n\nIn C:/bob/project on win32.")
    );

    environment.borrow_mut().model = "model-b".to_string();
    let third = render_prompt(&sp.assemble(&AssembleContext::new()).unwrap()).unwrap();
    assert_eq!(third, second.replace("Model model-a.", "Model model-b."));
}

#[test]
fn live_iterates_variables_registered_by_an_earlier_provider() {
    let sp = mount(Config::default());
    let late: Rc<RefCell<Option<Registration>>> = Rc::new(RefCell::new(None));
    let sp_late = sp.clone();
    let _first = sp
        .variable("first", move |_| {
            if late.borrow().is_none() {
                *late.borrow_mut() = Some(
                    sp_late
                        .variable("late", |_| Some("second value".to_string()))
                        .unwrap(),
                );
            }
            Some("first value".to_string())
        })
        .unwrap();

    let assembly = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(assembly.variables.get("first"), Some(Some("first value")));
    assert_eq!(assembly.variables.get("late"), Some(Some("second value")));
}

#[test]
fn snapshots_tool_provider_membership_before_evaluating_an_assembly() {
    let sp = mount(Config::default());
    let late: Rc<RefCell<Option<Registration>>> = Rc::new(RefCell::new(None));
    let sp_late = sp.clone();
    let _first = sp.tools(move |_| {
        if late.borrow().is_none() {
            *late.borrow_mut() = Some(sp_late.tools(|_| ToolProviderResult {
                schemas: vec![tool("late")],
                known_names: None,
            }));
        }
        ToolProviderResult {
            schemas: vec![tool("first")],
            known_names: None,
        }
    });

    let first = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(tool_names(&first), ["first"]);
    let second = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(tool_names(&second), ["first", "late"]);
}

#[test]
fn removes_contributions_when_the_registration_is_dropped() {
    let sp = mount(Config::default());
    let section = sp
        .section(PromptSection::static_text("direct", 0.0, "direct section"))
        .unwrap();
    let tools = sp.tools(|_| ToolProviderResult {
        schemas: vec![tool("direct-tool")],
        known_names: None,
    });
    assert_eq!(
        contributed(&sp.assemble(&AssembleContext::new()).unwrap()).len(),
        1
    );
    assert_eq!(
        tool_names(&sp.assemble(&AssembleContext::new()).unwrap()),
        ["direct-tool"]
    );

    drop(section);
    drop(tools);
    assert!(contributed(&sp.assemble(&AssembleContext::new()).unwrap()).is_empty());
    assert!(tool_names(&sp.assemble(&AssembleContext::new()).unwrap()).is_empty());
}

#[test]
fn refresh_re_resolves_registered_contexts_without_reassembling_anything_else() {
    let sp = mount(Config::default());
    let current = Rc::new(RefCell::new("before".to_string()));
    let live_calls = Rc::new(Cell::new(0usize));
    let variable_calls = Rc::new(Cell::new(0usize));
    let tool_calls = Rc::new(Cell::new(0usize));

    let live_text = Rc::clone(&current);
    let live_count = Rc::clone(&live_calls);
    let _live = sp
        .context(
            PromptContext::provider("live", 0.0, move |_| {
                live_count.set(live_count.get() + 1);
                live_text.borrow().clone()
            })
            .with_interpolate(false),
        )
        .unwrap();
    let fixed_text = Rc::clone(&current);
    let fixed_count = Rc::clone(&variable_calls);
    let _fixed = sp
        .variable("fixed", move |_| {
            fixed_count.set(fixed_count.get() + 1);
            Some(fixed_text.borrow().clone())
        })
        .unwrap();
    let tool_text = Rc::clone(&current);
    let tool_count = Rc::clone(&tool_calls);
    let _tools = sp.tools(move |_| {
        tool_count.set(tool_count.get() + 1);
        let mut schema = ToolSchema::new("accepted_tool", tool_text.borrow().clone(), json!({}));
        schema.defer_loading = true;
        ToolProviderResult {
            schemas: vec![schema],
            known_names: None,
        }
    });

    let accepted = sp.assemble(&AssembleContext::new()).unwrap();
    *current.borrow_mut() = "after".to_string();
    let refreshed = sp
        .refresh_context(&accepted, &AssembleContext::new())
        .unwrap();

    assert_eq!(
        refreshed.contexts,
        vec![AssembledContext {
            name: "live".to_string(),
            text: "after".to_string(),
            interpolate: Some(false),
        }]
    );
    assert_eq!(accepted.contexts[0].text, "before");
    assert_eq!(refreshed.sections, accepted.sections);
    assert_eq!(refreshed.tools, accepted.tools);
    assert_eq!(refreshed.tools[0].description, "before");
    assert_eq!(refreshed.variables, accepted.variables);
    assert_eq!(
        (live_calls.get(), variable_calls.get(), tool_calls.get()),
        (2, 1, 1)
    );
}

#[test]
fn refresh_retains_required_context_when_optional_context_is_suppressed() {
    let sp = mount(Config {
        include_runtime_context: Some(false),
        ..Config::default()
    });
    let _directory = sp
        .context(
            PromptContext::static_text("directory", 0.0, "Directory: /project").with_required(true),
        )
        .unwrap();
    let _optional = sp
        .context(PromptContext::provider("optional", 1.0, |_| {
            panic!("suppressed provider ran")
        }))
        .unwrap();

    let assembly = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(
        assembly.contexts,
        vec![AssembledContext {
            name: "directory".to_string(),
            text: "Directory: /project".to_string(),
            interpolate: None,
        }]
    );
    let refreshed = sp
        .refresh_context(&assembly, &AssembleContext::new())
        .unwrap();
    assert_eq!(refreshed.contexts, assembly.contexts);
    assert!(render_context_snapshot(&assembly)
        .unwrap()
        .contains("Directory: /project"));
    // A runtime context is never a prompt section.
    assert!(!render_prompt(&assembly)
        .unwrap()
        .contains("Directory: /project"));
}

#[test]
fn refresh_restores_missing_required_contexts_before_a_later_registered_context() {
    for later in [false, true] {
        let sp = mount(Config::default());
        let _required = sp
            .context(PromptContext::static_text("required", 0.0, "fresh").with_required(true))
            .unwrap();
        let _later = sp
            .context(PromptContext::static_text("later", 10.0, "last"))
            .unwrap();
        let mut accepted = sp.assemble(&AssembleContext::new()).unwrap();
        accepted.contexts = if later {
            vec![
                AssembledContext {
                    name: "expert".to_string(),
                    text: "extra".to_string(),
                    interpolate: None,
                },
                accepted.contexts[1].clone(),
            ]
        } else {
            Vec::new()
        };

        let refreshed = sp
            .refresh_context(&accepted, &AssembleContext::new())
            .unwrap();
        let expected = if later {
            vec![
                AssembledContext {
                    name: "expert".to_string(),
                    text: "extra".to_string(),
                    interpolate: None,
                },
                AssembledContext {
                    name: "required".to_string(),
                    text: "fresh".to_string(),
                    interpolate: None,
                },
                AssembledContext {
                    name: "later".to_string(),
                    text: "last".to_string(),
                    interpolate: None,
                },
            ]
        } else {
            vec![AssembledContext {
                name: "required".to_string(),
                text: "fresh".to_string(),
                interpolate: None,
            }]
        };
        assert_eq!(refreshed.contexts, expected);
    }
}

#[test]
fn refresh_applies_current_suppression_without_evaluating_optional_providers() {
    let sp = mount(Config::default());
    let suppressed = Rc::new(Cell::new(false));
    let _required = sp
        .context(
            PromptContext::provider("required", 0.0, |_| "current".to_string()).with_required(true),
        )
        .unwrap();
    let flag = Rc::clone(&suppressed);
    let _optional = sp
        .context(PromptContext::provider("optional", 10.0, move |_| {
            assert!(!flag.get(), "suppressed context was evaluated");
            "optional".to_string()
        }))
        .unwrap();

    let mut accepted = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(accepted.contexts.len(), 2);
    // A contribution only the (unported) waterfall could have added.
    accepted.contexts.push(AssembledContext {
        name: "expert".to_string(),
        text: "accepted optional".to_string(),
        interpolate: None,
    });

    let _suppressor = sp.suppress_runtime_context();
    suppressed.set(true);
    let refreshed = sp
        .refresh_context(&accepted, &AssembleContext::new())
        .unwrap();
    assert_eq!(
        refreshed.contexts,
        vec![AssembledContext {
            name: "required".to_string(),
            text: "current".to_string(),
            interpolate: None,
        }]
    );
}
