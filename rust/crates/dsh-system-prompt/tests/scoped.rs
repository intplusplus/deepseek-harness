//! Ported from `packages/core/system-prompt/tests/scoped.spec.ts`.
//!
//! The TypeScript suite mints real scopes through `createScope` /
//! `scopeOf` and disposes them; those are `@deepseek-ai/dsh-scope`
//! lifetime mechanics and are not ported. What is ported is the shadowing
//! rule itself — a scoped registration shadows the global one with the
//! same name, for that scope only — against a registry keyed by a plain
//! scope string.
//!
//! Not ported, because each is scope-package lifetime: minting and
//! disposing a scope, re-minting the same key starting clean, and the
//! scoped assemble listener shaping only its own scope. The
//! self-replacing scoped variable is ported — it needs only the
//! registry's live-iteration and drain-detach rules, both owned here.
//!
//! One shape had to change: TypeScript reaches its own disposer through
//! the closure's scope, while a Rust guard must be stashed in a cell the
//! closure can take from.

use std::cell::RefCell;
use std::rc::Rc;

use dsh_llm::ToolSchema;
use dsh_system_prompt::*;
use serde_json::json;

fn mount(config: Config) -> SystemPrompt {
    SystemPrompt::new(config).unwrap()
}

fn schema(name: &str) -> ToolSchema {
    ToolSchema::new(
        name,
        format!("tool {name}"),
        json!({ "type": "object", "properties": {} }),
    )
}

fn tool_names(assembly: &PromptAssembly) -> Vec<&str> {
    assembly
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect()
}

#[test]
fn a_scoped_persona_shadows_the_deployment_one_for_that_scope_only() {
    let sp = mount(Config {
        persona_prefix: Some("You are the deployment.".to_string()),
        ..Config::default()
    });
    let _shadow = sp
        .section_in(
            "child",
            PromptSection::static_text(PERSONA_PREFIX_SECTION, 0.0, "You run tests."),
        )
        .unwrap();

    let scoped =
        render_prompt(&sp.assemble(&AssembleContext::with_scope("child")).unwrap()).unwrap();
    let global = render_prompt(&sp.assemble(&AssembleContext::new()).unwrap()).unwrap();
    assert!(scoped.contains("You run tests."));
    assert!(!scoped.contains("You are the deployment."));
    assert!(global.contains("You are the deployment."));
    assert!(!global.contains("You run tests."));
}

#[test]
fn a_scoped_variable_shadows_its_global_name_twin_for_that_scope() {
    let sp = mount(Config {
        persona_prefix: Some("Mode: {{mode}}.".to_string()),
        ..Config::default()
    });
    let _global = sp.variable("mode", |_| Some("normal".to_string())).unwrap();
    let _scoped = sp
        .variable_in("child", "mode", |_| Some("strict".to_string()))
        .unwrap();

    let scoped =
        render_prompt(&sp.assemble(&AssembleContext::with_scope("child")).unwrap()).unwrap();
    let global = render_prompt(&sp.assemble(&AssembleContext::new()).unwrap()).unwrap();
    assert!(scoped.contains("Mode: strict."));
    assert!(global.contains("Mode: normal."));
}

#[test]
fn a_scoped_context_shadows_the_global_one_and_suppression_is_per_scope() {
    let sp = mount(Config::default());
    let _global = sp
        .context(PromptContext::static_text("policy", 1.0, "global policy"))
        .unwrap();
    let _scoped = sp
        .context_in(
            "child-context",
            PromptContext::static_text("policy", 1.0, "scoped policy"),
        )
        .unwrap();
    let err = sp
        .context_in(
            "child-context",
            PromptContext::static_text("policy", 2.0, "duplicate"),
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("prompt context \"policy\" is already registered in this scope"),
        "{err}"
    );

    let scoped = sp
        .assemble(&AssembleContext::with_scope("child-context"))
        .unwrap();
    assert!(render_context_snapshot(&scoped)
        .unwrap()
        .contains("scoped policy"));
    let global = sp.assemble(&AssembleContext::new()).unwrap();
    assert!(render_context_snapshot(&global)
        .unwrap()
        .contains("global policy"));

    // Suppression is per scope: the global assembly keeps its context.
    let sp = mount(Config::default());
    let _policy = sp
        .context(PromptContext::static_text("policy", 1.0, "global policy"))
        .unwrap();
    let suppressor = sp.suppress_runtime_context_in("suppressed-context");
    let suppressed = sp
        .assemble(&AssembleContext::with_scope("suppressed-context"))
        .unwrap();
    assert!(suppressed.contexts.is_empty());
    let global = sp.assemble(&AssembleContext::new()).unwrap();
    assert!(render_context_snapshot(&global)
        .unwrap()
        .contains("global policy"));
    drop(suppressor);
    let restored = sp
        .assemble(&AssembleContext::with_scope("suppressed-context"))
        .unwrap();
    assert!(render_context_snapshot(&restored)
        .unwrap()
        .contains("global policy"));
}

#[test]
fn scoped_tool_providers_are_consulted_only_for_their_scope() {
    let sp = mount(Config::default());
    let _global = sp.tools(|_| ToolProviderResult {
        schemas: vec![schema("global_tool")],
        known_names: None,
    });
    let _scoped = sp.tools_in("child", |_| ToolProviderResult {
        schemas: vec![schema("scoped_tool")],
        known_names: None,
    });

    let scoped = sp.assemble(&AssembleContext::with_scope("child")).unwrap();
    assert_eq!(tool_names(&scoped), ["global_tool", "scoped_tool"]);
    let global = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(tool_names(&global), ["global_tool"]);
}

#[test]
fn defers_a_scoped_variable_that_replaces_the_last_provider_in_its_generation() {
    let sp = mount(Config {
        persona_prefix: Some("Mode: {{mode}}.".to_string()),
        ..Config::default()
    });
    let calls = Rc::new(RefCell::new(Vec::<&str>::new()));
    // The provider's own guard, stashed where the closure can take it:
    // the Rust stand-in for calling the TypeScript disposer in place.
    let cell: Rc<RefCell<Option<Registration>>> = Rc::new(RefCell::new(None));
    let seat = Rc::clone(&cell);
    let log = Rc::clone(&calls);
    let sp_inner = sp.clone();
    let first = sp
        .variable_in("child", "mode", move |_| {
            log.borrow_mut().push("first");
            let disposal = seat.borrow_mut().take();
            drop(disposal);
            let seat = Rc::clone(&seat);
            let log = Rc::clone(&log);
            *seat.borrow_mut() = Some(
                sp_inner
                    .variable_in("child", "mode", move |_| {
                        log.borrow_mut().push("replacement");
                        Some("replacement".to_string())
                    })
                    .unwrap(),
            );
            Some("first".to_string())
        })
        .unwrap();
    *cell.borrow_mut() = Some(first);

    // The drained table detached this assembly's walk: the replacement is
    // deferred to the next generation, exactly as the TypeScript test
    // asserts — 'first' now, 'replacement' only on the next assemble.
    let scoped =
        render_prompt(&sp.assemble(&AssembleContext::with_scope("child")).unwrap()).unwrap();
    assert!(scoped.contains("Mode: first."), "{scoped}");
    assert_eq!(*calls.borrow(), ["first"]);

    let scoped =
        render_prompt(&sp.assemble(&AssembleContext::with_scope("child")).unwrap()).unwrap();
    assert!(scoped.contains("Mode: replacement."), "{scoped}");
    assert_eq!(*calls.borrow(), ["first", "replacement"]);
}

#[test]
fn duplicate_names_throw_per_layer_with_scope_specific_messages() {
    let sp = mount(Config::default());
    let _x = sp
        .section(PromptSection::static_text("x", 1.0, "a"))
        .unwrap();
    let err = sp
        .section(PromptSection::static_text("x", 1.0, "b"))
        .unwrap_err();
    assert!(
        err.to_string().contains(
            "prompt section \"x\" is already registered (for a per-agent override, register through that agent's `agent.ctx` instead)"
        ),
        "{err}"
    );
    let _y = sp
        .section_in("child", PromptSection::static_text("y", 1.0, "a"))
        .unwrap();
    let err = sp
        .section_in("child", PromptSection::static_text("y", 1.0, "b"))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("prompt section \"y\" is already registered in this scope"),
        "{err}"
    );
}
