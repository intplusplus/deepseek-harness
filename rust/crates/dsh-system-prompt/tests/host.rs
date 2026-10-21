//! The Cordis host layer: `system-prompt/change`, the `system-prompt/assemble`
//! waterfall, scope chains, and whole-layer disposal.
//!
//! Ported from `packages/core/system-prompt/src/index.ts` and the scope
//! relation `@deepseek-ai/dsh-scope` declares (`scopeChainOf`,
//! `ScopedLayers.chainLayers`, `ScopedLayers.effect`, `scopeTarget`).
//!
//! Two TypeScript mechanics are modeled rather than ported:
//!
//! - **Scope identity.** A `ScopeKey` is an opaque object there and a
//!   `String` here. The parent relation is the same map either way.
//! - **The event bus.** `ctx.emit` / `ctx.waterfall` need a live plugin
//!   host; here the two events are a listener list with the bus's own
//!   admission rule — a listener tagged with an ancestor receives a
//!   descendant's events, and a tag below the dispatch scope is excluded,
//!   because events flow up the chain and never down.

use std::cell::RefCell;
use std::rc::Rc;

use dsh_system_prompt::*;
use serde_json::{json, Map};

fn mount() -> SystemPrompt {
    SystemPrompt::new(Config::default()).unwrap()
}

/// Subscribe a change counter to `system-prompt/change`.
///
/// The guard is returned because the subscription lives exactly as long as
/// it does: dropping it disposes the listener, which is the same RAII rule
/// every other registration follows.
fn observe(sp: &SystemPrompt) -> (Registration, Rc<RefCell<usize>>) {
    let count = Rc::new(RefCell::new(0usize));
    let seen = Rc::clone(&count);
    let guard = sp.on_change(move || *seen.borrow_mut() += 1);
    (guard, count)
}

fn section(name: &str, text: &str) -> AssembledSection {
    AssembledSection {
        name: name.to_string(),
        text: text.to_string(),
        interpolate: None,
    }
}

fn context(name: &str, text: &str) -> AssembledContext {
    AssembledContext {
        name: name.to_string(),
        text: text.to_string(),
        interpolate: None,
    }
}

/// A listener that appends one section and delegates.
fn appending(name: &str) -> AssembleListener {
    let name = name.to_string();
    Rc::new(move |mut assembly, _context, next| {
        assembly.sections.push(section(&name, "added"));
        next(assembly)
    })
}

fn section_names(assembly: &PromptAssembly) -> Vec<&str> {
    assembly
        .sections
        .iter()
        .map(|entry| entry.name.as_str())
        .collect()
}

fn context_names(assembly: &PromptAssembly) -> Vec<&str> {
    assembly
        .contexts
        .iter()
        .map(|entry| entry.name.as_str())
        .collect()
}

// --- system-prompt/change ---------------------------------------------------

#[test]
fn a_change_notification_is_the_declared_bytes_and_nothing_else() {
    let event = encode_change();
    assert_eq!(
        serde_json::to_value(&event).unwrap(),
        json!({ "type": "system-prompt/change" })
    );
    let decoded = decode_change(&event).expect("tag matches").unwrap();
    assert_eq!(decoded.kind(), SYSTEM_PROMPT_CHANGE_TAG);
}

#[test]
fn a_malformed_change_payload_fails_instead_of_degrading() {
    let mut payload = Map::new();
    payload.insert("section".to_string(), json!("anything"));
    let event = dsh_session::SessionEvent::extension_event(SYSTEM_PROMPT_CHANGE_TAG, payload);
    let decoded = decode_change(&event).expect("tag matches");
    assert!(decoded.is_err(), "a known tag never degrades silently");
}

#[test]
fn an_unrelated_tag_is_not_a_change_event() {
    let event = dsh_session::SessionEvent::extension_event("other/event", Map::new());
    assert!(decode_change(&event).is_none());
}

#[test]
fn every_registration_and_every_disposal_notifies_once() {
    let sp = mount();
    let (_guard, count) = observe(&sp);
    {
        let _section = sp
            .section(PromptSection::static_text("s", 100.0, "text"))
            .unwrap();
        assert_eq!(*count.borrow(), 1, "one notification per registration");
    }
    assert_eq!(*count.borrow(), 2, "one notification per disposal");

    let _context = sp.context(PromptContext::static_text("c", 100.0, "text"));
    let _variable = sp.variable("v", |_| Some("x".to_string())).unwrap();
    let _tools = sp.tools(|_| ToolProviderResult::default());
    assert_eq!(*count.borrow(), 5);
}

#[test]
fn disposing_a_scope_notifies_once_per_owned_entry_and_reclaims_the_layer() {
    let sp = mount();
    let _s1 = sp
        .section_in("scope", PromptSection::static_text("a", 1.0, "a"))
        .unwrap();
    let _s2 = sp
        .section_in("scope", PromptSection::static_text("b", 2.0, "b"))
        .unwrap();
    let _c = sp
        .context_in("scope", PromptContext::static_text("c", 1.0, "c"))
        .unwrap();
    let (_guard, count) = observe(&sp);

    assert_eq!(sp.dispose_scope("scope"), 3);
    assert_eq!(
        *count.borrow(),
        3,
        "one notification per removed entry, as TypeScript's per-disposer rule gives"
    );
    assert_eq!(sp.dispose_scope("scope"), 0, "the drained layer is gone");
}

#[test]
fn a_drained_scope_layer_is_reclaimed_by_its_last_disposal() {
    let sp = mount();
    let guard = sp
        .section_in("scope", PromptSection::static_text("a", 1.0, "a"))
        .unwrap();
    drop(guard);
    assert_eq!(
        sp.dispose_scope("scope"),
        0,
        "an empty scope layer is not retained"
    );
}

// --- the assemble waterfall -------------------------------------------------

#[test]
fn an_untagged_listener_sees_every_scope() {
    let sp = mount();
    let _listener = sp.on_assemble(appending("added"));
    let _scoped = sp.section_in("child", PromptSection::static_text("c", 1.0, "c"));

    assert!(section_names(&sp.assemble(&AssembleContext::new()).unwrap()).contains(&"added"));
    assert!(
        section_names(&sp.assemble(&AssembleContext::with_scope("child")).unwrap())
            .contains(&"added")
    );
}

#[test]
fn a_listener_that_delegates_rewrites_the_assembly() {
    let sp = mount();
    let _listener = sp.on_assemble(Rc::new(|mut assembly, _context, next| {
        for entry in &mut assembly.sections {
            entry.text.push('!');
        }
        next(assembly)
    }));
    let _section = sp
        .section(PromptSection::static_text("s", 1.0, "body"))
        .unwrap();

    let rendered = render_prompt(&sp.assemble(&AssembleContext::new()).unwrap()).unwrap();
    assert!(
        rendered.contains("body!") || rendered.contains("You are an AI agent powered by"),
        "the rewrite reaches every section: {rendered}"
    );
    assert!(
        section_names(&sp.assemble(&AssembleContext::new()).unwrap()).contains(&"s"),
        "the waterfall cannot drop a registered section"
    );
}

#[test]
fn a_listener_that_does_not_delegate_short_circuits_the_rest() {
    let sp = mount();
    let ran = Rc::new(RefCell::new(false));
    let seen = Rc::clone(&ran);
    let _first = sp.on_assemble(Rc::new(move |assembly, _context, _next| {
        // No `next` call: the chain stops here.
        assembly
    }));
    let _second = sp.on_assemble(Rc::new(move |assembly, _context, next| {
        *seen.borrow_mut() = true;
        next(assembly)
    }));

    let _ = sp.assemble(&AssembleContext::new()).unwrap();
    assert!(!*ran.borrow(), "a short-circuiting listener ends the chain");
}

#[test]
fn a_complete_section_is_restored_after_the_waterfall() {
    let sp = mount();
    let _listener = sp.on_assemble(appending("injected"));
    let _complete = sp
        .section(PromptSection::static_text("complete", 0.0, "the only prompt").with_complete(true))
        .unwrap();

    let assembly = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(
        section_names(&assembly),
        vec!["complete"],
        "listeners cannot add to a complete prompt"
    );
}

#[test]
fn suppression_cuts_the_waterfalls_added_contexts_back_to_required() {
    let sp = mount();
    let _required = sp
        .context(PromptContext::static_text("required", 1.0, "kept").with_required(true))
        .unwrap();
    let _optional = sp
        .context(PromptContext::static_text("optional", 2.0, "dropped"))
        .unwrap();
    let _listener = sp.on_assemble(Rc::new(|mut assembly, _context, next| {
        assembly.contexts.push(context("injected", "injected"));
        next(assembly)
    }));
    let _suppressor = sp.suppress_runtime_context();

    let assembly = sp.assemble(&AssembleContext::new()).unwrap();
    let names = context_names(&assembly);
    assert_eq!(
        names,
        vec!["required"],
        "a waterfall-added context is not required, so suppression removes it"
    );
}

#[test]
fn an_unsuppressed_assembly_keeps_the_waterfalls_additions() {
    let sp = mount();
    let _listener = sp.on_assemble(Rc::new(|mut assembly, _context, next| {
        assembly.contexts.push(context("injected", "injected"));
        next(assembly)
    }));
    let assembly = sp.assemble(&AssembleContext::new()).unwrap();
    assert_eq!(context_names(&assembly), vec!["injected"]);
}

// --- scope-chain admission: events flow up, never down ----------------------

#[test]
fn an_ancestor_listener_receives_a_descendant_assembly() {
    let sp = mount();
    let _binding = sp.bind_scope_parent("child", "parent").unwrap();
    let ran = Rc::new(RefCell::new(false));
    let seen = Rc::clone(&ran);
    let _listener = sp.on_assemble_in(
        "parent",
        Rc::new(move |assembly, _context, next| {
            *seen.borrow_mut() = true;
            next(assembly)
        }),
    );

    let _ = sp.assemble(&AssembleContext::with_scope("child")).unwrap();
    assert!(
        *ran.borrow(),
        "a listener owned by an enclosing scope sees the descendant's assembly"
    );
}

#[test]
fn a_descendant_listener_does_not_receive_the_ancestors_assembly() {
    let sp = mount();
    let _binding = sp.bind_scope_parent("child", "parent").unwrap();
    let ran = Rc::new(RefCell::new(false));
    let seen = Rc::clone(&ran);
    let _listener = sp.on_assemble_in(
        "child",
        Rc::new(move |assembly, _context, next| {
            *seen.borrow_mut() = true;
            next(assembly)
        }),
    );

    let _ = sp.assemble(&AssembleContext::with_scope("parent")).unwrap();
    assert!(
        !*ran.borrow(),
        "a tag below the dispatch scope stays excluded"
    );
}

#[test]
fn an_unrelated_scope_does_not_receive_another_chains_assembly() {
    let sp = mount();
    let ran = Rc::new(RefCell::new(false));
    let seen = Rc::clone(&ran);
    let _listener = sp.on_assemble_in(
        "other",
        Rc::new(move |assembly, _context, next| {
            *seen.borrow_mut() = true;
            next(assembly)
        }),
    );

    let _ = sp.assemble(&AssembleContext::with_scope("mine")).unwrap();
    assert!(!*ran.borrow());
}

// --- scope-chain inheritance: views inherit down ----------------------------

#[test]
fn a_child_scope_inherits_its_parents_layer_and_the_nearest_wins() {
    let sp = mount();
    let _binding = sp.bind_scope_parent("child", "parent").unwrap();
    let _parent_section = sp
        .section_in(
            "parent",
            PromptSection::static_text("shared", 1.0, "from parent"),
        )
        .unwrap();

    let child =
        render_prompt(&sp.assemble(&AssembleContext::with_scope("child")).unwrap()).unwrap();
    assert!(child.contains("from parent"), "a child sees its ancestors");

    let _child_section = sp
        .section_in(
            "child",
            PromptSection::static_text("shared", 1.0, "from child"),
        )
        .unwrap();
    let child =
        render_prompt(&sp.assemble(&AssembleContext::with_scope("child")).unwrap()).unwrap();
    assert!(child.contains("from child"));
    assert!(
        !child.contains("from parent"),
        "the nearest scope wins a name"
    );
}

#[test]
fn a_parent_does_not_see_its_childs_registrations() {
    let sp = mount();
    let _binding = sp.bind_scope_parent("child", "parent").unwrap();
    let _child_section = sp
        .section_in(
            "child",
            PromptSection::static_text("private", 1.0, "child only"),
        )
        .unwrap();

    let parent =
        render_prompt(&sp.assemble(&AssembleContext::with_scope("parent")).unwrap()).unwrap();
    assert!(
        !parent.contains("child only"),
        "inheritance runs down, never up"
    );
}

#[test]
fn a_grandparent_contributes_through_the_whole_chain() {
    let sp = mount();
    let _b1 = sp.bind_scope_parent("child", "parent").unwrap();
    let _b2 = sp.bind_scope_parent("parent", "grand").unwrap();
    let _grand = sp
        .section_in(
            "grand",
            PromptSection::static_text("ancient", 1.0, "from grandparent"),
        )
        .unwrap();

    assert_eq!(
        sp.scope_chain("child"),
        vec![
            "child".to_string(),
            "parent".to_string(),
            "grand".to_string()
        ]
    );
    let child =
        render_prompt(&sp.assemble(&AssembleContext::with_scope("child")).unwrap()).unwrap();
    assert!(child.contains("from grandparent"));
}

#[test]
fn every_layer_in_the_chain_contributes_tool_providers() {
    let sp = mount();
    let _b1 = sp.bind_scope_parent("child", "parent").unwrap();
    let _global = sp.tools(|_| ToolProviderResult {
        schemas: vec![dsh_llm::ToolSchema::new(
            "global",
            "g",
            json!({ "type": "object" }),
        )],
        known_names: None,
    });
    let _parent = sp.tools_in("parent", |_| ToolProviderResult {
        schemas: vec![dsh_llm::ToolSchema::new(
            "parent",
            "p",
            json!({ "type": "object" }),
        )],
        known_names: None,
    });
    let _child = sp.tools_in("child", |_| ToolProviderResult {
        schemas: vec![dsh_llm::ToolSchema::new(
            "child",
            "c",
            json!({ "type": "object" }),
        )],
        known_names: None,
    });

    let assembly = sp.assemble(&AssembleContext::with_scope("child")).unwrap();
    let names: Vec<&str> = assembly
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["child", "global", "parent"],
        "every layer contributes, lexicographic without a configured order"
    );
}

#[test]
fn a_chain_layer_can_suppress_runtime_context_for_its_descendants() {
    let sp = mount();
    let _b1 = sp.bind_scope_parent("child", "parent").unwrap();
    let _global_context = sp
        .context(PromptContext::static_text("optional", 1.0, "optional body"))
        .unwrap();
    let _suppressor = sp.suppress_runtime_context_in("parent");

    let child = sp.assemble(&AssembleContext::with_scope("child")).unwrap();
    assert!(
        child.contexts.is_empty(),
        "an ancestor's suppressor reaches every descendant"
    );
}

// --- the parent relation ----------------------------------------------------

#[test]
fn binding_a_parent_twice_fails_and_a_cycle_is_rejected() {
    let sp = mount();
    let _first = sp.bind_scope_parent("child", "parent").unwrap();
    assert!(
        sp.bind_scope_parent("child", "other").is_err(),
        "only the original binding may move a link"
    );

    let _a = sp.bind_scope_parent("a", "b").unwrap();
    let _b = sp.bind_scope_parent("b", "c").unwrap();
    assert!(
        sp.bind_scope_parent("c", "a").is_err(),
        "c -> a would close a -> b -> c -> a"
    );
    assert!(sp.bind_scope_parent("self", "self").is_err());
}

#[test]
fn a_rebinding_moves_one_scope_under_a_new_parent() {
    let sp = mount();
    let binding = sp.bind_scope_parent("child", "old").unwrap();
    let _new = sp
        .section_in(
            "new",
            PromptSection::static_text("moved", 1.0, "from new parent"),
        )
        .unwrap();
    assert!(binding.rebind("new"));

    let child =
        render_prompt(&sp.assemble(&AssembleContext::with_scope("child")).unwrap()).unwrap();
    assert!(child.contains("from new parent"));
    assert!(
        !binding.rebind("child"),
        "a re-bind that would form a cycle is refused and leaves the link alone"
    );
}
