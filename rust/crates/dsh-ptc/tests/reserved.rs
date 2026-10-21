//! Reserved-name validation.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node/tests/bindings.spec.ts`
//! and the three exclusion sets in `packages/ptc-runtime/ptc-runtime/src/index.ts`.

use dsh_ptc::{
    is_dunder_member, is_identifier, validate_bindings, BindingErrorClass, BindingNamespace,
    ReservedError, PORTABLE_RESERVED_WORDS, RESERVED_BINDING_GLOBALS, RESERVED_ERROR_MEMBERS,
};

fn namespace(global: &str, functions: &[&str]) -> BindingNamespace {
    BindingNamespace::new(
        global,
        functions.iter().map(|name| name.to_string()).collect(),
    )
}

fn with_class(global: &str, class: &str, member: &str) -> BindingNamespace {
    namespace(global, &[]).with_error_class(BindingErrorClass::new(class, member))
}

#[test]
fn rejects_unusable_globals() {
    for global in ["$name", "", "while", "lambda", "1tool", "name-with-dash"] {
        assert!(
            matches!(
                validate_bindings(vec![namespace(global, &[])]),
                Err(ReservedError::UnusableGlobal { .. })
            ),
            "expected {global:?} to be refused"
        );
    }
}

#[test]
fn rejects_backend_owned_globals() {
    for global in [
        "console",
        "__dsh_main__",
        "__builtins__",
        "__name__",
        "__debug__",
    ] {
        assert!(
            matches!(
                validate_bindings(vec![namespace(global, &[])]),
                Err(ReservedError::ReservedGlobal { .. })
            ),
            "expected {global:?} to be refused"
        );
    }
    assert_eq!(RESERVED_BINDING_GLOBALS.len(), 5);
}

#[test]
fn accepts_arbitrary_own_member_names() {
    let validated = validate_bindings(vec![namespace(
        "tools",
        &["__proto__", "constructor", "toString", "a b", ""],
    )])
    .expect("member names are unconstrained");
    assert_eq!(
        validated.namespaces()[0].functions,
        ["__proto__", "constructor", "toString", "a b", ""]
    );
}

#[test]
fn rejects_duplicate_globals() {
    let error = validate_bindings(vec![namespace("tools", &[]), namespace("tools", &[])])
        .expect_err("duplicates are refused");
    assert!(matches!(error, ReservedError::DuplicateGlobal { .. }));
    assert!(error
        .to_string()
        .contains("duplicate binding global \"tools\""));
}

#[test]
fn rejects_invalid_error_class_names() {
    for name in ["$name", "while", "console", "tools"] {
        assert!(
            validate_bindings(vec![with_class("tools", name, "member")]).is_err(),
            "expected class {name:?} to be refused"
        );
    }
}

#[test]
fn rejects_invalid_error_member_properties() {
    for member in [
        "",
        "name",
        "message",
        "stack",
        "args",
        "with_traceback",
        "add_note",
        "__member__",
    ] {
        assert!(
            validate_bindings(vec![with_class("tools", "ToolError", member)]).is_err(),
            "expected member {member:?} to be refused"
        );
    }
    assert_eq!(RESERVED_ERROR_MEMBERS.len(), 6);
}

#[test]
fn rejects_two_injected_constructors_with_one_name() {
    let error = validate_bindings(vec![
        with_class("first", "ToolError", "member"),
        with_class("second", "ToolError", "member"),
    ])
    .expect_err("duplicate injected globals are refused");
    assert!(matches!(
        error,
        ReservedError::DuplicateInjectedGlobal { .. }
    ));
}

#[test]
fn an_error_class_may_not_shadow_a_later_namespace_global() {
    // The first pass finishes every global before the second considers any
    // class, so a class colliding with a namespace declared *after* it is
    // still refused.
    let error = validate_bindings(vec![
        with_class("first", "second", "member"),
        namespace("second", &[]),
    ])
    .expect_err("the class name collides with a later global");
    assert!(matches!(
        error,
        ReservedError::DuplicateInjectedGlobal { .. }
    ));
}

#[test]
fn accepting_a_namespace_preserves_declaration_order() {
    let validated = validate_bindings(vec![
        namespace("tools", &["b", "a"])
            .with_error_class(BindingErrorClass::new("ToolCallError", "toolName")),
        namespace("state", &[]),
    ])
    .expect("valid declarations");
    assert_eq!(validated.len(), 2);
    assert_eq!(validated.namespaces()[0].global, "tools");
    assert_eq!(validated.namespaces()[0].functions, ["b", "a"]);
    assert_eq!(
        validated.namespaces()[0].error_class,
        Some(BindingErrorClass::new("ToolCallError", "toolName"))
    );
}

#[test]
fn identifier_rule_is_ascii_and_anchored() {
    for name in ["a", "_", "_a1", "A1_b"] {
        assert!(is_identifier(name), "expected {name:?} to be an identifier");
    }
    for name in ["", "1a", "a-b", "a b", "机构", "a.b"] {
        assert!(!is_identifier(name), "expected {name:?} to be refused");
    }
}

#[test]
fn dunder_form_needs_a_non_empty_middle() {
    assert!(is_dunder_member("__x__"));
    assert!(is_dunder_member("__member__"));
    assert!(!is_dunder_member("____"));
    assert!(!is_dunder_member("__"));
    assert!(!is_dunder_member("_x__"));
    assert!(!is_dunder_member("__x_"));
}

#[test]
fn the_portable_word_union_covers_both_target_languages() {
    for word in ["await", "lambda", "nonlocal", "match", "type", "_"] {
        assert!(PORTABLE_RESERVED_WORDS.contains(&word), "missing {word}");
    }
    // A duplicated entry would mean one language's list was pasted twice, and
    // a word in both sets would make the refusal reason ambiguous.
    let mut seen: Vec<&str> = Vec::new();
    for word in PORTABLE_RESERVED_WORDS {
        assert!(!seen.contains(word), "duplicate reserved word {word}");
        assert!(
            !RESERVED_BINDING_GLOBALS.contains(word),
            "{word} is both a reserved word and a reserved global"
        );
        seen.push(word);
    }
}
