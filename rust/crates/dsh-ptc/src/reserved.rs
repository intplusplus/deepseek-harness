//! Names a program may not claim, and the binding-declaration validator.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime/src/index.ts` (the three
//! shared exclusion sets plus the dunder form) and
//! `packages/ptc-runtime/ptc-runtime-node/src/bindings.ts`
//! (`validateBindings`).
//!
//! The sets are *shared across backends by design*: a namespace list that is
//! valid for one execution backend must be valid for every backend, so the
//! sets name the union of the backend-owned slots rather than one backend's
//! own. `console` is Node's log-capture slot; `__dsh_main__`,
//! `__builtins__`, `__name__` and `__debug__` are the Python wrapper's
//! seeded globals and CPython's compile-time constant.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Globals every backend refuses because some backend owns the slot.
pub const RESERVED_BINDING_GLOBALS: &[&str] = &[
    "console",
    "__dsh_main__",
    "__builtins__",
    "__name__",
    "__debug__",
];

/// Error-member property names every backend refuses.
pub const RESERVED_ERROR_MEMBERS: &[&str] = &[
    "name",
    "message",
    "stack",
    "args",
    "with_traceback",
    "add_note",
];

/// Reserved words of every portable target language (ECMAScript ∪ Python).
///
/// Kept in the TypeScript declaration's order so the two lists can be audited
/// side by side. Membership is the contract; a per-language check would let
/// `lambda` pass one backend and fail the other.
pub const PORTABLE_RESERVED_WORDS: &[&str] = &[
    // ECMAScript reserved words and reserved-in-strict-mode names.
    "await",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "function",
    "if",
    "import",
    "in",
    "instanceof",
    "new",
    "null",
    "return",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "var",
    "void",
    "while",
    "with",
    "yield",
    "let",
    "static",
    "implements",
    "interface",
    "package",
    "private",
    "protected",
    "public",
    "arguments",
    "eval",
    // Python 3.x keywords plus the `type`/`_` soft keywords, reserved for safety.
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "def",
    "del",
    "elif",
    "except",
    "from",
    "global",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "match",
    "type",
    "_",
];

/// Whether `name` is the dunder form `__x__` with a non-empty middle.
///
/// This is TypeScript's `/^__.+__$/` without a regex dependency. The dunder
/// form is refused wholesale for error members because several are
/// constrained interpreter descriptors whose `setattr` raises while the
/// rejection is constructed, and the exact set is interpreter-version
/// dependent.
pub fn is_dunder_member(name: &str) -> bool {
    name.len() >= 5 && name.starts_with("__") && name.ends_with("__")
}

/// Whether `name` is a language-portable identifier (`[A-Za-z_][A-Za-z0-9_]*`).
///
/// ASCII-only on purpose: the TypeScript regex is ASCII, and a name that
/// passes here must be spellable in every target language.
pub fn is_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    match bytes.next() {
        Some(first) if first.is_ascii_alphabetic() || first == b'_' => {}
        _ => return false,
    }
    bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// The program-visible typed rejection contract for one namespace.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BindingErrorClass {
    /// Constructor global and resulting error name.
    pub name: String,
    /// Non-empty own property carrying the rejected member's name.
    pub member_name_property: String,
}

impl BindingErrorClass {
    /// Create one rejection-class descriptor.
    pub fn new(name: impl Into<String>, member_name_property: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            member_name_property: member_name_property.into(),
        }
    }
}

/// One declared binding namespace, as the host hands it to a runtime.
///
/// The functions themselves stay host-side — the program only receives their
/// names — so the declaration carries `functions: Vec<String>` rather than
/// callables. Order is preserved: it is the member order the program sees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingNamespace {
    /// The global identifier the program sees.
    pub global: String,
    /// The callable member names, in declaration order.
    pub functions: Vec<String>,
    /// Optional program-visible typed rejection contract.
    pub error_class: Option<BindingErrorClass>,
}

impl BindingNamespace {
    /// A namespace with no rejection class.
    pub fn new(global: impl Into<String>, functions: Vec<String>) -> Self {
        Self {
            global: global.into(),
            functions,
            error_class: None,
        }
    }

    /// A namespace that injects the given rejection class.
    pub fn with_error_class(mut self, error_class: BindingErrorClass) -> Self {
        self.error_class = Some(error_class);
        self
    }
}

/// Declarations that passed validation, in declaration order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedBindings {
    namespaces: Vec<BindingNamespace>,
}

impl ValidatedBindings {
    /// The validated declarations, in declaration order.
    pub fn namespaces(&self) -> &[BindingNamespace] {
        &self.namespaces
    }

    /// The validated declarations, consuming the wrapper.
    pub fn into_namespaces(self) -> Vec<BindingNamespace> {
        self.namespaces
    }

    /// The number of declared namespaces.
    pub fn len(&self) -> usize {
        self.namespaces.len()
    }

    /// Whether no namespace was declared.
    pub fn is_empty(&self) -> bool {
        self.namespaces.is_empty()
    }
}

/// Why a binding declaration was refused.
///
/// Every message names the offending value in JSON form, and the variant
/// distinguishes the rule that fired, so a caller can locate the bad
/// declaration without string matching.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReservedError {
    /// The global is not a portable identifier, or is a portable reserved word.
    #[error("dsh-ptc: binding global {global} is not a usable identifier")]
    UnusableGlobal {
        /// The refused global, JSON-quoted.
        global: String,
    },
    /// The global names a backend-owned slot.
    #[error("dsh-ptc: reserved binding global {global}")]
    ReservedGlobal {
        /// The refused global, JSON-quoted.
        global: String,
    },
    /// Two namespaces declared the same global.
    #[error("dsh-ptc: duplicate binding global {global}")]
    DuplicateGlobal {
        /// The duplicated global, JSON-quoted.
        global: String,
    },
    /// The error-class name is not a portable identifier, or is a reserved word.
    #[error("dsh-ptc: binding error class {name} is not a usable identifier")]
    UnusableErrorClass {
        /// The refused class name, JSON-quoted.
        name: String,
    },
    /// The error-class name collides with another injected global.
    #[error("dsh-ptc: duplicate injected global {name}")]
    DuplicateInjectedGlobal {
        /// The colliding class name, JSON-quoted.
        name: String,
    },
    /// The member-name property is empty, reserved, or dunder-form.
    #[error("dsh-ptc: binding error member property {member} is not usable")]
    UnusableErrorMember {
        /// The refused property name, JSON-quoted.
        member: String,
    },
}

fn quoted(value: &str) -> String {
    serde_json::Value::String(value.to_owned()).to_string()
}

fn unusable_identifier(name: &str) -> bool {
    !is_identifier(name) || PORTABLE_RESERVED_WORDS.contains(&name)
}

fn reserved_global(name: &str) -> bool {
    RESERVED_BINDING_GLOBALS.contains(&name)
}

/// Validate binding declarations before any program starts.
///
/// The two passes are ordered, and the order is observable: every global is
/// checked and de-duplicated before any error class is considered, so an
/// error class colliding with a *later* namespace's global is still refused
/// (TypeScript's first loop completes first for the same reason).
pub fn validate_bindings(
    namespaces: Vec<BindingNamespace>,
) -> Result<ValidatedBindings, ReservedError> {
    for namespace in &namespaces {
        if unusable_identifier(&namespace.global) {
            return Err(ReservedError::UnusableGlobal {
                global: quoted(&namespace.global),
            });
        }
        if reserved_global(&namespace.global) {
            return Err(ReservedError::ReservedGlobal {
                global: quoted(&namespace.global),
            });
        }
        if namespaces
            .iter()
            .filter(|other| other.global == namespace.global)
            .count()
            > 1
        {
            return Err(ReservedError::DuplicateGlobal {
                global: quoted(&namespace.global),
            });
        }
    }

    let mut error_class_names: Vec<&str> = Vec::new();
    for namespace in &namespaces {
        let Some(descriptor) = namespace.error_class.as_ref() else {
            continue;
        };
        if unusable_identifier(&descriptor.name) {
            return Err(ReservedError::UnusableErrorClass {
                name: quoted(&descriptor.name),
            });
        }
        if reserved_global(&descriptor.name) {
            return Err(ReservedError::ReservedGlobal {
                global: quoted(&descriptor.name),
            });
        }
        if namespaces
            .iter()
            .any(|other| other.global == descriptor.name)
            || error_class_names.contains(&descriptor.name.as_str())
        {
            return Err(ReservedError::DuplicateInjectedGlobal {
                name: quoted(&descriptor.name),
            });
        }
        let member = &descriptor.member_name_property;
        if member.is_empty()
            || RESERVED_ERROR_MEMBERS.contains(&member.as_str())
            || is_dunder_member(member)
        {
            return Err(ReservedError::UnusableErrorMember {
                member: quoted(member),
            });
        }
        error_class_names.push(&descriptor.name);
    }

    Ok(ValidatedBindings { namespaces })
}
