//! Registry input types and the assembled model input.
//!
//! [`PromptSection`] and [`PromptContext`] are what plugins register;
//! [`AssembledSection`] and [`AssembledContext`] are the same contributions
//! with their text resolved; [`PromptAssembly`] is the merge-extensible
//! model input the waterfall carries in TypeScript. Sections and contexts
//! stay uninterpolated until [`render_prompt`](crate::render_prompt); tools
//! are already in canonical order.
//!
//! Wire format follows the project convention (§3.9 of the workspace
//! README): `camelCase` fields, and a `field?: boolean` slot that TypeScript
//! spells "present only when defined" becomes
//! `#[serde(default, skip_serializing_if = "Option::is_none")]` — an
//! `interpolate` that was never set must not appear on the wire, because
//! `interpolate: false` and an absent `interpolate` mean different things.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;

use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use dsh_llm::ToolSchema;

/// Static text, or a provider evaluated at each assembly with that
/// assembly's [`AssembleContext`].
#[derive(Clone)]
pub enum PromptText {
    Static(String),
    Provider(Rc<dyn Fn(&AssembleContext) -> String>),
}

impl PromptText {
    /// Resolve the text against one assembly context.
    pub fn resolve(&self, context: &AssembleContext) -> String {
        match self {
            Self::Static(text) => text.clone(),
            Self::Provider(provider) => provider(context),
        }
    }
}

impl fmt::Debug for PromptText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static(text) => f.debug_tuple("Static").field(text).finish(),
            Self::Provider(_) => f.write_str("Provider(<provider>)"),
        }
    }
}

/// One contributed section of the system prompt (registry input).
#[derive(Clone, Debug)]
pub struct PromptSection {
    /// Unique name — a duplicate registration fails (see
    /// [`SystemPrompt::section`](crate::SystemPrompt::section)).
    pub name: String,
    /// Sections are concatenated in ascending order. Equal orders use
    /// code-unit name order.
    pub order: f64,
    /// Static text or a provider. The text may reference `{{variable}}`s —
    /// they are interpolated later, by [`render_prompt`](crate::render_prompt),
    /// unless [`PromptSection::interpolate`] is `Some(false)`.
    pub text: PromptText,
    /// Whether to interpolate prompt variables. `None` (unset) defaults to
    /// true and is omitted from the wire; `Some(false)` preserves literal
    /// text.
    pub interpolate: Option<bool>,
    /// Treat this contribution as the complete system prompt.
    pub complete: bool,
}

impl PromptSection {
    /// A section with static text.
    pub fn static_text(name: impl Into<String>, order: f64, text: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            order,
            text: PromptText::Static(text.into()),
            interpolate: None,
            complete: false,
        }
    }

    /// A section whose text is evaluated at each assembly.
    pub fn provider(
        name: impl Into<String>,
        order: f64,
        provider: impl Fn(&AssembleContext) -> String + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            order,
            text: PromptText::Provider(Rc::new(provider)),
            interpolate: None,
            complete: false,
        }
    }

    /// Set the interpolation flag.
    pub fn with_interpolate(mut self, interpolate: bool) -> Self {
        self.interpolate = Some(interpolate);
        self
    }

    /// Mark this section as the complete prompt.
    pub fn with_complete(mut self, complete: bool) -> Self {
        self.complete = complete;
        self
    }
}

/// Dynamic model context materialized as a durable user-role snapshot.
#[derive(Clone, Debug)]
pub struct PromptContext {
    /// Unique name — a duplicate registration fails (see
    /// [`SystemPrompt::context`](crate::SystemPrompt::context)).
    pub name: String,
    /// Contexts are joined in ascending order.
    pub order: f64,
    /// Static text or a provider. Empty text contributes nothing.
    pub text: PromptText,
    /// Whether to interpolate prompt variables; `None` defaults to true.
    pub interpolate: Option<bool>,
    /// Keep this operational context when optional runtime context is
    /// disabled.
    pub required: bool,
}

impl PromptContext {
    /// A context with static text.
    pub fn static_text(name: impl Into<String>, order: f64, text: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            order,
            text: PromptText::Static(text.into()),
            interpolate: None,
            required: false,
        }
    }

    /// A context whose text is evaluated at each assembly and refresh.
    pub fn provider(
        name: impl Into<String>,
        order: f64,
        provider: impl Fn(&AssembleContext) -> String + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            order,
            text: PromptText::Provider(Rc::new(provider)),
            interpolate: None,
            required: false,
        }
    }

    /// Set the interpolation flag.
    pub fn with_interpolate(mut self, interpolate: bool) -> Self {
        self.interpolate = Some(interpolate);
        self
    }

    /// Mark this context as required under suppression.
    pub fn with_required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }
}

/// One section of an assembly: a [`PromptSection`] with its text resolved.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssembledSection {
    /// The contributing section's unique name.
    pub name: String,
    /// The resolved (but not yet interpolated) section text.
    pub text: String,
    /// Whether to interpolate prompt variables; serialized only when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpolate: Option<bool>,
}

/// One resolved dynamic context contribution.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssembledContext {
    /// The contributing context's unique name.
    pub name: String,
    /// The resolved text before variable interpolation.
    pub text: String,
    /// Whether to interpolate prompt variables; serialized only when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpolate: Option<bool>,
}

/// One named contribution to a runtime-context snapshot, in assembly order.
///
/// TypeScript declares this in `@deepseek-ai/dsh-llm`; the Rust `dsh-llm`
/// crate has not ported it yet, so it lives here for now. When `dsh-llm`
/// grows the type, switch to re-exporting it instead of defining it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextSnapshotSection {
    /// The contributing subsystem's name.
    pub name: String,
    /// That contribution's model-facing text, exactly as assembled.
    pub text: String,
}

/// Tool schemas visible in one assembly and their pre-restriction name set.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolProviderResult {
    /// The schemas this provider contributes to THIS assembly.
    pub schemas: Vec<ToolSchema>,
    /// The pre-restriction name universe for config validation; defaults to
    /// `schemas`' names when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub known_names: Option<Vec<String>>,
}

/// Merge-extensible assembled model input.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptAssembly {
    /// Resolved sections in canonical order.
    pub sections: Vec<AssembledSection>,
    /// Resolved contexts in canonical order.
    pub contexts: Vec<AssembledContext>,
    /// Tool schemas in canonical order.
    pub tools: Vec<ToolSchema>,
    /// Registered prompt variables and their values for this assembly.
    pub variables: PromptVariables,
}

/// Registered prompt variables: a name maps to its value, or to "registered
/// but without a value for this assembly".
///
/// This is the Rust shape of TypeScript's `Record<string, string |
/// undefined>`: key presence means registered, and `None` means the
/// provider returned `undefined`. Serializing skips valueless entries,
/// exactly as `JSON.stringify` drops `undefined` values; deserializing maps
/// a JSON `null` back to "registered without a value".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PromptVariables(BTreeMap<String, Option<String>>);

impl PromptVariables {
    /// The variable's value: `None` when unregistered, `Some(None)` when
    /// registered without a value for this assembly.
    pub fn get(&self, name: &str) -> Option<Option<&str>> {
        self.0.get(name).map(|value| value.as_deref())
    }

    /// Whether the name is registered, value or not.
    pub fn contains_key(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    /// Record one variable's value for this assembly.
    pub fn insert(&mut self, name: impl Into<String>, value: Option<String>) {
        self.0.insert(name.into(), value);
    }

    /// How many variables are registered.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether no variables are registered.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The registered names, sorted, as the diagnostics list them.
    pub(crate) fn known_list(&self) -> String {
        if self.0.is_empty() {
            return "(none)".to_string();
        }
        self.0.keys().cloned().collect::<Vec<_>>().join(", ")
    }
}

impl From<BTreeMap<String, Option<String>>> for PromptVariables {
    fn from(variables: BTreeMap<String, Option<String>>) -> Self {
        Self(variables)
    }
}

impl<'a> FromIterator<(&'a str, Option<&'a str>)> for PromptVariables {
    fn from_iter<I: IntoIterator<Item = (&'a str, Option<&'a str>)>>(iter: I) -> Self {
        Self(
            iter.into_iter()
                .map(|(name, value)| (name.to_string(), value.map(str::to_string)))
                .collect(),
        )
    }
}

impl Serialize for PromptVariables {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let valued = self.0.iter().filter(|(_, value)| value.is_some()).count();
        let mut map = serializer.serialize_map(Some(valued))?;
        for (name, value) in &self.0 {
            if let Some(value) = value {
                map.serialize_entry(name, value)?;
            }
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for PromptVariables {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = BTreeMap::<String, Option<String>>::deserialize(deserializer)?;
        Ok(Self(raw))
    }
}

/// Merge-extensible context for one prompt assembly.
///
/// TypeScript lets plugins smuggle extra fields through this object (the
/// loop passes `{ agent }`); the typed [`scope`](AssembleContext::scope)
/// slot plus a free-form field map keeps that extensibility. The
/// `signal` member of the TypeScript type is an `AbortSignal` — a host
/// lifetime concern that is not ported.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssembleContext {
    /// Scope whose providers participate. When absent, only global
    /// providers participate.
    pub scope: Option<String>,
    fields: BTreeMap<String, Value>,
}

impl AssembleContext {
    /// An empty global context.
    pub fn new() -> Self {
        Self::default()
    }

    /// A context for one named scope.
    pub fn with_scope(scope: impl Into<String>) -> Self {
        Self {
            scope: Some(scope.into()),
            fields: BTreeMap::new(),
        }
    }

    /// One plugin-defined assembly field, when present.
    pub fn field(&self, key: &str) -> Option<&Value> {
        self.fields.get(key)
    }

    /// Set one plugin-defined assembly field.
    pub fn set_field(&mut self, key: impl Into<String>, value: impl Into<Value>) -> &mut Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    /// Set one plugin-defined assembly field, builder-style.
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.set_field(key, value);
        self
    }
}

/// Plugin config: the deployment-authored fragment of the system prompt.
///
/// Every field is optional; an absent field takes the documented default
/// (`true` for both include flags, `''` for both personas, lexicographic
/// tool order for `tool_order`). `tool_order` stays omitted rather than
/// defaulting to `[]`, because an explicit empty order lacks the rest
/// marker and must fail validation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Config {
    /// Include the fixed DeepSeek Harness identity before the deployment
    /// persona (default true).
    pub include_harness_identity: Option<bool>,
    /// Include optional runtime-context snapshots in model history
    /// (default true); required context remains.
    pub include_runtime_context: Option<bool>,
    /// Deployment-wide persona prefix template before first-party guidance.
    pub persona_prefix: Option<String>,
    /// Persona suffix template after first-party guidance.
    pub persona_suffix: Option<String>,
    /// Model-facing tool names in order, with [`TOOL_ORDER_REST`] exactly
    /// once. Omitted means lexicographic order.
    pub tool_order: Option<Vec<String>>,
}
