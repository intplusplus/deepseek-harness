//! The prompt registry: registration, canonical assembly, and refresh.
//!
//! TypeScript stores registrations in `ScopedLayers` of `NamedEntries` /
//! `AnonymousEntries` and reaches them through Cordis effect disposers.
//! Here each layer is a plain value inside one `Rc<RefCell<Registry>>`, and
//! a [`Registration`] guard removes exactly what it added when dropped —
//! the RAII shape of the TypeScript disposer. Drop a guard to dispose;
//! keep it alive to keep the contribution.
//!
//! Two collection semantics are ported deliberately:
//!
//! - **Variables live-iterate.** A provider registered by an earlier
//!   provider during the same assembly is visible to that assembly,
//!   mirroring the live `Map` iterator TypeScript iterates — including its
//!   detach rule: a table that drains mid-pass stops seeing insertions.
//! - **Tool providers snapshot.** Membership is captured before any
//!   provider runs, so a provider that registers another provider does not
//!   change the assembly in flight.
//!
//! Ordering is the part that must not drift: sections sort by
//! `(order, code-unit name)`, contexts by order, tools by configured order
//! with unlisted ones lexicographic at [`TOOL_ORDER_REST`]. The sort is
//! stable, so tools sharing a name keep collection order, and section /
//! context ties fall back to the name comparison instead of registration
//! order.

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use dsh_llm::ToolSchema;

use super::error::PromptError;
use super::orders::{
    PromptContextOrderName, PromptSectionOrderName, HARNESS_IDENTITY_SECTION,
    HARNESS_IDENTITY_TEXT, PERSONA_PREFIX_SECTION, PERSONA_SUFFIX_SECTION, TOOL_ORDER_REST,
};
use super::render::is_valid_variable_name;
use super::types::{
    AssembleContext, AssembledContext, AssembledSection, Config, PromptAssembly, PromptContext,
    PromptSection, PromptVariables, ToolProviderResult,
};
use super::waterfall::{run_assemble_waterfall, AssembleListener};

/// A variable provider, evaluated per assembly.
pub type VariableProvider = Rc<dyn Fn(&AssembleContext) -> Option<String>>;

/// A tool-schema provider, evaluated per assembly.
pub type ToolProvider = Rc<dyn Fn(&AssembleContext) -> ToolProviderResult>;

/// A `system-prompt/change` subscriber: no payload, because the TypeScript
/// notification takes none.
pub type ChangeListener = Rc<dyn Fn()>;

/// All prompt registrations owned by one global or scoped layer.
#[derive(Default)]
struct Layer {
    /// Named entries in registration order; insertion rejects duplicates.
    sections: Vec<(String, PromptSection)>,
    /// Named entries in registration order; insertion rejects duplicates.
    contexts: Vec<(String, PromptContext)>,
    /// Named entries with the live-iteration semantics of the TypeScript
    /// `NamedEntries` table; see [`VariableTable`].
    variables: VariableTable,
    /// Anonymous entries with an identity token for removal.
    tool_providers: Vec<(Rc<()>, ToolProvider)>,
    /// Active runtime-context suppressors; non-empty means suppressed.
    suppressors: usize,
}

impl Layer {
    /// Whether this layer owns no prompt registrations.
    ///
    /// TypeScript reclaims a scoped layer the moment it drains
    /// (`ScopedLayers.effect`'s disposer deletes an empty scope layer), which
    /// is what makes a scope's later re-registration start from a clean
    /// overlay instead of a table that remembers deleted slots.
    fn is_empty(&self) -> bool {
        self.sections.is_empty()
            && self.contexts.is_empty()
            && self.variables.slots.iter().all(Option::is_none)
            && self.tool_providers.is_empty()
            && self.suppressors == 0
    }
}

/// One layer's prompt variables, reproducing the iteration semantics of
/// the TypeScript `NamedEntries` table over a `Map`.
///
/// Deleting an entry leaves a hole instead of compacting the table, and
/// emptying it bumps the generation: TypeScript swaps in a fresh `Map` at
/// that moment, which detaches live iterators from every insertion that
/// follows. Assembly walks the slots the way the native `Map` iterator
/// walks the table — skipping holes, seeing appends, and stopping at a
/// generation change.
#[derive(Default)]
struct VariableTable {
    /// Insertion order with holes; `None` marks a deleted slot.
    slots: Vec<Option<(String, VariableProvider)>>,
    /// Bumped each time a delete empties the table.
    generation: u64,
}

impl VariableTable {
    /// Append one unique name; `false` when the name is already live.
    fn insert(&mut self, name: &str, provider: VariableProvider) -> bool {
        if self
            .slots
            .iter()
            .flatten()
            .any(|(existing, _)| existing == name)
        {
            return false;
        }
        self.slots.push(Some((name.to_string(), provider)));
        true
    }

    /// Remove one live name, bumping the generation when the table drains.
    fn remove(&mut self, name: &str) -> Option<(String, VariableProvider)> {
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.as_ref().is_some_and(|(existing, _)| existing == name))?;
        let removed = slot.take();
        if self.slots.iter().all(|slot| slot.is_none()) {
            self.generation += 1;
        }
        removed
    }
}

/// The global layer plus one layer per scope key.
///
/// TypeScript chains parent scopes through `ScopedLayers.chainLayers` over a
/// parent relation owned by `@deepseek-ai/dsh-scope`. That relation is
/// ported here as [`Registry::parents`]: a scope key is a plain `String`
/// instead of an object identity, but the resulting chain — and every rule
/// the system-prompt package states about it — is the same. Scoped entries
/// shadow globals by name, the nearest scope wins a name, and every layer in
/// the chain contributes tool providers.
struct Registry {
    global: Layer,
    scopes: BTreeMap<String, Layer>,
    /// Child scope to its enclosing scope. Only ever written through
    /// [`SystemPrompt::bind_scope_parent`], whose cycle check is what keeps
    /// every chain walk finite.
    parents: BTreeMap<String, String>,
    /// `system-prompt/change` subscribers, notified on every registration
    /// and every disposal.
    change_listeners: Vec<(Rc<()>, ChangeListener)>,
    /// `system-prompt/assemble` waterfall listeners with the scope tag they
    /// were registered under; `None` means the listener is untagged and
    /// receives every scope's assemblies.
    assemble_listeners: Vec<(Option<String>, Rc<()>, AssembleListener)>,
    tool_order: Option<Vec<String>>,
}

impl Registry {
    /// The layer registrations go into, creating a scope layer on first use.
    fn layer_for_insert(&mut self, scope: Option<&str>) -> &mut Layer {
        match scope {
            None => &mut self.global,
            Some(scope) => self.scopes.entry(scope.to_string()).or_default(),
        }
    }

    /// The layer to read or remove from, without creating one.
    fn layer_mut(&mut self, scope: Option<&str>) -> Option<&mut Layer> {
        match scope {
            None => Some(&mut self.global),
            Some(scope) => self.scopes.get_mut(scope),
        }
    }

    /// A scope's ancestry, nearest first: `[scope, parent, grandparent, ..]`.
    ///
    /// This is TypeScript's `scopeChainOf`. It is the one relation the
    /// scope package owns: registration views inherit DOWN the chain (a child
    /// scope sees its ancestors' layers), while event admission extends UP it
    /// (a listener tagged with an ancestor receives a descendant's events) —
    /// see [`SystemPrompt::assemble_listeners_for`].
    fn scope_chain(&self, scope: &str) -> Vec<String> {
        let mut chain = Vec::new();
        let mut cursor = Some(scope.to_string());
        while let Some(key) = cursor {
            chain.push(key.clone());
            cursor = self.parents.get(&key).cloned();
        }
        chain
    }

    /// Existing layers along the scope's ancestry, farthest ancestor first
    /// and the exact scope last, so a caller layering them in order gives
    /// the nearest scope the final word. Absent overlays are skipped.
    fn chain_layers(&self, scope: Option<&str>) -> Vec<&Layer> {
        let Some(scope) = scope else {
            return Vec::new();
        };
        self.scope_chain(scope)
            .iter()
            .rev()
            .filter_map(|key| self.scopes.get(key))
            .collect()
    }

    /// The same chain as [`Registry::chain_layers`], as keys.
    ///
    /// Callers that must release the registry borrow between steps — the
    /// variable walk, which runs a provider per entry — cannot hold the
    /// layer references across the gap, so they re-look-up by key instead.
    fn chain_layer_keys(&self, scope: Option<&str>) -> Vec<String> {
        let Some(scope) = scope else {
            return Vec::new();
        };
        self.scope_chain(scope)
            .into_iter()
            .rev()
            .filter(|key| self.scopes.contains_key(key))
            .collect()
    }

    /// Drop a scoped layer that drained, the way TypeScript's disposer
    /// reclaims an empty aggregate layer.
    fn reclaim_if_empty(&mut self, scope: Option<&str>) {
        let Some(scope) = scope else { return };
        let drained = self.scopes.get(scope).is_some_and(|layer| layer.is_empty());
        if drained {
            self.scopes.remove(scope);
        }
    }
}

/// What one [`Registration`] removes when dropped.
#[derive(Debug)]
enum RegistrationKind {
    Section {
        scope: Option<String>,
        name: String,
    },
    Context {
        scope: Option<String>,
        name: String,
    },
    Variable {
        scope: Option<String>,
        name: String,
    },
    ToolProvider {
        scope: Option<String>,
        token: Rc<()>,
    },
    Suppressor {
        scope: Option<String>,
    },
    ChangeListener {
        token: Rc<()>,
    },
    AssembleListener {
        scope: Option<String>,
        token: Rc<()>,
    },
}

/// A live prompt registration; dropping it removes the contribution, the
/// way calling the TypeScript effect disposer does.
///
/// Built-in sections (the harness identity and the deployment persona)
/// belong to the service itself and have no guard: they live as long as
/// the [`SystemPrompt`], matching their membership of the service fiber in
/// TypeScript.
pub struct Registration {
    registry: Rc<RefCell<Registry>>,
    kind: RegistrationKind,
}

impl fmt::Debug for Registration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registration")
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let scope = self.scope().cloned();
        // Every arm removes its entry inside a block, so the registry borrow
        // is released before the removed value is dropped: a provider may
        // hold another registration into this same registry, and dropping it
        // under the borrow would re-enter `borrow_mut`.
        match &self.kind {
            RegistrationKind::Section { scope, name } => {
                let removed = {
                    let mut registry = self.registry.borrow_mut();
                    registry
                        .layer_mut(scope.as_deref())
                        .and_then(|layer| take_named(&mut layer.sections, name))
                };
                drop(removed);
            }
            RegistrationKind::Context { scope, name } => {
                let removed = {
                    let mut registry = self.registry.borrow_mut();
                    registry
                        .layer_mut(scope.as_deref())
                        .and_then(|layer| take_named(&mut layer.contexts, name))
                };
                drop(removed);
            }
            RegistrationKind::Variable { scope, name } => {
                let removed = {
                    let mut registry = self.registry.borrow_mut();
                    registry
                        .layer_mut(scope.as_deref())
                        .and_then(|layer| layer.variables.remove(name))
                };
                drop(removed);
            }
            RegistrationKind::ToolProvider { scope, token } => {
                let removed = {
                    let mut registry = self.registry.borrow_mut();
                    registry
                        .layer_mut(scope.as_deref())
                        .and_then(|layer| take_token(&mut layer.tool_providers, token))
                };
                drop(removed);
            }
            RegistrationKind::Suppressor { scope } => {
                let mut registry = self.registry.borrow_mut();
                if let Some(layer) = registry.layer_mut(scope.as_deref()) {
                    layer.suppressors = layer.suppressors.saturating_sub(1);
                }
            }
            RegistrationKind::ChangeListener { token } => {
                let removed = {
                    let mut registry = self.registry.borrow_mut();
                    take_token(&mut registry.change_listeners, token)
                };
                drop(removed);
            }
            RegistrationKind::AssembleListener { token, .. } => {
                let removed = {
                    let mut registry = self.registry.borrow_mut();
                    let position = registry
                        .assemble_listeners
                        .iter()
                        .position(|(_, existing, _)| Rc::ptr_eq(existing, token));
                    position.map(|position| registry.assemble_listeners.remove(position))
                };
                drop(removed);
            }
        }
        // A drained scoped layer is reclaimed, then one change notification
        // is emitted for this disposal — TypeScript's `ScopedLayers.effect`
        // disposer does the same two things in the same order.
        {
            let mut registry = self.registry.borrow_mut();
            registry.reclaim_if_empty(scope.as_deref());
        }
        self.emit_change();
    }
}

impl Registration {
    /// The scope this registration was made under, when it belongs to one.
    fn scope(&self) -> Option<&String> {
        match &self.kind {
            RegistrationKind::Section { scope, .. }
            | RegistrationKind::Context { scope, .. }
            | RegistrationKind::Variable { scope, .. }
            | RegistrationKind::ToolProvider { scope, .. }
            | RegistrationKind::Suppressor { scope }
            | RegistrationKind::AssembleListener { scope, .. } => scope.as_ref(),
            RegistrationKind::ChangeListener { .. } => None,
        }
    }

    /// Notify every `system-prompt/change` subscriber that one registration
    /// changed, without holding the registry borrow while they run.
    fn emit_change(&self) {
        let listeners: Vec<ChangeListener> = {
            let registry = self.registry.borrow();
            registry
                .change_listeners
                .iter()
                .map(|(_, listener)| Rc::clone(listener))
                .collect()
        };
        for listener in listeners {
            listener();
        }
    }
}

/// Remove the one live entry with this name, if present.
fn take_named<T>(entries: &mut Vec<(String, T)>, name: &str) -> Option<(String, T)> {
    let position = entries.iter().position(|(existing, _)| existing == name)?;
    Some(entries.remove(position))
}

/// Remove the one anonymous entry carrying this identity token, if present.
fn take_token<T>(providers: &mut Vec<(Rc<()>, T)>, token: &Rc<()>) -> Option<(Rc<()>, T)> {
    let position = providers
        .iter()
        .position(|(existing, _)| Rc::ptr_eq(existing, token))?;
    Some(providers.remove(position))
}

/// The privileged handle to move one scope's parent link.
///
/// TypeScript grants re-linking only to the original binder, and only "while
/// nothing produced under the old parent is retained" — the blank-session
/// recompose contract — because the relation cannot see what a session
/// logged. The same restriction holds here: there is no public re-bind.
pub struct ScopeParentBinding {
    registry: Rc<RefCell<Registry>>,
    child: String,
}

impl ScopeParentBinding {
    /// Re-link the bound scope to a different parent, with the same cycle
    /// check as the bind.
    ///
    /// Returns `false` when the link would form a cycle; the existing link is
    /// then left in place.
    pub fn rebind(&self, parent: &str) -> bool {
        let mut registry = self.registry.borrow_mut();
        let mut cursor = Some(parent.to_string());
        while let Some(key) = cursor {
            if key == self.child {
                return false;
            }
            cursor = registry.parents.get(&key).cloned();
        }
        registry
            .parents
            .insert(self.child.clone(), parent.to_string());
        true
    }
}

impl fmt::Debug for ScopeParentBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopeParentBinding")
            .field("child", &self.child)
            .finish_non_exhaustive()
    }
}

/// Registry service for the prompt inputs assembled before each model step.
#[derive(Clone)]
pub struct SystemPrompt {
    registry: Rc<RefCell<Registry>>,
}

impl fmt::Debug for SystemPrompt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemPrompt").finish_non_exhaustive()
    }
}

impl SystemPrompt {
    /// Build the registry from a deployment config.
    ///
    /// Registers the built-in sections — the harness identity (unless
    /// disabled) and both deployment persona slots — and applies the
    /// configured runtime-context suppression. A `tool_order` that is
    /// present but invalid fails here, at load, exactly as
    /// `validateToolOrder` throws in the TypeScript constructor.
    pub fn new(config: Config) -> Result<Self, PromptError> {
        let tool_order = validate_tool_order(config.tool_order.as_deref())?;
        let mut global = Layer::default();
        if config.include_harness_identity.unwrap_or(true) {
            global.sections.push((
                HARNESS_IDENTITY_SECTION.to_string(),
                PromptSection::static_text(
                    HARNESS_IDENTITY_SECTION,
                    PromptSectionOrderName::HarnessIdentity.order(),
                    HARNESS_IDENTITY_TEXT,
                ),
            ));
        }
        global.sections.push((
            PERSONA_PREFIX_SECTION.to_string(),
            PromptSection::static_text(
                PERSONA_PREFIX_SECTION,
                PromptSectionOrderName::DeploymentPersonaPrefix.order(),
                config.persona_prefix.unwrap_or_default(),
            ),
        ));
        global.sections.push((
            PERSONA_SUFFIX_SECTION.to_string(),
            PromptSection::static_text(
                PERSONA_SUFFIX_SECTION,
                PromptSectionOrderName::DeploymentPersonaSuffix.order(),
                config.persona_suffix.unwrap_or_default(),
            ),
        ));
        if !config.include_runtime_context.unwrap_or(true) {
            global.suppressors += 1;
        }
        Ok(Self {
            registry: Rc::new(RefCell::new(Registry {
                global,
                scopes: BTreeMap::new(),
                parents: BTreeMap::new(),
                change_listeners: Vec::new(),
                assemble_listeners: Vec::new(),
                tool_order,
            })),
        })
    }

    /// Bind `parent` as `child`'s enclosing scope, once.
    ///
    /// A child that already has a parent fails: TypeScript has no open
    /// re-link path, so a scope's ancestry can only be moved by whoever
    /// holds the [`ScopeParentBinding`] returned here. A link that would
    /// close a cycle is rejected, because every chain consumer walks
    /// parents to the root and would never terminate.
    ///
    /// In TypeScript this relation belongs to `@deepseek-ai/dsh-scope` and
    /// keys are opaque objects; here a scope is a `String`, so the same
    /// relation is a name-to-name map.
    pub fn bind_scope_parent(
        &self,
        child: &str,
        parent: &str,
    ) -> Result<ScopeParentBinding, PromptError> {
        let mut registry = self.registry.borrow_mut();
        if registry.parents.contains_key(child) {
            return Err(PromptError::ScopeParentBound);
        }
        // Walk up from the prospective parent: reaching the child means the
        // link would close a cycle.
        let mut cursor = Some(parent.to_string());
        while let Some(key) = cursor {
            if key == child {
                return Err(PromptError::ScopeParentCycle);
            }
            cursor = registry.parents.get(&key).cloned();
        }
        registry
            .parents
            .insert(child.to_string(), parent.to_string());
        Ok(ScopeParentBinding {
            registry: Rc::clone(&self.registry),
            child: child.to_string(),
        })
    }

    /// A scope's ancestry, nearest first: `[scope, parent, grandparent, ..]`.
    ///
    /// Registration views inherit DOWN this chain — a child scope sees its
    /// ancestors' layers — while event admission extends UP it.
    pub fn scope_chain(&self, scope: &str) -> Vec<String> {
        self.registry.borrow().scope_chain(scope)
    }

    /// Subscribe to `system-prompt/change`.
    ///
    /// The notification carries no payload, exactly as the TypeScript event
    /// takes no arguments: it only says "a prompt provider changed", and it
    /// is unfiltered because a global change affects every scope.
    pub fn on_change(&self, listener: impl Fn() + 'static) -> Registration {
        let token = Rc::new(());
        self.registry
            .borrow_mut()
            .change_listeners
            .push((Rc::clone(&token), Rc::new(listener)));
        Registration {
            registry: Rc::clone(&self.registry),
            kind: RegistrationKind::ChangeListener { token },
        }
    }

    /// Register an untagged `system-prompt/assemble` waterfall listener.
    ///
    /// An untagged listener receives every scope's assemblies; a listener
    /// registered through [`SystemPrompt::on_assemble_in`] receives only
    /// assemblies for its own scope and its descendants, because events flow
    /// up the scope chain and never down.
    pub fn on_assemble(&self, listener: AssembleListener) -> Registration {
        self.insert_assemble_listener(None, listener)
    }

    /// Register an `system-prompt/assemble` waterfall listener owned by one
    /// scope: it receives assemblies dispatched to that scope or to any
    /// scope nested under it.
    pub fn on_assemble_in(&self, scope: &str, listener: AssembleListener) -> Registration {
        self.insert_assemble_listener(Some(scope.to_string()), listener)
    }

    fn insert_assemble_listener(
        &self,
        scope: Option<String>,
        listener: AssembleListener,
    ) -> Registration {
        let token = Rc::new(());
        self.registry.borrow_mut().assemble_listeners.push((
            scope.clone(),
            Rc::clone(&token),
            listener,
        ));
        Registration {
            registry: Rc::clone(&self.registry),
            kind: RegistrationKind::AssembleListener { scope, token },
        }
    }

    /// The waterfall listeners admitted for one scope, in registration
    /// order: untagged listeners, then those tagged with the scope itself or
    /// with any of its ancestors.
    fn assemble_listeners_for(&self, scope: Option<&str>) -> Vec<AssembleListener> {
        let registry = self.registry.borrow();
        let chain: Vec<String> = match scope {
            None => Vec::new(),
            Some(scope) => registry.scope_chain(scope),
        };
        registry
            .assemble_listeners
            .iter()
            .filter(|(tag, _, _)| match tag {
                None => true,
                Some(tag) => scope == Some(tag.as_str()) || chain.iter().any(|key| key == tag),
            })
            .map(|(_, _, listener)| Rc::clone(listener))
            .collect()
    }

    /// Dispose every registration owned by one scope's layer at once.
    ///
    /// This is the Rust shape of disposing a scope's Cordis fiber: the
    /// layer's entries are removed together and the (now empty) layer is
    /// reclaimed. One `system-prompt/change` notification is emitted per
    /// removed entry, because TypeScript's each-disposer-once rule makes the
    /// count observable to a host that records the notifications.
    ///
    /// Returns how many entries were removed.
    pub fn dispose_scope(&self, scope: &str) -> usize {
        let removed = {
            let mut registry = self.registry.borrow_mut();
            let Some(layer) = registry.scopes.remove(scope) else {
                return 0;
            };
            layer.sections.len()
                + layer.contexts.len()
                + layer.variables.slots.iter().flatten().count()
                + layer.tool_providers.len()
                + layer.suppressors
        };
        for _ in 0..removed {
            self.emit_change();
        }
        removed
    }

    /// Notify every `system-prompt/change` subscriber.
    ///
    /// The listener list is snapshotted and the borrow released before any
    /// listener runs, so a subscriber may register, dispose, or assemble
    /// without re-entering the registry.
    fn emit_change(&self) {
        let listeners: Vec<ChangeListener> = {
            let registry = self.registry.borrow();
            registry
                .change_listeners
                .iter()
                .map(|(_, listener)| Rc::clone(listener))
                .collect()
        };
        for listener in listeners {
            listener();
        }
    }

    /// Resolve the centrally owned placement of a repository prompt section.
    pub fn get_section_order(name: PromptSectionOrderName) -> f64 {
        name.order()
    }

    /// Resolve the centrally owned placement of a repository runtime context.
    pub fn get_context_order(name: PromptContextOrderName) -> f64 {
        name.order()
    }

    /// Register an ordered prompt section in the global layer. A scoped
    /// section shadows a global section with the same name; duplicates
    /// within one layer and non-finite orders fail.
    pub fn section(&self, section: PromptSection) -> Result<Registration, PromptError> {
        self.insert_section(None, section)
    }

    /// Register an ordered prompt section in one scope's layer.
    pub fn section_in(
        &self,
        scope: &str,
        section: PromptSection,
    ) -> Result<Registration, PromptError> {
        self.insert_section(Some(scope), section)
    }

    fn insert_section(
        &self,
        scope: Option<&str>,
        section: PromptSection,
    ) -> Result<Registration, PromptError> {
        if !section.order.is_finite() {
            return Err(PromptError::NonFiniteSectionOrder { name: section.name });
        }
        let name = section.name.clone();
        let mut registry = self.registry.borrow_mut();
        let layer = registry.layer_for_insert(scope);
        if layer.sections.iter().any(|(existing, _)| existing == &name) {
            return Err(match scope {
                Some(_) => PromptError::DuplicateScopedSection { name },
                None => PromptError::DuplicateSection { name },
            });
        }
        layer.sections.push((name.clone(), section));
        drop(registry);
        // One notification per registration, as `ScopedLayers.effect` does
        // after a successful action.
        self.emit_change();
        Ok(Registration {
            registry: Rc::clone(&self.registry),
            kind: RegistrationKind::Section {
                scope: scope.map(str::to_string),
                name,
            },
        })
    }

    /// Register ordered dynamic context in the global layer. Scoped entries
    /// shadow global entries with the same name.
    pub fn context(&self, context: PromptContext) -> Result<Registration, PromptError> {
        self.insert_context(None, context)
    }

    /// Register ordered dynamic context in one scope's layer.
    pub fn context_in(
        &self,
        scope: &str,
        context: PromptContext,
    ) -> Result<Registration, PromptError> {
        self.insert_context(Some(scope), context)
    }

    fn insert_context(
        &self,
        scope: Option<&str>,
        context: PromptContext,
    ) -> Result<Registration, PromptError> {
        if !context.order.is_finite() {
            return Err(PromptError::NonFiniteContextOrder { name: context.name });
        }
        let name = context.name.clone();
        let mut registry = self.registry.borrow_mut();
        let layer = registry.layer_for_insert(scope);
        if layer.contexts.iter().any(|(existing, _)| existing == &name) {
            return Err(match scope {
                Some(_) => PromptError::DuplicateScopedContext { name },
                None => PromptError::DuplicateContext { name },
            });
        }
        layer.contexts.push((name.clone(), context));
        drop(registry);
        self.emit_change();
        Ok(Registration {
            registry: Rc::clone(&self.registry),
            kind: RegistrationKind::Context {
                scope: scope.map(str::to_string),
                name,
            },
        })
    }

    /// Register a prompt variable in the global layer. Scoped values shadow
    /// globals; invalid or duplicate names fail. A provider may return
    /// `None`, but rendering a section that references that value then
    /// fails.
    pub fn variable(
        &self,
        name: &str,
        provider: impl Fn(&AssembleContext) -> Option<String> + 'static,
    ) -> Result<Registration, PromptError> {
        self.insert_variable(None, name, provider)
    }

    /// Register a prompt variable in one scope's layer.
    pub fn variable_in(
        &self,
        scope: &str,
        name: &str,
        provider: impl Fn(&AssembleContext) -> Option<String> + 'static,
    ) -> Result<Registration, PromptError> {
        self.insert_variable(Some(scope), name, provider)
    }

    fn insert_variable(
        &self,
        scope: Option<&str>,
        name: &str,
        provider: impl Fn(&AssembleContext) -> Option<String> + 'static,
    ) -> Result<Registration, PromptError> {
        if !is_valid_variable_name(name) {
            return Err(PromptError::InvalidVariableName {
                name: name.to_string(),
            });
        }
        let mut registry = self.registry.borrow_mut();
        let layer = registry.layer_for_insert(scope);
        if !layer
            .variables
            .insert(name, Rc::new(provider) as VariableProvider)
        {
            return Err(match scope {
                Some(_) => PromptError::DuplicateScopedVariable {
                    name: name.to_string(),
                },
                None => PromptError::DuplicateVariable {
                    name: name.to_string(),
                },
            });
        }
        drop(registry);
        self.emit_change();
        Ok(Registration {
            registry: Rc::clone(&self.registry),
            kind: RegistrationKind::Variable {
                scope: scope.map(str::to_string),
                name: name.to_string(),
            },
        })
    }

    /// Register a tool-schema provider in the global layer. Global and
    /// matching scoped providers both contribute; returning the reserved
    /// [`TOOL_ORDER_REST`] name makes assembly fail.
    pub fn tools(
        &self,
        provider: impl Fn(&AssembleContext) -> ToolProviderResult + 'static,
    ) -> Registration {
        self.insert_tool_provider(None, provider)
    }

    /// Register a tool-schema provider in one scope's layer.
    pub fn tools_in(
        &self,
        scope: &str,
        provider: impl Fn(&AssembleContext) -> ToolProviderResult + 'static,
    ) -> Registration {
        self.insert_tool_provider(Some(scope), provider)
    }

    fn insert_tool_provider(
        &self,
        scope: Option<&str>,
        provider: impl Fn(&AssembleContext) -> ToolProviderResult + 'static,
    ) -> Registration {
        let token = Rc::new(());
        let mut registry = self.registry.borrow_mut();
        registry
            .layer_for_insert(scope)
            .tool_providers
            .push((Rc::clone(&token), Rc::new(provider)));
        drop(registry);
        self.emit_change();
        Registration {
            registry: Rc::clone(&self.registry),
            kind: RegistrationKind::ToolProvider {
                scope: scope.map(str::to_string),
                token,
            },
        }
    }

    /// Suppress optional dynamic runtime-context contributions in the
    /// global layer without changing the services that own or enforce
    /// those facts. Multiple suppressors remain independently disposable.
    pub fn suppress_runtime_context(&self) -> Registration {
        self.insert_suppressor(None)
    }

    /// Suppress optional dynamic runtime-context contributions in one
    /// scope's layer.
    pub fn suppress_runtime_context_in(&self, scope: &str) -> Registration {
        self.insert_suppressor(Some(scope))
    }

    fn insert_suppressor(&self, scope: Option<&str>) -> Registration {
        let mut registry = self.registry.borrow_mut();
        registry.layer_for_insert(scope).suppressors += 1;
        drop(registry);
        self.emit_change();
        Registration {
            registry: Rc::clone(&self.registry),
            kind: RegistrationKind::Suppressor {
                scope: scope.map(str::to_string),
            },
        }
    }

    /// Assemble global and scoped providers, detach tool parameters, apply
    /// canonical ordering, then run the assembly waterfall.
    ///
    /// This is the whole TypeScript `assemble()`: variables resolve with
    /// scope-chain shadowing (the nearest scope wins a name), sections and
    /// contexts resolve with the same shadowing before the deterministic
    /// order sort, tools canonicalize against the configured order, and the
    /// cooperative waterfall runs on the result.
    ///
    /// The waterfall's value is authoritative except for two restorations
    /// TypeScript applies afterwards: an effective `complete` section is put
    /// back as the sole prompt section — listeners cannot add to or replace
    /// a complete prompt — and, under runtime-context suppression, contexts
    /// are cut back to the required ones, which drops any context the
    /// waterfall added.
    pub fn assemble(&self, context: &AssembleContext) -> Result<PromptAssembly, PromptError> {
        let scope = context.scope.as_deref();
        let tool_order = self.registry.borrow().tool_order.clone();
        let chain_keys = self.registry.borrow().chain_layer_keys(scope);
        let suppressed = is_suppressed(&self.registry, scope);

        // Scoped variables shadow globals; the map keeps one value per name.
        // The chain runs farthest ancestor first, so the nearest scope wins.
        let mut variables = PromptVariables::default();
        evaluate_variables(&self.registry, None, context, &mut variables);
        for key in &chain_keys {
            evaluate_variables(&self.registry, Some(key), context, &mut variables);
        }

        // Scoped sections and contexts shadow globals before the
        // deterministic order sort.
        let mut section_defs = merged_entries(&self.registry, scope, |layer| &layer.sections);
        let context_defs = merged_entries(&self.registry, scope, |layer| &layer.contexts);

        // Validate order against pre-restriction names while collecting
        // visible schemas.
        let providers = tool_provider_snapshot(&self.registry, scope);
        let mut collected: Vec<ToolSchema> = Vec::new();
        let mut known_names: BTreeSet<String> = BTreeSet::new();
        for provider in &providers {
            let result = provider(context);
            for schema in &result.schemas {
                collected.push(schema.clone());
            }
            match &result.known_names {
                Some(names) => known_names.extend(names.iter().cloned()),
                None => known_names.extend(result.schemas.iter().map(|tool| tool.name.clone())),
            }
        }

        section_defs.sort_by(|a, b| compare_prompt_sections(&a.1, &b.1));
        let complete: Vec<&PromptSection> = section_defs
            .iter()
            .map(|(_, section)| section)
            .filter(|section| section.complete)
            .collect();
        if complete.len() > 1 {
            let names = complete
                .iter()
                .map(|section| format!("{:?}", section.name))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(PromptError::MultipleCompleteSections { names });
        }
        let mut complete_section: Option<AssembledSection> = None;
        let mut sections = Vec::with_capacity(section_defs.len());
        for (_, def) in &section_defs {
            let assembled = AssembledSection {
                name: def.name.clone(),
                text: def.text.resolve(context),
                interpolate: def.interpolate,
            };
            if def.complete {
                complete_section = Some(assembled.clone());
            }
            sections.push(assembled);
        }

        // The required names come from the merged map, not the filtered
        // list: suppression is what the post-waterfall cut-back re-applies,
        // and a required context added by the waterfall is not one of them.
        let required_contexts: BTreeSet<String> = context_defs
            .iter()
            .map(|(_, entry)| entry)
            .filter(|entry| entry.required)
            .map(|entry| entry.name.clone())
            .collect();

        let mut contexts: Vec<&PromptContext> = context_defs
            .iter()
            .map(|(_, entry)| entry)
            .filter(|entry| !suppressed || entry.required)
            .collect();
        contexts.sort_by(|a, b| a.order.total_cmp(&b.order));
        let contexts: Vec<AssembledContext> = contexts
            .iter()
            .map(|entry| resolve_context(entry, context))
            .collect();

        let tools = order_tools(&collected, tool_order.as_deref(), &known_names)?;

        let assembly = PromptAssembly {
            sections,
            contexts,
            tools,
            variables,
        };

        // The waterfall runs on the assembled value; its result is
        // authoritative, then TypeScript's two restorations are applied.
        let listeners = self.assemble_listeners_for(scope);
        let transformed = run_assemble_waterfall(&listeners, context, assembly);
        if complete_section.is_none() && !suppressed {
            return Ok(transformed);
        }
        Ok(PromptAssembly {
            sections: match &complete_section {
                Some(complete_section) => vec![complete_section.clone()],
                None => transformed.sections,
            },
            contexts: if suppressed {
                transformed
                    .contexts
                    .into_iter()
                    .filter(|entry| required_contexts.contains(&entry.name))
                    .collect()
            } else {
                transformed.contexts
            },
            ..transformed
        })
    }

    /// Refresh accepted registered runtime facts for request admission.
    ///
    /// Contexts added only by the assembly waterfall retain their accepted
    /// values. Current suppression removes optional contexts, and missing
    /// required registrations are restored in registry order. Sections,
    /// tools, and interpolation variables retain the accepted assembly;
    /// their providers do not rerun.
    pub fn refresh_context(
        &self,
        assembly: &PromptAssembly,
        context: &AssembleContext,
    ) -> Result<PromptAssembly, PromptError> {
        let scope = context.scope.as_deref();
        let providers = merged_entries(&self.registry, scope, |layer| &layer.contexts);
        let suppressed = is_suppressed(&self.registry, scope);
        let mut required: Vec<(String, PromptContext)> = providers
            .iter()
            .filter(|(_, entry)| entry.required)
            .cloned()
            .collect();
        let mut contexts: Vec<AssembledContext> = Vec::with_capacity(assembly.contexts.len());
        for accepted in &assembly.contexts {
            let provider = providers
                .iter()
                .find(|(name, _)| name == &accepted.name)
                .map(|(_, entry)| entry);
            if let Some(position) = required.iter().position(|(name, _)| name == &accepted.name) {
                required.remove(position);
            }
            if suppressed && !provider.is_some_and(|entry| entry.required) {
                continue;
            }
            contexts.push(match provider {
                Some(entry) => resolve_context(entry, context),
                None => accepted.clone(),
            });
        }
        for (_, provider) in required {
            let position = contexts.iter().position(|entry| {
                providers
                    .iter()
                    .find(|(name, _)| name == &entry.name)
                    .is_some_and(|(_, next)| next.order > provider.order)
            });
            contexts.insert(
                position.unwrap_or(contexts.len()),
                resolve_context(&provider, context),
            );
        }
        Ok(PromptAssembly {
            contexts,
            ..assembly.clone()
        })
    }
}

/// Whether optional runtime context is suppressed for this scope.
///
/// Any layer in the chain can suppress: the global layer, or any ancestor
/// overlay — TypeScript tests `global.runtimeContextSuppressors` and then
/// every chained layer.
fn is_suppressed(registry: &Rc<RefCell<Registry>>, scope: Option<&str>) -> bool {
    let registry = registry.borrow();
    if registry.global.suppressors > 0 {
        return true;
    }
    registry
        .chain_layers(scope)
        .iter()
        .any(|layer| layer.suppressors > 0)
}

/// Evaluate one layer's variables the way TypeScript iterates its live
/// `Map`: a provider registered by an earlier provider in the same pass
/// is still seen, a hole left by a disposal is skipped, and a table that
/// drained mid-pass detaches the walk from everything inserted after.
fn evaluate_variables(
    registry: &Rc<RefCell<Registry>>,
    scope: Option<&str>,
    context: &AssembleContext,
    out: &mut PromptVariables,
) {
    let mut cursor = 0usize;
    let generation = {
        let registry = registry.borrow();
        let Some(layer) = layer_ref(&registry, scope) else {
            return;
        };
        layer.variables.generation
    };
    loop {
        // The borrow is released before the provider runs, so a provider
        // may register another variable — or dispose this one — safely.
        let next = {
            let registry = registry.borrow();
            let Some(layer) = layer_ref(&registry, scope) else {
                return;
            };
            let table = &layer.variables;
            if table.generation != generation {
                return;
            }
            let mut found = None;
            while cursor < table.slots.len() {
                if let Some(entry) = &table.slots[cursor] {
                    found = Some(entry.clone());
                    break;
                }
                cursor += 1;
            }
            found
        };
        match next {
            Some((name, provider)) => {
                out.insert(name, provider(context));
                cursor += 1;
            }
            None => return,
        }
    }
}

/// The layer to read for a scope, without creating one.
fn layer_ref<'a>(registry: &'a Registry, scope: Option<&str>) -> Option<&'a Layer> {
    match scope {
        None => Some(&registry.global),
        Some(scope) => registry.scopes.get(scope),
    }
}

/// Merge the global layer's named entries with every scope-chain layer's,
/// scoped entries shadowing globals by name while keeping the global
/// position.
///
/// That is `Map.set` semantics: overwriting a key does not move it, which
/// is what makes the later stable order sort deterministic. The chain is
/// walked farthest ancestor first, so the nearest scope's entry wins.
fn merged_entries<T: Clone>(
    registry: &Rc<RefCell<Registry>>,
    scope: Option<&str>,
    pick: impl for<'b> Fn(&'b Layer) -> &'b [(String, T)],
) -> Vec<(String, T)> {
    let registry = registry.borrow();
    let mut merged = pick(&registry.global).to_vec();
    for layer in registry.chain_layers(scope) {
        for (name, entry) in pick(layer) {
            match merged.iter_mut().find(|(existing, _)| existing == name) {
                Some(slot) => slot.1 = entry.clone(),
                None => merged.push((name.clone(), entry.clone())),
            }
        }
    }
    merged
}

/// Snapshot tool-provider membership for one assembly: global providers
/// first, then every chained scope layer's, farthest ancestor first, in
/// registration order. Every layer in the chain contributes.
fn tool_provider_snapshot(
    registry: &Rc<RefCell<Registry>>,
    scope: Option<&str>,
) -> Vec<ToolProvider> {
    let registry = registry.borrow();
    let mut providers: Vec<ToolProvider> = registry
        .global
        .tool_providers
        .iter()
        .map(|(_, provider)| Rc::clone(provider))
        .collect();
    for layer in registry.chain_layers(scope) {
        providers.extend(
            layer
                .tool_providers
                .iter()
                .map(|(_, provider)| Rc::clone(provider)),
        );
    }
    providers
}

/// Resolve one registered context against the assembly context.
fn resolve_context(entry: &PromptContext, context: &AssembleContext) -> AssembledContext {
    AssembledContext {
        name: entry.name.clone(),
        text: entry.text.resolve(context),
        interpolate: entry.interpolate,
    }
}

/// Order prompt sections by their explicit placement, then
/// deterministically by name.
fn compare_prompt_sections(a: &PromptSection, b: &PromptSection) -> Ordering {
    a.order
        .total_cmp(&b.order)
        .then_with(|| a.name.cmp(&b.name))
}

/// Validate duplicate names and the required [`TOOL_ORDER_REST`] marker.
/// Registered names are checked later, at assembly.
fn validate_tool_order(tool_order: Option<&[String]>) -> Result<Option<Vec<String>>, PromptError> {
    let Some(tool_order) = tool_order else {
        return Ok(None);
    };
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for name in tool_order {
        if !seen.insert(name.as_str()) {
            return Err(PromptError::DuplicateToolOrderEntry { name: name.clone() });
        }
    }
    if !seen.contains(TOOL_ORDER_REST) {
        return Err(PromptError::MissingRestEntry);
    }
    Ok(Some(tool_order.to_vec()))
}

/// Apply configured tool order, inserting unlisted tools lexicographically
/// at [`TOOL_ORDER_REST`]. Unknown configured names fail; known but
/// restricted names may be absent.
fn order_tools(
    tools: &[ToolSchema],
    tool_order: Option<&[String]>,
    known_names: &BTreeSet<String>,
) -> Result<Vec<ToolSchema>, PromptError> {
    if tools.iter().any(|tool| tool.name == TOOL_ORDER_REST) {
        return Err(PromptError::ReservedToolName);
    }
    let Some(tool_order) = tool_order else {
        let mut sorted = tools.to_vec();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        return Ok(sorted);
    };
    let unknown: Vec<&String> = tool_order
        .iter()
        .filter(|name| name.as_str() != TOOL_ORDER_REST && !known_names.contains(name.as_str()))
        .collect();
    if !unknown.is_empty() {
        let known = if known_names.is_empty() {
            "(none)".to_string()
        } else {
            known_names.iter().cloned().collect::<Vec<_>>().join(", ")
        };
        return Err(PromptError::UnknownToolOrderNames {
            plural: if unknown.len() > 1 { "s" } else { "" },
            names: unknown
                .iter()
                .map(|name| format!("{name:?}"))
                .collect::<Vec<_>>()
                .join(", "),
            known,
        });
    }
    let listed: BTreeSet<&str> = tool_order.iter().map(String::as_str).collect();
    let mut rest: Vec<ToolSchema> = tools
        .iter()
        .filter(|tool| !listed.contains(tool.name.as_str()))
        .cloned()
        .collect();
    rest.sort_by(|a, b| a.name.cmp(&b.name));
    let mut ordered = Vec::with_capacity(tools.len());
    for name in tool_order {
        if name == TOOL_ORDER_REST {
            ordered.append(&mut rest);
        } else {
            ordered.extend(tools.iter().filter(|tool| &tool.name == name).cloned());
        }
    }
    Ok(ordered)
}
