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

use std::cell::{RefCell, RefMut};
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

/// A variable provider, evaluated per assembly.
pub type VariableProvider = Rc<dyn Fn(&AssembleContext) -> Option<String>>;

/// A tool-schema provider, evaluated per assembly.
pub type ToolProvider = Rc<dyn Fn(&AssembleContext) -> ToolProviderResult>;

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
/// TypeScript chains parent scopes through `ScopedLayers.chainLayers`, a
/// rule owned by `@deepseek-ai/dsh-scope`; that walk is not ported. One
/// global layer and one named scope layer reproduce every rule the
/// system-prompt package itself states: scoped entries shadow globals by
/// name, and both layers' tool providers contribute.
struct Registry {
    global: Layer,
    scopes: BTreeMap<String, Layer>,
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
        let mut registry = self.registry.borrow_mut();
        match &self.kind {
            RegistrationKind::Section { scope, name } => release(
                registry
                    .layer_mut(scope.as_deref())
                    .and_then(|layer| take_named(&mut layer.sections, name)),
                registry,
            ),
            RegistrationKind::Context { scope, name } => release(
                registry
                    .layer_mut(scope.as_deref())
                    .and_then(|layer| take_named(&mut layer.contexts, name)),
                registry,
            ),
            RegistrationKind::Variable { scope, name } => release(
                registry
                    .layer_mut(scope.as_deref())
                    .and_then(|layer| layer.variables.remove(name)),
                registry,
            ),
            RegistrationKind::ToolProvider { scope, token } => release(
                registry
                    .layer_mut(scope.as_deref())
                    .and_then(|layer| take_token(&mut layer.tool_providers, token)),
                registry,
            ),
            RegistrationKind::Suppressor { scope } => {
                if let Some(layer) = registry.layer_mut(scope.as_deref()) {
                    layer.suppressors = layer.suppressors.saturating_sub(1);
                }
            }
        }
    }
}

/// Drop a removed entry only after its registry borrow is released.
///
/// A provider closure can hold another registration into this same
/// registry — the self-replacing-variable pattern — and dropping it
/// under the borrow would re-enter `borrow_mut` on a cell already held.
fn release<T>(removed: Option<T>, registry: RefMut<'_, Registry>) {
    drop(registry);
    drop(removed);
}

/// Remove the one live entry with this name, if present.
fn take_named<T>(entries: &mut Vec<(String, T)>, name: &str) -> Option<(String, T)> {
    let position = entries.iter().position(|(existing, _)| existing == name)?;
    Some(entries.remove(position))
}

/// Remove the one provider carrying this identity token, if present.
fn take_token(
    providers: &mut Vec<(Rc<()>, ToolProvider)>,
    token: &Rc<()>,
) -> Option<(Rc<()>, ToolProvider)> {
    let position = providers
        .iter()
        .position(|(existing, _)| Rc::ptr_eq(existing, token))?;
    Some(providers.remove(position))
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
                tool_order,
            })),
        })
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
        Registration {
            registry: Rc::clone(&self.registry),
            kind: RegistrationKind::Suppressor {
                scope: scope.map(str::to_string),
            },
        }
    }

    /// Assemble global and scoped providers, detach tool parameters, and
    /// apply canonical ordering.
    ///
    /// This is the TypeScript `assemble()` up to the waterfall: variables
    /// resolve with scoped shadowing, sections and contexts resolve with
    /// scoped shadowing and order sorting, tools canonicalize against the
    /// configured order, and a single effective `complete` section becomes
    /// the sole prompt. The post-waterfall restorations (a complete section
    /// replacing whatever the waterfall produced; required-only contexts
    /// under suppression) are folded in, because without a waterfall the
    /// registry's own output already satisfies them.
    pub fn assemble(&self, context: &AssembleContext) -> Result<PromptAssembly, PromptError> {
        let scope = context.scope.as_deref();
        let tool_order = self.registry.borrow().tool_order.clone();
        let suppressed = is_suppressed(&self.registry, scope);

        // Scoped variables shadow globals; the map keeps one value per name.
        let mut variables = PromptVariables::default();
        evaluate_variables(&self.registry, None, context, &mut variables);
        if let Some(scope) = scope {
            evaluate_variables(&self.registry, Some(scope), context, &mut variables);
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

        let mut assembly = PromptAssembly {
            sections,
            contexts,
            tools,
            variables,
        };
        if let Some(complete_section) = complete_section {
            assembly.sections = vec![complete_section];
        }
        Ok(assembly)
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
fn is_suppressed(registry: &Rc<RefCell<Registry>>, scope: Option<&str>) -> bool {
    let registry = registry.borrow();
    if registry.global.suppressors > 0 {
        return true;
    }
    scope
        .and_then(|scope| registry.scopes.get(scope))
        .is_some_and(|layer| layer.suppressors > 0)
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

/// Merge one layer's named entries with the scope layer's, scoped entries
/// shadowing globals by name while keeping the global position.
///
/// That is `Map.set` semantics: overwriting a key does not move it, which
/// is what makes the later stable order sort deterministic.
fn merged_entries<T: Clone>(
    registry: &Rc<RefCell<Registry>>,
    scope: Option<&str>,
    pick: impl for<'b> Fn(&'b Layer) -> &'b [(String, T)],
) -> Vec<(String, T)> {
    let registry = registry.borrow();
    let mut merged = pick(&registry.global).to_vec();
    if let Some(scope) = scope {
        if let Some(layer) = registry.scopes.get(scope) {
            for (name, entry) in pick(layer) {
                match merged.iter_mut().find(|(existing, _)| existing == name) {
                    Some(slot) => slot.1 = entry.clone(),
                    None => merged.push((name.clone(), entry.clone())),
                }
            }
        }
    }
    merged
}

/// Snapshot tool-provider membership for one assembly: global providers
/// first, then the scope's, in registration order.
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
    if let Some(scope) = scope {
        if let Some(layer) = registry.scopes.get(scope) {
            providers.extend(
                layer
                    .tool_providers
                    .iter()
                    .map(|(_, provider)| Rc::clone(provider)),
            );
        }
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
