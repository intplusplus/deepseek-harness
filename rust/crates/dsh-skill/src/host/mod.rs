//! The host layer: provider registry, scope chains, and the collect cache.
//!
//! TypeScript implements this in the Cordis `SkillRegistry` service
//! (`packages/skill/skill/src/index.ts`). The port keeps the registry's
//! observable rules and replaces its async machinery:
//!
//! - **Registration is an RAII guard.** The TypeScript effect disposer becomes
//!   a [`ProviderRegistration`] / [`RuntimeRegistration`] guard in the shape
//!   of `dsh-system-prompt`'s `Registration`: dropping it removes exactly
//!   what it added, reclaims a drained scope layer, and invalidates caches.
//! - **Discovery is synchronous.** Providers are sync [`SkillProvider`] trait
//!   objects; the abort-signal race points of the TypeScript registry have no
//!   await to hang on and are not ported.
//! - **Merging reuses the pure layer.** Per-layer duplicate resolution goes
//!   through [`crate::merge::resolve_layer`], the cross-layer overlay through
//!   [`crate::merge::merge_layers`], runtime contributions through
//!   [`crate::merge::runtime_layer`], and catalog ordering through
//!   [`crate::merge::sort_summaries`]. The host never re-implements a rank
//!   or shadowing rule.
//! - **The collect cache is keyed by registry content fingerprint.** The key
//!   is `(cwd, scope chain, revision)` exactly like the TypeScript
//!   `collectCacheKey`; the revision counter is the fingerprint of registry
//!   content, bumped by every registration, disposal, provider invalidation,
//!   and runtime mutation. Provider content that changes without such a
//!   mutation must call [`ProviderControl::invalidate`], the same contract
//!   the TypeScript `SkillProviderControl` states. Incomplete observations
//!   are never cached, a mid-collect invalidation retries once and then
//!   leaves the result uncached, and eviction removes the oldest entry past
//!   the configured maximum.
//!
//! Scope keys are plain strings with a parent relation bound through
//! [`SkillHost::bind_scope_parent`], mirroring `dsh-system-prompt`'s port of
//! `@deepseek-ai/dsh-scope`. A read merges the global layer with the viewing
//! scope's chain; the nearest layer's entry wins a duplicate name outright.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::{Rc, Weak};

use crate::merge::{
    merge_layers, resolve_layer, runtime_layer, sort_summaries, IndexedCandidate, RUNTIME_PROVIDER,
};
use crate::name::{is_skill_name, SkillName};
use crate::types::{SkillCandidate, SkillDefinition, SkillRegistration, SkillSummary};
use crate::validate::{validate_candidate, validate_definition, SkillError};

mod filesystem;
mod provider;

pub use filesystem::{
    is_potential_skill_path, is_relevant_watch_event, FsEntry, FsEntryKind, FsProviderConfig,
    FsSkillProvider, LocalLocator, SkillFiles, SkillRoot, StdFiles, WatchEvent,
    PROJECT_AGENTS_RANK, PROJECT_DSH_RANK, USER_AGENTS_RANK, USER_DSH_RANK,
};
pub use provider::{ProviderError, ProviderObservation, SkillLookupOptions, SkillProvider};

/// Maximum collect attempts when a provider invalidates mid-discovery,
/// matching the TypeScript `MAX_COLLECT_ATTEMPTS`.
const MAX_COLLECT_ATTEMPTS: usize = 2;

/// Default completed catalogs kept in memory, matching the TypeScript
/// `DEFAULT_COLLECT_CACHE_ENTRIES`.
const DEFAULT_COLLECT_CACHE_ENTRIES: usize = 128;

/// Host registry configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostConfig {
    /// Maximum number of completed cwd/provider catalogs kept in memory.
    pub collect_cache_max_entries: usize,
}

impl Default for HostConfig {
    fn default() -> Self {
        Self {
            collect_cache_max_entries: DEFAULT_COLLECT_CACHE_ENTRIES,
        }
    }
}

/// A host operation that cannot be completed.
#[derive(Clone, Debug, PartialEq)]
pub enum HostError {
    /// A provider with this name is already registered in the target layer;
    /// provider-name uniqueness is per layer, not global.
    DuplicateProvider {
        /// The layer the duplicate registration targeted, when scoped.
        scope: Option<String>,
        /// The rejected provider name.
        name: String,
    },
    /// The provider name is reserved for runtime skill registrations.
    ReservedProviderName {
        /// The rejected provider name.
        name: String,
    },
    /// Binding the parent would make the scope chain cyclic.
    ScopeCycle {
        /// The child scope of the rejected binding.
        child: String,
        /// The parent scope of the rejected binding.
        parent: String,
    },
    /// A configuration value is out of range.
    InvalidConfig {
        /// The rejected configuration field.
        field: &'static str,
    },
    /// A provider candidate or loaded definition failed validation.
    Candidate(SkillError),
    /// A provider load failed.
    Load(ProviderError),
}

impl fmt::Display for HostError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateProvider { scope, name } => write!(
                formatter,
                "a skill provider named \"{name}\" is already registered{}",
                if scope.is_some() {
                    " in this scope"
                } else {
                    ""
                }
            ),
            Self::ReservedProviderName { name } => write!(
                formatter,
                "\"{name}\" is reserved for runtime skill registrations"
            ),
            Self::ScopeCycle { child, parent } => write!(
                formatter,
                "binding scope \"{child}\" to parent \"{parent}\" would create a cycle"
            ),
            Self::InvalidConfig { field } => write!(
                formatter,
                "skill: {field} must be an integer greater than or equal to 1"
            ),
            Self::Candidate(error) => write!(formatter, "{error}"),
            Self::Load(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for HostError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Candidate(error) => Some(error),
            Self::Load(error) => Some(error),
            _ => None,
        }
    }
}

/// What loads a collected candidate.
///
/// The registry merges runtime contributions and provider candidates in one
/// pass, so both ride the same candidate type with the owner as locator.
#[derive(Clone)]
enum Owner<L> {
    /// A runtime registration; the definition is the body.
    Runtime(Box<SkillDefinition>),
    /// A provider-backed candidate with its opaque discovery locator.
    Provider {
        /// The provider that owns the skill body.
        provider: Rc<dyn SkillProvider<L>>,
        /// The opaque locator handed back to the provider on load.
        locator: L,
    },
}

impl<L> fmt::Debug for Owner<L> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(definition) => {
                formatter.debug_tuple("Runtime").field(definition).finish()
            }
            Self::Provider { .. } => formatter.debug_struct("Provider").finish_non_exhaustive(),
        }
    }
}

/// The `skills/change` subscriber table.
type ChangeListeners = Vec<(Rc<()>, Rc<dyn Fn()>)>;

/// The listeners to notify once no registry borrow is held.
type ChangeNotifications = Vec<Rc<dyn Fn()>>;

/// One collect pass's merged catalog plus whether it may be cached.
type Collected<L> = (Vec<SkillCandidate<Owner<L>>>, bool);

/// One provider registration retained by its layer.
struct RegisteredProvider<L> {
    provider: Rc<dyn SkillProvider<L>>,
    /// Service-wide monotonic registration order, the within-layer rank
    /// tiebreak.
    order: u64,
    /// Identity token proving the exact registration is still live.
    token: Rc<()>,
}

/// One scope's complete skill-registry contribution.
struct SkillLayer<L> {
    /// Providers registered through this scope, insertion-ordered; insertion
    /// rejects duplicate names.
    providers: Vec<(String, RegisteredProvider<L>)>,
    /// Runtime skills registered through this scope, insertion-ordered;
    /// first registration for a name wins.
    runtime: Vec<(SkillName, SkillDefinition)>,
}

impl<L> Default for SkillLayer<L> {
    fn default() -> Self {
        Self {
            providers: Vec::new(),
            runtime: Vec::new(),
        }
    }
}

impl<L> SkillLayer<L> {
    /// Whether every contribution table in this aggregate layer is empty.
    fn is_empty(&self) -> bool {
        self.providers.is_empty() && self.runtime.is_empty()
    }
}

/// The cache key: cwd, scope chain, and the registry content fingerprint.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CollectKey {
    cwd: Option<String>,
    chain: Vec<String>,
    revision: u64,
}

type CachedCollect<L> = (CollectKey, Vec<SkillCandidate<Owner<L>>>);

/// The mutable registry state behind one [`SkillHost`].
struct HostInner<L> {
    config: HostConfig,
    global: SkillLayer<L>,
    scopes: BTreeMap<String, SkillLayer<L>>,
    /// Child scope to its enclosing scope; only ever written through
    /// [`SkillHost::bind_scope_parent`], whose cycle check keeps every chain
    /// walk finite.
    parents: BTreeMap<String, String>,
    collect_cache: Vec<CachedCollect<L>>,
    revision: u64,
    next_provider_order: u64,
    change_listeners: ChangeListeners,
    warnings: Vec<String>,
}

impl<L> HostInner<L> {
    /// The layer to read or remove from, without creating one.
    fn layer_mut(&mut self, scope: Option<&str>) -> Option<&mut SkillLayer<L>> {
        match scope {
            None => Some(&mut self.global),
            Some(scope) => self.scopes.get_mut(scope),
        }
    }

    /// The layer registrations go into, creating a scope layer on first use.
    fn layer_for_insert(&mut self, scope: Option<&str>) -> &mut SkillLayer<L> {
        match scope {
            None => &mut self.global,
            Some(scope) => self.scopes.entry(scope.to_string()).or_default(),
        }
    }

    /// A scope's ancestry, nearest first: `[scope, parent, grandparent, ..]`.
    fn scope_chain(&self, scope: Option<&str>) -> Vec<String> {
        let Some(scope) = scope else {
            return Vec::new();
        };
        let mut chain = vec![scope.to_string()];
        let mut cursor = self.parents.get(scope).cloned();
        while let Some(key) = cursor {
            chain.push(key.clone());
            cursor = self.parents.get(&key).cloned();
        }
        chain
    }

    /// Existing layers along the scope's ancestry, farthest ancestor first
    /// and the exact scope last, so a caller layering them in order gives
    /// the nearest scope the final word.
    fn chain_layer_keys(&self, scope: Option<&str>) -> Vec<String> {
        self.scope_chain(scope)
            .iter()
            .rev()
            .filter(|key| self.scopes.contains_key(*key))
            .cloned()
            .collect()
    }

    /// Drop a scoped layer that drained, the way the TypeScript disposer
    /// reclaims an empty aggregate layer.
    fn reclaim_if_empty(&mut self, scope: Option<&str>) {
        let Some(scope) = scope else { return };
        let drained = self.scopes.get(scope).is_some_and(|layer| layer.is_empty());
        if drained {
            self.scopes.remove(scope);
        }
    }

    /// Bump the revision and drop completed catalogs, returning the change
    /// listeners to notify once no borrow is held.
    fn invalidate(&mut self) -> ChangeNotifications {
        self.revision += 1;
        self.collect_cache.clear();
        self.change_listeners
            .iter()
            .map(|(_, listener)| Rc::clone(listener))
            .collect()
    }

    fn cache_get(&self, key: &CollectKey) -> Option<&Vec<SkillCandidate<Owner<L>>>> {
        self.collect_cache
            .iter()
            .find(|(cached, _)| cached == key)
            .map(|(_, entries)| entries)
    }

    fn cache_put(&mut self, key: CollectKey, entries: Vec<SkillCandidate<Owner<L>>>) {
        self.collect_cache.push((key, entries));
        while self.collect_cache.len() > self.config.collect_cache_max_entries {
            self.collect_cache.remove(0);
        }
    }

    /// Whether a provider registration with this exact token is live.
    fn has_provider_token(&self, token: &Rc<()>) -> bool {
        self.global
            .providers
            .iter()
            .any(|(_, registered)| Rc::ptr_eq(&registered.token, token))
            || self.scopes.values().any(|layer| {
                layer
                    .providers
                    .iter()
                    .any(|(_, registered)| Rc::ptr_eq(&registered.token, token))
            })
    }

    /// Whether this exact provider registration is live anywhere.
    fn has_provider(&self, provider: &Rc<dyn SkillProvider<L>>) -> bool {
        self.global
            .providers
            .iter()
            .any(|(_, registered)| Rc::ptr_eq(&registered.provider, provider))
            || self.scopes.values().any(|layer| {
                layer
                    .providers
                    .iter()
                    .any(|(_, registered)| Rc::ptr_eq(&registered.provider, provider))
            })
    }
}

/// One layer's discovery inputs, snapshotted before any provider runs.
struct LayerInput<L> {
    runtime: Vec<SkillDefinition>,
    providers: Vec<(Rc<dyn SkillProvider<L>>, u64)>,
}

/// The invocation-neutral catalog plus whether discovery completed within a
/// stable registry revision.
#[derive(Clone, Debug)]
pub struct CatalogSnapshot {
    /// Sorted summaries collected in this observation.
    pub skills: Vec<SkillSummary>,
    /// Whether every registered provider completed without a concurrent
    /// registry revision. Incomplete snapshots are never cached.
    pub complete: bool,
}

/// Layered registry of skill providers.
///
/// A registration files into the layer of its scope: `None` registers
/// globally, a scoped registration lands in that scope's layer. A read
/// merges the global layer with the viewing scope's chain — the nearest
/// layer's entry wins a duplicate name outright, and rank decides duplicates
/// only within one layer.
pub struct SkillHost<L> {
    inner: Rc<RefCell<HostInner<L>>>,
}

impl<L: Clone + 'static> Default for SkillHost<L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: Clone + 'static> SkillHost<L> {
    /// A host with the default configuration.
    pub fn new() -> Self {
        Self::with_config(HostConfig::default()).expect("the default config is valid")
    }

    /// A host with an explicit configuration.
    ///
    /// # Errors
    /// [`HostError::InvalidConfig`] when `collect_cache_max_entries` is zero.
    pub fn with_config(config: HostConfig) -> Result<Self, HostError> {
        if config.collect_cache_max_entries < 1 {
            return Err(HostError::InvalidConfig {
                field: "collectCacheMaxEntries",
            });
        }
        Ok(Self {
            inner: Rc::new(RefCell::new(HostInner {
                config,
                global: SkillLayer::default(),
                scopes: BTreeMap::new(),
                parents: BTreeMap::new(),
                collect_cache: Vec::new(),
                revision: 0,
                next_provider_order: 0,
                change_listeners: Vec::new(),
                warnings: Vec::new(),
            })),
        })
    }

    /// Bind a child scope to a parent scope, extending the inheritance chain.
    ///
    /// Rebinding overwrites the previous parent; a binding that would make
    /// the chain cyclic is rejected. The chain is part of the collect cache
    /// key rather than an invalidation trigger, matching the TypeScript
    /// comment on `collectCacheKey`.
    ///
    /// # Errors
    /// [`HostError::ScopeCycle`] when the binding would create a cycle.
    pub fn bind_scope_parent(&self, child: &str, parent: &str) -> Result<(), HostError> {
        let mut inner = self.inner.borrow_mut();
        if child == parent {
            return Err(HostError::ScopeCycle {
                child: child.to_string(),
                parent: parent.to_string(),
            });
        }
        let mut cursor = Some(parent.to_string());
        while let Some(key) = cursor {
            if key == child {
                return Err(HostError::ScopeCycle {
                    child: child.to_string(),
                    parent: parent.to_string(),
                });
            }
            cursor = inner.parents.get(&key).cloned();
        }
        inner.parents.insert(child.to_string(), parent.to_string());
        Ok(())
    }

    /// Register a provider into the given layer.
    ///
    /// The factory receives this registration's invalidation control; the
    /// control's `invalidate` only applies while the exact registration
    /// stays live. The returned guard unregisters the provider and
    /// invalidates catalog caches when dropped.
    ///
    /// # Errors
    /// [`HostError::ReservedProviderName`] for the runtime provider name and
    /// [`HostError::DuplicateProvider`] when the name is already registered
    /// in the target layer.
    pub fn register_provider(
        &self,
        scope: Option<&str>,
        create: impl FnOnce(ProviderControl<L>) -> Rc<dyn SkillProvider<L>>,
    ) -> Result<ProviderRegistration<L>, HostError> {
        let token = Rc::new(());
        let control = ProviderControl {
            host: Rc::downgrade(&self.inner),
            token: Rc::downgrade(&token),
        };
        let provider = create(control);
        let name = provider.name().to_string();
        if name == RUNTIME_PROVIDER {
            return Err(HostError::ReservedProviderName { name });
        }
        let listeners = {
            let mut inner = self.inner.borrow_mut();
            let order = inner.next_provider_order;
            inner.next_provider_order += 1;
            {
                let layer = inner.layer_for_insert(scope);
                if layer
                    .providers
                    .iter()
                    .any(|(existing, _)| existing == &name)
                {
                    return Err(HostError::DuplicateProvider {
                        scope: scope.map(str::to_string),
                        name,
                    });
                }
                layer.providers.push((
                    name.clone(),
                    RegisteredProvider {
                        provider,
                        order,
                        token: Rc::clone(&token),
                    },
                ));
            }
            inner.invalidate()
        };
        for listener in listeners {
            listener();
        }
        Ok(ProviderRegistration {
            host: Rc::downgrade(&self.inner),
            scope: scope.map(str::to_string),
            name,
            token,
        })
    }

    /// Register a runtime skill into the given layer.
    ///
    /// Project entries outrank runtime entries, which outrank user entries,
    /// within one layer. Same-name runtime entries in one layer are
    /// first-wins: a duplicate records a warning and yields a no-op guard so
    /// it cannot remove the winner.
    ///
    /// # Errors
    /// The [`SkillError`] of [`SkillRegistration::into_definition`].
    pub fn register_runtime(
        &self,
        scope: Option<&str>,
        registration: SkillRegistration,
    ) -> Result<RuntimeRegistration<L>, SkillError> {
        let definition = registration.into_definition()?;
        let name = definition.summary.name.clone();
        let (listeners, active) = {
            let mut inner = self.inner.borrow_mut();
            let duplicate = inner
                .layer_mut(scope)
                .is_some_and(|layer| layer.runtime.iter().any(|(existing, _)| existing == &name));
            if duplicate {
                inner.warnings.push(format!(
                    "runtime skill \"{name}\" ignored because it is already registered"
                ));
                (Vec::new(), false)
            } else {
                inner
                    .layer_for_insert(scope)
                    .runtime
                    .push((name.clone(), definition));
                (inner.invalidate(), true)
            }
        };
        for listener in listeners {
            listener();
        }
        Ok(RuntimeRegistration {
            host: Rc::downgrade(&self.inner),
            scope: scope.map(str::to_string),
            name,
            active,
        })
    }

    /// Subscribe to catalog invalidations. The listener runs on every
    /// registration, disposal, and provider invalidation. Dropping the
    /// returned guard unsubscribes.
    pub fn on_change(&self, listener: impl Fn() + 'static) -> ChangeRegistration<L> {
        let token = Rc::new(());
        self.inner
            .borrow_mut()
            .change_listeners
            .push((Rc::clone(&token), Rc::new(listener)));
        ChangeRegistration {
            host: Rc::downgrade(&self.inner),
            token,
        }
    }

    /// Drain the warnings recorded so far: skipped providers, ignored
    /// duplicate runtime skills.
    pub fn take_warnings(&self) -> Vec<String> {
        std::mem::take(&mut self.inner.borrow_mut().warnings)
    }

    /// List invocation-neutral skill summaries for the viewing scope.
    ///
    /// # Errors
    /// [`HostError::Candidate`] when a provider returned an invalid
    /// candidate.
    pub fn list(
        &self,
        scope: Option<&str>,
        options: &SkillLookupOptions,
    ) -> Result<Vec<SkillSummary>, HostError> {
        Ok(self.snapshot(scope, options)?.skills)
    }

    /// Observe the current catalog and whether discovery completed within a
    /// stable revision.
    ///
    /// # Errors
    /// [`HostError::Candidate`] when a provider returned an invalid
    /// candidate.
    pub fn snapshot(
        &self,
        scope: Option<&str>,
        options: &SkillLookupOptions,
    ) -> Result<CatalogSnapshot, HostError> {
        let (entries, complete) = self.collect(scope, options)?;
        let mut skills: Vec<SkillSummary> =
            entries.into_iter().map(|entry| entry.summary).collect();
        sort_summaries(&mut skills);
        Ok(CatalogSnapshot { skills, complete })
    }

    /// Load and validate the winning candidate, passing its opaque discovery
    /// locator back to the provider.
    ///
    /// # Errors
    /// [`HostError::Load`] when the provider load fails and
    /// [`HostError::Candidate`] when the loaded definition fails validation.
    pub fn get(
        &self,
        name: &str,
        scope: Option<&str>,
        options: &SkillLookupOptions,
    ) -> Result<Option<SkillDefinition>, HostError> {
        if !is_skill_name(name) {
            return Ok(None);
        }
        let skill_name = SkillName::new(name).expect("the grammar check passed");
        let (entries, _) = self.collect(scope, options)?;
        let Some(entry) = entries
            .into_iter()
            .find(|entry| entry.summary.name == skill_name)
        else {
            return Ok(None);
        };
        match entry.locator {
            Owner::Runtime(definition) => {
                validate_definition(&definition).map_err(HostError::Candidate)?;
                Ok(Some(*definition))
            }
            Owner::Provider { provider, locator } => {
                let candidate = SkillCandidate {
                    summary: entry.summary.clone(),
                    rank: entry.rank,
                    locator,
                    metadata: entry.metadata.clone(),
                };
                let Some(definition) =
                    provider.get(&candidate, options).map_err(HostError::Load)?
                else {
                    return Ok(None);
                };
                validate_definition(&definition).map_err(HostError::Candidate)?;
                if definition.summary.name != candidate.summary.name {
                    // A stale definition load invalidates the catalog, but
                    // only while the exact registration that produced the
                    // entry is still live.
                    self.invalidate_entry(&provider);
                    return Ok(None);
                }
                Ok(Some(definition))
            }
        }
    }

    /// Collect the merged catalog for the viewing scope.
    fn collect(
        &self,
        scope: Option<&str>,
        options: &SkillLookupOptions,
    ) -> Result<Collected<L>, HostError> {
        let mut attempt = 1usize;
        loop {
            let (revision, key) = {
                let inner = self.inner.borrow();
                let chain = inner.scope_chain(scope);
                (
                    inner.revision,
                    CollectKey {
                        cwd: options.cwd.clone(),
                        chain,
                        revision: inner.revision,
                    },
                )
            };
            if let Some(cached) = self.inner.borrow().cache_get(&key) {
                return Ok((cached.clone(), true));
            }

            // Snapshot the layer inputs, release the borrow, then let
            // providers run: a provider may re-enter the host through its
            // control, and a mid-collect invalidation is handled by the
            // revision check below.
            let layer_inputs = {
                let inner = self.inner.borrow();
                let mut inputs = vec![LayerInput {
                    runtime: inner
                        .global
                        .runtime
                        .iter()
                        .map(|(_, definition)| definition.clone())
                        .collect(),
                    providers: inner
                        .global
                        .providers
                        .iter()
                        .map(|(_, registered)| (Rc::clone(&registered.provider), registered.order))
                        .collect(),
                }];
                for key in inner.chain_layer_keys(scope) {
                    let layer = &inner.scopes[&key];
                    inputs.push(LayerInput {
                        runtime: layer
                            .runtime
                            .iter()
                            .map(|(_, definition)| definition.clone())
                            .collect(),
                        providers: layer
                            .providers
                            .iter()
                            .map(|(_, registered)| {
                                (Rc::clone(&registered.provider), registered.order)
                            })
                            .collect(),
                    });
                }
                inputs
            };

            let mut merged_layers = Vec::new();
            let mut cacheable = true;
            for input in &layer_inputs {
                let (winners, complete) = self.collect_layer(input, options)?;
                if !complete {
                    cacheable = false;
                }
                merged_layers.push(winners);
            }
            let entries = merge_layers(merged_layers);

            if self.inner.borrow().revision != revision {
                if attempt < MAX_COLLECT_ATTEMPTS {
                    attempt += 1;
                    continue;
                }
                return Ok((entries, false));
            }
            if cacheable {
                self.inner.borrow_mut().cache_put(key, entries.clone());
            }
            return Ok((entries, cacheable));
        }
    }

    /// Resolve one layer's runtime contributions and provider candidates.
    fn collect_layer(
        &self,
        input: &LayerInput<L>,
        options: &SkillLookupOptions,
    ) -> Result<Collected<L>, HostError> {
        let mut indexed: Vec<IndexedCandidate<Owner<L>>> = Vec::new();
        for entry in runtime_layer(input.runtime.clone()) {
            indexed.push(IndexedCandidate::new(
                entry
                    .candidate
                    .map_locator(|definition| Owner::Runtime(Box::new(definition))),
                entry.provider_order,
                entry.local_order,
            ));
        }
        let mut cacheable = true;
        for (provider, order) in &input.providers {
            let mut local_order = 0u64;
            match provider.list(options) {
                Ok(observation) => {
                    if !observation.is_complete() {
                        cacheable = false;
                    }
                    for candidate in observation.into_candidates() {
                        validate_candidate(&candidate, provider.name())
                            .map_err(HostError::Candidate)?;
                        indexed.push(IndexedCandidate::new(
                            candidate.map_locator(|locator| Owner::Provider {
                                provider: Rc::clone(provider),
                                locator,
                            }),
                            *order as i64,
                            local_order,
                        ));
                        local_order += 1;
                    }
                }
                Err(error) => {
                    // A failing provider is skipped, warned about, and makes
                    // the collect uncacheable, matching the TypeScript
                    // registry's containment.
                    self.inner.borrow_mut().warnings.push(format!(
                        "skill provider \"{}\" skipped: {}",
                        provider.name(),
                        error
                    ));
                    cacheable = false;
                }
            }
        }
        Ok((resolve_layer(indexed), cacheable))
    }

    /// Invalidate after a stale definition load, only while the exact
    /// registration that produced the entry is still live.
    fn invalidate_entry(&self, provider: &Rc<dyn SkillProvider<L>>) {
        let listeners = {
            let mut inner = self.inner.borrow_mut();
            if inner.has_provider(provider) {
                inner.invalidate()
            } else {
                Vec::new()
            }
        };
        for listener in listeners {
            listener();
        }
    }
}

/// An invalidation capability borrowed by one provider registration.
///
/// `invalidate` applies only while the exact registration that received this
/// control is still live; after disposal or a failed registration it is a
/// no-op. The TypeScript control's abort signal has no synchronous
/// counterpart.
pub struct ProviderControl<L> {
    host: Weak<RefCell<HostInner<L>>>,
    token: Weak<()>,
}

impl<L> ProviderControl<L> {
    /// Invalidate completed catalogs while this exact registration is live.
    pub fn invalidate(&self) {
        let Some(token) = self.token.upgrade() else {
            return;
        };
        let Some(host) = self.host.upgrade() else {
            return;
        };
        let listeners = {
            let mut inner = host.borrow_mut();
            if inner.has_provider_token(&token) {
                inner.invalidate()
            } else {
                Vec::new()
            }
        };
        for listener in listeners {
            listener();
        }
    }
}

/// A live provider registration; dropping it unregisters the provider and
/// invalidates catalog caches, the way the TypeScript effect disposer does.
pub struct ProviderRegistration<L> {
    host: Weak<RefCell<HostInner<L>>>,
    scope: Option<String>,
    name: String,
    token: Rc<()>,
}

impl<L> fmt::Debug for ProviderRegistration<L> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderRegistration")
            .field("scope", &self.scope)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<L> Drop for ProviderRegistration<L> {
    fn drop(&mut self) {
        let Some(host) = self.host.upgrade() else {
            return;
        };
        let listeners = {
            let mut inner = host.borrow_mut();
            if let Some(layer) = inner.layer_mut(self.scope.as_deref()) {
                if let Some(position) = layer.providers.iter().position(|(existing, registered)| {
                    existing == &self.name && Rc::ptr_eq(&registered.token, &self.token)
                }) {
                    layer.providers.remove(position);
                }
            }
            inner.reclaim_if_empty(self.scope.as_deref());
            inner.invalidate()
        };
        for listener in listeners {
            listener();
        }
    }
}

/// A live runtime skill registration; dropping it removes the contribution.
///
/// A guard produced for a duplicate registration is a no-op, matching the
/// TypeScript disposer handed to ignored duplicates.
pub struct RuntimeRegistration<L> {
    host: Weak<RefCell<HostInner<L>>>,
    scope: Option<String>,
    name: SkillName,
    active: bool,
}

impl<L> fmt::Debug for RuntimeRegistration<L> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeRegistration")
            .field("scope", &self.scope)
            .field("name", &self.name)
            .field("active", &self.active)
            .finish()
    }
}

impl<L> Drop for RuntimeRegistration<L> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Some(host) = self.host.upgrade() else {
            return;
        };
        let listeners = {
            let mut inner = host.borrow_mut();
            if let Some(layer) = inner.layer_mut(self.scope.as_deref()) {
                layer.runtime.retain(|(existing, _)| existing != &self.name);
            }
            inner.reclaim_if_empty(self.scope.as_deref());
            inner.invalidate()
        };
        for listener in listeners {
            listener();
        }
    }
}

/// A live `skills/change` subscription; dropping it unsubscribes.
pub struct ChangeRegistration<L> {
    host: Weak<RefCell<HostInner<L>>>,
    token: Rc<()>,
}

impl<L> fmt::Debug for ChangeRegistration<L> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ChangeRegistration").finish()
    }
}

impl<L> Drop for ChangeRegistration<L> {
    fn drop(&mut self) {
        let Some(host) = self.host.upgrade() else {
            return;
        };
        let mut inner = host.borrow_mut();
        if let Some(position) = inner
            .change_listeners
            .iter()
            .position(|(existing, _)| Rc::ptr_eq(existing, &self.token))
        {
            inner.change_listeners.remove(position);
        }
    }
}
