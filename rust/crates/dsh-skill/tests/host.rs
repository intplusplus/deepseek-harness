//! Host-layer behavior, taken from the TypeScript registry's public
//! contracts (`packages/skill/skill/tests/skill.spec.ts`) and the
//! filesystem provider's discovery rules.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dsh_skill::host::{
    is_potential_skill_path, is_relevant_watch_event, FsProviderConfig, FsSkillProvider,
    HostConfig, LocalLocator, ProviderControl, ProviderError, ProviderObservation, SkillHost,
    SkillLookupOptions, SkillProvider, StdFiles, WatchEvent,
};
use dsh_skill::{
    render_skill_content, ProviderRegistration, SkillCandidate, SkillDefinition, SkillError,
    SkillInvocationPolicy, SkillName, SkillRank, SkillRegistration, SkillSource, SkillSummary,
};

fn name(value: &str) -> SkillName {
    SkillName::new(value).expect("test skill names are valid")
}

fn summary(skill_name: &str, provider: &str) -> SkillSummary {
    SkillSummary {
        name: name(skill_name),
        path: None,
        description: format!("{skill_name} description"),
        when_to_use: None,
        invocation: SkillInvocationPolicy::default(),
        source: SkillSource::new("memory"),
        provider: provider.to_string(),
        resource_base: None,
    }
}

fn memory_candidate(
    skill_name: &str,
    provider: &str,
    description: &str,
    rank: f64,
    body: &str,
) -> SkillCandidate<String> {
    let mut value = summary(skill_name, provider);
    value.description = description.to_string();
    SkillCandidate::new(
        value,
        SkillRank::new(rank).expect("finite"),
        body.to_string(),
    )
}

/// A configurable in-memory provider standing in for the TypeScript
/// `MemoryProvider` and its misbehaving variants.
struct MemoryProvider {
    provider_name: String,
    state: RefCell<Vec<SkillCandidate<String>>>,
    list_calls: Cell<usize>,
    fail: Cell<bool>,
    complete: Cell<bool>,
    get_returns_none: Cell<bool>,
    get_error: Cell<bool>,
    control: RefCell<Option<ProviderControl<String>>>,
}

impl MemoryProvider {
    fn new(provider_name: &str, candidates: Vec<SkillCandidate<String>>) -> Rc<Self> {
        Rc::new(Self {
            provider_name: provider_name.to_string(),
            state: RefCell::new(candidates),
            list_calls: Cell::new(0),
            fail: Cell::new(false),
            complete: Cell::new(true),
            get_returns_none: Cell::new(false),
            get_error: Cell::new(false),
            control: RefCell::new(None),
        })
    }

    fn replace(&self, candidates: Vec<SkillCandidate<String>>) {
        *self.state.borrow_mut() = candidates;
    }
}

impl SkillProvider<String> for MemoryProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn list(
        &self,
        _options: &SkillLookupOptions,
    ) -> Result<ProviderObservation<String>, ProviderError> {
        self.list_calls.set(self.list_calls.get() + 1);
        if self.fail.get() {
            return Err(ProviderError("transient discovery failure".to_string()));
        }
        let candidates = self.state.borrow().clone();
        Ok(if self.complete.get() {
            ProviderObservation::Complete(candidates)
        } else {
            ProviderObservation::Incomplete(candidates)
        })
    }

    fn get(
        &self,
        candidate: &SkillCandidate<String>,
        _options: &SkillLookupOptions,
    ) -> Result<Option<SkillDefinition>, ProviderError> {
        if self.get_error.get() {
            return Err(ProviderError("load failed".to_string()));
        }
        if self.get_returns_none.get() {
            return Ok(None);
        }
        Ok(Some(SkillDefinition {
            summary: candidate.summary.clone(),
            content: candidate.locator.clone(),
            metadata: candidate.metadata.clone(),
        }))
    }
}

/// A provider that invalidates the host on every discovery, the port of the
/// TypeScript self-invalidating provider test.
struct SelfInvalidatingProvider {
    provider_name: String,
    control: RefCell<Option<ProviderControl<String>>>,
    list_calls: Cell<usize>,
}

impl SkillProvider<String> for SelfInvalidatingProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn list(
        &self,
        _options: &SkillLookupOptions,
    ) -> Result<ProviderObservation<String>, ProviderError> {
        self.list_calls.set(self.list_calls.get() + 1);
        if let Some(control) = self.control.borrow().as_ref() {
            control.invalidate();
        }
        let candidate = memory_candidate(
            "bounded-skill",
            &self.provider_name,
            &format!("Attempt {}", self.list_calls.get()),
            10.0,
            "bounded body",
        );
        Ok(ProviderObservation::Complete(vec![candidate]))
    }

    fn get(
        &self,
        candidate: &SkillCandidate<String>,
        _options: &SkillLookupOptions,
    ) -> Result<Option<SkillDefinition>, ProviderError> {
        Ok(Some(SkillDefinition {
            summary: candidate.summary.clone(),
            content: candidate.locator.clone(),
            metadata: candidate.metadata.clone(),
        }))
    }
}

/// A provider whose loaded definition renames itself, the port of the
/// TypeScript stale-definition invalidation test.
struct RenamingProvider {
    list_calls: Cell<usize>,
}

impl SkillProvider<String> for RenamingProvider {
    fn name(&self) -> &str {
        "renamed"
    }

    fn list(
        &self,
        _options: &SkillLookupOptions,
    ) -> Result<ProviderObservation<String>, ProviderError> {
        self.list_calls.set(self.list_calls.get() + 1);
        Ok(ProviderObservation::Complete(vec![memory_candidate(
            "old-name", "renamed", "Old name", 1.0, "old body",
        )]))
    }

    fn get(
        &self,
        candidate: &SkillCandidate<String>,
        _options: &SkillLookupOptions,
    ) -> Result<Option<SkillDefinition>, ProviderError> {
        let mut definition = SkillDefinition {
            summary: candidate.summary.clone(),
            content: "Fresh body.".to_string(),
            metadata: candidate.metadata.clone(),
        };
        definition.summary.name = name("new-name");
        Ok(Some(definition))
    }
}

fn register(
    host: &SkillHost<String>,
    provider: &Rc<MemoryProvider>,
) -> ProviderRegistration<String> {
    let cloned = Rc::clone(provider);
    host.register_provider(None, |control| {
        *cloned.control.borrow_mut() = Some(control);
        cloned as Rc<dyn SkillProvider<String>>
    })
    .expect("registers")
}

fn runtime_registration(skill_name: &str, description: &str, content: &str) -> SkillRegistration {
    SkillRegistration {
        name: name(skill_name),
        description: description.to_string(),
        when_to_use: None,
        path: None,
        source: SkillSource::new(SkillSource::RUNTIME),
        resource_base: None,
        content: content.to_string(),
        metadata: None,
        invocation: None,
        provider: None,
    }
}

fn list_names<L: Clone + 'static>(host: &SkillHost<L>, scope: Option<&str>) -> Vec<String> {
    host.list(scope, &SkillLookupOptions::default())
        .expect("lists")
        .iter()
        .map(|skill| skill.name.as_str().to_string())
        .collect()
}

/// A temporary directory tree removed when the test ends.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "dsh-skill-host-{tag}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&path).expect("creates temp dir");
        Self { path }
    }

    fn write(&self, relative: &[&str], content: &str) -> PathBuf {
        let path = self.child(relative);
        std::fs::create_dir_all(path.parent().expect("has a parent")).expect("creates parent");
        std::fs::write(&path, content).expect("writes file");
        path
    }

    fn child(&self, relative: &[&str]) -> PathBuf {
        let mut path = self.path.clone();
        for segment in relative {
            path.push(segment);
        }
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Register a filesystem provider over `root`'s custom skill dir, handing
/// the provider and its registration back for observation.
fn register_fs_provider(
    root: &TempDir,
    mut config: FsProviderConfig,
) -> (
    SkillHost<LocalLocator>,
    ProviderRegistration<LocalLocator>,
    Rc<FsSkillProvider<StdFiles>>,
) {
    config.custom_skill_dirs = Some(vec![root.path.to_string_lossy().into_owned()]);
    // Isolated providers must see only their explicit roots, or every such
    // provider would re-discover the environment's bundled skills.
    config.include_default_roots = Some(false);
    let host = SkillHost::new();
    let provider_cell: Rc<RefCell<Option<Rc<FsSkillProvider<StdFiles>>>>> =
        Rc::new(RefCell::new(None));
    let cell_for_factory = Rc::clone(&provider_cell);
    let registration = host
        .register_provider(None, move |control| {
            let provider = Rc::new(FsSkillProvider::new(StdFiles, config, control));
            *cell_for_factory.borrow_mut() = Some(Rc::clone(&provider));
            provider as Rc<dyn SkillProvider<LocalLocator>>
        })
        .expect("registers");
    let provider = provider_cell
        .borrow()
        .as_ref()
        .expect("the factory ran")
        .clone();
    (host, registration, provider)
}

// ---------------------------------------------------------------------------
// Registry core
// ---------------------------------------------------------------------------

#[test]
fn registers_providers_resolves_duplicate_names_and_disposes() {
    let host: SkillHost<String> = SkillHost::new();
    let memory = MemoryProvider::new(
        "memory",
        vec![
            memory_candidate("z-skill", "memory", "Z skill", 20.0, "z body"),
            memory_candidate("a-skill", "memory", "A skill", 10.0, "a body"),
            memory_candidate("shadowed", "memory", "Lower priority", 20.0, "memory body"),
        ],
    );
    let _memory_registration = register(&host, &memory);
    let override_provider = MemoryProvider::new(
        "override",
        vec![memory_candidate(
            "shadowed",
            "override",
            "Higher priority",
            5.0,
            "Override body.",
        )],
    );
    let _override_registration = register(&host, &override_provider);

    let listed = host
        .list(None, &SkillLookupOptions::default())
        .expect("lists")
        .iter()
        .map(|skill| (skill.name.as_str().to_string(), skill.provider.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        listed,
        [
            ("a-skill".to_string(), "memory".to_string()),
            ("shadowed".to_string(), "override".to_string()),
            ("z-skill".to_string(), "memory".to_string()),
        ]
    );
    let loaded = host
        .get("shadowed", None, &SkillLookupOptions::default())
        .expect("loads")
        .expect("found");
    assert_eq!(loaded.content, "Override body.");

    drop(_memory_registration);
    assert_eq!(list_names(&host, None), ["shadowed"]);
}

#[test]
fn a_duplicate_provider_name_is_rejected_per_layer() {
    let host: SkillHost<String> = SkillHost::new();
    let _first = register(&host, &MemoryProvider::new("memory", vec![]));
    let preset_a = MemoryProvider::new("memory", vec![]);
    let _scoped = host
        .register_provider(Some("preset-a"), |_| {
            Rc::clone(&preset_a) as Rc<dyn SkillProvider<String>>
        })
        .expect("a different layer accepts the same name");
    let preset_again = MemoryProvider::new("memory", vec![]);
    let error = host
        .register_provider(Some("preset-a"), |control| {
            *preset_again.control.borrow_mut() = Some(control);
            preset_again as Rc<dyn SkillProvider<String>>
        })
        .expect_err("the same layer rejects the duplicate");
    assert!(
        error
            .to_string()
            .contains("already registered in this scope"),
        "{error}"
    );
    let global_again = MemoryProvider::new("memory", vec![]);
    let error = host
        .register_provider(None, |control| {
            *global_again.control.borrow_mut() = Some(control);
            global_again as Rc<dyn SkillProvider<String>>
        })
        .expect_err("the global layer rejects the duplicate");
    assert!(!error.to_string().contains("in this scope"), "{error}");
}

#[test]
fn the_runtime_provider_name_is_reserved() {
    let host: SkillHost<String> = SkillHost::new();
    let reserved = MemoryProvider::new("runtime", vec![]);
    let error = host
        .register_provider(None, |_| {
            Rc::clone(&reserved) as Rc<dyn SkillProvider<String>>
        })
        .expect_err("rejected");
    assert!(
        error
            .to_string()
            .contains("reserved for runtime skill registrations"),
        "{error}"
    );
}

#[test]
fn runtime_registrations_default_and_ignore_duplicates() {
    let host: SkillHost<String> = SkillHost::new();
    let _first = host
        .register_runtime(
            None,
            runtime_registration("same-skill", "First", "first body"),
        )
        .expect("registers");
    let listed = host
        .list(None, &SkillLookupOptions::default())
        .expect("lists");
    assert_eq!(
        listed[0].invocation,
        SkillInvocationPolicy {
            model_invocable: true,
            user_invocable: true
        }
    );
    let loaded = host
        .get("same-skill", None, &SkillLookupOptions::default())
        .expect("loads")
        .expect("found");
    assert_eq!(loaded.summary.provider, "runtime");

    // A duplicate records a warning and yields a no-op guard.
    let _second = host
        .register_runtime(
            None,
            runtime_registration("same-skill", "Second", "second body"),
        )
        .expect("registers as a no-op");
    assert!(host
        .take_warnings()
        .iter()
        .any(|warning| warning.contains("runtime skill \"same-skill\" ignored")));
    let loaded = host
        .get("same-skill", None, &SkillLookupOptions::default())
        .expect("loads")
        .expect("found");
    assert_eq!(loaded.content, "first body");
    drop(_second);
    assert!(
        host.get("same-skill", None, &SkillLookupOptions::default())
            .expect("loads")
            .is_some(),
        "the no-op guard cannot remove the winner"
    );
    drop(_first);
    assert!(host
        .get("same-skill", None, &SkillLookupOptions::default())
        .expect("loads")
        .is_none());
}

#[test]
fn runtime_validation_and_lookup_misses() {
    let host: SkillHost<String> = SkillHost::new();
    let invalid = SkillRegistration {
        description: String::new(),
        ..runtime_registration("demo", "ignored", "body")
    };
    assert!(matches!(
        host.register_runtime(None, invalid),
        Err(SkillError::EmptyDescription { .. })
    ));
    let options = SkillLookupOptions::default();
    assert!(host
        .get("Bad_Name", None, &options)
        .expect("never fails on a bad name")
        .is_none());
    assert!(host
        .get("missing-skill", None, &options)
        .expect("loads")
        .is_none());
}

// ---------------------------------------------------------------------------
// Collect cache
// ---------------------------------------------------------------------------

#[test]
fn cache_hits_last_until_an_invalidation() {
    let host: SkillHost<String> = SkillHost::new();
    let provider = MemoryProvider::new(
        "memory",
        vec![memory_candidate(
            "first-skill",
            "memory",
            "First",
            10.0,
            "first body",
        )],
    );
    let _registration = register(&host, &provider);

    assert_eq!(list_names(&host, None), ["first-skill"]);
    provider.replace(vec![memory_candidate(
        "second-skill",
        "memory",
        "Second",
        10.0,
        "second body",
    )]);
    assert_eq!(list_names(&host, None), ["first-skill"]);
    assert_eq!(provider.list_calls.get(), 1);

    let runtime = host
        .register_runtime(
            None,
            runtime_registration("runtime-skill", "Runtime", "Runtime body."),
        )
        .expect("registers");
    assert_eq!(list_names(&host, None), ["runtime-skill", "second-skill"]);
    assert_eq!(provider.list_calls.get(), 2);
    drop(runtime);
    assert_eq!(list_names(&host, None), ["second-skill"]);
    assert_eq!(provider.list_calls.get(), 3);
}

#[test]
fn cache_keys_distinguish_cwd_and_evict_oldest() {
    let host = SkillHost::<String>::with_config(HostConfig {
        collect_cache_max_entries: 1,
    })
    .expect("valid config");
    let provider = MemoryProvider::new("memory", vec![]);
    let _registration = register(&host, &provider);

    host.list(None, &SkillLookupOptions::default())
        .expect("lists");
    assert_eq!(provider.list_calls.get(), 1);
    host.list(None, &SkillLookupOptions::with_cwd("a"))
        .expect("lists");
    assert_eq!(provider.list_calls.get(), 2);
    // Same key: cached.
    host.list(None, &SkillLookupOptions::with_cwd("a"))
        .expect("lists");
    assert_eq!(provider.list_calls.get(), 2);
    // A different cwd evicts the single entry.
    host.list(None, &SkillLookupOptions::with_cwd("b"))
        .expect("lists");
    assert_eq!(provider.list_calls.get(), 3);
    host.list(None, &SkillLookupOptions::with_cwd("a"))
        .expect("lists");
    assert_eq!(provider.list_calls.get(), 4);
}

#[test]
fn failing_providers_are_skipped_and_leave_the_catalog_uncached() {
    let host: SkillHost<String> = SkillHost::new();
    let good = MemoryProvider::new(
        "good",
        vec![memory_candidate(
            "good-skill",
            "good",
            "Good",
            10.0,
            "good body",
        )],
    );
    let _good_registration = register(&host, &good);
    let flaky = MemoryProvider::new("flaky", vec![]);
    flaky.fail.set(true);
    let _flaky_registration = register(&host, &flaky);

    let snapshot = host
        .snapshot(None, &SkillLookupOptions::default())
        .expect("snapshots");
    assert_eq!(
        snapshot
            .skills
            .iter()
            .map(|skill| skill.name.as_str())
            .collect::<Vec<_>>(),
        ["good-skill"]
    );
    assert!(!snapshot.complete);
    assert_eq!(flaky.list_calls.get(), 1);
    let warnings = host.take_warnings();
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("skill provider \"flaky\" skipped"));

    // Uncached: the next observation retries discovery.
    host.snapshot(None, &SkillLookupOptions::default())
        .expect("snapshots");
    assert_eq!(flaky.list_calls.get(), 2);

    flaky.fail.set(false);
    flaky.replace(vec![memory_candidate(
        "flaky-skill",
        "flaky",
        "Flaky",
        10.0,
        "flaky body",
    )]);
    let snapshot = host
        .snapshot(None, &SkillLookupOptions::default())
        .expect("snapshots");
    assert!(snapshot.complete);
    assert_eq!(
        snapshot
            .skills
            .iter()
            .map(|skill| skill.name.as_str())
            .collect::<Vec<_>>(),
        ["flaky-skill", "good-skill"]
    );
    assert_eq!(flaky.list_calls.get(), 3);
    // Now cached.
    host.snapshot(None, &SkillLookupOptions::default())
        .expect("snapshots");
    assert_eq!(flaky.list_calls.get(), 3);
}

#[test]
fn incomplete_observations_stay_loadable_without_caching() {
    let host: SkillHost<String> = SkillHost::new();
    let provider = MemoryProvider::new(
        "incomplete",
        vec![memory_candidate(
            "available-skill",
            "incomplete",
            "Available",
            10.0,
            "available body",
        )],
    );
    provider.complete.set(false);
    let _registration = register(&host, &provider);

    let snapshot = host
        .snapshot(None, &SkillLookupOptions::default())
        .expect("snapshots");
    assert!(!snapshot.complete);
    assert_eq!(
        snapshot
            .skills
            .iter()
            .map(|skill| skill.name.as_str())
            .collect::<Vec<_>>(),
        ["available-skill"]
    );

    let loaded = host
        .get("available-skill", None, &SkillLookupOptions::default())
        .expect("loads")
        .expect("found");
    assert_eq!(loaded.content, "available body");
    host.list(None, &SkillLookupOptions::default())
        .expect("lists");
    assert_eq!(provider.list_calls.get(), 3, "never cached");
}

#[test]
fn control_invalidation_only_applies_while_the_registration_is_live() {
    let host: SkillHost<String> = SkillHost::new();
    let provider = MemoryProvider::new(
        "watched",
        vec![memory_candidate(
            "first-skill",
            "watched",
            "First",
            10.0,
            "first body",
        )],
    );
    let registration = register(&host, &provider);

    assert_eq!(list_names(&host, None), ["first-skill"]);
    provider.replace(vec![memory_candidate(
        "second-skill",
        "watched",
        "Second",
        10.0,
        "second body",
    )]);
    assert_eq!(list_names(&host, None), ["first-skill"]);
    provider
        .control
        .borrow()
        .as_ref()
        .expect("the control was handed to the factory")
        .invalidate();
    assert_eq!(list_names(&host, None), ["second-skill"]);
    drop(registration);

    // A late invalidation from a disposed registration is a no-op.
    let replacement = MemoryProvider::new(
        "replacement",
        vec![memory_candidate(
            "replacement-skill",
            "replacement",
            "Replacement",
            10.0,
            "replacement body",
        )],
    );
    let _replacement_registration = register(&host, &replacement);
    assert_eq!(list_names(&host, None), ["replacement-skill"]);
    provider.replace(vec![memory_candidate(
        "ignored-skill",
        "watched",
        "Ignored",
        10.0,
        "ignored body",
    )]);
    provider
        .control
        .borrow()
        .as_ref()
        .expect("the control exists")
        .invalidate();
    assert_eq!(list_names(&host, None), ["replacement-skill"]);
    assert_eq!(replacement.list_calls.get(), 1);
}

#[test]
fn change_listeners_observe_the_catalog_lifecycle() {
    let host: SkillHost<String> = SkillHost::new();
    let changes = Rc::new(Cell::new(0usize));
    let counter = Rc::clone(&changes);
    let _listener = host.on_change(move || counter.set(counter.get() + 1));

    let provider = MemoryProvider::new("memory", vec![]);
    let registration = register(&host, &provider);
    assert_eq!(changes.get(), 1);
    provider
        .control
        .borrow()
        .as_ref()
        .expect("control")
        .invalidate();
    assert_eq!(changes.get(), 2);

    let runtime = host
        .register_runtime(
            None,
            runtime_registration("runtime-skill", "Runtime", "Runtime body."),
        )
        .expect("registers");
    assert_eq!(changes.get(), 3);
    drop(runtime);
    assert_eq!(changes.get(), 4);
    drop(registration);
    assert_eq!(changes.get(), 5);
    // A disposed registration's control cannot notify anymore.
    provider
        .control
        .borrow()
        .as_ref()
        .expect("control")
        .invalidate();
    assert_eq!(changes.get(), 5);
}

#[test]
fn a_self_invalidating_provider_is_bounded_and_stays_uncached() {
    let host: SkillHost<String> = SkillHost::new();
    let provider = Rc::new(SelfInvalidatingProvider {
        provider_name: "self-invalidating".to_string(),
        control: RefCell::new(None),
        list_calls: Cell::new(0),
    });
    let provider_for_factory = Rc::clone(&provider);
    let _registration = host
        .register_provider(None, |control| {
            *provider_for_factory.control.borrow_mut() = Some(control);
            provider_for_factory as Rc<dyn SkillProvider<String>>
        })
        .expect("registers");

    let snapshot = host
        .snapshot(None, &SkillLookupOptions::default())
        .expect("snapshots");
    assert!(!snapshot.complete);
    assert_eq!(
        snapshot.skills[0].description, "Attempt 2",
        "the retry's result wins and stays uncached"
    );
    assert_eq!(provider.list_calls.get(), 2);

    host.snapshot(None, &SkillLookupOptions::default())
        .expect("snapshots");
    assert_eq!(provider.list_calls.get(), 4, "bounded at two attempts");
}

#[test]
fn a_renamed_loaded_definition_invalidates_and_disappears() {
    let host: SkillHost<String> = SkillHost::new();
    let provider = Rc::new(RenamingProvider {
        list_calls: Cell::new(0),
    });
    let _registration = host
        .register_provider(None, |_| {
            Rc::clone(&provider) as Rc<dyn SkillProvider<String>>
        })
        .expect("registers");

    let loaded = host
        .get("old-name", None, &SkillLookupOptions::default())
        .expect("loads");
    assert!(loaded.is_none(), "a renamed definition is not returned");
    host.list(None, &SkillLookupOptions::default())
        .expect("lists");
    assert_eq!(provider.list_calls.get(), 2, "the stale load invalidated");
}

#[test]
fn load_failures_propagate_and_vanished_candidates_return_none() {
    let host: SkillHost<String> = SkillHost::new();
    let failing = MemoryProvider::new(
        "failing-loader",
        vec![memory_candidate(
            "failing-skill",
            "failing-loader",
            "Failing",
            10.0,
            "failing body",
        )],
    );
    failing.get_error.set(true);
    let _failing_registration = register(&host, &failing);
    let error = host
        .get("failing-skill", None, &SkillLookupOptions::default())
        .expect_err("the load failure propagates");
    assert!(error.to_string().contains("load failed"), "{error}");

    let vanished = MemoryProvider::new(
        "vanished-body",
        vec![memory_candidate(
            "vanished-skill",
            "vanished-body",
            "Vanished",
            10.0,
            "vanished body",
        )],
    );
    vanished.get_returns_none.set(true);
    let _vanished_registration = register(&host, &vanished);
    assert!(host
        .get("vanished-skill", None, &SkillLookupOptions::default())
        .expect("loads")
        .is_none());
}

// ---------------------------------------------------------------------------
// Scope chains
// ---------------------------------------------------------------------------

#[test]
fn a_scoped_provider_is_visible_only_in_its_scope_view() {
    let host: SkillHost<String> = SkillHost::new();
    let _global = register(
        &host,
        &MemoryProvider::new(
            "memory",
            vec![memory_candidate(
                "global-skill",
                "memory",
                "Global",
                100.0,
                "global body",
            )],
        ),
    );
    let preset_provider = MemoryProvider::new(
        "preset-local",
        vec![memory_candidate(
            "preset-skill",
            "preset-local",
            "Preset",
            300.0,
            "Preset body.",
        )],
    );
    let _preset = host
        .register_provider(Some("preset"), |_| {
            Rc::clone(&preset_provider) as Rc<dyn SkillProvider<String>>
        })
        .expect("registers");

    assert_eq!(list_names(&host, None), ["global-skill"]);
    assert_eq!(
        list_names(&host, Some("preset")),
        ["global-skill", "preset-skill"]
    );
    let loaded = host
        .get(
            "preset-skill",
            Some("preset"),
            &SkillLookupOptions::default(),
        )
        .expect("loads")
        .expect("found");
    assert_eq!(loaded.content, "Preset body.");
    assert!(host
        .get("preset-skill", None, &SkillLookupOptions::default())
        .expect("loads")
        .is_none());
}

#[test]
fn the_nearest_layer_wins_a_duplicate_name_regardless_of_rank() {
    let host: SkillHost<String> = SkillHost::new();
    let _global = register(
        &host,
        &MemoryProvider::new(
            "memory",
            vec![memory_candidate(
                "shared-name",
                "memory",
                "Global wins ranks",
                10.0,
                "global body",
            )],
        ),
    );
    let shadow = MemoryProvider::new(
        "preset-local",
        vec![memory_candidate(
            "shared-name",
            "preset-local",
            "Preset shadow",
            900.0,
            "Preset shadow body.",
        )],
    );
    let _shadow = host
        .register_provider(Some("preset"), |_| {
            Rc::clone(&shadow) as Rc<dyn SkillProvider<String>>
        })
        .expect("registers");

    let scoped = host
        .list(Some("preset"), &SkillLookupOptions::default())
        .expect("lists");
    assert_eq!(scoped.len(), 1);
    assert_eq!(scoped[0].description, "Preset shadow");
    let loaded = host
        .get(
            "shared-name",
            Some("preset"),
            &SkillLookupOptions::default(),
        )
        .expect("loads")
        .expect("found");
    assert_eq!(loaded.content, "Preset shadow body.");
    let global = host
        .list(None, &SkillLookupOptions::default())
        .expect("lists");
    assert_eq!(global[0].description, "Global wins ranks");
}

#[test]
fn an_agent_key_inherits_its_preset_layer_and_follows_rebind() {
    let host: SkillHost<String> = SkillHost::new();
    let _a = host
        .register_runtime(
            Some("preset-a"),
            runtime_registration("skill-a", "Skill A", "Body A."),
        )
        .expect("registers");
    let _b = host
        .register_runtime(
            Some("preset-b"),
            runtime_registration("skill-b", "Skill B", "Body B."),
        )
        .expect("registers");

    host.bind_scope_parent("agent", "preset-a").expect("binds");
    assert_eq!(list_names(&host, Some("agent")), ["skill-a"]);
    host.bind_scope_parent("agent", "preset-b")
        .expect("rebinds");
    assert_eq!(list_names(&host, Some("agent")), ["skill-b"]);
    assert!(host.bind_scope_parent("agent", "agent").is_err());
}

#[test]
fn runtime_skills_shadow_across_layers_and_first_wins_within_one() {
    let host: SkillHost<String> = SkillHost::new();
    let _global = host
        .register_runtime(
            None,
            runtime_registration("told-twice", "Global runtime", "Global body."),
        )
        .expect("registers");
    let shadow = host
        .register_runtime(
            Some("preset"),
            runtime_registration("told-twice", "Preset runtime", "Preset body."),
        )
        .expect("registers");
    assert!(host.take_warnings().is_empty());
    let ignored = host
        .register_runtime(
            Some("preset"),
            runtime_registration("told-twice", "Ignored", "Ignored."),
        )
        .expect("registers as a no-op");
    assert!(host
        .take_warnings()
        .iter()
        .any(|warning| warning.contains("runtime skill \"told-twice\" ignored")));

    let scoped = host
        .get("told-twice", Some("preset"), &SkillLookupOptions::default())
        .expect("loads")
        .expect("found");
    assert_eq!(scoped.content, "Preset body.");
    let global = host
        .get("told-twice", None, &SkillLookupOptions::default())
        .expect("loads")
        .expect("found");
    assert_eq!(global.content, "Global body.");

    // The no-op guard cannot remove the winner; the real one can.
    drop(ignored);
    assert!(host
        .get("told-twice", Some("preset"), &SkillLookupOptions::default())
        .expect("loads")
        .is_some());
    drop(shadow);
    let scoped = host
        .get("told-twice", Some("preset"), &SkillLookupOptions::default())
        .expect("loads")
        .expect("found");
    assert_eq!(scoped.content, "Global body.");
    drop(_global);
    assert!(host
        .get("told-twice", None, &SkillLookupOptions::default())
        .expect("loads")
        .is_none());
}

#[test]
fn a_disposed_scoped_registration_drops_from_its_scope_view() {
    let host: SkillHost<String> = SkillHost::new();
    let provider = MemoryProvider::new(
        "scoped",
        vec![memory_candidate(
            "scoped-skill",
            "scoped",
            "Scoped",
            100.0,
            "scoped body",
        )],
    );
    let registration = host
        .register_provider(Some("preset"), |_| {
            Rc::clone(&provider) as Rc<dyn SkillProvider<String>>
        })
        .expect("registers");
    assert_eq!(list_names(&host, Some("preset")), ["scoped-skill"]);
    drop(registration);
    assert_eq!(list_names(&host, Some("preset")), Vec::<String>::new());
}

// ---------------------------------------------------------------------------
// Filesystem provider
// ---------------------------------------------------------------------------

#[test]
fn discovers_directory_bundles_and_flat_markdown_skills_end_to_end() {
    let root = TempDir::new("fs-e2e");
    let alpha_dir = root.child(&["alpha"]);
    root.write(
        &["alpha", "SKILL.md"],
        "---\nname: alpha\ndescription: Alpha skill\nmetadata:\n  owner: test\n---\n\nAlpha body.\n",
    );
    root.write(
        &["beta.md"],
        "---\nname: beta\ndescription: Beta skill\n---\nBeta body.\n",
    );
    let (host, _registration, _provider) = register_fs_provider(&root, FsProviderConfig::default());

    assert_eq!(list_names(&host, None), ["alpha", "beta"]);
    let loaded = host
        .get("alpha", None, &SkillLookupOptions::default())
        .expect("loads")
        .expect("found");
    assert_eq!(loaded.content, "Alpha body.");
    assert_eq!(
        loaded.summary.resource_base,
        Some(dsh_skill::SkillResourceBase::Directory {
            path: alpha_dir.to_string_lossy().into_owned()
        })
    );
    assert_eq!(loaded.metadata.as_ref().expect("metadata")["owner"], "test");

    let rendered = render_skill_content(&loaded);
    assert!(rendered.contains("<skill_content name=\"alpha\">"));
    assert!(rendered.contains(&format!(
        "Base directory for this skill: {}",
        alpha_dir.to_string_lossy()
    )));
    assert!(rendered.contains("Alpha body."));
}

#[test]
fn project_roots_rank_above_user_roots_for_the_same_name() {
    let project = TempDir::new("fs-project");
    std::fs::create_dir_all(project.child(&[".git"])).expect("creates .git");
    project.write(
        &[".dsh", "skills", "shared", "SKILL.md"],
        "---\nname: shared\ndescription: Project wins\n---\nProject body.\n",
    );
    let home = TempDir::new("fs-home");
    home.write(
        &["skills", "shared", "SKILL.md"],
        "---\nname: shared\ndescription: User loses\n---\nUser body.\n",
    );
    let agents = TempDir::new("fs-agents");

    let host = SkillHost::new();
    let config = FsProviderConfig {
        include_default_roots: Some(true),
        dsh_home: Some(home.path.to_string_lossy().into_owned()),
        agents_home: Some(agents.path.to_string_lossy().into_owned()),
        ..FsProviderConfig::default()
    };
    let provider_cell: Rc<RefCell<Option<Rc<FsSkillProvider<StdFiles>>>>> =
        Rc::new(RefCell::new(None));
    let cell_for_factory = Rc::clone(&provider_cell);
    let _registration = host
        .register_provider(None, move |control| {
            let provider = Rc::new(FsSkillProvider::new(StdFiles, config, control));
            *cell_for_factory.borrow_mut() = Some(Rc::clone(&provider));
            provider as Rc<dyn SkillProvider<LocalLocator>>
        })
        .expect("registers");

    let options = SkillLookupOptions::with_cwd(project.path.to_string_lossy().into_owned());
    let loaded = host
        .get("shared", None, &options)
        .expect("loads")
        .expect("found");
    assert_eq!(loaded.summary.description, "Project wins");
    assert_eq!(loaded.summary.source.as_str(), "project-dsh");
    assert_eq!(loaded.content, "Project body.");
}

#[test]
fn invalid_skill_files_are_skipped_with_warnings() {
    let root = TempDir::new("fs-invalid");
    root.write(&["no-frontmatter.md"], "Just text.\n");
    root.write(&["bad-yaml.md"], "---\nthis line has no colon\n---\nbody\n");
    root.write(
        &["no-description.md"],
        "---\nname: no-description\n---\nbody\n",
    );
    root.write(
        &["Bad_Name.md"],
        "---\nname: Bad_Name\ndescription: bad name\n---\nbody\n",
    );
    root.write(
        &["legacy.md"],
        "---\nname: legacy\ndescription: legacy key\ndisableModelInvocation: true\n---\nbody\n",
    );
    root.write(
        &["bad-boolean.md"],
        "---\nname: bad-boolean\ndescription: bad boolean\ndisable-model-invocation: sometimes\n---\nbody\n",
    );
    root.write(
        &["good.md"],
        "---\nname: good\ndescription: Good skill\n---\nGood body.\n",
    );
    let (host, _registration, provider) = register_fs_provider(&root, FsProviderConfig::default());

    assert_eq!(list_names(&host, None), ["good"]);
    let warnings = provider.take_warnings();
    assert_eq!(warnings.len(), 6, "{warnings:?}");
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("missing YAML frontmatter")));
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("invalid YAML frontmatter")));
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("frontmatter requires name and description")));
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("invalid skill name")));
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("disableModelInvocation")
            && warning.contains("disable-model-invocation")));
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("must be a boolean")));
}

#[test]
fn frontmatter_invocation_policy_is_parsed() {
    let root = TempDir::new("fs-invocation");
    root.write(
        &["model-off.md"],
        "---\nname: model-off\ndescription: Model off\nuser-invocable: false\n---\nbody\n",
    );
    root.write(
        &["user-off.md"],
        "---\nname: user-off\ndescription: User off\ndisable-model-invocation: true\n---\nbody\n",
    );
    root.write(
        &["both-on.md"],
        "---\nname: both-on\ndescription: Both on\n---\nbody\n",
    );
    let (host, _registration, _provider) = register_fs_provider(&root, FsProviderConfig::default());

    let listed = host
        .list(None, &SkillLookupOptions::default())
        .expect("lists");
    let policies: std::collections::BTreeMap<String, SkillInvocationPolicy> = listed
        .iter()
        .map(|skill| (skill.name.as_str().to_string(), skill.invocation))
        .collect();
    assert_eq!(
        policies["model-off"],
        SkillInvocationPolicy {
            model_invocable: true,
            user_invocable: false
        }
    );
    assert_eq!(
        policies["user-off"],
        SkillInvocationPolicy {
            model_invocable: false,
            user_invocable: true
        }
    );
    assert_eq!(
        policies["both-on"],
        SkillInvocationPolicy {
            model_invocable: true,
            user_invocable: true
        }
    );
}

#[test]
fn a_deleted_skill_file_loads_as_none_and_host_mutation_invalidates() {
    let root = TempDir::new("fs-mutation");
    let skill_file = root.write(
        &["demo.md"],
        "---\nname: demo\ndescription: Demo skill\n---\nDemo body.\n",
    );
    let (host, _registration, provider) = register_fs_provider(&root, FsProviderConfig::default());
    let options = SkillLookupOptions::default();

    assert_eq!(list_names(&host, None), ["demo"]);
    assert!(std::fs::remove_file(&skill_file).is_ok());

    // The cached candidate still resolves to a load, which reports the
    // vanished body as absent.
    assert!(host.get("demo", None, &options).expect("loads").is_none());

    // A mutation outside the skill roots does not invalidate.
    provider.observe_host_mutation(root.child(&["unrelated.txt"]).to_string_lossy().as_ref());
    assert_eq!(list_names(&host, None), ["demo"]);

    // A mutation at the skill path does.
    provider.observe_host_mutation(skill_file.to_string_lossy().as_ref());
    assert_eq!(list_names(&host, None), Vec::<String>::new());
}

#[test]
fn watch_relevance_rules_match_the_typescript_filters() {
    let root = TempDir::new("fs-watch");
    root.write(&["top.md"], "body");
    root.write(&["nested", "SKILL.md"], "body");
    let root_path = root.path.to_string_lossy().into_owned();
    let skill_root = dsh_skill::SkillRoot {
        path: root_path.clone(),
        source: SkillSource::new(SkillSource::CUSTOM),
        rank: 300.0,
        skip_system: false,
    };
    let at = |segments: &[&str]| root.child(segments).to_string_lossy().into_owned();

    // The root itself only matters on add/unlink of the directory.
    assert!(is_relevant_watch_event(
        &skill_root,
        WatchEvent::AddDir,
        &root_path
    ));
    assert!(is_relevant_watch_event(
        &skill_root,
        WatchEvent::UnlinkDir,
        &root_path
    ));
    assert!(!is_relevant_watch_event(
        &skill_root,
        WatchEvent::Change,
        &root_path
    ));
    // A top-level markdown file matters; other files do not.
    assert!(is_relevant_watch_event(
        &skill_root,
        WatchEvent::Change,
        &at(&["top.md"])
    ));
    assert!(!is_relevant_watch_event(
        &skill_root,
        WatchEvent::Change,
        &at(&["top.txt"])
    ));
    assert!(is_relevant_watch_event(
        &skill_root,
        WatchEvent::AddDir,
        &at(&["nested"])
    ));
    // A bundle's SKILL.md matters in either direction, but not its directory.
    assert!(is_relevant_watch_event(
        &skill_root,
        WatchEvent::Change,
        &at(&["nested", "SKILL.md"])
    ));
    assert!(!is_relevant_watch_event(
        &skill_root,
        WatchEvent::AddDir,
        &at(&["nested", "SKILL.md"])
    ));
    // Deeper paths and outside paths never matter.
    assert!(!is_relevant_watch_event(
        &skill_root,
        WatchEvent::Change,
        &at(&["nested", "extra", "SKILL.md"])
    ));
    assert!(!is_relevant_watch_event(
        &skill_root,
        WatchEvent::Change,
        "C:/definitely/elsewhere/skill.md"
    ));

    let system_root = dsh_skill::SkillRoot {
        skip_system: true,
        ..skill_root.clone()
    };
    assert!(!is_relevant_watch_event(
        &system_root,
        WatchEvent::Change,
        &at(&[".system", "hidden.md"])
    ));
    assert!(!is_potential_skill_path(
        &system_root,
        &at(&[".system", "hidden.md"])
    ));

    assert!(is_potential_skill_path(&skill_root, &at(&["top.md"])));
    assert!(is_potential_skill_path(
        &skill_root,
        &at(&["nested", "SKILL.md"])
    ));
    assert!(!is_potential_skill_path(&skill_root, &root_path));
    assert!(!is_potential_skill_path(
        &skill_root,
        &at(&["nested", "extra", "SKILL.md"])
    ));
    assert!(!is_potential_skill_path(
        &skill_root,
        &at(&["nested", "README.md"])
    ));
}
