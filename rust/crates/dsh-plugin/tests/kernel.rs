//! Kernel behavior: reversible service registrations, contained observer
//! failures, waterfall delegation, and plugin load/unload semantics.

use std::{
    error::Error,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use dsh_plugin::{Context, Plugin, PluginHost, Scope};

#[derive(Debug)]
struct Config {
    value: u32,
}

#[derive(Debug)]
struct Other;

/// An observer event.
struct Ping {
    #[allow(dead_code)]
    hits: usize,
}

/// A waterfall event.
struct Audit {
    note: String,
}

#[test]
fn service_registration_restores_the_previous_provider_on_unwind() {
    let context = Arc::new(Context::new());
    let guard = context.insert(Arc::new(Config { value: 1 }));
    assert_eq!(context.get::<Config>().unwrap().value, 1);

    let replacement = context.insert(Arc::new(Config { value: 2 }));
    assert_eq!(context.get::<Config>().unwrap().value, 2);

    drop(replacement);
    assert_eq!(context.get::<Config>().unwrap().value, 1);

    drop(guard);
    assert!(context.get::<Config>().is_none());
}

#[test]
fn require_fails_explicitly_when_the_service_is_absent() {
    let context = Context::new();
    let error = context.require::<Config>().unwrap_err();
    assert!(error.to_string().contains("not registered"), "{error}");
}

#[test]
fn observer_failures_are_contained_and_cannot_veto() {
    let context = Context::new();
    let _guard = context.observe(|_: &Ping| panic!("observer boom"));
    // Must not panic: observer failures are contained.
    context.emit(&Ping { hits: 1 });
}

#[test]
fn observer_registration_unwinds_when_its_guard_drops() {
    let context = Context::new();
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    let guard = context.observe(move |_: &Ping| {
        counter.fetch_add(1, Ordering::SeqCst);
    });
    context.emit(&Ping { hits: 1 });
    drop(guard);
    context.emit(&Ping { hits: 2 });
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[test]
fn waterfall_listeners_may_rewrite_and_must_delegate() {
    let context = Context::new();
    let first = context.waterfall(|mut audit: Audit, next| {
        audit.note.push_str("[first]");
        next(audit)
    });
    let second = context.waterfall(|mut audit: Audit, next| {
        audit.note.push_str("[second]");
        next(audit)
    });
    let final_audit = context
        .dispatch(Audit {
            note: String::new(),
        })
        .unwrap();
    assert_eq!(final_audit.note, "[first][second]");

    drop(second);
    let after = context
        .dispatch(Audit {
            note: String::new(),
        })
        .unwrap();
    assert_eq!(after.note, "[first]");

    drop(first);
    let terminal = context
        .dispatch(Audit {
            note: "kept".to_string(),
        })
        .unwrap();
    assert_eq!(terminal.note, "kept");
}

#[test]
fn a_listener_that_never_delegates_short_circuits_the_chain() {
    let context = Context::new();
    let _short = context.waterfall(|audit: Audit, _next| Ok(audit));
    let _tail = context.waterfall(|mut audit: Audit, next| {
        audit.note.push_str("[tail]");
        next(audit)
    });
    let result = context
        .dispatch(Audit {
            note: String::new(),
        })
        .unwrap();
    assert_eq!(result.note, "");
}

struct GoodPlugin;

impl Plugin for GoodPlugin {
    fn name(&self) -> &str {
        "good"
    }

    fn load(&self, scope: &mut Scope) -> Result<(), Box<dyn Error + Send + Sync>> {
        scope.set(Arc::new(Config { value: 7 }));
        Ok(())
    }
}

struct BadPlugin {
    rolled_back: Arc<AtomicUsize>,
}

impl Plugin for BadPlugin {
    fn name(&self) -> &str {
        "bad"
    }

    fn load(&self, scope: &mut Scope) -> Result<(), Box<dyn Error + Send + Sync>> {
        scope.set(Arc::new(Other));
        let rolled_back = Arc::clone(&self.rolled_back);
        scope.defer(move || {
            rolled_back.fetch_add(1, Ordering::SeqCst);
        });
        Err("boom".into())
    }
}

#[test]
fn unload_unwinds_every_effect_the_plugin_registered() {
    let mut host = PluginHost::new();
    host.load(&GoodPlugin).unwrap();
    assert_eq!(host.context().get::<Config>().unwrap().value, 7);
    assert!(host.unload("good"));
    assert!(host.context().get::<Config>().is_none());
    assert!(!host.unload("good"));
}

#[test]
fn a_failed_load_rolls_back_every_effect_and_keeps_other_plugins() {
    let rolled_back = Arc::new(AtomicUsize::new(0));
    let mut host = PluginHost::new();
    host.load(&GoodPlugin).unwrap();

    let error = host
        .load(&BadPlugin {
            rolled_back: Arc::clone(&rolled_back),
        })
        .unwrap_err();
    assert!(error.to_string().contains("boom"), "{error}");
    assert_eq!(rolled_back.load(Ordering::SeqCst), 1);
    assert!(host.context().get::<Other>().is_none());
    assert_eq!(host.context().get::<Config>().unwrap().value, 7);
    assert_eq!(host.loaded_names(), vec!["good"]);
}
