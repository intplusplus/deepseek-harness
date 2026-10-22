//! Deterministic LIFO-drain tests for the unified plugin-host assembly.
//!
//! These follow the offline instantiation discipline of `dsh-ptc-host` and
//! `dsh-skill`: hosts are constructed in memory and never spawn a real
//! network/Node child process. The only async work is the retry/persistent
//! drains, which resolve immediately because no recovery is in flight and no
//! retained process exists.

use std::sync::{Arc, Mutex};

use dsh_agent::Agent;
use dsh_host::{HostConfig, LaunchConfig, RetryLifecycle, SessionScope, SkillHost};
use dsh_interaction::questions::UserQuestionService;
use dsh_session::SessionId;

/// A tracer that records each host name, in the order its drain guard fires.
fn tracing_tracer(log: &Arc<Mutex<Vec<String>>>) -> Arc<dyn Fn(&str) + Send + Sync> {
    let log = Arc::clone(log);
    Arc::new(move |name| log.lock().unwrap().push(name.to_string()))
}

#[tokio::test]
async fn drops_hosts_in_lifo_order() {
    // Mount order: retry, ptc, persistent, approval, question, skill.
    // LIFO unwind means the drain order is the reverse.
    let log = Arc::new(Mutex::new(Vec::new()));
    let tracer = tracing_tracer(&log);

    let mut scope = SessionScope::with_tracer(tracer)
        .with_agent(Agent::new(SessionId::new("agent-1").expect("valid id")));

    scope.mount_retry(RetryLifecycle::new());
    scope
        .mount_ptc(HostConfig::default(), LaunchConfig::new("node", vec![]))
        .expect("ptc host constructs offline");
    scope
        .mount_persistent(HostConfig::default(), LaunchConfig::new("node", vec![]))
        .expect("persistent host constructs offline");
    scope.mount_approval_default();
    scope.mount_question(UserQuestionService::new());
    scope.mount_skill(SkillHost::<()>::new());

    // Hosts are retrievable through the kernel context while the scope lives.
    assert!(scope.context().require::<RetryLifecycle>().is_ok());
    assert!(scope.context().require::<dsh_host::PtcHost>().is_ok());
    assert!(scope
        .context()
        .require::<dsh_host::PersistentHost>()
        .is_ok());

    drop(scope);

    assert_eq!(
        *log.lock().unwrap(),
        vec![
            "skill",
            "question",
            "approval",
            "persistent",
            "ptc",
            "retry"
        ],
        "the scope must unwind hosts in LIFO (reverse registration) order"
    );
}

#[tokio::test]
async fn sends_each_host_through_the_kernel_service_registry() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut scope = SessionScope::with_tracer(tracing_tracer(&log));

    scope
        .mount_persistent(HostConfig::default(), LaunchConfig::new("node", vec![]))
        .expect("constructs offline");
    scope.mount_question(UserQuestionService::new());

    // Both are registered as retrievable services before teardown.
    assert!(scope
        .context()
        .require::<dsh_host::PersistentHost>()
        .is_ok());
    assert!(scope.context().require::<UserQuestionService>().is_ok());

    drop(scope);
    assert_eq!(*log.lock().unwrap(), vec!["question", "persistent"]);
}

#[test]
fn drains_without_a_runtime_for_the_skill_and_sync_hosts() {
    // No tokio runtime is present here; the retry drain's await is simply
    // skipped while its synchronous abort still takes effect.
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut scope = SessionScope::with_tracer(tracing_tracer(&log));

    scope.mount_retry(RetryLifecycle::new());
    scope.mount_skill(SkillHost::<()>::new());

    drop(scope);

    assert_eq!(*log.lock().unwrap(), vec!["skill", "retry"]);
}

#[test]
fn ptc_construction_failure_surfaces_without_spawning() {
    let mut scope = SessionScope::with_tracer(tracing_tracer(&Arc::new(Mutex::new(Vec::new()))));

    // An empty executable is rejected by `PtcHost::new` before any child spawns.
    let result = scope.mount_ptc(HostConfig::default(), LaunchConfig::new("", vec![]));
    assert!(result.is_err(), "an empty executable must be rejected");

    // The failed mount left no drain guard behind.
    drop(scope);
}

#[test]
fn agent_identity_is_recorded_when_supplied() {
    let scope =
        SessionScope::new().with_agent(Agent::new(SessionId::new("agent-7").expect("valid id")));
    assert_eq!(
        scope.agent().map(|a| a.id.clone()),
        Some(SessionId::new("agent-7").unwrap())
    );

    let empty = SessionScope::new();
    assert!(empty.agent().is_none());
}
