//! Offline integration tests over real `node -e` children.
//!
//! Every sandbox in these tests is a Node child speaking the exact frame
//! protocol `dsh-ptc` defines: 4-byte big-endian length-prefixed JSON over
//! stdin/stdout, readiness first, then `run {id, cwd, data}` envelopes with
//! `{id, payload}` answers. The child's behavior is selected by the program
//! text (`ECHO:`, `HANG:`, `CRASH:` markers) so no network and no fixtures
//! are involved.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dsh_ptc::policy::{PolicyError, Timeout, TimeoutRequest};
use dsh_ptc::types::PtcFailureKind;
use dsh_ptc::{PolicyKey, SandboxMode};
use dsh_ptc_host::{
    startup_environment, ConfigError, FnBinding, HostConfig, HostError, HostNamespace,
    LaunchConfig, PersistentHost, PtcHost, RunRequest, STARTUP_ENVIRONMENT_NAMES,
};
use serde_json::{json, Value};

/// The sandbox stand-in: frames over stdin/stdout, behaviors keyed by the
/// program text's marker prefix.
const CHILD_SCRIPT: &str = r#"
const fs = require('fs');
function send(obj) {
  const body = Buffer.from(JSON.stringify(obj), 'utf8');
  const head = Buffer.alloc(4);
  head.writeUInt32BE(body.length, 0);
  fs.writeSync(1, head);
  fs.writeSync(1, body);
}
function readExact(n) {
  const out = Buffer.alloc(n);
  let got = 0;
  while (got < n) {
    let read;
    try { read = fs.readSync(0, out, got, n - got); } catch (error) { return null; }
    if (read <= 0) return null;
    got += read;
  }
  return out;
}
function recv() {
  const head = readExact(4);
  if (!head) return null;
  const len = head.readUInt32BE(0);
  const body = readExact(len);
  if (!body) return null;
  return JSON.parse(body.toString('utf8'));
}
function sleep(ms) { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms); }
function wireOf(value, tokens) {
  tokens = tokens || [];
  if (value === null || typeof value !== 'object') { tokens.push(value); return tokens; }
  if (Array.isArray(value)) {
    tokens.push({ kind: 'array', length: value.length });
    for (const item of value) wireOf(item, tokens);
    return tokens;
  }
  const keys = Object.keys(value);
  tokens.push({ kind: 'object', keys: keys });
  for (const key of keys) wireOf(value[key], tokens);
  return tokens;
}
function unwire(tokens) {
  let at = 0;
  function take() {
    const token = tokens[at++];
    if (token === null || typeof token !== 'object') return token;
    if (token.kind === 'array') { const out = []; for (let i = 0; i < token.length; i++) out.push(take()); return out; }
    if (token.kind === 'object') { const out = {}; for (const key of token.keys) out[key] = take(); return out; }
    throw new Error('bad wire token');
  }
  return take();
}
const vars = Object.create(null);
const pid = process.pid;
const replies = [];
function emit(id, payload) { send({ id: id, payload: payload }); }
function done(id, payload) { emit(id, Object.assign({ type: 'done' }, payload)); }
function awaitReplies(count) {
  while (replies.length < count) {
    const msg = recv();
    if (!msg) process.exit(9);
    if (msg.type === 'reply') replies.push(msg);
  }
  return replies.splice(0, count);
}
function evaluate(msg) {
  const id = msg.id;
  const code = msg.data.code;
  if (code.startsWith('ECHO:')) { done(id, { value: wireOf(JSON.parse(code.slice(5))) }); return; }
  if (code.startsWith('PID')) { done(id, { value: [pid] }); return; }
  if (code.startsWith('CWD')) { done(id, { value: [process.cwd()] }); return; }
  if (code.startsWith('SET:')) {
    const eq = code.indexOf('=', 4);
    vars[code.slice(4, eq)] = code.slice(eq + 1);
    done(id, {});
    return;
  }
  if (code.startsWith('GET:')) {
    const name = code.slice(4);
    if (!(name in vars)) { done(id, { error: { kind: 'exception', message: 'undefined variable ' + name } }); return; }
    done(id, { value: wireOf(vars[name]) });
    return;
  }
  if (code.startsWith('LOG:')) {
    const bar = code.indexOf('|', 4);
    emit(id, { type: 'log', text: code.slice(4, bar) });
    const tail = code.slice(bar + 1);
    if (tail === '-') done(id, {});
    else done(id, { value: wireOf(JSON.parse(tail)) });
    return;
  }
  if (code.startsWith('BIGLOG:')) {
    const count = parseInt(code.slice(7), 10);
    for (let i = 0; i < count; i++) emit(id, { type: 'log', text: '0123456789' });
    done(id, {});
    return;
  }
  if (code.startsWith('CALL:')) {
    emit(id, { type: 'call', id: 1, global: 'tools', name: 'echo', args: wireOf(JSON.parse(code.slice(5))) });
    const reply = awaitReplies(1)[0];
    if (reply.ok) done(id, { value: wireOf(unwire(reply.value)) });
    else done(id, { value: [reply.message] });
    return;
  }
  if (code === 'CALLFAIL') {
    emit(id, { type: 'call', id: 1, global: 'tools', name: 'boom', args: [1] });
    const reply = awaitReplies(1)[0];
    done(id, { value: wireOf([reply.ok ? 'unexpected-ok' : reply.message]) });
    return;
  }
  if (code === 'CALL2') {
    emit(id, { type: 'call', id: 1, global: 'tools', name: 'slow', args: wireOf([1]) });
    emit(id, { type: 'call', id: 2, global: 'tools', name: 'slow', args: wireOf([2]) });
    const both = awaitReplies(2);
    const first = both.find(r => r.id === 1);
    const second = both.find(r => r.id === 2);
    done(id, { value: wireOf([unwire(first.value), unwire(second.value)]) });
    return;
  }
  if (code === 'CALLBADID') {
    emit(id, { type: 'call', id: 7, global: 'tools', name: 'echo', args: [1] });
    while (recv() !== null) {}
    process.exit(0);
  }
  if (code === 'CALLUNKNOWN') {
    emit(id, { type: 'call', id: 1, global: 'nope', name: 'echo', args: [1] });
    while (recv() !== null) {}
    process.exit(0);
  }
  if (code.startsWith('FRAGMENT:')) {
    const body = Buffer.from(JSON.stringify({ id: id, payload: { type: 'done', value: wireOf(JSON.parse(code.slice(9))) } }), 'utf8');
    const frame = Buffer.alloc(body.length + 4);
    frame.writeUInt32BE(body.length, 0);
    body.copy(frame, 4);
    for (const byte of frame) { fs.writeSync(1, Buffer.from([byte])); sleep(2); }
    return;
  }
  if (code.startsWith('HANG:')) { sleep(parseInt(code.slice(5), 10)); done(id, {}); return; }
  if (code.startsWith('THROW:')) { done(id, { error: { kind: 'exception', message: code.slice(6) } }); return; }
  if (code === 'BADVAL') { done(id, { value: { a: 1 } }); return; }
  if (code.startsWith('CRASH:')) { fs.writeSync(2, 'crashing now'); process.exit(parseInt(code.slice(6), 10)); }
  done(id, { error: { kind: 'exception', message: 'unknown program: ' + code } });
}
const argv = process.argv;
if (argv.includes('silent')) { sleep(60000); process.exit(0); }
if (argv.includes('crashnow')) { process.exit(5); }
if (argv.includes('garbage')) {
  send({ type: 'ready' });
  const body = Buffer.from('this is not json', 'utf8');
  const head = Buffer.alloc(4);
  head.writeUInt32BE(body.length, 0);
  fs.writeSync(1, head);
  fs.writeSync(1, body);
  sleep(60000);
  process.exit(0);
}
if (argv.includes('oversized')) {
  send({ type: 'ready' });
  const head = Buffer.alloc(4);
  head.writeUInt32BE(100000000, 0);
  fs.writeSync(1, head);
  sleep(60000);
  process.exit(0);
}
send({ type: 'ready' });
for (;;) {
  const msg = recv();
  if (!msg) break;
  if (msg.type === 'run') evaluate(msg);
}
process.exit(0);
"#;

fn test_config(max_output_bytes: usize) -> HostConfig {
    HostConfig {
        timeout_ms: 5_000,
        max_timeout_ms: 8_000,
        max_output_bytes,
        max_message_bytes: 65_536,
        max_pending_calls: 16,
        grace_ms: 1_000,
        idle_timeout_ms: None,
        max_old_generation_size_mb: Some(48),
    }
}

fn launch(mode: &str) -> LaunchConfig {
    LaunchConfig::new(
        "node",
        vec!["-e".to_owned(), CHILD_SCRIPT.to_owned(), mode.to_owned()],
    )
}

fn policy_key(session: Option<&str>) -> PolicyKey {
    PolicyKey::new(
        SandboxMode::WorkspaceWrite,
        r"C:\workspace".to_owned(),
        session.map(str::to_owned),
    )
}

fn tools() -> Vec<HostNamespace> {
    vec![HostNamespace::new("tools")
        .with_fn(
            "echo",
            FnBinding::shared(|args: Value| async move { Ok(args) }),
        )
        .with_fn(
            "boom",
            FnBinding::shared(|_args: Value| async move { Err("binding refused".to_owned()) }),
        )
        .with_fn(
            "slow",
            FnBinding::shared(|args: Value| async move {
                tokio::time::sleep(Duration::from_millis(120)).await;
                Ok(args)
            }),
        )]
}

/// `slow` bindings instrumented to prove concurrent dispatch: the peak
/// in-flight count is observable after the run.
fn instrumented_tools(max_inflight: Arc<AtomicUsize>) -> Vec<HostNamespace> {
    vec![HostNamespace::new("tools").with_fn(
        "slow",
        FnBinding::shared(move |args: Value| {
            let max_inflight = max_inflight.clone();
            async move {
                let now = INFLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
                max_inflight.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(120)).await;
                INFLIGHT.fetch_sub(1, Ordering::SeqCst);
                Ok(args)
            }
        }),
    )]
}

static INFLIGHT: AtomicUsize = AtomicUsize::new(0);

fn request(program: &str) -> RunRequest {
    RunRequest {
        program: program.to_owned(),
        cwd: std::env::current_dir()
            .expect("a working directory")
            .to_string_lossy()
            .to_string(),
        timeout: TimeoutRequest::Budget(5_000),
        bindings: tools(),
        policy: policy_key(None),
        state_owner: None,
    }
}

fn owner_request(owner: &str, program: &str) -> RunRequest {
    let mut request = request(program);
    request.state_owner = Some(owner.to_owned());
    request
}

fn session_request(owner: &str, program: &str, session: &str, mode: SandboxMode) -> RunRequest {
    let mut request = request(program);
    request.state_owner = Some(owner.to_owned());
    request.policy = PolicyKey::new(mode, r"C:\workspace".to_owned(), Some(session.to_owned()));
    request
}

async fn persistent_host(idle_timeout_ms: Option<u64>) -> PersistentHost {
    let config = HostConfig {
        idle_timeout_ms,
        ..test_config(4_096)
    };
    PersistentHost::new(config, launch("persistent")).expect("a valid persistent host")
}

// ---------------------------------------------------------------------------
// Configuration and environment (pure)
// ---------------------------------------------------------------------------

#[test]
fn config_rejects_zero_knobs() {
    let config = HostConfig {
        timeout_ms: 0,
        ..HostConfig::default()
    };
    assert!(matches!(
        config.validate(),
        Err(ConfigError::Positive { field: "timeoutMs" })
    ));
    let config = HostConfig {
        grace_ms: 0,
        ..HostConfig::default()
    };
    assert!(matches!(
        config.validate(),
        Err(ConfigError::Positive { field: "graceMs" })
    ));
}

#[test]
fn config_rejects_impossible_limits() {
    let config = HostConfig {
        max_output_bytes: 3,
        ..HostConfig::default()
    };
    assert!(matches!(
        config.validate(),
        Err(ConfigError::OutputTooSmall)
    ));
    let config = HostConfig {
        max_message_bytes: u32::MAX as usize + 1,
        ..HostConfig::default()
    };
    assert!(matches!(
        config.validate(),
        Err(ConfigError::MessageTooLarge)
    ));
    let config = HostConfig {
        idle_timeout_ms: Some(0),
        ..HostConfig::default()
    };
    assert!(matches!(config.validate(), Err(ConfigError::IdleTimeout)));
}

#[test]
fn startup_environment_keeps_only_launch_variables() {
    let environment = startup_environment();
    assert!(
        !environment.is_empty(),
        "a host environment always carries PATH or TEMP"
    );
    for (key, _) in environment {
        let upper = key.to_ascii_uppercase();
        assert!(
            STARTUP_ENVIRONMENT_NAMES.contains(&upper.as_str()),
            "unexpected startup variable {key}"
        );
    }
}

#[test]
fn resolve_clamps_and_refuses_budgets() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    assert_eq!(
        host.resolve_timeout(TimeoutRequest::Budget(50_000))
            .expect("clamped"),
        Timeout::Budget(8_000)
    );
    assert_eq!(
        host.resolve_timeout(TimeoutRequest::Default)
            .expect("default"),
        Timeout::Budget(5_000)
    );
    assert_eq!(
        host.resolve_timeout(TimeoutRequest::NoDeadline)
            .expect("no deadline"),
        Timeout::NoDeadline
    );
    assert!(matches!(
        host.resolve_timeout(TimeoutRequest::Budget(0)),
        Err(PolicyError::InvalidTimeout { .. })
    ));
}

// ---------------------------------------------------------------------------
// One-shot runs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn oneshot_returns_value_and_logs() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let result = host
        .run(request("LOG:hello from node|\"done\""))
        .await
        .expect("the run resolves");
    assert!(result.error.is_none());
    assert_eq!(result.logs, vec!["hello from node".to_owned()]);
    assert_eq!(result.value, Some(json!("done")));
    assert_eq!(
        result.sandbox.as_ref().map(|sandbox| sandbox.mode),
        Some(SandboxMode::WorkspaceWrite)
    );
}

#[tokio::test]
async fn oneshot_honors_cwd() {
    let dir = std::env::temp_dir().join("dsh-ptc-host-cwd");
    std::fs::create_dir_all(&dir).expect("a temp directory");
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let mut request = request("CWD");
    request.cwd = dir.to_string_lossy().to_string();
    let result = host.run(request).await.expect("the run resolves");
    let observed = result
        .value
        .expect("a cwd value")
        .as_str()
        .expect("a string cwd")
        .replace('/', "\\")
        .to_ascii_lowercase();
    let expected = dir
        .to_string_lossy()
        .replace('/', "\\")
        .to_ascii_lowercase();
    assert_eq!(observed, expected);
}

#[tokio::test]
async fn oneshot_reports_program_exceptions() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let result = host
        .run(request("THROW:boom happened"))
        .await
        .expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::Exception));
    assert_eq!(result.error.expect("an error").message, "boom happened");
    assert!(result.value.is_none());
}

#[tokio::test]
async fn oneshot_rejects_invalid_completion_output() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let result = host.run(request("BADVAL")).await.expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::InvalidOutput));
    assert!(result
        .error
        .expect("an error")
        .message
        .contains("lossless JSON"));
}

#[tokio::test]
async fn oneshot_enforces_output_limit() {
    // 40 log entries x 12 serialized bytes each crosses the 120-byte cap.
    let host = PtcHost::new(test_config(120), launch("once")).expect("a valid host");
    let result = host
        .run(request("BIGLOG:40"))
        .await
        .expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::OutputLimit));
    assert!(
        serde_json::to_string(&result.logs)
            .expect("logs serialize")
            .len()
            <= 120,
        "the retained prefix still fits the cap"
    );
    assert!(result
        .error
        .expect("an error")
        .message
        .contains("outer output exceeded 120 bytes"));
}

#[tokio::test]
async fn oneshot_decodes_byte_by_byte_frames() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let result = host
        .run(request(r#"FRAGMENT:{"ok":true}"#))
        .await
        .expect("the run resolves");
    assert!(result.error.is_none());
    assert_eq!(result.value, Some(json!({"ok": true})));
}

#[tokio::test]
async fn oneshot_fails_protocol_on_garbage_frames() {
    let host = PtcHost::new(test_config(4_096), launch("garbage")).expect("a valid host");
    let result = host.run(request("PID")).await.expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::Protocol));
}

#[tokio::test]
async fn oneshot_fails_protocol_on_oversized_frames() {
    let host = PtcHost::new(test_config(4_096), launch("oversized")).expect("a valid host");
    let result = host.run(request("PID")).await.expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::Protocol));
}

#[tokio::test]
async fn oneshot_reports_crash_as_worker_exit_with_diagnostics() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let result = host
        .run(request("CRASH:3"))
        .await
        .expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::WorkerExit));
    let message = result.error.expect("an error").message;
    assert!(message.contains('3'), "exit code in message: {message}");
    assert!(message.contains("crashing now"), "stderr tail: {message}");
}

#[tokio::test]
async fn oneshot_reports_pre_readiness_crash() {
    let host = PtcHost::new(test_config(4_096), launch("crashnow")).expect("a valid host");
    let result = host.run(request("PID")).await.expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::WorkerExit));
    assert!(result
        .error
        .expect("an error")
        .message
        .contains("exit code 5"));
}

#[tokio::test]
async fn oneshot_bridges_binding_calls() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let result = host
        .run(request(r#"CALL:{"hello":[1,2,true,null]}"#))
        .await
        .expect("the run resolves");
    assert!(result.error.is_none());
    assert_eq!(result.value, Some(json!({"hello": [1, 2, true, null]})));
}

#[tokio::test]
async fn oneshot_bridges_binding_rejections() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let result = host
        .run(request("CALLFAIL"))
        .await
        .expect("the run resolves");
    assert!(
        result.error.is_none(),
        "unexpected error: {:?}",
        result.error
    );
    assert_eq!(result.value, Some(json!(["binding refused"])));
}

#[tokio::test]
async fn oneshot_dispatches_concurrent_binding_calls() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let max_inflight = Arc::new(AtomicUsize::new(0));
    let mut request = request("CALL2");
    request.bindings = instrumented_tools(max_inflight.clone());
    let result = host.run(request).await.expect("the run resolves");
    assert!(
        result.error.is_none(),
        "unexpected error: {:?}",
        result.error
    );
    assert_eq!(result.value, Some(json!([[1], [2]])));
    assert!(
        max_inflight.load(Ordering::SeqCst) >= 2,
        "both binding calls must be in flight at once"
    );
}

#[tokio::test]
async fn oneshot_fails_protocol_on_unknown_binding() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let result = host
        .run(request("CALLUNKNOWN"))
        .await
        .expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::Protocol));
    assert!(result
        .error
        .expect("an error")
        .message
        .contains("undeclared binding"));
}

#[tokio::test]
async fn oneshot_fails_protocol_on_call_id_desync() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let result = host
        .run(request("CALLBADID"))
        .await
        .expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::Protocol));
    assert!(result
        .error
        .expect("an error")
        .message
        .contains("binding call identity"));
}

#[tokio::test]
async fn oneshot_times_out_before_readiness() {
    let config = HostConfig {
        timeout_ms: 400,
        max_timeout_ms: 1_000,
        ..test_config(4_096)
    };
    let host = PtcHost::new(config, launch("silent")).expect("a valid host");
    let started = Instant::now();
    let mut request = request("PID");
    request.timeout = TimeoutRequest::Budget(400);
    let result = host.run(request).await.expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::Timeout));
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "the budget, not the hang, ends the run"
    );
}

#[tokio::test]
async fn oneshot_times_out_mid_evaluation_and_kills_the_child() {
    let config = HostConfig {
        timeout_ms: 400,
        max_timeout_ms: 1_000,
        ..test_config(4_096)
    };
    let host = PtcHost::new(config, launch("once")).expect("a valid host");
    let mut request = request("HANG:60000");
    request.timeout = TimeoutRequest::Budget(400);
    let result = host.run(request).await.expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::Timeout));
    assert!(result
        .error
        .expect("an error")
        .message
        .contains("execution deadline reached (400ms)"));
}

#[tokio::test]
async fn oneshot_concurrent_runs_do_not_cross_talk() {
    let host = Arc::new(PtcHost::new(test_config(4_096), launch("once")).expect("a valid host"));
    let first_host = host.clone();
    let second_host = host.clone();
    let first = tokio::spawn(async move { first_host.run(request("ECHO:1")).await });
    let second = tokio::spawn(async move { second_host.run(request("ECHO:2")).await });
    let first = first.await.expect("joined").expect("the run resolves");
    let second = second.await.expect("joined").expect("the run resolves");
    assert_eq!(first.value, Some(json!(1)));
    assert_eq!(second.value, Some(json!(2)));
}

#[tokio::test]
async fn oneshot_rejects_relative_cwd() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let mut request = request("PID");
    request.cwd = "relative/path".to_owned();
    assert!(matches!(
        host.run(request).await,
        Err(HostError::Resolve(_))
    ));
}

#[tokio::test]
async fn oneshot_refuses_reserved_binding_globals() {
    let host = PtcHost::new(test_config(4_096), launch("once")).expect("a valid host");
    let mut request = request("PID");
    request.bindings = vec![HostNamespace::new("console")];
    assert!(matches!(
        host.run(request).await,
        Err(HostError::Bindings(_))
    ));
}

// ---------------------------------------------------------------------------
// Persistent pool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn persistent_reuses_process_across_runs() {
    let host = persistent_host(None).await;
    let first = host
        .run(owner_request("agent-1", "PID"))
        .await
        .expect("the run resolves");
    let second = host
        .run(owner_request("agent-1", "PID"))
        .await
        .expect("the run resolves");
    assert!(
        first.error.is_none() && second.error.is_none(),
        "first: {:?}, second: {:?}",
        first.error,
        second.error
    );
    assert_eq!(first.value, second.value, "one child serves both runs");
    host.destroy_all().await.expect("cleanup");
}

#[tokio::test]
async fn persistent_distinct_owners_get_distinct_processes() {
    let host = persistent_host(None).await;
    let first = host
        .run(owner_request("agent-1", "PID"))
        .await
        .expect("the run resolves");
    let second = host
        .run(owner_request("agent-2", "PID"))
        .await
        .expect("the run resolves");
    assert_ne!(first.value, second.value);
    host.destroy_all().await.expect("cleanup");
}

#[tokio::test]
async fn persistent_state_survives_runs_and_exceptions() {
    let host = persistent_host(None).await;
    host.run(owner_request("agent-1", "SET:x=kept"))
        .await
        .expect("the run resolves");
    let failed = host
        .run(owner_request("agent-1", "THROW:later failure"))
        .await
        .expect("the run resolves");
    assert_eq!(failed.failure_kind(), Some(PtcFailureKind::Exception));
    let retained = host
        .run(owner_request("agent-1", "GET:x"))
        .await
        .expect("the run resolves");
    assert!(retained.error.is_none());
    assert_eq!(retained.value, Some(json!("kept")));
    host.destroy_all().await.expect("cleanup");
}

#[tokio::test]
async fn persistent_rejects_concurrent_runs_for_one_owner() {
    let host = Arc::new(persistent_host(None).await);
    let first_host = host.clone();
    let first = tokio::spawn(async move {
        first_host
            .run(owner_request("owner", "HANG:800"))
            .await
            .expect("the run resolves")
    });
    tokio::time::sleep(Duration::from_millis(150)).await;
    let second = host.run(owner_request("owner", "PID")).await;
    assert!(matches!(second, Err(HostError::Policy(PolicyError::Busy))));
    first.await.expect("joined");
    host.destroy_all().await.expect("cleanup");
}

#[tokio::test]
async fn persistent_rejects_policy_change() {
    let host = persistent_host(None).await;
    host.run(owner_request("owner", "PID"))
        .await
        .expect("the run resolves");
    let mut changed = owner_request("owner", "PID");
    changed.policy = PolicyKey::new(
        SandboxMode::DangerFullAccess,
        r"C:\workspace".to_owned(),
        None,
    );
    assert!(matches!(
        host.run(changed).await,
        Err(HostError::Policy(PolicyError::PolicyChange))
    ));
    host.destroy_all().await.expect("cleanup");
}

#[tokio::test]
async fn persistent_mode_change_invalidates_the_process() {
    let host = persistent_host(None).await;
    let request = session_request("owner", "PID", "s1", SandboxMode::ReadOnly);
    let before = host.run(request.clone()).await.expect("the run resolves");
    host.invalidate("s1", SandboxMode::DangerFullAccess)
        .await
        .expect("invalidation settles");
    let (_, state) = &host.status()[0];
    assert!(state.closed, "an invalidated environment is closed");
    let after = host.run(request).await.expect("the run resolves");
    assert_ne!(before.value, after.value, "a fresh child replaces it");
    host.destroy_all().await.expect("cleanup");
}

#[tokio::test]
async fn persistent_destroy_then_respawn_collects_exit() {
    let host = persistent_host(None).await;
    let before = host
        .run(owner_request("owner", "PID"))
        .await
        .expect("the run resolves");
    host.destroy("owner").await.expect("cleanup");
    let exit = host.exit_info("owner").await.expect("an exit record");
    assert!(exit.code.is_some(), "the exit code was collected");
    let after = host
        .run(owner_request("owner", "PID"))
        .await
        .expect("the run resolves");
    assert_ne!(before.value, after.value, "a destroyed owner respawns");
    host.destroy_all().await.expect("cleanup");
}

#[tokio::test]
async fn persistent_idle_timeout_reaps_the_process() {
    let host = persistent_host(Some(120)).await;
    let before = host
        .run(owner_request("owner", "PID"))
        .await
        .expect("the run resolves");
    tokio::time::sleep(Duration::from_millis(600)).await;
    let (_, state) = &host.status()[0];
    assert!(state.closed, "the reaper closed the idle environment");
    let after = host
        .run(owner_request("owner", "PID"))
        .await
        .expect("the run resolves");
    assert_ne!(before.value, after.value, "a reaped owner respawns");
    host.destroy_all().await.expect("cleanup");
}

#[tokio::test]
async fn persistent_timeout_discards_retained_state() {
    let host = persistent_host(None).await;
    host.run(owner_request("owner", "SET:x=1"))
        .await
        .expect("the run resolves");
    let mut hung = owner_request("owner", "HANG:30000");
    hung.timeout = TimeoutRequest::Budget(300);
    let result = host.run(hung).await.expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::Timeout));
    let message = result.error.expect("an error").message;
    assert!(message.contains("execution deadline reached (300ms)"));
    assert!(message.contains("retained variables were cleared"));
    let (_, state) = &host.status()[0];
    assert!(state.closed, "a timed-out environment is closed");
    let fresh = host
        .run(owner_request("owner", "GET:x"))
        .await
        .expect("the run resolves");
    assert_eq!(fresh.failure_kind(), Some(PtcFailureKind::Exception));
    assert!(fresh.error.expect("an error").message.contains("undefined"));
    host.destroy_all().await.expect("cleanup");
}

#[tokio::test]
async fn persistent_crash_discards_retained_state_and_collects_exit() {
    let host = persistent_host(None).await;
    host.run(owner_request("owner", "SET:x=1"))
        .await
        .expect("the run resolves");
    let result = host
        .run(owner_request("owner", "CRASH:3"))
        .await
        .expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::WorkerExit));
    let message = result.error.expect("an error").message;
    assert!(message.contains('3'), "exit code in message: {message}");
    assert!(message.contains("retained variables were cleared"));
    let exit = host.exit_info("owner").await.expect("an exit record");
    assert_eq!(exit.code, Some(3));
    let fresh = host
        .run(owner_request("owner", "GET:x"))
        .await
        .expect("the run resolves");
    assert_eq!(fresh.failure_kind(), Some(PtcFailureKind::Exception));
    host.destroy_all().await.expect("cleanup");
}

#[tokio::test]
async fn persistent_output_limit_discards_retained_state() {
    let config = HostConfig {
        idle_timeout_ms: None,
        ..test_config(120)
    };
    let host = PersistentHost::new(config, launch("persistent")).expect("a valid host");
    host.run(owner_request("owner", "SET:x=1"))
        .await
        .expect("the run resolves");
    let result = host
        .run(owner_request("owner", "BIGLOG:40"))
        .await
        .expect("the run resolves");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::OutputLimit));
    assert!(result
        .error
        .expect("an error")
        .message
        .contains("retained variables were cleared; outer output exceeded 120 bytes"));
    let fresh = host
        .run(owner_request("owner", "GET:x"))
        .await
        .expect("the run resolves");
    assert_eq!(fresh.failure_kind(), Some(PtcFailureKind::Exception));
    host.destroy_all().await.expect("cleanup");
}

#[tokio::test]
async fn persistent_destroy_aborts_the_active_run() {
    let host = Arc::new(persistent_host(None).await);
    let first_host = host.clone();
    let first = tokio::spawn(async move {
        first_host
            .run(owner_request("owner", "HANG:30000"))
            .await
            .expect("the run resolves")
    });
    tokio::time::sleep(Duration::from_millis(150)).await;
    host.destroy("owner").await.expect("cleanup");
    let result = first.await.expect("joined");
    assert_eq!(result.failure_kind(), Some(PtcFailureKind::Abort));
    assert!(result.error.expect("an error").message.contains("disposed"));
    let (_, state) = &host.status()[0];
    assert!(state.closed);
    host.destroy_all().await.expect("cleanup");
}
