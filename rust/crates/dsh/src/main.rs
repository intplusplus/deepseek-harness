//! Minimal headless-profile skeleton over the Rust foundation.
//!
//! Exercises the whole surface end to end: the plugin kernel (services,
//! observer and waterfall events, reversible effects), one scripted session
//! turn through the core event vocabulary, plugin-declared extension events,
//! the pure folds, and JSONL publication followed by a verified reload.

#![forbid(unsafe_code)]

use std::{
    env,
    error::Error,
    fs,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use dsh_agent::{fold_inbox, Agent, InboxSplice, InboxTarget};
use dsh_agent_loop::AgentLoop;
use dsh_llm::{ContentBlock, MessageId, SystemMessage, ToolSchema, UserMessage};
use dsh_plugin::{Plugin, PluginHost, Scope};
use dsh_provider::{text_response, tool_call_response, ScriptedProvider};
use dsh_session::{
    derive_messages, fold_request_header, fold_turn_boundary, jsonl, ExtensionEvent,
    RequestContext, Session, SessionEvent, SessionId, SessionMeta, SessionSeq, SystemPromptUpdate,
};
use dsh_tools::{
    argument, CancelSignal, ToolCallInput, ToolExecutionMode, ToolExecutionResult, ToolHandler,
    ToolRegistry,
};
use serde::{Deserialize, Serialize};

/// A typed observer event (the live domain).
#[derive(Debug)]
struct SessionAppended {
    seq: SessionSeq,
}

/// A typed waterfall event (the extension-point domain): the pre-step decision.
#[derive(Debug, Default)]
struct PreStepDecision {
    accepted: bool,
    notes: Vec<String>,
}

/// A plugin-provided service (the capability seam).
#[derive(Debug)]
struct DemoRoute {
    provider: String,
    model: String,
}

/// A service only the deliberately failing plugin would have registered.
#[derive(Debug)]
struct MarkerService;

/// A real tool: the handler owns its schema, its mode, and its body.
struct EchoTool;

impl ToolHandler for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "echo",
            "Echo text back",
            serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } }
            }),
        )
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Parallel
    }

    fn call(&self, input: &ToolCallInput, _cancel: &CancelSignal) -> ToolExecutionResult {
        let text = argument(input, "text")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        ToolExecutionResult::success(vec![ContentBlock::text(text.clone())])
            .with_value(serde_json::json!({ "echoed": text }))
    }
}

/// A plugin-declared extension event payload (the compaction seam).
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CompactionSummary {
    summary: String,
}

/// The kernel plugin: one service, one observer, one waterfall listener.
struct KernelPlugin;

impl Plugin for KernelPlugin {
    fn name(&self) -> &str {
        "demo-kernel"
    }

    fn load(&self, scope: &mut Scope) -> Result<(), Box<dyn Error + Send + Sync>> {
        scope.set(Arc::new(DemoRoute {
            provider: "demo".to_string(),
            model: "demo-1".to_string(),
        }));
        scope.observe(|event: &SessionAppended| {
            println!(
                "    [observer] session event committed at seq {}",
                event.seq.get()
            );
        });
        scope.waterfall(
            |mut decision: PreStepDecision,
             next: &mut dyn FnMut(
                PreStepDecision,
            )
                -> Result<PreStepDecision, dsh_plugin::EventError>| {
                decision.notes.push("audited-by-kernel".to_string());
                next(decision)
            },
        );
        Ok(())
    }
}

/// A plugin whose load fails halfway: every effect it made must unwind.
struct FailingPlugin;

impl Plugin for FailingPlugin {
    fn name(&self) -> &str {
        "demo-failing"
    }

    fn load(&self, scope: &mut Scope) -> Result<(), Box<dyn Error + Send + Sync>> {
        scope.set(Arc::new(MarkerService));
        scope.defer(|| println!("    [rollback] failed plugin's deferred effect ran"));
        Err("intentional load failure".into())
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn payload_of<T: Serialize>(
    value: &T,
) -> Result<serde_json::Map<String, serde_json::Value>, Box<dyn Error>> {
    match serde_json::to_value(value)? {
        serde_json::Value::Object(map) => Ok(map),
        other => Err(format!("expected a JSON object payload, got {other}").into()),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("== dsh (Rust foundation) - headless skeleton ==\n");

    // 1. The plugin kernel: load one plugin beside the (empty) core.
    let mut host = PluginHost::new();
    host.load(&KernelPlugin)?;
    println!("loaded plugins: {:?}\n", host.loaded_names());

    // 2. One scripted turn through the core event vocabulary.
    let route = host.context().require::<DemoRoute>()?;
    let id = SessionId::new("rust-foundation-demo")?;
    let mut session = Session::create(
        id.clone(),
        SessionMeta {
            cwd: Some(env::current_dir()?.to_string_lossy().into_owned()),
            ..Default::default()
        },
        now_ms(),
    );

    // The registry owns the tools; the request header is assembled from it, so
    // what the model is offered and what can run can never drift apart.
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool));

    // A real system prompt is still a subsystem, so the demo commits one
    // directly: it is derived history, not part of the request header.
    let system_message = SystemMessage::new(
        MessageId::random(),
        vec![ContentBlock::text("You are terse.")],
    );
    session.append(SessionEvent::system_message(1, 1, system_message))?;

    // The model is a scripted provider: deterministic, offline, and enough to
    // drive a genuine two-step turn.
    let provider = ScriptedProvider::new(vec![
        tool_call_response(
            &route.provider,
            &route.model,
            "call-1",
            "echo",
            r#"{"text":"hello"}"#,
        ),
        text_response(&route.provider, &route.model, "hello"),
    ]);

    let commit = |session: &mut Session,
                  event: SessionEvent|
     -> Result<SessionSeq, dsh_session::SessionError> {
        let seq = session.append(event)?;
        host.context().emit(&SessionAppended { seq });
        Ok(seq)
    };

    println!("== one driven turn ==");
    let before_turn = session.len();
    let report = {
        let mut driver = AgentLoop::new(
            &mut session,
            &provider,
            &registry,
            RequestContext {
                provider: route.provider.clone(),
                model: route.model.clone(),
                context_window: Some(128_000),
                system_prompt_update: Some(SystemPromptUpdate::InHistory),
            },
        );
        driver.send(UserMessage::new(
            MessageId::random(),
            vec![ContentBlock::text("Say hello in one word.")],
        ));
        driver.run_turn().await?
    };
    for seq in before_turn..session.len() {
        host.context().emit(&SessionAppended {
            seq: SessionSeq::new(seq).expect("in bounds"),
        });
    }
    println!(
        "turn {} ended after {} step(s): {:?}",
        report.turn, report.steps, report.reason
    );
    // 3. Plugin-declared extension events: declaration merging's Rust slot.
    println!("\n== extension events ==");
    commit(
        &mut session,
        SessionEvent::extension_event("compaction/start", serde_json::Map::new()),
    )?;
    commit(
        &mut session,
        SessionEvent::Extension(ExtensionEvent::new(
            "compaction/summary",
            payload_of(&CompactionSummary {
                summary: "greeting exchange".to_string(),
            })?,
        )),
    )?;
    commit(
        &mut session,
        SessionEvent::extension_event("compaction/end", serde_json::Map::new()),
    )?;
    let splice = InboxSplice {
        target: InboxTarget::NextTurn,
        start: 0,
        removed_count: None,
        inserted: vec![UserMessage::new(
            MessageId::random(),
            vec![ContentBlock::text("then say goodbye")],
        )],
        outcome: None,
    };
    commit(
        &mut session,
        SessionEvent::Extension(ExtensionEvent::new(
            "agent/inbox/spliced",
            payload_of(&splice)?,
        )),
    )?;

    // 4. The waterfall seam: listeners audit, then delegate.
    println!("\n== waterfall dispatch ==");
    let decision = host.context().dispatch(PreStepDecision {
        accepted: true,
        notes: Vec::new(),
    })?;
    println!(
        "decision: accepted={} notes={:?}",
        decision.accepted, decision.notes
    );

    // 5. The pure folds.
    println!("\n== folds ==");
    let header = fold_request_header(session.events()).expect("a request header was logged");
    println!(
        "request header: provider={} model={} tools={}",
        header.config.provider,
        header.config.model,
        header.tools.as_ref().map_or(0, |tools| tools.len())
    );
    let boundary = fold_turn_boundary(session.events());
    println!(
        "turn boundary: open={:?} last_step={:?} last_turn={}",
        boundary.open_turn_start_seq.map(|seq| seq.get()),
        boundary.last_step_boundary,
        boundary.last_turn
    );
    let messages = derive_messages(session.events());
    println!("derived {} model-visible messages:", messages.len());
    for message in &messages {
        println!(
            "  {:<9} {} block(s)",
            message.role(),
            message.content().len()
        );
    }
    let inbox = fold_inbox(session.events());
    println!(
        "inbox: next-turn={} next-step={}",
        inbox.next_turn.len(),
        inbox.next_step.len()
    );

    // 6. Persistence: publish, then reload and verify.
    println!("\n== jsonl persistence ==");
    let directory = env::temp_dir().join(format!("dsh-rust-demo-{}", std::process::id()));
    let _ = fs::remove_dir_all(&directory);
    let path = jsonl::write_session(&directory, &session)?;
    println!("published: {}", path.display());
    let reloaded = jsonl::read_session(&directory)?;
    println!(
        "reloaded session equals the original: {}",
        reloaded == session
    );
    println!("stored bytes: {}", fs::metadata(&path)?.len());
    let _ = fs::remove_dir_all(&directory);

    // 7. Plugin lifecycle: a failed load rolls back; an unload unwinds.
    println!("\n== plugin lifecycle ==");
    if let Err(error) = host.load(&FailingPlugin) {
        println!("failed load (expected): {error}");
    }
    println!(
        "marker service present after rollback: {}",
        host.context().get::<MarkerService>().is_some()
    );
    println!(
        "demo route still present: {}",
        host.context().get::<DemoRoute>().is_some()
    );
    host.unload("demo-kernel");
    println!(
        "after unload, demo route present: {}",
        host.context().get::<DemoRoute>().is_some()
    );
    println!("after unload, loaded plugins: {:?}", host.loaded_names());

    // 8. The agent face over the session.
    let agent = Agent::new(session.id().clone());
    println!(
        "\nagent {} owns {} events ({} pending inbox)",
        agent.id.as_str(),
        session.len(),
        inbox.len()
    );

    println!("\ndone.");
    Ok(())
}
