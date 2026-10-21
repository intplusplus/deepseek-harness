//! The evaluation pump shared by the one-shot and persistent paths.
//!
//! Ported from the frame-driven cores of `process.ts` (`execute`) and
//! `persistent.ts` (`Kernel::evaluate`): wait for readiness, dispatch one
//! run frame, then admit logs against the output ledger, dispatch binding
//! calls concurrently under the pending limits, and settle on the terminal
//! frame, a budget expiry, or substrate death.
//!
//! Host failures are synthesized exactly where the TypeScript provider
//! synthesizes them: a deadline kill becomes a `timeout` failure, an exit
//! before settling becomes `worker-exit` with the exit code and stderr tail,
//! and a broken frame becomes `protocol`. A dispose during the run becomes
//! `abort`, the persistent provider's dispose semantics.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dsh_ptc::protocol::{
    HostToProgram, ProgramBootData, ProgramEnvelope, ProgramFailure, ProgramFailureKind,
    ProgramToHost, Reply,
};
use dsh_ptc::{decode_json_wire, encode_json_wire, OutputLedger, Timeout};
use serde_json::Value;
use tokio::task::JoinSet;

use crate::bindings::NamespaceTable;
use crate::channel::{Channel, ChannelFailure, Inbound, InboundStream};
use crate::process::SandboxProcess;

/// How one evaluation settled, before ledger composition.
pub(crate) enum Terminal {
    /// A clean `done` frame with an optional completion value.
    Completed(Option<Value>),
    /// A `done` frame carrying a program-reported failure.
    ProgramFailure(ProgramFailure),
    /// The elapsed budget expired; the child was killed.
    Timeout(u64),
    /// The child died before settling.
    WorkerExit(String),
    /// The child sent traffic the protocol refuses.
    Protocol(String),
    /// The combined output cap was crossed.
    OutputLimit,
    /// The owning environment was disposed mid-run.
    Disposed,
}

/// A pump's full report.
pub(crate) struct Report {
    /// How the evaluation settled.
    pub terminal: Terminal,
    /// Every admitted log entry, in emission order.
    pub logs: Vec<String>,
}

/// One evaluation's fixed inputs.
pub(crate) struct Params {
    /// The evaluation id stamped on the run frame.
    pub eval_id: u64,
    /// Working directory for this evaluation.
    pub cwd: String,
    /// The boot payload.
    pub boot: ProgramBootData,
    /// The effective elapsed budget.
    pub timeout: Timeout,
    /// Maximum control frame and queued-write bytes.
    pub max_message_bytes: usize,
    /// Maximum simultaneous host binding calls.
    pub max_pending_calls: usize,
    /// Drain window after an observed exit, before declaring worker-exit.
    pub grace_ms: u64,
    /// Whether this child already completed its readiness handshake — true
    /// for every evaluation after the first on a retained process.
    pub already_ready: bool,
}

/// Drive one evaluation to its terminal on an already-spawned child.
pub(crate) async fn drive(
    process: &SandboxProcess,
    namespaces: &NamespaceTable,
    params: Params,
    disposed: Arc<AtomicBool>,
    ledger: &mut OutputLedger,
    inbound: &mut InboundStream,
) -> Report {
    let channel = process.channel().clone();
    let settled = Arc::new(AtomicBool::new(false));
    let pending = Arc::new(AtomicUsize::new(0));
    let pending_bytes = Arc::new(AtomicUsize::new(0));
    let mut join_set = JoinSet::new();
    let mut next_call: u64 = 1;
    let mut logs: Vec<String> = Vec::new();
    let channel_open = Arc::new(AtomicBool::new(true));
    let mut exit_rx = process.exit_watcher();
    let budget_ms = match params.timeout {
        Timeout::Budget(ms) => Some(ms),
        Timeout::NoDeadline => None,
    };
    let mut sleep = budget_ms.map(|ms| Box::pin(tokio::time::sleep(Duration::from_millis(ms))));
    // Set when the child's exit is observed: frames already written get a
    // bounded window before the death is classified, mirroring the
    // TypeScript provider's `setImmediate` deference to queued frames.
    let mut drain_deadline: Option<Instant> = None;

    let mut terminal: Option<Terminal> = None;
    let mut ready = params.already_ready;
    if ready {
        // A retained process handshake happens once; later evaluations go
        // straight to their run frame.
        terminal = dispatch_run(&params, &channel, process, &channel_open).await;
    }

    while terminal.is_none() {
        if let Some(deadline) = drain_deadline {
            let now = Instant::now();
            if now >= deadline {
                terminal = Some(classify_exit(process, &disposed).await);
                break;
            }
            match tokio::time::timeout(deadline - now, inbound.recv()).await {
                Ok(Some(Inbound::Frame(frame))) => {
                    if let Some(failure) = handle_frame(
                        frame,
                        &mut ready,
                        &params,
                        namespaces,
                        &channel,
                        process,
                        ledger,
                        &mut logs,
                        &mut next_call,
                        &pending,
                        &pending_bytes,
                        &settled,
                        &channel_open,
                        &mut join_set,
                    )
                    .await
                    {
                        terminal = Some(failure);
                    }
                }
                Ok(Some(Inbound::Failed(ChannelFailure::Protocol(message)))) => {
                    terminal = Some(Terminal::Protocol(message));
                }
                Ok(Some(_)) | Ok(None) | Err(_) => {
                    terminal = Some(classify_exit(process, &disposed).await);
                    break;
                }
            }
            continue;
        }

        tokio::select! {
            biased;
            message = recv_or_pending(inbound, channel_open.load(Ordering::SeqCst)) => {
                match message {
                    Some(Inbound::Frame(frame)) => {
                        if let Some(failure) = handle_frame(
                            frame, &mut ready, &params, namespaces, &channel, process,
                            ledger, &mut logs, &mut next_call, &pending, &pending_bytes,
                            &settled, &channel_open, &mut join_set,
                        )
                        .await
                        {
                            terminal = Some(failure);
                        }
                    }
                    Some(Inbound::Failed(ChannelFailure::Protocol(message))) => {
                        terminal = Some(Terminal::Protocol(message));
                    }
                    Some(Inbound::Failed(ChannelFailure::Io(_))) => {
                        // The pipe broke; let the exit watcher classify the death.
                        channel_open.store(false, Ordering::SeqCst);
                        process.kill();
                    }
                    Some(Inbound::Closed) | None => {
                        // The child ended its stream; only its exit remains.
                        channel_open.store(false, Ordering::SeqCst);
                    }
                }
            }
            _ = async {
                match sleep.as_mut() {
                    Some(sleep) => sleep.as_mut().await,
                    None => std::future::pending::<()>().await,
                }
            } => {
                process.kill();
                terminal = Some(Terminal::Timeout(budget_ms.unwrap_or(0)));
            }
            _ = async {
                loop {
                    if exit_rx.borrow().is_some() {
                        break;
                    }
                    if exit_rx.changed().await.is_err() {
                        break;
                    }
                }
            } => {
                drain_deadline =
                    Some(Instant::now() + Duration::from_millis(params.grace_ms.clamp(20, 200)));
            }
        }
    }

    // In-flight binding calls race the terminal; their replies are dropped,
    // exactly as the TypeScript provider drops them.
    settled.store(true, Ordering::SeqCst);
    join_set.detach_all();

    Report {
        terminal: terminal.expect("the pump always settles on a terminal"),
        logs,
    }
}

/// Read one inbound message, or wait forever while the channel is closed.
///
/// A closed channel must not spin the select: the exit watcher owns the
/// remaining outcomes.
async fn recv_or_pending(inbound: &mut InboundStream, open: bool) -> Option<Inbound> {
    if open {
        inbound.recv().await
    } else {
        std::future::pending().await
    }
}

/// Classify a death the pump observed without a terminal frame.
async fn classify_exit(process: &SandboxProcess, disposed: &AtomicBool) -> Terminal {
    if disposed.load(Ordering::SeqCst) {
        return Terminal::Disposed;
    }
    let exit = process.wait_exit().await;
    Terminal::WorkerExit(exit_failure_message(exit.code, &process.stderr_tail()))
}

/// Classify and act on one frame; `None` means the evaluation continues.
#[allow(clippy::too_many_arguments)]
async fn handle_frame(
    frame: Value,
    ready: &mut bool,
    params: &Params,
    namespaces: &NamespaceTable,
    channel: &Channel,
    process: &SandboxProcess,
    ledger: &mut OutputLedger,
    logs: &mut Vec<String>,
    next_call: &mut u64,
    pending: &Arc<AtomicUsize>,
    pending_bytes: &Arc<AtomicUsize>,
    settled: &Arc<AtomicBool>,
    channel_open: &Arc<AtomicBool>,
    join_set: &mut JoinSet<()>,
) -> Option<Terminal> {
    if !*ready {
        return handle_readiness(frame, ready, params, channel, process, channel_open).await;
    }
    let frame_bytes = serde_json::to_vec(&frame)
        .map(|bytes| bytes.len())
        .unwrap_or(0);
    match ProgramEnvelope::from_json(&frame) {
        Ok(envelope) => {
            if envelope.id != params.eval_id {
                return Some(Terminal::Protocol(
                    "frame does not belong to the active evaluation".to_owned(),
                ));
            }
            absorb_payload(
                envelope.payload,
                frame_bytes,
                params,
                namespaces,
                channel,
                ledger,
                logs,
                next_call,
                pending,
                pending_bytes,
                settled,
                join_set,
            )
        }
        Err(error) => {
            // The TypeScript provider distinguishes an ill-typed `done.value`
            // (invalid-output) from a frame that cannot be decoded at all
            // (protocol); `dsh-ptc`'s strict decoder reports both as errors.
            let is_done = frame
                .get("payload")
                .and_then(|payload| payload.get("type"))
                .and_then(Value::as_str)
                == Some("done");
            if is_done {
                Some(Terminal::ProgramFailure(ProgramFailure {
                    kind: ProgramFailureKind::InvalidOutput,
                    message: "program completion must be lossless JSON".to_owned(),
                }))
            } else {
                Some(Terminal::Protocol(format!(
                    "invalid sandbox frame: {error}"
                )))
            }
        }
    }
}

async fn handle_readiness(
    frame: Value,
    ready: &mut bool,
    params: &Params,
    channel: &Channel,
    process: &SandboxProcess,
    channel_open: &Arc<AtomicBool>,
) -> Option<Terminal> {
    match HostToProgram::from_json(&frame) {
        Ok(HostToProgram::Ready) => {
            *ready = true;
            dispatch_run(params, channel, process, channel_open).await
        }
        _ => Some(Terminal::Protocol("expected sandbox readiness".to_owned())),
    }
}

/// Send this evaluation's run frame.
async fn dispatch_run(
    params: &Params,
    channel: &Channel,
    process: &SandboxProcess,
    channel_open: &Arc<AtomicBool>,
) -> Option<Terminal> {
    let run = HostToProgram::Run {
        id: params.eval_id,
        cwd: params.cwd.clone(),
        data: params.boot.clone(),
    };
    match channel.send(&run.to_json()).await {
        Ok(()) => None,
        Err(ChannelFailure::Protocol(message)) => Some(Terminal::Protocol(message)),
        Err(ChannelFailure::Io(_)) => {
            channel_open.store(false, Ordering::SeqCst);
            process.kill();
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn absorb_payload(
    payload: ProgramToHost,
    frame_bytes: usize,
    params: &Params,
    namespaces: &NamespaceTable,
    channel: &Channel,
    ledger: &mut OutputLedger,
    logs: &mut Vec<String>,
    next_call: &mut u64,
    pending: &Arc<AtomicUsize>,
    pending_bytes: &Arc<AtomicUsize>,
    settled: &Arc<AtomicBool>,
    join_set: &mut JoinSet<()>,
) -> Option<Terminal> {
    match payload {
        ProgramToHost::Log { text } => {
            if ledger.admit(&text, logs) {
                None
            } else {
                // The overflowing entry joins the logs so the limit result
                // retains a fitting prefix of it.
                logs.push(text);
                Some(Terminal::OutputLimit)
            }
        }
        ProgramToHost::OutputLimit => Some(Terminal::OutputLimit),
        ProgramToHost::Done { value, error } => match error {
            Some(failure) => Some(Terminal::ProgramFailure(failure)),
            None => match value {
                None => Some(Terminal::Completed(None)),
                Some(wire) => match decode_json_wire(&wire) {
                    Some(value) => Some(Terminal::Completed(Some(value))),
                    None => Some(Terminal::ProgramFailure(ProgramFailure {
                        kind: ProgramFailureKind::InvalidOutput,
                        message: "program completion must be lossless JSON".to_owned(),
                    })),
                },
            },
        },
        ProgramToHost::Call {
            id,
            global,
            name,
            args,
        } => {
            if id != *next_call {
                return Some(Terminal::Protocol(
                    "invalid binding call identity".to_owned(),
                ));
            }
            *next_call += 1;
            let Some(args) = decode_json_wire(&args) else {
                return Some(Terminal::Protocol(
                    "binding arguments must be lossless JSON".to_owned(),
                ));
            };
            let now_pending = pending.fetch_add(1, Ordering::SeqCst) + 1;
            let now_bytes = pending_bytes.fetch_add(frame_bytes, Ordering::SeqCst) + frame_bytes;
            if now_pending > params.max_pending_calls || now_bytes > params.max_message_bytes {
                return Some(Terminal::Protocol(
                    "pending binding calls exceed configured limits".to_owned(),
                ));
            }
            let Some(function) = namespaces
                .get(&global)
                .and_then(|namespace| namespace.functions.get(&name))
            else {
                return Some(Terminal::Protocol(
                    "program requested an undeclared binding".to_owned(),
                ));
            };
            let function = function.clone();
            let channel = channel.clone();
            let settled = settled.clone();
            let pending = pending.clone();
            let pending_bytes = pending_bytes.clone();
            join_set.spawn(async move {
                let outcome = function.call(args).await;
                let reply = match outcome {
                    Ok(value) => Reply::Ok {
                        id,
                        value: encode_json_wire(&value),
                    },
                    Err(message) => Reply::Error { id, message },
                };
                pending.fetch_sub(1, Ordering::SeqCst);
                pending_bytes.fetch_sub(frame_bytes, Ordering::SeqCst);
                if !settled.load(Ordering::SeqCst) {
                    let _ = channel.send(&HostToProgram::Reply(reply).to_json()).await;
                }
            });
            None
        }
    }
}

/// The TypeScript provider's per-call `processFinished` diagnostic.
fn exit_failure_message(code: Option<i32>, stderr: &str) -> String {
    let code = code
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_owned());
    let mut message = format!("Node process exited before completing (exit code {code})");
    if !stderr.is_empty() {
        message.push_str(": ");
        message.push_str(stderr);
    }
    message
}
