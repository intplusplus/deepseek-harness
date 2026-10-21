//! The framed control channel over one child's stdin/stdout.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node/src/channel.ts`
//! (`JsonChannel`): 4-byte big-endian length-prefixed JSON frames, a receive
//! side that refuses malformed framing as a *protocol* failure, and a send
//! side bounded by the same cap on queued bytes. The frame codec itself is
//! `dsh-ptc`'s — this module only wires it to pipes with two background
//! tasks, one per direction, so concurrent senders (binding replies) keep
//! frame order.
//!
//! The TypeScript channel runs over a dedicated third pipe; this host uses
//! the child's stdin/stdout directly, so raw program output arrives through
//! `log` frames and stderr stays a diagnostics channel.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use dsh_ptc::{encode_frame, FrameDecoder};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{ChildStdin, ChildStdout};
use tokio::sync::mpsc;

/// Why the channel stopped carrying traffic.
///
/// Mirrors `JsonChannel`'s two failure kinds: a *protocol* failure is a frame
/// the peer must never have written; an io failure is the pipe itself dying.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ChannelFailure {
    /// The pipe broke or the peer's end closed.
    #[error("control channel io failed: {0}")]
    Io(String),
    /// A frame violated the format or the queue budget.
    #[error("control channel protocol failed: {0}")]
    Protocol(String),
}

/// One message delivered from the child to the pump.
pub enum Inbound {
    /// One complete frame.
    Frame(Value),
    /// The stream refused a frame; the channel is done.
    Failed(ChannelFailure),
    /// The stream ended on a frame boundary.
    Closed,
}

/// The pump-owned receive side of a channel.
pub type InboundStream = mpsc::UnboundedReceiver<Inbound>;

/// The pump-shared send side of a channel.
///
/// Cloning is cheap; every clone enforces the same queued-byte budget, so
/// binding replies racing with the pump cannot overflow the child's pipe
/// buffer unboundedly.
#[derive(Clone)]
pub struct Channel {
    write_tx: mpsc::UnboundedSender<Vec<u8>>,
    queued: Arc<AtomicUsize>,
    failed: Arc<AtomicBool>,
    sealed: Arc<AtomicBool>,
    max_message_bytes: usize,
}

impl Channel {
    /// Send one frame, or refuse it.
    ///
    /// A frame the format cannot carry, or one that would push queued output
    /// over the cap, is a protocol failure — the caller stops the run rather
    /// than queueing what the child must never see.
    pub async fn send(&self, value: &Value) -> Result<(), ChannelFailure> {
        if self.failed.load(Ordering::SeqCst) {
            return Err(ChannelFailure::Io("control channel is closed".to_owned()));
        }
        let frame = encode_frame(value, self.max_message_bytes)
            .map_err(|error| ChannelFailure::Protocol(error.to_string()))?;
        let body_bytes = frame.len().saturating_sub(4);
        if self.queued.load(Ordering::SeqCst) + body_bytes > self.max_message_bytes {
            return Err(ChannelFailure::Protocol(format!(
                "control output exceeds {} queued bytes",
                self.max_message_bytes
            )));
        }
        self.queued.fetch_add(body_bytes, Ordering::SeqCst);
        self.write_tx
            .send(frame)
            .map_err(|_| ChannelFailure::Io("control channel is closed".to_owned()))
    }

    /// End the host side: no further writes are accepted and the child sees
    /// end-of-file on stdin once the writer task drains.
    pub fn seal(&self) {
        self.sealed.store(true, Ordering::SeqCst);
    }
}

/// Wire one child's stdin/stdout into a framed channel.
///
/// The returned [`Channel`] is the send side (shareable across tasks); the
/// [`InboundStream`] is the receive side, owned by whichever pump drives the
/// child.
pub fn wire(
    stdout: ChildStdout,
    stdin: ChildStdin,
    max_message_bytes: usize,
) -> (Channel, InboundStream) {
    let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
    let (write_tx, write_rx) = mpsc::unbounded_channel();
    let queued = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    let sealed = Arc::new(AtomicBool::new(false));

    tokio::spawn(reader_task(stdout, inbound_tx, max_message_bytes));
    tokio::spawn(writer_task(
        stdin,
        write_rx,
        queued.clone(),
        failed.clone(),
        sealed.clone(),
    ));

    (
        Channel {
            write_tx,
            queued,
            failed,
            sealed,
            max_message_bytes,
        },
        inbound_rx,
    )
}

async fn reader_task(
    mut stdout: ChildStdout,
    inbound_tx: mpsc::UnboundedSender<Inbound>,
    max_message_bytes: usize,
) {
    let mut decoder = FrameDecoder::new(max_message_bytes);
    let mut chunk = vec![0u8; 8192];
    loop {
        match stdout.read(&mut chunk).await {
            Ok(0) => {
                // End of stream: a half-read tail is a refusal, not silence.
                if let Err(error) = decoder.finish() {
                    let _ = inbound_tx
                        .send(Inbound::Failed(ChannelFailure::Protocol(error.to_string())));
                }
                let _ = inbound_tx.send(Inbound::Closed);
                return;
            }
            Ok(n) => match decoder.push(&chunk[..n]) {
                Ok(frames) => {
                    for frame in frames {
                        if inbound_tx.send(Inbound::Frame(frame)).is_err() {
                            return;
                        }
                    }
                }
                Err(error) => {
                    let _ = inbound_tx
                        .send(Inbound::Failed(ChannelFailure::Protocol(error.to_string())));
                    return;
                }
            },
            Err(error) => {
                let _ = inbound_tx.send(Inbound::Failed(ChannelFailure::Io(error.to_string())));
                return;
            }
        }
    }
}

async fn writer_task(
    mut stdin: ChildStdin,
    mut write_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    queued: Arc<AtomicUsize>,
    failed: Arc<AtomicBool>,
    sealed: Arc<AtomicBool>,
) {
    while let Some(frame) = write_rx.recv().await {
        queued.fetch_sub(frame.len().saturating_sub(4), Ordering::SeqCst);
        if stdin.write_all(&frame).await.is_err() {
            failed.store(true, Ordering::SeqCst);
            return;
        }
        if sealed.load(Ordering::SeqCst) {
            return;
        }
    }
}
