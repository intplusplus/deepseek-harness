//! One managed sandbox child: spawn, stderr capture, and exit collection.
//!
//! Ported from the subprocess seams of `process.ts` / `persistent.ts` plus
//! `output-stream.ts`: the host owns the child, kills it on teardown, waits
//! out a drain grace, and keeps a bounded stderr tail so a substrate death
//! can be diagnosed (`exitMessage` in the TypeScript provider).
//!
//! The child environment starts from `environment.ts`'s contract: stripped to
//! the startup variables a native executable needs, so model-written code
//! never sees host credentials.

#![forbid(unsafe_code)]

use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::{watch, Notify};

use crate::channel::{wire, Channel, InboundStream};
use crate::config::LaunchConfig;
use crate::env::startup_environment;

/// How a child's process exited.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExitInfo {
    /// The reported exit code, when the platform reported one.
    pub code: Option<i32>,
}

/// One spawned sandbox child with its framed control channel.
///
/// Dropping the handle asks the child to die; the exit watcher performs the
/// kill, so teardown never leaks a process even on an abrupt drop.
pub struct SandboxProcess {
    pid: u32,
    channel: Channel,
    stderr: Arc<Mutex<String>>,
    exit_rx: watch::Receiver<Option<ExitInfo>>,
    kill: Arc<Notify>,
}

impl SandboxProcess {
    /// Spawn one child.
    ///
    /// `args` is the full argument list (the heap flag included); `cwd` must
    /// already be validated as absolute. Returns the process and its inbound
    /// frame stream.
    pub fn spawn(
        launch: &LaunchConfig,
        args: &[String],
        cwd: &str,
        max_message_bytes: usize,
        stderr_max: usize,
    ) -> std::io::Result<(Self, InboundStream)> {
        let mut command = Command::new(&launch.executable);
        command.args(args).current_dir(cwd).env_clear();
        for (key, value) in startup_environment() {
            command.env(key, value);
        }
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            // CREATE_NO_WINDOW: a headless host must not pop consoles.
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command.spawn()?;
        let pid = child.id().unwrap_or(0);
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");

        let (channel, inbound) = wire(stdout, stdin, max_message_bytes);

        let tail = Arc::new(Mutex::new(String::new()));
        tokio::spawn(stderr_task(stderr, tail.clone(), stderr_max));

        let (exit_tx, exit_rx) = watch::channel(None);
        let kill = Arc::new(Notify::new());
        tokio::spawn(exit_task(child, exit_tx, kill.clone()));

        Ok((
            Self {
                pid,
                channel,
                stderr: tail,
                exit_rx,
                kill,
            },
            inbound,
        ))
    }

    /// The child's OS process id.
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// The shared send side of the control channel.
    pub fn channel(&self) -> &Channel {
        &self.channel
    }

    /// End the host side of the channel; the child sees stdin end-of-file.
    pub fn seal_stdin(&self) {
        self.channel.seal();
    }

    /// The last `max` bytes of stderr, as captured.
    pub fn stderr_tail(&self) -> String {
        self.stderr.lock().expect("stderr lock").clone()
    }

    /// A cloneable watcher for the child's exit outcome.
    pub fn exit_watcher(&self) -> watch::Receiver<Option<ExitInfo>> {
        self.exit_rx.clone()
    }

    /// Ask the child to die. Idempotent; safe after exit.
    pub fn kill(&self) {
        self.kill.notify_one();
    }

    /// Wait for the child's exit, however long it takes.
    pub async fn wait_exit(&self) -> ExitInfo {
        let mut exit_rx = self.exit_rx.clone();
        loop {
            if let Some(info) = exit_rx.borrow().as_ref() {
                return info.clone();
            }
            if exit_rx.changed().await.is_err() {
                return ExitInfo { code: None };
            }
        }
    }

    /// Kill the child and wait up to `grace` for it to exit.
    ///
    /// Returns whether the exit was observed inside the grace window — the
    /// TypeScript `terminateProcess` "drained" boolean.
    pub async fn terminate(&self, grace: Duration) -> bool {
        self.seal_stdin();
        self.kill();
        tokio::time::timeout(grace, self.wait_exit()).await.is_ok()
    }
}

impl Drop for SandboxProcess {
    fn drop(&mut self) {
        self.kill.notify_one();
    }
}

async fn exit_task(
    mut child: tokio::process::Child,
    exit_tx: watch::Sender<Option<ExitInfo>>,
    kill: Arc<Notify>,
) {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let _ = exit_tx.send(Some(ExitInfo {
                    code: status.code(),
                }));
                return;
            }
            Ok(None) => {}
            Err(_) => {
                let _ = exit_tx.send(Some(ExitInfo { code: None }));
                return;
            }
        }
        tokio::select! {
            biased;
            _ = kill.notified() => {
                // Killing an already-exited child reports an error we ignore.
                let _ = child.start_kill();
            }
            _ = tokio::time::sleep(Duration::from_millis(20)) => {}
        }
    }
}

async fn stderr_task(
    mut stderr: tokio::process::ChildStderr,
    tail: Arc<Mutex<String>>,
    max: usize,
) {
    let mut chunk = vec![0u8; 4096];
    loop {
        match stderr.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                let text = String::from_utf8_lossy(&chunk[..n]);
                let mut guard = tail.lock().expect("stderr lock");
                guard.push_str(&text);
                if guard.len() > max {
                    let mut split = guard.len() - max;
                    while !guard.is_char_boundary(split) {
                        split += 1;
                    }
                    guard.drain(..split);
                }
            }
        }
    }
}
