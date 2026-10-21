//! `dsh-agent-loop` — drives one session through turn and step boundaries.
//!
//! Ported from `packages/core/agent-loop`. The driver is deliberately small:
//! every request is derived from the log, every outcome is an event, and the
//! driver itself holds no conversational state beyond the turn and step
//! counters. That is what makes a replay a re-derivation rather than a
//! reconstruction.
//!
//! ## Shape of a turn
//!
//! ```text
//! turn/start
//!   user/message  (claimed input)
//!   step/start
//!     request/header, request/context   (when they change)
//!     compaction/start, compaction/summary, user/message, compaction/end
//!     llm/retry, llm/retry-started      (per failed attempt that recovers)
//!     assistant/message                 (what the model said)
//!     tool/call, tool/result            (scheduled in model order)
//!   step/end
//!   ... more steps while the model keeps asking for tools
//! turn/end
//! ```
//!
//! A step that returns no tool calls ends the turn. A step whose tool results
//! conclude it also ends the turn. Everything else opens another step.
//!
//! ## Recovery
//!
//! A provider failure is offered to `dsh_retry::plan_retry` before the turn
//! is ended. Recovery only runs when a policy is configured for the request's
//! provider, so a loop without retry behaves exactly as it did before. The
//! retry wait is asynchronous and races the configured cancellation source.
//!
//! ## Compaction
//!
//! `dsh-compaction` owns the transaction; this crate owns the trigger points.
//! With [`CompactionConfig`] installed, each step first measures the request its
//! surface would produce (through the host's [`TokenMeter`]) and compacts when
//! the routed threshold is crossed, and a canonical context-window overflow is
//! offered one forced reduction before the step is retried. Both are optional:
//! without a config the log carries no `compaction/*` events at all.
//!
//! ## Not ported
//!
//! System-prompt assembly and the `agent/pre-step` waterfall are absent: they
//! need the system-prompt subsystem. The step boundary where they hook is
//! already present.

#![forbid(unsafe_code)]

mod driver;

pub use driver::{
    AgentLoop, CompactionConfig, LoopConfig, LoopError, TokenMeter, TurnReport, DEFAULT_MAX_STEPS,
};
