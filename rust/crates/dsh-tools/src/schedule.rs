//! Schedules one assistant step's tool calls.
//!
//! Ported from `packages/core/agent-loop/tool-calls.ts`. Two invariants carry
//! over and are the whole point of this module:
//!
//! 1. **Results commit in model order.** Dispatch overlaps; policy, results,
//!    and result contexts never reorder. The log pairs each `tool/result` with
//!    the `tool/call` that requested it, so a replay reconstructs the step
//!    exactly.
//! 2. **Every model call settles.** A call the scheduler never starts still
//!    gets a durable `tool/call` + `tool/result` pair carrying a synthetic
//!    abort error, so `derive_messages()` never sees an unanswered call.
//!
//! ## Concurrency
//!
//! A parallel group runs up to `max_parallel_tool_calls` bodies at the same
//! time. Each body is dispatched onto the Tokio blocking pool, because a
//! [`ToolHandler`](crate::ToolHandler) is a synchronous function; the scheduler
//! refills the pool as bodies settle. Only *when* a body runs is concurrent:
//! calls are appended in model order and results are committed in model order,
//! so the durable log is the same one the sequential scheduler produced.
//! Exclusive calls remain barriers — they start alone, and the next call starts
//! only after the barrier's result committed.
//!
//! ## The pre-dispatch gate
//!
//! [`execute_tool_calls_gated`] routes every call through a [`PreToolGate`]
//! before its body runs: a denied or cancelled call never reaches the handler
//! and settles with the gate's synthesized result, still paired with its
//! `tool/call`. [`execute_tool_calls`] is the ungated entry point and delegates
//! to it with [`AllowAllGate`], which is why its behavior is unchanged. The
//! gate runs in model order on the scheduler's own task, so a policy observes
//! calls in the order the model asked for them.
//!
//! ## The post-execute stage
//!
//! Every result the scheduler settles passes a [`PostToolGate`] and its final
//! content transformation before it is committed: a dispatched body, an
//! unknown tool, and a gate denial or cancellation. A call that cancellation
//! never started is the exception: its canonical abort pair is recorded
//! directly, without post-execute, exactly as the TypeScript
//! `appendSkippedToolCall` does — no body ran, so no policy rewrites it. That
//! is the `tools/post-execute` waterfall plus `finalize` of the TypeScript
//! registry. The stage runs at commit time, in model order, so a policy sees
//! results in the same order they become durable and an `accept` rewrite or a
//! `block` cannot reorder the step. A rewrite changes the committed result
//! only: it cannot create or remove a call, and a call that was denied stays
//! denied even when its denial text is rewritten.

#![forbid(unsafe_code)]

use std::sync::Arc;

use dsh_llm::{ContentBlock, MessageId, ToolResultMessage, UserMessage};
use dsh_session::{Session, SessionError, SessionEvent, ToolErrorInfo};
use tokio::task::{Id, JoinSet};

use super::{
    cancel::CancelSignal,
    gate::{
        aborted_before_dispatch_result, apply_post_tool, evaluate_pre_dispatch, AllowAllGate,
        GateOutcome, PostToolGate, PreToolGate,
    },
    registry::{ToolHandler, ToolRegistry},
    types::{
        parse_arguments, ToolCallInput, ToolExecutionMode, ToolExecutionResult, TOOL_UNKNOWN_TOOL,
    },
};

/// Default maximum in-flight parallel-safe calls per agent step.
pub const DEFAULT_MAX_PARALLEL_TOOL_CALLS: usize = 10;

/// Scheduler bounds for one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolScheduleConfig {
    /// Maximum calls in one parallel group that may run a body at the same
    /// time. Values below one are treated as one.
    pub max_parallel_tool_calls: usize,
}

impl Default for ToolScheduleConfig {
    fn default() -> Self {
        Self {
            max_parallel_tool_calls: DEFAULT_MAX_PARALLEL_TOOL_CALLS,
        }
    }
}

/// What one step's scheduling produced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToolCallOutcome {
    /// Whether any committed result asked the turn to conclude.
    pub concluded: bool,
    /// Whether cancellation cut dispatch short.
    pub aborted: bool,
    /// How many model calls received a durable result.
    pub settled: usize,
}

/// One model call after argument parsing, ready to schedule.
struct PlannedCall {
    /// Raw JSON exactly as the model produced it, for the durable `tool/call`.
    arguments: String,
    /// Parsed input shared by the gate, the body, and post-execute.
    input: ToolCallInput,
}

/// Everything one step needs to schedule its tool calls.
///
/// Bundled because a step's coordinates, registry, bounds, and cancellation
/// always travel together; the log is the only remaining input.
#[derive(Clone, Copy, Debug)]
pub struct ToolSchedule<'a> {
    /// The turn the calls belong to.
    pub turn: u64,
    /// The step the calls belong to.
    pub step: u64,
    /// The live tool set.
    pub registry: &'a ToolRegistry,
    /// Scheduler bounds for this step.
    pub config: ToolScheduleConfig,
    /// The signal shared by the step's calls.
    pub cancel: &'a CancelSignal,
}

/// Schedule one assistant step's tool calls by their live concurrency mode.
///
/// `calls` are the assistant's tool-call blocks in model order; blocks that are
/// not tool calls are ignored. Exclusive calls form barriers: each runs alone.
/// Parallel calls form one bounded group whose bodies may overlap. Cancellation
/// stops new dispatches and records synthetic results for every remaining call,
/// in model order.
///
/// Deferred result contexts are handed to `accept_context` in call order for
/// the caller to stage in the next step's inbox.
///
/// This is the ungated entry point: every call is allowed, exactly as the
/// TypeScript waterfall's all-delegating default, and every settled result is
/// accepted by the default post-execute policy. Use
/// [`execute_tool_calls_gated`] to interpose a permission gate and
/// [`execute_tool_calls_with_post_execute`] to interpose a post-execute policy.
/// A call skipped by cancellation is recorded without post-execute, matching
/// the TypeScript `appendSkippedToolCall`.
///
/// Must be polled inside a Tokio runtime: parallel bodies run on its blocking
/// pool.
pub async fn execute_tool_calls(
    session: &mut Session,
    schedule: &ToolSchedule<'_>,
    calls: &[ContentBlock],
    accept_context: &mut (dyn FnMut(UserMessage) + Send),
) -> Result<ToolCallOutcome, SessionError> {
    execute_tool_calls_gated(session, schedule, calls, &AllowAllGate, accept_context).await
}

/// Schedule one assistant step's tool calls through a pre-dispatch gate.
///
/// Identical to [`execute_tool_calls`] except that every call passes `gate`
/// before its body runs. A gate denial or cancellation settles the call with a
/// synthesized failure — the reason-carrying denial, or the canonical
/// before-dispatch abort — and the handler is never invoked, but the call
/// still receives its paired `tool/call` + `tool/result`, so both §3.5
/// invariants hold: results commit in model order and every model call
/// settles. A gated failure is an ordinary failed result, not an abort: it
/// does not stop the step and later calls still run.
///
/// Settled results are accepted by the default post-execute policy; use
/// [`execute_tool_calls_with_post_execute`] to interpose one. A call skipped by
/// cancellation is recorded without post-execute.
pub async fn execute_tool_calls_gated(
    session: &mut Session,
    schedule: &ToolSchedule<'_>,
    calls: &[ContentBlock],
    gate: &dyn PreToolGate,
    accept_context: &mut (dyn FnMut(UserMessage) + Send),
) -> Result<ToolCallOutcome, SessionError> {
    execute_tool_calls_with_post_execute(
        session,
        schedule,
        calls,
        gate,
        &super::gate::AcceptAllPostTool,
        accept_context,
    )
    .await
}

/// Schedule one assistant step's tool calls through both policy seams.
///
/// [`execute_tool_calls_gated`] plus the post-execute stage: every result the
/// scheduler settles — dispatched, unknown-tool, gate-denied, or
/// gate-cancelled — is passed to `post` and its final content transformation
/// before it is committed, in model order. A call that cancellation never
/// started is recorded with the canonical abort without post-execute, matching
/// the TypeScript `appendSkippedToolCall`. A post-execute rewrite changes the
/// committed result, never the sequence of calls or the order in which results
/// land.
pub async fn execute_tool_calls_with_post_execute(
    session: &mut Session,
    schedule: &ToolSchedule<'_>,
    calls: &[ContentBlock],
    gate: &dyn PreToolGate,
    post: &dyn PostToolGate,
    accept_context: &mut (dyn FnMut(UserMessage) + Send),
) -> Result<ToolCallOutcome, SessionError> {
    let planned: Vec<PlannedCall> = calls
        .iter()
        .filter_map(ContentBlock::as_tool_call)
        .map(|call| PlannedCall {
            arguments: call.arguments.to_owned(),
            input: ToolCallInput {
                call_id: call.id.clone(),
                name: call.name.to_owned(),
                arguments: parse_arguments(call.arguments),
            },
        })
        .collect();

    let mut outcome = ToolCallOutcome::default();
    let mut next = 0;
    while next < planned.len() {
        if schedule.cancel.is_cancelled() {
            skip_remaining(session, schedule, &planned[next..])?;
            outcome.aborted = true;
            outcome.settled = planned.len();
            return Ok(outcome);
        }

        // Re-read the mode each iteration: a registry change between groups can
        // turn a later call into a barrier.
        let exclusive = schedule.registry.execution_mode(&planned[next].input.name)
            == Some(ToolExecutionMode::Exclusive);
        let end = if exclusive {
            next + 1
        } else {
            parallel_group_end(
                &planned,
                next,
                schedule.config.max_parallel_tool_calls.max(1),
                schedule.registry,
            )
        };

        let group_outcome = run_group(
            session,
            schedule,
            &planned[next..end],
            gate,
            post,
            accept_context,
        )
        .await?;
        outcome.concluded |= group_outcome.concluded;
        outcome.settled += group_outcome.settled;
        next = end;

        if group_outcome.aborted {
            skip_remaining(session, schedule, &planned[next..])?;
            outcome.aborted = true;
            outcome.settled = planned.len();
            return Ok(outcome);
        }
    }
    Ok(outcome)
}

/// The end index of one bounded run of parallel calls starting at `from`.
fn parallel_group_end(
    planned: &[PlannedCall],
    from: usize,
    max: usize,
    registry: &ToolRegistry,
) -> usize {
    let mut end = from;
    while end < planned.len()
        && end - from < max
        && registry.execution_mode(&planned[end].input.name) == Some(ToolExecutionMode::Parallel)
    {
        end += 1;
    }
    // A group is never empty: the leading call always runs.
    end.max(from + 1)
}

/// The slots of one group and how far the model-ordered prefix has committed.
struct CommitState {
    slots: Vec<Option<ToolExecutionResult>>,
    committed: usize,
    concluded: bool,
}

impl CommitState {
    fn new(len: usize) -> Self {
        Self {
            slots: (0..len).map(|_| None).collect(),
            committed: 0,
            concluded: false,
        }
    }

    /// Record one settled body's result, or drop it when the index is out of
    /// range (unreachable: every index comes from this group).
    fn settle(&mut self, index: usize, result: ToolExecutionResult) {
        if let Some(slot) = self.slots.get_mut(index) {
            *slot = Some(result);
        }
    }

    /// Commit every contiguous slot from the current frontier, running
    /// post-execute and finalize first. Stops at the first gap, which keeps the
    /// committed prefix in model order no matter which body finished first.
    async fn commit_ready(
        &mut self,
        session: &mut Session,
        schedule: &ToolSchedule<'_>,
        group: &[PlannedCall],
        post: &dyn PostToolGate,
        accept_context: &mut (dyn FnMut(UserMessage) + Send),
    ) -> Result<(), SessionError> {
        while let Some(call) = group.get(self.committed) {
            let Some(result) = self
                .slots
                .get_mut(self.committed)
                .and_then(|slot| slot.take())
            else {
                break;
            };
            let result = run_post_execute(post, &call.input, result).await;
            self.concluded |= result.concludes_turn();
            commit_result(session, schedule, call, &result)?;
            for context in result.additional_contexts() {
                accept_context(context.clone());
            }
            self.committed += 1;
        }
        Ok(())
    }
}

/// Run one exclusive barrier or parallel pool, committing results in model
/// order. Cancellation stops new starts, drains the bodies already running,
/// then records synthetic results for the rest of the group.
///
/// Bodies run on the blocking pool while the scheduler stays on its own task,
/// so the gate, the log, and post-execute are never touched concurrently.
fn group_body(
    handler: Arc<dyn ToolHandler>,
    input: ToolCallInput,
    cancel: CancelSignal,
    index: usize,
) -> (usize, ToolExecutionResult) {
    let result = handler.call(&input, &cancel);
    (index, result)
}

async fn run_group(
    session: &mut Session,
    schedule: &ToolSchedule<'_>,
    group: &[PlannedCall],
    gate: &dyn PreToolGate,
    post: &dyn PostToolGate,
    accept_context: &mut (dyn FnMut(UserMessage) + Send),
) -> Result<ToolCallOutcome, SessionError> {
    let max = schedule.config.max_parallel_tool_calls.max(1);
    let mut state = CommitState::new(group.len());
    let mut in_flight: JoinSet<(usize, ToolExecutionResult)> = JoinSet::new();
    // Task identity → model index, so a body that panicked still settles its
    // own slot instead of aborting the step.
    let mut task_indices: Vec<(Id, usize)> = Vec::new();
    let mut started = 0usize;
    let mut aborted = false;

    loop {
        while !aborted && started < group.len() && in_flight.len() < max {
            if schedule.cancel.is_cancelled() {
                aborted = true;
                break;
            }
            let Some(call) = group.get(started) else {
                break;
            };
            let index = started;
            session.append(SessionEvent::tool_call(
                schedule.turn,
                schedule.step,
                call.input.call_id.clone(),
                call.input.name.clone(),
                call.arguments.clone(),
            ))?;
            started += 1;

            // The gate sees the same parsed input the body would, including a
            // call naming an unregistered tool: the TypeScript pre-execute
            // waterfall observes every name that reaches the registry, and a
            // denial there short-circuits before the dispatch-stage
            // unknown-tool failure.
            match evaluate_pre_dispatch(gate, &call.input) {
                GateOutcome::Run => match schedule.registry.get(&call.input.name) {
                    Some(handler) => {
                        let handler = Arc::clone(handler);
                        let input = call.input.clone();
                        let cancel = schedule.cancel.clone();
                        let handle = in_flight
                            .spawn_blocking(move || group_body(handler, input, cancel, index));
                        task_indices.push((handle.id(), index));
                    }
                    None => state.settle(index, unknown_tool_result(&call.input.name)),
                },
                GateOutcome::Denied(result) | GateOutcome::Cancelled(result) => {
                    state.settle(index, result);
                }
            }

            state
                .commit_ready(session, schedule, group, post, accept_context)
                .await?;
            // Cancellation may have arrived while the gate or post-execute ran.
            if schedule.cancel.is_cancelled() {
                aborted = true;
            }
        }

        if in_flight.is_empty() {
            break;
        }

        match in_flight.join_next_with_id().await {
            Some(Ok((id, (index, result)))) => {
                task_indices.retain(|(task, _)| *task != id);
                state.settle(index, result);
            }
            Some(Err(error)) => {
                // A panicking handler is a failed call, not a failed step: the
                // call still settles with a result the model can see.
                let index = take_task_index(&mut task_indices, error.id());
                if let Some(index) = index {
                    let name = group
                        .get(index)
                        .map_or("unknown", |call| call.input.name.as_str());
                    state.settle(
                        index,
                        ToolExecutionResult::failure(format!("tool \"{name}\" failed: {error}")),
                    );
                }
            }
            None => break,
        }

        state
            .commit_ready(session, schedule, group, post, accept_context)
            .await?;
        if schedule.cancel.is_cancelled() {
            aborted = true;
        }
    }

    let mut outcome = ToolCallOutcome {
        concluded: state.concluded,
        aborted: false,
        settled: state.committed,
    };
    if aborted {
        // Every started call committed during the drain; the rest settle with
        // the canonical before-dispatch abort, in model order, without
        // post-execute because no body ran.
        skip_remaining(session, schedule, &group[started..])?;
        outcome.aborted = true;
        outcome.settled = group.len();
    }
    Ok(outcome)
}

/// Remove and return the model index registered for `id`.
fn take_task_index(task_indices: &mut Vec<(Id, usize)>, id: Id) -> Option<usize> {
    let position = task_indices.iter().position(|(task, _)| *task == id)?;
    Some(task_indices.remove(position).1)
}

/// Apply the post-execute seam to one settled result.
///
/// The decision is applied with [`apply_post_tool`], then the policy's final
/// content transformation runs. A decision that cannot be applied — replacing
/// the value of a failed result — fails the call the way the TypeScript
/// `TypeError` does: the error becomes the committed result, and finalization
/// still runs.
async fn run_post_execute(
    post: &dyn PostToolGate,
    input: &ToolCallInput,
    result: ToolExecutionResult,
) -> ToolExecutionResult {
    let decision = post.decide(input, &result).await;
    let shaped = match apply_post_tool(&decision, &result) {
        Ok(shaped) => shaped,
        Err(error) => ToolExecutionResult::failure(error.to_string()),
    };
    match post.finalize(input, &shaped) {
        Some(content) => shaped.with_content(content),
        None => shaped,
    }
}

/// The result for a call naming a tool the registry does not own.
///
/// The failure reproduces the TypeScript `ToolNotFoundError`
/// (`packages/core/tools/src/index.ts:511-526`): the message is
/// `unknown tool "<name>"` — plain double quotes, matching the TS template
/// literal — and `info.name` is `ToolNotFoundError`, the class name TS puts on
/// the routed error.
///
/// TS has a second form, `unknown tool "<name>": <route>`, for a name that IS
/// visible and only the presentation layer denies calling it directly (the
/// reserved PTC transport). The Rust registry has neither a scope layer nor
/// PTC, so only the base form is ported — same reasoning as
/// `validate_restriction`.
fn unknown_tool_result(name: &str) -> ToolExecutionResult {
    ToolExecutionResult::failure(format!("unknown tool \"{name}\"")).with_error_info(
        ToolErrorInfo {
            name: "ToolNotFoundError".to_string(),
            code: TOOL_UNKNOWN_TOOL.to_string(),
            reason: None,
        },
    )
}

/// Append the durable result of one started or failed call.
fn commit_result(
    session: &mut Session,
    schedule: &ToolSchedule<'_>,
    call: &PlannedCall,
    result: &ToolExecutionResult,
) -> Result<(), SessionError> {
    let message = ToolResultMessage::new(
        MessageId::random(),
        result.content().to_vec(),
        call.input.call_id.clone(),
        result.is_error(),
    );
    session.append(SessionEvent::tool_result_full(
        schedule.turn,
        schedule.step,
        message,
        result.error_info().cloned(),
        result.meta().cloned(),
    ))?;
    Ok(())
}

/// Record the durable pair for a model call that never started, settling it
/// with the canonical before-dispatch cancellation result.
///
/// No post-execute runs: the TypeScript `appendSkippedToolCall` writes this
/// pair directly, because no body ran and no policy may rewrite the abort.
fn skip_remaining(
    session: &mut Session,
    schedule: &ToolSchedule<'_>,
    calls: &[PlannedCall],
) -> Result<(), SessionError> {
    for call in calls {
        session.append(SessionEvent::tool_call(
            schedule.turn,
            schedule.step,
            call.input.call_id.clone(),
            call.input.name.clone(),
            call.arguments.clone(),
        ))?;
        commit_result(session, schedule, call, &aborted_before_dispatch_result())?;
    }
    Ok(())
}
