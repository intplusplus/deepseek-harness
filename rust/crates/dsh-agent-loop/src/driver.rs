//! The turn/step driver.

#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::sync::Arc;

use dsh_compaction::{
    acquire_lock, compaction_attempt_limit, fold_surface, overflow_compaction_allowed,
    plan_overflow_recovery, release_lock, routed_target, should_compact, should_recover_overflow,
    still_above_threshold, CompactPlan, CompactionError, CompactionEvent, CompactionService,
    OverflowRecovery, ResolvedCompactSpec, ResolvedPruneConfig, DEFAULT_LEASE_MS,
};
use dsh_llm::{FinishReason, LlmCallConfig, Message, UserMessage};
use dsh_provider::{LlmProvider, LlmRequest};
use dsh_retry::{
    encode_retry, encode_retry_started, fold_retry_state, plan_retry, RetryConfig, RetryStateError,
};
use dsh_session::{
    derive_messages, fold_request_header, CoreEvent, EpochHeader, RequestContext,
    RequestHeaderReason, Session, SessionEvent, SystemPromptUpdate, TurnEndCancelCause,
    TurnEndReason,
};
use dsh_tools::{execute_tool_calls, CancelSignal, ToolRegistry, ToolSchedule, ToolScheduleConfig};

/// Guard against a model that never stops asking for tools.
///
/// This has no TypeScript counterpart — the real loop relies on the model
/// terminating — but an offline driver needs a bound. Exceeding it ends the
/// turn with an `error` reason rather than hanging.
pub const DEFAULT_MAX_STEPS: usize = 32;

/// Bounds the driver applies to one turn.
#[derive(Clone, Debug)]
pub struct LoopConfig {
    /// Maximum steps in one turn.
    pub max_steps: usize,
    /// Scheduler bounds for each step's tool calls.
    pub tool_schedule: ToolScheduleConfig,
    /// How a failed model request may be retried, when configured at all.
    ///
    /// `None` — the default — keeps the pre-retry behaviour: a provider
    /// failure ends the turn immediately.
    pub retry: Option<RetryConfig>,
    /// Automatic compaction at step boundaries and on context overflow.
    ///
    /// `None` — the default — keeps the pre-compaction behaviour: the log
    /// carries no `compaction/*` events and every request is the raw history.
    pub compaction: Option<CompactionConfig>,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            tool_schedule: ToolScheduleConfig::default(),
            retry: None,
            compaction: None,
        }
    }
}

impl LoopConfig {
    /// Install retry recovery for failed model requests.
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Install automatic compaction at step boundaries and on context overflow.
    pub fn with_compaction(mut self, compaction: CompactionConfig) -> Self {
        self.compaction = Some(compaction);
        self
    }
}

/// The measured size of the request the current history would produce.
///
/// The host owns the token meter: `dsh-session-stats` carries part of the
/// input, but the loop takes the size as a caller-supplied capability so it
/// stays independent of any one meter. The event slice is the log the loop is
/// about to derive its request from.
pub trait TokenMeter: Send + Sync {
    /// The estimated request size for this history, in tokens.
    fn total_tokens(&self, events: &[SessionEvent]) -> u64;
}

impl<F> TokenMeter for F
where
    F: Fn(&[SessionEvent]) -> u64 + Send + Sync,
{
    fn total_tokens(&self, events: &[SessionEvent]) -> u64 {
        self(events)
    }
}

/// The automatic compaction capability the loop may exercise.
///
/// It pairs a runnable [`CompactionService`] with the routed pressure budget
/// the loop measures against. Everything else — the transaction, the lock, the
/// summary call, the event order — stays inside `dsh-compaction`; the loop only
/// decides *when* to call it and appends what it returns.
#[derive(Clone)]
pub struct CompactionConfig {
    service: Arc<CompactionService>,
    spec: ResolvedCompactSpec,
    meter: Arc<dyn TokenMeter>,
    prune: Option<ResolvedPruneConfig>,
    lease_ms: i64,
    clock: Arc<dyn Fn() -> i64 + Send + Sync>,
}

impl CompactionConfig {
    /// Wire automatic compaction to a runnable service and its routed budget.
    pub fn new(
        service: Arc<CompactionService>,
        spec: ResolvedCompactSpec,
        meter: impl TokenMeter + 'static,
    ) -> Self {
        Self {
            service,
            spec,
            meter: Arc::new(meter),
            prune: None,
            lease_ms: DEFAULT_LEASE_MS,
            clock: Arc::new(system_now_ms),
        }
    }

    /// Install the optional model-free tool-result prune pass, which runs
    /// inside the transaction before the summary range is selected.
    pub fn with_prune(mut self, prune: ResolvedPruneConfig) -> Self {
        self.prune = Some(prune);
        self
    }

    /// Use a caller-supplied clock, in epoch milliseconds, for the durable lock.
    pub fn with_clock(mut self, clock: Arc<dyn Fn() -> i64 + Send + Sync>) -> Self {
        self.clock = clock;
        self
    }

    /// Override the durable lock lease.
    pub fn with_lease_ms(mut self, lease_ms: i64) -> Self {
        self.lease_ms = lease_ms;
        self
    }
}

impl std::fmt::Debug for CompactionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The service, meter, and clock are capabilities without a printable
        // identity; the budget is what a debug read wants anyway.
        f.debug_struct("CompactionConfig")
            .field("spec", &self.spec)
            .field("prune", &self.prune)
            .field("lease_ms", &self.lease_ms)
            .finish_non_exhaustive()
    }
}

/// Wall clock in epoch milliseconds, the unit the compaction lease speaks.
fn system_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or_default()
}

/// Why a turn could not be driven to a normal end.
#[derive(Debug, thiserror::Error)]
pub enum LoopError {
    /// The log rejected an event.
    #[error("session append failed: {0}")]
    Session(#[from] dsh_session::SessionError),
    /// The provider failed.
    #[error("model call failed: {0}")]
    Llm(#[from] dsh_llm::LlmFailure),
    /// A retry event could not be written to, or read back from, the log.
    #[error("retry log round-trip failed: {0}")]
    Retry(#[from] RetryStateError),
    /// A compaction event could not be encoded for, or admitted by, the log.
    ///
    /// A failed *transaction* never reaches this: compaction failures are
    /// recorded and the turn continues. This is the log rejecting the events
    /// the loop tried to append, which is corruption, not policy.
    #[error("compaction log write failed: {0}")]
    Compaction(#[from] CompactionError),
    /// The step guard tripped.
    #[error("turn exceeded its {limit}-step limit")]
    StepLimit {
        /// The limit that tripped.
        limit: usize,
    },
    /// The per-step attempt guard tripped.
    #[error("step exhausted its {limit}-attempt limit")]
    AttemptLimit {
        /// The limit that tripped.
        limit: usize,
    },
}

/// What one turn produced.
#[derive(Clone, Debug, PartialEq)]
pub struct TurnReport {
    /// The turn number that ran.
    pub turn: u64,
    /// How many steps it spent.
    pub steps: u64,
    /// Why it ended.
    pub reason: TurnEndReason,
}

/// What one model-request attempt decided about its step.
///
/// A step is not just "one request": it may spend several attempts recovering
/// from failures, and it may end with tool calls that open the next step. The
/// two "not done yet" outcomes are deliberately distinct — collapsing them is
/// how a retry loop silently swallows the step boundary.
#[derive(Debug)]
enum StepAttempt {
    /// The same step must try the request again.
    Retry,
    /// The turn must open another step.
    Continue,
    /// The turn ends here.
    Ended(TurnEndReason),
}

/// What one automatic compaction call landed.
///
/// The three outcomes are distinct on purpose: a landed plan grows the log, no
/// range keeps it untouched, and a failure must be recorded without ending the
/// turn.
#[derive(Debug)]
enum Landed {
    /// A full transaction was appended.
    Plan,
    /// No compactable range exists, so nothing was written.
    Nothing,
    /// The transaction failed; the caller decides how to record it.
    Failed(CompactionError),
}

/// Drives one session through turn boundaries.
///
/// The driver owns the turn and step counters and nothing else: history lives
/// in the log, tools live in the registry, and the model lives behind the
/// provider.
pub struct AgentLoop<'a> {
    session: &'a mut Session,
    provider: &'a dyn LlmProvider,
    registry: &'a ToolRegistry,
    config: LoopConfig,
    route: RequestContext,
    turn: u64,
    pending: VecDeque<UserMessage>,
    /// Overflow recoveries spent since the last successful response.
    ///
    /// Scoped like the TypeScript per-agent counter: a successful
    /// `assistant/message` (or a new turn) starts a fresh sequence.
    overflow_retries: u64,
}

impl<'a> AgentLoop<'a> {
    /// A driver over one session.
    pub fn new(
        session: &'a mut Session,
        provider: &'a dyn LlmProvider,
        registry: &'a ToolRegistry,
        route: RequestContext,
    ) -> Self {
        Self {
            session,
            provider,
            registry,
            config: LoopConfig::default(),
            route,
            turn: 0,
            pending: VecDeque::new(),
            overflow_retries: 0,
        }
    }

    /// Override the driver bounds.
    pub fn with_config(mut self, config: LoopConfig) -> Self {
        self.config = config;
        self
    }

    /// Queue input for the next turn.
    pub fn send(&mut self, message: UserMessage) {
        self.pending.push_back(message);
    }

    /// The turn number the next turn will take.
    pub fn next_turn(&self) -> u64 {
        self.turn + 1
    }

    /// Whether input is queued.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Run one turn to its end.
    ///
    /// Queued input is claimed as `user/message` events first, so it is part of
    /// the derived history before the first request is built. The returned
    /// future must be polled inside a Tokio runtime: the model call is async
    /// and tool bodies run on the blocking pool.
    pub async fn run_turn(&mut self) -> Result<TurnReport, LoopError> {
        let turn = self.turn + 1;
        self.append(SessionEvent::turn_start(turn))?;
        self.turn = turn;
        // A new turn starts a fresh overflow-recovery sequence, matching the
        // TypeScript counter reset on the agent's idle transition.
        self.overflow_retries = 0;

        let claimed: Vec<UserMessage> = self.pending.drain(..).collect();
        for message in claimed {
            self.append(SessionEvent::user_message(message))?;
        }

        let reason = self.run_steps(turn).await?;
        self.append(SessionEvent::turn_end(turn, reason.clone()))?;
        Ok(TurnReport {
            turn,
            steps: self.steps_for(turn),
            reason,
        })
    }

    /// Run the step loop, returning why the turn ended.
    async fn run_steps(&mut self, turn: u64) -> Result<TurnEndReason, LoopError> {
        let mut step = 0u64;
        loop {
            if step as usize >= self.config.max_steps {
                return Ok(step_limit_reason(self.config.max_steps));
            }
            step += 1;

            self.append(SessionEvent::step_start(turn, step))?;
            let ended = self.run_step(turn, step).await?;
            self.append(SessionEvent::step_end(turn, step))?;

            if let Some(reason) = ended {
                return Ok(reason);
            }
        }
    }

    /// Run one step: build the request, call the model, schedule its tool calls.
    ///
    /// A provider failure is offered to retry recovery before the turn is
    /// ended, so one step may spend several attempts on the model request.
    /// Returns `None` when the turn must open another step.
    async fn run_step(&mut self, turn: u64, step: u64) -> Result<Option<TurnEndReason>, LoopError> {
        self.log_route()?;
        // The TypeScript `agent/pre-step` hook runs compaction before the
        // request is derived; here the equivalent point is the top of the step,
        // after the request header exists (compaction needs the routed target)
        // and before `derive_messages`. The bracket this writes stays inside
        // the step, so no step event can land between `compaction/start` and
        // its `compaction/end`.
        self.compact_pressure().await?;
        let mut attempts = 0usize;
        loop {
            attempts += 1;
            match self.run_attempt(turn, step, attempts).await? {
                StepAttempt::Retry => continue,
                StepAttempt::Continue => return Ok(None),
                StepAttempt::Ended(reason) => return Ok(Some(reason)),
            }
        }
    }

    /// One model-request attempt plus its tool calls.
    async fn run_attempt(
        &mut self,
        turn: u64,
        step: u64,
        attempts: usize,
    ) -> Result<StepAttempt, LoopError> {
        let messages: Vec<Message> = derive_request_messages(self.session.events());
        let request = LlmRequest {
            config: self.call_config(),
            messages,
            tools: self.registry.schemas(),
        };

        let response = match self.provider.generate(&request).await {
            Ok(response) => response,
            Err(failure) => return self.recover_request(turn, step, attempts, failure).await,
        };

        self.append(SessionEvent::assistant_message(
            turn,
            step,
            response.message.clone(),
            response.stream.clone(),
            response.usage.clone(),
            false,
        ))?;
        // A delivered response proves the current surface fits, so the next
        // overflow starts a fresh recovery sequence.
        self.overflow_retries = 0;

        let calls: usize = response.tool_calls().len();
        if calls == 0 {
            return Ok(StepAttempt::Ended(match response.finish_reason {
                FinishReason::MaxTokens => TurnEndReason::MaxTokens,
                FinishReason::Aborted { failure } | FinishReason::Error { failure } => {
                    TurnEndReason::error((*failure).clone())
                }
                FinishReason::Stop | FinishReason::ToolCalls => TurnEndReason::Completed,
            }));
        }

        let cancel = CancelSignal::new();
        let schedule = ToolSchedule {
            turn,
            step,
            registry: self.registry,
            config: self.config.tool_schedule,
            cancel: &cancel,
        };
        // Deferred tool contexts become the next step's input, exactly as the
        // TS loop stages them in the step-boundary inbox.
        let mut deferred: Vec<UserMessage> = Vec::new();
        let outcome = execute_tool_calls(
            self.session,
            &schedule,
            &response.message.content,
            &mut |context| deferred.push(context),
        )
        .await?;
        for context in deferred {
            self.append(SessionEvent::user_message(context))?;
        }

        if outcome.aborted {
            return Ok(StepAttempt::Ended(TurnEndReason::Aborted {
                reason: TurnEndCancelCause::User,
            }));
        }
        if outcome.concluded {
            return Ok(StepAttempt::Ended(TurnEndReason::Completed));
        }
        Ok(StepAttempt::Continue)
    }

    /// Consider automatic pressure compaction before this step's request.
    ///
    /// This is the `agent/pre-step` trigger of the TypeScript loop: measure the
    /// request the current surface would produce, and compact when it has
    /// crossed the routed threshold. The TypeScript loop retries the shrink up
    /// to `compactionRetries + 1` times and remeasures between attempts; when
    /// the budget is spent it records the failure and lets the turn continue on
    /// whatever surface it has.
    ///
    /// A failed compaction never ends the turn: the request then simply carries
    /// the full history, which is what the TypeScript listener's "warn and
    /// continue the turn" amounts to.
    async fn compact_pressure(&mut self) -> Result<(), LoopError> {
        let Some(compaction) = self.config.compaction.clone() else {
            return Ok(());
        };
        let total_tokens = compaction.meter.total_tokens(self.session.events());
        if !should_compact(total_tokens, &compaction.spec) {
            return Ok(());
        }

        let mut landed = false;
        for _ in 0..compaction_attempt_limit(&compaction.spec) {
            match self
                .land_compaction(&compaction, compaction.spec.retain_tokens)
                .await?
            {
                // No range: nothing to shrink, and no event may be written.
                Landed::Nothing => return Ok(()),
                Landed::Plan => landed = true,
                Landed::Failed(error) => {
                    self.record_compaction_failure(&compaction, &error)?;
                    return Ok(());
                }
            }
            let total_tokens = compaction.meter.total_tokens(self.session.events());
            if !should_compact(total_tokens, &compaction.spec) {
                return Ok(());
            }
        }

        if landed {
            let total_tokens = compaction.meter.total_tokens(self.session.events());
            let error = still_above_threshold(total_tokens, &compaction.spec);
            self.record_compaction_failure(&compaction, &error)?;
        }
        Ok(())
    }

    /// Run one compaction transaction over the current log and land its events.
    async fn land_compaction(
        &mut self,
        compaction: &CompactionConfig,
        retain_tokens: u64,
    ) -> Result<Landed, LoopError> {
        let now = (compaction.clock)();
        let outcome = compaction
            .service
            .compact_if_needed(
                self.session.events(),
                retain_tokens,
                now,
                compaction.lease_ms,
                compaction.prune,
            )
            .await;
        match outcome {
            Ok(Some(plan)) => {
                self.append_plan(plan)?;
                Ok(Landed::Plan)
            }
            Ok(None) => Ok(Landed::Nothing),
            Err(error) => Ok(Landed::Failed(error)),
        }
    }

    /// Append one transaction in its contractual order.
    ///
    /// The checkpoint `user/message` must stay adjacent to `compaction/summary`
    /// (that adjacency is what makes the replacement effective on replay), and
    /// each prune replacement must stay adjacent to its `compaction/prune`
    /// shadow price.
    fn append_plan(&mut self, plan: CompactPlan) -> Result<(), LoopError> {
        self.append(plan.start)?;
        for event in plan.prune_events {
            self.append(event)?;
        }
        self.append(plan.summary)?;
        self.append(plan.checkpoint)?;
        self.append(plan.end)?;
        Ok(())
    }

    /// Record a failed compaction attempt as an empty bracket.
    ///
    /// `compact` is a transaction that returns its events instead of writing
    /// them, so a failed attempt leaves no `compaction/start` behind. The
    /// TypeScript log still shows the attempt, and so does this: the host opens
    /// and closes a bracket of its own, carrying the diagnostic in
    /// `compaction/end`. A genuinely held lock leaves nothing to record.
    fn record_compaction_failure(
        &mut self,
        compaction: &CompactionConfig,
        error: &CompactionError,
    ) -> Result<(), LoopError> {
        let now = (compaction.clock)();
        let Ok(start) = acquire_lock(
            self.session.events(),
            false,
            compaction.lease_ms,
            now,
            "automatic compaction",
        ) else {
            return Ok(());
        };
        let end = release_lock(&start, Some(error.to_string()));
        self.append(encode_compaction(CompactionEvent::Start(start))?)?;
        self.append(encode_compaction(CompactionEvent::End(end))?)?;
        Ok(())
    }

    /// Offer a canonical context-window overflow to forced compaction.
    ///
    /// Returns `Some(Retry)` when a durable surface replacement lets this same
    /// step try the request again. `None` means the caller should fall through
    /// to the ordinary retry policy, which preserves the original overflow
    /// failure when the budget is spent or no progress was made.
    ///
    /// Overflow deliberately bypasses both the pressure threshold and the
    /// retained tail: the provider has already rejected the request, so one
    /// forced balanced reduction is the whole recovery.
    async fn recover_overflow(
        &mut self,
        failure: &dsh_llm::LlmFailure,
    ) -> Result<Option<StepAttempt>, LoopError> {
        let Some(compaction) = self.config.compaction.clone() else {
            return Ok(None);
        };
        // An abort is not a provider failure here: the driver ends an aborted
        // step through its own paths, so recovery only ever sees `failure`.
        if !should_recover_overflow(&failure.code, false) {
            return Ok(None);
        }
        let routed = routed_target(fold_request_header(self.session.events()).as_ref()).is_some();
        if !overflow_compaction_allowed(
            self.overflow_retries,
            compaction.spec.max_overflow_retries,
            routed,
        ) {
            return Ok(None);
        }

        let before = replace_generation(self.session.events());
        if let Landed::Failed(error) = self.land_compaction(&compaction, 0).await? {
            self.record_compaction_failure(&compaction, &error)?;
        }
        let after = replace_generation(self.session.events());

        match plan_overflow_recovery(self.overflow_retries, false, before, after) {
            OverflowRecovery::Retry { retries } => {
                self.overflow_retries = retries;
                Ok(Some(StepAttempt::Retry))
            }
            OverflowRecovery::Delegate => Ok(None),
        }
    }

    /// Offer a failed model request to retry recovery.
    ///
    /// This is the driver's `agent/request-error` recovery point. Without a
    /// configured policy for the request's provider — or with no retry
    /// configuration at all — the failure ends the turn, which is exactly what
    /// the driver did before recovery existed.
    ///
    /// The durable `llm/retry` event is written *before* the wait and
    /// `llm/retry-started` after it, so a log always shows that a retry was
    /// scheduled even if the process never got to spend the delay.
    ///
    /// The wait is raced against [`RetryConfig::cancel`]: an external
    /// cancellation ends it without writing `llm/retry-started`, and the step
    /// ends as a user abort, exactly as a cancelled delay does.
    async fn recover_request(
        &mut self,
        turn: u64,
        step: u64,
        attempts: usize,
        failure: dsh_llm::LlmFailure,
    ) -> Result<StepAttempt, LoopError> {
        // Overflow recovery is deterministic and takes precedence over a
        // provider retry policy: retrying an over-window request unchanged can
        // only fail again.
        if let Some(attempt) = self.recover_overflow(&failure).await? {
            return Ok(attempt);
        }
        let Some(retry) = self.config.retry.clone() else {
            return Ok(StepAttempt::Ended(TurnEndReason::error(failure)));
        };
        let provider = self.call_config().provider;
        let Some(policy) = retry.policies.get(&provider) else {
            return Ok(StepAttempt::Ended(TurnEndReason::error(failure)));
        };
        if attempts >= retry.max_attempts {
            return Ok(StepAttempt::Ended(attempt_limit_reason(retry.max_attempts)));
        }

        // Retry state is derived, not stored: the fold resets at `step/start`,
        // so replaying the whole log yields exactly this step's state.
        let state = fold_retry_state(self.session.events())?;
        let random = retry.random.clone();
        let Some(plan) = plan_retry(&provider, &failure, policy, &state, move || random()) else {
            return Ok(StepAttempt::Ended(TurnEndReason::error(failure)));
        };

        self.append(
            encode_retry(&plan.retry_event(turn, step, failure)).map_err(RetryStateError::from)?,
        )?;
        let waited = {
            let delay = retry.delay.wait(plan.delay_ms);
            let cancelled = retry.cancel.cancelled();
            tokio::pin!(delay);
            tokio::pin!(cancelled);
            tokio::select! {
                waited = &mut delay => waited,
                () = &mut cancelled => false,
            }
        };
        if !waited {
            return Ok(StepAttempt::Ended(TurnEndReason::Aborted {
                reason: TurnEndCancelCause::User,
            }));
        }
        self.append(
            encode_retry_started(&plan.started_event(turn, step)).map_err(RetryStateError::from)?,
        )?;
        Ok(StepAttempt::Retry)
    }

    /// Log the request header and route context the first time they are needed.
    fn log_route(&mut self) -> Result<(), LoopError> {
        if fold_request_header(self.session.events()).is_some() {
            return Ok(());
        }
        // An empty tool list is non-canonical: the log rejects `tools: []`
        // rather than normalizing it, so a tool-less route omits the field.
        let tools = self.registry.schemas();
        self.append(SessionEvent::request_header_series(
            EpochHeader {
                config: self.call_config(),
                adapter_defaults: None,
                tools: (!tools.is_empty()).then_some(tools),
            },
            RequestHeaderReason::Initial,
        ))?;
        self.append(SessionEvent::request_context(RequestContext {
            provider: self.route.provider.clone(),
            model: self.route.model.clone(),
            context_window: self.route.context_window,
            system_prompt_update: Some(SystemPromptUpdate::InHistory),
        }))?;
        Ok(())
    }

    /// The call configuration for the next request.
    fn call_config(&self) -> LlmCallConfig {
        fold_request_header(self.session.events())
            .map(|header| header.config)
            .unwrap_or_else(|| LlmCallConfig {
                provider: self.route.provider.clone(),
                model: self.route.model.clone(),
                reasoning_effort: None,
                temperature: None,
                max_tokens: None,
                stop: None,
            })
    }

    fn append(&mut self, event: SessionEvent) -> Result<(), LoopError> {
        self.session.append(event)?;
        Ok(())
    }

    /// How many `step/end` events the turn logged.
    fn steps_for(&self, turn: u64) -> u64 {
        self.session
            .events()
            .iter()
            .filter(|event| matches!(event.core(), Some(dsh_session::CoreEvent::StepEnd { turn: t, .. }) if *t == turn))
            .count() as u64
    }
}

/// The reason a tripped step guard produces, carrying its limit in the text.
fn step_limit_reason(limit: usize) -> TurnEndReason {
    TurnEndReason::error(dsh_llm::LlmFailure {
        message: format!("turn exceeded its {limit}-step limit"),
        code: "STEP_LIMIT".to_string(),
        status: None,
        provider_retry_after_ms: None,
        request_id: None,
        offload_images: None,
    })
}

/// The reason a tripped attempt guard produces.
///
/// A `normal` policy bounds itself by `maxRetries`, but an `always` policy is
/// unbounded by design, so the driver still needs a ceiling of its own.
fn attempt_limit_reason(limit: usize) -> TurnEndReason {
    TurnEndReason::error(dsh_llm::LlmFailure {
        message: format!("step exhausted its {limit}-attempt limit"),
        code: "ATTEMPT_LIMIT".to_string(),
        status: None,
        provider_retry_after_ms: None,
        request_id: None,
        offload_images: None,
    })
}

/// Derive the model history, honouring any landed compaction replacements.
///
/// [`derive_messages`] folds the raw log and cannot see a replacement: the
/// shadowing record and the replacement message live outside the events they
/// replace. The surface fold is what removes shadowed nodes, so it owns the
/// history whenever a replacement has landed. Without one the two folds are
/// identical, and the log order is preserved either way.
fn derive_request_messages(events: &[SessionEvent]) -> Vec<Message> {
    let Ok(surface) = fold_surface(events) else {
        // A malformed `compaction/*` payload makes the surface unusable; the
        // raw history is the only reading left, and it is what a turn without
        // compaction would have sent.
        return derive_messages(events);
    };
    if surface.replacements().is_empty() {
        return derive_messages(events);
    }
    surface
        .nodes()
        .iter()
        .filter_map(|node| events.get(node.seq().get() as usize))
        .filter_map(message_from_event)
        .collect()
}

/// Reconstruct one model message from a message-bearing core event.
fn message_from_event(event: &SessionEvent) -> Option<Message> {
    match event.core()? {
        CoreEvent::SystemMessage { message, .. } => Some(Message::System((**message).clone())),
        CoreEvent::DeveloperMessage { message, .. } => {
            Some(Message::Developer((**message).clone()))
        }
        CoreEvent::UserMessage { message } => Some(Message::User((**message).clone())),
        CoreEvent::AssistantMessage { message, .. } => {
            Some(Message::Assistant((**message).clone()))
        }
        CoreEvent::ToolResult { message, .. } => Some(Message::Tool((**message).clone())),
        _ => None,
    }
}

/// How many positional replacements the current surface carries.
///
/// This is the durable proof an overflow recovery actually shrank the request:
/// a summary or prune pair that landed advanced it, a failed transaction left
/// it unchanged.
fn replace_generation(events: &[SessionEvent]) -> u64 {
    fold_surface(events)
        .map(|surface| surface.replace_generation() as u64)
        .unwrap_or(0)
}

/// Encode one compaction event for the log.
fn encode_compaction(event: CompactionEvent) -> Result<SessionEvent, LoopError> {
    event
        .encode()
        .map_err(|error| LoopError::Compaction(CompactionError::Encode(error)))
}
