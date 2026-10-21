//! The turn/step driver.

#![forbid(unsafe_code)]

use std::collections::VecDeque;

use dsh_llm::{FinishReason, LlmCallConfig, Message, UserMessage};
use dsh_provider::{LlmProvider, LlmRequest};
use dsh_retry::{
    encode_retry, encode_retry_started, fold_retry_state, plan_retry, RetryConfig, RetryStateError,
};
use dsh_session::{
    derive_messages, fold_request_header, EpochHeader, RequestContext, RequestHeaderReason,
    Session, SessionEvent, SystemPromptUpdate, TurnEndCancelCause, TurnEndReason,
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
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            tool_schedule: ToolScheduleConfig::default(),
            retry: None,
        }
    }
}

impl LoopConfig {
    /// Install retry recovery for failed model requests.
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = Some(retry);
        self
    }
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
        let messages: Vec<Message> = derive_messages(self.session.events());
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
