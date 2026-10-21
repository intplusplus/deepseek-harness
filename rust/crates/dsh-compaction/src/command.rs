//! Manual compaction command entry and the [`CompactionService`] the agent
//! loop may call at a step boundary.
//!
//! [`compact`] is pure and testable: it takes the current durable log (a
//! `&[SessionEvent]`), runs the full lifecycle — acquire the durable lock,
//! optionally prune over-budget tool results, summarize the selected region
//! through a [`SummarizationBackend`], and emit the `compaction/start`,
//! `compaction/prune`/`tool/result`, `compaction/summary`, checkpoint
//! `user/message`, and `compaction/end` events — and returns the new events for
//! the host to append. No log is mutated here.
//!
//! The agent-loop integration is a pure policy decision, [`should_compact`]: a
//! caller resolves the routed pressure spec and asks whether the measured
//! request has crossed the threshold. The loop drives the actual
//! `compact_now`/`compact_if_needed` call; this crate only exposes the decision
//! and the [`CompactionService`] that can perform it.

#![forbid(unsafe_code)]

use std::sync::Arc;

use dsh_llm::{ContentBlock, Message, MessageId, UserMessage};
use dsh_session::{CoreEvent, SessionEvent, SessionSeq};

use super::backend::{frame_summary, SummarizationBackend, SummarizationInput, SummaryResult};
use super::brand::CommandId;
use super::checkpoint::compact_checkpoint_source;
use super::error::{CompactionError, SurfaceError};
use super::events::{CompactionEvent, CompactionResult, CompactionSummary, ShadowedRange};
use super::policy::{ResolvedConfig, ResolvedCompactSpec, ResolvedRetention};
use super::prune::{measure_content, plan_prune_session, PrunePlan, ResolvedPruneConfig};
use super::state::{select_compactable_range, validate_surface_region, NodePrice};
use super::surface::{fold_surface, SurfaceNodeKind, SurfaceProjection};

/// Options for one manual or automatic compaction transaction.
#[derive(Clone, Debug)]
pub struct CompactOptions {
    /// Whether this is a manual (between-turns) transaction.
    pub manual: bool,
    /// Initiating manual command, when present.
    pub source_command_id: Option<CommandId>,
    /// Epoch milliseconds at acquisition, for lease-expiry.
    pub now: i64,
    /// Lease duration in milliseconds.
    pub lease_ms: i64,
    /// Recent-tail retention budget; `0` compacts the whole eligible range.
    pub retain_tokens: u64,
    /// Summarization provider route.
    pub provider: String,
    /// Summarization model.
    pub model: String,
    /// Summarization generation cap.
    pub max_tokens: Option<u64>,
    /// Optional model-free tool-result prune configuration.
    pub prune_config: Option<ResolvedPruneConfig>,
}

/// The durable events a compaction transaction produces, in append order, plus
/// the decoded result a caller can surface.
#[derive(Clone, Debug)]
pub struct CompactPlan {
    /// The lock-opening `compaction/start` event.
    pub start: SessionEvent,
    /// Model-free prune metering events and their replacement `tool/result`s.
    pub prune_events: Vec<SessionEvent>,
    /// The completed-summary `compaction/summary` event.
    pub summary: SessionEvent,
    /// The replacement checkpoint `user/message` immediately after the summary.
    pub checkpoint: SessionEvent,
    /// The lock-closing `compaction/end` event.
    pub end: SessionEvent,
    /// Decoded result, with sequence positions relative to the pre-compaction log.
    pub result: CompactionResult,
}

/// Run one full compaction transaction over the current log.
///
/// Returns `Ok(None)` when no compactable range exists yet (the caller should
/// report "no compactable history"). On any failure the caller is expected to
/// still append `compaction/end` with the error; this pure entry returns the
/// error so the host decides how to record it.
pub async fn compact(
    events: &[SessionEvent],
    backend: &dyn SummarizationBackend,
    config: &ResolvedConfig,
    options: &CompactOptions,
) -> Result<Option<CompactPlan>, CompactionError> {
    let stage = if options.manual {
        "manual compaction"
    } else {
        "automatic compaction"
    };

    // Acquire the durable lock before any asynchronous work (matches TS order).
    let start = super::lock::acquire_lock(events, options.manual, options.lease_ms, options.now, stage)?;
    let compaction_id = start.compaction_id.clone();
    let turn = start.turn;
    let source_command_id = start.source_command_id.clone();

    // Work on a mutable copy so the model-free prune pass lands before selection.
    let mut working = events.to_vec();
    let mut prune_events = Vec::new();

    if let Some(prune_config) = &options.prune_config {
        let surface = fold_surface(&working)?;
        let plan: PrunePlan = plan_prune_session(&surface, &working, prune_config)?;
        for decision in plan.decisions {
            let original = working
                .get(decision.original_seq.get() as usize)
                .ok_or_else(|| CompactionError::RangeStartNotFound {
                    start: decision.original_seq,
                })?;
            let (orig_turn, orig_step) = match original {
                SessionEvent::Core(CoreEvent::ToolResult { turn, step, .. }) => (*turn, *step),
                _ => {
                    return Err(CompactionError::Surface(SurfaceError::MissingToolResult {
                        seq: decision.original_seq,
                    }));
                }
            };
            let price = measure_content(&decision.replacement.content);
            let prune_event = CompactionEvent::Prune(decision.prune_event(price)).encode()?;
            let rewritten = SessionEvent::tool_result(orig_turn, orig_step, decision.replacement);
            prune_events.push(prune_event.clone());
            prune_events.push(rewritten.clone());
            working.push(prune_event);
            working.push(rewritten);
        }
    }

    let surface = fold_surface(&working)?;
    let prices = estimate_surface_prices(&working, &surface);
    let range = select_compactable_range(&surface.nodes(), &prices, options.retain_tokens)?;
    let Some(range) = range else {
        return Ok(None);
    };
    let region = validate_surface_region(&surface.nodes(), range.start, range.end)?;

    let input = build_summarization_input(&working, &surface, &region.shadowed_seqs)?;
    let result: SummaryResult = backend
        .summarize(
            &input,
            &options.provider,
            &options.model,
            options.max_tokens,
        )
        .await?;

    let shadowed_token_count = region
        .shadowed_seqs
        .iter()
        .filter_map(|seq| prices.iter().find(|price| price.seq == *seq))
        .fold(0_u64, |total, price| total.saturating_add(price.tokens));

    let summary_event = CompactionSummary {
        compaction_id: compaction_id.clone(),
        source_command_id: source_command_id.clone(),
        summary: result.summary.clone(),
        raw_output: result.raw_output.clone(),
        llm_stream_call: result.llm_stream_call,
        shadowed_range: ShadowedRange::new(range.start, range.end),
        shadowed_seqs: region.shadowed_seqs.clone(),
        shadowed_token_count,
        provider: result.provider.clone(),
        model: result.model.clone(),
        max_tokens: result.max_tokens,
        usage: result.usage.clone(),
    };
    let summary = CompactionEvent::Summary(Box::new(summary_event)).encode()?;

    let source = compact_checkpoint_source(compaction_id.clone(), source_command_id.clone());
    let checkpoint = SessionEvent::user_message(UserMessage::with_source(
        MessageId::random(),
        frame_summary(&result.summary),
        source.to_message_source(),
    ));

    let end = super::lock::release_lock(&start, None);
    let end_event = CompactionEvent::End(end).encode()?;

    // Sequence positions are relative to the pre-compaction log; the host
    // appends these events after `events`, so indices shift by `events.len()`.
    let base = events.len() as u64;
    let prune_count = prune_events.len() as u64;
    let start_seq = SessionSeq::new(base)
        .map_err(|_| CompactionError::Surface(SurfaceError::IndexOutOfRange { index: base as usize }))?;
    let summary_seq = SessionSeq::new(base + 1 + prune_count).map_err(|_| {
        CompactionError::Surface(SurfaceError::IndexOutOfRange {
            index: (base + 1 + prune_count) as usize,
        })
    })?;
    let end_seq = SessionSeq::new(base + 2 + prune_count).map_err(|_| {
        CompactionError::Surface(SurfaceError::IndexOutOfRange {
            index: (base + 2 + prune_count) as usize,
        })
    })?;

    let result = CompactionResult {
        compaction_id,
        source_command_id,
        start_seq,
        summary_seq,
        end_seq,
        summary: result.summary,
        shadowed_range: ShadowedRange::new(range.start, range.end),
        shadowed_seqs: region.shadowed_seqs.clone(),
        shadowed_token_count,
    };

    Ok(Some(CompactPlan {
        start: CompactionEvent::Start(start).encode()?,
        prune_events,
        summary,
        checkpoint,
        end: end_event,
        result,
    }))
}

/// Measure a heuristic per-node price from message content, so range selection
/// works without the TypeScript token meter.
fn estimate_surface_prices(events: &[SessionEvent], surface: &SurfaceProjection) -> Vec<NodePrice> {
    surface
        .nodes()
        .iter()
        .filter_map(|node| {
            let event = events.get(node.seq().get() as usize)?;
            let content = event_content(event)?;
            Some(NodePrice::new(node.seq(), measure_content(content)))
        })
        .collect()
}

/// The model-facing content blocks of one core message event.
fn event_content(event: &SessionEvent) -> Option<&[ContentBlock]> {
    match event {
        SessionEvent::Core(core) => match core {
            CoreEvent::SystemMessage { message, .. } => Some(&message.content),
            CoreEvent::DeveloperMessage { message, .. } => Some(&message.content),
            CoreEvent::UserMessage { message } => Some(&message.content),
            CoreEvent::AssistantMessage { message, .. } => Some(&message.content),
            CoreEvent::ToolResult { message, .. } => Some(&message.content),
            _ => None,
        },
        SessionEvent::Extension(_) => None,
    }
}

/// Reconstruct one `Message` from a core event, for the summarization input.
fn derive_message(event: &SessionEvent) -> Option<Message> {
    match event {
        SessionEvent::Core(core) => match core {
            CoreEvent::SystemMessage { message, .. } => {
                Some(Message::System((**message).clone()))
            }
            CoreEvent::UserMessage { message } => Some(Message::User((**message).clone())),
            CoreEvent::DeveloperMessage { message, .. } => {
                Some(Message::Developer((**message).clone()))
            }
            CoreEvent::AssistantMessage { message, .. } => {
                Some(Message::Assistant((**message).clone()))
            }
            CoreEvent::ToolResult { message, .. } => Some(Message::Tool((**message).clone())),
            _ => None,
        },
        SessionEvent::Extension(_) => None,
    }
}

/// Build the summarization input: the system head (when node 0), then the
/// region's messages in surface order.
fn build_summarization_input(
    events: &[SessionEvent],
    surface: &SurfaceProjection,
    shadowed_seqs: &[SessionSeq],
) -> Result<SummarizationInput, CompactionError> {
    let mut messages = Vec::new();
    if let Some(head) = surface.nodes().first() {
        if matches!(head.kind(), SurfaceNodeKind::System) {
            if let Some(event) = events.get(head.seq().get() as usize) {
                if let Some(message) = derive_message(event) {
                    messages.push(message);
                }
            }
        }
    }
    for seq in shadowed_seqs {
        let event = events
            .get(seq.get() as usize)
            .ok_or(CompactionError::RangeStartNotFound { start: *seq })?;
        if let Some(message) = derive_message(event) {
            messages.push(message);
        }
    }
    Ok(SummarizationInput {
        messages,
        tools: None,
    })
}

/// A pure trigger decision: should the agent loop compact at this step boundary?
///
/// `total_tokens` is the latest measured request size (the host supplies it from
/// its token meter); `spec` is the resolved pressure budget for the routed
/// target. The decision is the threshold test the TypeScript loop runs before
/// calling `compactIfNeeded`.
pub fn should_compact(total_tokens: u64, spec: &ResolvedCompactSpec) -> bool {
    total_tokens >= spec.threshold_tokens
}

/// Resolve the retention budget a transaction uses, defaulting to `0` (compact
/// the whole eligible range) when no absolute or ratio retention is configured.
pub fn resolve_retain_tokens(config: &ResolvedConfig, context_window: u64) -> u64 {
    match config.retention {
        ResolvedRetention::Tokens(tokens) => tokens,
        ResolvedRetention::Ratio(ratio) => (context_window as f64 * ratio).floor().max(0.0) as u64,
    }
}

/// A runnable compaction capability the agent loop can hold and call.
///
/// It bundles the async [`SummarizationBackend`] and the resolved policy, and
/// exposes the pure [`should_compact`] decision alongside the async [`compact`]
/// entry. The loop is not modified by this crate; it calls these methods at its
/// step boundary if it chooses to.
pub struct CompactionService {
    backend: Arc<dyn SummarizationBackend>,
    config: ResolvedConfig,
    provider: String,
    model: String,
}

impl CompactionService {
    /// Build a service from a backend and resolved policy.
    pub fn new(
        backend: Arc<dyn SummarizationBackend>,
        config: ResolvedConfig,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            backend,
            config,
            provider: provider.into(),
            model: model.into(),
        }
    }

    /// The pure trigger decision for the latest measured request.
    pub fn should_compact(&self, total_tokens: u64, spec: &ResolvedCompactSpec) -> bool {
        should_compact(total_tokens, spec)
    }

    /// Run one manual (between-turns) compaction transaction.
    pub async fn compact_now(
        &self,
        events: &[SessionEvent],
        source_command_id: Option<CommandId>,
        now: i64,
        lease_ms: i64,
        prune_config: Option<ResolvedPruneConfig>,
    ) -> Result<Option<CompactPlan>, CompactionError> {
        let options = CompactOptions {
            manual: true,
            source_command_id,
            now,
            lease_ms,
            retain_tokens: 0,
            provider: self.provider.clone(),
            model: self.model.clone(),
            max_tokens: Some(self.config.max_tokens),
            prune_config,
        };
        compact(events, self.backend.as_ref(), &self.config, &options).await
    }

    /// Run one automatic (open-turn) compaction transaction for a measured request.
    pub async fn compact_if_needed(
        &self,
        events: &[SessionEvent],
        retain_tokens: u64,
        now: i64,
        lease_ms: i64,
        prune_config: Option<ResolvedPruneConfig>,
    ) -> Result<Option<CompactPlan>, CompactionError> {
        let options = CompactOptions {
            manual: false,
            source_command_id: None,
            now,
            lease_ms,
            retain_tokens,
            provider: self.provider.clone(),
            model: self.model.clone(),
            max_tokens: Some(self.config.max_tokens),
            prune_config,
        };
        compact(events, self.backend.as_ref(), &self.config, &options).await
    }
}
