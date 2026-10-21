//! Compaction vocabulary and pure replay rules ported from
//! `packages/compaction`.
//!
//! This crate owns the parts of the compaction layer that need no context,
//! clock, filesystem, network, or async runtime:
//!
//! * [`events`] — the durable `compaction/*` records and their exact field
//!   order;
//! * [`checkpoint`] — the correlated replacement message source;
//! * [`surface`] — the pure surface fold that applies summary and prune
//!   replacements by their contractual adjacency;
//! * [`pairing`] — tool-call/result balance over current surface positions;
//! * [`prune`] — deterministic head/middle/tail tool-result pruning;
//! * [`state`] — lock inspection, range validation, and the replay projection;
//! * [`policy`] — configuration, capacity budgets, pressure qualification,
//!   bounded retries, and overflow recovery.
//!
//! The asynchronous backend, LLM call, token meter, attachment service, and
//! session append boundary stay in their owning packages; see `rust/README.md`
//! for the deliberate gaps.

#![forbid(unsafe_code)]

pub mod backend;
pub mod brand;
pub mod checkpoint;
pub mod command;
pub mod error;
pub mod events;
pub mod lock;
pub mod pairing;
pub mod policy;
pub mod prune;
pub mod state;
pub mod surface;

pub use brand::{CommandId, CompactionId};
pub use backend::{
    frame_summary, LlmSummarizationBackend, SummarizationBackend, SummarizationInput, SummaryResult,
};
pub use checkpoint::{
    compact_checkpoint_source, is_compact_checkpoint_source, CompactionCheckpointSource,
    COMPACT_CHECKPOINT_KIND,
};
pub use command::{
    compact, should_compact, resolve_retain_tokens, CompactOptions, CompactPlan, CompactionService,
};
pub use error::{
    CompactionError, ConfigError, PruneError, SurfaceError, TargetPressureConfigError,
    ToolPairingError,
};
pub use events::{
    CompactionEnd, CompactionEvent, CompactionPrune, CompactionResult, CompactionStart,
    CompactionSummary, CompactionTurn, ShadowedRange, COMPACTION_END_TAG, COMPACTION_EVENT_TAGS,
    COMPACTION_PRUNE_TAG, COMPACTION_START_TAG, COMPACTION_SUMMARY_TAG,
};
pub use lock::{acquire_lock, inspect_lock, release_lock, LockState, DEFAULT_LEASE_MS};
pub use pairing::{tool_pairing_balanced_after, tool_pairing_balanced_before, ToolPairing};
pub use policy::{
    compaction_attempt_limit, conversation_target, overflow_compaction_allowed,
    plan_overflow_recovery, plan_pressure_after_prune, plan_pressure_step, pressure_cleared,
    reserved_completion_tokens, resolve_compact_spec, resolve_config, resolve_target_policy,
    routed_target, should_recover_overflow, still_above_threshold, BasicCompactionConfig,
    CompactionPolicyConfig, CompactionTrigger, ModelCompactPolicyConfig, ModelTarget,
    OverflowRecovery, PressurePlan, ResolvedCompactSpec, ResolvedConfig, ResolvedRetention,
    ResolvedTargetPolicy, CONTEXT_WINDOW_EXCEEDED_CODE, DEFAULT_COMPACTION_RETRIES,
    DEFAULT_HEADROOM_TOKENS, DEFAULT_MAX_OVERFLOW_RETRIES, DEFAULT_RETAIN_RATIO,
    DEFAULT_THRESHOLD_RATIO,
};
pub use prune::{
    measure_content, plan_prune_session, prune_content, resolve_prune_config, PruneDecision,
    PrunePlan, ResolvedPruneConfig, ToolResultPruneConfig, DEFAULT_HEAD_CHARS, DEFAULT_TAIL_CHARS,
    DEFAULT_THRESHOLD_CHARS, PRUNE_MARKER,
};
pub use state::{
    assert_compaction_inactive, fold_compaction_state, inspect_compaction_entry_state,
    plan_compaction_owner, select_compactable_range, validate_surface_region, CompactionEntryState,
    CompactionState, NodePrice, SurfaceRegion,
};
pub use surface::{
    fold_surface, SurfaceNode, SurfaceNodeKind, SurfaceProjection, SurfaceReplacement,
};
