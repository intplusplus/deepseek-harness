//! Failure types for compaction policy, event decoding, surface folding, and
//! pruning.
//!
//! The variants keep the TypeScript diagnostics intact where a human command or
//! a log warning already depends on them. Configuration failures carry their
//! exact provider/model key so a caller can suppress a repeated warning for one
//! routed target.

#![forbid(unsafe_code)]

use dsh_session::SessionSeq;

/// A validation failure in the basic-compaction configuration.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// A field, pair, or duplicate-target rule failed validation.
    #[error("{0}")]
    Invalid(String),
}

impl ConfigError {
    /// Create one configuration failure with an actionable message.
    pub fn new(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

/// A capacity or retention failure for one exact provider/model route.
///
/// The TypeScript backend catches this type to warn once per target instead of
/// once per turn. `message` is the complete diagnostic; `target_key` is the
/// `provider/model` warning key.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct TargetPressureConfigError {
    /// Exact provider/model route used as the warning key.
    pub target_key: String,
    /// Complete, actionable configuration failure detail.
    pub message: String,
}

impl TargetPressureConfigError {
    /// Create one target-specific pressure failure.
    pub fn new(target_key: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            target_key: target_key.into(),
            message: message.into(),
        }
    }
}

/// A failure while deriving tool-call/result balance over a surface.
#[derive(Debug, thiserror::Error)]
pub enum ToolPairingError {
    /// A requested cut names a sequence that is not a current surface node.
    #[error("tool-pairing balance: surface seq {seq} not found")]
    MissingSeq {
        /// The absent surface sequence.
        seq: SessionSeq,
    },
    /// A result appears without a preceding unanswered call.
    #[error(
        "tool-pairing balance: tool/result at surface seq {seq} has no matching tool-call (corrupt surface)"
    )]
    OrphanResult {
        /// The offending result sequence.
        seq: SessionSeq,
    },
}

/// A failure while replaying compaction replacements over a session surface.
#[derive(Debug, thiserror::Error)]
pub enum SurfaceError {
    /// A known `compaction/*` payload is malformed.
    #[error(transparent)]
    Event(#[from] serde_json::Error),
    /// The metering event named no shadowed sequences.
    #[error("compaction: shadowed seqs must not be empty")]
    EmptyShadowedSeqs,
    /// The declared start/end do not name the first/last shadowed sequence.
    #[error(
        "compaction: shadowedRange {start}-{end} does not match the first and last shadowed seqs"
    )]
    RangeMismatch {
        /// Declared first surface-node sequence.
        start: SessionSeq,
        /// Declared last surface-node sequence.
        end: SessionSeq,
    },
    /// A shadowed sequence is absent from the current surface.
    #[error("compaction: shadowed seq {seq} not found in surface")]
    MissingShadowedSeq {
        /// The absent sequence.
        seq: SessionSeq,
    },
    /// The shadowed sequences are not one contiguous, ordered surface span.
    #[error("compaction: shadowed seq {seq} is not contiguous in the current surface")]
    NotContiguous {
        /// The sequence that broke contiguity.
        seq: SessionSeq,
    },
    /// A `compaction/prune` event must rewrite exactly one tool result.
    #[error("compaction: prune replacement must rewrite exactly one current tool/result")]
    PruneSpanNotSingle,
    /// The replacement changed a field other than the tool-result content.
    #[error("compaction: prune may change only tool-result content")]
    PruneRewriteMismatch,
    /// The surface node named by a prune event is not a tool result.
    #[error("compaction: surface seq {seq} is not a tool/result event (corrupt surface)")]
    MissingToolResult {
        /// The offending surface sequence.
        seq: SessionSeq,
    },
    /// Surface node 0 holds a protected system prompt.
    #[error(
        "compaction: node 0 holds the system prompt and may be rewritten only by a system/message over exactly that node"
    )]
    SystemHeadProtected,
    /// A sequence in the fold names no event at its log position.
    #[error("compaction: surface seq {seq} has no matching session event (corrupt surface)")]
    MissingSurfaceEvent {
        /// The absent log position.
        seq: SessionSeq,
    },
    /// A log index exceeds the safe-integer bound the sequence brand admits.
    #[error("compaction: surface index {index} exceeds the safe-integer bound")]
    IndexOutOfRange {
        /// The offending vector index.
        index: usize,
    },
}

/// A failure while planning or measuring a tool-result prune.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PruneError {
    /// Character budgets failed validation.
    #[error("{0}")]
    Config(String),
    /// No text block intersected the configured removed span.
    #[error("tool-result prune: failed to locate the removed text span")]
    MissingRemovedSpan,
    /// The emitted replacement was not smaller than the input.
    #[error("tool-result prune: replacement must be smaller and within threshold")]
    ReplacementNotSmaller,
}

/// A failure from one of the compaction capabilities.
#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
    /// Basic-compaction configuration is invalid.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// One routed target cannot be scaled into a pressure budget.
    #[error(transparent)]
    TargetPressure(#[from] TargetPressureConfigError),
    /// Tool-pairing balance could not be derived.
    #[error(transparent)]
    Pairing(#[from] ToolPairingError),
    /// A tool-result prune failed.
    #[error(transparent)]
    Prune(#[from] PruneError),
    /// A compaction replacement could not be folded.
    #[error(transparent)]
    Surface(#[from] SurfaceError),
    /// A compaction extension event failed to encode to the durable log.
    #[error(transparent)]
    Encode(#[from] serde_json::Error),
    /// Another compaction transaction still owns the durable lock.
    #[error(
        "{stage}: compaction already in progress; the session compaction lock is already active"
    )]
    CompactionBusy {
        /// Operation label included in the diagnostic.
        stage: String,
    },
    /// The summarization provider call failed.
    #[error("compaction summarization failed: {0}")]
    SummaryProvider(String),
    /// Summarization produced no usable text content.
    #[error("compaction summarization produced no text summary content")]
    SummaryEmpty,
    /// The priced nodes are not the current positional surface.
    #[error("compaction: token-meter surface does not match the current session surface")]
    TokenMeterSurfaceMismatch,
    /// The requested range start is not a current surface node.
    #[error("compactRegion: start seq {start} not found in surface")]
    RangeStartNotFound {
        /// The absent start sequence.
        start: SessionSeq,
    },
    /// The requested range end is not a current surface node.
    #[error("compactRegion: end seq {end} not found in surface")]
    RangeEndNotFound {
        /// The absent end sequence.
        end: SessionSeq,
    },
    /// The start position is after the end position.
    #[error(
        "compactRegion: start seq {start} (position {start_position}) is after end seq {end} (position {end_position}) on the surface"
    )]
    RangeReversed {
        /// The requested start sequence.
        start: SessionSeq,
        /// The requested end sequence.
        end: SessionSeq,
        /// Surface position of `start`.
        start_position: usize,
        /// Surface position of `end`.
        end_position: usize,
    },
    /// The cut before the range start splits a tool-call/result pair.
    #[error(
        "compactRegion: start seq {start} is not a balanced boundary (would split a step's tool-call/result pair)"
    )]
    RangeStartUnbalanced {
        /// The requested start sequence.
        start: SessionSeq,
    },
    /// The cut after the range end splits a pair or leaves a call open.
    #[error(
        "compactRegion: end seq {end} is not a balanced boundary (would split a step, or the step is still open)"
    )]
    RangeEndUnbalanced {
        /// The requested end sequence.
        end: SessionSeq,
    },
    /// Automatic compaction requires an open turn owner.
    #[error(
        "compactRegion: no open turn - automatic compaction events must be enclosed in a turn"
    )]
    NoOpenTurn,
    /// Manual compaction requires the session to be between turns.
    #[error("manual compaction: the session already has an open turn")]
    ManualTurnOpen,
    /// The bounded summary attempts did not lower pressure below the threshold.
    #[error(
        "compaction still above threshold after {attempts} compaction attempts ({total_tokens} estimated tokens >= threshold {threshold_tokens})"
    )]
    StillAboveThreshold {
        /// Number of attempts performed (`compactionRetries + 1`).
        attempts: u64,
        /// Latest measured total tokens.
        total_tokens: u64,
        /// The routed pressure threshold.
        threshold_tokens: u64,
    },
}
