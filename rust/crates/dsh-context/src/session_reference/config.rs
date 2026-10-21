//! Session-reference configuration, budgets, and stable diagnostics.
//!
//! Ported from `packages/context/session-reference/src/config.ts`. The raw
//! configuration keeps JSON numbers as `f64` so the safe-integer diagnostics
//! match the TypeScript constructor exactly; resolved values are integers.

#![forbid(unsafe_code)]

use thiserror::Error;

/// Hard maximum references accepted by one message.
pub const MAX_REFERENCES: usize = 3;

/// Default number of discovery candidates returned to a host.
pub const DEFAULT_CANDIDATE_LIMIT: u64 = 50;

/// Minimum automatic UTF-8 budget for one rendered reference JSON object.
pub const DEFAULT_MAX_REFERENCE_BYTES: u64 = 65_536;

/// Default fraction of a model context window reserved per source.
pub const DEFAULT_REFERENCE_CONTEXT_FRACTION: f64 = 0.2;

/// The largest JavaScript safe integer, exactly representable as `f64`.
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

/// Session-reference service configuration.
///
/// Every field is optional and takes the documented TypeScript default when
/// absent. `max_reference_bytes` stays absent rather than defaulting to 64 KiB,
/// because an absent value selects the model-relative budget while an explicit
/// one is an exact per-source cap.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Config {
    /// Maximum distinct source sessions referenced by one message, 1 to 3.
    pub max_references: Option<f64>,
    /// Default host candidate-list limit.
    pub candidate_limit: Option<f64>,
    /// Explicit maximum rendered UTF-8 bytes per source.
    pub max_reference_bytes: Option<f64>,
    /// Fraction of the model context window per source, between zero and one.
    pub reference_context_fraction: Option<f64>,
}

/// Configuration with defaults applied and numeric fields validated.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedConfig {
    max_references: usize,
    candidate_limit: u64,
    max_reference_bytes: Option<u64>,
    reference_context_fraction: f64,
}

impl Default for ResolvedConfig {
    fn default() -> Self {
        Self {
            max_references: MAX_REFERENCES,
            candidate_limit: DEFAULT_CANDIDATE_LIMIT,
            max_reference_bytes: None,
            reference_context_fraction: DEFAULT_REFERENCE_CONTEXT_FRACTION,
        }
    }
}

impl ResolvedConfig {
    /// Validate a deployment-authored configuration and apply its defaults.
    ///
    /// # Errors
    ///
    /// Returns [`SessionReferenceError::InvalidPositiveInteger`] for a
    /// non-positive or non-safe-integer cap,
    /// [`SessionReferenceError::MaxReferencesTooHigh`] above three, and
    /// [`SessionReferenceError::InvalidReferenceContextFraction`] outside
    /// zero through one (including `NaN`).
    pub fn new(config: &Config) -> Result<Self, SessionReferenceError> {
        let max_references = match config.max_references {
            Some(value) => validate_positive_safe_integer("maxReferences", value)?,
            None => MAX_REFERENCES as u64,
        };
        let candidate_limit = match config.candidate_limit {
            Some(value) => validate_positive_safe_integer("candidateLimit", value)?,
            None => DEFAULT_CANDIDATE_LIMIT,
        };
        let max_reference_bytes = match config.max_reference_bytes {
            Some(value) => Some(validate_positive_safe_integer("maxReferenceBytes", value)?),
            None => None,
        };
        let reference_context_fraction = config
            .reference_context_fraction
            .unwrap_or(DEFAULT_REFERENCE_CONTEXT_FRACTION);
        if !(0.0..=1.0).contains(&reference_context_fraction) {
            return Err(SessionReferenceError::InvalidReferenceContextFraction);
        }
        if max_references > MAX_REFERENCES as u64 {
            return Err(SessionReferenceError::MaxReferencesTooHigh {
                max: MAX_REFERENCES,
            });
        }
        Ok(Self {
            max_references: max_references as usize,
            candidate_limit,
            max_reference_bytes,
            reference_context_fraction,
        })
    }

    /// Maximum distinct sources one message may reference.
    pub fn max_references(&self) -> usize {
        self.max_references
    }

    /// Default discovery result cap.
    pub fn candidate_limit(&self) -> u64 {
        self.candidate_limit
    }

    /// Explicit per-source byte cap, when configured.
    pub fn max_reference_bytes(&self) -> Option<u64> {
        self.max_reference_bytes
    }

    /// Fraction of the context window reserved per source.
    pub fn reference_context_fraction(&self) -> f64 {
        self.reference_context_fraction
    }

    /// The per-source UTF-8 byte budget for one rendered reference object.
    ///
    /// An explicit cap wins. Otherwise the budget is four bytes per token of
    /// `context_window` scaled by [`Self::reference_context_fraction`], with a
    /// 64 KiB floor. Missing model capacity selects the floor.
    pub fn reference_budget(&self, context_window: Option<u64>) -> u64 {
        if let Some(max_bytes) = self.max_reference_bytes {
            return max_bytes;
        }
        let Some(context_window) = context_window else {
            return DEFAULT_MAX_REFERENCE_BYTES;
        };
        let scaled = (context_window as f64 * 4.0 * self.reference_context_fraction).floor();
        // A non-negative finite `f64` cast saturates at `u64::MAX`.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let scaled = scaled as u64;
        scaled.max(DEFAULT_MAX_REFERENCE_BYTES)
    }
}

/// Validate one positive JavaScript safe integer.
fn validate_positive_safe_integer(
    name: &'static str,
    value: f64,
) -> Result<u64, SessionReferenceError> {
    if !value.is_finite() || value.fract() != 0.0 || !(1.0..=MAX_SAFE_INTEGER).contains(&value) {
        return Err(SessionReferenceError::InvalidPositiveInteger { name });
    }
    // A non-negative integral `f64` at or below 2^53 - 1 converts exactly.
    Ok(value as u64)
}

/// Stable failure codes exposed to host adapters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionReferenceErrorCode {
    /// The service configuration is invalid.
    InvalidConfig,
    /// A reference, mention, or URI is malformed.
    InvalidReference,
    /// A message tried to reference its own session.
    SelfReference,
    /// A message exceeded the configured reference count.
    TooMany,
    /// A source session could not be read.
    ReadFailed,
    /// Fixed snapshot data does not fit the byte budget.
    BudgetExceeded,
    /// Preparation was cancelled.
    Cancelled,
}

impl SessionReferenceErrorCode {
    /// The wire code used by host protocol error mapping.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidConfig => "SESSION_REFERENCE_INVALID_CONFIG",
            Self::InvalidReference => "SESSION_REFERENCE_INVALID_REFERENCE",
            Self::SelfReference => "SESSION_REFERENCE_SELF_REFERENCE",
            Self::TooMany => "SESSION_REFERENCE_TOO_MANY",
            Self::ReadFailed => "SESSION_REFERENCE_READ_FAILED",
            Self::BudgetExceeded => "SESSION_REFERENCE_BUDGET_EXCEEDED",
            Self::Cancelled => "SESSION_REFERENCE_CANCELLED",
        }
    }
}

/// Why session-reference configuration, parsing, or preparation failed.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SessionReferenceError {
    /// A configured cap is not a positive JavaScript safe integer.
    #[error("session-reference: {name} must be a positive safe integer")]
    InvalidPositiveInteger {
        /// The configuration field name.
        name: &'static str,
    },
    /// `maxReferences` exceeds the hard maximum.
    #[error("session-reference: maxReferences must not exceed {max}")]
    MaxReferencesTooHigh {
        /// The hard maximum.
        max: usize,
    },
    /// The context fraction is outside zero through one.
    #[error("session-reference: referenceContextFraction must be between zero and one")]
    InvalidReferenceContextFraction,
    /// A direct candidate limit argument is invalid.
    #[error("candidate limit must be a positive safe integer")]
    InvalidCandidateLimit,
    /// A message referenced its own session.
    #[error("session {session_id} cannot reference itself")]
    SelfReference {
        /// The target session id rendered as a JSON string literal.
        session_id: String,
    },
    /// A message referenced more distinct sessions than allowed.
    #[error("a message may reference at most {max} sessions")]
    TooMany {
        /// The configured maximum.
        max: usize,
    },
    /// A source session could not be read.
    #[error("failed to read referenced session: {message}")]
    ReadFailed {
        /// The host-supplied read failure.
        message: String,
    },
    /// Fixed snapshot data cannot fit the configured byte budget.
    #[error("referenced session snapshot cannot fit the configured byte budget")]
    BudgetExceeded,
    /// Preparation was cancelled.
    #[error("session reference preparation was cancelled")]
    Cancelled,
    /// A URI is not a canonical encoded session reference.
    #[error("invalid session reference URI {uri}")]
    InvalidUri {
        /// The rejected URI rendered as a JSON string literal.
        uri: String,
    },
    /// The reference data cannot be represented as JSON.
    #[error("session-reference data is not JSON-serializable")]
    NotJsonSerializable,
    /// The caller-supplied snapshots do not line up with the normalized inputs.
    #[error("referenced session snapshots do not match normalized references")]
    SnapshotMismatch,
}

impl SessionReferenceError {
    /// The stable host-routing code for this failure.
    pub fn code(&self) -> SessionReferenceErrorCode {
        match self {
            Self::InvalidPositiveInteger { .. }
            | Self::MaxReferencesTooHigh { .. }
            | Self::InvalidReferenceContextFraction => SessionReferenceErrorCode::InvalidConfig,
            Self::InvalidCandidateLimit | Self::InvalidUri { .. } | Self::NotJsonSerializable => {
                SessionReferenceErrorCode::InvalidReference
            }
            Self::SelfReference { .. } => SessionReferenceErrorCode::SelfReference,
            Self::TooMany { .. } => SessionReferenceErrorCode::TooMany,
            Self::ReadFailed { .. } | Self::SnapshotMismatch => {
                SessionReferenceErrorCode::ReadFailed
            }
            Self::BudgetExceeded => SessionReferenceErrorCode::BudgetExceeded,
            Self::Cancelled => SessionReferenceErrorCode::Cancelled,
        }
    }
}
