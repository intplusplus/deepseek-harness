//! The public projection contract: state validation, typed unit definitions,
//! cursors, checkpoints, and snapshots.

#![forbid(unsafe_code)]

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use dsh_session::{
    SessionEvent, SessionHeader, SessionLogOffset, SessionSeq, SessionSeqCursor, MAX_SAFE_INTEGER,
};
use serde::de::{self, DeserializeOwned, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

/// A projection's persisted fold state.
///
/// The TypeScript contract requires plain JSON so the projection cache can
/// persist a row as `(sessionId, key, ver, seq, val)`. Rust expresses the same
/// precondition through this trait's `Serialize`/`DeserializeOwned` bounds, and
/// adds [`ProjectionState::validate`] for the invariants serde's structural
/// parsing cannot express (non-negative counters, strict ordering, preview
/// budgets).
///
/// `serde` is deliberately not a replacement for the TypeScript Zod schemas:
/// a derive checks field types and presence, while `validate` owns semantic
/// rules. State structs additionally use `#[serde(deny_unknown_fields)]` so a
/// persisted row written by a different schema version cannot silently lose
/// fields.
pub trait ProjectionState: Serialize + DeserializeOwned + Send + Sync + 'static {
    /// Reject a state value that parsed but violates this unit's invariants.
    ///
    /// Returns a stable, human-readable reason; the caller wraps it in
    /// [`ProjectionError::InvalidState`].
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

pub(crate) type ErasedWire = Arc<dyn Any>;

pub(crate) type InitFn<S> = Box<dyn Fn(&SessionHeader, SessionLogOffset) -> S>;
pub(crate) type ApplyFn<S> = Box<dyn Fn(&S, &SessionEvent, SessionSeq) -> Option<S>>;
pub(crate) type ViewFn<S> = Box<dyn Fn(&S) -> ErasedWire>;
pub(crate) type WireJsonFn = Box<dyn Fn(&ErasedWire) -> Result<Value, String>>;

pub(crate) struct WireView<S> {
    pub(crate) view: ViewFn<S>,
    pub(crate) to_json: WireJsonFn,
}

/// One domain's state-driven computation unit: a pure synchronous fold plus
/// its optional client view.
///
/// The definition is registered on a [`crate::ProjectionRegistry`], which owns
/// the drive, the per-session watermark, and the change feed. Domain code owns
/// only the computation and must not call back into the registry from `init`,
/// `apply`, or `view`.
///
/// ## Identity contract
///
/// `apply` returns `Option<S>` instead of `S`: `None` means “this event is not
/// mine” and the registry reuses the exact cached `Arc`, so no state is cloned
/// and no downstream work happens. Returning `Some(state)` publishes a new
/// state reference; returning `Some` with an equal value still counts as a
/// change. This is Rust's equivalent of the TypeScript `Object.is` gate, made
/// impossible to violate by accident.
///
/// A client `view` returns `Arc<W>` for the same reason: the registry compares
/// two consecutive view allocations with [`Arc::ptr_eq`]. A unit whose view is
/// derived from a sub-value stored in an `Arc` (for example a turn list)
/// returns a clone of that `Arc` to suppress publication across internal-only
/// state changes; a unit that always allocates a fresh view publishes on every
/// state change, exactly like the TypeScript implementation that returns a
/// fresh object literal.
pub struct ProjectionDefinition<S: ProjectionState> {
    pub(crate) key: &'static str,
    pub(crate) state_version: u64,
    pub(crate) init: InitFn<S>,
    pub(crate) apply: ApplyFn<S>,
    pub(crate) wire: Option<WireView<S>>,
}

impl<S: ProjectionState> ProjectionDefinition<S> {
    /// Create a host-only definition.
    ///
    /// Its state is checkpointed like every other unit but never appears in a
    /// client snapshot and never reaches the change feed. An empty key is
    /// rejected later by [`crate::ProjectionRegistry::register`].
    pub fn host_only(
        key: &'static str,
        state_version: u64,
        init: impl Fn(&SessionHeader, SessionLogOffset) -> S + 'static,
        apply: impl Fn(&S, &SessionEvent, SessionSeq) -> Option<S> + 'static,
    ) -> Self {
        Self {
            key,
            state_version,
            init: Box::new(init),
            apply: Box::new(apply),
            wire: None,
        }
    }

    /// Attach a client-visible view to this definition.
    ///
    /// `view` must return the whole current client value for the key. It is
    /// called only when a state change occurred and at least one change
    /// listener is subscribed; snapshots call it on demand. `validate` owns
    /// the semantic rules the wire schema enforces in TypeScript (for example
    /// preview length and strict turn ordering).
    ///
    /// The returned `Arc` is the comparison unit: returning the same allocation
    /// as the previous call suppresses the change notification.
    #[must_use]
    pub fn with_wire<W>(
        mut self,
        view: impl Fn(&S) -> Arc<W> + 'static,
        validate: impl Fn(&W) -> Result<(), String> + 'static,
    ) -> Self
    where
        W: Serialize + DeserializeOwned + 'static,
    {
        self.wire = Some(WireView {
            view: Box::new(move |state| {
                let wire: Arc<W> = view(state);
                wire
            }),
            to_json: Box::new(move |wire: &ErasedWire| {
                let typed = wire
                    .downcast_ref::<W>()
                    .ok_or_else(|| "client view produced a different wire type".to_string())?;
                validate(typed)?;
                let value = serde_json::to_value(typed).map_err(|error| error.to_string())?;
                // A second parse proves the value survives JSON with its
                // declared shape (the TypeScript viewSchema role).
                let _: W =
                    serde_json::from_value(value.clone()).map_err(|error| error.to_string())?;
                Ok(value)
            }),
        });
        self
    }

    /// The projection key this unit owns.
    pub fn key(&self) -> &'static str {
        self.key
    }

    /// The persisted-cache invalidation version.
    ///
    /// Must be at most [`MAX_SAFE_INTEGER`] to stay representable in a
    /// TypeScript cache row; [`crate::ProjectionRegistry::register`] enforces
    /// the bound.
    pub fn state_version(&self) -> u64 {
        self.state_version
    }

    /// Whether this unit contributes a client-visible view.
    pub fn has_wire(&self) -> bool {
        self.wire.is_some()
    }
}

/// An inclusive session watermark, or the `-1` sentinel before the first event.
///
/// TypeScript uses a branded number with `-1` for the empty log. Rust's
/// [`dsh_session::SessionSeq`] cannot be negative, so this newtype carries the
/// absence explicitly while serializing to the TypeScript bytes (`-1` or the
/// sequence number).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ProjectionCursor(Option<SessionSeq>);

impl ProjectionCursor {
    /// The cursor before the first event (serialized as `-1`).
    pub const BEFORE_LOG: Self = Self(None);

    /// A cursor at one existing event.
    pub fn at(seq: SessionSeq) -> Self {
        Self(Some(seq))
    }

    /// The cursor immediately before `offset`; `0` maps to [`Self::BEFORE_LOG`].
    pub fn before_offset(offset: SessionLogOffset) -> Self {
        Self::from_position(offset.get() as i64 - 1)
    }

    /// Build a cursor from a raw position, mapping negative positions to absence.
    ///
    /// Positions above the safe-integer bound cannot be represented by
    /// [`SessionSeq`]; this is unreachable for values derived from a session log
    /// length and falls back to absence rather than panicking.
    pub fn from_position(position: i64) -> Self {
        if position < 0 {
            return Self(None);
        }
        match u64::try_from(position) {
            Ok(value) => match SessionSeq::new(value) {
                Ok(seq) => Self(Some(seq)),
                Err(_) => Self(None),
            },
            Err(_) => Self(None),
        }
    }

    /// The explicit cursor.
    pub fn cursor(self) -> SessionSeqCursor {
        self.0
    }

    /// The positional form used for ordered comparisons (`-1` for absence).
    pub fn position(self) -> i64 {
        match self.0 {
            Some(seq) => seq.get() as i64,
            None => -1,
        }
    }
}

impl From<SessionSeqCursor> for ProjectionCursor {
    fn from(cursor: SessionSeqCursor) -> Self {
        Self(cursor)
    }
}

impl From<ProjectionCursor> for SessionSeqCursor {
    fn from(cursor: ProjectionCursor) -> Self {
        cursor.0
    }
}

impl Serialize for ProjectionCursor {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_i64(self.position())
    }
}

impl<'de> Deserialize<'de> for ProjectionCursor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct CursorVisitor;

        impl Visitor<'_> for CursorVisitor {
            type Value = ProjectionCursor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a session cursor: -1 or a safe non-negative integer")
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                if value < -1 {
                    return Err(E::custom(format!("session cursor {value} is below -1")));
                }
                if value == -1 {
                    return Ok(ProjectionCursor::BEFORE_LOG);
                }
                match u64::try_from(value) {
                    Ok(value) if value <= MAX_SAFE_INTEGER => SessionSeq::new(value)
                        .map(ProjectionCursor::at)
                        .map_err(|error| E::custom(error.to_string())),
                    _ => Err(E::custom(format!(
                        "session cursor {value} exceeds the safe-integer bound"
                    ))),
                }
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                if value > MAX_SAFE_INTEGER {
                    return Err(E::custom(format!(
                        "session cursor {value} exceeds the safe-integer bound"
                    )));
                }
                SessionSeq::new(value)
                    .map(ProjectionCursor::at)
                    .map_err(|error| E::custom(error.to_string()))
            }
        }

        deserializer.deserialize_any(CursorVisitor)
    }
}

/// One unit's checkpoint row: state plus the watermark that produced it.
///
/// Field order matches the TypeScript `ProjectionCheckpointRow`
/// (`ver`, `seq`, `val`). The row is a fold shortcut, never authoritative:
/// restore discards it on a version mismatch or when it claims events past the
/// stored log end.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProjectionCheckpointRow {
    /// The registering unit's `stateVersion` at fold time.
    pub ver: u64,
    /// The last event folded into `val`; `-1` for the empty log.
    pub seq: ProjectionCursor,
    /// The unit's internal state, plain JSON per [`ProjectionState`].
    pub val: Value,
}

/// Checkpoint rows keyed by projection key.
pub type ProjectionCheckpoint = BTreeMap<String, ProjectionCheckpointRow>;

/// One consistent read cut over every registered client-visible unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectionSnapshot {
    /// Seq of the last event every value reflects; `-1` for an empty log.
    pub as_of_seq: ProjectionCursor,
    /// Whole current client value per registered key, in registration order.
    pub values: Map<String, Value>,
}

/// One validated client-view change.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionChange {
    /// The session whose cell changed.
    pub session_id: dsh_session::SessionId,
    /// The unit's projection key.
    pub key: &'static str,
    /// The schema-validated view value at `seq`.
    pub value: Value,
    /// The sequence number of the event that caused the change.
    pub seq: SessionSeq,
}

/// The result of folding a stored log suffix over a checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct RestoreOutcome {
    /// Values at the supplied log end.
    pub snapshot: ProjectionSnapshot,
    /// Refreshed rows at that cut, ready for a durable write-back.
    pub checkpoint: ProjectionCheckpoint,
}

/// A projection registration, key, or persisted-row failure.
#[derive(Debug, thiserror::Error)]
pub enum ProjectionError {
    /// A projection key must not be empty.
    #[error("projection key must not be empty")]
    EmptyKey,
    /// A registration carries a version no TypeScript cache row can represent.
    #[error("projection `{key}` stateVersion {version} exceeds the safe-integer bound {max}")]
    InvalidStateVersion {
        /// The affected key.
        key: &'static str,
        /// The rejected version.
        version: u64,
        /// The largest representable version.
        max: u64,
    },
    /// The same key was registered under a different persisted-cache version.
    #[error(
        "projection key `{key}` is already registered at stateVersion {existing}; refusing to share it with stateVersion {requested}"
    )]
    StateVersionConflict {
        /// The key already registered.
        key: &'static str,
        /// The version already installed.
        existing: u64,
        /// The version the caller tried to install.
        requested: u64,
    },
    /// The cached state behind a key is not the type the caller requested.
    #[error("projection `{key}` state type does not match the registered definition")]
    StateTypeMismatch {
        /// The affected key.
        key: &'static str,
    },
    /// The client view behind a key is not the type the definition produces.
    #[error("projection `{key}` client view type does not match the registered definition")]
    WireTypeMismatch {
        /// The affected key.
        key: &'static str,
    },
    /// A persisted state parsed as JSON but failed its semantic validation.
    #[error("projection `{key}` rejected persisted state: {reason}")]
    InvalidState {
        /// The affected key.
        key: &'static str,
        /// Why the value was rejected.
        reason: String,
    },
    /// A client view failed validation or JSON encoding.
    #[error("projection `{key}` rejected its client view: {reason}")]
    InvalidWire {
        /// The affected key.
        key: &'static str,
        /// Why the value was rejected.
        reason: String,
    },
    /// The fold needed an event that the session does not contain.
    #[error("projection `{key}` cannot advance across missing seq {seq}")]
    MissingSeq {
        /// The affected key.
        key: &'static str,
        /// The sequence that was required.
        seq: u64,
    },
    /// A registration exists but its per-session cell is absent.
    #[error("projection `{key}` has no cell for this session")]
    MissingCell {
        /// The affected key.
        key: &'static str,
    },
    /// A checkpoint row cannot seed a tail read and the caller asked for one.
    #[error(
        "projection `{key}` cannot restore from seq {base_seq}: its checkpoint row is missing, version-mismatched, or beyond the supplied log end; re-read from seq 0"
    )]
    UnrestorableCheckpoint {
        /// The affected key.
        key: &'static str,
        /// The first sequence supplied to the tail read.
        base_seq: u64,
    },
    /// The requested event is outside the session log.
    #[error("session has no event at seq {seq}")]
    MissingEvent {
        /// The requested sequence.
        seq: u64,
    },
    /// A session offset or restore cut overflowed its branded bound.
    #[error("session log offset overflow: {0}")]
    Offset(String),
}
