//! Type erasure between the typed registration surface and the registry's
//! heterogeneous cell table.

#![forbid(unsafe_code)]

use std::any::Any;
use std::sync::Arc;

use dsh_session::{SessionEvent, SessionHeader, SessionLogOffset, SessionSeq};
use serde_json::Value;

use crate::types::{ErasedWire, ProjectionDefinition, ProjectionError, ProjectionState};

pub(crate) type ErasedState = Arc<dyn Any + Send + Sync>;

/// One registered unit after its state type has been erased.
///
/// Every downcast is checked. A mismatch is a [`ProjectionError`], never a
/// panic, because the registry's typed registration surface is the only
/// producer of typed units but the public state lookup accepts any
/// [`ProjectionState`] type.
pub(crate) trait ErasedUnit {
    fn key(&self) -> &'static str;
    fn state_version(&self) -> u64;
    fn has_wire(&self) -> bool;
    fn init(&self, header: &SessionHeader, inherited: SessionLogOffset) -> ErasedState;
    fn apply(
        &self,
        state: &ErasedState,
        event: &SessionEvent,
        seq: SessionSeq,
    ) -> Result<ErasedState, ProjectionError>;
    fn seed(&self, value: &Value) -> Result<ErasedState, ProjectionError>;
    fn validate_state(&self, state: &ErasedState) -> Result<(), ProjectionError>;
    fn wire(&self, state: &ErasedState) -> Result<ErasedWire, ProjectionError>;
    fn wire_json(&self, wire: &ErasedWire) -> Result<Value, ProjectionError>;
    fn state_json(&self, state: &ErasedState) -> Result<Value, ProjectionError>;
}

pub(crate) struct TypedUnit<S: ProjectionState> {
    definition: ProjectionDefinition<S>,
}

impl<S: ProjectionState> TypedUnit<S> {
    pub(crate) fn new(definition: ProjectionDefinition<S>) -> Self {
        Self { definition }
    }

    fn state<'a>(&self, state: &'a ErasedState) -> Result<&'a S, ProjectionError> {
        state
            .downcast_ref::<S>()
            .ok_or(ProjectionError::StateTypeMismatch {
                key: self.definition.key,
            })
    }
}

impl<S: ProjectionState> ErasedUnit for TypedUnit<S> {
    fn key(&self) -> &'static str {
        self.definition.key
    }

    fn state_version(&self) -> u64 {
        self.definition.state_version
    }

    fn has_wire(&self) -> bool {
        self.definition.wire.is_some()
    }

    fn init(&self, header: &SessionHeader, inherited: SessionLogOffset) -> ErasedState {
        Arc::new((self.definition.init)(header, inherited))
    }

    fn apply(
        &self,
        state: &ErasedState,
        event: &SessionEvent,
        seq: SessionSeq,
    ) -> Result<ErasedState, ProjectionError> {
        let previous = self.state(state)?;
        match (self.definition.apply)(previous, event, seq) {
            Some(next) => Ok(Arc::new(next)),
            None => Ok(Arc::clone(state)),
        }
    }

    fn seed(&self, value: &Value) -> Result<ErasedState, ProjectionError> {
        let state: S = serde_json::from_value(value.clone()).map_err(|error| {
            ProjectionError::InvalidState {
                key: self.definition.key,
                reason: error.to_string(),
            }
        })?;
        state
            .validate()
            .map_err(|reason| ProjectionError::InvalidState {
                key: self.definition.key,
                reason,
            })?;
        Ok(Arc::new(state))
    }

    fn validate_state(&self, state: &ErasedState) -> Result<(), ProjectionError> {
        self.state(state)?
            .validate()
            .map_err(|reason| ProjectionError::InvalidState {
                key: self.definition.key,
                reason,
            })
    }

    fn wire(&self, state: &ErasedState) -> Result<ErasedWire, ProjectionError> {
        let typed = self.state(state)?;
        let wire = self
            .definition
            .wire
            .as_ref()
            .ok_or(ProjectionError::WireTypeMismatch {
                key: self.definition.key,
            })?;
        Ok((wire.view)(typed))
    }

    fn wire_json(&self, wire: &ErasedWire) -> Result<Value, ProjectionError> {
        let view = self
            .definition
            .wire
            .as_ref()
            .ok_or(ProjectionError::WireTypeMismatch {
                key: self.definition.key,
            })?;
        (view.to_json)(wire).map_err(|reason| ProjectionError::InvalidWire {
            key: self.definition.key,
            reason,
        })
    }

    fn state_json(&self, state: &ErasedState) -> Result<Value, ProjectionError> {
        serde_json::to_value(self.state(state)?).map_err(|error| ProjectionError::InvalidState {
            key: self.definition.key,
            reason: error.to_string(),
        })
    }
}
