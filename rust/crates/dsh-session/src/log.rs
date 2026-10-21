//! The in-memory session: an append-only log of typed events with runtime
//! admission at the append boundary.

#![forbid(unsafe_code)]

use super::{
    event::{CoreEvent, SessionEvent},
    types::*,
};

/// The in-memory, event-sourced model of one session.
///
/// A session is the single source of truth for an agent's whole interaction
/// history: the LLM message history is *derived* from the log, never stored
/// separately; replay is re-derivation from the same events.
#[derive(Clone, Debug, PartialEq)]
pub struct Session {
    id: SessionId,
    header: SessionHeader,
    events: Vec<SessionEvent>,
}

impl Session {
    /// Create a fresh session stamped with the current format version.
    pub fn create(id: SessionId, meta: SessionMeta, created_at: u64) -> Self {
        let header = SessionHeader {
            version: SESSION_FORMAT_VERSION,
            id,
            created_at,
            cwd: meta.cwd,
            parent_session: meta.parent_session,
            is_seeded: meta.is_seeded,
            origin: meta.origin,
            delegation_depth: meta.delegation_depth,
            agent_preset: meta.agent_preset,
        };
        Self {
            id: header.id.clone(),
            header,
            events: Vec::new(),
        }
    }

    /// Restore a session from storage: the header plus its events.
    ///
    /// Every restored event is re-admitted through the same runtime
    /// validation as a live append, so a stored log cannot carry data the
    /// writer would have rejected.
    pub fn restore(header: SessionHeader, events: Vec<SessionEvent>) -> Result<Self, SessionError> {
        match header.version.cmp(&SESSION_FORMAT_VERSION) {
            std::cmp::Ordering::Greater => {
                return Err(SessionError::UnsupportedVersion {
                    found: header.version,
                    current: SESSION_FORMAT_VERSION,
                });
            }
            std::cmp::Ordering::Less => {
                return Err(SessionError::MigrationRequired {
                    found: header.version,
                    current: SESSION_FORMAT_VERSION,
                });
            }
            std::cmp::Ordering::Equal => {}
        }
        let mut session = Self {
            id: header.id.clone(),
            header,
            events: Vec::new(),
        };
        for event in events {
            session.append(event)?;
        }
        Ok(session)
    }

    /// The session's id.
    pub fn id(&self) -> &SessionId {
        &self.id
    }

    /// The immutable validated storage metadata.
    pub fn header(&self) -> &SessionHeader {
        &self.header
    }

    /// The full event log, in append order.
    pub fn events(&self) -> &[SessionEvent] {
        &self.events
    }

    /// The number of events in the log (also the next sequence number).
    pub fn len(&self) -> u64 {
        self.events.len() as u64
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// The event at `seq`, or `None` when out of range.
    pub fn event(&self, seq: SessionSeq) -> Option<&SessionEvent> {
        self.events.get(seq.get() as usize)
    }

    /// Append one event and return its sequence number.
    ///
    /// Sequence numbers stay contiguous by construction: the appended event
    /// receives the next position. Admission runs first — see
    /// [`Session::validate`].
    pub fn append(&mut self, event: SessionEvent) -> Result<SessionSeq, SessionError> {
        self.validate(&event)?;
        self.events.push(event);
        Ok(SessionSeq::unchecked(self.events.len() as u64 - 1))
    }

    /// Runtime admission for one event:
    ///
    /// 1. **Lossless JSON** — the `isJsonValue` rule: the event must encode and
    ///    decode back to itself, so a non-serializable payload is rejected at
    ///    the source.
    /// 2. **Canonical request headers** — `tools: []` and an all-empty
    ///    `adapterDefaults` must be omitted rather than normalized.
    /// 3. **Legacy governance** — the retired v0 `request/header-delta` event
    ///    is rejected, not replayed incompletely.
    fn validate(&self, event: &SessionEvent) -> Result<(), SessionError> {
        let value = serde_json::to_value(event)
            .map_err(|error| SessionError::NotJson(error.to_string()))?;
        let round_tripped: SessionEvent = serde_json::from_value(value)
            .map_err(|error| SessionError::NotJson(error.to_string()))?;
        if &round_tripped != event {
            return Err(SessionError::NotJson(
                "event data did not round-trip losslessly".to_string(),
            ));
        }
        if let SessionEvent::Core(CoreEvent::RequestHeader { header, .. }) = event {
            if let Some(tools) = &header.tools {
                if tools.is_empty() {
                    return Err(SessionError::NonCanonicalHeader(
                        "`tools: []` must be omitted".to_string(),
                    ));
                }
            }
            if let Some(defaults) = &header.adapter_defaults {
                if defaults.is_empty() {
                    return Err(SessionError::NonCanonicalHeader(
                        "an empty `adapterDefaults` must be omitted".to_string(),
                    ));
                }
            }
        }
        if let SessionEvent::Extension(extension) = event {
            if extension.kind() == "request/header-delta" {
                return Err(SessionError::LegacyEvent(extension.kind().to_string()));
            }
        }
        Ok(())
    }
}
