//! The synchronous projection registry: registration, eager drive, per-session
//! watermarks, the change feed, and the persisted-cache read ladder.
//!
//! This is the Rust counterpart of `SessionProjectionRegistry` without the
//! Cordis service lifecycle. Registrations and listener subscriptions are RAII
//! guards instead of effects on a plugin fiber; the host decides when to call
//! [`ProjectionRegistry::drive_event`] after committing an event.

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use dsh_session::{
    Session, SessionEvent, SessionHeader, SessionId, SessionLogOffset, SessionSeq, MAX_SAFE_INTEGER,
};
use serde_json::{Map, Value};

use crate::erased::{ErasedState, ErasedUnit, TypedUnit};
use crate::types::{
    ErasedWire, ProjectionChange, ProjectionCheckpoint, ProjectionCheckpointRow, ProjectionCursor,
    ProjectionDefinition, ProjectionError, ProjectionSnapshot, ProjectionState, RestoreOutcome,
};

struct Cell {
    state: ErasedState,
    observed: ProjectionCursor,
    /// `[previous, current]` view allocations; `None` means no usable baseline.
    views: [Option<ErasedWire>; 2],
}

impl Cell {
    fn new(state: ErasedState) -> Self {
        Self {
            state,
            observed: ProjectionCursor::BEFORE_LOG,
            views: [None, None],
        }
    }
}

#[derive(Default)]
struct SessionCells {
    cells: BTreeMap<&'static str, Cell>,
}

struct RegistrationEntry {
    unit: Arc<dyn ErasedUnit>,
    refs: usize,
}

type ChangeListener = Rc<dyn Fn(&ProjectionChange)>;

#[derive(Default)]
struct Inner {
    registrations: Vec<RegistrationEntry>,
    listeners: Vec<(u64, ChangeListener)>,
    next_listener: u64,
    sessions: BTreeMap<SessionId, SessionCells>,
}

impl Inner {
    fn registration(&self, key: &str) -> Option<&RegistrationEntry> {
        self.registrations
            .iter()
            .find(|entry| entry.unit.key() == key)
    }

    fn registration_index(&self, key: &str) -> Option<usize> {
        self.registrations
            .iter()
            .position(|entry| entry.unit.key() == key)
    }
}

/// The projection unit table and its synchronous drive.
///
/// The registry is single-threaded (`Rc`/`RefCell`) because the host drives it
/// synchronously on the session commit path. Sessions are keyed by
/// [`SessionId`]; two `Session` values with the same id share one watermark and
/// one set of cells. The TypeScript registry uses `WeakMap<Session, …>` object
/// identity; this is a deliberate Rust difference, documented on
/// [`ProjectionRegistry`].
///
/// ## Whole-value event rule
///
/// A state-carrying log event must carry the complete post-change state, never
/// a bare delta. That keeps every [`ProjectionDefinition::apply`] transition
/// trivially cheap and every served value self-describing.
#[derive(Clone, Default)]
pub struct ProjectionRegistry {
    inner: Rc<RefCell<Inner>>,
}

impl ProjectionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one domain's unit.
    ///
    /// A second registration of the same key with the same `stateVersion` adds
    /// a reference to the existing unit and shares its cells; the first
    /// definition's functions win, exactly like the TypeScript registry. A
    /// different version is refused because the persisted row shape and fold
    /// semantics cannot be shared.
    ///
    /// The returned guard removes this reference on drop; the key (and every
    /// cached cell) disappears with the last release.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionError::EmptyKey`],
    /// [`ProjectionError::InvalidStateVersion`], or
    /// [`ProjectionError::StateVersionConflict`].
    pub fn register<S: ProjectionState>(
        &self,
        definition: ProjectionDefinition<S>,
    ) -> Result<Registration, ProjectionError> {
        if definition.key.is_empty() {
            return Err(ProjectionError::EmptyKey);
        }
        if definition.state_version > MAX_SAFE_INTEGER {
            return Err(ProjectionError::InvalidStateVersion {
                key: definition.key,
                version: definition.state_version,
                max: MAX_SAFE_INTEGER,
            });
        }
        let key = definition.key;
        let version = definition.state_version;
        let mut inner = self.inner.borrow_mut();
        if let Some(index) = inner.registration_index(key) {
            let existing = inner.registrations[index].unit.state_version();
            if existing != version {
                return Err(ProjectionError::StateVersionConflict {
                    key,
                    existing,
                    requested: version,
                });
            }
            inner.registrations[index].refs += 1;
        } else {
            inner.registrations.push(RegistrationEntry {
                unit: Arc::new(TypedUnit::new(definition)),
                refs: 1,
            });
        }
        Ok(Registration {
            inner: Rc::clone(&self.inner),
            key,
        })
    }

    /// Subscribe to the change feed.
    ///
    /// The listener receives one validated value per client-visible unit whose
    /// raw view allocation changed, per driven event. Host-only units never
    /// appear. Notifications run in registration order after the event has
    /// been applied to every unit (a consistent cut), so a listener may read
    /// the registry re-entrantly.
    pub fn on_changed(&self, listener: impl Fn(&ProjectionChange) + 'static) -> ChangeSubscription {
        let mut inner = self.inner.borrow_mut();
        let id = inner.next_listener;
        inner.next_listener += 1;
        inner.listeners.push((id, Rc::new(listener)));
        ChangeSubscription {
            inner: Rc::clone(&self.inner),
            id,
        }
    }

    /// Eagerly drive one committed event through every registered unit.
    ///
    /// Missing earlier events are folded silently first, so a host may call
    /// this once per append or after a batch. The event at `seq` is the only
    /// one that can emit a change notification.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionError::MissingEvent`] when `seq` is outside the
    /// session log, or a fold/view error from a unit.
    pub fn drive_event(
        &self,
        session: &Session,
        inherited: SessionLogOffset,
        seq: SessionSeq,
    ) -> Result<(), ProjectionError> {
        let event = session
            .event(seq)
            .ok_or(ProjectionError::MissingEvent { seq: seq.get() })?;
        let target = ProjectionCursor::at(seq);
        let before = ProjectionCursor::from_position(seq.get() as i64 - 1);

        let mut inner = self.inner.borrow_mut();
        ensure_session_cells(&mut inner, session, inherited);
        for index in 0..inner.registrations.len() {
            let unit = Arc::clone(&inner.registrations[index].unit);
            let cell = cell_mut(&mut inner, session.id(), unit.key())?;
            advance_cell(unit.as_ref(), cell, session, before)?;
        }

        let listeners_present = !inner.listeners.is_empty();
        let mut changes = Vec::new();
        for index in 0..inner.registrations.len() {
            let unit = Arc::clone(&inner.registrations[index].unit);
            let cell = cell_mut(&mut inner, session.id(), unit.key())?;
            if cell.observed.position() >= target.position() {
                continue;
            }
            let previous = Arc::clone(&cell.state);
            let next = unit.apply(&previous, event, seq)?;
            let changed = !Arc::ptr_eq(&next, &previous);
            cell.state = next;
            cell.observed = target;
            if !changed || !unit.has_wire() {
                continue;
            }
            cell.views[0] = cell.views[1].take();
            if !listeners_present {
                cell.views[1] = None;
                continue;
            }
            let wire = unit.wire(&cell.state)?;
            let value = unit.wire_json(&wire)?;
            let notify = match &cell.views[0] {
                Some(previous) => !Arc::ptr_eq(previous, &wire),
                None => true,
            };
            cell.views[1] = Some(wire);
            if notify {
                changes.push(ProjectionChange {
                    session_id: session.id().clone(),
                    key: unit.key(),
                    value,
                    seq,
                });
            }
        }
        let listeners: Vec<ChangeListener> = if changes.is_empty() {
            Vec::new()
        } else {
            inner
                .listeners
                .iter()
                .map(|(_, listener)| Rc::clone(listener))
                .collect()
        };
        drop(inner);
        for change in &changes {
            for listener in &listeners {
                listener(change);
            }
        }
        Ok(())
    }

    /// Read one unit's current host state, materializing every registered unit
    /// at the session cursor first.
    ///
    /// Returns `Ok(None)` when the key is not registered. The returned value is
    /// the live cached state; callers must treat it as read-only.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionError::StateTypeMismatch`] when `S` is not the type
    /// the key was registered with.
    pub fn state_of<S: ProjectionState>(
        &self,
        session: &Session,
        inherited: SessionLogOffset,
        key: &str,
    ) -> Result<Option<Arc<S>>, ProjectionError> {
        {
            let mut inner = self.inner.borrow_mut();
            if inner.registration(key).is_none() {
                return Ok(None);
            }
            sync_session(&mut inner, session, inherited)?;
        }
        let inner = self.inner.borrow();
        let Some(entry) = inner.registration(key) else {
            return Ok(None);
        };
        let cells = inner
            .sessions
            .get(session.id())
            .ok_or(ProjectionError::MissingCell {
                key: entry.unit.key(),
            })?;
        let cell = cells.cells.get(key).ok_or(ProjectionError::MissingCell {
            key: entry.unit.key(),
        })?;
        let state = Arc::downcast::<S>(Arc::clone(&cell.state)).map_err(|_| {
            ProjectionError::StateTypeMismatch {
                key: entry.unit.key(),
            }
        })?;
        Ok(Some(state))
    }

    /// One consistent cut over every registered client-visible unit.
    ///
    /// Materialization is complete: every unit folds to the session cursor,
    /// even when only selected wire values are requested.
    ///
    /// # Errors
    ///
    /// Returns a fold or client-view validation error.
    pub fn snapshot(
        &self,
        session: &Session,
        inherited: SessionLogOffset,
    ) -> Result<ProjectionSnapshot, ProjectionError> {
        self.snapshot_selected(session, inherited, None)
    }

    /// One consistent cut over the selected client-visible units.
    ///
    /// # Errors
    ///
    /// Returns a fold or client-view validation error.
    pub fn snapshot_selected(
        &self,
        session: &Session,
        inherited: SessionLogOffset,
        keys: Option<&[&str]>,
    ) -> Result<ProjectionSnapshot, ProjectionError> {
        {
            let mut inner = self.inner.borrow_mut();
            sync_session(&mut inner, session, inherited)?;
        }
        let selected = keys.map(|keys| keys.iter().copied().collect::<BTreeSet<&str>>());
        let inner = self.inner.borrow();
        let mut values = Map::new();
        for entry in &inner.registrations {
            if !entry.unit.has_wire() {
                continue;
            }
            if let Some(selected) = &selected {
                if !selected.contains(entry.unit.key()) {
                    continue;
                }
            }
            let cell = cell_ref(&inner, session.id(), entry.unit.key())?;
            values.insert(
                entry.unit.key().to_string(),
                wire_value(entry.unit.as_ref(), &cell.state)?,
            );
        }
        Ok(ProjectionSnapshot {
            as_of_seq: last_cursor(session),
            values,
        })
    }

    /// Read only already-materialized client-visible cells, without folding.
    ///
    /// Values may trail the live session and are hints, not a complete
    /// baseline. Returns `None` when no client-visible cell exists.
    ///
    /// # Errors
    ///
    /// Returns a client-view validation error.
    pub fn cached_snapshot(
        &self,
        session: &Session,
        keys: Option<&[&str]>,
    ) -> Result<Option<ProjectionSnapshot>, ProjectionError> {
        let selected = keys.map(|keys| keys.iter().copied().collect::<BTreeSet<&str>>());
        let inner = self.inner.borrow();
        let mut values = Map::new();
        let mut as_of: Option<ProjectionCursor> = None;
        for entry in &inner.registrations {
            if !entry.unit.has_wire() {
                continue;
            }
            if let Some(selected) = &selected {
                if !selected.contains(entry.unit.key()) {
                    continue;
                }
            }
            let Some(cells) = inner.sessions.get(session.id()) else {
                continue;
            };
            let Some(cell) = cells.cells.get(entry.unit.key()) else {
                continue;
            };
            values.insert(
                entry.unit.key().to_string(),
                wire_value(entry.unit.as_ref(), &cell.state)?,
            );
            as_of = Some(match as_of {
                Some(current) if current.position() <= cell.observed.position() => current,
                _ => cell.observed,
            });
        }
        Ok(as_of.map(|as_of| ProjectionSnapshot {
            as_of_seq: as_of,
            values,
        }))
    }

    /// State-level checkpoint of every persisted unit.
    ///
    /// Each `val` is a detached JSON clone, never the live cell reference.
    ///
    /// # Errors
    ///
    /// Returns a state validation or JSON encoding error.
    pub fn checkpoint(
        &self,
        session: &Session,
        inherited: SessionLogOffset,
    ) -> Result<ProjectionCheckpoint, ProjectionError> {
        {
            let mut inner = self.inner.borrow_mut();
            sync_session(&mut inner, session, inherited)?;
        }
        let inner = self.inner.borrow();
        let mut rows = BTreeMap::new();
        for entry in &inner.registrations {
            let cell = cell_ref(&inner, session.id(), entry.unit.key())?;
            entry.unit.validate_state(&cell.state)?;
            rows.insert(
                entry.unit.key().to_string(),
                ProjectionCheckpointRow {
                    ver: entry.unit.state_version(),
                    seq: cell.observed,
                    val: entry.unit.state_json(&cell.state)?,
                },
            );
        }
        Ok(rows)
    }

    /// The stored seq a tail read must start at: one event below the lowest
    /// usable watermark.
    ///
    /// A row is usable when its `ver` matches the live unit's `stateVersion`;
    /// an absent or mismatched row pulls the floor to `0` (that key must
    /// refold the full log). The one-below anchor proves how far the stored log
    /// still extends, so [`Self::restore`] can detect a shrunken log.
    ///
    /// Returns `None` when no unit is registered.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionError::Offset`] when the anchored offset cannot be
    /// branded (unreachable for safe checkpoints).
    pub fn restore_floor(
        &self,
        checkpoint: &ProjectionCheckpoint,
    ) -> Result<Option<SessionLogOffset>, ProjectionError> {
        let inner = self.inner.borrow();
        if inner.registrations.is_empty() {
            return Ok(None);
        }
        let mut floor: Option<u64> = None;
        for entry in &inner.registrations {
            let row = checkpoint.get(entry.unit.key());
            let need = match row {
                Some(row) if row.ver == entry.unit.state_version() => {
                    row.seq.position().saturating_add(1).max(0) as u64
                }
                _ => 0,
            };
            floor = Some(match floor {
                Some(current) => current.min(need),
                None => need,
            });
        }
        let Some(floor) = floor else {
            return Ok(None);
        };
        let offset = floor.saturating_sub(1);
        let offset = SessionLogOffset::new(offset)
            .map_err(|error| ProjectionError::Offset(error.to_string()))?;
        Ok(Some(offset))
    }

    /// View version-matching checkpoint rows without any log read.
    ///
    /// Malformed or version-mismatched rows leave their key absent. A
    /// schema-valid row whose view fails validation is an error, matching the
    /// TypeScript read path.
    ///
    /// # Errors
    ///
    /// Returns a client-view validation error.
    pub fn view_checkpoint(
        &self,
        checkpoint: &ProjectionCheckpoint,
        keys: Option<&[&str]>,
    ) -> Result<Map<String, Value>, ProjectionError> {
        let selected = keys.map(|keys| keys.iter().copied().collect::<BTreeSet<&str>>());
        let inner = self.inner.borrow();
        let mut values = Map::new();
        for entry in &inner.registrations {
            if !entry.unit.has_wire() {
                continue;
            }
            if let Some(selected) = &selected {
                if !selected.contains(entry.unit.key()) {
                    continue;
                }
            }
            let Some(row) = checkpoint.get(entry.unit.key()) else {
                continue;
            };
            if row.ver != entry.unit.state_version() {
                continue;
            }
            let state = match entry.unit.seed(&row.val) {
                Ok(state) => state,
                Err(_) => continue,
            };
            values.insert(
                entry.unit.key().to_string(),
                wire_value(entry.unit.as_ref(), &state)?,
            );
        }
        Ok(values)
    }

    /// Cold read: fold every persisted unit over a stored log suffix, seeding
    /// each from its checkpoint row when usable.
    ///
    /// `events` MUST be the stored events at or past `base_seq`, in contiguous
    /// sequence order. Rust events carry no `seq` envelope, so the cut is
    /// positional: the event at index `i` is assumed to have seq
    /// `base_seq + i`. A usable row must not predate `base_seq - 1` and must
    /// not claim events past the supplied end. A discarded row with
    /// `base_seq > 0` is an error: only a full re-read from seq 0 is sound.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionError::UnrestorableCheckpoint`] for an unusable row
    /// in a tail read, or a fold/state/view error.
    pub fn restore(
        &self,
        checkpoint: &ProjectionCheckpoint,
        events: &[SessionEvent],
        base_seq: SessionLogOffset,
        header: &SessionHeader,
        inherited: SessionLogOffset,
    ) -> Result<RestoreOutcome, ProjectionError> {
        let restored = self.restore_inner(checkpoint, events, base_seq, header, inherited)?;
        Ok(RestoreOutcome {
            snapshot: restored.snapshot,
            checkpoint: restored.checkpoint,
        })
    }

    /// Restore an exact cut and install its states on the supplied session.
    ///
    /// A later [`Self::snapshot`] or [`Self::drive_event`] advances any suffix
    /// the session owns exactly once. Cells already ahead of the restored cut
    /// are kept.
    ///
    /// `events` MUST be the stored events at or past `base_seq` for this
    /// session's log prefix, with the same positional contract as
    /// [`Self::restore`].
    ///
    /// # Errors
    ///
    /// Returns a restore, fold, state, or client-view error.
    pub fn hydrate(
        &self,
        session: &Session,
        checkpoint: &ProjectionCheckpoint,
        events: &[SessionEvent],
        base_seq: SessionLogOffset,
        inherited: SessionLogOffset,
    ) -> Result<ProjectionSnapshot, ProjectionError> {
        let restored =
            self.restore_inner(checkpoint, events, base_seq, session.header(), inherited)?;
        {
            let mut inner = self.inner.borrow_mut();
            ensure_session_cells(&mut inner, session, inherited);
            for restored_cell in &restored.cells {
                let cell = cell_mut(&mut inner, session.id(), restored_cell.key)?;
                if cell.observed.position() > restored_cell.observed.position() {
                    continue;
                }
                cell.state = Arc::clone(&restored_cell.state);
                cell.observed = restored_cell.observed;
                cell.views = [None, None];
            }
        }
        Ok(restored.snapshot)
    }

    fn restore_inner(
        &self,
        checkpoint: &ProjectionCheckpoint,
        events: &[SessionEvent],
        base_seq: SessionLogOffset,
        header: &SessionHeader,
        inherited: SessionLogOffset,
    ) -> Result<Restored, ProjectionError> {
        let units: Vec<Arc<dyn ErasedUnit>> = self
            .inner
            .borrow()
            .registrations
            .iter()
            .map(|entry| Arc::clone(&entry.unit))
            .collect();
        let base_position = base_seq.get() as i64;
        let before = ProjectionCursor::from_position(base_position - 1);
        let end = if events.is_empty() {
            before
        } else {
            let length = i64::try_from(events.len())
                .map_err(|error| ProjectionError::Offset(error.to_string()))?;
            let last = base_position
                .checked_add(length - 1)
                .ok_or_else(|| ProjectionError::Offset("restore cut overflowed".to_string()))?;
            if last > MAX_SAFE_INTEGER as i64 {
                return Err(ProjectionError::Offset(
                    "restore cut exceeds the safe-integer bound".to_string(),
                ));
            }
            ProjectionCursor::from_position(last)
        };

        let mut values = Map::new();
        let mut rows = BTreeMap::new();
        let mut cells = Vec::new();
        for unit in &units {
            let key = unit.key();
            let row = checkpoint.get(key);
            let usable = match row {
                Some(row) => {
                    row.ver == unit.state_version()
                        && row.seq.position() >= before.position()
                        && row.seq.position() <= end.position()
                }
                None => false,
            };
            if !usable && base_seq.get() > 0 {
                return Err(ProjectionError::UnrestorableCheckpoint {
                    key,
                    base_seq: base_seq.get(),
                });
            }
            let (mut state, from) = match (usable, row) {
                (true, Some(row)) => (unit.seed(&row.val)?, row.seq.position()),
                _ => (unit.init(header, inherited), before.position()),
            };
            let start = from - base_position + 1;
            let start = usize::try_from(start)
                .map_err(|error| ProjectionError::Offset(error.to_string()))?;
            for (offset, event) in events.iter().enumerate().skip(start) {
                let position = base_position
                    .checked_add(offset as i64)
                    .ok_or_else(|| ProjectionError::Offset("restore cut overflowed".to_string()))?;
                let seq = SessionSeq::new(position as u64)
                    .map_err(|error| ProjectionError::Offset(error.to_string()))?;
                state = unit.apply(&state, event, seq)?;
            }
            unit.validate_state(&state)?;
            if unit.has_wire() {
                values.insert(key.to_string(), wire_value(unit.as_ref(), &state)?);
            }
            rows.insert(
                key.to_string(),
                ProjectionCheckpointRow {
                    ver: unit.state_version(),
                    seq: end,
                    val: unit.state_json(&state)?,
                },
            );
            cells.push(RestoredCell {
                key,
                state,
                observed: end,
            });
        }

        Ok(Restored {
            snapshot: ProjectionSnapshot {
                as_of_seq: end,
                values,
            },
            checkpoint: rows,
            cells,
        })
    }
}

struct RestoredCell {
    key: &'static str,
    state: ErasedState,
    observed: ProjectionCursor,
}

struct Restored {
    snapshot: ProjectionSnapshot,
    checkpoint: ProjectionCheckpoint,
    cells: Vec<RestoredCell>,
}

/// One live registration; dropping it releases the key reference.
#[must_use = "dropping the registration immediately unregisters the projection"]
pub struct Registration {
    inner: Rc<RefCell<Inner>>,
    key: &'static str,
}

impl std::fmt::Debug for Registration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Registration")
            .field("key", &self.key)
            .finish()
    }
}

impl Registration {
    /// The projection key this guard holds.
    pub fn key(&self) -> &'static str {
        self.key
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let mut inner = self.inner.borrow_mut();
        let Some(index) = inner.registration_index(self.key) else {
            return;
        };
        if inner.registrations[index].refs > 1 {
            inner.registrations[index].refs -= 1;
            return;
        }
        inner.registrations.remove(index);
        for cells in inner.sessions.values_mut() {
            cells.cells.remove(self.key);
        }
    }
}

/// One change-feed subscription; dropping it unsubscribes.
#[must_use = "dropping the subscription immediately stops change notifications"]
pub struct ChangeSubscription {
    inner: Rc<RefCell<Inner>>,
    id: u64,
}

impl std::fmt::Debug for ChangeSubscription {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChangeSubscription")
            .field("id", &self.id)
            .finish()
    }
}

impl Drop for ChangeSubscription {
    fn drop(&mut self) {
        self.inner
            .borrow_mut()
            .listeners
            .retain(|(id, _)| *id != self.id);
    }
}

fn ensure_session_cells(inner: &mut Inner, session: &Session, inherited: SessionLogOffset) {
    let Inner {
        registrations,
        sessions,
        ..
    } = inner;
    let cells = sessions.entry(session.id().clone()).or_default();
    for entry in registrations.iter() {
        cells
            .cells
            .entry(entry.unit.key())
            .or_insert_with(|| Cell::new(entry.unit.init(session.header(), inherited)));
    }
}

fn sync_session(
    inner: &mut Inner,
    session: &Session,
    inherited: SessionLogOffset,
) -> Result<(), ProjectionError> {
    ensure_session_cells(inner, session, inherited);
    let target = last_cursor(session);
    for index in 0..inner.registrations.len() {
        let unit = Arc::clone(&inner.registrations[index].unit);
        let cell = cell_mut(inner, session.id(), unit.key())?;
        advance_cell(unit.as_ref(), cell, session, target)?;
    }
    Ok(())
}

/// Fold silently through `target`, invalidating the view baseline on a state
/// change without notifying.
fn advance_cell(
    unit: &dyn ErasedUnit,
    cell: &mut Cell,
    session: &Session,
    target: ProjectionCursor,
) -> Result<(), ProjectionError> {
    if cell.observed.position() >= target.position() {
        return Ok(());
    }
    let mut position = cell.observed.position() + 1;
    while position <= target.position() {
        let seq = SessionSeq::new(position as u64)
            .map_err(|error| ProjectionError::Offset(error.to_string()))?;
        let event = session.event(seq).ok_or(ProjectionError::MissingSeq {
            key: unit.key(),
            seq: position as u64,
        })?;
        let next = unit.apply(&cell.state, event, seq)?;
        if !Arc::ptr_eq(&next, &cell.state) {
            cell.views[0] = cell.views[1].take();
            cell.views[1] = None;
        }
        cell.state = next;
        cell.observed = ProjectionCursor::at(seq);
        position += 1;
    }
    Ok(())
}

fn cell_mut<'a>(
    inner: &'a mut Inner,
    session: &SessionId,
    key: &'static str,
) -> Result<&'a mut Cell, ProjectionError> {
    inner
        .sessions
        .get_mut(session)
        .and_then(|cells| cells.cells.get_mut(key))
        .ok_or(ProjectionError::MissingCell { key })
}

fn cell_ref<'a>(
    inner: &'a Inner,
    session: &SessionId,
    key: &'static str,
) -> Result<&'a Cell, ProjectionError> {
    inner
        .sessions
        .get(session)
        .and_then(|cells| cells.cells.get(key))
        .ok_or(ProjectionError::MissingCell { key })
}

fn wire_value(unit: &dyn ErasedUnit, state: &ErasedState) -> Result<Value, ProjectionError> {
    let wire = unit.wire(state)?;
    unit.wire_json(&wire)
}

fn last_cursor(session: &Session) -> ProjectionCursor {
    match i64::try_from(session.len()) {
        Ok(0) => ProjectionCursor::BEFORE_LOG,
        Ok(length) => ProjectionCursor::from_position(length - 1),
        Err(_) => ProjectionCursor::from_position(MAX_SAFE_INTEGER as i64),
    }
}
