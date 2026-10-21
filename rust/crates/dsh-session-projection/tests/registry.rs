//! Projection registry behavior: registration reference counting, eager drive,
//! per-session watermarks, the change feed's two-slot view identity, the
//! persisted-cache read ladder, and exact wire validation.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use dsh_session::{
    CoreEvent, Session, SessionEvent, SessionHeader, SessionId, SessionLogOffset, SessionMeta,
    SessionSeq, TurnEndReason, MAX_SAFE_INTEGER,
};
use dsh_session_projection::{
    ProjectionChange, ProjectionCheckpoint, ProjectionCheckpointRow, ProjectionCursor,
    ProjectionDefinition, ProjectionError, ProjectionRegistry, ProjectionSnapshot, ProjectionState,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

fn session(id: &str) -> Session {
    Session::create(SessionId::new(id).unwrap(), SessionMeta::default(), 0)
}

fn offset(value: u64) -> SessionLogOffset {
    SessionLogOffset::new(value).unwrap()
}

fn seq(value: u64) -> SessionSeq {
    SessionSeq::new(value).unwrap()
}

fn mark_event(marks: &[&str]) -> SessionEvent {
    let mut payload = Map::new();
    payload.insert("marks".to_string(), json!(marks));
    SessionEvent::extension_event("test/mark", payload)
}

fn append(session: &mut Session, event: SessionEvent) -> SessionSeq {
    session.append(event).unwrap()
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MarksState {
    marks: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MarksView {
    marks: Vec<String>,
}

impl ProjectionState for MarksState {}

#[derive(Deserialize)]
struct MarkPayload {
    marks: Vec<String>,
}

fn marks_unit() -> ProjectionDefinition<MarksState> {
    ProjectionDefinition::host_only(
        "test/marks",
        1,
        |_header: &SessionHeader, _inherited: SessionLogOffset| MarksState::default(),
        |_state, event, _seq| {
            let extension = event.extension()?;
            if extension.kind() != "test/mark" {
                return None;
            }
            let payload: MarkPayload = extension.decode().ok()?;
            Some(MarksState {
                marks: payload.marks,
            })
        },
    )
    .with_wire(
        |state| {
            Arc::new(MarksView {
                marks: state.marks.clone(),
            })
        },
        |_view: &MarksView| Ok(()),
    )
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
struct CountState(u64);

impl ProjectionState for CountState {}

fn count_unit(version: u64) -> ProjectionDefinition<CountState> {
    ProjectionDefinition::host_only(
        "test/count",
        version,
        |_header: &SessionHeader, _inherited: SessionLogOffset| CountState::default(),
        |state: &CountState, _event, _seq| Some(CountState(state.0 + 1)),
    )
}

fn count_wire_unit(version: u64) -> ProjectionDefinition<CountState> {
    count_unit(version).with_wire(|state: &CountState| Arc::new(state.0), |_view: &u64| Ok(()))
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
struct CutState(u64);

impl ProjectionState for CutState {}

fn cut_unit() -> ProjectionDefinition<CutState> {
    ProjectionDefinition::host_only(
        "test/cut",
        1,
        |_header: &SessionHeader, inherited: SessionLogOffset| CutState(inherited.get()),
        |_state: &CutState, _event, _seq| None,
    )
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StableViewState {
    revision: u64,
    value: Arc<Vec<String>>,
}

impl ProjectionState for StableViewState {}

fn stable_view_unit(calls: Rc<Cell<usize>>) -> ProjectionDefinition<StableViewState> {
    ProjectionDefinition::host_only(
        "test/stable-view",
        1,
        |_header: &SessionHeader, _inherited: SessionLogOffset| StableViewState {
            revision: 0,
            value: Arc::new(Vec::new()),
        },
        |state: &StableViewState, event, _seq| match event.core() {
            Some(CoreEvent::TurnStart { .. }) => Some(StableViewState {
                revision: state.revision + 1,
                value: Arc::clone(&state.value),
            }),
            _ => {
                let extension = event.extension()?;
                if extension.kind() != "test/mark" {
                    return None;
                }
                let payload: MarkPayload = extension.decode().ok()?;
                Some(StableViewState {
                    revision: state.revision + 1,
                    value: Arc::new(payload.marks),
                })
            }
        },
    )
    .with_wire(
        move |state: &StableViewState| {
            calls.set(calls.get() + 1);
            Arc::clone(&state.value)
        },
        |_view: &Vec<String>| Ok(()),
    )
}

fn bounded_marks_unit() -> ProjectionDefinition<MarksState> {
    ProjectionDefinition::host_only(
        "test/bounded",
        1,
        |_header: &SessionHeader, _inherited: SessionLogOffset| MarksState::default(),
        |_state, event, _seq| {
            let extension = event.extension()?;
            if extension.kind() != "test/mark" {
                return None;
            }
            let payload: MarkPayload = extension.decode().ok()?;
            Some(MarksState {
                marks: payload.marks,
            })
        },
    )
    .with_wire(
        |state| {
            Arc::new(MarksView {
                marks: state.marks.clone(),
            })
        },
        |view: &MarksView| {
            if view.marks.len() > 1 {
                Err("a preview carries at most one mark".to_string())
            } else {
                Ok(())
            }
        },
    )
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
struct BoundedState(u64);

impl ProjectionState for BoundedState {
    fn validate(&self) -> Result<(), String> {
        if self.0 > 1 {
            Err("bounded state overflowed".to_string())
        } else {
            Ok(())
        }
    }
}

fn bounded_state_unit() -> ProjectionDefinition<BoundedState> {
    ProjectionDefinition::host_only(
        "test/bounded-state",
        1,
        |_header: &SessionHeader, _inherited: SessionLogOffset| BoundedState::default(),
        |state: &BoundedState, _event, _seq| Some(BoundedState(state.0 + 1)),
    )
}

fn row(version: u64, cursor: ProjectionCursor, value: Value) -> ProjectionCheckpointRow {
    ProjectionCheckpointRow {
        ver: version,
        seq: cursor,
        val: value,
    }
}

fn checkpoint<const N: usize>(rows: [(&str, ProjectionCheckpointRow); N]) -> ProjectionCheckpoint {
    rows.into_iter()
        .map(|(key, row)| (key.to_string(), row))
        .collect()
}

fn record(changes: &Rc<RefCell<Vec<ProjectionChange>>>) -> impl Fn(&ProjectionChange) + 'static {
    let changes = Rc::clone(changes);
    move |change| changes.borrow_mut().push(change.clone())
}

#[test]
fn one_registration_serves_until_the_last_guard_is_released() {
    let registry = ProjectionRegistry::new();
    let mut session = session("shared-guards");
    let first = registry.register(marks_unit()).unwrap();
    let second = registry.register(marks_unit()).unwrap();
    assert_eq!(first.key(), "test/marks");
    append(&mut session, mark_event(&["kept"]));

    let snapshot = registry.snapshot(&session, offset(0)).unwrap();
    assert_eq!(
        snapshot.values.get("test/marks"),
        Some(&json!({"marks": ["kept"]}))
    );

    drop(first);
    let snapshot = registry.snapshot(&session, offset(0)).unwrap();
    assert_eq!(
        snapshot.values.get("test/marks"),
        Some(&json!({"marks": ["kept"]}))
    );

    drop(second);
    assert!(registry
        .snapshot(&session, offset(0))
        .unwrap()
        .values
        .is_empty());
    assert!(registry
        .state_of::<MarksState>(&session, offset(0), "test/marks")
        .unwrap()
        .is_none());
}

#[test]
fn a_second_registration_does_not_replace_the_first_definition() {
    let registry = ProjectionRegistry::new();
    let _first = registry.register(marks_unit()).unwrap();
    let _second = registry
        .register(ProjectionDefinition::host_only(
            "test/marks",
            1,
            |_header: &SessionHeader, _inherited: SessionLogOffset| MarksState {
                marks: vec!["second-init".to_string()],
            },
            |_state: &MarksState, _event, _seq| {
                Some(MarksState {
                    marks: vec!["second-apply".to_string()],
                })
            },
        ))
        .unwrap();
    let session = session("first-definition-wins");

    assert_eq!(
        registry
            .snapshot(&session, offset(0))
            .unwrap()
            .values
            .get("test/marks"),
        Some(&json!({"marks": []}))
    );
}

#[test]
fn registration_rejects_an_empty_key() {
    let registry = ProjectionRegistry::new();
    let error = registry
        .register(ProjectionDefinition::host_only(
            "",
            1,
            |_header: &SessionHeader, _inherited: SessionLogOffset| CountState::default(),
            |state: &CountState, _event, _seq| Some(state.clone()),
        ))
        .unwrap_err();
    assert!(matches!(error, ProjectionError::EmptyKey));
}

#[test]
fn registration_refuses_to_share_a_key_across_a_state_version_change() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let error = registry
        .register(ProjectionDefinition::host_only(
            "test/marks",
            9,
            |_header: &SessionHeader, _inherited: SessionLogOffset| MarksState::default(),
            |_state, _event, _seq| None,
        ))
        .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::StateVersionConflict {
            key: "test/marks",
            existing: 1,
            requested: 9
        }
    ));
}

#[test]
fn registration_rejects_a_state_version_beyond_the_safe_integer_bound() {
    let registry = ProjectionRegistry::new();
    let error = registry
        .register(count_unit(MAX_SAFE_INTEGER + 1))
        .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::InvalidStateVersion {
            key: "test/count",
            version,
            max
        } if version == MAX_SAFE_INTEGER + 1 && max == MAX_SAFE_INTEGER
    ));
}

#[test]
fn an_empty_log_serves_init_state_at_the_before_log_cursor() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let session = session("empty-log");

    let snapshot = registry.snapshot(&session, offset(0)).unwrap();
    assert_eq!(snapshot.as_of_seq, ProjectionCursor::BEFORE_LOG);
    assert_eq!(
        snapshot.values.get("test/marks"),
        Some(&json!({"marks": []}))
    );

    let rows = registry.checkpoint(&session, offset(0)).unwrap();
    let row = rows.get("test/marks").unwrap();
    assert_eq!(row.ver, 1);
    assert_eq!(row.seq, ProjectionCursor::BEFORE_LOG);
    assert_eq!(row.val, json!({"marks": []}));
}

#[test]
fn a_snapshot_materializes_every_unit_and_excludes_host_only_state() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_unit(7)).unwrap();
    let mut session = session("snapshot-cut");
    append(&mut session, mark_event(&["a", "b"]));

    let snapshot = registry.snapshot(&session, offset(0)).unwrap();
    assert_eq!(snapshot.as_of_seq, ProjectionCursor::at(seq(0)));
    assert_eq!(snapshot.values.len(), 1);
    assert_eq!(
        snapshot.values.get("test/marks"),
        Some(&json!({"marks": ["a", "b"]}))
    );
    assert_eq!(
        registry
            .state_of::<CountState>(&session, offset(0), "test/count")
            .unwrap()
            .unwrap()
            .0,
        1
    );

    let selected = registry
        .snapshot_selected(&session, offset(0), Some(&["test/marks", "test/missing"]))
        .unwrap();
    assert_eq!(selected.values.len(), 1);
    assert_eq!(
        selected.values.get("test/marks"),
        Some(&json!({"marks": ["a", "b"]}))
    );

    let none = registry
        .snapshot_selected(&session, offset(0), Some(&["test/missing"]))
        .unwrap();
    assert!(none.values.is_empty());
}

#[test]
fn a_snapshot_can_select_keys_without_materializing_only_those_units() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_unit(1)).unwrap();
    let mut session = session("selected-materialization");
    append(&mut session, mark_event(&["a"]));

    let selected = registry
        .snapshot_selected(&session, offset(0), Some(&["test/marks"]))
        .unwrap();
    assert!(selected.values.contains_key("test/marks"));
    assert_eq!(
        registry
            .state_of::<CountState>(&session, offset(0), "test/count")
            .unwrap()
            .unwrap()
            .0,
        1
    );
}

#[test]
fn drive_notifies_once_with_the_validated_view_and_the_causing_seq() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let mut session = session("drive-notify");
    append(&mut session, SessionEvent::turn_start(1));
    let mark = append(&mut session, mark_event(&["a"]));
    let changes = Rc::new(RefCell::new(Vec::new()));
    let _subscription = registry.on_changed(record(&changes));

    registry.drive_event(&session, offset(0), seq(0)).unwrap();
    assert!(
        changes.borrow().is_empty(),
        "an unmatched event keeps the state reference"
    );

    registry.drive_event(&session, offset(0), mark).unwrap();
    let changes = changes.borrow();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].key, "test/marks");
    assert_eq!(changes[0].value, json!({"marks": ["a"]}));
    assert_eq!(changes[0].seq, mark);
    assert_eq!(changes[0].session_id, *session.id());
}

#[test]
fn drive_folds_missing_prefix_events_before_the_requested_seq() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let mut session = session("late-drive");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, mark_event(&["one"]));
    let tail = append(&mut session, mark_event(&["one", "two"]));
    let changes = Rc::new(RefCell::new(Vec::new()));
    let _subscription = registry.on_changed(record(&changes));

    registry.drive_event(&session, offset(0), tail).unwrap();

    let changes = changes.borrow();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].value, json!({"marks": ["one", "two"]}));
    assert_eq!(changes[0].seq, tail);
    assert_eq!(
        registry
            .snapshot(&session, offset(0))
            .unwrap()
            .values
            .get("test/marks"),
        Some(&json!({"marks": ["one", "two"]}))
    );
}

#[test]
fn an_unmatched_event_keeps_the_state_allocation_and_stays_quiet() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let mut session = session("identity-gate");
    let mark = append(&mut session, mark_event(&["one"]));
    let changes = Rc::new(RefCell::new(Vec::new()));
    let _subscription = registry.on_changed(record(&changes));

    registry.drive_event(&session, offset(0), mark).unwrap();
    let before = registry
        .state_of::<MarksState>(&session, offset(0), "test/marks")
        .unwrap()
        .unwrap();
    append(&mut session, SessionEvent::turn_start(1));
    registry.drive_event(&session, offset(0), seq(1)).unwrap();
    let after = registry
        .state_of::<MarksState>(&session, offset(0), "test/marks")
        .unwrap()
        .unwrap();

    assert!(Arc::ptr_eq(&before, &after));
    assert_eq!(
        changes.borrow().len(),
        1,
        "only the marking event changed the view"
    );
}

#[test]
fn views_are_only_computed_while_a_listener_is_subscribed() {
    let registry = ProjectionRegistry::new();
    let calls = Rc::new(Cell::new(0));
    let _stable = registry
        .register(stable_view_unit(Rc::clone(&calls)))
        .unwrap();
    let mut session = session("lazy-views");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, SessionEvent::turn_start(2));

    registry.drive_event(&session, offset(0), seq(0)).unwrap();
    registry.drive_event(&session, offset(0), seq(1)).unwrap();
    assert_eq!(calls.get(), 0);

    let subscription = registry.on_changed(|_change| {});
    append(&mut session, SessionEvent::turn_start(3));
    registry.drive_event(&session, offset(0), seq(2)).unwrap();
    assert_eq!(calls.get(), 1);

    drop(subscription);
    append(&mut session, SessionEvent::turn_start(4));
    registry.drive_event(&session, offset(0), seq(3)).unwrap();
    assert_eq!(calls.get(), 1);

    registry.snapshot(&session, offset(0)).unwrap();
    assert_eq!(calls.get(), 2);
}

#[test]
fn a_stable_view_allocation_suppresses_internal_only_changes() {
    let registry = ProjectionRegistry::new();
    let calls = Rc::new(Cell::new(0));
    let _stable = registry
        .register(stable_view_unit(Rc::clone(&calls)))
        .unwrap();
    let mut session = session("stable-view");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, SessionEvent::turn_start(2));
    append(&mut session, mark_event(&["changed"]));
    let seen = Rc::new(RefCell::new(Vec::new()));
    let _subscription = registry.on_changed({
        let seen = Rc::clone(&seen);
        move |change| seen.borrow_mut().push(change.value.clone())
    });

    registry.drive_event(&session, offset(0), seq(0)).unwrap();
    registry.drive_event(&session, offset(0), seq(1)).unwrap();
    registry.drive_event(&session, offset(0), seq(2)).unwrap();

    assert_eq!(*seen.borrow(), vec![json!([]), json!(["changed"])]);
    assert_eq!(
        calls.get(),
        3,
        "each changed state computes a view, but identity gates publication"
    );
}

#[test]
fn a_listener_gap_clears_the_comparison_baseline_and_republishes() {
    let registry = ProjectionRegistry::new();
    let calls = Rc::new(Cell::new(0));
    let _stable = registry
        .register(stable_view_unit(Rc::clone(&calls)))
        .unwrap();
    let mut session = session("listener-gap");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, SessionEvent::turn_start(2));
    append(&mut session, SessionEvent::turn_start(3));
    let first = Rc::new(RefCell::new(Vec::new()));
    let first_subscription = registry.on_changed({
        let first = Rc::clone(&first);
        move |change| first.borrow_mut().push(change.value.clone())
    });

    registry.drive_event(&session, offset(0), seq(0)).unwrap();
    drop(first_subscription);
    registry.drive_event(&session, offset(0), seq(1)).unwrap();
    assert_eq!(calls.get(), 1);

    let resumed = Rc::new(RefCell::new(Vec::new()));
    let _resumed_subscription = registry.on_changed({
        let resumed = Rc::clone(&resumed);
        move |change| resumed.borrow_mut().push(change.value.clone())
    });
    registry.drive_event(&session, offset(0), seq(2)).unwrap();

    assert_eq!(*first.borrow(), vec![json!([])]);
    assert_eq!(
        *resumed.borrow(),
        vec![json!([])],
        "the first view after a gap has no baseline"
    );
    assert_eq!(calls.get(), 2);
}

#[test]
fn a_listener_can_read_the_registry_at_the_consistent_cut() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_wire_unit(1)).unwrap();
    let mut session = session("reentrant-read");
    append(&mut session, mark_event(&["one"]));
    let observed = Rc::new(RefCell::new(Vec::new()));
    let reader = registry.clone();
    let read_session = session.clone();
    let _subscription = registry.on_changed({
        let observed = Rc::clone(&observed);
        move |change| {
            if change.key == "test/marks" {
                let count = reader
                    .state_of::<CountState>(&read_session, offset(0), "test/count")
                    .unwrap()
                    .unwrap();
                observed.borrow_mut().push(count.0);
            }
        }
    });

    registry.drive_event(&session, offset(0), seq(0)).unwrap();

    assert_eq!(
        *observed.borrow(),
        vec![1],
        "every unit advanced before listeners run"
    );
}

#[test]
fn host_only_units_advance_without_publishing_to_the_change_feed() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_unit(1)).unwrap();
    let mut session = session("host-only");
    append(&mut session, SessionEvent::turn_start(1));
    let changes = Rc::new(RefCell::new(Vec::new()));
    let _subscription = registry.on_changed(record(&changes));

    registry.drive_event(&session, offset(0), seq(0)).unwrap();

    assert!(changes.borrow().is_empty());
    assert_eq!(
        registry
            .state_of::<CountState>(&session, offset(0), "test/count")
            .unwrap()
            .unwrap()
            .0,
        1
    );
}

#[test]
fn drive_rejects_a_sequence_outside_the_session_log() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let mut session = session("missing-event");
    append(&mut session, SessionEvent::turn_start(1));

    let error = registry
        .drive_event(&session, offset(0), seq(4))
        .unwrap_err();
    assert!(matches!(error, ProjectionError::MissingEvent { seq: 4 }));
}

#[test]
fn checkpoints_are_detached_cuts_with_the_registered_versions() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_unit(7)).unwrap();
    let mut session = session("checkpoint");
    let mark = append(&mut session, mark_event(&["a"]));

    let rows = registry.checkpoint(&session, offset(0)).unwrap();
    let marks = rows.get("test/marks").unwrap();
    assert_eq!(
        *marks,
        row(1, ProjectionCursor::at(mark), json!({"marks": ["a"]}))
    );
    assert_eq!(
        *rows.get("test/count").unwrap(),
        row(7, ProjectionCursor::at(mark), json!(1))
    );

    let mut rows = rows;
    rows.get_mut("test/marks").unwrap().val["marks"] = json!(["INJECTED"]);
    assert_eq!(
        registry
            .snapshot(&session, offset(0))
            .unwrap()
            .values
            .get("test/marks"),
        Some(&json!({"marks": ["a"]}))
    );
}

#[test]
fn checkpoint_rows_and_snapshots_serialize_in_typescript_field_order() {
    let sample = row(1, ProjectionCursor::at(seq(3)), json!({"marks": ["a"]}));
    assert_eq!(
        serde_json::to_string(&sample).unwrap(),
        r#"{"ver":1,"seq":3,"val":{"marks":["a"]}}"#
    );
    assert_eq!(
        serde_json::to_string(&row(1, ProjectionCursor::BEFORE_LOG, json!({"marks": []}))).unwrap(),
        r#"{"ver":1,"seq":-1,"val":{"marks":[]}}"#
    );
    assert!(
        serde_json::from_str::<ProjectionCheckpointRow>(
            r#"{"ver":1,"seq":0,"val":null,"extra":true}"#
        )
        .is_err(),
        "unknown row fields must not be dropped"
    );

    let mut values = Map::new();
    values.insert("test/marks".to_string(), json!({"marks": ["a"]}));
    let snapshot = ProjectionSnapshot {
        as_of_seq: ProjectionCursor::at(seq(3)),
        values,
    };
    let text = serde_json::to_string(&snapshot).unwrap();
    assert_eq!(
        text,
        r#"{"asOfSeq":3,"values":{"test/marks":{"marks":["a"]}}}"#
    );
    assert_eq!(
        serde_json::from_str::<ProjectionSnapshot>(&text).unwrap(),
        snapshot
    );
}

#[test]
fn cursors_round_trip_the_before_log_sentinel_and_reject_unsafe_values() {
    assert_eq!(
        serde_json::to_string(&ProjectionCursor::BEFORE_LOG).unwrap(),
        "-1"
    );
    assert_eq!(
        serde_json::to_string(&ProjectionCursor::at(seq(7))).unwrap(),
        "7"
    );
    assert_eq!(
        serde_json::from_str::<ProjectionCursor>("-1").unwrap(),
        ProjectionCursor::BEFORE_LOG
    );
    assert!(serde_json::from_str::<ProjectionCursor>("-2").is_err());
    assert!(serde_json::from_str::<ProjectionCursor>(&(MAX_SAFE_INTEGER + 1).to_string()).is_err());
    assert!(serde_json::from_str::<ProjectionCursor>("1.5").is_err());
}

#[test]
fn reading_a_key_as_the_wrong_state_type_fails_loudly() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let mut session = session("state-mismatch");
    append(&mut session, SessionEvent::turn_start(1));

    let error = registry
        .state_of::<CountState>(&session, offset(0), "test/marks")
        .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::StateTypeMismatch { key: "test/marks" }
    ));
    assert!(registry
        .state_of::<MarksState>(&session, offset(0), "test/missing")
        .unwrap()
        .is_none());
}

#[test]
fn a_view_that_violates_its_schema_fails_loud_and_publishes_nothing() {
    let registry = ProjectionRegistry::new();
    let _bounded = registry.register(bounded_marks_unit()).unwrap();
    let mut session = session("invalid-view");
    append(&mut session, mark_event(&["a", "b"]));
    let changes = Rc::new(RefCell::new(Vec::new()));
    let _subscription = registry.on_changed(record(&changes));

    let error = registry
        .drive_event(&session, offset(0), seq(0))
        .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::InvalidWire {
            key: "test/bounded",
            ..
        }
    ));
    assert!(changes.borrow().is_empty());
    let error = registry.snapshot(&session, offset(0)).unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::InvalidWire {
            key: "test/bounded",
            ..
        }
    ));
}

#[test]
fn checkpoint_and_restore_reject_state_that_violates_its_validation() {
    let registry = ProjectionRegistry::new();
    let _bounded = registry.register(bounded_state_unit()).unwrap();
    let mut session = session("invalid-state");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, SessionEvent::turn_start(2));

    let error = registry.checkpoint(&session, offset(0)).unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::InvalidState {
            key: "test/bounded-state",
            ..
        }
    ));

    let rows = checkpoint([(
        "test/bounded-state",
        row(1, ProjectionCursor::BEFORE_LOG, json!(2)),
    )]);
    let error = registry
        .restore(&rows, &[], offset(0), session.header(), offset(0))
        .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::InvalidState {
            key: "test/bounded-state",
            ..
        }
    ));
}

#[test]
fn restore_floor_anchors_one_below_the_lowest_usable_watermark() {
    let registry = ProjectionRegistry::new();
    assert_eq!(
        registry
            .restore_floor(&ProjectionCheckpoint::new())
            .unwrap(),
        None
    );

    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_unit(1)).unwrap();
    assert_eq!(
        registry
            .restore_floor(&ProjectionCheckpoint::new())
            .unwrap(),
        Some(offset(0))
    );

    let rows = checkpoint([
        (
            "test/marks",
            row(1, ProjectionCursor::at(seq(10)), json!({"marks": []})),
        ),
        ("test/count", row(1, ProjectionCursor::at(seq(5)), json!(6))),
    ]);
    assert_eq!(registry.restore_floor(&rows).unwrap(), Some(offset(5)));

    let mismatched = checkpoint([
        (
            "test/marks",
            row(2, ProjectionCursor::at(seq(10)), json!({"marks": []})),
        ),
        ("test/count", row(1, ProjectionCursor::at(seq(5)), json!(6))),
    ]);
    assert_eq!(
        registry.restore_floor(&mismatched).unwrap(),
        Some(offset(0))
    );

    let fresh = checkpoint([
        (
            "test/marks",
            row(1, ProjectionCursor::BEFORE_LOG, json!({"marks": []})),
        ),
        ("test/count", row(1, ProjectionCursor::BEFORE_LOG, json!(0))),
    ]);
    assert_eq!(registry.restore_floor(&fresh).unwrap(), Some(offset(0)));
}

#[test]
fn view_checkpoint_serves_only_matching_rows_and_client_visible_units() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_unit(1)).unwrap();
    let rows = checkpoint([
        (
            "test/marks",
            row(
                1,
                ProjectionCursor::at(seq(4)),
                json!({"marks": ["stored"]}),
            ),
        ),
        ("test/count", row(1, ProjectionCursor::at(seq(4)), json!(5))),
    ]);

    let values = registry.view_checkpoint(&rows, None).unwrap();
    assert_eq!(values.len(), 1);
    assert_eq!(
        values.get("test/marks"),
        Some(&json!({"marks": ["stored"]}))
    );
    assert!(registry
        .view_checkpoint(&rows, Some(&["test/count"]))
        .unwrap()
        .is_empty());
    assert!(registry
        .view_checkpoint(&rows, Some(&["test/missing"]))
        .unwrap()
        .is_empty());

    let mismatched = checkpoint([(
        "test/marks",
        row(
            2,
            ProjectionCursor::at(seq(4)),
            json!({"marks": ["stored"]}),
        ),
    )]);
    assert!(registry
        .view_checkpoint(&mismatched, None)
        .unwrap()
        .is_empty());

    let malformed = checkpoint([(
        "test/marks",
        row(
            1,
            ProjectionCursor::at(seq(4)),
            json!({"marks": "not-an-array"}),
        ),
    )]);
    assert!(registry
        .view_checkpoint(&malformed, None)
        .unwrap()
        .is_empty());
}

#[test]
fn restore_folds_only_past_each_usable_watermark() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_unit(1)).unwrap();
    let rows = checkpoint([
        (
            "test/marks",
            row(1, ProjectionCursor::at(seq(4)), json!({"marks": ["done"]})),
        ),
        ("test/count", row(1, ProjectionCursor::at(seq(2)), json!(3))),
    ]);
    let header = session("restore-tail").header().clone();
    let events = vec![SessionEvent::turn_start(2), mark_event(&["new"])];

    let restored = registry
        .restore(&rows, &events, offset(3), &header, offset(0))
        .unwrap();

    assert_eq!(restored.snapshot.as_of_seq, ProjectionCursor::at(seq(4)));
    assert_eq!(
        restored.snapshot.values.get("test/marks"),
        Some(&json!({"marks": ["done"]})),
        "a row at the tail end re-applies nothing"
    );
    assert!(!restored.snapshot.values.contains_key("test/count"));
    assert_eq!(
        *restored.checkpoint.get("test/count").unwrap(),
        row(1, ProjectionCursor::at(seq(4)), json!(5)),
        "the tail folds exactly the events past the checkpoint watermark"
    );
    assert_eq!(
        *restored.checkpoint.get("test/marks").unwrap(),
        row(1, ProjectionCursor::at(seq(4)), json!({"marks": ["done"]}))
    );
}

#[test]
fn restore_serves_an_exact_empty_tail_cut() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_unit(1)).unwrap();
    let rows = checkpoint([
        (
            "test/marks",
            row(1, ProjectionCursor::at(seq(4)), json!({"marks": ["done"]})),
        ),
        ("test/count", row(1, ProjectionCursor::at(seq(4)), json!(5))),
    ]);
    let header = session("empty-tail").header().clone();

    let restored = registry
        .restore(&rows, &[], offset(5), &header, offset(0))
        .unwrap();

    assert_eq!(restored.snapshot.as_of_seq, ProjectionCursor::at(seq(4)));
    assert_eq!(
        *restored.checkpoint.get("test/count").unwrap(),
        rows["test/count"]
    );
}

#[test]
fn restore_refuses_a_tail_read_without_a_usable_row() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_unit(1)).unwrap();
    let header = session("missing-row").header().clone();
    let rows = checkpoint([(
        "test/marks",
        row(1, ProjectionCursor::at(seq(2)), json!({"marks": ["old"]})),
    )]);

    let error = registry
        .restore(&rows, &[], offset(3), &header, offset(0))
        .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::UnrestorableCheckpoint {
            key: "test/count",
            base_seq: 3
        }
    ));

    let error = registry
        .restore(
            &ProjectionCheckpoint::new(),
            &[],
            offset(3),
            &header,
            offset(0),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::UnrestorableCheckpoint {
            key: "test/marks",
            base_seq: 3
        }
    ));
}

#[test]
fn restore_refolds_from_init_on_a_full_read_when_a_row_is_unusable() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_unit(1)).unwrap();
    let mut session = session("full-refold");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, mark_event(&["old"]));
    append(&mut session, mark_event(&["old", "2"]));
    append(
        &mut session,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );
    append(&mut session, mark_event(&["new"]));
    let rows = checkpoint([
        (
            "test/marks",
            row(
                1,
                ProjectionCursor::at(seq(2)),
                json!({"marks": ["old", "2"]}),
            ),
        ),
        (
            "test/count",
            row(99, ProjectionCursor::at(seq(2)), json!(3)),
        ),
    ]);

    let restored = registry
        .restore(
            &rows,
            session.events(),
            offset(0),
            session.header(),
            offset(0),
        )
        .unwrap();

    assert_eq!(restored.snapshot.as_of_seq, ProjectionCursor::at(seq(4)));
    assert_eq!(
        restored.snapshot.values.get("test/marks"),
        Some(&json!({"marks": ["new"]}))
    );
    assert_eq!(
        *restored.checkpoint.get("test/count").unwrap(),
        row(1, ProjectionCursor::at(seq(4)), json!(5))
    );
    assert_eq!(
        *restored.checkpoint.get("test/marks").unwrap(),
        row(1, ProjectionCursor::at(seq(4)), json!({"marks": ["new"]}))
    );
}

#[test]
fn restore_refuses_a_row_claiming_events_past_the_supplied_log_end() {
    let registry = ProjectionRegistry::new();
    let _count = registry.register(count_unit(1)).unwrap();
    let header = session("shrunk-log").header().clone();
    let rows = checkpoint([(
        "test/count",
        row(1, ProjectionCursor::at(seq(9)), json!(10)),
    )]);

    let error = registry
        .restore(&rows, &[], offset(9), &header, offset(0))
        .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::UnrestorableCheckpoint {
            key: "test/count",
            base_seq: 9
        }
    ));

    let events = vec![
        SessionEvent::turn_start(1),
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    ];
    let restored = registry
        .restore(&rows, &events, offset(0), &header, offset(0))
        .unwrap();
    assert_eq!(restored.snapshot.as_of_seq, ProjectionCursor::at(seq(1)));
    assert_eq!(
        *restored.checkpoint.get("test/count").unwrap(),
        row(1, ProjectionCursor::at(seq(1)), json!(2))
    );
}

#[test]
fn restore_rejects_a_version_matching_row_whose_state_does_not_parse() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let header = session("drifted-row").header().clone();
    let rows = checkpoint([(
        "test/marks",
        row(
            1,
            ProjectionCursor::at(seq(2)),
            json!({"marks": "not-an-array"}),
        ),
    )]);

    let error = registry
        .restore(&rows, &[], offset(3), &header, offset(0))
        .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::InvalidState {
            key: "test/marks",
            ..
        }
    ));
}

#[test]
fn inherited_prefix_length_reaches_live_restore_and_hydrate_initialization() {
    let registry = ProjectionRegistry::new();
    let _cut = registry.register(cut_unit()).unwrap();
    let mut live = session("cut-live");
    append(&mut live, SessionEvent::turn_start(1));
    append(
        &mut live,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );
    let inherited = offset(2);

    assert_eq!(
        registry
            .state_of::<CutState>(&live, inherited, "test/cut")
            .unwrap()
            .unwrap()
            .0,
        2
    );

    let restored = registry
        .restore(
            &ProjectionCheckpoint::new(),
            live.events(),
            offset(0),
            live.header(),
            inherited,
        )
        .unwrap();
    assert_eq!(restored.snapshot.as_of_seq, ProjectionCursor::at(seq(1)));
    assert_eq!(restored.checkpoint.get("test/cut").unwrap().val, json!(2));

    let mut prepared = session("cut-prepared");
    append(&mut prepared, SessionEvent::turn_start(1));
    append(
        &mut prepared,
        SessionEvent::turn_end(1, TurnEndReason::Completed),
    );
    let snapshot = registry
        .hydrate(
            &prepared,
            &ProjectionCheckpoint::new(),
            prepared.events(),
            offset(0),
            inherited,
        )
        .unwrap();
    assert_eq!(snapshot.as_of_seq, ProjectionCursor::at(seq(1)));
    assert_eq!(
        registry
            .state_of::<CutState>(&prepared, inherited, "test/cut")
            .unwrap()
            .unwrap()
            .0,
        2
    );
}

#[test]
fn hydrate_installs_a_cut_and_folds_the_owned_suffix_exactly_once() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let _count = registry.register(count_unit(1)).unwrap();
    let mut session = session("hydrate-suffix");
    append(&mut session, SessionEvent::turn_start(1));
    append(&mut session, mark_event(&["a"]));
    append(&mut session, mark_event(&["a", "b"]));
    let rows = checkpoint([
        (
            "test/marks",
            row(1, ProjectionCursor::at(seq(1)), json!({"marks": ["a"]})),
        ),
        ("test/count", row(1, ProjectionCursor::at(seq(1)), json!(2))),
    ]);

    let restored = registry
        .hydrate(
            &session,
            &rows,
            &session.events()[..2],
            offset(0),
            offset(0),
        )
        .unwrap();
    assert_eq!(restored.as_of_seq, ProjectionCursor::at(seq(1)));
    assert_eq!(
        restored.values.get("test/marks"),
        Some(&json!({"marks": ["a"]})),
        "the returned snapshot is the supplied cut, not the live log end"
    );

    let marks = registry
        .state_of::<MarksState>(&session, offset(0), "test/marks")
        .unwrap()
        .unwrap();
    assert_eq!(marks.marks, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(
        registry
            .state_of::<CountState>(&session, offset(0), "test/count")
            .unwrap()
            .unwrap()
            .0,
        3
    );
    assert_eq!(
        registry
            .state_of::<CountState>(&session, offset(0), "test/count")
            .unwrap()
            .unwrap()
            .0,
        3,
        "a second read must not replay the suffix"
    );
}

#[test]
fn hydrate_does_not_regress_a_cell_already_past_the_restored_cut() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let mut session = session("hydrate-ahead");
    append(&mut session, mark_event(&["a"]));
    append(&mut session, mark_event(&["a", "b"]));
    registry.snapshot(&session, offset(0)).unwrap();
    let rows = checkpoint([(
        "test/marks",
        row(1, ProjectionCursor::BEFORE_LOG, json!({"marks": []})),
    )]);

    registry
        .hydrate(
            &session,
            &rows,
            &session.events()[..1],
            offset(0),
            offset(0),
        )
        .unwrap();

    let marks = registry
        .state_of::<MarksState>(&session, offset(0), "test/marks")
        .unwrap()
        .unwrap();
    assert_eq!(marks.marks, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn cached_snapshot_reports_the_lowest_materialized_watermark() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let mut session = session("cached-cut");
    append(&mut session, mark_event(&["a"]));
    append(&mut session, mark_event(&["a", "b"]));
    registry.snapshot(&session, offset(0)).unwrap();
    let _count = registry.register(count_wire_unit(1)).unwrap();

    let cached = registry.cached_snapshot(&session, None).unwrap().unwrap();
    assert_eq!(cached.as_of_seq, ProjectionCursor::at(seq(1)));
    assert_eq!(
        cached.values.len(),
        1,
        "a unit without a materialized cell is omitted"
    );

    let rows = checkpoint([
        (
            "test/marks",
            row(1, ProjectionCursor::at(seq(0)), json!({"marks": ["stale"]})),
        ),
        ("test/count", row(1, ProjectionCursor::at(seq(0)), json!(1))),
    ]);
    registry
        .hydrate(
            &session,
            &rows,
            &session.events()[..1],
            offset(0),
            offset(0),
        )
        .unwrap();

    let cached = registry.cached_snapshot(&session, None).unwrap().unwrap();
    assert_eq!(
        cached.as_of_seq,
        ProjectionCursor::at(seq(0)),
        "the cut is the lowest cell"
    );
    assert_eq!(
        cached.values.get("test/marks"),
        Some(&json!({"marks": ["a", "b"]})),
        "a cell ahead of the restored cut keeps its value"
    );
    assert_eq!(cached.values.get("test/count"), Some(&json!(1)));

    registry.snapshot(&session, offset(0)).unwrap();
    let cached = registry
        .cached_snapshot(&session, Some(&["test/count"]))
        .unwrap()
        .unwrap();
    assert_eq!(cached.as_of_seq, ProjectionCursor::at(seq(1)));
    assert_eq!(cached.values.get("test/count"), Some(&json!(2)));
    assert_eq!(cached.values.len(), 1);
}

#[test]
fn registered_units_drive_independently_per_session_id() {
    let registry = ProjectionRegistry::new();
    let _marks = registry.register(marks_unit()).unwrap();
    let mut first = session("session-one");
    let mut second = session("session-two");
    append(&mut first, mark_event(&["one"]));
    append(&mut second, mark_event(&["two"]));

    assert_eq!(
        registry
            .snapshot(&first, offset(0))
            .unwrap()
            .values
            .get("test/marks"),
        Some(&json!({"marks": ["one"]}))
    );
    assert_eq!(
        registry
            .snapshot(&second, offset(0))
            .unwrap()
            .values
            .get("test/marks"),
        Some(&json!({"marks": ["two"]}))
    );

    let handle = first.clone();
    assert_eq!(
        registry
            .state_of::<MarksState>(&handle, offset(0), "test/marks")
            .unwrap()
            .unwrap()
            .marks,
        vec!["one".to_string()],
        "a clone with the same id shares the session cell"
    );
}
