//! Sub-call identity and the ordered commit lane.
//!
//! The behaviour comes from the per-run scheduler in
//! `packages/core/tools/src/ptc.ts`; the scenarios mirror
//! `packages/core/tools/tests/ptc.spec.ts` ("overlaps concurrency-safe calls",
//! "an exclusive call bars overlap", abandonment after a run ends), with the
//! async driver replaced by explicit `poll`/`settle`/`committed` calls.

use dsh_llm::ToolCallId;
use dsh_ptc::{
    run_over_message, Classification, DispatchLane, LaneStep, RunOverPhase, SubCallSequence,
};

fn id(value: &str) -> ToolCallId {
    ToolCallId::new(value)
}

fn start(value: &str, classification: Classification) -> LaneStep {
    LaneStep::Start {
        id: id(value),
        classification,
    }
}

fn commit(value: &str, classification: Classification) -> LaneStep {
    LaneStep::Commit {
        id: id(value),
        classification,
    }
}

#[test]
fn sub_call_ids_are_minted_in_submission_order() {
    let parent = id("call-1");
    let mut sequence = SubCallSequence::new();
    assert_eq!(sequence.mint(&parent).as_str(), "call-1:ptc:1");
    assert_eq!(sequence.mint(&parent).as_str(), "call-1:ptc:2");
    assert_eq!(sequence.mint(&parent).as_str(), "call-1:ptc:3");
    assert_eq!(sequence.issued(), 3);

    // A second run starts over: the id is scoped to its enclosing call.
    let mut second = SubCallSequence::new();
    assert_eq!(second.mint(&parent).as_str(), "call-1:ptc:1");
}

#[test]
fn settle_records_commit_in_submission_order_not_settle_order() {
    let mut lane = DispatchLane::new(4);
    for name in ["a", "b", "c"] {
        lane.submit(id(name), "read", Classification::Parallel);
    }
    assert_eq!(lane.poll(), start("a", Classification::Parallel));
    assert_eq!(lane.poll(), start("b", Classification::Parallel));
    assert_eq!(lane.poll(), start("c", Classification::Parallel));
    assert_eq!(lane.poll(), LaneStep::Wait);
    assert_eq!(lane.in_flight(), 3);

    // The middle and last calls finish first.
    assert!(lane.settle(&id("b")));
    assert!(lane.settle(&id("c")));
    assert_eq!(lane.poll(), LaneStep::Wait, "the head has not settled yet");

    assert!(lane.settle(&id("a")));
    assert_eq!(lane.poll(), commit("a", Classification::Parallel));
    assert!(lane.committed(&id("a")));
    assert_eq!(lane.poll(), commit("b", Classification::Parallel));
    assert!(lane.committed(&id("b")));
    assert_eq!(lane.poll(), commit("c", Classification::Parallel));
    assert!(lane.committed(&id("c")));
    assert_eq!(lane.poll(), LaneStep::Quiescent);
}

#[test]
fn a_parallel_group_overlaps_up_to_its_width() {
    let mut lane = DispatchLane::new(2);
    for name in ["a", "b", "c"] {
        lane.submit(id(name), "read", Classification::Parallel);
    }
    assert_eq!(lane.poll(), start("a", Classification::Parallel));
    assert_eq!(lane.poll(), start("b", Classification::Parallel));
    assert_eq!(lane.poll(), LaneStep::Wait, "the pool is full");
    assert_eq!(lane.pending(), 1);

    assert!(lane.settle(&id("a")));
    assert_eq!(lane.poll(), commit("a", Classification::Parallel));
    assert!(lane.committed(&id("a")));
    assert_eq!(lane.poll(), start("c", Classification::Parallel));
    assert_eq!(lane.in_flight(), 2);
}

#[test]
fn an_exclusive_call_drains_the_pool_runs_alone_and_holds_its_barrier() {
    let mut lane = DispatchLane::new(4);
    lane.submit(id("p1"), "read", Classification::Parallel);
    lane.submit(id("x"), "write", Classification::Exclusive);
    lane.submit(id("p2"), "read", Classification::Parallel);

    assert_eq!(lane.poll(), start("p1", Classification::Parallel));
    assert_eq!(
        lane.poll(),
        LaneStep::Wait,
        "the exclusive call waits for the pool"
    );

    assert!(lane.settle(&id("p1")));
    assert_eq!(lane.poll(), commit("p1", Classification::Parallel));
    assert!(lane.committed(&id("p1")));
    assert_eq!(lane.poll(), start("x", Classification::Exclusive));
    assert_eq!(
        lane.poll(),
        LaneStep::Wait,
        "a later parallel call must not overlap the barrier"
    );

    assert!(lane.settle(&id("x")));
    assert_eq!(lane.poll(), commit("x", Classification::Exclusive));
    // The barrier covers the whole commit: the lane keeps asking for the same
    // commit, and nothing else starts, until the driver acknowledges it.
    lane.submit(id("p3"), "read", Classification::Parallel);
    assert_eq!(lane.poll(), commit("x", Classification::Exclusive));
    assert!(lane.committed(&id("x")));
    assert_eq!(lane.poll(), start("p2", Classification::Parallel));
    assert_eq!(lane.poll(), start("p3", Classification::Parallel));
}

#[test]
fn a_queued_call_can_be_reclassified_before_it_starts() {
    let mut lane = DispatchLane::new(4);
    lane.submit(id("a"), "read", Classification::Parallel);
    assert_eq!(lane.poll(), start("a", Classification::Parallel));
    lane.submit(id("b"), "later_exclusive", Classification::Parallel);

    // The registry changed while the call waited.
    assert!(lane.reclassify(&id("b"), Classification::Exclusive));

    assert!(lane.settle(&id("a")));
    assert_eq!(lane.poll(), commit("a", Classification::Parallel));
    assert!(lane.committed(&id("a")));
    assert_eq!(lane.poll(), start("b", Classification::Exclusive));
    // A started call keeps the mode it started under.
    assert!(!lane.reclassify(&id("b"), Classification::Parallel));
}

#[test]
fn an_aborted_run_abandons_queued_calls_and_still_commits_started_ones() {
    let mut lane = DispatchLane::new(1);
    lane.submit(id("a"), "read", Classification::Parallel);
    lane.submit(id("b"), "write", Classification::Parallel);
    lane.submit(id("c"), "write", Classification::Parallel);
    assert_eq!(lane.poll(), start("a", Classification::Parallel));

    lane.abort("run_code settled");
    assert_eq!(lane.abort_reason(), Some("run_code settled"));

    // Queued calls are abandoned unstarted: no start event, so a UI never
    // shows a call that never ran.
    assert_eq!(
        lane.poll(),
        LaneStep::Abandon {
            id: id("b"),
            message: "run_code run is over (run_code settled); write tool call abandoned".into(),
        }
    );
    assert_eq!(
        lane.poll(),
        LaneStep::Abandon {
            id: id("c"),
            message: "run_code run is over (run_code settled); write tool call abandoned".into(),
        }
    );
    assert_eq!(
        lane.poll(),
        LaneStep::Wait,
        "the started call still owes a commit"
    );

    assert!(lane.settle(&id("a")));
    assert_eq!(lane.poll(), commit("a", Classification::Parallel));
    assert!(lane.committed(&id("a")));
    assert_eq!(lane.poll(), LaneStep::Quiescent);
}

#[test]
fn the_lane_refuses_a_settle_it_never_started() {
    let mut lane = DispatchLane::new(2);
    lane.submit(id("a"), "read", Classification::Parallel);
    assert_eq!(lane.poll(), start("a", Classification::Parallel));

    assert!(!lane.settle(&id("ghost")), "an unknown id cannot settle");
    assert!(
        !lane.committed(&id("a")),
        "a call that has not settled cannot be committed"
    );
    assert!(lane.settle(&id("a")));
    assert!(!lane.settle(&id("a")), "a call settles exactly once");
    assert!(
        !lane.committed(&id("b")),
        "only the settled head may be committed"
    );
    assert!(lane.committed(&id("a")));
}

#[test]
fn the_run_over_message_names_the_phase_and_the_reason() {
    assert_eq!(
        run_over_message("read", "run_code settled", RunOverPhase::NotDispatched),
        "run_code run is over (run_code settled); read not dispatched"
    );
    assert_eq!(
        run_over_message("read", "run_code settled", RunOverPhase::Abandoned),
        "run_code run is over (run_code settled); read tool call abandoned"
    );
    assert_eq!(
        run_over_message("read", "run_code settled", RunOverPhase::Discarded),
        "run_code run is over (run_code settled); read result discarded"
    );
}
