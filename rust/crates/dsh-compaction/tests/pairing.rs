//! Tool-call/result balance over current surface positions.
//!
//! Ported from `packages/compaction/compaction/tests/tool-pairing.spec.ts`.

mod common;

use common::seq;
use dsh_compaction::{
    tool_pairing_balanced_after, tool_pairing_balanced_before, SurfaceNode, SurfaceNodeKind,
    ToolPairing, ToolPairingError,
};

fn node(value: u64, kind: SurfaceNodeKind) -> SurfaceNode {
    SurfaceNode::new(seq(value), kind)
}

fn call_node(value: u64, tool_calls: usize) -> SurfaceNode {
    node(value, SurfaceNodeKind::Assistant { tool_calls })
}

#[test]
fn classifies_closed_and_open_single_call_steps() {
    let nodes = [
        node(0, SurfaceNodeKind::User),
        call_node(1, 1),
        node(2, SurfaceNodeKind::Tool),
    ];
    let pairing = ToolPairing::new(&nodes).unwrap();
    assert_eq!(pairing.cuts(), &[true, true, false, true]);
    assert!(pairing.balanced_before(seq(0)).unwrap());
    assert!(pairing.balanced_after(seq(0)).unwrap());
    assert!(pairing.balanced_before(seq(1)).unwrap());
    assert!(!pairing.balanced_after(seq(1)).unwrap());
    assert!(!pairing.balanced_before(seq(2)).unwrap());
    assert!(pairing.balanced_after(seq(2)).unwrap());

    let open = [call_node(0, 1)];
    let pairing = ToolPairing::new(&open).unwrap();
    assert!(!pairing.balanced_after(seq(0)).unwrap());
}

#[test]
fn requires_every_result_from_a_multiple_call_assistant_message() {
    let nodes = [
        call_node(0, 2),
        node(1, SurfaceNodeKind::Tool),
        node(2, SurfaceNodeKind::Tool),
    ];
    let pairing = ToolPairing::new(&nodes).unwrap();
    assert!(!pairing.balanced_after(seq(1)).unwrap());
    assert!(pairing.balanced_after(seq(2)).unwrap());
}

#[test]
fn keeps_neutral_nodes_inside_an_open_pair_unbalanced() {
    let nodes = [
        call_node(0, 1),
        node(1, SurfaceNodeKind::User),
        node(2, SurfaceNodeKind::Tool),
    ];
    let pairing = ToolPairing::new(&nodes).unwrap();
    assert!(!pairing.balanced_before(seq(1)).unwrap());
    assert!(!pairing.balanced_after(seq(1)).unwrap());

    let free = [node(0, SurfaceNodeKind::User)];
    let pairing = ToolPairing::new(&free).unwrap();
    assert!(pairing.balanced_before(seq(0)).unwrap());
    assert!(pairing.balanced_after(seq(0)).unwrap());
}

#[test]
fn rejects_an_orphan_result() {
    let nodes = [node(0, SurfaceNodeKind::Tool)];
    let error = ToolPairing::new(&nodes).unwrap_err();
    assert!(matches!(error, ToolPairingError::OrphanResult { seq: at } if at == seq(0)));
    assert!(error.to_string().contains("no matching tool-call"));
}

#[test]
fn rejects_a_missing_sequence_before_and_after() {
    let pairing = ToolPairing::new(&[]).unwrap();
    assert_eq!(pairing.cuts(), &[true]);
    for result in [
        pairing.balanced_before(seq(999)),
        pairing.balanced_after(seq(999)),
    ] {
        assert!(matches!(
            result.unwrap_err(),
            ToolPairingError::MissingSeq { seq: at } if at == seq(999)
        ));
    }
}

#[test]
fn the_cut_after_the_tail_is_balanced() {
    let nodes = [
        node(0, SurfaceNodeKind::System),
        node(1, SurfaceNodeKind::User),
    ];
    let pairing = ToolPairing::new(&nodes).unwrap();
    assert_eq!(pairing.cuts().len(), nodes.len() + 1);
    assert_eq!(pairing.cuts().last(), Some(&true));
    assert!(pairing.balanced_after(seq(1)).unwrap());
}

#[test]
fn free_helpers_match_the_typed_pairing() {
    let nodes = [
        node(0, SurfaceNodeKind::User),
        call_node(1, 1),
        node(2, SurfaceNodeKind::Tool),
    ];
    assert!(!tool_pairing_balanced_before(&nodes, seq(2)).unwrap());
    assert!(tool_pairing_balanced_after(&nodes, seq(2)).unwrap());
    assert!(!tool_pairing_balanced_after(&nodes, seq(1)).unwrap());
}
