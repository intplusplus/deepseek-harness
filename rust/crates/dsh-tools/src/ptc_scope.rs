//! The `ptc` presentation seam over the tool registry.
//!
//! Ported from the parts of `packages/core/tools/src/ptc.ts` that only need
//! the registry: which tools a program may call, the dispatch scope that mints
//! sub-call identities, and the presentation collapse that keeps a model from
//! calling a native tool directly under `ptc` mode.
//!
//! Nothing here executes a tool, starts a program, or appends an event. The
//! registry's existing entry points are used as they are — this module adds a
//! read-only view plus one small context object — so the scheduler's
//! guarantees and every existing call site are untouched.
//!
//! The wiring that a real `run_code` tool needs on top of this (execute a
//! program in a sandbox, bridge each call back through
//! [`DispatchScope`] + [`dsh_ptc::DispatchLane`], append
//! [`dsh_ptc::PtcEvent`]s) is deliberately absent: it needs a process, which
//! this workspace has not ported.

#![forbid(unsafe_code)]

use dsh_llm::ToolCallId;
use dsh_ptc::SubCallSequence;

use super::registry::ToolRegistry;

/// The tool name the `ptc` presentation reserves for its own transport.
pub use dsh_ptc::RUN_CODE_NAME;

/// The tool names a PTC program may call as `await tools.<name>(...)`.
///
/// This is the calling agent's visible set minus the `run_code` transport
/// itself, in registry order — the same set the SDK section promised the
/// model, so a program can bind exactly what its prompt declared. Sub-calls
/// re-resolve through the registry per call, which is why this returns names
/// rather than handlers: a tool registered while the program runs is
/// reachable, and a removed one fails as an unknown tool.
pub fn binding_names(registry: &ToolRegistry) -> Vec<String> {
    registry
        .schemas()
        .into_iter()
        .map(|schema| schema.name)
        .filter(|name| name != RUN_CODE_NAME)
        .collect()
}

/// Whether the `ptc` presentation denies a model-direct call to `name`.
///
/// Under `ptc` mode the only tool the model may name directly is the
/// `run_code` transport; every other name is reachable only from inside a
/// program. A nested call (one carrying a dispatch scope) bypasses the
/// collapse — that bypass is the entire mechanism, so the predicate has
/// exactly one home.
pub fn collapses_to_run_code(name: &str, nested: bool, ptc_mode: bool) -> bool {
    !nested && ptc_mode && name != RUN_CODE_NAME
}

/// The dispatch identity one PTC run hands to its nested calls.
///
/// `root_call_id` names the model-requested call that owns the execution tree;
/// `parent_call_id` is the enclosing `run_code` call. Both are carried by every
/// `tool/ptc-dispatch*` record so a reader can rebuild the tree without
/// timestamps, and the sub-call id is minted from the parent in submission
/// order — `<parent>:ptc:<n>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchScope {
    root_call_id: ToolCallId,
    parent_call_id: ToolCallId,
    sequence: SubCallSequence,
}

impl DispatchScope {
    /// Open a scope for one `run_code` execution.
    pub fn new(root_call_id: ToolCallId, parent_call_id: ToolCallId) -> Self {
        Self {
            root_call_id,
            parent_call_id,
            sequence: SubCallSequence::new(),
        }
    }

    /// The root model-requested call.
    pub fn root_call_id(&self) -> &ToolCallId {
        &self.root_call_id
    }

    /// The enclosing `run_code` call.
    pub fn parent_call_id(&self) -> &ToolCallId {
        &self.parent_call_id
    }

    /// Mint the next sub-call id, numbered from one in submission order.
    pub fn mint(&mut self) -> ToolCallId {
        self.sequence.mint(&self.parent_call_id)
    }

    /// How many sub-calls this scope has admitted.
    pub fn issued(&self) -> u64 {
        self.sequence.issued()
    }
}
