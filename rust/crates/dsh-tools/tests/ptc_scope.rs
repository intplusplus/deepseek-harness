//! The registry-facing PTC seam.
//!
//! The binding set and the collapse predicate come from
//! `packages/core/tools/src/ptc.ts`; the sub-call id shape from its scheduler.

use std::sync::Arc;

use dsh_llm::{ToolCallId, ToolSchema};
use dsh_tools::{
    binding_names, collapses_to_run_code, CancelSignal, DispatchScope, ToolCallInput,
    ToolExecutionResult, ToolHandler, ToolRegistry, RUN_CODE_NAME,
};
use serde_json::json;

struct Stub {
    name: &'static str,
}

impl ToolHandler for Stub {
    fn name(&self) -> &str {
        self.name
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name,
            format!("{} tool", self.name),
            json!({ "type": "object" }),
        )
    }

    fn call(&self, _input: &ToolCallInput, _cancel: &CancelSignal) -> ToolExecutionResult {
        ToolExecutionResult::success(Vec::new())
    }
}

fn registry(names: &[&'static str]) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for name in names {
        registry.register(Arc::new(Stub { name }));
    }
    registry
}

#[test]
fn a_program_binds_every_tool_except_its_own_transport() {
    let registry = registry(&["read", "write", RUN_CODE_NAME]);
    assert_eq!(binding_names(&registry), ["read", "write"]);
    assert!(
        !binding_names(&registry).contains(&RUN_CODE_NAME.to_owned()),
        "a program must not be able to call run_code recursively"
    );
}

#[test]
fn an_empty_registry_binds_nothing() {
    assert!(binding_names(&ToolRegistry::new()).is_empty());
}

#[test]
fn the_collapse_hides_native_tools_from_the_model_but_not_from_a_program() {
    // Under ptc mode a model-direct call may name only the transport.
    assert!(collapses_to_run_code("read", false, true));
    assert!(!collapses_to_run_code(RUN_CODE_NAME, false, true));
    // A nested dispatch — the program's own call — always reaches the tool.
    assert!(!collapses_to_run_code("read", true, true));
    // Native presentation has no collapse at all.
    assert!(!collapses_to_run_code("read", false, false));
}

#[test]
fn the_dispatch_scope_numbers_sub_calls_from_one_in_submission_order() {
    let mut scope = DispatchScope::new(ToolCallId::new("call-1"), ToolCallId::new("call-2"));
    assert_eq!(scope.root_call_id().as_str(), "call-1");
    assert_eq!(scope.parent_call_id().as_str(), "call-2");
    assert_eq!(scope.mint().as_str(), "call-2:ptc:1");
    assert_eq!(scope.mint().as_str(), "call-2:ptc:2");
    assert_eq!(scope.issued(), 2);
}

#[test]
fn the_scope_feeds_the_durable_record_it_will_append() {
    let mut scope = DispatchScope::new(ToolCallId::new("call-1"), ToolCallId::new("call-1"));
    let sub_call_id = scope.mint();
    let event = dsh_ptc::PtcDispatchStart::new(
        scope.root_call_id().clone(),
        scope.parent_call_id().clone(),
        sub_call_id.clone(),
        "read",
        json!({ "path": "a.ts" }),
    );
    let encoded = dsh_ptc::PtcEvent::Start(event).encode().expect("encodable");
    let payload = encoded.extension().expect("extension").payload();
    assert_eq!(payload["subCallId"], json!(sub_call_id.as_str()));
    assert_eq!(
        payload["parentCallId"],
        json!(scope.parent_call_id().as_str())
    );
}
