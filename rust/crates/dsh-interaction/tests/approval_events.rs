//! The approval audit events: the request id and the durable payloads.

use dsh_interaction::approval::{
    approval_event_payload, ApprovalAsked, ApprovalDecided, ApprovalError, ApprovalRequestId,
};
use dsh_llm::ToolCallId;
use dsh_tools::gate::ApprovalOutcome;
use serde_json::json;

#[test]
fn a_request_id_must_not_be_empty() {
    assert!(ApprovalRequestId::parse("").is_err());
    let error = ApprovalRequestId::parse("").unwrap_err();
    assert!(matches!(error, ApprovalError::EmptyRequestId));
    assert_eq!(error.to_string(), "approval request id must not be empty");
}

#[test]
fn a_request_id_brands_any_other_string() {
    let id = ApprovalRequestId::parse("req-1").unwrap();
    assert_eq!(id.as_str(), "req-1");
}

#[test]
fn a_request_id_round_trips_through_json() {
    let id = ApprovalRequestId::parse("3f2504e0-4f89-41d3-9a0c-0305e82c3301").unwrap();
    let value = serde_json::to_value(&id).unwrap();
    assert_eq!(value, json!("3f2504e0-4f89-41d3-9a0c-0305e82c3301"));
    let decoded: ApprovalRequestId = serde_json::from_value(value).unwrap();
    assert_eq!(decoded, id);
}

#[test]
fn the_asked_event_round_trips_through_json() {
    let asked = ApprovalAsked {
        id: ApprovalRequestId::parse("req-1").unwrap(),
        tool_name: "bash".to_string(),
        call_id: Some(ToolCallId::new("call-1")),
        reason: Some("hook says ask".to_string()),
    };
    let value = serde_json::to_value(&asked).unwrap();
    assert_eq!(
        value,
        json!({
            "id": "req-1",
            "toolName": "bash",
            "callId": "call-1",
            "reason": "hook says ask",
        })
    );
    let decoded: ApprovalAsked = serde_json::from_value(value).unwrap();
    assert_eq!(decoded, asked);
}

#[test]
fn absent_optional_fields_stay_absent_on_the_wire() {
    let asked = ApprovalAsked {
        id: ApprovalRequestId::parse("req-1").unwrap(),
        tool_name: "bash".to_string(),
        call_id: None,
        reason: None,
    };
    let value = serde_json::to_value(&asked).unwrap();
    assert_eq!(value, json!({"id": "req-1", "toolName": "bash"}));
    assert!(value.get("callId").is_none());
    assert!(value.get("reason").is_none());
}

#[test]
fn the_decided_event_round_trips_through_json() {
    for outcome in [
        ApprovalOutcome::AllowedOnce,
        ApprovalOutcome::Rejected,
        ApprovalOutcome::Cancelled,
        ApprovalOutcome::Unavailable,
    ] {
        let decided = ApprovalDecided {
            id: ApprovalRequestId::parse("req-1").unwrap(),
            outcome,
        };
        let value = serde_json::to_value(&decided).unwrap();
        let decoded: ApprovalDecided = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(decoded, decided);
        assert_eq!(value["id"], json!("req-1"));
    }
}

#[test]
fn the_outcome_vocabulary_matches_the_wire_format() {
    assert_eq!(
        serde_json::to_value(ApprovalOutcome::AllowedOnce).unwrap(),
        json!("allowed-once")
    );
    assert_eq!(
        serde_json::to_value(ApprovalOutcome::Rejected).unwrap(),
        json!("rejected")
    );
    assert_eq!(
        serde_json::to_value(ApprovalOutcome::Cancelled).unwrap(),
        json!("cancelled")
    );
    assert_eq!(
        serde_json::to_value(ApprovalOutcome::Unavailable).unwrap(),
        json!("unavailable")
    );
}

#[test]
fn an_audit_payload_is_built_as_a_json_object() {
    let asked = ApprovalAsked {
        id: ApprovalRequestId::parse("req-1").unwrap(),
        tool_name: "bash".to_string(),
        call_id: None,
        reason: None,
    };
    let payload = approval_event_payload(&asked).unwrap();
    assert_eq!(payload.get("toolName").unwrap(), &json!("bash"));
}
