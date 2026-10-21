//! Async approval seam: the mounted `ApprovalService`, the approval waterfall,
//! and cancel/timeout reconciliation of an in-flight approval.
//!
//! Every test runs with a scripted backend — no real UI — and pins the §3.6 /
//! §5 behavior that `crate::gate` left as the unported async half: an `Ask`
//! escalates to an `ApprovalService`, listeners vote in a converging waterfall,
//! and a cancel/timeout reconciles the pending request.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dsh_llm::ToolCallId;
use dsh_tools::*;
use serde_json::Value;

fn call(name: &str) -> ToolCallInput {
    ToolCallInput {
        call_id: ToolCallId::new("c1"),
        name: name.to_string(),
        arguments: Value::Null,
    }
}

fn request_for<'a>(
    call: &'a ToolCallInput,
    signal: Option<&'a CancelSignal>,
) -> ApprovalRequest<'a> {
    ApprovalRequest {
        tool_name: &call.name,
        call_id: &call.call_id,
        reason: Some("needs approval"),
        display_reason: None,
        signal,
    }
}

/// A backend whose future sets `dropped` when it is cancelled/dropped, proving
/// the in-flight approval was reconciled rather than left hanging.
struct DropSpyApprovalService {
    dropped: Arc<AtomicBool>,
}

#[async_trait]
impl ApprovalService for DropSpyApprovalService {
    async fn request(&self, _req: ApprovalRequest<'_>) -> ApprovalResult {
        struct Spy(Arc<AtomicBool>);
        impl Drop for Spy {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let _spy = Spy(self.dropped.clone());
        std::future::pending().await
    }
}

// ---------------------------------------------------------------------------
// Approval waterfall convergence
// ---------------------------------------------------------------------------

#[tokio::test]
async fn waterfall_allow_short_circuits_the_backend() {
    // An Allow listener wins immediately; a never-resolving backend is never
    // consulted, so the round-trip still completes as AllowedOnce.
    let call = call("x");
    let svc = WaterfallApprovalService::new(Arc::new(PendingApprovalService))
        .register(Arc::new(AllowListener));
    let result = svc.request(request_for(&call, None)).await;
    assert_eq!(result, ApprovalResult::AllowedOnce);
}

#[tokio::test]
async fn waterfall_deny_short_circuits_with_reason() {
    let call = call("x");
    let svc = WaterfallApprovalService::new(Arc::new(PendingApprovalService))
        .register(Arc::new(DenyListener::new("policy says no")));
    let result = svc.request(request_for(&call, None)).await;
    assert_eq!(
        result,
        ApprovalResult::Rejected {
            reason: Some("policy says no".to_string())
        }
    );
}

#[tokio::test]
async fn waterfall_all_delegate_falls_through_to_backend() {
    let call = call("x");
    let svc = WaterfallApprovalService::new(Arc::new(ConstApprovalService::new(
        ApprovalResult::AllowedOnce,
    )))
    .register(Arc::new(DelegateListener))
    .register(Arc::new(DelegateListener));
    let result = svc.request(request_for(&call, None)).await;
    assert_eq!(result, ApprovalResult::AllowedOnce);
}

#[tokio::test]
async fn waterfall_first_non_delegate_wins() {
    // The leading Delegate is skipped; the Deny listener wins over the tail.
    let call = call("x");
    let svc = WaterfallApprovalService::new(Arc::new(PendingApprovalService))
        .register(Arc::new(DelegateListener))
        .register(Arc::new(DenyListener::new("second")))
        .register(Arc::new(DenyListener::new("third")));
    let result = svc.request(request_for(&call, None)).await;
    assert_eq!(
        result,
        ApprovalResult::Rejected {
            reason: Some("second".to_string())
        }
    );
}

#[tokio::test]
async fn waterfall_fn_listener_can_vote_on_the_request() {
    let call = call("secret");
    let svc = WaterfallApprovalService::new(Arc::new(ConstApprovalService::new(
        ApprovalResult::AllowedOnce,
    )))
    .register(Arc::new(FnApprovalListener::new(|req| {
        if req.tool_name == "secret" {
            ApprovalVote::Deny {
                reason: "name is secret".to_string(),
            }
        } else {
            ApprovalVote::Delegate
        }
    })));
    let result = svc.request(request_for(&call, None)).await;
    assert_eq!(
        result,
        ApprovalResult::Rejected {
            reason: Some("name is secret".to_string())
        }
    );
}

// ---------------------------------------------------------------------------
// ask → allow / deny mapping
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ask_allows_when_backend_allows() {
    let call = call("x");
    let svc: Arc<dyn ApprovalService> = Arc::new(WaterfallApprovalService::new(Arc::new(
        ConstApprovalService::new(ApprovalResult::AllowedOnce),
    )));
    let svc_ref: &dyn ApprovalService = &*svc;
    let resolved = await_approval(&call, Some("needs approval"), Some(svc_ref), None).await;
    assert_eq!(resolved.decision, PreToolDecision::Allow);
    assert!(!resolved.approval_cancelled);
}

#[tokio::test]
async fn ask_denies_when_backend_rejects() {
    let call = call("x");
    let svc: Arc<dyn ApprovalService> = Arc::new(WaterfallApprovalService::new(Arc::new(
        ConstApprovalService::new(ApprovalResult::Rejected { reason: None }),
    )));
    let svc_ref: &dyn ApprovalService = &*svc;
    let resolved = await_approval(&call, Some("needs approval"), Some(svc_ref), None).await;
    assert_eq!(
        resolved.decision,
        PreToolDecision::deny("the user rejected tool \"x\"")
    );
    assert!(!resolved.approval_cancelled);
}

#[tokio::test]
async fn ask_degrades_to_deny_without_a_service() {
    let call = call("x");
    let resolved = await_approval(&call, Some("needs approval"), None, None).await;
    assert_eq!(resolved.decision, PreToolDecision::deny("needs approval"));
    assert!(!resolved.approval_cancelled);
}

#[tokio::test]
async fn ask_unavailable_denies_with_its_own_reason() {
    let call = call("x");
    let svc: Arc<dyn ApprovalService> = Arc::new(WaterfallApprovalService::new(Arc::new(
        ConstApprovalService::new(ApprovalResult::Unavailable),
    )));
    let svc_ref: &dyn ApprovalService = &*svc;
    let resolved = await_approval(&call, Some("needs approval"), Some(svc_ref), None).await;
    assert_eq!(
        resolved.decision,
        PreToolDecision::deny("tool \"x\" requires approval, but no approval channel is available")
    );
}

// ---------------------------------------------------------------------------
// Timeout handling
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pending_approval_times_out() {
    let call = call("x");
    let svc = WaterfallApprovalService::new(Arc::new(PendingApprovalService))
        .with_timeout(Duration::from_millis(20));
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        svc.request(request_for(&call, None)),
    )
    .await
    .expect("timeout must fire before the test guard");
    assert_eq!(result, ApprovalResult::Timeout);

    let resolved = resolve_approval("x", None, Some(result));
    assert_eq!(
        resolved.decision,
        PreToolDecision::deny("approval for tool \"x\" timed out")
    );
    assert!(!resolved.approval_cancelled);
}

// ---------------------------------------------------------------------------
// Cancel reconciliation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn in_flight_approval_reconciles_as_cancelled() {
    let call = call("x");
    let dropped = Arc::new(AtomicBool::new(false));
    let svc = WaterfallApprovalService::new(Arc::new(DropSpyApprovalService {
        dropped: dropped.clone(),
    }));
    let signal = CancelSignal::new();
    // Cancel while the approval is still in flight (the backend never resolves
    // on its own), and let the round-trip observe the cancellation.
    signal.cancel();
    let req = request_for(&call, Some(&signal));
    let result = svc.request(req).await;

    // The round-trip reconciles as Cancelled and the in-flight backend future
    // was dropped, not left hanging.
    assert_eq!(result, ApprovalResult::Cancelled);
    assert!(dropped.load(Ordering::SeqCst));

    let resolved = resolve_approval(&call.name, Some("needs approval"), Some(result));
    assert_eq!(
        resolved.decision,
        PreToolDecision::deny("approval for tool \"x\" was cancelled")
    );
    assert!(resolved.approval_cancelled);
}

#[tokio::test]
async fn cancel_takes_precedence_over_backend_decision() {
    // A fired signal reconciles the request as Cancelled even though the
    // backend would otherwise have answered (it is still pending here).
    let call = call("x");
    let svc = WaterfallApprovalService::new(Arc::new(PendingApprovalService));
    let signal = CancelSignal::new();
    signal.cancel();

    let req = request_for(&call, Some(&signal));
    let result = svc.request(req).await;
    assert_eq!(result, ApprovalResult::Cancelled);
}

#[tokio::test]
async fn cancel_arriving_mid_flight_reconciles() {
    // The cancellation signal fires only after the approval is already in
    // flight, matching the TypeScript registry recheck that swaps a delayed
    // approval for the canonical before-dispatch abort.
    let call = call("x");
    let dropped = Arc::new(AtomicBool::new(false));
    let svc = WaterfallApprovalService::new(Arc::new(DropSpyApprovalService {
        dropped: dropped.clone(),
    }));
    let signal = CancelSignal::new();

    let call = call.clone();
    let svc = svc.clone();
    let signal = signal.clone();
    let signal_for_cancel = signal.clone();
    let dropped = dropped.clone();
    let handle = tokio::spawn(async move {
        let req = request_for(&call, Some(&signal));
        svc.request(req).await
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    signal_for_cancel.cancel();

    let result = handle.await.unwrap();
    assert_eq!(result, ApprovalResult::Cancelled);
    assert!(dropped.load(Ordering::SeqCst));
}

// ---------------------------------------------------------------------------
// Pure resolver mapping
// ---------------------------------------------------------------------------

#[test]
fn resolve_approval_none_degrades_to_deny() {
    let resolved = resolve_approval("x", Some("r"), None);
    assert_eq!(resolved.decision, PreToolDecision::deny("r"));
    assert!(!resolved.approval_cancelled);
}

#[test]
fn resolve_approval_listener_reason_is_preserved() {
    let resolved = resolve_approval(
        "x",
        Some("r"),
        Some(ApprovalResult::Rejected {
            reason: Some("custom".to_string()),
        }),
    );
    assert_eq!(resolved.decision, PreToolDecision::deny("custom"));
    assert!(!resolved.approval_cancelled);
}
