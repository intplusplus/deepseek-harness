//! The approval policy layer: the closed vocabulary, the durable override, the
//! effective-policy fold, and the audited request path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use dsh_interaction::approval::{
    approval_policy_override, effective_approval_policy, has_open_turn, set_approval_policy,
    set_approval_policy_str, ApprovalError, ApprovalIdGenerator, ApprovalPolicy,
    AuditedApprovalRequest, AuditedApprovalService, CountingApprovalIds, RandomApprovalIds,
    APPROVAL_ASKED_EVENT, APPROVAL_DECIDED_EVENT, APPROVAL_POLICIES, APPROVAL_POLICY_EVENT,
    ASK_SENTENCE, NEVER_SENTENCE,
};
use dsh_llm::ToolCallId;
use dsh_session::{Session, SessionEvent, SessionId, SessionMeta};
use dsh_tools::approval::{ApprovalResult, ConstApprovalService, FnApprovalService};
use dsh_tools::cancel::CancelSignal;
use dsh_tools::gate::ApprovalOutcome;

/// A session with one open turn, so the audited path's enclosure precondition
/// is satisfied.
fn session_with_open_turn(id: &str) -> Session {
    let mut session = Session::create(SessionId::new(id).unwrap(), SessionMeta::default(), 0);
    session.append(SessionEvent::turn_start(1)).unwrap();
    session
}

/// The durable extension events of a session, in append order.
fn extension_events(session: &Session) -> Vec<&SessionEvent> {
    session
        .events()
        .iter()
        .filter(|event| event.extension().is_some())
        .collect()
}

#[test]
fn the_policy_vocabulary_is_closed() {
    assert_eq!(
        APPROVAL_POLICIES,
        [ApprovalPolicy::Ask, ApprovalPolicy::Never]
    );
    assert_eq!(ApprovalPolicy::Ask.as_str(), "ask");
    assert_eq!(ApprovalPolicy::Never.as_str(), "never");
    assert_eq!(ApprovalPolicy::parse("ask"), Some(ApprovalPolicy::Ask));
    assert_eq!(ApprovalPolicy::parse("never"), Some(ApprovalPolicy::Never));
    assert_eq!(ApprovalPolicy::parse("sometimes"), None);
    assert_eq!(ApprovalPolicy::parse(""), None);
    assert_eq!(ApprovalPolicy::parse("Ask"), None);
}

#[test]
fn a_policy_override_is_durable() {
    let mut session = session_with_open_turn("sess-policy");
    set_approval_policy(&mut session, ApprovalPolicy::Never).unwrap();

    let last = session.events().last().unwrap();
    assert_eq!(last.extension().unwrap().kind(), APPROVAL_POLICY_EVENT);
    let value = serde_json::to_value(last).unwrap();
    assert_eq!(value["type"], serde_json::json!("approval/policy"));
    assert_eq!(value["policy"], serde_json::json!("never"));
    // The delegation source is absent for a runtime switch.
    assert!(value.get("source").is_none());
}

#[test]
fn an_unknown_policy_is_rejected_before_the_log_changes() {
    let mut session = session_with_open_turn("sess-unknown");
    let before = session.len();

    let error = set_approval_policy_str(&mut session, "sometimes").unwrap_err();
    assert!(matches!(error, ApprovalError::UnknownPolicy));
    assert_eq!(
        error.to_string(),
        "approval policy must be one of \"ask\" or \"never\""
    );
    assert_eq!(session.len(), before);
}

#[test]
fn the_policy_override_folds_to_the_last_event() {
    let mut session = session_with_open_turn("sess-fold");
    assert_eq!(approval_policy_override(session.events()), None);

    set_approval_policy(&mut session, ApprovalPolicy::Never).unwrap();
    assert_eq!(
        approval_policy_override(session.events()),
        Some(ApprovalPolicy::Never)
    );

    set_approval_policy(&mut session, ApprovalPolicy::Ask).unwrap();
    assert_eq!(
        approval_policy_override(session.events()),
        Some(ApprovalPolicy::Ask)
    );
}

#[test]
fn the_session_override_outranks_the_configured_default() {
    let mut session = session_with_open_turn("sess-override");
    assert_eq!(
        effective_approval_policy(session.events(), ApprovalPolicy::Never),
        ApprovalPolicy::Never
    );

    set_approval_policy(&mut session, ApprovalPolicy::Ask).unwrap();
    assert_eq!(
        effective_approval_policy(session.events(), ApprovalPolicy::Never),
        ApprovalPolicy::Ask
    );

    set_approval_policy(&mut session, ApprovalPolicy::Never).unwrap();
    assert_eq!(
        effective_approval_policy(session.events(), ApprovalPolicy::Ask),
        ApprovalPolicy::Never
    );
}

#[test]
fn each_policy_contributes_its_model_facing_sentence() {
    assert_eq!(ApprovalPolicy::Ask.sentence(), ASK_SENTENCE);
    assert_eq!(ApprovalPolicy::Never.sentence(), NEVER_SENTENCE);
    assert!(ASK_SENTENCE.contains("Approval policy: ask."));
    assert!(NEVER_SENTENCE.contains("Approval prompts are disabled in this session"));
}

#[test]
fn the_policy_change_notice_names_both_policies() {
    let notice =
        dsh_interaction::approval::policy_change_notice(ApprovalPolicy::Ask, ApprovalPolicy::Never);
    assert_eq!(
        notice,
        "The approval policy changed from \"ask\" to \"never\" (changed by the user)."
    );
}

#[test]
fn an_open_turn_satisfies_the_enclosure_precondition() {
    let session = session_with_open_turn("sess-open");
    assert!(has_open_turn(session.events()));

    let mut closed = session_with_open_turn("sess-closed");
    closed
        .append(SessionEvent::turn_end(
            1,
            dsh_session::TurnEndReason::Completed,
        ))
        .unwrap();
    assert!(!has_open_turn(closed.events()));

    let idle = Session::create(
        SessionId::new("sess-idle").unwrap(),
        SessionMeta::default(),
        0,
    );
    assert!(!has_open_turn(idle.events()));
}

#[tokio::test]
async fn an_idle_ask_is_rejected_before_anything_is_appended() {
    let session = Arc::new(Mutex::new(Session::create(
        SessionId::new("sess-idle-ask").unwrap(),
        SessionMeta::default(),
        0,
    )));
    let answerer = Arc::new(ConstApprovalService::new(ApprovalResult::AllowedOnce));
    let service = AuditedApprovalService::new(Arc::clone(&session), answerer, ApprovalPolicy::Ask);

    let error = service
        .request(AuditedApprovalRequest::new("echo"))
        .await
        .unwrap_err();
    assert!(matches!(error, ApprovalError::NoOpenTurn));
    assert!(extension_events(&session.lock().unwrap()).is_empty());
}

#[tokio::test]
async fn the_audit_pair_records_what_was_asked_and_how_it_settled() {
    let session = Arc::new(Mutex::new(session_with_open_turn("sess-audit")));
    let answerer = Arc::new(ConstApprovalService::new(ApprovalResult::AllowedOnce));
    let service = AuditedApprovalService::new(Arc::clone(&session), answerer, ApprovalPolicy::Ask)
        .with_id_generator(Arc::new(CountingApprovalIds::new("req")));

    let call_id = ToolCallId::new("call-1");
    let outcome = service
        .request(
            AuditedApprovalRequest::new("echo")
                .with_call_id(&call_id)
                .with_reason("hook says ask"),
        )
        .await
        .unwrap();
    assert_eq!(outcome, ApprovalOutcome::AllowedOnce);

    let session = session.lock().unwrap();
    let events = extension_events(&session);
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].extension().unwrap().kind(), APPROVAL_ASKED_EVENT);
    assert_eq!(
        events[1].extension().unwrap().kind(),
        APPROVAL_DECIDED_EVENT
    );

    let asked = serde_json::to_value(events[0]).unwrap();
    assert_eq!(asked["toolName"], serde_json::json!("echo"));
    assert_eq!(asked["callId"], serde_json::json!("call-1"));
    assert_eq!(asked["reason"], serde_json::json!("hook says ask"));

    let decided = serde_json::to_value(events[1]).unwrap();
    assert_eq!(decided["outcome"], serde_json::json!("allowed-once"));
    // The pair shares one id.
    assert_eq!(asked["id"], decided["id"]);
}

#[tokio::test]
async fn absent_optional_fields_are_omitted_from_the_asked_event() {
    let session = Arc::new(Mutex::new(session_with_open_turn("sess-minimal")));
    let answerer = Arc::new(ConstApprovalService::new(ApprovalResult::AllowedOnce));
    let service = AuditedApprovalService::new(Arc::clone(&session), answerer, ApprovalPolicy::Ask);

    service
        .request(AuditedApprovalRequest::new("echo"))
        .await
        .unwrap();

    let session = session.lock().unwrap();
    let asked = serde_json::to_value(extension_events(&session)[0]).unwrap();
    let mut keys = asked
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();
    assert_eq!(
        keys,
        vec!["id".to_string(), "toolName".to_string(), "type".to_string()]
    );
}

#[tokio::test]
async fn the_never_policy_rejects_without_consulting_any_answerer() {
    let session = Arc::new(Mutex::new(session_with_open_turn("sess-never")));
    let consulted = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&consulted);
    let answerer = Arc::new(FnApprovalService::new(move |_| {
        flag.store(true, Ordering::SeqCst);
        ApprovalResult::AllowedOnce
    }));
    let service =
        AuditedApprovalService::new(Arc::clone(&session), answerer, ApprovalPolicy::Never);

    let outcome = service
        .request(AuditedApprovalRequest::new("bash"))
        .await
        .unwrap();
    assert_eq!(outcome, ApprovalOutcome::Rejected);
    assert!(!consulted.load(Ordering::SeqCst));

    // The audit pair still lands: a `never` rejection is still an ask that was
    // asked and decided.
    let session = session.lock().unwrap();
    let events = extension_events(&session);
    assert_eq!(events.len(), 2);
    let decided = serde_json::to_value(events[1]).unwrap();
    assert_eq!(decided["outcome"], serde_json::json!("rejected"));
}

#[tokio::test]
async fn a_session_override_outranks_the_configured_default_in_both_directions() {
    let mut session = session_with_open_turn("sess-both");
    set_approval_policy(&mut session, ApprovalPolicy::Ask).unwrap();
    let session = Arc::new(Mutex::new(session));
    let answerer = Arc::new(ConstApprovalService::new(ApprovalResult::AllowedOnce));
    let service =
        AuditedApprovalService::new(Arc::clone(&session), answerer, ApprovalPolicy::Never);

    let outcome = service
        .request(AuditedApprovalRequest::new("bash"))
        .await
        .unwrap();
    assert_eq!(outcome, ApprovalOutcome::AllowedOnce);
}

#[tokio::test]
async fn an_aborted_signal_settles_the_request_as_cancelled() {
    let session = Arc::new(Mutex::new(session_with_open_turn("sess-cancel")));
    let consulted = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&consulted);
    let answerer = Arc::new(FnApprovalService::new(move |_| {
        flag.store(true, Ordering::SeqCst);
        ApprovalResult::AllowedOnce
    }));
    let service = AuditedApprovalService::new(Arc::clone(&session), answerer, ApprovalPolicy::Ask);

    let signal = CancelSignal::new();
    signal.cancel();
    let outcome = service
        .request(AuditedApprovalRequest::new("echo").with_signal(&signal))
        .await
        .unwrap();
    assert_eq!(outcome, ApprovalOutcome::Cancelled);
    assert!(!consulted.load(Ordering::SeqCst));

    let session = session.lock().unwrap();
    let decided = serde_json::to_value(extension_events(&session)[1]).unwrap();
    assert_eq!(decided["outcome"], serde_json::json!("cancelled"));
}

#[tokio::test]
async fn a_throwing_answerer_fails_the_question_closed() {
    let session = Arc::new(Mutex::new(session_with_open_turn("sess-throw")));
    let answerer = Arc::new(FnApprovalService::new(|_| -> ApprovalResult {
        panic!("answerer died");
    }));
    let service = AuditedApprovalService::new(Arc::clone(&session), answerer, ApprovalPolicy::Ask);

    let outcome = service
        .request(AuditedApprovalRequest::new("echo"))
        .await
        .unwrap();
    assert_eq!(outcome, ApprovalOutcome::Unavailable);

    let session = session.lock().unwrap();
    let decided = serde_json::to_value(extension_events(&session)[1]).unwrap();
    assert_eq!(decided["outcome"], serde_json::json!("unavailable"));
}

#[tokio::test]
async fn every_request_issues_a_fresh_id() {
    let session = Arc::new(Mutex::new(session_with_open_turn("sess-ids")));
    let answerer = Arc::new(ConstApprovalService::new(ApprovalResult::AllowedOnce));
    let service = AuditedApprovalService::new(Arc::clone(&session), answerer, ApprovalPolicy::Ask)
        .with_id_generator(Arc::new(CountingApprovalIds::new("req")));

    service
        .request(AuditedApprovalRequest::new("echo"))
        .await
        .unwrap();
    service
        .request(AuditedApprovalRequest::new("echo"))
        .await
        .unwrap();

    let session = session.lock().unwrap();
    let ids = extension_events(&session)
        .iter()
        .filter(|event| {
            event
                .extension()
                .is_some_and(|extension| extension.kind() == APPROVAL_ASKED_EVENT)
        })
        .map(|event| serde_json::to_value(event).unwrap()["id"].clone())
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
}

#[test]
fn the_counting_generator_issues_a_deterministic_sequence() {
    let generator = CountingApprovalIds::new("req");
    assert_eq!(generator.next_request_id().as_str(), "req-1");
    assert_eq!(generator.next_request_id().as_str(), "req-2");
}

#[test]
fn the_random_generator_issues_distinct_ids() {
    let generator = RandomApprovalIds;
    let first = generator.next_request_id();
    let second = generator.next_request_id();
    assert_ne!(first, second);
    assert_eq!(first.as_str().len(), 36);
}
