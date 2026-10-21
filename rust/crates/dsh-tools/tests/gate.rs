//! Tool permission and approval: the decision unions, the convergence rules,
//! the restriction filter, and the gated dispatch path.
//!
//! The pure-rule tests port the assertions of `packages/core/tools/tests/
//! tools.spec.ts` and `scoped.spec.ts` that need no async runtime and no
//! mounted approval service; the scheduler tests pin the §3.5 invariants for
//! calls the gate stopped — the body never runs, yet the call still settles
//! with a paired `tool/call` + `tool/result`, in model order.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use dsh_llm::{ContentBlock, MessageId, ToolCallId, UserMessage};
use dsh_session::{CoreEvent, Session, SessionId, SessionMeta, ToolErrorInfo};
use dsh_tools::*;

fn session() -> Session {
    Session::create(SessionId::new("t").unwrap(), SessionMeta::default(), 0)
}

fn call(id: &str, name: &str, arguments: &str) -> ContentBlock {
    ContentBlock::tool_call(ToolCallId::new(id), name, arguments)
}

fn schedule<'a>(registry: &'a ToolRegistry, cancel: &'a CancelSignal) -> ToolSchedule<'a> {
    ToolSchedule {
        turn: 1,
        step: 1,
        registry,
        config: ToolScheduleConfig::default(),
        cancel,
    }
}

fn text(content: &str) -> ContentBlock {
    ContentBlock::text(content)
}

fn user(texts: &[&str]) -> UserMessage {
    UserMessage::new(MessageId::random(), texts.iter().map(|t| text(t)).collect())
}

/// A tool that echoes its arguments and records the order it ran in.
struct EchoTool {
    name: String,
    mode: ToolExecutionMode,
    log: Arc<Mutex<Vec<String>>>,
}

impl EchoTool {
    fn new(name: &str, mode: ToolExecutionMode, log: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            name: name.to_string(),
            mode,
            log,
        }
    }
}

impl ToolHandler for EchoTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> dsh_llm::ToolSchema {
        dsh_llm::ToolSchema::new(
            self.name.clone(),
            "echo",
            serde_json::json!({ "type": "object", "properties": { "text": { "type": "string" } } }),
        )
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        self.mode
    }

    fn call(&self, input: &ToolCallInput, _cancel: &CancelSignal) -> ToolExecutionResult {
        self.log
            .lock()
            .unwrap()
            .push(input.call_id.as_str().to_string());
        let echoed = argument(input, "text")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        ToolExecutionResult::success(vec![text(&format!("{}:{}", self.name, echoed))])
            .with_value(serde_json::json!({ "echoed": echoed }))
    }
}

/// One committed result, as the durable log records it.
#[derive(Debug, PartialEq)]
struct CommittedResult {
    call_id: String,
    texts: Vec<String>,
    error: Option<(String, String)>,
}

/// Every committed result with the call id it answers, in commit order.
///
/// Results commit in model order, but bodies overlap, so a result is paired
/// with its call by the id it cites rather than by adjacency in the log.
fn committed_results(session: &Session) -> Vec<CommittedResult> {
    session
        .events()
        .iter()
        .filter_map(|event| match event.core() {
            Some(CoreEvent::ToolResult { message, error, .. }) => Some(CommittedResult {
                call_id: message.tool_call_id.as_str().to_string(),
                texts: message
                    .content
                    .iter()
                    .filter_map(ContentBlock::as_text)
                    .map(str::to_owned)
                    .collect(),
                error: error
                    .as_ref()
                    .map(|info| (info.name.clone(), info.code.clone())),
            }),
            _ => None,
        })
        .collect()
}

/// A gate that denies the named tools, allows everything else, and records
/// every call it was consulted for.
struct DenyGate {
    denied: Vec<String>,
    seen: Arc<Mutex<Vec<String>>>,
}

impl DenyGate {
    fn new(denied: Vec<String>, seen: Arc<Mutex<Vec<String>>>) -> Self {
        Self { denied, seen }
    }
}

impl PreToolGate for DenyGate {
    fn decide(&self, call: &ToolCallInput) -> PreToolDecision {
        self.seen
            .lock()
            .unwrap()
            .push(call.call_id.as_str().to_string());
        if self.denied.iter().any(|name| name == &call.name) {
            PreToolDecision::deny(format!("{} is not permitted", call.name))
        } else {
            PreToolDecision::Allow
        }
    }
}

// ---------------------------------------------------------------------------
// The waterfall convergence rules
// ---------------------------------------------------------------------------

/// The first vote that does not delegate wins; the tail is never consulted.
#[test]
fn the_first_non_delegating_vote_wins_and_the_tail_is_never_consulted() {
    let votes = [
        PreToolVote::Delegate,
        PreToolVote::decide(PreToolDecision::deny("first denial")),
        PreToolVote::decide(PreToolDecision::allow()),
    ];
    let converged = converge_pre_tool(&votes, PreToolDecision::allow());
    assert_eq!(converged, PreToolDecision::deny("first denial"));
}

/// An all-delegating chain — including an empty one — converges to the
/// registry's `allow` default.
#[test]
fn an_all_delegating_chain_converges_to_the_default() {
    let votes = [PreToolVote::Delegate, PreToolVote::Delegate];
    assert_eq!(
        converge_pre_tool(&votes, PreToolDecision::allow()),
        PreToolDecision::Allow
    );
    assert_eq!(
        converge_pre_tool(&[], PreToolDecision::allow()),
        PreToolDecision::Allow
    );
    assert_eq!(
        converge_pre_tool(&[], PreToolDecision::deny("closed by default")),
        PreToolDecision::deny("closed by default")
    );
}

/// A guard can deny an allowed call, but no guard can re-allow a call the
/// waterfall already denied, and an abstaining guard changes nothing.
#[test]
fn a_guard_can_deny_an_allowed_call_but_never_re_allow_a_denied_one() {
    let allowed = apply_guard(PreToolDecision::Allow, Some("terminal policy".to_string()));
    assert_eq!(allowed, PreToolDecision::deny("terminal policy"));

    let abstained = apply_guard(PreToolDecision::Allow, None);
    assert_eq!(abstained, PreToolDecision::Allow);

    let denied = PreToolDecision::deny_with_info(
        "waterfall denial",
        ToolErrorInfo {
            name: "AutoReviewDeniedError".to_string(),
            code: "AUTO_REVIEW_DENIED".to_string(),
            reason: None,
        },
    );
    let re_examined = apply_guard(denied.clone(), None);
    assert_eq!(
        re_examined, denied,
        "a non-allow decision is never re-examined"
    );
}

// ---------------------------------------------------------------------------
// The ask degradation rules
// ---------------------------------------------------------------------------

/// `ask` with no approval seam denies with the ask's own reason.
#[test]
fn ask_without_an_approval_seam_degrades_to_a_denial_with_the_ask_reason() {
    let resolved = resolve_ask("echo", Some("needs approval"), None);
    assert_eq!(resolved.decision, PreToolDecision::deny("needs approval"));
    assert!(!resolved.approval_cancelled);
}

/// `ask` with no reason degrades to the default denial message.
#[test]
fn ask_without_a_reason_degrades_to_the_default_denial_message() {
    let resolved = resolve_ask("echo", None, None);
    assert_eq!(
        resolved.decision,
        PreToolDecision::deny("tool \"echo\" requires approval (not yet supported)")
    );
}

/// The four approval outcomes map one-to-one; only `cancelled` reports it.
#[test]
fn approval_outcomes_map_one_to_one_onto_allow_or_deny() {
    let granted = resolve_ask("echo", None, Some(ApprovalOutcome::AllowedOnce));
    assert_eq!(granted.decision, PreToolDecision::Allow);
    assert!(!granted.approval_cancelled);

    let rejected = resolve_ask("echo", None, Some(ApprovalOutcome::Rejected));
    assert_eq!(
        rejected.decision,
        PreToolDecision::deny("the user rejected tool \"echo\"")
    );

    let cancelled = resolve_ask("echo", None, Some(ApprovalOutcome::Cancelled));
    assert_eq!(
        cancelled.decision,
        PreToolDecision::deny("approval for tool \"echo\" was cancelled")
    );
    assert!(cancelled.approval_cancelled);

    let unavailable = resolve_ask("echo", None, Some(ApprovalOutcome::Unavailable));
    assert_eq!(
        unavailable.decision,
        PreToolDecision::deny(
            "tool \"echo\" requires approval, but no approval channel is available"
        )
    );
    assert!(!unavailable.approval_cancelled);
}

// ---------------------------------------------------------------------------
// Mapping a converged decision onto the scheduler action
// ---------------------------------------------------------------------------

/// Only `allow` runs the body; every other decision settles the call.
#[test]
fn evaluate_pre_tool_maps_each_decision_onto_the_scheduler_action() {
    assert_eq!(
        evaluate_pre_tool("echo", &PreToolDecision::Allow),
        GateOutcome::Run
    );
    assert_eq!(
        evaluate_pre_tool("echo", &PreToolDecision::cancel()),
        GateOutcome::Cancelled(aborted_before_dispatch_result())
    );
    assert_eq!(
        evaluate_pre_tool("echo", &PreToolDecision::ask(Some("needs approval"))),
        GateOutcome::Denied(ToolExecutionResult::failure("needs approval"))
    );
}

/// A denial materializes its reason as `Error: {reason}` content and as the
/// failure message, with the optional structured identity attached.
#[test]
fn a_denial_materializes_its_reason_and_structured_identity() {
    let info = ToolErrorInfo {
        name: "AutoReviewDeniedError".to_string(),
        code: "AUTO_REVIEW_DENIED".to_string(),
        reason: Some(" raw\nreason ".to_string()),
    };
    let decision = PreToolDecision::deny_with_info("denied by policy", info.clone());
    let GateOutcome::Denied(result) = evaluate_pre_tool("echo", &decision) else {
        panic!("a denial must block the call");
    };
    assert!(result.is_error());
    assert_eq!(result.content(), &[text("Error: denied by policy")]);
    assert_eq!(result.error_info(), Some(&info));
    assert!(!result.concludes_turn(), "a failure never concludes a turn");

    // A reason-only denial carries no structured identity.
    let GateOutcome::Denied(plain) =
        evaluate_pre_tool("echo", &PreToolDecision::deny("denied by policy"))
    else {
        panic!("a denial must block the call");
    };
    assert_eq!(plain.error_info(), None);
    assert_eq!(plain.content(), &[text("Error: denied by policy")]);
}

/// `cancel` selects the canonical before-dispatch cancellation, the same
/// result the scheduler synthesizes for a call cancelled before dispatch.
#[test]
fn cancel_selects_the_canonical_before_dispatch_result() {
    let result = aborted_before_dispatch_result();
    assert!(result.is_error());
    assert_eq!(
        result.content(),
        &[text("Error: tool call aborted before dispatch")]
    );
    assert_eq!(
        result.error_info(),
        Some(&ToolErrorInfo {
            name: "AbortError".to_string(),
            code: TOOL_ABORTED_BEFORE_DISPATCH.to_string(),
            reason: None,
        })
    );
}

/// The guard stage is consulted only when the waterfall allowed: a waterfall
/// denial wins and the guard never runs.
#[test]
fn a_guard_never_runs_when_the_waterfall_already_decided() {
    struct WaterfallGate {
        consulted: Arc<Mutex<bool>>,
    }

    impl PreToolGate for WaterfallGate {
        fn decide(&self, _call: &ToolCallInput) -> PreToolDecision {
            PreToolDecision::deny("waterfall denial")
        }

        fn guard(&self, _call: &ToolCallInput) -> Option<String> {
            *self.consulted.lock().unwrap() = true;
            Some("guard denial".to_string())
        }
    }

    let consulted = Arc::new(Mutex::new(false));
    let gate = WaterfallGate {
        consulted: consulted.clone(),
    };
    let input = ToolCallInput {
        call_id: ToolCallId::new("c1"),
        name: "echo".to_string(),
        arguments: serde_json::json!({}),
    };
    let GateOutcome::Denied(result) = evaluate_pre_dispatch(&gate, &input) else {
        panic!("the waterfall denial must block the call");
    };
    assert_eq!(result.content(), &[text("Error: waterfall denial")]);
    assert!(!*consulted.lock().unwrap(), "the guard must not run");
}

// ---------------------------------------------------------------------------
// Wire-format fidelity
// ---------------------------------------------------------------------------

/// The pre-tool decision serializes to the exact TypeScript shapes.
#[test]
fn the_pre_tool_decision_wire_format_matches_the_typescript_shapes() {
    assert_eq!(
        serde_json::to_string(&PreToolDecision::allow()).unwrap(),
        r#"{"kind":"allow"}"#
    );
    assert_eq!(
        serde_json::to_string(&PreToolDecision::cancel()).unwrap(),
        r#"{"kind":"cancel"}"#
    );
    assert_eq!(
        serde_json::to_string(&PreToolDecision::deny("nope")).unwrap(),
        r#"{"kind":"deny","reason":"nope"}"#
    );
    assert_eq!(
        serde_json::to_string(&PreToolDecision::ask(None)).unwrap(),
        r#"{"kind":"ask"}"#
    );

    // Optional fields appear only when present, in the TypeScript order.
    let deny = serde_json::to_value(PreToolDecision::deny_with_info(
        "denied by policy",
        ToolErrorInfo {
            name: "AutoReviewDeniedError".to_string(),
            code: "AUTO_REVIEW_DENIED".to_string(),
            reason: Some(" raw\nreason ".to_string()),
        },
    ))
    .unwrap();
    assert_eq!(
        deny,
        serde_json::json!({
            "kind": "deny",
            "reason": "denied by policy",
            "info": {
                "name": "AutoReviewDeniedError",
                "code": "AUTO_REVIEW_DENIED",
                "reason": " raw\nreason ",
            },
        })
    );

    let ask = serde_json::to_value(PreToolDecision::Ask {
        reason: Some("hook wants a human".to_string()),
        display_reason: Some(DisplayReason {
            en: "Allow it?".to_string(),
            locales: [("zh".to_string(), "Allow it? (zh)".to_string())]
                .into_iter()
                .collect(),
        }),
    })
    .unwrap();
    assert_eq!(
        ask,
        serde_json::json!({
            "kind": "ask",
            "reason": "hook wants a human",
            "displayReason": { "en": "Allow it?", "zh": "Allow it? (zh)" },
        })
    );

    // Every shape round-trips.
    for decision in [
        PreToolDecision::allow(),
        PreToolDecision::cancel(),
        PreToolDecision::deny("nope"),
        PreToolDecision::ask(Some("needs approval")),
    ] {
        let wire = serde_json::to_string(&decision).unwrap();
        assert_eq!(
            serde_json::from_str::<PreToolDecision>(&wire).unwrap(),
            decision
        );
    }

    // A malformed known tag is an error, not a silent default.
    assert!(serde_json::from_str::<PreToolDecision>(r#"{"kind":"deny"}"#).is_err());
    assert!(serde_json::from_str::<PreToolDecision>(r#"{"kind":"escalate"}"#).is_err());
}

/// The post-tool decision serializes to the exact TypeScript shapes, and the
/// two accept projections stay mutually exclusive on the wire too.
#[test]
fn the_post_tool_decision_wire_format_matches_the_typescript_shapes() {
    assert_eq!(
        serde_json::to_string(&PostToolDecision::accept()).unwrap(),
        r#"{"kind":"accept"}"#
    );
    assert_eq!(
        serde_json::to_string(&PostToolDecision::accept_content(vec![text("rewritten")])).unwrap(),
        r#"{"kind":"accept","content":[{"type":"text","text":"rewritten"}]}"#
    );
    assert_eq!(
        serde_json::to_string(&PostToolDecision::accept_value(serde_json::json!(
            "replacement"
        )))
        .unwrap(),
        r#"{"kind":"accept","value":"replacement"}"#
    );
    assert_eq!(
        serde_json::to_string(&PostToolDecision::block(vec![text("rejected")])).unwrap(),
        r#"{"kind":"block","feedback":[{"type":"text","text":"rejected"}]}"#
    );

    // An explicit empty content list is a real replacement, not a dropped field.
    assert_eq!(
        serde_json::to_string(&PostToolDecision::accept_content(Vec::new())).unwrap(),
        r#"{"kind":"accept","content":[]}"#
    );
    // A null value is a real replacement too.
    assert_eq!(
        serde_json::to_string(&PostToolDecision::accept_value(serde_json::Value::Null)).unwrap(),
        r#"{"kind":"accept","value":null}"#
    );

    // Contexts ride under `additionalContexts`; the random message id is not
    // part of the assertion.
    let with_contexts = serde_json::to_value(
        PostToolDecision::accept_content(vec![text("rewritten")])
            .with_contexts(vec![user(&["fyi"])]),
    )
    .unwrap();
    assert_eq!(
        with_contexts,
        serde_json::json!({
            "kind": "accept",
            "content": [{ "type": "text", "text": "rewritten" }],
            "additionalContexts": [{
                "content": [{ "type": "text", "text": "fyi" }],
                "id": with_contexts["additionalContexts"][0]["id"].clone(),
                "source": { "kind": "user" },
            }],
        })
    );

    // Every shape round-trips.
    for decision in [
        PostToolDecision::accept(),
        PostToolDecision::accept_content(vec![text("rewritten")]),
        PostToolDecision::accept_value(serde_json::json!({ "ok": true })),
        PostToolDecision::block(vec![text("rejected")]),
    ] {
        let wire = serde_json::to_string(&decision).unwrap();
        assert_eq!(
            serde_json::from_str::<PostToolDecision>(&wire).unwrap(),
            decision
        );
    }

    // The runtime check the TypeScript union enforces at the type level.
    let both =
        serde_json::from_str::<PostToolDecision>(r#"{"kind":"accept","content":[],"value":1}"#)
            .unwrap_err();
    assert!(
        both.to_string()
            .contains("cannot replace both value and content"),
        "unexpected error: {both}"
    );
    assert!(serde_json::from_str::<PostToolDecision>(r#"{"kind":"block"}"#).is_err());
    assert!(serde_json::from_str::<PostToolDecision>(r#"{"kind":"escalate"}"#).is_err());
}

// ---------------------------------------------------------------------------
// Applying a post-execute decision
// ---------------------------------------------------------------------------

/// A block turns the call into a valueless failure whose content is the
/// corrective feedback, verbatim.
#[test]
fn a_block_turns_the_result_into_a_valueless_failure_with_feedback() {
    let dispatched =
        ToolExecutionResult::success(vec![text("body")]).with_value(serde_json::json!({}));
    let decision = PostToolDecision::block(vec![text("output rejected: try again")]);
    let blocked = apply_post_tool(&decision, &dispatched).unwrap();

    assert!(blocked.is_error());
    assert_eq!(blocked.content(), &[text("output rejected: try again")]);
    assert!(!blocked.concludes_turn());
    // The failure message is derived from the feedback blocks.
    if let ToolExecutionResult::Failure { error, .. } = &blocked {
        assert_eq!(error.message, "output rejected: try again");
    } else {
        panic!("a block must produce a failure");
    }
}

/// A block exposes only the contexts the decision supplied; the ones the body
/// deferred are discarded.
#[test]
fn a_block_discards_body_deferred_contexts_and_keeps_only_its_own() {
    let dispatched = ToolExecutionResult::success(vec![text("body")])
        .with_contexts(vec![user(&["body deferred"])]);
    let decision =
        PostToolDecision::block(vec![text("rejected")]).with_contexts(vec![user(&["why"])]);
    let blocked = apply_post_tool(&decision, &dispatched).unwrap();

    // The random message id is not part of the assertion: compare the content.
    let context_texts: Vec<Vec<ContentBlock>> = blocked
        .additional_contexts()
        .iter()
        .map(|context| context.content.clone())
        .collect();
    assert_eq!(context_texts, vec![vec![text("why")]]);
}

/// An accept replaces the content projection and keeps the call successful.
#[test]
fn an_accept_replaces_the_content_projection() {
    let dispatched =
        ToolExecutionResult::success(vec![text("body")]).with_value(serde_json::json!({ "a": 1 }));
    let accepted = apply_post_tool(
        &PostToolDecision::accept_content(vec![text("rewritten")]),
        &dispatched,
    )
    .unwrap();

    assert!(!accepted.is_error());
    assert_eq!(accepted.content(), &[text("rewritten")]);
    // The canonical value survives a content replacement.
    assert!(
        matches!(&accepted, ToolExecutionResult::Success { value, .. } if value == &serde_json::json!({ "a": 1 }))
    );
}

/// An accept cannot replace the value of a failed result.
#[test]
fn an_accept_cannot_replace_the_value_of_a_failed_result() {
    let failed = ToolExecutionResult::failure("boom");
    let error = apply_post_tool(
        &PostToolDecision::accept_value(serde_json::json!("replacement")),
        &failed,
    )
    .unwrap_err();
    assert_eq!(error, PostToolError::ValueReplacementOfFailure);
}

/// An accept replaces the canonical value of a success, appending the
/// decision's contexts after the result's own.
#[test]
fn an_accept_replaces_the_canonical_value_and_appends_its_contexts() {
    let dispatched = ToolExecutionResult::success(vec![text("body")])
        .with_value(serde_json::json!("original"))
        .with_contexts(vec![user(&["body deferred"])]);
    let decision = PostToolDecision::accept_value(serde_json::json!("replacement"))
        .with_contexts(vec![user(&["decision context"])]);
    let accepted = apply_post_tool(&decision, &dispatched).unwrap();

    assert!(!accepted.is_error());
    assert!(
        matches!(&accepted, ToolExecutionResult::Success { value, .. } if value == &serde_json::json!("replacement"))
    );
    assert_eq!(accepted.content(), &[text("body")]);
    let context_texts: Vec<Vec<ContentBlock>> = accepted
        .additional_contexts()
        .iter()
        .map(|context| context.content.clone())
        .collect();
    assert_eq!(
        context_texts,
        vec![vec![text("body deferred")], vec![text("decision context")]]
    );
}

/// The failure message of a block renders text blocks and placeholders.
#[test]
fn failure_message_from_content_renders_text_blocks_and_placeholders() {
    assert_eq!(
        failure_message_from_content(&[text("first"), text("second")]),
        "first\nsecond"
    );
    assert_eq!(
        failure_message_from_content(&[text("note"), ContentBlock::reasoning("hidden")]),
        "note\n[reasoning content]"
    );
    assert_eq!(
        failure_message_from_content(&[]),
        "tool result blocked by post-execute policy"
    );
}

// ---------------------------------------------------------------------------
// The restriction filter
// ---------------------------------------------------------------------------

/// An allow-list admits only the names it lists.
#[test]
fn an_allow_list_admits_only_the_names_it_lists() {
    let filter = ToolRestriction::new(Some(vec!["read".to_string(), "bash".to_string()]), None);
    assert!(filter.admits("read"));
    assert!(filter.admits("bash"));
    assert!(!filter.admits("web"));
}

/// A deny-list removes only the names it lists.
#[test]
fn a_deny_list_removes_only_the_names_it_lists() {
    let filter = ToolRestriction::new(None, Some(vec!["bash".to_string()]));
    assert!(!filter.admits("bash"));
    assert!(filter.admits("read"));
    assert!(filter.admits("web"));
}

/// Both lists compose: an allow-list bounds the surface, a deny-list carves
/// holes in it.
#[test]
fn allow_and_deny_lists_compose_within_one_filter() {
    let filter = ToolRestriction::new(
        Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        Some(vec!["b".to_string()]),
    );
    assert!(filter.admits("a"));
    assert!(!filter.admits("b"));
    assert!(
        !filter.admits("d"),
        "an unlisted name never passes an allow-list"
    );
}

/// Restrictions intersect across the chain: one filter may mask what another
/// admits.
#[test]
fn restrictions_intersect_across_the_chain() {
    let filters = [
        ToolRestriction::new(Some(vec!["a".to_string(), "b".to_string()]), None),
        ToolRestriction::new(None, Some(vec!["b".to_string()])),
    ];
    assert!(admits_all(&filters, "a"));
    assert!(!admits_all(&filters, "b"));
    assert!(admits_all(&[], "anything"), "no filters admit everything");
}

/// A filter snapshots its name sets at registration: later mutation of the
/// caller's arrays is not observed.
#[test]
fn a_restriction_snapshots_its_name_sets_at_compile_time() {
    let mut deny = vec!["a".to_string()];
    let filter = ToolRestriction::new(None, Some(deny.clone()));
    deny.push("b".to_string());
    assert!(!filter.admits("a"));
    assert!(
        filter.admits("b"),
        "the compiled filter must not observe the mutation"
    );
}

/// An empty allow-list admits nothing; an empty deny-list removes nothing.
#[test]
fn empty_name_lists_are_real_answers_not_absent_filters() {
    let allow_none = ToolRestriction::new(Some(Vec::new()), None);
    assert!(!allow_none.admits("a"));

    let deny_none = ToolRestriction::new(None, Some(Vec::new()));
    assert!(deny_none.admits("a"));
}

/// Registration-time validation fails loud on an empty filter.
#[test]
fn validate_restriction_rejects_an_empty_filter() {
    let known: BTreeSet<String> = ["read".to_string()].into_iter().collect();
    assert_eq!(
        validate_restriction(None, None, &known),
        Err(RestrictionError::EmptyFilter)
    );
    assert!(validate_restriction(Some(&[]), None, &known).is_ok());
}

/// Registration-time validation names the unknown tools and the known ones.
#[test]
fn validate_restriction_rejects_unknown_names_with_the_known_list() {
    let known: BTreeSet<String> = ["bash".to_string(), "read".to_string()]
        .into_iter()
        .collect();
    assert_eq!(
        validate_restriction(None, Some(&["ghost".to_string()]), &known),
        Err(RestrictionError::UnknownTools {
            plural: "",
            names: "\"ghost\"".to_string(),
            known: "bash, read".to_string(),
        })
    );
    // Plurals, caller order, and duplicates follow the TypeScript message.
    assert_eq!(
        validate_restriction(
            Some(&["zeta".to_string(), "alpha".to_string()]),
            Some(&["ghost".to_string(), "alpha".to_string()]),
            &known,
        ),
        Err(RestrictionError::UnknownTools {
            plural: "s",
            names: "\"zeta\", \"alpha\", \"ghost\", \"alpha\"".to_string(),
            known: "bash, read".to_string(),
        })
    );
    // An empty registry reports `(none)`.
    assert_eq!(
        validate_restriction(None, Some(&["ghost".to_string()]), &BTreeSet::new()),
        Err(RestrictionError::UnknownTools {
            plural: "",
            names: "\"ghost\"".to_string(),
            known: "(none)".to_string(),
        })
    );
    assert!(validate_restriction(Some(&["read".to_string()]), None, &known).is_ok());
}

// ---------------------------------------------------------------------------
// The gated dispatch path
// ---------------------------------------------------------------------------

/// A denying gate never runs the body, yet the call still settles with a
/// paired `tool/call` + `tool/result` carrying the denial.
#[tokio::test]
async fn a_denying_gate_never_runs_the_body_and_still_settles_the_call() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "bash",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));

    let mut session = session();
    let cancel = CancelSignal::new();
    let gate = DenyGate::new(vec!["bash".to_string()], seen.clone());
    let outcome = execute_tool_calls_gated(
        &mut session,
        &schedule(&registry, &cancel),
        &[
            call("c1", "bash", r#"{"text":"rm -rf"}"#),
            call("c2", "bash", r#"{"text":"ls"}"#),
        ],
        &gate,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 2);
    assert!(
        !outcome.aborted,
        "a gated denial is a failure, not an abort"
    );
    assert!(
        log.lock().unwrap().is_empty(),
        "no body may run under a denial"
    );
    assert_eq!(
        *seen.lock().unwrap(),
        vec!["c1", "c2"],
        "the gate sees every call"
    );

    let results = committed_results(&session);
    assert_eq!(results.len(), 2, "every model call settles");
    assert_eq!(results[0].call_id, "c1");
    assert_eq!(
        results[0].texts,
        vec!["Error: bash is not permitted".to_string()]
    );
    assert_eq!(
        results[0].error, None,
        "a reason-only denial carries no structured identity"
    );
    assert_eq!(results[1].call_id, "c2");
    assert_eq!(
        results[1].texts,
        vec!["Error: bash is not permitted".to_string()]
    );
}

/// A denial with a structured identity persists it beside the model-facing
/// text, and the failure never concludes the turn.
#[tokio::test]
async fn a_denial_with_structured_identity_persists_the_identity() {
    struct InfoGate;

    impl PreToolGate for InfoGate {
        fn decide(&self, call: &ToolCallInput) -> PreToolDecision {
            if call.name == "echo" {
                return PreToolDecision::deny_with_info(
                    "denied by policy",
                    ToolErrorInfo {
                        name: "AutoReviewDeniedError".to_string(),
                        code: "AUTO_REVIEW_DENIED".to_string(),
                        reason: Some(" raw\nreason ".to_string()),
                    },
                );
            }
            PreToolDecision::Allow
        }
    }

    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));
    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls_gated(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "echo", r#"{"text":"hi"}"#)],
        &InfoGate,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 1);
    assert!(!outcome.concluded);
    assert!(log.lock().unwrap().is_empty());
    let results = committed_results(&session);
    assert_eq!(
        results[0].texts,
        vec!["Error: denied by policy".to_string()]
    );
    assert_eq!(
        results[0].error,
        Some((
            "AutoReviewDeniedError".to_string(),
            "AUTO_REVIEW_DENIED".to_string()
        ))
    );
}

/// A cancelling gate selects the canonical before-dispatch result without
/// running the body, and the call still settles.
#[tokio::test]
async fn a_cancelling_gate_selects_the_canonical_result_and_never_runs_the_body() {
    struct CancelGate;

    impl PreToolGate for CancelGate {
        fn decide(&self, _call: &ToolCallInput) -> PreToolDecision {
            PreToolDecision::cancel()
        }
    }

    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));
    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls_gated(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "echo", r#"{"text":"hi"}"#)],
        &CancelGate,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 1);
    assert!(!outcome.aborted);
    assert!(log.lock().unwrap().is_empty());
    let results = committed_results(&session);
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].texts,
        vec!["Error: tool call aborted before dispatch".to_string()]
    );
    assert_eq!(
        results[0].error,
        Some((
            "AbortError".to_string(),
            TOOL_ABORTED_BEFORE_DISPATCH.to_string()
        ))
    );
}

/// A gated denial is an ordinary failed result: it does not abort the step,
/// and later calls still run and commit in model order.
#[tokio::test]
async fn a_gated_denial_is_a_failure_not_an_abort_and_later_calls_still_run() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "bash",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));

    let mut session = session();
    let cancel = CancelSignal::new();
    let gate = DenyGate::new(vec!["bash".to_string()], seen);
    let outcome = execute_tool_calls_gated(
        &mut session,
        &schedule(&registry, &cancel),
        &[
            call("c1", "bash", r#"{"text":"rm"}"#),
            call("c2", "echo", r#"{"text":"ok"}"#),
            call("c3", "bash", r#"{"text":"rm"}"#),
            call("c4", "echo", r#"{"text":"ok"}"#),
        ],
        &gate,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 4);
    assert!(!outcome.aborted);
    let mut bodies = log.lock().unwrap().clone();
    bodies.sort();
    assert_eq!(
        bodies,
        vec!["c2", "c4"],
        "only the allowed calls run a body"
    );
    let results = committed_results(&session);
    let call_ids: Vec<&str> = results.iter().map(|r| r.call_id.as_str()).collect();
    assert_eq!(
        call_ids,
        vec!["c1", "c2", "c3", "c4"],
        "model order never reorders"
    );
    assert_eq!(
        results[0].texts,
        vec!["Error: bash is not permitted".to_string()]
    );
    assert_eq!(results[1].texts, vec!["echo:ok".to_string()]);
    assert_eq!(
        results[2].texts,
        vec!["Error: bash is not permitted".to_string()]
    );
    assert_eq!(results[3].texts, vec!["echo:ok".to_string()]);
}

/// The gate observes every name that reaches the registry, so a denial
/// outranks the dispatch-stage unknown-tool failure.
#[tokio::test]
async fn a_gate_denial_outranks_the_unknown_tool_failure() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let registry = ToolRegistry::new();
    let mut session = session();
    let cancel = CancelSignal::new();
    let gate = DenyGate::new(vec!["ghost".to_string()], seen.clone());
    let outcome = execute_tool_calls_gated(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "ghost", "{}")],
        &gate,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 1);
    assert_eq!(*seen.lock().unwrap(), vec!["c1"]);
    let results = committed_results(&session);
    assert_eq!(
        results[0].texts,
        vec!["Error: ghost is not permitted".to_string()]
    );
    assert_eq!(results[0].error, None, "not the unknown-tool identity");
}

/// A monotonic guard denies after an allowing waterfall decision, and the
/// body never runs.
#[tokio::test]
async fn a_guard_denial_stops_the_body_after_an_allowing_waterfall() {
    struct GuardGate;

    impl PreToolGate for GuardGate {
        fn decide(&self, _call: &ToolCallInput) -> PreToolDecision {
            PreToolDecision::Allow
        }

        fn guard(&self, _call: &ToolCallInput) -> Option<String> {
            Some("terminal policy".to_string())
        }
    }

    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));
    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls_gated(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "echo", r#"{"text":"hi"}"#)],
        &GuardGate,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 1);
    assert!(!outcome.aborted);
    assert!(log.lock().unwrap().is_empty());
    let results = committed_results(&session);
    assert_eq!(results[0].texts, vec!["Error: terminal policy".to_string()]);
}

/// An `ask` degrades to a denial — with the ask's reason when it has one, and
/// with the default message when it does not — without running the body.
#[tokio::test]
async fn an_ask_gate_degrades_to_a_denial_without_running_the_body() {
    struct AskGate {
        reason: Option<String>,
    }

    impl PreToolGate for AskGate {
        fn decide(&self, _call: &ToolCallInput) -> PreToolDecision {
            PreToolDecision::ask(self.reason.as_deref())
        }
    }

    async fn run(reason: Option<String>) -> CommittedResult {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool::new(
            "echo",
            ToolExecutionMode::Parallel,
            log.clone(),
        )));
        let mut session = session();
        let cancel = CancelSignal::new();
        execute_tool_calls_gated(
            &mut session,
            &schedule(&registry, &cancel),
            &[call("c1", "echo", r#"{"text":"hi"}"#)],
            &AskGate { reason },
            &mut |_| {},
        )
        .await
        .unwrap();
        assert!(
            log.lock().unwrap().is_empty(),
            "no body may run under an ask"
        );
        committed_results(&session).pop().unwrap()
    }

    assert_eq!(
        run(Some("needs approval".to_string())).await.texts,
        vec!["Error: needs approval".to_string()]
    );
    assert_eq!(
        run(None).await.texts,
        vec!["Error: tool \"echo\" requires approval (not yet supported)".to_string()]
    );
}

/// The gate sees the parsed arguments, so a policy may deny on their content.
#[tokio::test]
async fn the_gate_sees_the_parsed_arguments() {
    struct ArgumentGate;

    impl PreToolGate for ArgumentGate {
        fn decide(&self, call: &ToolCallInput) -> PreToolDecision {
            match argument(call, "text").and_then(|value| value.as_str()) {
                Some("forbidden") => PreToolDecision::deny("that text is forbidden"),
                _ => PreToolDecision::Allow,
            }
        }
    }

    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "echo",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));
    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls_gated(
        &mut session,
        &schedule(&registry, &cancel),
        &[
            call("c1", "echo", r#"{"text":"forbidden"}"#),
            call("c2", "echo", r#"{"text":"fine"}"#),
        ],
        &ArgumentGate,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 2);
    assert_eq!(*log.lock().unwrap(), vec!["c2"]);
    let results = committed_results(&session);
    assert_eq!(
        results[0].texts,
        vec!["Error: that text is forbidden".to_string()]
    );
    assert_eq!(results[1].texts, vec!["echo:fine".to_string()]);
}

/// A compiled restriction composes with a gate: a masked name is denied by
/// policy and settles as a failure.
#[tokio::test]
async fn a_restriction_composes_with_the_gate_to_mask_a_tool() {
    struct RestrictingGate {
        filters: Vec<ToolRestriction>,
    }

    impl PreToolGate for RestrictingGate {
        fn decide(&self, call: &ToolCallInput) -> PreToolDecision {
            if admits_all(&self.filters, &call.name) {
                PreToolDecision::Allow
            } else {
                PreToolDecision::deny(format!("{} is not available in this scope", call.name))
            }
        }
    }

    let log = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(
        "bash",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));
    registry.register(Arc::new(EchoTool::new(
        "read",
        ToolExecutionMode::Parallel,
        log.clone(),
    )));
    let gate = RestrictingGate {
        filters: vec![ToolRestriction::new(None, Some(vec!["bash".to_string()]))],
    };

    let mut session = session();
    let cancel = CancelSignal::new();
    let outcome = execute_tool_calls_gated(
        &mut session,
        &schedule(&registry, &cancel),
        &[call("c1", "bash", "{}"), call("c2", "read", "{}")],
        &gate,
        &mut |_| {},
    )
    .await
    .unwrap();

    assert_eq!(outcome.settled, 2);
    assert_eq!(*log.lock().unwrap(), vec!["c2"]);
    let results = committed_results(&session);
    assert_eq!(
        results[0].texts,
        vec!["Error: bash is not available in this scope".to_string()]
    );
    assert_eq!(results[1].texts, vec!["read:".to_string()]);
}

/// A gate-denied exclusive call still forms its own group: the barrier
/// grouping reads the registry's mode, never the gate's answer.
///
/// What is compared against the ungated run is the **call sequence and its
/// pairing**, not the committed bytes: the denied call's result text is
/// necessarily different from the allowed one, so only the identities line up.
#[tokio::test]
async fn an_exclusive_barrier_still_settles_a_gated_call_in_model_order() {
    async fn run(gate: &dyn PreToolGate) -> (Vec<String>, Vec<CommittedResult>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool::new(
            "echo",
            ToolExecutionMode::Parallel,
            log.clone(),
        )));
        registry.register(Arc::new(EchoTool::new(
            "barrier",
            ToolExecutionMode::Exclusive,
            log.clone(),
        )));
        let mut session = session();
        let cancel = CancelSignal::new();
        execute_tool_calls_gated(
            &mut session,
            &schedule(&registry, &cancel),
            &[
                call("c1", "echo", r#"{"text":"a"}"#),
                call("c2", "barrier", r#"{"text":"b"}"#),
                call("c3", "echo", r#"{"text":"c"}"#),
            ],
            gate,
            &mut |_| {},
        )
        .await
        .unwrap();
        let bodies = log.lock().unwrap().clone();
        (bodies, committed_results(&session))
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let (bodies, results) = run(&DenyGate::new(vec!["barrier".to_string()], seen)).await;
    assert_eq!(
        bodies,
        vec!["c1", "c3"],
        "the denied barrier never runs a body"
    );
    let call_ids: Vec<&str> = results.iter().map(|r| r.call_id.as_str()).collect();
    assert_eq!(call_ids, vec!["c1", "c2", "c3"]);
    assert_eq!(
        results[1].texts,
        vec!["Error: barrier is not permitted".to_string()]
    );

    // With an allow-all gate the same calls commit in the same order, with the
    // same pairing — the texts differ, only the identities line up.
    let (allowed_bodies, allowed_results) = run(&AllowAllGate).await;
    assert_eq!(allowed_bodies, vec!["c1", "c2", "c3"]);
    assert_eq!(
        allowed_results
            .iter()
            .map(|r| r.call_id.clone())
            .collect::<Vec<_>>(),
        results
            .iter()
            .map(|r| r.call_id.clone())
            .collect::<Vec<_>>()
    );
}

/// An allow-all gate is indistinguishable from the ungated scheduler.
#[tokio::test]
async fn an_allow_all_gate_is_indistinguishable_from_the_ungated_scheduler() {
    async fn run(gated: bool) -> (Vec<String>, Vec<CommittedResult>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool::new(
            "echo",
            ToolExecutionMode::Parallel,
            log.clone(),
        )));
        registry.register(Arc::new(EchoTool::new(
            "barrier",
            ToolExecutionMode::Exclusive,
            log.clone(),
        )));
        let mut session = session();
        let cancel = CancelSignal::new();
        let calls = [
            call("c1", "echo", r#"{"text":"a"}"#),
            call("c2", "barrier", r#"{"text":"b"}"#),
            call("c3", "ghost", "{}"),
            call("c4", "echo", r#"{"text":"d"}"#),
        ];
        if gated {
            execute_tool_calls_gated(
                &mut session,
                &schedule(&registry, &cancel),
                &calls,
                &AllowAllGate,
                &mut |_| {},
            )
            .await
            .unwrap();
        } else {
            execute_tool_calls(
                &mut session,
                &schedule(&registry, &cancel),
                &calls,
                &mut |_| {},
            )
            .await
            .unwrap();
        }
        let bodies = log.lock().unwrap().clone();
        (bodies, committed_results(&session))
    }

    let (plain_bodies, plain_results) = run(false).await;
    let (gated_bodies, gated_results) = run(true).await;
    assert_eq!(plain_bodies, gated_bodies);
    assert_eq!(plain_results, gated_results);
    assert_eq!(
        plain_bodies,
        vec!["c1", "c2", "c4"],
        "the ghost never runs a body"
    );
}
