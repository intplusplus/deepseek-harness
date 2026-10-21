//! The `exit_plan_mode` tool and `/plan` command, ported from the matching
//! `describe` blocks in `packages/plan/plan-mode/tests/plan-mode.spec.ts`.
//!
//! The review itself is a host interaction, so these tests drive its pure
//! edges: the pre-ask checks, the question and intent handed to the reviewer,
//! the answer classification, the post-review decision, the presenters, and
//! the command copy. Exact bytes are asserted wherever the TypeScript side
//! pins model-visible or user-visible output.

use dsh_llm::{ContentBlock, CoreContentBlock, ToolCallId};
use dsh_plan::{
    classify_review, command_metadata, decide_exit, first_heading, output_schema, parameter_schema,
    plan_review_question, present_call, present_result, require_review_channel, run_plan_command,
    valid_plan_heading, validate_exit_plan, ExitPlanModeError, ExitPlanModeOutput, ExitReview,
    PlanCommandKind, PlanMachine, PlanReviewQuestion, PlanSetDecision, PlanSetOutcome,
    PlanUnitState, ReviewAnswer, ReviewAnswerItem, APPROVE_LABEL, APPROVE_OPTION_DESCRIPTION,
    EXIT_APPROVED_TEXT, EXIT_DESCRIPTION, EXIT_DISMISSED_TEXT, EXIT_KEEP_PLANNING_TEXT,
    EXIT_PLAN_ARGUMENT_DESCRIPTION, EXIT_PLAN_MODE, KEEP_PLANNING_LABEL,
    KEEP_PLANNING_OPTION_DESCRIPTION, PLAN_ALREADY_INACTIVE_TEXT, PLAN_ATTACHMENTS_OFF_TEXT,
    PLAN_COMMAND_DEFINITION_ID, PLAN_COMMAND_DESCRIPTION, PLAN_COMMAND_HINT, PLAN_COMMAND_NAME,
    PLAN_ENTERING_TEXT, PLAN_ENTRY_CANCELLED_TEXT, PLAN_LEAVING_TEXT, PLAN_OFF_TEXT, PLAN_ON_TEXT,
    PLAN_REVIEW_INTENT, REVIEW_HEADER, REVIEW_ID, REVIEW_QUESTION,
};
use dsh_tools::ToolExecutionResult;
use serde_json::json;

fn content_text(content: &[ContentBlock]) -> String {
    match content {
        [ContentBlock::Core(CoreContentBlock::Text { text })] => text.clone(),
        other => format!("{other:?}"),
    }
}

fn result_text(result: &ToolExecutionResult) -> String {
    content_text(result.content())
}

fn answer(selected: &[&str], custom: Option<&str>) -> ReviewAnswer {
    ReviewAnswer {
        answers: vec![ReviewAnswerItem {
            id: REVIEW_ID.to_string(),
            selected: selected.iter().map(|label| label.to_string()).collect(),
            custom: custom.map(str::to_string),
        }],
    }
}

fn item(id: &str, selected: &[&str], custom: Option<&str>) -> ReviewAnswerItem {
    ReviewAnswerItem {
        id: id.to_string(),
        selected: selected.iter().map(|label| label.to_string()).collect(),
        custom: custom.map(str::to_string),
    }
}

fn success_value(result: &ToolExecutionResult) -> String {
    match result {
        ToolExecutionResult::Success { value, .. } => value.to_string(),
        ToolExecutionResult::Failure { .. } => panic!("expected a successful result"),
    }
}

#[test]
fn the_tool_description_and_schemas_are_the_typescript_bytes() {
    assert_eq!(EXIT_PLAN_MODE, "exit_plan_mode");
    assert!(EXIT_DESCRIPTION.starts_with("Use only in plan mode."));
    assert_eq!(
        EXIT_DESCRIPTION,
        concat!(
            "Use only in plan mode. Present your plan for the user's review and, on approval, ",
            "leave plan mode. The user may approve (carry out the plan from your next step) or ",
            "keep planning — their feedback comes back in the tool result; revise and present ",
            "again."
        )
    );
    assert_eq!(
        parameter_schema().to_string(),
        concat!(
            r#"{"type":"object","properties":{"plan":{"type":"string","description":"#,
            r#""The complete plan, as markdown, starting with a # heading that names it."}},"#,
            r#""required":["plan"]}"#
        )
    );
    assert_eq!(
        EXIT_PLAN_ARGUMENT_DESCRIPTION,
        "The complete plan, as markdown, starting with a # heading that names it."
    );
    assert_eq!(
        output_schema().to_string(),
        concat!(
            r#"{"type":"object","additionalProperties":false,"properties":"#,
            r#"{"approved":{"type":"boolean"}},"required":["approved"]}"#
        )
    );
    assert_eq!(
        ExitPlanModeOutput { approved: true }.to_value().to_string(),
        r#"{"approved":true}"#
    );
    assert_eq!(
        ExitPlanModeOutput { approved: false }
            .to_value()
            .to_string(),
        r#"{"approved":false}"#
    );
}

#[test]
fn the_pre_ask_checks_run_in_typescript_order() {
    assert_eq!(
        validate_exit_plan(false, false, ""),
        Err(ExitPlanModeError::NoCallingAgent)
    );
    assert_eq!(
        validate_exit_plan(true, false, ""),
        Err(ExitPlanModeError::NotInPlanMode)
    );
    assert_eq!(
        validate_exit_plan(true, true, ""),
        Err(ExitPlanModeError::InvalidPlan)
    );
    assert_eq!(validate_exit_plan(true, true, "# The plan"), Ok(()));

    assert_eq!(
        ExitPlanModeError::NoCallingAgent.to_string(),
        "exit_plan_mode requires a calling agent (no session to switch)"
    );
    assert_eq!(
        ExitPlanModeError::NotInPlanMode.to_string(),
        "exit_plan_mode is only available in plan mode"
    );
    assert_eq!(
        ExitPlanModeError::InvalidPlan.to_string(),
        "exit_plan_mode requires a non-empty markdown plan starting with a # heading"
    );
    assert_eq!(
        require_review_channel(false),
        Err(ExitPlanModeError::NoReviewChannel)
    );
    assert_eq!(require_review_channel(true), Ok(()));
    assert_eq!(
        ExitPlanModeError::NoReviewChannel.to_string(),
        "no user-questions channel is available to review the plan; ask the user to switch the session mode instead"
    );
    assert_eq!(
        ExitPlanModeError::Reloaded.to_string(),
        "the plan-mode service was reloaded while the plan was under review; present the plan again"
    );
}

#[test]
fn only_a_single_hash_heading_satisfies_the_exit_rule() {
    for plan in [
        "# The plan",
        "  # The plan  ",
        "#\tPlan",
        "#\u{FEFF}Plan",
        "#\nPlan",
    ] {
        assert!(
            valid_plan_heading(plan),
            "expected a valid heading: {plan:?}"
        );
    }
    for plan in [
        "",
        "   ",
        "do things",
        "## Plan",
        "#",
        "# ",
        "#\u{0085}Plan",
        "\u{0085}# Plan",
    ] {
        assert!(
            !valid_plan_heading(plan),
            "expected an invalid heading: {plan:?}"
        );
    }
}

#[test]
fn the_review_question_carries_the_plan_options_and_intent() {
    let call_id = ToolCallId::new("call-1");
    let question = plan_review_question("# The plan\n\nbody", &call_id);
    assert_eq!(question.id, REVIEW_ID);
    assert_eq!(question.header, REVIEW_HEADER);
    assert_eq!(question.question, REVIEW_QUESTION);
    assert_eq!(question.detail, "# The plan\n\nbody");
    assert_eq!(
        question
            .options
            .iter()
            .map(|option| option.label.as_str())
            .collect::<Vec<_>>(),
        [APPROVE_LABEL, KEEP_PLANNING_LABEL]
    );
    assert_eq!(question.options[0].description, APPROVE_OPTION_DESCRIPTION);
    assert_eq!(
        question.options[1].description,
        KEEP_PLANNING_OPTION_DESCRIPTION
    );
    assert_eq!(question.intent.kind, PLAN_REVIEW_INTENT);
    assert_eq!(question.intent.approve, APPROVE_LABEL);
    assert_eq!(question.intent.call_id.as_str(), "call-1");

    assert_eq!(
        serde_json::to_string(&question).expect("the question serializes"),
        concat!(
            r##"{"id":"plan-review","header":"Plan review","question":"Approve this plan and leave plan mode?","##,
            r##""detail":"# The plan\n\nbody","options":[{"label":"Approve","description":"##,
            r##""Leave plan mode; the plan is carried out from the next step."},"##,
            r##"{"label":"Keep planning","description":"Stay in plan mode; feedback goes back to the model."}],"##,
            r##""intent":{"kind":"plan-review","approve":"Approve","callId":"call-1"}}"##
        )
    );
}

#[test]
fn only_exactly_one_approve_selection_without_text_is_consent() {
    assert_eq!(
        classify_review(&answer(&[APPROVE_LABEL], None)),
        ExitReview::Approved
    );
    assert_eq!(
        classify_review(&answer(&[APPROVE_LABEL, KEEP_PLANNING_LABEL], None)),
        ExitReview::KeepPlanning {
            feedback: String::new()
        }
    );
    assert_eq!(
        classify_review(&answer(&[], None)),
        ExitReview::KeepPlanning {
            feedback: String::new()
        }
    );
    assert_eq!(
        classify_review(&answer(&[KEEP_PLANNING_LABEL], None)),
        ExitReview::KeepPlanning {
            feedback: String::new()
        }
    );
    assert_eq!(
        classify_review(&answer(&[APPROVE_LABEL], Some("change the tests"))),
        ExitReview::KeepPlanning {
            feedback: "change the tests".to_string()
        }
    );
    assert_eq!(
        classify_review(&answer(&[], Some("add tests first"))),
        ExitReview::KeepPlanning {
            feedback: "add tests first".to_string()
        }
    );
    assert_eq!(
        classify_review(&answer(&[APPROVE_LABEL], Some(""))),
        ExitReview::KeepPlanning {
            feedback: String::new()
        }
    );
}

#[test]
fn duplicate_or_foreign_answer_items_are_not_consent() {
    let duplicate = ReviewAnswer {
        answers: vec![
            item(REVIEW_ID, &[APPROVE_LABEL], None),
            item(REVIEW_ID, &[KEEP_PLANNING_LABEL], None),
        ],
    };
    assert_eq!(
        classify_review(&duplicate),
        ExitReview::KeepPlanning {
            feedback: String::new()
        }
    );

    let missing = ReviewAnswer {
        answers: Vec::new(),
    };
    assert_eq!(
        classify_review(&missing),
        ExitReview::KeepPlanning {
            feedback: String::new()
        }
    );

    let foreign = ReviewAnswer {
        answers: vec![item("another-question", &[APPROVE_LABEL], None)],
    };
    assert_eq!(
        classify_review(&foreign),
        ExitReview::KeepPlanning {
            feedback: String::new()
        }
    );
}

#[test]
fn an_approved_exit_parks_a_silent_boundary_selection() {
    let decision = decide_exit(&ExitReview::Approved, false);
    assert!(!decision.result.is_error());
    assert_eq!(success_value(&decision.result), r#"{"approved":true}"#);
    assert_eq!(result_text(&decision.result), EXIT_APPROVED_TEXT);
    assert_eq!(decision.pending, Some(false));

    let reloaded = decide_exit(&ExitReview::Approved, true);
    assert!(reloaded.result.is_error());
    assert_eq!(
        result_text(&reloaded.result),
        "Error: the plan-mode service was reloaded while the plan was under review; present the plan again"
    );
    assert_eq!(reloaded.pending, None);
}

#[test]
fn a_dismissed_review_concludes_the_turn_with_a_success() {
    let decision = decide_exit(&ExitReview::Dismissed, false);
    assert!(!decision.result.is_error());
    assert!(decision.result.concludes_turn());
    assert_eq!(success_value(&decision.result), r#"{"approved":false}"#);
    assert_eq!(result_text(&decision.result), EXIT_DISMISSED_TEXT);
    assert_eq!(decision.pending, None);
}

#[test]
fn keep_planning_is_a_corrective_error_with_the_feedback_verbatim() {
    assert_eq!(
        ExitPlanModeError::KeepPlanning.to_string(),
        EXIT_KEEP_PLANNING_TEXT
    );
    let generic = decide_exit(
        &ExitReview::KeepPlanning {
            feedback: String::new(),
        },
        false,
    );
    assert!(generic.result.is_error());
    assert_eq!(
        result_text(&generic.result),
        "Error: The user chose to keep planning; revise the plan and present it again."
    );
    assert_eq!(generic.pending, None);

    let feedback = decide_exit(
        &ExitReview::KeepPlanning {
            feedback: "consider the resume path".to_string(),
        },
        false,
    );
    assert!(feedback.result.is_error());
    assert_eq!(
        result_text(&feedback.result),
        "Error: The user chose to keep planning; their feedback: consider the resume path"
    );
    assert_eq!(
        ExitPlanModeError::keep_planning("consider the resume path").to_string(),
        "The user chose to keep planning; their feedback: consider the resume path"
    );
    assert_eq!(
        ExitPlanModeError::keep_planning(""),
        ExitPlanModeError::KeepPlanning
    );
}

#[test]
fn every_other_review_failure_keeps_its_message() {
    let decision = decide_exit(
        &ExitReview::Failed {
            message: "ask_user_question was aborted before the user answered".to_string(),
        },
        false,
    );
    assert!(decision.result.is_error());
    assert_eq!(
        result_text(&decision.result),
        "Error: ask_user_question was aborted before the user answered"
    );
    assert_eq!(decision.pending, None);
}

#[test]
fn the_first_heading_accepts_any_level_and_trims() {
    assert_eq!(
        first_heading("## Fix the flake\n\nsteps"),
        Some("Fix the flake".to_string())
    );
    assert_eq!(first_heading("no heading here"), None);
    assert_eq!(
        first_heading("###   Deep  title  "),
        Some("Deep  title".to_string())
    );
    assert_eq!(first_heading("# "), None);
    assert_eq!(first_heading("#  "), Some(" ".to_string()));
    assert_eq!(first_heading("#\t\t"), Some("\t".to_string()));
    assert_eq!(first_heading("#\u{FEFF}title"), Some("title".to_string()));
    assert_eq!(first_heading("###### six"), Some("six".to_string()));
    assert_eq!(first_heading("####### seven"), None);
    assert_eq!(first_heading("#\u{0085}title"), None);
}

#[test]
fn the_presenters_are_generic_cards() {
    assert_eq!(
        serde_json::to_string(&present_call("## Fix the flake\n\nsteps"))
            .expect("the call card serializes"),
        concat!(
            r###"{"card":"generic","title":"Fix the flake","kind":"other","###,
            r###""content":[{"type":"text","text":"## Fix the flake\n\nsteps"}]}"###
        )
    );
    assert_eq!(present_call("no heading here").title, "Plan");
    assert_eq!(
        serde_json::to_string(&present_result(&[ContentBlock::text("ok")]))
            .expect("the result card serializes"),
        r#"{"card":"generic","title":"Plan review","content":[{"type":"text","text":"ok"}]}"#
    );
}

#[test]
fn the_command_metadata_is_the_registered_definition() {
    let metadata = command_metadata();
    assert_eq!(metadata.definition_id, PLAN_COMMAND_DEFINITION_ID);
    assert_eq!(metadata.name, PLAN_COMMAND_NAME);
    assert_eq!(metadata.description, PLAN_COMMAND_DESCRIPTION);
    assert_eq!(metadata.input.hint, PLAN_COMMAND_HINT);
    assert!(metadata.input.attachments);
    assert_eq!(
        serde_json::to_string(&metadata).expect("the metadata serializes"),
        concat!(
            r#"{"definitionId":"@deepseek-ai/dsh-plan-mode","name":"plan","#,
            r#""description":"Enter or leave plan mode","input":"#,
            r#"{"hint":"[off|message]","attachments":true}}"#
        )
    );
}

#[test]
fn an_idle_command_commits_and_an_open_turn_queues() {
    let mut idle = PlanMachine::new(PlanUnitState::default(), false);
    let entering = run_plan_command(&mut idle, "", false);
    assert_eq!(entering.result.kind, PlanCommandKind::Success);
    assert_eq!(entering.result.text, PLAN_ON_TEXT);
    assert_eq!(entering.steer, None);
    assert_eq!(
        entering.set,
        PlanSetDecision::Commit {
            active: true,
            narration: None
        }
    );
    idle.commit_set(&entering.set);
    assert!(idle.active());

    let mut open = PlanMachine::new(PlanUnitState::default(), true);
    let queued = run_plan_command(&mut open, "   ", false);
    assert_eq!(queued.result.text, PLAN_ENTERING_TEXT);
    assert_eq!(queued.steer, None);
    assert_eq!(queued.set.outcome(), PlanSetOutcome::Queued);
    assert_eq!(open.status().pending, Some(true));

    let mut message = PlanMachine::new(PlanUnitState::default(), true);
    let steer = run_plan_command(&mut message, "  draft the migration  ", false);
    assert_eq!(steer.steer, Some("draft the migration".to_string()));
    assert_eq!(steer.result.text, PLAN_ENTERING_TEXT);

    let mut attachments = PlanMachine::new(PlanUnitState::default(), true);
    let bare = run_plan_command(&mut attachments, "", true);
    assert_eq!(bare.steer, Some(String::new()));
    assert_eq!(bare.result.text, PLAN_ENTERING_TEXT);
    assert_eq!(attachments.status().pending, Some(true));
}

#[test]
fn plan_off_commits_queues_cancels_and_repeats() {
    let mut inactive = PlanMachine::new(PlanUnitState::default(), false);
    let already = run_plan_command(&mut inactive, " off ", false);
    assert_eq!(already.result.kind, PlanCommandKind::Success);
    assert_eq!(already.result.text, PLAN_ALREADY_INACTIVE_TEXT);
    assert_eq!(already.set, PlanSetDecision::Noop);
    assert_eq!(inactive.status().pending, None);

    let mut active = PlanMachine::new(
        PlanUnitState {
            active: true,
            ..PlanUnitState::default()
        },
        false,
    );
    let leaving = run_plan_command(&mut active, "off", false);
    assert_eq!(leaving.result.text, PLAN_OFF_TEXT);
    assert_eq!(leaving.set.outcome(), PlanSetOutcome::Committed);
    active.commit_set(&leaving.set);
    assert!(!active.active());

    let mut open = PlanMachine::new(
        PlanUnitState {
            active: true,
            ..PlanUnitState::default()
        },
        true,
    );
    let queued = run_plan_command(&mut open, "off", false);
    assert_eq!(queued.result.text, PLAN_LEAVING_TEXT);
    assert_eq!(queued.set.outcome(), PlanSetOutcome::Queued);
    let repeat = run_plan_command(&mut open, "off", false);
    assert_eq!(repeat.result.text, PLAN_LEAVING_TEXT);
    assert_eq!(repeat.set, PlanSetDecision::Noop);
    assert_eq!(open.status().pending, Some(false));

    let mut entering = PlanMachine::new(PlanUnitState::default(), true);
    let _ = run_plan_command(&mut entering, "", false);
    let cancelled = run_plan_command(&mut entering, "off", false);
    assert_eq!(cancelled.result.text, PLAN_ENTRY_CANCELLED_TEXT);
    assert_eq!(cancelled.set, PlanSetDecision::Cancelled);
    assert_eq!(entering.status().pending, Some(false));
    let flush = entering
        .decide_flush()
        .expect("the cancellation waits for the boundary");
    assert_eq!(flush.active(), None);
    entering.commit_flush(&flush);
    assert_eq!(entering.status().pending, None);
}

#[test]
fn attachments_cannot_accompany_plan_off() {
    let mut machine = PlanMachine::new(
        PlanUnitState {
            active: true,
            ..PlanUnitState::default()
        },
        true,
    );
    let refused = run_plan_command(&mut machine, "off", true);
    assert_eq!(refused.result.kind, PlanCommandKind::Error);
    assert_eq!(refused.result.text, PLAN_ATTACHMENTS_OFF_TEXT);
    assert_eq!(refused.steer, None);
    assert_eq!(refused.set, PlanSetDecision::Noop);
    assert!(machine.active());
    assert_eq!(machine.pending(), None);
}

#[test]
fn the_command_result_serializes_as_the_executor_row() {
    assert_eq!(
        serde_json::to_string(&dsh_plan::PlanCommandResult::success("Plan mode off."))
            .expect("the result serializes"),
        r#"{"kind":"success","text":"Plan mode off."}"#
    );
    assert_eq!(
        serde_json::to_string(&dsh_plan::PlanCommandResult::error("bad input"))
            .expect("the result serializes"),
        r#"{"kind":"error","text":"bad input"}"#
    );
}

#[test]
fn the_review_question_type_is_serializable_as_a_plain_value() {
    let question: PlanReviewQuestion = plan_review_question("# P", &ToolCallId::new("call-2"));
    let value = serde_json::to_value(&question).expect("the question serializes");
    assert_eq!(value["id"], json!(REVIEW_ID));
    assert_eq!(value["intent"]["callId"], json!("call-2"));
    assert_eq!(value["options"][0]["label"], json!(APPROVE_LABEL));
}
