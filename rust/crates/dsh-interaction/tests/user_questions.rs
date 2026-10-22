//! The user-question state machine: validation, the answerer seam, the live
//! caller rules, and the late reply.

use std::sync::{Arc, Mutex};

use dsh_agent::Agent;
use dsh_interaction::questions::*;
use dsh_llm::{MessageId, ToolCallId, UserMessage};
use dsh_session::SessionId;
use dsh_tools::cancel::CancelSignal;

/// An answerer that records every request it sees.
struct ScriptedAnswerer {
    seen: Mutex<Vec<AskUserQuestionRequest>>,
    answer: AskUserQuestionAnswer,
}

impl ScriptedAnswerer {
    fn new(answer: AskUserQuestionAnswer) -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
            answer,
        }
    }
}

impl UserQuestionAnswerer for ScriptedAnswerer {
    fn ask(
        &self,
        request: &AskUserQuestionRequest,
    ) -> Result<AskUserQuestionAnswer, UserQuestionError> {
        self.seen.lock().unwrap().push(request.clone());
        Ok(self.answer.clone())
    }
}

/// An answerer that always fails.
struct FailingAnswerer;

impl UserQuestionAnswerer for FailingAnswerer {
    fn ask(
        &self,
        _request: &AskUserQuestionRequest,
    ) -> Result<AskUserQuestionAnswer, UserQuestionError> {
        Err(UserQuestionError::AskTimedOut)
    }
}

/// A live-agent registry with fixed answers.
struct StubAgents {
    live: bool,
    root: bool,
}

impl LiveAgentRegistry for StubAgents {
    fn is_live(&self, _agent: &Agent) -> bool {
        self.live
    }

    fn is_root(&self, _agent: &Agent) -> bool {
        self.root
    }
}

/// A reply sink that records every steered message.
struct StubSink {
    steered: Mutex<Vec<UserMessage>>,
}

impl ReplySink for StubSink {
    fn steer(&self, _agent: &Agent, message: UserMessage) -> Result<(), UserQuestionError> {
        self.steered.lock().unwrap().push(message);
        Ok(())
    }
}

fn agent(id: &str) -> Agent {
    Agent::new(SessionId::new(id).unwrap())
}

fn question(id: &str, text: &str) -> AskUserQuestionItem {
    AskUserQuestionItem {
        id: id.to_string(),
        question: text.to_string(),
        detail: None,
        header: None,
        options: None,
        multi_select: None,
        intent: None,
    }
}

fn plan_review_question() -> AskUserQuestionItem {
    AskUserQuestionItem {
        id: "plan".to_string(),
        question: "Approve the plan?".to_string(),
        detail: Some("# Plan".to_string()),
        header: None,
        options: Some(vec![
            AskUserQuestionOption {
                label: "Approve".to_string(),
                description: Some("Ship it.".to_string()),
            },
            AskUserQuestionOption {
                label: "Refuse".to_string(),
                description: None,
            },
        ]),
        multi_select: Some(false),
        intent: Some(AskUserQuestionIntent {
            kind: AskUserQuestionIntentKind::PlanReview,
            approve: "Approve".to_string(),
            call_id: None,
        }),
    }
}

fn answer_for(id: &str, selected: &[&str]) -> AskUserQuestionAnswer {
    AskUserQuestionAnswer {
        answers: vec![AskUserQuestionAnswerItem {
            id: id.to_string(),
            selected: selected.iter().map(|label| (*label).to_string()).collect(),
            custom: None,
        }],
    }
}

#[test]
fn an_empty_question_batch_is_rejected() {
    let error = validate_questions(&[]).unwrap_err();
    assert_eq!(error.code(), "EMPTY_QUESTIONS");
    assert_eq!(
        error.to_string(),
        "ask_user_question requires at least one question"
    );
}

#[test]
fn an_intent_naming_no_option_is_rejected() {
    let mut item = plan_review_question();
    item.intent.as_mut().unwrap().approve = "Ship it".to_string();
    let error = validate_questions(&[item]).unwrap_err();
    assert_eq!(error.code(), "BAD_INTENT");
    assert!(error.to_string().contains("names none of its options"));
}

#[test]
fn an_intent_without_detail_is_rejected() {
    let mut item = plan_review_question();
    item.detail = None;
    let error = validate_questions(&[item]).unwrap_err();
    assert_eq!(error.code(), "BAD_INTENT");
    assert!(error.to_string().contains("without the detail it reviews"));
}

#[test]
fn a_well_formed_question_batch_is_accepted() {
    assert!(validate_questions(&[question("scope", "Which scope?")]).is_ok());
    assert!(validate_questions(&[plan_review_question()]).is_ok());
}

#[test]
fn the_answer_batch_must_name_each_question_exactly_once() {
    let questions = vec![question("a", "A?"), question("b", "B?")];
    let call_id = ToolCallId::new("call-1");

    // Exactly one item per question.
    let good = AskUserQuestionAnswer {
        answers: vec![
            AskUserQuestionAnswerItem {
                id: "a".to_string(),
                selected: vec!["yes".to_string()],
                custom: None,
            },
            AskUserQuestionAnswerItem {
                id: "b".to_string(),
                selected: vec![],
                custom: Some("later".to_string()),
            },
        ],
    };
    assert!(validate_answer_batch(&call_id, &questions, &good).is_ok());

    // One item too few.
    let error =
        validate_answer_batch(&call_id, &questions, &answer_for("a", &["yes"])).unwrap_err();
    assert_eq!(error.code(), "BAD_ANSWER");
    assert!(error
        .to_string()
        .contains("must name each of its 2 questions"));

    // The same question named twice.
    let twice = AskUserQuestionAnswer {
        answers: vec![
            AskUserQuestionAnswerItem {
                id: "a".to_string(),
                selected: vec![],
                custom: None,
            },
            AskUserQuestionAnswerItem {
                id: "a".to_string(),
                selected: vec![],
                custom: None,
            },
        ],
    };
    assert_eq!(
        validate_answer_batch(&call_id, &questions, &twice)
            .unwrap_err()
            .code(),
        "BAD_ANSWER"
    );
}

#[test]
fn an_ask_without_an_answerer_fails_closed() {
    let service = UserQuestionService::new();
    let request = AskUserQuestionRequest::new(vec![question("continue", "Continue?")]);
    let error = service.ask(&request).unwrap_err();
    assert_eq!(error.code(), "NO_PROVIDER");
}

#[test]
fn an_ask_with_an_aborted_signal_is_rejected_before_the_answerer() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for("continue", &["yes"])));
    let service = UserQuestionService::new().with_answerer(answerer.clone());
    let signal = CancelSignal::new();
    signal.cancel();
    let request =
        AskUserQuestionRequest::new(vec![question("continue", "Continue?")]).with_signal(signal);
    let error = service.ask(&request).unwrap_err();
    assert_eq!(error.code(), "ASK_ABORTED");
    assert!(answerer.seen.lock().unwrap().is_empty());
}

#[test]
fn a_caller_that_is_not_the_live_instance_is_rejected() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for("continue", &["yes"])));
    let service = UserQuestionService::new()
        .with_answerer(answerer.clone())
        .with_live_agents(Arc::new(StubAgents {
            live: false,
            root: true,
        }));
    let request = AskUserQuestionRequest::new(vec![question("continue", "Continue?")])
        .with_agent(agent("caller"));
    let error = service.ask(&request).unwrap_err();
    assert_eq!(error.code(), "CALLER_NOT_LIVE");
    assert!(answerer.seen.lock().unwrap().is_empty());
}

#[test]
fn a_caller_owned_by_another_agent_is_rejected() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for("continue", &["yes"])));
    let service = UserQuestionService::new()
        .with_answerer(answerer.clone())
        .with_live_agents(Arc::new(StubAgents {
            live: true,
            root: false,
        }));
    let request = AskUserQuestionRequest::new(vec![question("continue", "Continue?")])
        .with_agent(agent("child"));
    let error = service.ask(&request).unwrap_err();
    assert_eq!(error.code(), "DELEGATED_CALLER");
    assert!(answerer.seen.lock().unwrap().is_empty());
}

#[test]
fn without_a_registry_the_caller_check_fails_closed() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for("continue", &["yes"])));
    let service = UserQuestionService::new().with_answerer(answerer.clone());
    let request = AskUserQuestionRequest::new(vec![question("continue", "Continue?")])
        .with_agent(agent("caller"));
    let error = service.ask(&request).unwrap_err();
    assert_eq!(error.code(), "CALLER_NOT_LIVE");
}

#[test]
fn a_answered_request_reaches_the_answerer_intact() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for("pkg", &["pnpm"])));
    let service = UserQuestionService::new().with_answerer(answerer.clone());
    let request = AskUserQuestionRequest::new(vec![question("pkg", "Which package manager?")]);
    let answer = service.ask(&request).unwrap();
    assert_eq!(answer.answers[0].id, "pkg");

    let seen = answerer.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].questions[0].id, "pkg");
    assert!(seen[0].wait.is_none());
}

#[test]
fn the_projection_view_lists_only_the_continued_questions() {
    let view = UserQuestionProjectionView {
        active: vec![
            PendingUserQuestion {
                call_id: ToolCallId::new("call-open"),
                questions: vec![question("a", "A?")],
                state: UserQuestionState::Open,
            },
            PendingUserQuestion {
                call_id: ToolCallId::new("call-continued"),
                questions: vec![question("b", "B?")],
                state: UserQuestionState::Continued,
            },
        ],
        settled: Vec::new(),
    };
    let continued = view.continued();
    assert_eq!(continued.len(), 1);
    assert_eq!(continued[0].call_id, ToolCallId::new("call-continued"));
}

#[test]
fn the_reply_message_names_the_call_and_carries_the_answers() {
    let questions = vec![question("scope", "Which scope?")];
    let answer = answer_for("scope", &["Tool only"]);
    let message = answer_reply_message(
        MessageId::new("msg-1"),
        &ToolCallId::new("call-1"),
        &questions,
        &answer,
    );

    assert_eq!(message.source.kind(), "user-question-reply");
    let payload: serde_json::Value =
        serde_json::from_str(message.content[0].as_text().unwrap()).unwrap();
    assert_eq!(
        payload["kind"],
        serde_json::json!("answer_to_pending_question")
    );
    assert_eq!(payload["tool"], serde_json::json!("ask_user_question"));
    assert_eq!(payload["callId"], serde_json::json!("call-1"));
    assert_eq!(payload["answers"][0]["id"], serde_json::json!("scope"));
    assert_eq!(
        payload["answers"][0]["selected"][0],
        serde_json::json!("Tool only")
    );
}

#[test]
fn answering_a_continued_question_steers_the_reply() {
    let answerer = Arc::new(ScriptedAnswerer::new(answer_for("scope", &["Tool only"])));
    let sink = Arc::new(StubSink {
        steered: Mutex::new(Vec::new()),
    });
    let service = UserQuestionService::new()
        .with_answerer(answerer)
        .with_live_agents(Arc::new(StubAgents {
            live: true,
            root: true,
        }))
        .with_reply_sink(sink.clone())
        .with_message_ids(Arc::new(CountingMessageIds::new("reply")));

    let view = UserQuestionProjectionView {
        active: vec![PendingUserQuestion {
            call_id: ToolCallId::new("call-1"),
            questions: vec![question("scope", "Which scope?")],
            state: UserQuestionState::Continued,
        }],
        settled: Vec::new(),
    };
    let caller = agent("root");
    let accepted = service
        .answer(
            &view,
            &caller,
            &ToolCallId::new("call-1"),
            &answer_for("scope", &["Tool only"]),
        )
        .unwrap();
    assert!(accepted);

    let steered = sink.steered.lock().unwrap();
    assert_eq!(steered.len(), 1);
    assert_eq!(steered[0].source.kind(), "user-question-reply");
    assert_eq!(steered[0].id, MessageId::new("reply-1"));

    // A second reply for the same question is refused while the first waits.
    let error = service
        .answer(
            &view,
            &caller,
            &ToolCallId::new("call-1"),
            &answer_for("scope", &["Tool only"]),
        )
        .unwrap_err();
    assert_eq!(error.code(), "REPLY_QUEUED");
}

#[test]
fn answering_an_unknown_question_reports_no_continuation() {
    let sink = Arc::new(StubSink {
        steered: Mutex::new(Vec::new()),
    });
    let service = UserQuestionService::new()
        .with_live_agents(Arc::new(StubAgents {
            live: true,
            root: true,
        }))
        .with_reply_sink(sink.clone());
    let view = UserQuestionProjectionView::empty();
    let accepted = service
        .answer(
            &view,
            &agent("root"),
            &ToolCallId::new("call-unknown"),
            &answer_for("scope", &["Tool only"]),
        )
        .unwrap();
    assert!(!accepted);
    assert!(sink.steered.lock().unwrap().is_empty());
}

#[test]
fn a_reply_with_a_malformed_batch_is_rejected() {
    let sink = Arc::new(StubSink {
        steered: Mutex::new(Vec::new()),
    });
    let service = UserQuestionService::new()
        .with_live_agents(Arc::new(StubAgents {
            live: true,
            root: true,
        }))
        .with_reply_sink(sink.clone());
    let view = UserQuestionProjectionView {
        active: vec![PendingUserQuestion {
            call_id: ToolCallId::new("call-1"),
            questions: vec![question("scope", "Which scope?")],
            state: UserQuestionState::Continued,
        }],
        settled: Vec::new(),
    };
    let error = service
        .answer(
            &view,
            &agent("root"),
            &ToolCallId::new("call-1"),
            &answer_for("other", &["Tool only"]),
        )
        .unwrap_err();
    assert_eq!(error.code(), "BAD_ANSWER");
    assert!(sink.steered.lock().unwrap().is_empty());
}

#[test]
fn a_timed_ask_that_times_out_reports_a_pending_result() {
    let answerer = Arc::new(FailingAnswerer);
    let service = UserQuestionService::new()
        .with_answerer(answerer)
        .with_live_agents(Arc::new(StubAgents {
            live: true,
            root: true,
        }));
    let request = AskUserQuestionRequest::new(vec![question("continue", "Continue?")])
        .with_agent(agent("root"));
    let result = service
        .ask_timed(&request, &ToolCallId::new("call-1"), 1_000)
        .unwrap();
    assert_eq!(
        result,
        TimedUserQuestionResult::Pending {
            call_id: ToolCallId::new("call-1")
        }
    );
}

#[test]
fn a_timed_ask_rejects_a_timeout_that_does_not_fit_a_platform_timer() {
    let service = UserQuestionService::new();
    let request = AskUserQuestionRequest::new(vec![question("continue", "Continue?")])
        .with_agent(agent("root"));
    let error = service
        .ask_timed(&request, &ToolCallId::new("call-1"), 0)
        .unwrap_err();
    assert_eq!(error.code(), "BAD_TIMEOUT");
}
