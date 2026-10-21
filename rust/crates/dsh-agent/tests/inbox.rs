//! Inbox fold behavior: splices apply in commit order, and the fold reads the
//! `agent/inbox/spliced` extension slot the session log carries.

use dsh_agent::{fold_inbox, Agent, InboxSplice, InboxState, InboxTarget, INBOX_SPLICED};
use dsh_llm::{ContentBlock, MessageId, UserMessage};
use dsh_session::{ExtensionEvent, Session, SessionEvent, SessionId, SessionMeta};
use serde_json::Map;

fn user(text: &str) -> UserMessage {
    UserMessage::new(
        MessageId::new(format!("m-{text}")),
        vec![ContentBlock::text(text)],
    )
}

fn splice_event(splice: &InboxSplice) -> SessionEvent {
    let payload = serde_json::to_value(splice).unwrap();
    let map = payload.as_object().unwrap().clone();
    SessionEvent::Extension(ExtensionEvent::new(INBOX_SPLICED, map))
}

#[test]
fn splices_apply_in_commit_order() {
    let mut state = InboxState::default();
    dsh_agent::apply_splice(
        &mut state,
        &InboxSplice {
            target: InboxTarget::NextTurn,
            start: 0,
            removed_count: None,
            inserted: vec![user("a"), user("b")],
            outcome: None,
        },
    );
    dsh_agent::apply_splice(
        &mut state,
        &InboxSplice {
            target: InboxTarget::NextStep,
            start: 0,
            removed_count: None,
            inserted: vec![user("c")],
            outcome: None,
        },
    );
    dsh_agent::apply_splice(
        &mut state,
        &InboxSplice {
            target: InboxTarget::NextTurn,
            start: 1,
            removed_count: Some(1),
            inserted: vec![user("B")],
            outcome: None,
        },
    );
    assert_eq!(state.target(InboxTarget::NextTurn).len(), 2);
    assert_eq!(state.target(InboxTarget::NextStep).len(), 1);
    assert_eq!(state.len(), 3);
    assert!(!state.is_empty());
}

#[test]
fn the_fold_reconstructs_pending_input_from_the_event_log() {
    let mut session = Session::create(SessionId::new("s").unwrap(), SessionMeta::default(), 0);
    session
        .append(splice_event(&InboxSplice {
            target: InboxTarget::NextTurn,
            start: 0,
            removed_count: None,
            inserted: vec![user("a"), user("b")],
            outcome: None,
        }))
        .unwrap();
    session
        .append(splice_event(&InboxSplice {
            target: InboxTarget::NextTurn,
            start: 0,
            removed_count: Some(2),
            inserted: vec![],
            outcome: Some("canceled".to_string()),
        }))
        .unwrap();
    let inbox = fold_inbox(session.events());
    assert!(inbox.is_empty());
}

#[test]
fn the_fold_ignores_other_extension_events() {
    let mut session = Session::create(SessionId::new("s").unwrap(), SessionMeta::default(), 0);
    session
        .append(SessionEvent::extension_event(
            "compaction/start",
            Map::new(),
        ))
        .unwrap();
    session
        .append(splice_event(&InboxSplice {
            target: InboxTarget::NextStep,
            start: 0,
            removed_count: None,
            inserted: vec![user("x")],
            outcome: None,
        }))
        .unwrap();
    let inbox = fold_inbox(session.events());
    assert_eq!(inbox.next_step.len(), 1);
    assert_eq!(inbox.next_turn.len(), 0);
}

#[test]
fn an_agent_is_its_session_identity() {
    let id = SessionId::new("agent-1").unwrap();
    let agent = Agent::new(id.clone());
    assert_eq!(agent.id, id);
}
