//! Regression locks for the fixed estimator, the surface fold, route pricing,
//! and the turn-usage fold.

use dsh_llm::{
    ContentBlock, CoreContentBlock, ImageAttachmentRef, Message, MessageId, StreamChunk, TokenUsage,
};
use dsh_session::{
    CoreEvent, Session, SessionEvent, SessionId, SessionLogOffset, SessionMeta, SessionSeq,
};
use dsh_session_projection::ProjectionRegistry;
use dsh_token_meter::estimate::{estimate_content, estimate_message, estimate_tools_tokens};
use dsh_token_meter::projection::{token_usage_projection_definition, TokenUsageState, KEY};
use dsh_token_meter::route_pricing::{price_surface, ImagePrice, ImageRequestPricing};
use dsh_token_meter::surface_fold::{
    commit_surface_tokens, plan_surface_tokens, MeterSurfaceNode, SurfaceEvent, SurfaceOp,
};
use dsh_token_meter::turn_usage::derive_turn_token_usage;

fn seq(value: u64) -> SessionSeq {
    SessionSeq::new(value).expect("in range")
}

fn offset(value: u64) -> SessionLogOffset {
    SessionLogOffset::new(value).unwrap()
}

fn text_message(text: &str) -> Message {
    Message::User(dsh_llm::UserMessage::new(
        MessageId::new("m1"),
        vec![ContentBlock::text(text)],
    ))
}

// --- estimate -------------------------------------------------------------

#[test]
fn utf16_len_matches_the_typescript_string_length() {
    // One BMP character is one UTF-16 unit; an emoji is a surrogate pair.
    assert_eq!(dsh_token_meter::utf16_len("中文"), 2);
    assert_eq!(dsh_token_meter::utf16_len("é"), 1);
    assert_eq!(dsh_token_meter::utf16_len("\u{1F600}"), 2);
}

#[test]
fn content_prices_by_utf16_density_plus_block_overhead() {
    let blocks = vec![ContentBlock::text("abcd")]; // 4 chars -> 1 token + 4 overhead
    assert_eq!(estimate_content(&blocks), 5);
}

#[test]
fn an_empty_tool_list_prices_at_zero() {
    assert_eq!(estimate_tools_tokens(None), 0);
}

#[test]
fn a_message_prices_its_content_plus_role_overhead() {
    let message = text_message("abcd");
    assert_eq!(estimate_message(&message), 5 + 4);
}

// --- surface fold ---------------------------------------------------------

fn user_event(text: &str) -> SessionEvent {
    SessionEvent::Core(CoreEvent::UserMessage {
        message: Box::new(dsh_llm::UserMessage::new(
            MessageId::new("m1"),
            vec![ContentBlock::text(text)],
        )),
    })
}

fn surface(seq_value: u64, event: SessionEvent, op: SurfaceOp) -> SurfaceEvent {
    SurfaceEvent {
        seq: seq(seq_value),
        event,
        op,
    }
}

#[test]
fn an_append_plans_the_message_price_and_commits_at_the_tail() {
    let mut nodes: Vec<MeterSurfaceNode> = Vec::new();
    let event = surface(0, user_event("abcd"), SurfaceOp::Append);
    let plan = plan_surface_tokens(&nodes, &event).expect("append plans");
    assert_eq!(plan.tokens, 5 + 4);
    assert_eq!(plan.delta_tokens, 5 + 4);
    assert_eq!(plan.target, dsh_token_meter::SurfaceTarget::Append);
    commit_surface_tokens(&mut nodes, plan);
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].seq, seq(0));
}

#[test]
fn a_replace_fails_loud_when_the_range_is_absent() {
    let nodes: Vec<MeterSurfaceNode> = Vec::new();
    let event = surface(
        7,
        user_event("abcd"),
        SurfaceOp::Replace {
            start_seq: seq(1),
            end_seq: seq(2),
        },
    );
    let error = plan_surface_tokens(&nodes, &event).expect_err("missing range");
    assert!(error.to_string().contains("invalid current range 1-2"));
}

#[test]
fn a_replace_subtracts_the_shadowed_heuristic_prices() {
    let mut nodes: Vec<MeterSurfaceNode> = Vec::new();
    for (index, text) in ["abcd", "efgh"].into_iter().enumerate() {
        let event = surface(index as u64, user_event(text), SurfaceOp::Append);
        let plan = plan_surface_tokens(&nodes, &event).expect("appends plan");
        commit_surface_tokens(&mut nodes, plan);
    }
    let event = surface(
        2,
        user_event("ok"),
        SurfaceOp::Replace {
            start_seq: seq(0),
            end_seq: seq(1),
        },
    );
    let plan = plan_surface_tokens(&nodes, &event).expect("replace plans");
    let removed = 2 * (5 + 4);
    assert_eq!(plan.delta_tokens, (1 + 4 + 4) - removed);
    commit_surface_tokens(&mut nodes, plan);
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].seq, seq(2));
}

#[test]
fn an_empty_system_message_prices_at_zero() {
    let event = surface(
        0,
        SessionEvent::Core(CoreEvent::SystemMessage {
            turn: 1,
            step: 1,
            message: Box::new(dsh_llm::SystemMessage::new(
                MessageId::new("s1"),
                Vec::new(),
            )),
        }),
        SurfaceOp::Append,
    );
    let plan = plan_surface_tokens(&[], &event).expect("plans");
    assert_eq!(plan.tokens, 0);
}

#[test]
fn image_occurrences_are_collected_for_route_pricing() {
    let event = surface(
        0,
        SessionEvent::Core(CoreEvent::UserMessage {
            message: Box::new(dsh_llm::UserMessage::new(
                MessageId::new("m1"),
                vec![
                    ContentBlock::Core(CoreContentBlock::Image {
                        attachment: image_ref("a1"),
                        offloaded: false,
                    }),
                    ContentBlock::Core(CoreContentBlock::File {
                        attachment: dsh_llm::FileAttachmentRef {
                            attachment_id: "f1".into(),
                            name: "notes.txt".into(),
                            bytes: 4,
                        },
                    }),
                ],
            )),
        }),
        SurfaceOp::Append,
    );
    let plan = plan_surface_tokens(&[], &event).expect("plans");
    assert_eq!(plan.node.images.len(), 1);
    assert_eq!(plan.node.files.len(), 1);
    assert!(plan.node.image_structural_tokens > 0);
    assert!(plan.node.file_structural_tokens > 0);
}

fn image_ref(id: &str) -> ImageAttachmentRef {
    ImageAttachmentRef {
        attachment_id: id.into(),
        media_type: "image/png".into(),
        bytes: 8,
        width: 2,
        height: 2,
        name: None,
        original_dimensions: None,
    }
}

// --- route pricing --------------------------------------------------------

struct FixedPricing(Vec<ImagePrice>);

impl ImageRequestPricing for FixedPricing {
    fn price_images(&self, _images: &[dsh_token_meter::MeterImage]) -> Vec<ImagePrice> {
        self.0.clone()
    }
}

#[test]
fn without_a_route_every_node_prices_at_its_heuristic() {
    let nodes = vec![MeterSurfaceNode {
        seq: seq(0),
        heuristic_tokens: 9,
        image_structural_tokens: 3,
        file_structural_tokens: 0,
        images: Vec::new(),
        files: Vec::new(),
    }];
    let priced = price_surface(&nodes, None, None).expect("prices");
    assert_eq!(priced.surface_tokens, 9);
    assert_eq!(priced.nodes[0].tokens, 9);
}

#[test]
fn a_misaligned_route_answer_fails_loud() {
    let nodes = vec![MeterSurfaceNode {
        seq: seq(0),
        heuristic_tokens: 9,
        image_structural_tokens: 3,
        file_structural_tokens: 0,
        images: vec![dsh_token_meter::MeterImage {
            attachment: image_ref("a1"),
            offloaded: false,
        }],
        files: Vec::new(),
    }];
    let pricing = FixedPricing(Vec::new());
    let error = price_surface(&nodes, Some(&pricing), None).expect_err("mismatch");
    assert!(error.to_string().contains("answered 0 prices for 1"));
}

#[test]
fn file_projection_replaces_the_structural_price() {
    let nodes = vec![MeterSurfaceNode {
        seq: seq(0),
        heuristic_tokens: 9,
        image_structural_tokens: 0,
        file_structural_tokens: 7,
        images: Vec::new(),
        files: vec![dsh_llm::FileAttachmentRef {
            attachment_id: "f1".into(),
            name: "notes.txt".into(),
            bytes: 4,
        }],
    }];
    let file_text = |_file: &dsh_llm::FileAttachmentRef| "abcd".to_string();
    let priced = price_surface(&nodes, None, Some(&file_text)).expect("prices");
    // The structural price is replaced by the projected text's heuristic.
    assert_eq!(priced.nodes[0].tokens, 9 - 7 + 5);
}

// --- turn usage -----------------------------------------------------------

fn usage_event(turn: u64, step: u64, usage: TokenUsage) -> SessionEvent {
    SessionEvent::assistant_message(
        turn,
        step,
        dsh_llm::AssistantMessage::new(
            MessageId::new("a1"),
            vec![ContentBlock::text("hi")],
            "deepseek-official",
            "deepseek-v4",
        ),
        Vec::new(),
        Some(usage),
        false,
    )
}

fn sample(input: u64, output: u64) -> TokenUsage {
    TokenUsage {
        input_tokens: input,
        output_tokens: output,
        // Without an exact total the fold requires both cache buckets to
        // derive one, so every priced sample states its exact total.
        total_tokens: Some(input + output),
        cache_read_tokens: None,
        cache_write_tokens: None,
        reasoning_tokens: None,
    }
}

fn bounded_turn(events: Vec<SessionEvent>) -> Vec<SessionEvent> {
    let mut all = vec![SessionEvent::Core(CoreEvent::TurnStart { turn: 1 })];
    all.extend(events);
    all.push(SessionEvent::Core(CoreEvent::TurnEnd {
        turn: 1,
        reason: dsh_session::TurnEndReason::Completed,
    }));
    all
}

#[test]
fn a_completed_step_reports_its_exact_usage() {
    let events = bounded_turn(vec![
        SessionEvent::Core(CoreEvent::StepStart { turn: 1, step: 1 }),
        usage_event(1, 1, sample(10, 5)),
        SessionEvent::Core(CoreEvent::StepEnd { turn: 1, step: 1 }),
    ]);
    let usage = derive_turn_token_usage(&events)
        .expect("fold")
        .expect("priced");
    assert_eq!(usage.uncached_input_tokens, 10);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
    assert_eq!(
        usage.routes.as_ref().expect("attributed")[0].model,
        "deepseek-v4"
    );
}

#[test]
fn a_turn_without_a_usage_sample_discloses_nothing() {
    let events = bounded_turn(vec![
        SessionEvent::Core(CoreEvent::StepStart { turn: 1, step: 1 }),
        SessionEvent::assistant_message(
            1,
            1,
            dsh_llm::AssistantMessage::new(
                MessageId::new("a1"),
                vec![ContentBlock::text("hi")],
                "deepseek-official",
                "deepseek-v4",
            ),
            Vec::new(),
            None,
            false,
        ),
        SessionEvent::Core(CoreEvent::StepEnd { turn: 1, step: 1 }),
    ]);
    let usage = derive_turn_token_usage(&events).expect("fold");
    assert_eq!(usage, None);
}

#[test]
fn a_missing_step_end_discloses_nothing() {
    let events = bounded_turn(vec![
        SessionEvent::Core(CoreEvent::StepStart { turn: 1, step: 1 }),
        usage_event(1, 1, sample(10, 5)),
    ]);
    let usage = derive_turn_token_usage(&events).expect("fold");
    assert_eq!(usage, None);
}

#[test]
fn a_stream_usage_sample_prices_when_the_field_is_absent() {
    let events = bounded_turn(vec![
        SessionEvent::Core(CoreEvent::StepStart { turn: 1, step: 1 }),
        SessionEvent::Core(CoreEvent::AssistantAttempt {
            turn: 1,
            step: 1,
            stream: vec![dsh_llm::AssistantStreamRecord::Chunk {
                time: 1,
                chunk: Box::new(StreamChunk::Usage {
                    usage: sample(3, 4),
                }),
            }],
        }),
        SessionEvent::Core(CoreEvent::StepEnd { turn: 1, step: 1 }),
    ]);
    let usage = derive_turn_token_usage(&events)
        .expect("fold")
        .expect("priced");
    assert_eq!(usage.total_tokens, 7);
    // A settled attempt without a message carries no route attribution.
    assert_eq!(usage.routes, None);
}

#[test]
fn a_contradictory_exact_total_discloses_nothing() {
    let mut usage = sample(10, 5);
    usage.total_tokens = Some(20);
    usage.cache_read_tokens = Some(2);
    usage.cache_write_tokens = Some(2); // known prompt 14 != exact prompt 15
    let events = bounded_turn(vec![
        SessionEvent::Core(CoreEvent::StepStart { turn: 1, step: 1 }),
        usage_event(1, 1, usage),
        SessionEvent::Core(CoreEvent::StepEnd { turn: 1, step: 1 }),
    ]);
    let usage = derive_turn_token_usage(&events).expect("fold");
    assert_eq!(usage, None);
}

#[test]
fn reasoning_above_output_discloses_nothing() {
    let mut usage = sample(10, 5);
    usage.reasoning_tokens = Some(6);
    let events = bounded_turn(vec![
        SessionEvent::Core(CoreEvent::StepStart { turn: 1, step: 1 }),
        usage_event(1, 1, usage),
        SessionEvent::Core(CoreEvent::StepEnd { turn: 1, step: 1 }),
    ]);
    let usage = derive_turn_token_usage(&events).expect("fold");
    assert_eq!(usage, None);
}

// --- tokenUsage projection ------------------------------------------------

fn session(id: &str) -> Session {
    Session::create(SessionId::new(id).unwrap(), SessionMeta::default(), 0)
}

fn append(session: &mut Session, event: SessionEvent) -> SessionSeq {
    session.append(event).unwrap()
}

#[test]
fn the_projection_accumulates_usage_across_attempts() {
    let registry = ProjectionRegistry::new();
    let _unit = registry
        .register(token_usage_projection_definition())
        .unwrap();
    let mut session = session("usage-accumulate");

    append(&mut session, usage_event(1, 1, sample(10, 5)));
    append(&mut session, usage_event(1, 2, sample(3, 2)));

    let state = registry
        .state_of::<TokenUsageState>(&session, offset(0), KEY)
        .unwrap()
        .expect("state");
    assert_eq!(state.totals.uncached_input_tokens, 13);
    assert_eq!(state.totals.output_tokens, 7);
    assert_eq!(state.totals.cache_read_tokens, 0);
}

#[test]
fn a_repeated_sample_for_the_same_attempt_replaces_rather_than_doubles() {
    let registry = ProjectionRegistry::new();
    let _unit = registry
        .register(token_usage_projection_definition())
        .unwrap();
    let mut session = session("usage-replace");

    append(&mut session, usage_event(1, 1, sample(10, 5)));
    let first = registry
        .state_of::<TokenUsageState>(&session, offset(0), KEY)
        .unwrap()
        .expect("state");

    // The same turn/step reporting again replaces its previous contribution.
    append(&mut session, usage_event(1, 1, sample(4, 4)));
    let second = registry
        .state_of::<TokenUsageState>(&session, offset(0), KEY)
        .unwrap()
        .expect("state");

    assert_eq!(first.totals.output_tokens, 5);
    assert_eq!(second.totals.uncached_input_tokens, 4);
    assert_eq!(second.totals.output_tokens, 4);
}

#[test]
fn a_retry_started_closes_the_replacement_slot() {
    let registry = ProjectionRegistry::new();
    let _unit = registry
        .register(token_usage_projection_definition())
        .unwrap();
    let mut session = session("usage-retry");

    append(&mut session, usage_event(1, 1, sample(10, 5)));
    let mut payload = serde_json::Map::new();
    payload.insert("retryId".into(), serde_json::json!("r1"));
    payload.insert("turn".into(), serde_json::json!(1));
    payload.insert("step".into(), serde_json::json!(1));
    payload.insert("retry".into(), serde_json::json!(1));
    append(
        &mut session,
        SessionEvent::Extension(dsh_session::ExtensionEvent::new(
            "llm/retry-started",
            payload,
        )),
    );
    // The retried attempt is a fresh slot: its sample adds again.
    append(&mut session, usage_event(1, 1, sample(6, 1)));

    let state = registry
        .state_of::<TokenUsageState>(&session, offset(0), KEY)
        .unwrap()
        .expect("state");
    assert_eq!(state.totals.uncached_input_tokens, 16);
    assert_eq!(state.totals.output_tokens, 6);
}
