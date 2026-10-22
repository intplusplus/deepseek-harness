//! Positional pricing for measurement: each retained node keeps its identity,
//! heuristic price, and the attachment details route pricing replaces.
//!
//! Transcribed from `packages/llm/token-meter/src/surface-fold.ts`. The fold is
//! a plan/commit pair: [`plan_surface_tokens`] runs every fallible step
//! read-only and [`commit_surface_tokens`] mutates in place, so an error leaves
//! the caller's state untouched and the same malformed event fails identically
//! on every retry.
//!
//! # The surface-op gap
//!
//! The TypeScript `SurfaceEvent` carries a `surfaceOp` marker on every
//! message-producing log row. The Rust core log does not: the marker is a
//! physical-row field ([`dsh_session::PhysicalRow::surface_op`]) and the
//! compaction crate reconstructs it from documented adjacency. This module
//! therefore takes the operation explicitly — the caller owns how it was
//! derived — while every price and validation rule stays byte-for-byte the
//! TypeScript one.

use dsh_llm::{ContentBlock, CoreContentBlock, FileAttachmentRef, ImageAttachmentRef, Message};
use dsh_session::{CoreEvent, SessionEvent, SessionSeq};
use thiserror::Error;

use crate::estimate::{estimate_message, estimate_structural_block};

/// The positional operation one surface event performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceOp {
    /// The event joins the surface tail.
    Append,
    /// The event replaces an inclusive range of current surface positions.
    Replace {
        /// Declared first replaced surface-node seq.
        start_seq: SessionSeq,
        /// Declared last replaced surface-node seq.
        end_seq: SessionSeq,
    },
}

/// One surface event paired with the positional operation it performs.
#[derive(Clone, Debug)]
pub struct SurfaceEvent {
    /// The event's durable sequence number.
    pub seq: SessionSeq,
    /// The committed event.
    pub event: SessionEvent,
    /// The positional operation.
    pub op: SurfaceOp,
}

/// One priced surface node with the image occurrences route pricing replaces.
#[derive(Clone, Debug, PartialEq)]
pub struct MeterSurfaceNode {
    /// Durable sequence number of the surface event.
    pub seq: SessionSeq,
    /// Fixed-heuristic price of the node's exact message.
    pub heuristic_tokens: i64,
    /// Structural JSON price replaced when the routed request projects images.
    pub image_structural_tokens: i64,
    /// Structural JSON price replaced when request assembly projects files to text.
    pub file_structural_tokens: i64,
    /// Durable image occurrences in message order, offload marks included.
    pub images: Vec<MeterImage>,
    /// Durable file occurrences in message order.
    pub files: Vec<FileAttachmentRef>,
}

/// One durable image occurrence, in message order.
#[derive(Clone, Debug, PartialEq)]
pub struct MeterImage {
    /// The attachment service reference.
    pub attachment: ImageAttachmentRef,
    /// Whether the image was offloaded from the request payload.
    pub offloaded: bool,
}

/// Commit position: `Append`, or the inclusive replaced index range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceTarget {
    /// Insert at the surface tail.
    Append,
    /// Replace the inclusive index range `[start_idx, end_idx]`.
    Replace {
        /// First replaced position.
        start_idx: usize,
        /// Last replaced position.
        end_idx: usize,
    },
}

/// One validated surface transition that has not mutated the priced surface yet.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceTokenPlan {
    /// Heuristic price of the event's own message; 0 when it derives none.
    pub tokens: i64,
    /// Signed change in the surface total: `tokens` minus anything shadowed.
    pub delta_tokens: i64,
    /// The priced node the commit inserts for this event.
    pub node: MeterSurfaceNode,
    /// Commit position.
    pub target: SurfaceTarget,
}

/// Errors raised by the surface-token fold.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SurfaceTokenError {
    /// A replacement named a range absent from the current surface.
    #[error("token surface: replace at seq {seq} has invalid current range {start}-{end}")]
    InvalidRange {
        /// Seq of the offending replacement event.
        seq: u64,
        /// Declared first replaced seq.
        start: u64,
        /// Declared last replaced seq.
        end: u64,
    },
}

/// Project a single event into the model-visible message it derives to, or
/// `None` when it produces none.
///
/// This is the per-event counterpart of [`dsh_session::derive_messages`] with
/// the TypeScript `deriveEventMessage` empty-content rule: an empty system,
/// developer, or assistant message keeps its surface position without adding a
/// wire message, so it prices at zero rather than injecting a content-less
/// turn.
pub fn derive_event_message(event: &SessionEvent) -> Option<Message> {
    let SessionEvent::Core(core) = event else {
        return None;
    };
    match core {
        CoreEvent::UserMessage { message } => Some(Message::User((**message).clone())),
        CoreEvent::SystemMessage { message, .. } => Some(Message::System((**message).clone()))
            .filter(|message| !message.content().is_empty()),
        CoreEvent::DeveloperMessage { message, .. } => {
            Some(Message::Developer((**message).clone()))
                .filter(|message| !message.content().is_empty())
        }
        CoreEvent::AssistantMessage { message, .. } => {
            Some(Message::Assistant((**message).clone()))
                .filter(|message| !message.content().is_empty())
        }
        CoreEvent::ToolResult { message, .. } => Some(Message::Tool((**message).clone())),
        _ => None,
    }
}

/// Collect projected attachment occurrences and their structural prices.
fn collect_projected_attachments(
    blocks: &[ContentBlock],
    images: &mut Vec<MeterImage>,
    files: &mut Vec<FileAttachmentRef>,
) -> (i64, i64) {
    let mut image_tokens = 0;
    let mut file_tokens = 0;
    for block in blocks {
        match block {
            ContentBlock::Core(CoreContentBlock::Image {
                attachment,
                offloaded,
            }) => {
                images.push(MeterImage {
                    attachment: attachment.clone(),
                    offloaded: *offloaded,
                });
                image_tokens += estimate_structural_block(block) as i64;
            }
            ContentBlock::Core(CoreContentBlock::File { attachment }) => {
                files.push(attachment.clone());
                file_tokens += estimate_structural_block(block) as i64;
            }
            _ => {}
        }
    }
    (image_tokens, file_tokens)
}

/// Build one priced node from a surface event's derived message.
fn analyze_node(seq: SessionSeq, message: Option<Message>) -> MeterSurfaceNode {
    let Some(message) = message else {
        return MeterSurfaceNode {
            seq,
            heuristic_tokens: 0,
            image_structural_tokens: 0,
            file_structural_tokens: 0,
            images: Vec::new(),
            files: Vec::new(),
        };
    };
    let heuristic_tokens = estimate_message(&message) as i64;
    let mut images = Vec::new();
    let mut files = Vec::new();
    let (image_tokens, file_tokens) =
        collect_projected_attachments(message.content(), &mut images, &mut files);
    MeterSurfaceNode {
        seq,
        heuristic_tokens,
        image_structural_tokens: image_tokens,
        file_structural_tokens: file_tokens,
        images,
        files,
    }
}

/// Validate and price one surface event without mutating the surface.
///
/// # Errors
/// [`SurfaceTokenError::InvalidRange`] when a replacement names a range absent
/// from `nodes` — committed logs are surface-validated at append time, so an
/// unresolvable range is log corruption and must fail loud rather than skip
/// the event.
pub fn plan_surface_tokens(
    nodes: &[MeterSurfaceNode],
    event: &SurfaceEvent,
) -> Result<SurfaceTokenPlan, SurfaceTokenError> {
    let message = derive_event_message(&event.event);
    let node = analyze_node(event.seq, message);
    let tokens = node.heuristic_tokens;
    match event.op {
        SurfaceOp::Append => Ok(SurfaceTokenPlan {
            tokens,
            delta_tokens: tokens,
            node,
            target: SurfaceTarget::Append,
        }),
        SurfaceOp::Replace { start_seq, end_seq } => {
            let start_idx = nodes
                .iter()
                .position(|candidate| candidate.seq == start_seq);
            let end_idx = nodes.iter().position(|candidate| candidate.seq == end_seq);
            let (Some(start_idx), Some(end_idx)) = (start_idx, end_idx) else {
                return Err(SurfaceTokenError::InvalidRange {
                    seq: event.seq.get(),
                    start: start_seq.get(),
                    end: end_seq.get(),
                });
            };
            if start_idx > end_idx {
                return Err(SurfaceTokenError::InvalidRange {
                    seq: event.seq.get(),
                    start: start_seq.get(),
                    end: end_seq.get(),
                });
            }
            let removed: i64 = nodes[start_idx..=end_idx]
                .iter()
                .map(|candidate| candidate.heuristic_tokens)
                .sum();
            Ok(SurfaceTokenPlan {
                tokens,
                delta_tokens: tokens - removed,
                node,
                target: SurfaceTarget::Replace { start_idx, end_idx },
            })
        }
    }
}

/// Apply one validated plan to the priced surface in place; infallible, so it
/// cannot leave a half-applied surface behind.
pub fn commit_surface_tokens(nodes: &mut Vec<MeterSurfaceNode>, plan: SurfaceTokenPlan) {
    match plan.target {
        SurfaceTarget::Append => nodes.push(plan.node),
        SurfaceTarget::Replace { start_idx, end_idx } => {
            nodes.splice(start_idx..=end_idx, [plan.node]);
        }
    }
}
