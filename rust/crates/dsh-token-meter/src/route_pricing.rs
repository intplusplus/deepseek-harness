//! Request-projected surface pricing: replaces attachment-block heuristics with
//! the image and file representations sent to the routed model.
//!
//! Transcribed from `packages/llm/token-meter/src/route-pricing.ts`. The
//! TypeScript `LlmImageRequestPricing` is an adapter-declared interface; the
//! Rust side models it as the [`ImageRequestPricing`] trait so a provider
//! adapter supplies its own image pricing without the core knowing it.

use dsh_llm::{ContentBlock, FileAttachmentRef};
use dsh_session::SessionSeq;

use crate::estimate::estimate_content;
use crate::surface_fold::MeterSurfaceNode;

/// One routed image occurrence's request price.
#[derive(Clone, Debug, PartialEq)]
pub struct ImagePrice {
    /// The provider's fixed visual-token price for the image.
    pub visual_tokens: u64,
    /// The projected textual framing of the image in the request.
    pub text: String,
}

/// The routed model's image pricing, as declared by its adapter.
///
/// Implementations must answer one price per occurrence, in the order given;
/// a misaligned answer is a route bug and [`price_surface`] fails loud on it.
pub trait ImageRequestPricing {
    /// Price every image occurrence of the current surface, in order.
    fn price_images(&self, images: &[crate::surface_fold::MeterImage]) -> Vec<ImagePrice>;
}

/// One surface priced for a request route: public nodes plus their total.
#[derive(Clone, Debug, PartialEq)]
pub struct PricedSurface {
    /// Positional nodes carrying both the route price and the fixed-heuristic price.
    pub nodes: Vec<TokenSurfaceNode>,
    /// Sum of the route prices across the surface.
    pub surface_tokens: i64,
}

/// One token-priced node in the current ordered session surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenSurfaceNode {
    /// Durable sequence number of the surface event.
    pub seq: SessionSeq,
    /// Request-pressure tokens for the exact message projected by this node
    /// under the measured route.
    pub tokens: i64,
    /// Fixed-heuristic tokens for the same message, independent of any route.
    pub heuristic_tokens: i64,
}

/// Price one ordered surface under its model-request attachment projection.
///
/// # Errors
/// The priced route must answer exactly as many image prices as there are
/// image occurrences; misalignment would silently misprice nodes, so a route
/// that answers a different occurrence count makes the call fail loud.
pub fn price_surface(
    nodes: &[MeterSurfaceNode],
    pricing: Option<&dyn ImageRequestPricing>,
    file_text: Option<&dyn Fn(&FileAttachmentRef) -> String>,
) -> Result<PricedSurface, RoutePricingError> {
    let mut images = Vec::new();
    if pricing.is_some() {
        for node in nodes {
            images.extend(node.images.iter().cloned());
        }
    }
    let has_files = file_text.is_some() && nodes.iter().any(|node| !node.files.is_empty());
    if (pricing.is_none() || images.is_empty()) && !has_files {
        let mut surface_tokens = 0;
        let public_nodes = nodes
            .iter()
            .map(|node| {
                surface_tokens += node.heuristic_tokens;
                TokenSurfaceNode {
                    seq: node.seq,
                    tokens: node.heuristic_tokens,
                    heuristic_tokens: node.heuristic_tokens,
                }
            })
            .collect();
        return Ok(PricedSurface {
            nodes: public_nodes,
            surface_tokens,
        });
    }
    let prices = pricing
        .map(|pricing| pricing.price_images(&images))
        .unwrap_or_default();
    if let Some(_pricing) = pricing {
        if prices.len() != images.len() {
            return Err(RoutePricingError::PriceCountMismatch {
                answered: prices.len(),
                occurrences: images.len(),
            });
        }
    }
    let mut cursor = 0;
    let mut surface_tokens = 0;
    let mut public_nodes = Vec::with_capacity(nodes.len());
    for node in nodes {
        let mut tokens = node.heuristic_tokens;
        if let Some(file_text) = file_text {
            if !node.files.is_empty() {
                tokens -= node.file_structural_tokens;
                for file in &node.files {
                    tokens += estimate_content(&[ContentBlock::text(file_text(file))]) as i64;
                }
            }
        }
        if let Some(_pricing) = pricing {
            if !node.images.is_empty() {
                tokens -= node.image_structural_tokens;
                for _occurrence in 0..node.images.len() {
                    let price = &prices[cursor];
                    cursor += 1;
                    tokens += price.visual_tokens as i64
                        + estimate_content(&[ContentBlock::text(&price.text)]) as i64;
                }
            }
        }
        surface_tokens += tokens;
        public_nodes.push(TokenSurfaceNode {
            seq: node.seq,
            tokens,
            heuristic_tokens: node.heuristic_tokens,
        });
    }
    Ok(PricedSurface {
        nodes: public_nodes,
        surface_tokens,
    })
}

/// Errors raised by route pricing.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RoutePricingError {
    /// The routed adapter answered a different image-occurrence count.
    #[error(
        "token meter: route image pricing answered {answered} prices for {occurrences} occurrences"
    )]
    PriceCountMismatch {
        /// Prices the route answered.
        answered: usize,
        /// Image occurrences the surface carries.
        occurrences: usize,
    },
}
