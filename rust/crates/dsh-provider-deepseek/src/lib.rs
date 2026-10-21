//! `dsh-provider-deepseek` — the real DeepSeek provider.
//!
//! [`DeepSeekProvider`] implements [`dsh_provider::LlmProvider`] over
//! DeepSeek's OpenAI-compatible `POST {base_url}/chat/completions` endpoint,
//! including Server-Sent Events for streaming calls. The canonical request is
//! translated to the wire format and the wire response back to
//! [`dsh_llm`]'s vocabulary inside this crate; no provider-native type escapes.
//!
//! ```no_run
//! use dsh_provider_deepseek::{DeepSeekConfig, DeepSeekProvider};
//!
//! # fn main() -> Result<(), dsh_llm::LlmFailure> {
//! let provider = DeepSeekProvider::new(DeepSeekConfig::new("sk-…"))?;
//! # let _ = provider;
//! # Ok(())
//! # }
//! ```
//!
//! ## What this crate guarantees
//!
//! - **Semantic equivalence, not byte equivalence.** The TypeScript
//!   `llm-deepseek` package speaks the in-house Messages transport; this crate
//!   speaks the OpenAI-compatible one. The goal is that the same
//!   [`dsh_provider::LlmRequest`] expresses the same intent and the same server
//!   answer produces the same [`dsh_provider::LlmResponse`]: the same content
//!   blocks in model order, the same tool calls with their ids and raw
//!   arguments, the same finish reason, and the same disjoint token counts.
//! - **Stable failure codes.** `LlmFailure::code` is provider-neutral and
//!   ordered the way `packages/llm/llm-deepseek/src/transport.ts` orders it, so
//!   `dsh-retry` can route on it: `AUTH`, `QUOTA`, `RATE_LIMIT`,
//!   `CONTEXT_WINDOW_EXCEEDED`, `INVALID_REQUEST`, `SERVER`, `HTTP_<n>`,
//!   `TRANSPORT`, `TIMEOUT`, `MALFORMED_RESPONSE`, `STREAM_CLOSED`,
//!   `UNSUPPORTED_CONTENT`, `EMPTY_RESPONSE`, `MISSING_CREDENTIAL`, and
//!   `INVALID_CREDENTIAL`.
//! - **Secrets stay out of diagnostics.** [`DeepSeekConfig`]'s `Debug` prints
//!   the key as `<redacted>`, and no failure message is built from request
//!   headers.
//!
//! ## Deliberate differences from the TypeScript adapter
//!
//! These are *not* implemented rather than silently approximated:
//!
//! - **Image and file bytes.** The Messages adapter uploads images (or inlines
//!   them) and references files by id, tracking a per-request byte budget,
//!   pixel budget, and offload quanta. This crate has no attachment store, so
//!   it can never send the bytes. `image` and `file` blocks are flattened to a
//!   placeholder that says so, and `LlmFailure::offload_images` is never set —
//!   image offload is a recovery path this transport cannot need.
//! - **Account and credential routing.** `resolveAuth`/`onRequestError` (key
//!   rotation, per-account rate-limit state) and the Files API are absent; a
//!   single configured key is used for every request.
//! - **Request pricing and model catalog.** `imageRequestPricing`,
//!   `imageRequestTarget`, `listModels`, and `resolveModel` have no counterpart:
//!   the crate does not model costs, context windows, or per-model request caps.
//! - **Signed thinking replay.** Messages returns a `signature` per thinking
//!   block and replays it on the next request. The OpenAI surface has no such
//!   field, and DeepSeek rejects `reasoning_content` in input, so assistant
//!   reasoning blocks are dropped from the request and
//!   `LlmResponse::stream` carries no `replay_state`.
//! - **Dynamic tool changes.** Messages uses `tool_addition`/`tool_removal`
//!   blocks with a beta header. Here every request re-declares the complete
//!   `tools` list, so those blocks only contribute their surrounding text.
//! - **Prompt and content flattening.** The wire format carries one string per
//!   message, so: every `system` message and every `developer` message's text
//!   folds into one leading `system` message (joined with a blank line, in
//!   encounter order); text blocks inside one message are joined with `\n`; and
//!   adjacent user messages are merged with a blank line. Messages keeps them
//!   as separate parts and can place a system update mid-conversation.
//! - **`ToolResultMessage::is_error`.** The wire format has no field for it, so
//!   a failed tool result reaches the model as ordinary tool text.
//! - **Attribution and extension headers.** `attributionHeaders()`, the
//!   per-session/user ids, the `x-deepseek-harness-*` headers, and the
//!   `@deepseek-ai/dsh-deepseek-llm-api-extensions` mechanism are not sent.
//! - **`Retry-After` HTTP-date form.** Only the delay-seconds form (integer or
//!   decimal) becomes `provider_retry_after_ms`; a date is ignored instead of
//!   being guessed, so no retry is scheduled from a mis-parsed clock.
//! - **Provider-native cache fields.** DeepSeek's `prompt_cache_hit_tokens` is
//!   preferred over the OpenAI-shaped `prompt_tokens_details.cached_tokens`;
//!   both map onto `TokenUsage::cache_read_tokens`, and the disjoint
//!   `input_tokens` is reduced by it. No cache-write bucket exists on this
//!   endpoint, so `cache_write_tokens` stays `None` rather than `0`.
//!
//! ## Streaming
//!
//! A streamed call is accumulated into one [`dsh_provider::LlmResponse`]:
//! `reasoning_content` deltas become a reasoning block, `content` deltas a text
//! block, and `tool_calls` deltas become tool-call blocks correlated by the
//! provider's `index` so parallel calls keep their argument fragments apart.
//! The exact timed stream is preserved in `LlmResponse::stream` as packed
//! `*Chunks` runs plus `block-start`/`block-end`/`usage`/`finish` records, so a
//! replay reproduces the cadence without re-calling the model.
//!
//! A `[DONE]` frame is terminal. A stream that ends cleanly after a
//! `finish_reason` but without `[DONE]` is accepted (the provider declared
//! itself finished); one that ends with neither is `STREAM_CLOSED`, and a body
//! read error is `TRANSPORT`.
#![forbid(unsafe_code)]

mod config;
mod error;
mod sse;
mod translate;
mod transport;
mod wire;

pub use config::{
    DeepSeekConfig, DeepSeekProvider, DEFAULT_BASE_URL, DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_STREAM_IDLE_TIMEOUT,
};
