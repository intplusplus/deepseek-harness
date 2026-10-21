//! `dsh-provider` — the model-call boundary.
//!
//! Everything below this line is vocabulary; everything above it is a driver.
//! The provider owns exactly one question — "given this request, what did the
//! model say?" — and answers it with the canonical vocabulary so the loop
//! never learns a provider's wire format.
//!
//! ## Why async, and why boxed futures
//!
//! [`LlmProvider::generate`] is asynchronous: a model call spends its time
//! waiting on a network, and a blocking trait would force the driver to own a
//! thread per request. The driver holds providers as `&dyn LlmProvider`, so
//! the trait must stay object-safe. A native `async fn` in a trait is not
//! dyn-compatible, so the trait uses `async_trait`: it gives every method the
//! same boxed, `Send` future an implementation would have to write by hand,
//! keeps `&dyn` working, and lets an implementation hold borrows such as
//! `&LlmRequest` across an await.
//!
//! Implementations are synchronous unless they need to await. [`FnProvider`]
//! keeps taking a plain closure, so a test's scripted answer does not have to
//! become a future; a provider that performs real I/O implements the trait
//! directly with an `async fn`.

#![forbid(unsafe_code)]

mod provider;
mod request;

pub use provider::{
    text_response, tool_call_response, user_message, EchoProvider, FnProvider, LlmProvider,
    ScriptedProvider,
};
pub use request::{LlmRequest, LlmResponse};
