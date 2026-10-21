//! `dsh-provider-deepseek` — the real DeepSeek provider.
//!
//! Skeleton only: the HTTP/SSE transport, wire types, and streaming assembly
//! are ported from `packages/llm/llm-deepseek` behind the canonical
//! [`dsh_provider::LlmProvider`] trait.
#![forbid(unsafe_code)]

mod config;

pub use config::{DeepSeekConfig, DeepSeekProvider};
