//! `dsh-session-format-catalog` — the session-format codec catalog: which
//! physical codec owns a stored session, and where its migration starts.
//!
//! This is the Rust transcription of the TypeScript
//! `packages/session/session-format-catalog` package (and the generic
//! dispatch machinery it inherits from `session-format`). The migration chain
//! itself lives in [`dsh_session::migration`]; this crate owns only the
//! dispatch layer:
//!
//! - codec registration and lookup for the released generations (v0–v4),
//!   validated with the TypeScript constructor's diagnostics;
//! - header classification (TS `readHeader`): current / migration-required /
//!   unsupported / malformed, with the TypeScript reason texts;
//! - the JSONL read path (TS `createRestore`'s dispatch half): the version
//!   stamped in a stored header selects the chain entry point, and a future
//!   generation is refused before any event row is opened;
//! - one parent's historical child evidence bound into the V3→V4 edge
//!   (TS `createSessionFormatCatalogWithChildren`);
//! - the historical prerequisite catalog that stops at v3
//!   (TS `historicalSessionFormatCatalog`).
//!
//! Developer tool history and agent-preset migration are behaviors of the
//! chain edges (tool-role lift, preset rename), exercised end-to-end through
//! this catalog's dispatch in the integration tests.
//!
//! # Error classification
//!
//! [`CatalogError`] mirrors the TypeScript error pair: [`CatalogError::Format`]
//! reports a damaged artifact, [`CatalogError::UnsupportedMigration`] reports
//! a readable artifact this build refuses to migrate. Both carry the
//! TypeScript diagnostic text verbatim.

#![forbid(unsafe_code)]

mod catalog;
mod error;

pub use catalog::{
    CatalogSpec, HeaderReadResult, HeaderReadStatus, RestoredArtifact, SessionFormatCatalog,
    HISTORICAL_CURRENT_VERSION,
};
pub use error::CatalogError;
