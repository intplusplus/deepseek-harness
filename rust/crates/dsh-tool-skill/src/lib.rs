//! `dsh-tool-skill` — the durable session skill catalog and the model-facing
//! `skill` loader tool.
//!
//! Transcribed from `packages/skill/tool-skill`. The TypeScript package is a
//! Cordis plugin that registers the loader tool and two `agent/pre-step`
//! listeners: one injecting user-explicit `/name` invocations, one publishing
//! and replacing the durable session catalog. The Rust crate keeps the parts
//! with a deterministic answer without a host:
//!
//! - [`digest`] — the SHA-256 identity over the published entry list, locked
//!   with the NIST vectors.
//! - [`catalog`] — entry normalization, the two rendered model-facing frames,
//!   the tolerant read of a published record, and the session-history fold.
//! - [`gesture`] — the `/name` gesture scan over claimed user messages.
//! - [`tool`] — the loader itself, a [`dsh_tools::ToolHandler`].
//!
//! # Not ported, and why
//!
//! - **The Cordis lifecycle.** The tool registers, and the two pre-step
//!   listeners subscribe, through a context; the Rust crate exposes pure
//!   functions and a handler, and a deployment composes them.
//! - **The agent/pre-step listeners.** `dsh-agent` exposes no pre-step seam,
//!   so the publication decision is exposed as data
//!   ([`catalog::catalog_history`] plus the two render functions) and the
//!   injection path as [`gesture::invoked_skill_names`] plus
//!   `dsh-skill`'s renderer.
//! - **The working-directory resolution.** The TypeScript lookup resolves the
//!   agent's working directory before listing; the injected
//!   [`tool::SkillCatalog`] seam owns that.
//!
//! Nothing here reads the clock, the filesystem, the environment, or a random
//! source.

#![forbid(unsafe_code)]

pub mod catalog;
pub mod digest;
pub mod gesture;
pub mod tool;

pub use catalog::{
    catalog_description, catalog_history, catalog_message, catalog_source_entries,
    read_catalog_entries, render_catalog_message, render_catalog_update, CatalogHistory,
    PublishedCatalog, DEFAULT_CATALOG_DESCRIPTION_MAX_LENGTH, SKILL_CATALOG_SOURCE_KIND,
};
pub use digest::{digest_catalog_entries, sha256_hex, CatalogEntry};
pub use gesture::{invoked_skill_names, scan_gestures};
pub use tool::{fn_skill_catalog, SkillCatalog, SkillTool, SkillToolError, SKILL_TOOL_NAME};
