//! The codec catalog: registration, lookup, and version dispatch.
//!
//! This is the Rust transcription of `packages/session/session-format-catalog`
//! (plus the generic machinery it inherits from `session-format`). The
//! migration chain itself lives in [`dsh_session::migration`]; this module
//! owns only *which codec owns a stored version* and how a stored artifact is
//! dispatched into the chain:
//!
//! - `SessionFormatCatalog::shared` — TS `sessionFormatCatalog`, the static
//!   v0..v4 catalog every current read goes through.
//! - `SessionFormatCatalog::with_children` — TS
//!   `createSessionFormatCatalogWithChildren`, one parent's historical child
//!   evidence bound into the v3→v4 edge.
//! - `SessionFormatCatalog::historical` — TS `historicalSessionFormatCatalog`,
//!   a V0–V3 catalog that never touches the current writer.
//!
//! Error texts are transcribed verbatim from the TypeScript sources
//! (`session-format/src/catalog.ts`, `chain.ts`, `json.ts`) so refusal
//! diagnostics match across readers.

use std::collections::BTreeSet;

use dsh_session::migration::{Envelope, Log};
use dsh_session::{SessionError, MAX_SAFE_INTEGER, SESSION_FORMAT_VERSION};
use serde_json::{json, Value};

use crate::error::CatalogError;

/// The historical restoration catalog's current version: it collects
/// migration prerequisites only and never publishes the current writer
/// (TS `historicalSessionFormatCatalog` stops at v3).
pub const HISTORICAL_CURRENT_VERSION: u32 = 3;

/// Name of every compiled migration edge, keyed by its source version. These
/// are the TypeScript package ids the chain names in refusal diagnostics.
fn edge_name(from: u32) -> &'static str {
    match from {
        0 => "@deepseek-ai/dsh-session-format-v0-to-v1",
        1 => "@deepseek-ai/dsh-session-format-v1-to-v2",
        2 => "@deepseek-ai/dsh-session-format-v2-to-v3",
        _ => "@deepseek-ai/dsh-session-format-v3-to-v4",
    }
}

/// Apply one compiled adjacent edge. The Rust chain exposes the four edges as
/// free functions over [`Log`]; the catalog is what maps a stored version to
/// the right entry point.
fn apply_edge(from: u32, log: Log, children: &[Value]) -> Result<Log, SessionError> {
    match from {
        0 => dsh_session::v0_to_v1(log),
        1 => dsh_session::v1_to_v2(log),
        2 => dsh_session::v2_to_v3(log),
        _ => dsh_session::v3_to_v4(log, children),
    }
}

/// The registration manifest a catalog is assembled from.
///
/// The Rust chain is build-static, so the manifest declares *versions* rather
/// than codec objects: one physical codec generation per released format, and
/// the adjacent migration edges mounted at build time. Construction validates
/// the manifest with the TypeScript constructor's checks.
#[derive(Debug, Clone)]
pub struct CatalogSpec {
    /// The version this build writes.
    pub current_version: u32,
    /// Every registered physical codec generation.
    pub codec_versions: Vec<u32>,
    /// Every mounted adjacent migration edge (`from`, `from + 1`).
    pub migration_edges: Vec<(u32, u32)>,
}

/// Classification of one stored physical header (TS `SessionFormatHeaderReadResult`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderReadStatus {
    /// The header is already at the catalog's current version.
    Current,
    /// The header is readable and migrates to the current version.
    MigrationRequired,
    /// The header is readable but this build refuses the migration.
    Unsupported,
    /// The header is damaged beyond dispatch.
    Malformed,
}

/// One classified stored header (TS `SessionFormatHeaderReadResult`).
#[derive(Debug, Clone)]
pub struct HeaderReadResult {
    pub status: HeaderReadStatus,
    /// The version stamped in the stored header, when it carries one.
    pub stored_version: Option<u32>,
    /// The version this build writes.
    pub target_version: u32,
    /// The migrated current-format header, when the read succeeded.
    pub header: Option<Value>,
    /// The verbatim diagnostic, when the read was refused.
    pub reason: Option<String>,
}

/// One restored artifact in the source artifact's physical coordinates
/// (TS `SessionFormatArtifact`): a current-format header plus the migrated
/// events and the fork-inherited prefix length.
#[derive(Debug, Clone)]
pub struct RestoredArtifact {
    pub header: Value,
    pub events: Vec<Envelope>,
    pub inherited_event_count: u64,
}

/// Physical codec dispatch and complete adjacent migration chain.
///
/// Immutable after construction; the catalog owns no per-artifact state, so
/// one instance serves every read.
#[derive(Debug, Clone)]
pub struct SessionFormatCatalog {
    current_version: u32,
    codec_versions: BTreeSet<u32>,
    migration_edges: Vec<(u32, u32)>,
    children: Vec<Value>,
}

impl SessionFormatCatalog {
    /// Validate and assemble a catalog from a manifest.
    ///
    /// Checks are transcribed from the TypeScript constructors
    /// (`session-format/src/catalog.ts` and `chain.ts`): duplicate, missing,
    /// and future codec generations; adjacent, duplicate, and complete
    /// migration edges.
    pub fn assemble(spec: CatalogSpec, children: Vec<Value>) -> Result<Self, CatalogError> {
        // Codec registration: exactly one codec per generation 0..=current,
        // and none newer than current (TS catalog.ts constructor).
        let mut codec_versions = BTreeSet::new();
        for version in spec.codec_versions {
            if !codec_versions.insert(version) {
                return Err(CatalogError::format(format!(
                    "Session format codec v{version} is duplicated"
                )));
            }
        }
        for version in 0..=spec.current_version {
            if !codec_versions.contains(&version) {
                return Err(CatalogError::format(format!(
                    "Session format codec v{version} is missing"
                )));
            }
        }
        if codec_versions.len() != spec.current_version as usize + 1 {
            let invalid = codec_versions
                .iter()
                .copied()
                .find(|version| *version > spec.current_version)
                .expect("a registered codec newer than current");
            return Err(CatalogError::format(format!(
                "Session format codec v{invalid} is newer than current v{}",
                spec.current_version
            )));
        }

        // Migration chain: adjacent, unique, and complete up to current
        // (TS chain.ts constructor).
        let mut edges = Vec::new();
        let mut seen_from = BTreeSet::new();
        for (from, to) in spec.migration_edges {
            if to != from + 1 {
                return Err(CatalogError::format(format!(
                    "Session migration v{from}->v{to} must be adjacent"
                )));
            }
            if !seen_from.insert(from) {
                return Err(CatalogError::format(format!(
                    "Session migration v{from}->v{to} is duplicated"
                )));
            }
            edges.push((from, to));
        }
        for version in 0..spec.current_version {
            if !seen_from.contains(&version) {
                return Err(CatalogError::unsupported(format!(
                    "Session migration v{version}->v{} is missing",
                    version + 1
                )));
            }
        }
        if edges.len() != spec.current_version as usize {
            let invalid = seen_from
                .iter()
                .copied()
                .find(|version| *version >= spec.current_version)
                .expect("a mounted edge that does not lead to current");
            return Err(CatalogError::format(format!(
                "Session migration from v{invalid} does not lead to current v{}",
                spec.current_version
            )));
        }
        edges.sort_by_key(|&(from, _)| from);

        Ok(Self {
            current_version: spec.current_version,
            codec_versions,
            migration_edges: edges,
            children,
        })
    }

    /// The static first-party catalog: every released codec generation, the
    /// complete adjacent chain, and no child evidence (TS `sessionFormatCatalog`).
    pub fn shared() -> Self {
        Self::assemble(static_spec(SESSION_FORMAT_VERSION), Vec::new())
            .expect("the static manifest is complete")
    }

    /// A catalog whose V3→V4 edge knows one parent's historical children
    /// (TS `createSessionFormatCatalogWithChildren`). An empty slice declares
    /// a parent without children.
    pub fn with_children(children: Vec<Value>) -> Self {
        Self::assemble(static_spec(SESSION_FORMAT_VERSION), children)
            .expect("the static manifest is complete")
    }

    /// The historical prerequisite catalog: V0–V3 decoding without the
    /// incoming catalog-completion edge (TS `historicalSessionFormatCatalog`).
    pub fn historical() -> Self {
        Self::assemble(static_spec(HISTORICAL_CURRENT_VERSION), Vec::new())
            .expect("the historical manifest is complete")
    }

    /// The version this catalog writes.
    pub fn current_version(&self) -> u32 {
        self.current_version
    }

    /// Every registered physical codec generation.
    pub fn codec_versions(&self) -> &BTreeSet<u32> {
        &self.codec_versions
    }

    /// The mounted adjacent migration edges.
    pub fn migration_edges(&self) -> &[(u32, u32)] {
        &self.migration_edges
    }

    /// TS `inspectSessionFormatVersion` + `artifactCodec`: the physical codec
    /// that owns a stored version, refused with the TypeScript diagnostics.
    fn codec_version(&self, stored: u64) -> Result<(), CatalogError> {
        let current = u64::from(self.current_version);
        if stored > current {
            return Err(CatalogError::unsupported(newer_text(stored, current)));
        }
        if !self.codec_versions.contains(&(stored as u32)) {
            return Err(CatalogError::unsupported(no_codec_text(stored)));
        }
        Ok(())
    }

    /// TS `readHeader`: classify one stored physical header without opening
    /// its event body.
    pub fn read_header(&self, header: &Value) -> HeaderReadResult {
        let target = self.current_version;
        let stored = match inspect_version(header) {
            Ok(version) => version,
            Err(error) => return malformed(None, target, &error),
        };
        if let Err(error) = self.codec_version(stored) {
            return HeaderReadResult {
                status: HeaderReadStatus::Unsupported,
                stored_version: Some(stored as u32),
                target_version: target,
                header: None,
                reason: Some(error.message().to_string()),
            };
        }
        // The codec check bounds the version by `current`, so it fits `u32`.
        let stored = stored as u32;
        match self.migrate_header_only(header) {
            Ok(migrated) => HeaderReadResult {
                status: if stored == target {
                    HeaderReadStatus::Current
                } else {
                    HeaderReadStatus::MigrationRequired
                },
                stored_version: Some(stored),
                target_version: target,
                header: Some(migrated),
                reason: None,
            },
            Err(error) => match error {
                CatalogError::UnsupportedMigration(reason) => HeaderReadResult {
                    status: HeaderReadStatus::Unsupported,
                    stored_version: Some(stored),
                    target_version: target,
                    header: None,
                    reason: Some(reason),
                },
                other => malformed(Some(stored), target, &other),
            },
        }
    }

    /// Decode one header at its stored version and migrate it through the
    /// mounted chain (TS `codec.decodeHeader` + `chain.migrateHeader`).
    ///
    /// The header is admitted through `Log::parse` so it receives the same
    /// per-version physical validation as a stored file's first line. With no
    /// event rows the inherited cut must be zero, which is why a seeded
    /// header-only read is refused here earlier than the TypeScript reader
    /// would refuse it (it defers the marker check to decoder finish).
    fn migrate_header_only(&self, header: &Value) -> Result<Value, CatalogError> {
        let line = serde_json::to_string(header)
            .map_err(|error| CatalogError::format(error.to_string()))?;
        let decoded = Log::parse(&line)?;
        let migrated = self.run_chain(
            Log::new(decoded.header, Vec::new(), decoded.cut),
            "Session header",
        )?;
        Ok(finalize_header(migrated.header, self.current_version))
    }

    /// Select the codec for one parsed stored log and migrate it to this
    /// catalog's current version. This is the dispatch core behind the JSONL
    /// read path: the stored version picks the chain entry point, and the
    /// catalog's child evidence feeds the V3→V4 edge.
    pub fn restore_log(&self, log: Log) -> Result<RestoredArtifact, CatalogError> {
        let stored = log.version().map_err(CatalogError::from)?;
        self.codec_version(u64::from(stored))?;
        let migrated = self.run_chain(log, "Session")?;
        Ok(RestoredArtifact {
            header: finalize_header(migrated.header, self.current_version),
            events: migrated.events,
            inherited_event_count: migrated.cut,
        })
    }

    /// Read one stored JSONL artifact: classify its header, select the codec
    /// by the stamped format version, and migrate every row to this catalog's
    /// current version. A future generation is refused before any event row
    /// is opened (TS `artifactCodec`).
    pub fn restore_jsonl(&self, text: &str) -> Result<RestoredArtifact, CatalogError> {
        let header_line = text
            .lines()
            .next()
            .filter(|line| !line.trim().is_empty())
            .ok_or_else(|| CatalogError::format("stored log is empty"))?;
        let raw: Value = serde_json::from_str(header_line)
            .map_err(|error| CatalogError::format(error.to_string()))?;
        let stored = inspect_version(&raw)?;
        self.codec_version(stored)?;
        let log = Log::parse(text)?;
        self.restore_log(log)
    }

    /// Run the mounted adjacent edges from the stored version to this
    /// catalog's current version, wrapping every refusal with the TypeScript
    /// chain's diagnostic (`throwUnsupportedRefusal`).
    fn run_chain(&self, log: Log, subject: &str) -> Result<Log, CatalogError> {
        let mut current = log;
        let mut from = current.version().map_err(CatalogError::from)?;
        while from < self.current_version {
            let next = from + 1;
            let migrated = apply_edge(from, current, &self.children)
                .map_err(|error| CatalogError::unsupported(refusal_text(from, subject, &error)))?;
            let advanced = migrated.version().map_err(CatalogError::from)?;
            if advanced != next {
                return Err(CatalogError::format(format!(
                    "{} returned v{advanced}; expected v{next}",
                    edge_name(from),
                )));
            }
            current = migrated;
            from = next;
        }
        Ok(current)
    }
}

/// The static manifest for a catalog over the compiled chain: one codec per
/// generation and every adjacent edge up to `current_version`.
fn static_spec(current_version: u32) -> CatalogSpec {
    CatalogSpec {
        current_version,
        codec_versions: (0..=current_version).collect(),
        migration_edges: (0..current_version).map(|from| (from, from + 1)).collect(),
    }
}

/// TS `inspectSessionFormatVersion`: read only the version required for
/// directional dispatch, refusing anything that is not a non-negative safe
/// integer with the TypeScript diagnostic.
fn inspect_version(header: &Value) -> Result<u64, CatalogError> {
    let map = header
        .as_object()
        .ok_or_else(|| CatalogError::format("Session header must be a JSON object"))?;
    let version = map.get("version").ok_or_else(bad_version_text)?;
    version
        .as_u64()
        .filter(|value| *value <= MAX_SAFE_INTEGER)
        .ok_or_else(bad_version_text)
}

fn bad_version_text() -> CatalogError {
    CatalogError::format("Session format version must be a non-negative safe integer")
}

fn newer_text(stored: u64, current: u64) -> String {
    format!("stored Session uses newer format v{stored}; this build writes v{current}")
}

fn no_codec_text(stored: u64) -> String {
    format!("this build has no Session format codec for v{stored}")
}

/// The bare detail a chain edge reports. TS wraps the *message* of the inner
/// error, so the `SessionError` prefixes are stripped here to keep the wrapped
/// diagnostic identical.
fn detail_of(error: &SessionError) -> String {
    match error {
        SessionError::Migration(detail) => detail.clone(),
        SessionError::UnsupportedVersion { found, current } => {
            newer_text(u64::from(*found), u64::from(*current))
        }
        other => other.to_string(),
    }
}

/// TS `throwUnsupportedRefusal`: the chain wraps every edge failure as a
/// migration refusal naming the edge and the format it came from.
fn refusal_text(from: u32, subject: &str, error: &SessionError) -> String {
    format!(
        "{} refuses this format v{from} {subject}: {}",
        edge_name(from),
        detail_of(error),
    )
}

/// One `malformed` classification (TS `malformed` in `catalog.ts`).
fn malformed(
    stored_version: Option<u32>,
    target_version: u32,
    error: &CatalogError,
) -> HeaderReadResult {
    HeaderReadResult {
        status: HeaderReadStatus::Malformed,
        stored_version,
        target_version,
        header: None,
        reason: Some(error.message().to_string()),
    }
}

/// Replicates `dsh_session::migration`'s finalization: drop the physical
/// `type` tag, pin the current version, and materialize the seeded flag.
fn finalize_header(mut header: Value, current_version: u32) -> Value {
    if let Some(map) = header.as_object_mut() {
        map.shift_remove("type");
        map.insert("version".to_string(), json!(current_version));
        if !map.contains_key("isSeeded") {
            map.insert("isSeeded".to_string(), json!(false));
        }
    }
    header
}
