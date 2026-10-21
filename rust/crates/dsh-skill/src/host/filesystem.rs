//! The local filesystem skill provider, ported from
//! `packages/skill/skill-filesystem`.
//!
//! The provider discovers directory-bundle (`<dir>/SKILL.md`) and flat
//! Markdown (`<dir>/<name>.md`) skills from project, custom, user, and
//! bundled roots, parses a constrained YAML frontmatter, and loads bodies
//! through the injected [`SkillFiles`] trait so tests can use real temporary
//! directories without touching the host filesystem directly.
//!
//! ## What is ported from the watcher
//!
//! Only the invalidation notification logic is portable without platform
//! APIs: [`is_relevant_watch_event`] and [`is_potential_skill_path`] decide
//! whether a path change under a root matters, and
//! [`FsSkillProvider::observe_host_mutation`] turns a first-party mutation
//! into a cache invalidation. The Chokidar / `watchFile` platform watchers,
//! the ancestor-watch fallback for missing roots, and the bounded project
//! watcher bookkeeping stay in the TypeScript host — they own timers and
//! native filesystem events, not decisions.

use std::cell::RefCell;
use std::io;
use std::path::{Path, PathBuf};

use crate::merge::BUNDLED_SKILL_RANK;
use crate::name::{is_skill_name, SkillName};
use crate::types::{
    SkillCandidate, SkillDefinition, SkillInvocationPolicy, SkillRank, SkillResourceBase,
    SkillSource, SkillSummary,
};
use crate::validate::{validate_candidate, SkillMetadata};

use super::provider::{ProviderError, ProviderObservation, SkillLookupOptions};
use super::{ProviderControl, SkillProvider};

/// Precedence rank of the project `.dsh/skills` root.
pub const PROJECT_DSH_RANK: f64 = 100.0;
/// Precedence rank of the project `.agents/skills` root.
pub const PROJECT_AGENTS_RANK: f64 = 200.0;
/// Precedence rank of custom skill roots.
pub const USER_DSH_RANK: f64 = 400.0;
/// Precedence rank of the user `.agents/skills` root.
pub const USER_AGENTS_RANK: f64 = 500.0;
/// Precedence rank of custom skill roots.
pub const CUSTOM_RANK: f64 = 300.0;

/// One skill root directory scanned by the filesystem provider.
#[derive(Clone, Debug, PartialEq)]
pub struct SkillRoot {
    /// Absolute root directory path.
    pub path: String,
    /// Origin bucket recorded on discovered candidates.
    pub source: SkillSource,
    /// Root precedence; lower ranks win duplicate names within one layer.
    pub rank: f64,
    /// Whether the `.system` subtree is excluded from discovery.
    pub skip_system: bool,
}

/// A directory entry kind reported by [`SkillFiles::list_dir`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsEntryKind {
    /// A directory; discovered as `<name>/SKILL.md`.
    Directory,
    /// A regular file; discovered when the name ends in `.md`.
    File,
    /// Any other entry kind, skipped by discovery.
    Other,
}

/// One directory entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FsEntry {
    /// The entry's file name.
    pub name: String,
    /// The entry's kind.
    pub kind: FsEntryKind,
}

/// The file operations the filesystem provider needs.
///
/// This is the injected seam standing in for the TypeScript `ctx.fs` service
/// plus the direct `node:fs` calls. Implementations over real temporary
/// directories keep the discovery tests end-to-end.
pub trait SkillFiles {
    /// Whether a path exists.
    fn path_exists(&self, path: &str) -> bool;

    /// List a directory's entries; absent paths are an empty listing's
    /// counterpart expressed through `io::ErrorKind::NotFound`.
    ///
    /// # Errors
    /// Any non-absent I/O failure, which the provider turns into an
    /// incomplete, uncached collect.
    fn list_dir(&self, path: &str) -> io::Result<Vec<FsEntry>>;

    /// Read a file's text.
    ///
    /// # Errors
    /// Any I/O failure; absent paths produce `io::ErrorKind::NotFound`.
    fn read_text(&self, path: &str) -> io::Result<String>;
}

/// The [`SkillFiles`] implementation over the process filesystem.
#[derive(Clone, Copy, Debug, Default)]
pub struct StdFiles;

impl SkillFiles for StdFiles {
    fn path_exists(&self, path: &str) -> bool {
        Path::new(path).exists()
    }

    fn list_dir(&self, path: &str) -> io::Result<Vec<FsEntry>> {
        let mut entries = Vec::new();
        for entry in fs_read_dir(path)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let kind = if file_type.is_dir() {
                FsEntryKind::Directory
            } else if file_type.is_file() {
                FsEntryKind::File
            } else {
                FsEntryKind::Other
            };
            entries.push(FsEntry {
                name: entry.file_name().to_string_lossy().into_owned(),
                kind,
            });
        }
        Ok(entries)
    }

    fn read_text(&self, path: &str) -> io::Result<String> {
        fs_read_to_string(path)
    }
}

fn fs_read_dir(path: &str) -> io::Result<std::fs::ReadDir> {
    std::fs::read_dir(path)
}

fn fs_read_to_string(path: &str) -> io::Result<String> {
    std::fs::read_to_string(path)
}

/// Filesystem provider configuration.
///
/// Homes default to `$DSH_HOME` / `~/.dsh` and `$DSH_AGENTS_HOME` /
/// `~/.agents`; the bundled root defaults to `$DSH_BUNDLED_SKILL_DIR` when
/// default roots are included, otherwise no bundled root is mounted — an
/// isolated provider must see only its explicit roots.
#[derive(Clone, Debug, Default)]
pub struct FsProviderConfig {
    /// Unique provider name; defaults to `filesystem`.
    pub provider_name: Option<String>,
    /// Whether project and user roots are included around custom roots;
    /// defaults to `true`.
    pub include_default_roots: Option<bool>,
    /// DeepSeek Harness config root; defaults to `$DSH_HOME` or `~/.dsh`.
    pub dsh_home: Option<String>,
    /// Shared agent config root; defaults to `$DSH_AGENTS_HOME` or
    /// `~/.agents`.
    pub agents_home: Option<String>,
    /// Additional skill roots scanned after project roots and before user
    /// roots.
    pub custom_skill_dirs: Option<Vec<String>>,
    /// Bundled skill root; see the type-level default rules.
    pub bundled_skill_dir: Option<String>,
}

/// The provider-owned locator for a local skill body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalLocator {
    /// The instruction file path handed to the parser on load.
    pub path: String,
    /// The directory relative resources resolve against.
    pub directory: String,
}

struct ParsedSkill {
    name: SkillName,
    description: String,
    when_to_use: Option<String>,
    invocation: SkillInvocationPolicy,
    metadata: Option<SkillMetadata>,
    path: String,
    content: String,
}

/// Provider that maps local project/user skill roots into the host registry.
pub struct FsSkillProvider<F: SkillFiles> {
    files: F,
    provider_name: String,
    include_default_roots: bool,
    dsh_home: String,
    agents_home: String,
    custom_skill_dirs: Vec<String>,
    bundled_skill_dir: Option<String>,
    control: ProviderControl<LocalLocator>,
    warnings: RefCell<Vec<String>>,
    /// Roots observed by the most recent discovery; the mutation filter
    /// checks incoming host-mutation paths against them.
    watched_roots: RefCell<Vec<SkillRoot>>,
}

impl<F: SkillFiles> FsSkillProvider<F> {
    /// Build the provider over the given file operations.
    pub fn new(files: F, config: FsProviderConfig, control: ProviderControl<LocalLocator>) -> Self {
        let include_default_roots = config.include_default_roots.unwrap_or(true);
        let bundled_skill_dir = config.bundled_skill_dir.or_else(|| {
            if include_default_roots {
                std::env::var("DSH_BUNDLED_SKILL_DIR").ok()
            } else {
                None
            }
        });
        Self {
            files,
            provider_name: config
                .provider_name
                .unwrap_or_else(|| "filesystem".to_string()),
            include_default_roots,
            dsh_home: config
                .dsh_home
                .unwrap_or_else(|| default_home("DSH_HOME", ".dsh")),
            agents_home: config
                .agents_home
                .unwrap_or_else(|| default_home("DSH_AGENTS_HOME", ".agents")),
            custom_skill_dirs: config.custom_skill_dirs.unwrap_or_default(),
            bundled_skill_dir,
            control,
            warnings: RefCell::new(Vec::new()),
            watched_roots: RefCell::new(Vec::new()),
        }
    }

    /// Drain the warnings recorded so far: ignored skill files and read
    /// failures.
    pub fn take_warnings(&self) -> Vec<String> {
        std::mem::take(&mut *self.warnings.borrow_mut())
    }

    /// Invalidate the host catalog synchronously after a first-party
    /// filesystem mutation, when the path could be a skill entry.
    pub fn observe_host_mutation(&self, path: &str) {
        if self
            .watched_roots
            .borrow()
            .iter()
            .any(|root| is_potential_skill_path(root, path))
        {
            self.control.invalidate();
        }
    }

    /// The roots scanned for a workspace, in discovery order.
    fn roots(&self, cwd: Option<&str>) -> Vec<SkillRoot> {
        let mut roots = Vec::new();
        if self.include_default_roots {
            if let Some(cwd) = cwd {
                let project_root = find_project_root(Path::new(cwd), &self.files);
                roots.push(SkillRoot {
                    path: join_path(Path::new(&project_root), &[".dsh", "skills"]),
                    source: SkillSource::new(SkillSource::PROJECT_DSH),
                    rank: PROJECT_DSH_RANK,
                    skip_system: false,
                });
                roots.push(SkillRoot {
                    path: join_path(Path::new(&project_root), &[".agents", "skills"]),
                    source: SkillSource::new(SkillSource::PROJECT_AGENTS),
                    rank: PROJECT_AGENTS_RANK,
                    skip_system: false,
                });
            }
        }
        for path in &self.custom_skill_dirs {
            roots.push(SkillRoot {
                path: path.clone(),
                source: SkillSource::new(SkillSource::CUSTOM),
                rank: CUSTOM_RANK,
                skip_system: false,
            });
        }
        if self.include_default_roots {
            roots.push(SkillRoot {
                path: join_path(Path::new(&self.dsh_home), &["skills"]),
                source: SkillSource::new(SkillSource::USER_DSH),
                rank: USER_DSH_RANK,
                skip_system: true,
            });
            roots.push(SkillRoot {
                path: join_path(Path::new(&self.agents_home), &["skills"]),
                source: SkillSource::new(SkillSource::USER_AGENTS),
                rank: USER_AGENTS_RANK,
                skip_system: false,
            });
        }
        if let Some(path) = &self.bundled_skill_dir {
            roots.push(SkillRoot {
                path: path.clone(),
                source: SkillSource::new(SkillSource::BUNDLED),
                rank: BUNDLED_SKILL_RANK,
                skip_system: false,
            });
        }
        roots
    }

    /// Discover one root's candidates. Absent roots contribute nothing;
    /// other I/O failures fail the whole discovery, matching the TypeScript
    /// containment where the host skips a failing provider entirely.
    fn discover_root(
        &self,
        root: &SkillRoot,
    ) -> Result<Vec<SkillCandidate<LocalLocator>>, ProviderError> {
        let mut entries = match self.files.list_dir(&root.path) {
            Ok(entries) => entries,
            Err(error) if is_absent_io_error(&error) => return Ok(Vec::new()),
            Err(error) => {
                return Err(ProviderError(format!(
                    "failed to list skill root {}: {error}",
                    root.path
                )));
            }
        };
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        let mut skills = Vec::new();
        for entry in entries {
            if root.skip_system && entry.name == ".system" {
                continue;
            }
            let entry_path = join_path(Path::new(&root.path), &[&entry.name]);
            let locator = match entry.kind {
                FsEntryKind::Directory => LocalLocator {
                    path: join_path(Path::new(&entry_path), &["SKILL.md"]),
                    directory: entry_path,
                },
                FsEntryKind::File if entry.name.ends_with(".md") => LocalLocator {
                    path: entry_path,
                    directory: root.path.clone(),
                },
                _ => continue,
            };
            let Some(parsed) = self.parse_skill_file(&locator.path) else {
                continue;
            };
            let rank = SkillRank::new(root.rank).expect("root ranks are finite constants");
            let summary = SkillSummary {
                name: parsed.name,
                path: Some(parsed.path),
                description: parsed.description,
                when_to_use: parsed.when_to_use,
                invocation: parsed.invocation,
                source: root.source.clone(),
                provider: self.provider_name.clone(),
                resource_base: Some(SkillResourceBase::Directory {
                    path: locator.directory.clone(),
                }),
            };
            let candidate_path = locator.path.clone();
            let candidate =
                SkillCandidate::new(summary, rank, locator).with_metadata(parsed.metadata);
            if let Err(error) = validate_candidate(&candidate, &self.provider_name) {
                self.warn(format!("skill file {candidate_path} ignored: {error}"));
                continue;
            }
            skills.push(candidate);
        }
        Ok(skills)
    }

    /// Read and parse one SKILL.md-style instruction file; a skill file that
    /// cannot be admitted is warned about and skipped, never fatal.
    fn parse_skill_file(&self, path: &str) -> Option<ParsedSkill> {
        let raw = match self.files.read_text(path) {
            Ok(text) => text,
            Err(error) if is_absent_io_error(&error) => return None,
            Err(error) => {
                self.warn(format!("skill file {path} ignored: {error}"));
                return None;
            }
        };
        let (fields, body) = match parse_frontmatter(&raw) {
            Ok(Some(parsed)) => parsed,
            Ok(None) => {
                self.warn(format!(
                    "skill file {path} ignored: missing YAML frontmatter"
                ));
                return None;
            }
            Err(error) => {
                self.warn(format!(
                    "skill file {path} ignored: invalid YAML frontmatter: {error}"
                ));
                return None;
            }
        };
        let Some(name) = fields.string_field("name") else {
            self.warn(format!(
                "skill file {path} ignored: frontmatter requires name and description"
            ));
            return None;
        };
        let Some(description) = fields.string_field("description") else {
            self.warn(format!(
                "skill file {path} ignored: frontmatter requires name and description"
            ));
            return None;
        };
        if !is_skill_name(&name) {
            self.warn(format!(
                "skill file {path} ignored: invalid skill name \"{name}\""
            ));
            return None;
        }
        let invocation = match parse_invocation_policy(&fields) {
            Ok(policy) => policy,
            Err(error) => {
                self.warn(format!(
                    "skill file {path} ignored: invalid invocation frontmatter: {error}"
                ));
                return None;
            }
        };
        Some(ParsedSkill {
            name: SkillName::new(name).expect("the grammar check passed"),
            description,
            when_to_use: fields.string_field("whenToUse"),
            invocation,
            metadata: fields.metadata,
            path: path.to_string(),
            content: body.trim().to_string(),
        })
    }

    fn warn(&self, message: String) {
        self.warnings.borrow_mut().push(message);
    }
}

impl<F: SkillFiles> SkillProvider<LocalLocator> for FsSkillProvider<F> {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn list(
        &self,
        options: &SkillLookupOptions,
    ) -> Result<ProviderObservation<LocalLocator>, ProviderError> {
        let roots = self.roots(options.cwd.as_deref());
        *self.watched_roots.borrow_mut() = roots.clone();
        let mut candidates = Vec::new();
        for root in &roots {
            for candidate in self.discover_root(root)? {
                candidates.push(candidate);
            }
        }
        // Watcher startup failures cannot occur without platform watchers,
        // so filesystem discovery is always reported complete.
        Ok(ProviderObservation::Complete(candidates))
    }

    fn get(
        &self,
        candidate: &SkillCandidate<LocalLocator>,
        _options: &SkillLookupOptions,
    ) -> Result<Option<SkillDefinition>, ProviderError> {
        let locator = &candidate.locator;
        let Some(parsed) = self.parse_skill_file(&locator.path) else {
            return Ok(None);
        };
        Ok(Some(SkillDefinition {
            summary: SkillSummary {
                name: parsed.name,
                path: Some(parsed.path),
                description: parsed.description,
                when_to_use: parsed.when_to_use,
                invocation: parsed.invocation,
                source: candidate.summary.source.clone(),
                provider: self.provider_name.clone(),
                resource_base: Some(SkillResourceBase::Directory {
                    path: locator.directory.clone(),
                }),
            },
            content: parsed.content,
            metadata: parsed.metadata,
        }))
    }
}

/// The watcher events the TypeScript Chokidar handle observes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WatchEvent {
    /// A file appeared.
    Add,
    /// A directory appeared.
    AddDir,
    /// A file changed.
    Change,
    /// A file disappeared.
    Unlink,
    /// A directory disappeared.
    UnlinkDir,
}

/// Whether a watcher event under a root is relevant to the skill catalog.
///
/// The port of the TypeScript `isRelevantWatchEvent`: the root itself only
/// matters when it appears or disappears, a top-level `.md` file matters,
/// and a two-segment path matters exactly when it is a skill bundle's
/// `SKILL.md`.
pub fn is_relevant_watch_event(root: &SkillRoot, event: WatchEvent, path: &str) -> bool {
    let Some(segments) = contained_segments(&root.path, path) else {
        return false;
    };
    if segments.is_empty() {
        return matches!(event, WatchEvent::AddDir | WatchEvent::UnlinkDir);
    }
    if root.skip_system && segments[0] == ".system" {
        return false;
    }
    if segments.len() == 1 {
        if matches!(event, WatchEvent::AddDir | WatchEvent::UnlinkDir) {
            return true;
        }
        return segments[0].ends_with(".md");
    }
    segments.len() == 2
        && segments[1] == "SKILL.md"
        && !matches!(event, WatchEvent::AddDir | WatchEvent::UnlinkDir)
}

/// Whether a path could be a skill entry under a root, used to filter
/// first-party mutation notifications.
///
/// The port of the TypeScript `isPotentialSkillPath`.
pub fn is_potential_skill_path(root: &SkillRoot, path: &str) -> bool {
    let Some(segments) = contained_segments(&root.path, path) else {
        return false;
    };
    if segments.is_empty() || segments.len() > 2 {
        return false;
    }
    if root.skip_system && segments[0] == ".system" {
        return false;
    }
    if segments.len() == 1 {
        return segments[0].ends_with(".md");
    }
    segments[1] == "SKILL.md"
}

/// The path's segments below the root: empty for the root itself, `None`
/// when the path is outside it.
fn contained_segments(root: &str, path: &str) -> Option<Vec<String>> {
    let child = Path::new(path).strip_prefix(root).ok()?;
    Some(
        child
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect(),
    )
}

/// Whether an I/O error means the path is absent.
fn is_absent_io_error(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::NotFound
}

fn join_path(base: &Path, segments: &[&str]) -> String {
    let mut path = base.to_path_buf();
    for segment in segments {
        path.push(segment);
    }
    path.to_string_lossy().into_owned()
}

/// Walk upward from the workspace to the nearest directory containing a
/// `.git` entry, falling back to the workspace itself, matching the
/// TypeScript `findProjectRoot`.
fn find_project_root(cwd: &Path, files: &impl SkillFiles) -> String {
    let mut current = cwd.to_path_buf();
    loop {
        if files.path_exists(&join_path(&current, &[".git"])) {
            return current.to_string_lossy().into_owned();
        }
        match current.parent() {
            Some(parent) if parent != current => current = parent.to_path_buf(),
            _ => return cwd.to_string_lossy().into_owned(),
        }
    }
}

fn default_home(env_key: &str, subdir: &str) -> String {
    if let Ok(home) = std::env::var(env_key) {
        return home;
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from);
    match home {
        Ok(home) => home.join(subdir).to_string_lossy().into_owned(),
        Err(_) => subdir.to_string(),
    }
}

/// Parsed frontmatter: top-level scalar fields plus an optional one-level
/// `metadata` map.
struct FrontmatterFields {
    scalars: Vec<(String, String)>,
    metadata: Option<SkillMetadata>,
}

impl FrontmatterFields {
    /// The last non-empty value for `key`, when present.
    fn string_field(&self, key: &str) -> Option<String> {
        let value = self
            .scalars
            .iter()
            .rev()
            .find(|(existing, _)| existing == key)
            .map(|(_, value)| value)?;
        (!value.is_empty()).then(|| value.clone())
    }

    fn contains_key(&self, key: &str) -> bool {
        self.scalars.iter().any(|(existing, _)| existing == key)
    }

    fn boolean_field(&self, key: &str) -> Result<Option<bool>, String> {
        let Some(value) = self
            .scalars
            .iter()
            .rev()
            .find(|(existing, _)| existing == key)
            .map(|(_, value)| value.as_str())
        else {
            return Ok(None);
        };
        parse_frontmatter_boolean(key, value).map(Some)
    }
}

/// Split a SKILL.md into its frontmatter fields and body.
///
/// Returns `Ok(None)` when the file has no frontmatter at all and `Err` for
/// malformed frontmatter, mirroring the TypeScript `parseFrontmatter` split
/// between "missing" and "invalid".
fn parse_frontmatter(raw: &str) -> Result<Option<(FrontmatterFields, String)>, String> {
    let Some(first_line_end) = raw.find('\n') else {
        return Ok(None);
    };
    let first_line = trim_cr(&raw[..first_line_end]);
    if first_line != "---" {
        return Ok(None);
    }
    let mut line_start = first_line_end + 1;
    let (closing_start, body_start) = loop {
        let Some(next_newline) = raw[line_start..].find('\n') else {
            if trim_cr(&raw[line_start..]) == "---" {
                break (line_start, raw.len());
            }
            return Ok(None);
        };
        let line_end = line_start + next_newline;
        if trim_cr(&raw[line_start..line_end]) == "---" {
            break (line_start, line_end + 1);
        }
        line_start = line_end + 1;
    };
    let fields = parse_yaml_fields(&raw[first_line_end + 1..closing_start])?;
    Ok(Some((fields, raw[body_start..].to_string())))
}

fn trim_cr(line: &str) -> &str {
    line.strip_suffix('\r').unwrap_or(line)
}

/// Parse the constrained YAML block: flat `key: value` scalars plus a
/// one-level indented map under `metadata`. Deeper structures are rejected
/// rather than misread — the TypeScript provider uses a full YAML parser,
/// which stays there; the fields this harness reads are all scalar.
fn parse_yaml_fields(yaml: &str) -> Result<FrontmatterFields, String> {
    let mut fields = FrontmatterFields {
        scalars: Vec::new(),
        metadata: None,
    };
    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            let Some(map) = fields.metadata.as_mut() else {
                return Err("unsupported frontmatter structure: unexpected indentation".to_string());
            };
            let (key, value) = split_field(trimmed)?;
            map.insert(key, scalar_value(&value));
            continue;
        }
        let (key, value) = split_field(line)?;
        if value.is_empty() {
            if key == "metadata" {
                fields.metadata = Some(SkillMetadata::new());
                continue;
            }
            return Err(format!("frontmatter field \"{key}\" has no value"));
        }
        fields.scalars.push((key.to_string(), unquote(&value)));
    }
    Ok(fields)
}

fn split_field(line: &str) -> Result<(String, String), String> {
    if let Some((key, value)) = line.split_once(": ") {
        return Ok((key.trim().to_string(), value.trim().to_string()));
    }
    if let Some(key) = line.strip_suffix(':') {
        return Ok((key.trim().to_string(), String::new()));
    }
    Err(format!("unsupported frontmatter line: {line}"))
}

fn unquote(value: &str) -> String {
    let bytes = value.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return value[1..value.len() - 1].to_string();
        }
    }
    value.to_string()
}

/// Coerce a metadata scalar into the JSON value the TypeScript YAML parser
/// would produce: booleans, numbers, then strings.
fn scalar_value(raw: &str) -> serde_json::Value {
    match raw {
        "true" => serde_json::Value::Bool(true),
        "false" => serde_json::Value::Bool(false),
        other => {
            if let Ok(integer) = other.parse::<i64>() {
                return serde_json::Value::from(integer);
            }
            if let Ok(float) = other.parse::<f64>() {
                if float.is_finite() {
                    return serde_json::Value::from(float);
                }
            }
            serde_json::Value::String(unquote(other))
        }
    }
}

fn parse_frontmatter_boolean(key: &str, value: &str) -> Result<bool, String> {
    match value {
        "1" => Ok(true),
        "0" => Ok(false),
        other => match other.to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" => Ok(true),
            "false" | "no" | "off" => Ok(false),
            _ => Err(format!("frontmatter field \"{key}\" must be a boolean")),
        },
    }
}

/// Resolve the invocation policy from frontmatter, rejecting the legacy
/// key spellings the TypeScript provider rejects.
fn parse_invocation_policy(fields: &FrontmatterFields) -> Result<SkillInvocationPolicy, String> {
    reject_legacy_key(fields, "disableModelInvocation", "disable-model-invocation")?;
    reject_legacy_key(fields, "modelInvocable", "disable-model-invocation")?;
    reject_legacy_key(fields, "userInvocable", "user-invocable")?;
    let disable_model_invocation = fields.boolean_field("disable-model-invocation")?;
    let user_invocable = fields.boolean_field("user-invocable")?;
    Ok(SkillInvocationPolicy {
        model_invocable: disable_model_invocation != Some(true),
        user_invocable: user_invocable != Some(false),
    })
}

fn reject_legacy_key(
    fields: &FrontmatterFields,
    legacy: &str,
    canonical: &str,
) -> Result<(), String> {
    if fields.contains_key(legacy) {
        return Err(format!(
            "frontmatter field \"{legacy}\" is unsupported; use \"{canonical}\""
        ));
    }
    Ok(())
}
