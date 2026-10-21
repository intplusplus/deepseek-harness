//! The skill catalog vocabulary.
//!
//! Field order follows the TypeScript object construction in
//! `packages/skill/skill/src/index.ts` (`toSummary`): `name`, `path`,
//! `description`, `whenToUse`, `invocation`, `source`, `provider`,
//! `resourceBase`. Optional keys are omitted rather than serialized as `null`,
//! matching the TypeScript spreads.

use std::fmt;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::merge::RUNTIME_PROVIDER;
use crate::name::SkillName;
use crate::validate::{validate_registration, SkillError, SkillMetadata};

/// Origin bucket for a skill contribution.
///
/// TypeScript models this as a union of known literals plus `(string & {})`:
/// the set is open, so Rust keeps the raw string and exposes the shipped
/// values as constants. The value is prompt-visible metadata, never precedence
/// by itself.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SkillSource(String);

impl SkillSource {
    /// The repository's `.dsh` skill directory.
    pub const PROJECT_DSH: &'static str = "project-dsh";
    /// The repository's `.agents` skill directory.
    pub const PROJECT_AGENTS: &'static str = "project-agents";
    /// An in-memory registration.
    pub const RUNTIME: &'static str = "runtime";
    /// The user's `.dsh` skill directory.
    pub const USER_DSH: &'static str = "user-dsh";
    /// The user's `.agents` skill directory.
    pub const USER_AGENTS: &'static str = "user-agents";
    /// A provider-defined source.
    pub const CUSTOM: &'static str = "custom";
    /// A packaged skill bundle.
    pub const BUNDLED: &'static str = "bundled";

    /// Wrap a provider-defined source label.
    pub fn new(source: impl Into<String>) -> Self {
        Self(source.into())
    }

    /// The source label.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SkillSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A provider-specific base used by a loaded skill body to resolve relative
/// resources.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum SkillResourceBase {
    /// A local directory; relative paths resolve against it.
    Directory {
        /// Absolute directory path.
        path: String,
    },
    /// A remote base; relative URLs resolve against it.
    Url {
        /// Absolute base URL.
        url: String,
    },
    /// Provider-owned resources that are neither a directory nor a URL.
    Opaque {
        /// Human-readable description of the resource location.
        description: String,
    },
}

/// Invocation controls shared by discovery consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillInvocationPolicy {
    /// Whether model-facing catalogs and loaders include this skill.
    pub model_invocable: bool,
    /// Whether human-facing command catalogs and loaders include this skill.
    pub user_invocable: bool,
}

impl Default for SkillInvocationPolicy {
    /// Both surfaces are enabled, matching the registration default.
    fn default() -> Self {
        Self {
            model_invocable: true,
            user_invocable: true,
        }
    }
}

/// A finite precedence rank: lower values win duplicate names within one layer.
///
/// The newtype makes a non-finite rank unrepresentable, which is what lets
/// [`crate::merge::resolve_layer`] sort with a total order instead of handling
/// `NaN` at every comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkillRank(f64);

impl SkillRank {
    /// Validate that `value` is finite.
    pub fn new(value: f64) -> Result<Self, SkillError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(SkillError::NonFiniteRank { rank: value })
        }
    }

    /// Wrap a value the caller already knows is finite.
    pub(crate) const fn from_finite(value: f64) -> Self {
        Self(value)
    }

    /// The rank as a plain number.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl Eq for SkillRank {}

impl PartialOrd for SkillRank {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SkillRank {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Serialize for SkillRank {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for SkillRank {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).map_err(D::Error::custom)
    }
}

/// Invocation-neutral skill metadata returned by catalog discovery.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillSummary {
    /// Kebab-case identifier used to address the skill.
    pub name: SkillName,
    /// Absolute instruction-file path when the provider supplies one; absent
    /// for virtual skills.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Short routing description shown by discovery consumers.
    pub description: String,
    /// Optional extra routing guidance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when_to_use: Option<String>,
    /// Resolved model and user invocation controls.
    pub invocation: SkillInvocationPolicy,
    /// Discovery source that produced this winning skill.
    pub source: SkillSource,
    /// Provider that owns this skill body.
    pub provider: String,
    /// Provider-specific base for relative resources.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_base: Option<SkillResourceBase>,
}

/// A provider catalog entry used by the registry to merge and later load
/// skills.
///
/// `L` is the provider-owned locator handed back to `provider.get()`. The
/// registry never inspects it, so it stays a caller-chosen type rather than a
/// JSON value; merge-only callers can use the `()` default.
#[derive(Clone, Debug, PartialEq)]
pub struct SkillCandidate<L = ()> {
    /// The invocation-neutral metadata used for merging and listing.
    pub summary: SkillSummary,
    /// Lower ranks win duplicate names within one layer.
    pub rank: SkillRank,
    /// Opaque provider handle passed back on load.
    pub locator: L,
    /// Provider-owned frontmatter metadata, when the provider parsed one.
    pub metadata: Option<SkillMetadata>,
}

impl<L> SkillCandidate<L> {
    /// Build a candidate from its summary and provider handle.
    pub fn new(summary: SkillSummary, rank: SkillRank, locator: L) -> Self {
        Self {
            summary,
            rank,
            locator,
            metadata: None,
        }
    }

    /// Attach provider frontmatter metadata.
    pub fn with_metadata(mut self, metadata: Option<SkillMetadata>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Replace the provider handle while keeping the rest of the candidate.
    pub fn map_locator<M>(self, map: impl FnOnce(L) -> M) -> SkillCandidate<M> {
        SkillCandidate {
            summary: self.summary,
            rank: self.rank,
            locator: map(self.locator),
            metadata: self.metadata,
        }
    }
}

/// A complete parsed skill definition, including the instruction body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillDefinition {
    /// The invocation-neutral metadata fields.
    #[serde(flatten)]
    pub summary: SkillSummary,
    /// Markdown instruction body after provider-specific metadata removal.
    pub content: String,
    /// Provider-owned frontmatter metadata, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<SkillMetadata>,
}

impl SkillDefinition {
    /// Build a definition from a summary and its body.
    pub fn new(summary: SkillSummary, content: impl Into<String>) -> Self {
        Self {
            summary,
            content: content.into(),
            metadata: None,
        }
    }

    /// Attach provider frontmatter metadata.
    pub fn with_metadata(mut self, metadata: Option<SkillMetadata>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// A runtime skill contribution accepted by a registry.
///
/// Omitting `invocation` enables both surfaces; omitting `provider` uses the
/// registry-owned `runtime` label, exactly like the TypeScript registration
/// defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct SkillRegistration {
    /// Kebab-case identifier used to address the skill.
    pub name: SkillName,
    /// Short routing description; must not be empty.
    pub description: String,
    /// Optional extra routing guidance.
    pub when_to_use: Option<String>,
    /// Absolute instruction-file path, when the skill has one.
    pub path: Option<String>,
    /// Discovery source recorded on the definition.
    pub source: SkillSource,
    /// Provider-specific base for relative resources.
    pub resource_base: Option<SkillResourceBase>,
    /// Markdown instruction body.
    pub content: String,
    /// Provider-owned frontmatter metadata.
    pub metadata: Option<SkillMetadata>,
    /// Invocation controls; omitted means both surfaces are enabled.
    pub invocation: Option<SkillInvocationPolicy>,
    /// Provider label; omitted uses `runtime`.
    pub provider: Option<String>,
}

impl SkillRegistration {
    /// Validate and resolve the optional fields into a complete definition.
    pub fn into_definition(self) -> Result<SkillDefinition, SkillError> {
        validate_registration(&self)?;
        let provider = self
            .provider
            .unwrap_or_else(|| RUNTIME_PROVIDER.to_string());
        let summary = SkillSummary {
            name: self.name,
            path: self.path,
            description: self.description,
            when_to_use: self.when_to_use,
            invocation: self.invocation.unwrap_or_default(),
            source: self.source,
            provider,
            resource_base: self.resource_base,
        };
        Ok(SkillDefinition {
            summary,
            content: self.content,
            metadata: self.metadata,
        })
    }
}

/// Durable source for the context message a user-explicit skill invocation
/// injects.
///
/// The user's own words ride a plain user message; the rendered body follows
/// as `instructions`-form context carrying this source, so transcript
/// consumers present the injection from metadata instead of re-parsing the
/// model-facing text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillInvocationSource {
    /// The source discriminator.
    pub kind: SkillInvocationSourceKind,
    /// Invoked skill name, validated user-invocable at the injecting boundary.
    pub name: SkillName,
    /// Injected bodies are instructions for the model to follow.
    pub form: SkillInvocationForm,
}

/// The only `kind` value of [`SkillInvocationSource`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillInvocationSourceKind {
    /// A user-explicit skill invocation injected by the host.
    #[serde(rename = "skill-invocation")]
    SkillInvocation,
}

/// The only `form` value of [`SkillInvocationSource`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillInvocationForm {
    /// The body is injected as instructions.
    #[serde(rename = "instructions")]
    Instructions,
}

impl SkillInvocationSource {
    /// Build the source descriptor for a validated skill name.
    pub fn new(name: SkillName) -> Self {
        Self {
            kind: SkillInvocationSourceKind::SkillInvocation,
            name,
            form: SkillInvocationForm::Instructions,
        }
    }
}
