//! The provider seam: discovery and loading contracts for one skill source.
//!
//! This is the TypeScript `SkillProvider` interface from
//! `packages/skill/skill/src/index.ts` with `Promise` removed: the Rust host
//! is synchronous, so discovery and loading are plain calls and there is no
//! await point for an uncooperative provider to hang on. The `AbortSignal`
//! members of the TypeScript lookup options therefore have no counterpart.

use crate::types::{SkillCandidate, SkillDefinition};

/// Workspace selector for cwd-sensitive provider work.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SkillLookupOptions {
    /// Workspace selector; `None` reads only cwd-independent roots.
    pub cwd: Option<String>,
}

impl SkillLookupOptions {
    /// Options with a workspace selector.
    pub fn with_cwd(cwd: impl Into<String>) -> Self {
        Self {
            cwd: Some(cwd.into()),
        }
    }
}

/// A provider discovery result.
///
/// The complete vector is the array shorthand TypeScript providers return;
/// the incomplete variant reports candidates from a discovery that could not
/// finish. Incomplete candidates stay loadable but the host never caches the
/// collect result that produced them.
#[derive(Clone, Debug)]
pub enum ProviderObservation<L> {
    /// Discovery completed; the candidates may be cached.
    Complete(Vec<SkillCandidate<L>>),
    /// Discovery did not complete; the candidates stay loadable but uncached.
    Incomplete(Vec<SkillCandidate<L>>),
}

impl<L> ProviderObservation<L> {
    /// The candidates in this observation.
    pub fn candidates(&self) -> &[SkillCandidate<L>] {
        match self {
            Self::Complete(candidates) | Self::Incomplete(candidates) => candidates,
        }
    }

    /// Whether these candidates may be cached.
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete(_))
    }

    /// Consume the observation into its candidates.
    pub fn into_candidates(self) -> Vec<SkillCandidate<L>> {
        match self {
            Self::Complete(candidates) | Self::Incomplete(candidates) => candidates,
        }
    }
}

/// A provider failure.
///
/// Discovery failures are contained: the host skips the provider, records a
/// warning, and marks the collect uncacheable. Load failures propagate to
/// the `get` caller, matching the TypeScript registry.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ProviderError(pub String);

/// One source of skills, such as local directories or a remote registry.
///
/// The locator type `L` is the opaque provider-owned handle carried by the
/// provider's candidates and handed back on load.
pub trait SkillProvider<L> {
    /// Unique provider name in the host registry.
    fn name(&self) -> &str;

    /// List available skill candidates for the lookup context.
    fn list(&self, options: &SkillLookupOptions) -> Result<ProviderObservation<L>, ProviderError>;

    /// Load a complete skill body for a previously listed candidate, or
    /// `None` when it is no longer loadable.
    fn get(
        &self,
        candidate: &SkillCandidate<L>,
        options: &SkillLookupOptions,
    ) -> Result<Option<SkillDefinition>, ProviderError>;
}
