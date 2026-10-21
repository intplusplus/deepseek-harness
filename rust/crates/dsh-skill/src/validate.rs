//! Admission checks for provider-supplied skills.

use serde_json::Value;

use crate::types::{SkillCandidate, SkillDefinition, SkillRegistration, SkillSummary};

/// A skill that cannot be admitted to a catalog.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum SkillError {
    /// The name does not match the public kebab-case grammar.
    #[error("invalid skill name \"{name}\"")]
    InvalidName {
        /// The rejected name.
        name: String,
    },
    /// The precedence rank is not a finite number.
    #[error("skill rank must be finite, got {rank}")]
    NonFiniteRank {
        /// The rejected rank.
        rank: f64,
    },
    /// The name is valid but the entry carries no routing description.
    #[error("{subject} requires a description")]
    EmptyDescription {
        /// The noun phrase naming the entry and its origin.
        subject: String,
    },
    /// A provider candidate names a different provider than the one that returned it.
    #[error("skill provider \"{provider}\" returned skill \"{name}\" for provider \"{actual}\"")]
    ProviderMismatch {
        /// The provider whose discovery returned the candidate.
        provider: String,
        /// The candidate's skill name.
        name: String,
        /// The provider name the candidate declared.
        actual: String,
    },
}

/// Validate one candidate returned by `provider_name`'s discovery.
///
/// The grammar and rank are already guaranteed by [`crate::SkillName`] and
/// [`crate::SkillRank`]. What remains is the routing description and provider
/// ownership: a provider must not smuggle in a skill owned by another source.
pub fn validate_candidate<L>(
    candidate: &SkillCandidate<L>,
    provider_name: &str,
) -> Result<(), SkillError> {
    validate_summary(&candidate.summary)?;
    if candidate.summary.provider != provider_name {
        return Err(SkillError::ProviderMismatch {
            provider: provider_name.to_string(),
            name: candidate.summary.name.to_string(),
            actual: candidate.summary.provider.clone(),
        });
    }
    Ok(())
}

/// Validate a complete definition loaded from a provider.
pub fn validate_definition(definition: &SkillDefinition) -> Result<(), SkillError> {
    validate_summary(&definition.summary)
}

/// Validate a runtime registration before it enters a registry layer.
///
/// The invocation policy and provider label are optional at registration time;
/// [`SkillRegistration::into_definition`] supplies their defaults.
pub fn validate_registration(registration: &SkillRegistration) -> Result<(), SkillError> {
    if registration.description.is_empty() {
        return Err(SkillError::EmptyDescription {
            subject: format!("skill \"{}\"", registration.name),
        });
    }
    Ok(())
}

/// The shared summary checks: a non-empty description.
fn validate_summary(summary: &SkillSummary) -> Result<(), SkillError> {
    if summary.description.is_empty() {
        return Err(SkillError::EmptyDescription {
            subject: format!("skill \"{}\"", summary.name),
        });
    }
    Ok(())
}

/// A provider's optional frontmatter object.
///
/// Kept as a plain JSON object because it is provider-owned metadata, not a
/// vocabulary this crate may narrow.
pub type SkillMetadata = serde_json::Map<String, Value>;
