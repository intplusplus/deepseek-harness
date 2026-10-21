//! Error vocabulary for the session-format catalog dispatch layer.

use std::fmt;

use dsh_session::SessionError;

/// One refused catalog operation.
///
/// The two variants mirror the TypeScript pair `SessionFormatError` /
/// `SessionFormatUnsupportedMigrationError`: a [`CatalogError::Format`] fault
/// means the artifact itself is damaged, while a
/// [`CatalogError::UnsupportedMigration`] fault means the artifact is readable
/// but this build refuses to migrate it losslessly. The [`fmt::Display`] text
/// is the TypeScript diagnostic, verbatim, so refusal messages stay aligned
/// across the two readers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogError {
    /// The artifact is damaged beyond dispatch (TS `SessionFormatError`).
    Format(String),
    /// The artifact is readable but the migration is refused
    /// (TS `SessionFormatUnsupportedMigrationError`).
    UnsupportedMigration(String),
}

impl CatalogError {
    pub(crate) fn format(message: impl Into<String>) -> Self {
        Self::Format(message.into())
    }

    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self::UnsupportedMigration(message.into())
    }

    /// The verbatim diagnostic (TS `error.message`).
    pub fn message(&self) -> &str {
        match self {
            Self::Format(message) | Self::UnsupportedMigration(message) => message,
        }
    }

    /// Whether this is the readable-but-refused classification.
    pub fn is_unsupported_migration(&self) -> bool {
        matches!(self, Self::UnsupportedMigration(_))
    }
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for CatalogError {}

impl From<SessionError> for CatalogError {
    /// Classify a core session error the way the TypeScript catalog classifies
    /// its own failures: a future stored generation is a migration refusal,
    /// everything else is a malformed artifact.
    fn from(error: SessionError) -> Self {
        match error {
            SessionError::UnsupportedVersion { found, current } => Self::unsupported(format!(
                "stored Session uses newer format v{found}; this build writes v{current}"
            )),
            other => Self::format(other.to_string()),
        }
    }
}
