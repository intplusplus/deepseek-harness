//! Errors raised while assembling the unified plugin host.

use thiserror::Error;

/// A failure mounting one host component onto the shared [`dsh_plugin::Scope`].
///
/// Construction failures are the only ones the assembly layer can surface:
/// every host's *drain* runs later, on scope teardown, and must be infallible
/// (a teardown error cannot abort the unwind of the remaining hosts).
#[derive(Debug, Error)]
pub enum HostAssemblyError {
    /// A host could not be constructed from the supplied configuration.
    #[error("failed to assemble host `{name}`: {source}")]
    Mount {
        /// The mounted component's stable name (used in drain traces).
        name: &'static str,
        /// The underlying construction error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl HostAssemblyError {
    /// Build a [`HostAssemblyError::Mount`] from any construction error.
    pub fn mount(
        name: &'static str,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Mount {
            name,
            source: source.into(),
        }
    }
}
