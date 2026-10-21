//! Startup environment filtering, ported from `environment.ts`.
//!
//! The child starts with an environment stripped to the variables a native
//! executable needs to launch itself — path resolution and Windows system
//! locations — so model-written code never sees host credentials through
//! `process.env`. The TypeScript provider clears the child's environment
//! object after spawn for the *program*; a Rust host cannot rebuild the child
//! view after the fact, so the same filter is applied at spawn time.

#![forbid(unsafe_code)]

/// Variables required by native executables before model evaluation:
/// executable search, Windows system paths, and sandbox temporary paths.
pub const STARTUP_ENVIRONMENT_NAMES: [&str; 6] =
    ["PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"];

/// The host environment filtered to [`STARTUP_ENVIRONMENT_NAMES`].
///
/// Comparison is ASCII-case-insensitive, matching the TypeScript
/// `key.toUpperCase()` membership test; the child sees the original spelling.
pub fn startup_environment() -> Vec<(String, String)> {
    std::env::vars()
        .filter(|(key, _)| {
            let upper = key.to_ascii_uppercase();
            STARTUP_ENVIRONMENT_NAMES.contains(&upper.as_str())
        })
        .collect()
}
