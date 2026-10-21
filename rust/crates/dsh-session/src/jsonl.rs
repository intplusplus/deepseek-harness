//! JSONL persistence: physical framing, generation selection, and exclusive
//! publication.
//!
//! Wire layout (Rust-native; byte-level interop with the TypeScript JSONL
//! provider is a follow-up): line 1 is the [`SessionHeader`], each later line
//! one [`SessionEvent`]. JSONL v0 uses `session.jsonl`, v1 and later use
//! lowercase `session.vN.jsonl`, and committed generation paths are never
//! renamed, replaced, or deleted.
//!
//! Generation selection is header-only: the numerically highest canonical
//! generation wins, a future version is refused, and an older one is normalized
//! by the migration chain in [`crate::migration`] before it is restored.

#![forbid(unsafe_code)]

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use super::{event::SessionEvent, log::Session, migration, types::*};

/// The file name a stored generation uses: the legacy v0 name, or `session.vN.jsonl`.
pub fn session_file_name(version: u32) -> String {
    if version == 0 {
        "session.jsonl".to_string()
    } else {
        format!("session.v{version}.jsonl")
    }
}

/// The generation encoded in a session file name, if it names one.
pub fn parse_generation(file_name: &str) -> Option<u32> {
    if file_name == "session.jsonl" {
        return Some(0);
    }
    let generation = file_name
        .strip_prefix("session.v")?
        .strip_suffix(".jsonl")?;
    generation.parse::<u32>().ok()
}

/// The numerically highest canonical generation in a session directory.
pub fn latest_generation(directory: &Path) -> Result<Option<(u32, PathBuf)>, SessionError> {
    let mut latest: Option<(u32, PathBuf)> = None;
    let entries = fs::read_dir(directory).map_err(|error| SessionError::Io(error.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|error| SessionError::Io(error.to_string()))?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        let Some(version) = parse_generation(file_name) else {
            continue;
        };
        if latest.as_ref().is_none_or(|(latest, _)| version > *latest) {
            latest = Some((version, entry.path()));
        }
    }
    Ok(latest)
}

/// Encode the session: the header line plus one line per event.
pub fn encode(session: &Session) -> Result<String, SessionError> {
    let mut text = serde_json::to_string(session.header())
        .map_err(|error| SessionError::Json(error.to_string()))?;
    text.push('\n');
    for event in session.events() {
        text.push_str(
            &serde_json::to_string(event).map_err(|error| SessionError::Json(error.to_string()))?,
        );
        text.push('\n');
    }
    Ok(text)
}

/// Decode an encoding back into a session, re-admitting every event.
///
/// Any stored generation below the current [`SESSION_FORMAT_VERSION`] is loaded
/// and normalized up to v4 by the migration chain before folding; a current
/// generation decodes directly; a future generation is refused.
pub fn decode(text: &str) -> Result<Session, SessionError> {
    let mut lines = text.lines();
    let header_line = lines.next().ok_or(SessionError::EmptyLog)?;
    let header_value: serde_json::Value =
        serde_json::from_str(header_line).map_err(|error| SessionError::Json(error.to_string()))?;
    let version = migration::detect_version(&header_value).ok_or_else(|| {
        SessionError::Json("session header is missing a format version".to_string())
    })?;

    if version == SESSION_FORMAT_VERSION {
        // Current generation: decode directly. The Rust-native flat log is the
        // only current layout; a TypeScript-physical v4 envelope is rejected by
        // the event decoder.
        return decode_current(&header_value, lines);
    }
    if version > SESSION_FORMAT_VERSION {
        return Err(SessionError::UnsupportedVersion {
            found: version,
            current: SESSION_FORMAT_VERSION,
        });
    }

    // Legacy generation: parse every line into an envelope, migrate to v4, then
    // restore through the usual admission path.
    let log = migration::Log::parse(text)?;
    let (v4_header, flat_events) = migration::migrate_session(log, &[])?;
    let header: SessionHeader =
        serde_json::from_value(v4_header).map_err(|error| SessionError::Json(error.to_string()))?;
    let mut events = Vec::with_capacity(flat_events.len());
    for flat in flat_events {
        events.push(
            serde_json::from_value(flat).map_err(|error| SessionError::Json(error.to_string()))?,
        );
    }
    Session::restore(header, events)
}

/// Decode a current-generation log directly: the header line plus one
/// [`SessionEvent`] per remaining line.
fn decode_current<'a, I>(
    header_value: &serde_json::Value,
    lines: I,
) -> Result<Session, SessionError>
where
    I: Iterator<Item = &'a str>,
{
    let header: SessionHeader = serde_json::from_value(header_value.clone())
        .map_err(|error| SessionError::Json(error.to_string()))?;
    let mut events = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let event: SessionEvent =
            serde_json::from_str(line).map_err(|error| SessionError::Json(error.to_string()))?;
        events.push(event);
    }
    Session::restore(header, events)
}

/// Verify an encoding decodes back to the same session.
pub fn verify(session: &Session) -> Result<(), SessionError> {
    let decoded = decode(&encode(session)?)?;
    if decoded != *session {
        return Err(SessionError::NotJson(
            "encoded session did not verify".to_string(),
        ));
    }
    Ok(())
}

/// Publish the current generation: encode, verify, then rename into place.
///
/// The staging file is synced before the rename, so the exclusive publication
/// is atomic within one volume and a crash never leaves a torn successor.
pub fn write_session(directory: &Path, session: &Session) -> Result<PathBuf, SessionError> {
    let version = session.header().version;
    if version != SESSION_FORMAT_VERSION {
        return Err(SessionError::UnsupportedVersion {
            found: version,
            current: SESSION_FORMAT_VERSION,
        });
    }
    verify(session)?;
    fs::create_dir_all(directory).map_err(|error| SessionError::Io(error.to_string()))?;
    let final_path = directory.join(session_file_name(version));
    let staging_path = directory.join(format!(
        "{}.staging-{}",
        session_file_name(version),
        std::process::id()
    ));
    let text = encode(session)?;
    {
        let mut file =
            fs::File::create(&staging_path).map_err(|error| SessionError::Io(error.to_string()))?;
        file.write_all(text.as_bytes())
            .map_err(|error| SessionError::Io(error.to_string()))?;
        file.sync_all()
            .map_err(|error| SessionError::Io(error.to_string()))?;
    }
    fs::rename(&staging_path, &final_path).map_err(|error| SessionError::Io(error.to_string()))?;
    Ok(final_path)
}

/// Read the highest stored generation from a session directory.
pub fn read_session(directory: &Path) -> Result<Session, SessionError> {
    let (_, path) = latest_generation(directory)?.ok_or(SessionError::NoStoredGeneration)?;
    let text = fs::read_to_string(&path).map_err(|error| SessionError::Io(error.to_string()))?;
    decode(&text)
}
