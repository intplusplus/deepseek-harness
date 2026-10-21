//! Injected time-zone resolution.
//!
//! The library never reads the system clock or a zone database. Callers supply
//! a [`TimeZoneResolver`]: production code adapts the platform's IANA data, and
//! tests use [`ZoneTable`], a small deterministic table with explicit
//! transitions. Fixed offsets and DST boundaries are therefore both ordinary
//! data, and no test depends on the host time zone.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use thiserror::Error;

/// The largest UTC offset magnitude a zone table admits, in seconds.
const MAX_OFFSET_SECONDS: i32 = 24 * 3_600;

/// One resolved zone observation: its canonical label and its UTC offset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZoneResolution {
    /// Canonical zone label, used verbatim in the timestamp brackets.
    pub canonical: String,
    /// Offset from UTC at the observed instant, in seconds.
    pub offset_seconds: i32,
}

/// Injected IANA-zone rules.
///
/// Implementations answer from compiled zone data, a platform API, or a fixed
/// table. `canonical` is the `Intl.DateTimeFormat(...).resolvedOptions().timeZone`
/// counterpart: an unknown zone returns `None`, and a known alias returns the
/// name it canonicalizes to.
pub trait TimeZoneResolver {
    /// The canonical label for `zone`, or `None` when the zone is unknown.
    fn canonical(&self, zone: &str) -> Option<String>;

    /// The zone's UTC offset at `epoch_ms`, or `None` when the zone is unknown.
    fn offset_seconds(&self, zone: &str, epoch_ms: i64) -> Option<i32>;
}

/// A deterministic zone table with explicit transition instants.
///
/// Each zone starts with a fixed offset and may gain transitions in any order;
/// the offset in force is the last transition at or before the instant. This is
/// the shape a compiled IANA table has, without the loader.
#[derive(Clone, Debug)]
pub struct ZoneTable {
    zones: BTreeMap<String, ZoneEntry>,
}

#[derive(Clone, Debug)]
struct ZoneEntry {
    canonical: String,
    transitions: Vec<(i64, i32)>,
}

impl Default for ZoneTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ZoneTable {
    /// A table containing only `UTC` at offset zero.
    pub fn new() -> Self {
        let mut table = Self {
            zones: BTreeMap::new(),
        };
        // The offset literal is zero and therefore always in range.
        let _ = table.insert_fixed("UTC", 0);
        table
    }

    /// Insert or replace a zone whose offset never changes.
    ///
    /// # Errors
    ///
    /// Returns [`ZoneTableError::OffsetOutOfRange`] when `offset_seconds` is at
    /// least one day in magnitude.
    pub fn insert_fixed(&mut self, zone: &str, offset_seconds: i32) -> Result<(), ZoneTableError> {
        validate_offset(offset_seconds)?;
        self.zones.insert(
            zone.to_string(),
            ZoneEntry {
                canonical: zone.to_string(),
                transitions: vec![(i64::MIN, offset_seconds)],
            },
        );
        Ok(())
    }

    /// Insert or replace a zone under an alias that canonicalizes to `canonical`.
    ///
    /// The target must already be present. This models `Etc/UTC` resolving to
    /// `UTC`, the case that makes a non-canonical browser zone detectable.
    ///
    /// # Errors
    ///
    /// Returns [`ZoneTableError::UnknownZone`] when `canonical` is absent.
    pub fn insert_alias(&mut self, alias: &str, canonical: &str) -> Result<(), ZoneTableError> {
        let entry =
            self.zones
                .get(canonical)
                .cloned()
                .ok_or_else(|| ZoneTableError::UnknownZone {
                    zone: canonical.to_string(),
                })?;
        self.zones.insert(
            alias.to_string(),
            ZoneEntry {
                canonical: entry.canonical,
                ..entry
            },
        );
        Ok(())
    }

    /// Add one transition to an existing zone.
    ///
    /// A transition at an existing instant replaces that transition's offset.
    /// The offset in force before the first transition is the zone's initial
    /// fixed offset.
    ///
    /// # Errors
    ///
    /// Returns [`ZoneTableError::UnknownZone`] for an absent zone and
    /// [`ZoneTableError::OffsetOutOfRange`] for an offset at least one day in
    /// magnitude.
    pub fn insert_transition(
        &mut self,
        zone: &str,
        at_epoch_ms: i64,
        offset_seconds: i32,
    ) -> Result<(), ZoneTableError> {
        validate_offset(offset_seconds)?;
        let entry = self
            .zones
            .get_mut(zone)
            .ok_or_else(|| ZoneTableError::UnknownZone {
                zone: zone.to_string(),
            })?;
        match entry
            .transitions
            .binary_search_by_key(&at_epoch_ms, |(at, _)| *at)
        {
            Ok(position) => entry.transitions[position].1 = offset_seconds,
            Err(position) => entry
                .transitions
                .insert(position, (at_epoch_ms, offset_seconds)),
        }
        Ok(())
    }
}

impl TimeZoneResolver for ZoneTable {
    fn canonical(&self, zone: &str) -> Option<String> {
        self.zones.get(zone).map(|entry| entry.canonical.clone())
    }

    fn offset_seconds(&self, zone: &str, epoch_ms: i64) -> Option<i32> {
        let entry = self.zones.get(zone)?;
        let position = match entry
            .transitions
            .binary_search_by_key(&epoch_ms, |(at, _)| *at)
        {
            Ok(position) => position,
            Err(0) => 0,
            Err(position) => position - 1,
        };
        entry.transitions.get(position).map(|(_, offset)| *offset)
    }
}

/// Reject offsets the formatter cannot mean.
fn validate_offset(offset_seconds: i32) -> Result<(), ZoneTableError> {
    if offset_seconds <= -MAX_OFFSET_SECONDS || offset_seconds >= MAX_OFFSET_SECONDS {
        return Err(ZoneTableError::OffsetOutOfRange { offset_seconds });
    }
    Ok(())
}

/// Why a [`ZoneTable`] edit was rejected.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ZoneTableError {
    /// The offset is at least 24 hours in magnitude.
    #[error("time zone offset {offset_seconds} seconds is outside the supported range")]
    OffsetOutOfRange {
        /// The rejected offset in seconds.
        offset_seconds: i32,
    },
    /// An alias or transition named a zone the table does not hold.
    #[error("unknown time zone `{zone}`")]
    UnknownZone {
        /// The absent zone name.
        zone: String,
    },
}
