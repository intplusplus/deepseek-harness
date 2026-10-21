//! ISO-shaped timestamp formatting for durable time-context readings.
//!
//! TypeScript calls `Intl.DateTimeFormat('en-US', …)` with numeric local
//! fields, `hourCycle: 'h23'`, and `timeZoneName: 'longOffset'`, then strips the
//! `GMT` prefix from the offset. The instant-to-calendar conversion is pure
//! arithmetic, so the counterpart here needs only the offset for the instant;
//! the IANA transitions that produce it are injected through
//! [`TimeZoneResolver`](crate::time_context::TimeZoneResolver).

#![forbid(unsafe_code)]

use std::fmt::Write as _;

/// Format `epoch_ms` in `time_zone` (already canonical) at `offset_seconds`.
///
/// The result is the durable text the model reads, for example
/// `2026-07-15T09:01:01+08:00[Asia/Shanghai]`. Milliseconds are dropped, as the
/// `Intl.DateTimeFormat` call that defines the TypeScript text does.
pub fn format_timestamp(epoch_ms: i64, offset_seconds: i32, time_zone: &str) -> String {
    let local_ms = epoch_ms.saturating_add(i64::from(offset_seconds).saturating_mul(1_000));
    let days = local_ms.div_euclid(86_400_000);
    let milliseconds_of_day = local_ms.rem_euclid(86_400_000);
    let seconds_of_day = milliseconds_of_day / 1_000;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    let mut rendered = String::with_capacity(time_zone.len() + 40);
    // Writing into a `String` cannot fail.
    let _ = write!(
        rendered,
        "{}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}{}[{time_zone}]",
        format_year(year),
        format_offset(offset_seconds),
    );
    rendered
}

/// Format a year the way `Intl.DateTimeFormat` with `year: 'numeric'` does.
///
/// Four-digit years are zero-padded; years outside `0000..=9999` keep their
/// plain decimal form, which is the closest representable shape for an instant
/// that far outside the product's range.
fn format_year(year: i64) -> String {
    if (0..=9_999).contains(&year) {
        format!("{year:04}")
    } else {
        year.to_string()
    }
}

/// Format a UTC offset as `±HH:MM`, with `:SS` only for whole-second offsets.
///
/// `Intl.DateTimeFormat` reports `GMT` for zero, which the TypeScript formatter
/// rewrites to `GMT+00:00` before stripping `GMT`; the output is therefore
/// always signed.
fn format_offset(offset_seconds: i32) -> String {
    let sign = if offset_seconds < 0 { '-' } else { '+' };
    let magnitude = i64::from(offset_seconds).abs();
    let hours = magnitude / 3_600;
    let minutes = (magnitude % 3_600) / 60;
    let seconds = magnitude % 60;
    if seconds == 0 {
        format!("{sign}{hours:02}:{minutes:02}")
    } else {
        format!("{sign}{hours:02}:{minutes:02}:{seconds:02}")
    }
}

/// Convert days since the Unix epoch to a proleptic Gregorian date.
///
/// Howard Hinnant's `civil_from_days`, valid for negative day counts.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days.saturating_add(719_468);
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted.saturating_sub(146_096)
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    (year, month as u32, day as u32)
}
