//! JavaScript-shaped JSON numbers.
//!
//! Every duration in the retry vocabulary is a JavaScript `number`, and
//! `JSON.stringify(500)` writes `500` — not `500.0`. Rust's `serde_json`
//! writes `500.0` for the same `f64`, which would put a different byte
//! sequence into the durable events and into `policyKey`. Since `policyKey`
//! is compared across replay (a changed policy starts a new retry chain),
//! the difference is observable, so numbers are written the way the
//! TypeScript side writes them.
//!
//! Only serialization is affected: `500` parses into `f64` either way.

#![forbid(unsafe_code)]

use serde::{Serialize, Serializer};
use serde_json::{Number, Value};

/// Largest integer exactly representable as a JavaScript number.
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

/// `value` as a JSON number shaped the way `JSON.stringify` shapes it.
///
/// Whole values within JavaScript's safe integer range become integers;
/// everything else keeps its `f64` rendering. Non-finite values are a bug in
/// the caller — every duration here is validated finite before it reaches the
/// wire — and collapse to zero rather than to `null`.
pub fn number(value: f64) -> Value {
    if value.is_finite() && value.fract() == 0.0 && value.abs() <= MAX_SAFE_INTEGER {
        return Value::Number((value as i64).into());
    }
    Value::Number(Number::from_f64(value).unwrap_or_else(|| Number::from(0)))
}

/// [`number`] as a `serialize_with` target.
pub fn serialize_number<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    number(*value).serialize(serializer)
}
