//! Exact JSON-byte accounting for one run's captured output.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node/src/output-json.ts` and
//! `output-ledger.ts`.
//!
//! The cap bounds the *serialized* outer payload — the `logs` array plus the
//! completion value or the failure diagnostic — because that is what the
//! consumer stores. Counting characters would be wrong twice over: a CJK
//! character is three UTF-8 bytes, and a control character costs six bytes
//! once escaped. Every length here is measured on the serialized form without
//! materializing it, and every measurement stops as soon as the cap is
//! crossed.
//!
//! Overflow is an explicit failure, never in-band truncation: the run reports
//! [`PtcFailureKind::OutputLimit`](crate::PtcFailureKind) and keeps only a
//! fitting prefix of the logs, so a model reading the result learns that
//! output was dropped.

#![forbid(unsafe_code)]

use serde_json::Value;

use super::types::{PtcFailureKind, PtcRunFailure, PtcRunResult};

/// Bytes one code point occupies inside a JSON string literal, quotes
/// excluded.
///
/// Mirrors `serializedCharacterBytes`: surrogate pairs count as one 4-byte
/// code point, `"`/`\` escape to two bytes, C0 controls escape to either two
/// (`\b \t \n \f \r`) or six (`\u00XX`) bytes, and everything else is its
/// UTF-8 length.
fn serialized_character_bytes(character: char) -> usize {
    match character {
        '"' | '\\' => 2,
        '\u{08}' | '\t' | '\n' | '\u{0c}' | '\r' => 2,
        character if (character as u32) < 0x20 => 6,
        character => character.len_utf8(),
    }
}

/// Measure one JSON string, quotes included, or `None` once it exceeds `max`.
///
/// Returns the exact serialized size, so a caller can admit by arithmetic
/// rather than by rendering.
pub fn json_string_bytes_up_to(text: &str, max_bytes: usize) -> Option<usize> {
    if max_bytes < 2 {
        return None;
    }
    let mut bytes = 2;
    for character in text.chars() {
        bytes += serialized_character_bytes(character);
        if bytes > max_bytes {
            return None;
        }
    }
    Some(bytes)
}

/// The longest code-point-aligned prefix whose JSON string encoding (quotes
/// included) fits `max_bytes`, or `""` when nothing useful fits.
///
/// Slicing on a code-point boundary matters: a byte-sliced prefix of a
/// multi-byte character is not a `String`, and a lossy replacement would
/// change the byte count this function promises.
pub fn truncate_json_string_bytes(text: &str, max_bytes: usize) -> String {
    if max_bytes < 2 {
        return String::new();
    }
    let mut bytes = 2;
    let mut end = 0;
    for character in text.chars() {
        let cost = serialized_character_bytes(character);
        if bytes + cost > max_bytes {
            break;
        }
        bytes += cost;
        end += character.len_utf8();
    }
    text[..end].to_owned()
}

/// A number's serialized length, measured through the same serializer that
/// will write the result.
///
/// The TypeScript ledger measures `String(number)`; a JavaScript number cannot
/// carry a trailing `.0`, while `serde_json` preserves the literal it parsed,
/// so the two spell `1.0` differently. Measuring with `serde_json`'s own form
/// keeps the accounting exact for the bytes *this* crate emits — the property
/// the cap exists to guarantee — instead of matching a form it never writes.
fn number_text(number: &serde_json::Number) -> String {
    number.to_string()
}

/// Measure one lossless JSON value's serialized size, or `None` once it
/// exceeds `max`.
///
/// Iterative: the traversal is bounded by the value's node count, not by the
/// call stack, so a deeply nested completion value is measured rather than
/// overflowing.
pub fn json_value_bytes_up_to(value: &Value, max_bytes: usize) -> Option<usize> {
    enum Task<'a> {
        Value(&'a Value),
        Key(&'a String),
        Comma,
        Colon,
    }

    let mut bytes = 0usize;
    let mut stack: Vec<Task<'_>> = vec![Task::Value(value)];
    while let Some(task) = stack.pop() {
        match task {
            Task::Comma => bytes += 1,
            Task::Colon => bytes += 1,
            Task::Key(key) => {
                // The separating colon is its own task, so it is charged once.
                bytes += json_string_bytes_up_to(key, max_bytes.checked_sub(bytes)?)?;
            }
            Task::Value(current) => match current {
                Value::Null => bytes += 4,
                Value::Bool(true) => bytes += 4,
                Value::Bool(false) => bytes += 5,
                Value::Number(number) => bytes += number_text(number).len(),
                Value::String(text) => {
                    bytes += json_string_bytes_up_to(text, max_bytes.checked_sub(bytes)?)?;
                }
                Value::Array(items) => {
                    bytes += 2;
                    for (index, item) in items.iter().enumerate().rev() {
                        stack.push(Task::Value(item));
                        if index > 0 {
                            stack.push(Task::Comma);
                        }
                    }
                }
                Value::Object(map) => {
                    bytes += 2;
                    let entries: Vec<(&String, &Value)> = map.iter().collect();
                    for (index, (key, item)) in entries.iter().enumerate().rev() {
                        // Popped in reverse: `,` `"key"` `:` value.
                        stack.push(Task::Value(item));
                        stack.push(Task::Colon);
                        stack.push(Task::Key(key));
                        if index > 0 {
                            stack.push(Task::Comma);
                        }
                    }
                }
            },
        }
        if bytes > max_bytes {
            return None;
        }
    }
    Some(bytes)
}

/// One run's combined outer-output ledger.
///
/// Binding values never enter it: a tool result crossing the bridge is
/// transported as JSON but is not *program output*, so it cannot consume the
/// model-visible cap.
///
/// The ledger is order-sensitive by construction — it is a running byte sum
/// over text in emission order — so "out of order" has no meaning here. The
/// ordering that PTC does have to satisfy is the *commit* order of nested
/// dispatches, which is [`crate::calls::DispatchLane`]'s job.
#[derive(Clone, Debug)]
pub struct OutputLedger {
    max_bytes: usize,
    entries: usize,
    bytes: usize,
    overflow_message: Option<String>,
}

impl OutputLedger {
    /// A ledger capping the serialized outer payload at `max_bytes`.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            entries: 0,
            // JSON serialization of the empty logs array: `[]`.
            bytes: 2,
            overflow_message: None,
        }
    }

    /// A ledger whose overflow diagnostic says why the output did not survive
    /// (the persistent provider uses this to explain that retained variables
    /// were cleared).
    pub fn with_overflow_message(mut self, message: impl Into<String>) -> Self {
        self.overflow_message = Some(message.into());
        self
    }

    /// The configured cap.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Bytes charged so far, including the log array's brackets.
    pub fn used_bytes(&self) -> usize {
        self.bytes
    }

    /// Exact remaining budget for the completion value or failure message.
    pub fn remaining_bytes(&self) -> usize {
        self.max_bytes.saturating_sub(self.bytes)
    }

    /// Admit one log entry, or report that the hard cap was crossed.
    ///
    /// A refused entry is *not* pushed: the caller decides whether to stop the
    /// run (the host does) or to continue capturing (the program-side buffer
    /// records the limit once).
    pub fn admit(&mut self, text: &str, sink: &mut Vec<String>) -> bool {
        let separator = if self.entries > 0 { 1 } else { 0 };
        let available = self
            .max_bytes
            .saturating_sub(self.bytes)
            .saturating_sub(separator);
        let Some(string_bytes) = json_string_bytes_up_to(text, available) else {
            return false;
        };
        self.bytes += string_bytes + separator;
        self.entries += 1;
        sink.push(text.to_owned());
        true
    }

    /// Finalize a successful absent-or-JSON completion against the cap.
    pub fn success(&self, logs: Vec<String>, value: Option<&Value>) -> PtcRunResult {
        if let Some(value) = value {
            if json_value_bytes_up_to(value, self.remaining_bytes()).is_none() {
                return self.limit(logs);
            }
        }
        let mut result = PtcRunResult::success(logs);
        if let Some(value) = value {
            result.value = Some(value.clone());
        }
        result
    }

    /// Finalize a failure diagnostic, with `output-limit` taking precedence
    /// when the combined bytes exceed the cap.
    pub fn failure(&self, logs: Vec<String>, error: PtcRunFailure) -> PtcRunResult {
        if json_string_bytes_up_to(&error.message, self.remaining_bytes()).is_none() {
            return self.limit(logs);
        }
        PtcRunResult::failure(logs, error)
    }

    /// Build the explicit output-limit failure, retaining a fitting prefix of
    /// the logs and truncating the diagnostic to what still fits.
    pub fn limit(&self, logs: Vec<String>) -> PtcRunResult {
        let full_message = self
            .overflow_message
            .clone()
            .unwrap_or_else(|| format!("outer output exceeded {} bytes", self.max_bytes));
        // The diagnostic is ASCII in the shipped configurations, where every
        // character is one byte plus the quotes.
        let message_bytes = full_message.len() + 2;
        let mut retained: Vec<String> = Vec::new();
        let mut retained_bytes = 2usize;
        let log_budget = self.max_bytes.saturating_sub(message_bytes);
        for text in logs {
            let separator = if retained.is_empty() { 0 } else { 1 };
            let available = log_budget.saturating_sub(retained_bytes + separator);
            if let Some(string_bytes) = json_string_bytes_up_to(&text, available) {
                retained.push(text);
                retained_bytes += string_bytes + separator;
                continue;
            }
            let prefix = truncate_json_string_bytes(&text, available);
            if !prefix.is_empty() {
                let Some(prefix_bytes) = json_string_bytes_up_to(&prefix, available) else {
                    unreachable!("a truncated prefix always fits its own budget");
                };
                retained.push(prefix);
                retained_bytes += prefix_bytes + separator;
            }
            break;
        }
        let message = truncate_json_string_bytes(
            &full_message,
            self.max_bytes.saturating_sub(retained_bytes),
        );
        PtcRunResult::failure(
            retained,
            PtcRunFailure::new(PtcFailureKind::OutputLimit, message),
        )
    }

    /// Render the log array's serialized form — the exact text the ledger's
    /// byte count describes, for a test that wants to compare accounting
    /// against the real thing.
    pub fn render_logs(logs: &[String]) -> String {
        serde_json::to_string(logs).expect("a log array is always serializable")
    }
}

/// What one program-side `push` produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProgramLogPush {
    /// The complete text was admitted; emit it.
    Whole,
    /// The text did not fit: emit this fitting prefix (possibly empty) and
    /// then report the limit exactly once.
    Truncated(String),
    /// The buffer already reported its limit; emit nothing.
    Suppressed,
}

/// The program side of the same budget.
///
/// Ported from `LogBuffer` in
/// `packages/ptc-runtime/ptc-runtime-node/src/bootstrap.ts`. The program
/// captures console and raw stream output *inside* the sandbox, so it must
/// stop charging at the same cap the host will apply — otherwise its own
/// `done` message could carry a value that the host's accounting then
/// rejects, and the model would see a confusing overflow for output it
/// already curated.
///
/// The limit is reported **once**: after the first over-budget push the buffer
/// is truncated for good, which is the de-duplication this seam has. Later
/// writes are dropped silently, because a second signal would produce a
/// second `output-limit` failure for one exhausted budget.
#[derive(Clone, Debug)]
pub struct ProgramLogBuffer {
    max_bytes: usize,
    entries: usize,
    bytes: usize,
    truncated: bool,
}

impl ProgramLogBuffer {
    /// A buffer capping the serialized log array at `max_bytes`.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            entries: 0,
            bytes: 2,
            truncated: false,
        }
    }

    /// Whether this buffer has already reported its limit.
    pub fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// Exact remaining budget for the completion value or failure message.
    pub fn remaining_bytes(&self) -> usize {
        self.max_bytes.saturating_sub(self.bytes)
    }

    /// Charge one captured text and report what the caller should emit.
    pub fn push(&mut self, text: &str) -> ProgramLogPush {
        if self.truncated {
            return ProgramLogPush::Suppressed;
        }
        let separator = if self.entries > 0 { 1 } else { 0 };
        let available = self
            .max_bytes
            .saturating_sub(self.bytes)
            .saturating_sub(separator);
        let Some(string_bytes) = json_string_bytes_up_to(text, available) else {
            self.truncated = true;
            let prefix = truncate_json_string_bytes(text, available);
            if prefix.is_empty() {
                return ProgramLogPush::Truncated(prefix);
            }
            let prefix_bytes = json_string_bytes_up_to(&prefix, available)
                .expect("a truncated prefix always fits its own budget");
            self.bytes += prefix_bytes + separator;
            self.entries += 1;
            return ProgramLogPush::Truncated(prefix);
        };
        self.bytes += string_bytes + separator;
        self.entries += 1;
        ProgramLogPush::Whole
    }
}
