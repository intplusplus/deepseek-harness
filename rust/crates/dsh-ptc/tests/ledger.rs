//! Outer-output byte accounting.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node/tests/output-ledger.spec.ts`
//! and `output-json.spec.ts`.

use dsh_ptc::{
    json_string_bytes_up_to, json_value_bytes_up_to, truncate_json_string_bytes, OutputLedger,
    ProgramLogBuffer, ProgramLogPush, PtcFailureKind, PtcRunFailure,
};
use serde_json::{json, Value};

fn ledger(max_bytes: usize) -> OutputLedger {
    OutputLedger::new(max_bytes)
}

#[test]
fn accounts_for_escaping_separators_and_optional_completion_values() {
    let mut ledger = ledger(20);
    let mut logs = Vec::new();
    assert!(ledger.admit("first", &mut logs));
    assert!(ledger.admit("second", &mut logs));
    assert!(!ledger.admit("third", &mut logs));
    assert_eq!(logs, ["first", "second"]);

    let success = ledger.success(logs.clone(), None);
    assert_eq!(success.logs, ["first", "second"]);
    assert!(success.error.is_none());

    let with_value = ledger.success(logs.clone(), Some(&json!(1)));
    assert_eq!(with_value.value, Some(json!(1)));

    let overflow = ledger.success(logs, Some(&json!("too long")));
    assert_eq!(overflow.failure_kind(), Some(PtcFailureKind::OutputLimit));
    let message = overflow.error.expect("a diagnostic").message;
    assert!(
        message.starts_with("outer output"),
        "the overflow diagnostic is truncated to the cap: {message:?}"
    );
}

/// `limit` on a 20-byte cap keeps only what fits, so the diagnostic itself is
/// shortened; a cap large enough for the whole sentence keeps it verbatim.
#[test]
fn a_generous_cap_keeps_the_whole_overflow_diagnostic() {
    let ledger = ledger(200);
    let result = ledger.limit(vec!["x".repeat(300)]);
    let message = result.error.expect("diagnostic").message;
    assert_eq!(message, "outer output exceeded 200 bytes");
    let bytes = OutputLedger::render_logs(&result.logs).len()
        + serde_json::to_string(&message).expect("serializable").len();
    assert!(bytes <= 200);
}

#[test]
fn the_ledger_matches_the_serialized_form_it_accounts_for() {
    // A one-byte budget cannot hold `[]` plus any string, but the empty array
    // itself is two bytes: the initial charge is exactly `[]`.
    let empty = ledger(2);
    assert_eq!(empty.used_bytes(), 2);
    assert_eq!(empty.remaining_bytes(), 0);
    assert_eq!(OutputLedger::render_logs(&[]), "[]");

    let mut ledger = ledger(64);
    let mut logs = Vec::new();
    assert!(ledger.admit("a\tb", &mut logs));
    // `"a\tb"` is four content bytes plus two escapes: 2 quotes + 1 + 2 + 1.
    assert_eq!(OutputLedger::render_logs(&logs), r#"["a\tb"]"#);
    assert_eq!(ledger.used_bytes(), OutputLedger::render_logs(&logs).len());
}

#[test]
fn string_measurement_counts_utf8_and_escapes_exactly() {
    assert_eq!(json_string_bytes_up_to("", 10), Some(2));
    assert_eq!(json_string_bytes_up_to("abc", 10), Some(5));
    // Three UTF-8 bytes each.
    assert_eq!(json_string_bytes_up_to("你好", 10), Some(8));
    // One astral code point is four UTF-8 bytes.
    assert_eq!(json_string_bytes_up_to("🙂", 10), Some(6));
    // `"` and `\` escape to two bytes each.
    assert_eq!(json_string_bytes_up_to("\"\\", 10), Some(6));
    // Named C0 escapes are two bytes; the rest are six (`\u00XX`).
    assert_eq!(json_string_bytes_up_to("\t\n", 10), Some(6));
    assert_eq!(json_string_bytes_up_to("\u{1}", 10), Some(8));
    // An over-cap measurement stops as soon as it crosses.
    assert_eq!(json_string_bytes_up_to("abc", 4), None);
    assert_eq!(json_string_bytes_up_to("abc", 1), None);
}

#[test]
fn value_measurement_matches_serialization() {
    let values = [
        json!(null),
        json!(true),
        json!(false),
        json!(0),
        json!(-1),
        json!(1.25),
        json!(1.0),
        json!("text"),
        json!([]),
        json!({}),
        json!([1, { "a": [2] }, "x"]),
        json!({ "b": 1, "a": null }),
    ];
    for value in values {
        let rendered = serde_json::to_string(&value).expect("serializable");
        let measured = json_value_bytes_up_to(&value, 1_000).expect("fits");
        assert_eq!(measured, rendered.len(), "measuring {value}");
    }
    assert_eq!(json_value_bytes_up_to(&json!("abcdef"), 6), None);
}

#[test]
fn truncation_stays_on_code_point_boundaries() {
    assert_eq!(truncate_json_string_bytes("abcdef", 5), "abc");
    assert_eq!(truncate_json_string_bytes("abcdef", 1), "");
    // A three-byte code point needs five bytes with quotes and the first byte.
    assert_eq!(truncate_json_string_bytes("你好🙂", 8), "你好");
    assert_eq!(truncate_json_string_bytes("你好🙂", 5), "你");
    assert_eq!(truncate_json_string_bytes("你好🙂", 4), "");
    // The prefix always fits its own budget.
    for max in 2..14 {
        let prefix = truncate_json_string_bytes("你好🙂", max);
        assert!(json_string_bytes_up_to(&prefix, max).is_some(), "max {max}");
    }
}

#[test]
fn the_limit_result_retains_a_fitting_prefix_and_fits_the_cap() {
    for max_bytes in [4usize, 8, 40, 80] {
        let ledger = ledger(max_bytes);
        let result = ledger.limit(vec!["a".into(), "b".into(), "你好🙂".repeat(50)]);
        assert_eq!(result.failure_kind(), Some(PtcFailureKind::OutputLimit));
        let message = &result.error.as_ref().expect("diagnostic").message;
        let bytes = OutputLedger::render_logs(&result.logs).len()
            + serde_json::to_string(message).expect("serializable").len();
        assert!(bytes <= max_bytes, "max {max_bytes}: {bytes} bytes");
    }
}

#[test]
fn a_failure_diagnostic_is_admitted_or_replaced_by_the_limit() {
    let ledger = ledger(80);
    let short = ledger.failure(
        Vec::new(),
        PtcRunFailure::new(PtcFailureKind::Exception, "short"),
    );
    assert_eq!(short.failure_kind(), Some(PtcFailureKind::Exception));
    assert_eq!(short.error.expect("diagnostic").message, "short");

    let long = ledger.failure(
        Vec::new(),
        PtcRunFailure::new(PtcFailureKind::Exception, "x".repeat(100)),
    );
    assert_eq!(long.failure_kind(), Some(PtcFailureKind::OutputLimit));
}

#[test]
fn a_cleared_environment_explains_itself_in_the_overflow_message() {
    let ledger = ledger(200)
        .with_overflow_message("retained variables were cleared; outer output exceeded 200 bytes");
    let result = ledger.limit(vec!["x".repeat(400)]);
    let message = result.error.expect("diagnostic").message;
    assert!(
        message.starts_with("retained variables were cleared"),
        "{message}"
    );
    let bytes = OutputLedger::render_logs(&result.logs).len()
        + serde_json::to_string(&message).expect("serializable").len();
    assert!(bytes <= 200);
}

#[test]
fn a_null_completion_value_counts_as_present() {
    // `Some(&Value::Null)` is a completion value; only `None` is "absent".
    let ledger = ledger(8);
    let result = ledger.success(Vec::new(), Some(&Value::Null));
    assert_eq!(result.value, Some(Value::Null));
    assert!(result.error.is_none());
    assert!(ledger.success(Vec::new(), None).value.is_none());
}

#[test]
fn the_program_side_buffer_reports_the_limit_exactly_once() {
    let mut buffer = ProgramLogBuffer::new(25);
    assert_eq!(buffer.push("first"), ProgramLogPush::Whole);
    assert_eq!(buffer.push("second"), ProgramLogPush::Whole);

    let third = buffer.push("third");
    assert_eq!(third, ProgramLogPush::Truncated("thir".into()));
    assert!(buffer.is_truncated());
    assert_eq!(buffer.remaining_bytes(), 0);

    // Later writes are dropped without a second signal: one exhausted budget
    // must not become two output-limit failures.
    assert_eq!(buffer.push("fourth"), ProgramLogPush::Suppressed);
    assert_eq!(buffer.push(""), ProgramLogPush::Suppressed);
}

#[test]
fn a_program_side_overflow_with_no_room_at_all_emits_no_prefix() {
    let mut buffer = ProgramLogBuffer::new(4);
    assert_eq!(buffer.push("abc"), ProgramLogPush::Truncated(String::new()));
    assert!(buffer.is_truncated());
}

#[test]
fn the_program_side_buffer_and_the_host_ledger_agree_on_the_budget() {
    // The same texts charge the same bytes on both sides, which is what keeps
    // a program's curated completion value admittable by the host.
    let mut buffer = ProgramLogBuffer::new(64);
    let mut ledger = ledger(64);
    let mut logs = Vec::new();
    for text in ["a\tb", "你好", "🙂"] {
        assert_eq!(buffer.push(text), ProgramLogPush::Whole);
        assert!(ledger.admit(text, &mut logs));
    }
    assert_eq!(buffer.remaining_bytes(), ledger.remaining_bytes());
}
