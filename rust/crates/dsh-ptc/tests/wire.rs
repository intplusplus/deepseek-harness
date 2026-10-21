//! The flat JSON wire format.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node/tests/json-wire.spec.ts`
//! (the `encodePtcJsonWire` / `decodePtcJsonWire` halves; the snapshot half has
//! no Rust counterpart because `serde_json::Value` is already lossless JSON).

use dsh_ptc::{decode_json_wire, encode_json_wire, MAX_SAFE_INTEGER};
use serde_json::{json, Map, Value};

fn round_trip(value: Value) {
    let wire = encode_json_wire(&value);
    assert_eq!(decode_json_wire(&wire), Some(value));
}

#[test]
fn round_trips_every_json_root_and_preserves_key_order() {
    for value in [
        json!(null),
        json!(false),
        json!(true),
        json!(1.25),
        json!("text"),
        json!([]),
        json!({}),
        json!([1, { "nested": [2] }]),
        json!({ "b": 1, "a": 2, "c": [] }),
    ] {
        round_trip(value);
    }

    let decoded = decode_json_wire(&encode_json_wire(&json!({ "__proto__": { "safe": true } })))
        .expect("a prototype-named key is an ordinary key");
    assert_eq!(decoded["__proto__"], json!({ "safe": true }));
    assert_eq!(
        decoded
            .as_object()
            .expect("object")
            .keys()
            .collect::<Vec<_>>(),
        ["__proto__"]
    );
}

#[test]
fn the_wire_is_a_flat_pre_order_token_stream() {
    assert_eq!(
        encode_json_wire(&json!([1, { "a": [2] }])),
        json!([
            { "kind": "array", "length": 2 },
            1,
            { "kind": "object", "keys": ["a"] },
            { "kind": "array", "length": 1 },
            2
        ])
    );
    // Container markers keep `kind` first so a hand-written frame matches.
    assert_eq!(
        encode_json_wire(&json!([])).to_string(),
        r#"[{"kind":"array","length":0}]"#
    );
}

#[test]
fn round_trips_deep_values_without_recursing() {
    // The codec is iterative, so its own depth bound is the token count. The
    // *value* still has a recursive `Drop` (serde_json owns that, not this
    // crate), so the assertion runs on a thread with room for it — the point
    // being that the traversal never recursed, not that json carries an
    // unlimited value.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let mut value = json!("leaf");
            for _ in 0..5_000 {
                value = Value::Array(vec![value]);
            }
            let wire = encode_json_wire(&value);
            assert_eq!(wire.as_array().expect("tokens").len(), 5_001);

            let decoded = decode_json_wire(&wire).expect("bounded-depth round trip");
            let mut cursor = &decoded;
            for depth in 0..5_000 {
                let items = cursor
                    .as_array()
                    .unwrap_or_else(|| panic!("depth {depth} is not an array"));
                cursor = &items[0];
            }
            assert_eq!(*cursor, json!("leaf"));
        })
        .expect("spawn")
        .join()
        .expect("the deep round trip holds");
}

#[test]
fn rejects_malformed_and_incomplete_wire_values() {
    for value in [
        json!(null),
        json!({}),
        json!([]),
        json!([1, 2]),
        json!([[]]),
        json!([{ "kind": "unknown" }]),
        json!([{ "kind": "array", "bogus": 0 }]),
        json!([{ "kind": "array" }]),
        json!([{ "kind": "array", "length": "1" }]),
        json!([{ "kind": "array", "length": -1 }]),
        json!([{ "kind": "array", "length": 1 }]),
        json!([{ "kind": "array", "length": 2 }, { "kind": "array", "length": 1 }, null]),
        json!([{ "kind": "array", "length": 0, "extra": true }]),
        json!([{ "kind": "object" }]),
        json!([{ "kind": "object", "keys": "x" }]),
        json!([{ "kind": "object", "keys": [1] }]),
        json!([{ "kind": "object", "keys": ["x", "x"] }, 1, 2]),
        json!([{ "kind": "object", "keys": ["x"] }]),
        json!([{ "kind": "object", "keys": [], "extra": true }]),
        // A safe-integer overflow would let the declared length disagree with
        // the tokens that follow.
        json!([{ "kind": "array", "length": MAX_SAFE_INTEGER + 1 }]),
    ] {
        assert_eq!(decode_json_wire(&value), None, "accepted {value}");
    }
}

#[test]
fn rejects_negative_zero() {
    // `-0` has no lossless JSON form, so it is not transportable even though
    // it is a valid `f64`.
    let wire = Value::Array(vec![Value::from(-0.0f64)]);
    assert_eq!(decode_json_wire(&wire), None);
    // `0` itself is fine.
    assert_eq!(decode_json_wire(&json!([0])), Some(json!(0)));
}

#[test]
fn a_container_marker_must_be_a_plain_two_field_object() {
    let mut marker = Map::new();
    marker.insert("length".to_owned(), json!(0));
    marker.insert("kind".to_owned(), json!("array"));
    // Same two fields in another order: the wire is order-insensitive on
    // input and normalizes to `kind` first on output.
    assert_eq!(
        decode_json_wire(&Value::Array(vec![Value::Object(marker)])),
        Some(json!([]))
    );

    let mut extra = Map::new();
    extra.insert("kind".to_owned(), json!("array"));
    extra.insert("length".to_owned(), json!(0));
    extra.insert("length2".to_owned(), json!(0));
    assert_eq!(
        decode_json_wire(&Value::Array(vec![Value::Object(extra)])),
        None
    );
}
