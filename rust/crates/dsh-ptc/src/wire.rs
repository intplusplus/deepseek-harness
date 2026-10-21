//! The flat lossless-JSON transport used across the process boundary.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node/src/json-wire.ts`.
//!
//! A pre-order token stream replaces the value itself, so serializing the
//! transport never recurses through the transported value's application
//! nesting. Container markers and scalar leaves share one flat array:
//!
//! ```text
//! [1, {kind:"array", length:2}, "a", {kind:"object", keys:["b"]}, 2]
//! ```
//!
//! Decoding is total and strict. Malformed, incomplete, ambiguous, and
//! non-lossless traffic all return `None` rather than a partially built value,
//! because the peer is model-generated code writing to an inherited
//! descriptor: it is an untrusted hostile writer, not a cooperating decoder.

#![forbid(unsafe_code)]

use serde_json::{Map, Value};

/// Largest integer the transport admits, matching JavaScript's
/// `Number.MAX_SAFE_INTEGER`: beyond it a container length is not exactly
/// representable, so a decoded length could disagree with the encoded one.
pub const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// Why a value could not be transported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WireError {
    /// The value did not fit the transport's representable range.
    #[error("ptc wire: value is not representable")]
    NotRepresentable,
}

/// A container marker, validated against the exact field set the wire admits.
enum Marker<'a> {
    Array { length: usize },
    Object { keys: Vec<&'a str> },
}

/// One partially built container during decoding.
enum Frame {
    Array {
        target: Vec<Value>,
        length: usize,
    },
    Object {
        target: Map<String, Value>,
        keys: Vec<String>,
        index: usize,
    },
}

/// Flatten one lossless JSON value into its pre-order token stream.
///
/// `serde_json::Value` cannot express `undefined`, a sparse slot, or a
/// decorated container, so — unlike the TypeScript encoder — this function
/// has no failure mode: every `Value` has a wire form.
pub fn encode_json_wire(value: &Value) -> Value {
    let mut tokens: Vec<Value> = Vec::new();
    let mut pending: Vec<&Value> = vec![value];
    while let Some(current) = pending.pop() {
        match current {
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
                tokens.push(current.clone());
            }
            Value::Array(items) => {
                let mut marker = Map::new();
                marker.insert("kind".to_owned(), Value::String("array".to_owned()));
                marker.insert("length".to_owned(), Value::from(items.len()));
                tokens.push(Value::Object(marker));
                pending.extend(items.iter().rev());
            }
            Value::Object(map) => {
                // `preserve_order` keeps the map in insertion order, which is
                // the key order the program's object literal had.
                let names: Vec<Value> = map.keys().map(|key| Value::String(key.clone())).collect();
                let mut marker = Map::new();
                marker.insert("kind".to_owned(), Value::String("object".to_owned()));
                marker.insert("keys".to_owned(), Value::Array(names));
                tokens.push(Value::Object(marker));
                pending.extend(map.values().rev());
            }
        }
    }
    Value::Array(tokens)
}

/// Rebuild one lossless JSON value from its token stream.
///
/// `None` means the input was not a well-formed stream: not a dense array, an
/// empty array, a malformed marker, a length that exceeds the tokens that
/// follow, a duplicate object key, a `-0`, or a non-finite number.
///
/// Traversal is iterative, so its own nesting is bounded by the input length
/// rather than by the call stack.
pub fn decode_json_wire(input: &Value) -> Option<Value> {
    let Value::Array(tokens) = input else {
        return None;
    };
    if tokens.is_empty() {
        return None;
    }

    let mut frames: Vec<Frame> = Vec::new();
    let mut root: Option<Value> = None;

    let attach = |frames: &mut Vec<Frame>, root: &mut Option<Value>, value: Value| -> bool {
        match frames.last_mut() {
            None => {
                if root.is_some() {
                    return false;
                }
                *root = Some(value);
                true
            }
            Some(Frame::Array { target, length }) => {
                if target.len() >= *length {
                    return false;
                }
                target.push(value);
                true
            }
            Some(Frame::Object {
                target,
                keys,
                index,
            }) => {
                let Some(key) = keys.get(*index) else {
                    return false;
                };
                if target.contains_key(key) {
                    return false;
                }
                let key = key.clone();
                target.insert(key, value);
                *index += 1;
                true
            }
        }
    };

    for (position, token) in tokens.iter().enumerate() {
        let remaining = tokens.len() - position - 1;
        match token {
            Value::Null | Value::Bool(_) | Value::String(_) => {
                if !attach(&mut frames, &mut root, token.clone()) {
                    return None;
                }
            }
            Value::Number(number) => {
                let float = number.as_f64()?;
                if !float.is_finite() {
                    return None;
                }
                // `-0` has no lossless JSON form: it survives no round trip
                // through most consumers and serializes as `0`.
                if float == 0.0 && float.is_sign_negative() {
                    return None;
                }
                if !attach(&mut frames, &mut root, token.clone()) {
                    return None;
                }
            }
            Value::Array(_) => return None,
            Value::Object(_) => {
                let marker = container_marker(token)?;
                let frame = match marker {
                    Marker::Array { length } => {
                        if length > remaining {
                            return None;
                        }
                        if length == 0 {
                            if !attach(&mut frames, &mut root, Value::Array(Vec::new())) {
                                return None;
                            }
                            None
                        } else {
                            Some(Frame::Array {
                                target: Vec::with_capacity(length.min(remaining)),
                                length,
                            })
                        }
                    }
                    Marker::Object { keys } => {
                        if keys.len() > remaining {
                            return None;
                        }
                        if keys.is_empty() {
                            if !attach(&mut frames, &mut root, Value::Object(Map::new())) {
                                return None;
                            }
                            None
                        } else {
                            Some(Frame::Object {
                                target: Map::new(),
                                keys: keys.into_iter().map(str::to_owned).collect(),
                                index: 0,
                            })
                        }
                    }
                };
                if let Some(frame) = frame {
                    frames.push(frame);
                }
            }
        }

        // Close every container that just filled, handing the finished value
        // to its parent. A container that filled before it was pushed (the
        // empty case above) was already attached directly.
        while frames.last().is_some_and(frame_is_complete) {
            let frame = frames.pop().expect("checked above");
            let finished = match frame {
                Frame::Array { target, .. } => Value::Array(target),
                Frame::Object { target, .. } => Value::Object(target),
            };
            if !attach(&mut frames, &mut root, finished) {
                return None;
            }
        }
    }

    if frames.is_empty() {
        root
    } else {
        None
    }
}

fn frame_is_complete(frame: &Frame) -> bool {
    match frame {
        Frame::Array { target, length } => target.len() == *length,
        Frame::Object { keys, index, .. } => *index == keys.len(),
    }
}

/// Validate one container marker: exactly the two declared fields, no extras.
fn container_marker(token: &Value) -> Option<Marker<'_>> {
    let Value::Object(map) = token else {
        return None;
    };
    if map.len() != 2 || !map.contains_key("kind") {
        return None;
    }
    match map.get("kind")?.as_str()? {
        "array" => {
            if !map.contains_key("length") {
                return None;
            }
            let length = map.get("length")?.as_u64()?;
            if length > MAX_SAFE_INTEGER {
                return None;
            }
            Some(Marker::Array {
                length: usize::try_from(length).ok()?,
            })
        }
        "object" => {
            let keys = map.get("keys")?.as_array()?;
            if keys.len() > MAX_SAFE_INTEGER as usize {
                return None;
            }
            let mut names: Vec<&str> = Vec::with_capacity(keys.len());
            for key in keys {
                let name = key.as_str()?;
                if names.contains(&name) {
                    return None;
                }
                names.push(name);
            }
            Some(Marker::Object { keys: names })
        }
        _ => None,
    }
}
