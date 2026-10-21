//! Length-prefixed control frames.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node/tests/channel.spec.ts`,
//! with the stream replaced by byte slices so every split point is explicit.

use dsh_ptc::{encode_frame, FrameDecoder, FrameError};
use serde_json::{json, Value};

fn feed(decoder: &mut FrameDecoder, bytes: &[u8]) -> Vec<Value> {
    decoder.push(bytes).expect("well-formed frames")
}

#[test]
fn reassembles_every_split_point_including_multibyte_bodies() {
    let value = json!({ "text": "你好🙂" });
    let frame = encode_frame(&value, 1_000).expect("frame fits");

    // One byte at a time splits headers, payload boundaries, and multi-byte
    // code points.
    let mut decoder = FrameDecoder::new(1_000);
    let mut received = Vec::new();
    for byte in &frame {
        received.extend(feed(&mut decoder, &[*byte]));
    }
    assert_eq!(received, vec![value.clone()]);
    assert!(!decoder.is_partial());
    decoder.finish().expect("a complete stream ends cleanly");

    // Every two-way split of the same frame yields the same single message.
    for split in 0..=frame.len() {
        let mut decoder = FrameDecoder::new(1_000);
        let mut received = feed(&mut decoder, &frame[..split]);
        received.extend(feed(&mut decoder, &frame[split..]));
        assert_eq!(received, vec![value.clone()], "split at {split}");
        decoder.finish().expect("clean end");
    }
}

#[test]
fn accepts_consecutive_frames_in_one_chunk() {
    let mut chunk = encode_frame(&json!({ "n": 1 }), 1_000).expect("frame");
    chunk.extend(encode_frame(&json!({ "n": 2 }), 1_000).expect("frame"));
    let mut decoder = FrameDecoder::new(1_000);
    assert_eq!(
        feed(&mut decoder, &chunk),
        vec![json!({ "n": 1 }), json!({ "n": 2 })]
    );
}

#[test]
fn rejects_an_empty_or_oversized_declaration_before_the_body() {
    for declared in [0u32, 65] {
        let mut decoder = FrameDecoder::new(64);
        let error = decoder
            .push(&declared.to_be_bytes())
            .expect_err("the declaration alone is refused");
        match (declared, error) {
            (0, FrameError::EmptyFrame { limit, offset }) => {
                assert_eq!(limit, 64);
                assert_eq!(offset, 4);
            }
            (
                65,
                FrameError::Oversized {
                    declared,
                    limit,
                    offset,
                },
            ) => {
                assert_eq!((declared, limit, offset), (65, 64, 4));
            }
            other => panic!("unexpected refusal {other:?}"),
        }
        // Nothing was buffered, so a later good frame still decodes: the
        // refusal is a decision, not a desynchronized stream.
        assert!(!decoder.is_partial());
    }
}

#[test]
fn a_truncated_frame_is_refused_at_end_of_stream() {
    let frame = encode_frame(&json!({ "value": "abcdefgh" }), 1_000).expect("frame");

    let mut in_header = FrameDecoder::new(1_000);
    in_header
        .push(&frame[..2])
        .expect("a partial header is not an error");
    assert_eq!(
        in_header.finish(),
        Err(FrameError::Truncated {
            expected: 4,
            received: 2,
            offset: 0,
        })
    );

    let mut in_body = FrameDecoder::new(1_000);
    in_body
        .push(&frame[..frame.len() - 3])
        .expect("a partial body is not an error");
    match in_body.finish() {
        Err(FrameError::Truncated {
            expected,
            received,
            offset,
        }) => {
            assert_eq!(expected, frame.len() - 4);
            assert_eq!(received, frame.len() - 4 - 3);
            // The header was complete, so the incomplete frame starts at 4.
            assert_eq!(offset, 4);
        }
        other => panic!("expected a truncation, got {other:?}"),
    }
}

#[test]
fn invalid_utf8_and_malformed_json_carry_their_offset() {
    for (body, expected_message) in [(vec![0xffu8], "UTF-8"), (b"{".to_vec(), "malformed JSON")] {
        let mut bytes = (body.len() as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(&body);
        let mut decoder = FrameDecoder::new(64);
        let error = decoder.push(&bytes).expect_err("the body is refused");
        assert!(
            error.to_string().contains(expected_message),
            "{error} should mention {expected_message}"
        );
        assert!(
            error.to_string().contains("byte 5"),
            "{error} should locate the end of the body"
        );
    }
}

#[test]
fn encoding_refuses_what_a_peer_must_reject() {
    assert!(matches!(
        encode_frame(&json!({ "text": "x".repeat(30) }), 20),
        Err(FrameError::Oversized { limit: 20, .. })
    ));
    assert!(encode_frame(&json!({ "ok": true }), 20).is_ok());
}

#[test]
fn a_zero_limit_refuses_every_body_including_the_empty_object() {
    let mut decoder = FrameDecoder::new(0);
    assert!(matches!(
        decoder.push(&2u32.to_be_bytes()),
        Err(FrameError::Oversized {
            declared: 2,
            limit: 0,
            ..
        })
    ));
    assert!(encode_frame(&json!({}), 0).is_err());
}
