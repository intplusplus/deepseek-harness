//! The bounded control channel's frame format, without the channel.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node/src/channel.ts`. One
//! frame is a 4-byte big-endian body length followed by that many bytes of
//! UTF-8 JSON. Both ends validate, because the program can write directly to
//! the inherited descriptor: a declared length of zero, a declared length
//! above the configured cap, invalid UTF-8, and malformed JSON are all
//! *protocol* failures, not recoverable input.
//!
//! [`FrameDecoder`] is a byte-level incremental parser so a test can split a
//! frame at any offset — including inside the header — and still see exactly
//! one frame. [`FrameDecoder::finish`] is what turns a truncated tail into a
//! refusal instead of silence: a peer that ends mid-frame has not settled.

#![forbid(unsafe_code)]

use serde_json::Value;

/// Why a frame was refused.
///
/// Every variant carries the byte offset at which the decoder noticed, so an
/// operator can locate the failure inside a captured stream.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    /// A header declared a zero-length body.
    #[error("ptc frame: empty frame at byte {offset} (limit {limit} bytes)")]
    EmptyFrame {
        /// Bytes consumed before the offending header completed.
        offset: u64,
        /// The configured cap.
        limit: usize,
    },
    /// A header declared more than the configured cap.
    #[error(
        "ptc frame: declared {declared} bytes at byte {offset} exceeds the {limit}-byte limit"
    )]
    Oversized {
        /// The declared body length.
        declared: usize,
        /// The configured cap.
        limit: usize,
        /// Bytes consumed before the offending header completed.
        offset: u64,
    },
    /// A body was not valid UTF-8.
    #[error("ptc frame: body ending at byte {offset} is not UTF-8 (first bad byte {valid_up_to})")]
    InvalidUtf8 {
        /// Bytes consumed through the offending body.
        offset: u64,
        /// Offset of the first invalid sequence inside the body.
        valid_up_to: usize,
    },
    /// A body was not valid JSON.
    #[error("ptc frame: malformed JSON in the body ending at byte {offset}: {message}")]
    Malformed {
        /// Bytes consumed through the offending body.
        offset: u64,
        /// The parser's diagnostic.
        message: String,
    },
    /// The stream ended inside a header or a body.
    #[error("ptc frame: stream ended after {received} of {expected} bytes at byte {offset}")]
    Truncated {
        /// Bytes of the incomplete header or body expected in total.
        expected: usize,
        /// Bytes of it actually received.
        received: usize,
        /// Bytes consumed before the incomplete frame.
        offset: u64,
    },
}

/// Byte-level decoder for length-prefixed JSON frames.
///
/// Feed it arbitrary chunk boundaries; it yields zero or more complete frames
/// per call and keeps the partial remainder. It never allocates for a body it
/// has not already validated against the cap.
#[derive(Clone, Debug)]
pub struct FrameDecoder {
    limit: usize,
    header: [u8; 4],
    header_len: usize,
    body: Vec<u8>,
    body_len: usize,
    body_active: bool,
    consumed: u64,
}

impl FrameDecoder {
    /// A decoder that refuses bodies larger than `limit` bytes.
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            header: [0; 4],
            header_len: 0,
            body: Vec::new(),
            body_len: 0,
            body_active: false,
            consumed: 0,
        }
    }

    /// The configured body cap.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Whether a frame is partially buffered.
    pub fn is_partial(&self) -> bool {
        self.header_len != 0 || self.body_active
    }

    /// Consume one chunk, returning every frame it completed.
    ///
    /// The decoder stops at the first refused frame; the bytes after it are
    /// not inspected, because a stream whose framing is wrong cannot be
    /// resynchronized. If a *later* frame in the same chunk is refused, the
    /// frames before it in that call are dropped with the error rather than
    /// returned: the host treats a protocol refusal as terminal (it stops the
    /// run), so no message is lost that could still have been acted on. Feed
    /// one frame per call if a caller ever needs per-frame delivery before a
    /// refusal.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<Value>, FrameError> {
        let mut frames = Vec::new();
        let mut at = 0usize;
        while at < chunk.len() {
            if !self.body_active {
                let want = 4 - self.header_len;
                let take = want.min(chunk.len() - at);
                self.header[self.header_len..self.header_len + take]
                    .copy_from_slice(&chunk[at..at + take]);
                self.header_len += take;
                at += take;
                self.consumed += take as u64;
                if self.header_len < 4 {
                    continue;
                }
                let declared = u32::from_be_bytes(self.header) as usize;
                self.header_len = 0;
                if declared == 0 {
                    return Err(FrameError::EmptyFrame {
                        offset: self.consumed,
                        limit: self.limit,
                    });
                }
                if declared > self.limit {
                    return Err(FrameError::Oversized {
                        declared,
                        limit: self.limit,
                        offset: self.consumed,
                    });
                }
                self.body.clear();
                self.body.resize(declared, 0);
                self.body_len = 0;
                self.body_active = true;
            }

            let want = self.body.len() - self.body_len;
            let take = want.min(chunk.len() - at);
            self.body[self.body_len..self.body_len + take].copy_from_slice(&chunk[at..at + take]);
            self.body_len += take;
            at += take;
            self.consumed += take as u64;
            if self.body_len < self.body.len() {
                continue;
            }

            let body = std::mem::take(&mut self.body);
            self.body_len = 0;
            self.body_active = false;
            let text = std::str::from_utf8(&body).map_err(|error| FrameError::InvalidUtf8 {
                offset: self.consumed,
                valid_up_to: error.valid_up_to(),
            })?;
            let value: Value =
                serde_json::from_str(text).map_err(|error| FrameError::Malformed {
                    offset: self.consumed,
                    message: error.to_string(),
                })?;
            frames.push(value);
        }
        Ok(frames)
    }

    /// Confirm the stream ended on a frame boundary.
    ///
    /// A clean end is `Ok(())`; a header or body left half-read is
    /// [`FrameError::Truncated`], which the caller reports as a protocol
    /// failure rather than treating the missing message as absent.
    pub fn finish(&self) -> Result<(), FrameError> {
        if self.body_active {
            return Err(FrameError::Truncated {
                expected: self.body.len(),
                received: self.body_len,
                offset: self.consumed - self.body_len as u64,
            });
        }
        if self.header_len != 0 {
            return Err(FrameError::Truncated {
                expected: 4,
                received: self.header_len,
                offset: self.consumed - self.header_len as u64,
            });
        }
        Ok(())
    }
}

/// Encode one JSON value as a single frame.
///
/// Refuses an empty body and a body above `limit`, mirroring the decoder: a
/// writer that would produce a frame the peer must reject fails here instead.
pub fn encode_frame(value: &Value, limit: usize) -> Result<Vec<u8>, FrameError> {
    let body = serde_json::to_vec(value).map_err(|error| FrameError::Malformed {
        offset: 0,
        message: error.to_string(),
    })?;
    if body.is_empty() || body.len() > limit || body.len() > u32::MAX as usize {
        return Err(FrameError::Oversized {
            declared: body.len(),
            limit,
            offset: 0,
        });
    }
    let mut frame = Vec::with_capacity(body.len() + 4);
    frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
    frame.extend_from_slice(&body);
    Ok(frame)
}
