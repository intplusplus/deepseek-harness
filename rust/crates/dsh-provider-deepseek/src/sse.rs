//! Server-Sent Events framing for the streaming endpoint.
//!
//! `eventsource-parser` does this in the TypeScript adapter; without a new
//! dependency the framing lives here. Only the parts the endpoint uses are
//! implemented: `data` fields are joined with `\n` per event and dispatched on
//! the blank line that ends the event, comments (`: keep-alive`) are dropped,
//! and `event`/`id`/`retry` fields are ignored because the payload itself
//! carries everything the translator needs.
//!
//! An unterminated tail is **not** an event: a connection that dies mid-frame
//! must not deliver a half-parsed chunk, so the trailing bytes stay buffered
//! and are discarded with the framer.

#![forbid(unsafe_code)]

/// Incremental SSE decoder over raw response bytes.
#[derive(Debug, Default)]
pub(crate) struct SseFramer {
    buffer: Vec<u8>,
    data: Vec<String>,
}

impl SseFramer {
    /// Feed raw bytes and return the `data` payload of every event that ended.
    ///
    /// Bytes are buffered until a whole line is available, so a line split
    /// across TCP frames is decoded once and never lossily halved mid
    /// character.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        let mut data = std::mem::take(&mut self.data);
        let mut start = 0;
        while let Some(offset) = self.buffer[start..].iter().position(|byte| *byte == b'\n') {
            let end = start + offset;
            {
                let raw = &self.buffer[start..end];
                let line = String::from_utf8_lossy(raw);
                let line = line.strip_suffix('\r').unwrap_or(line.as_ref());
                if line.is_empty() {
                    if !data.is_empty() {
                        events.push(data.join("\n"));
                        data.clear();
                    }
                } else if let Some(value) = line.strip_prefix("data:") {
                    // The SSE grammar strips exactly one leading space.
                    data.push(value.strip_prefix(' ').unwrap_or(value).to_string());
                } else if line == "data" {
                    data.push(String::new());
                }
            }
            start = end + 1;
        }
        self.buffer.drain(..start);
        self.data = data;
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatches_on_the_blank_line_not_on_the_data_line() {
        let mut framer = SseFramer::default();
        assert!(framer.push(b"data: {\"a\":1}\n").is_empty());
        assert_eq!(framer.push(b"\n"), vec!["{\"a\":1}".to_string()]);
    }

    #[test]
    fn joins_a_line_split_across_tcp_frames() {
        let mut framer = SseFramer::default();
        assert!(framer.push(b"data: {\"cho").is_empty());
        assert!(framer.push(b"ices\":[]}\n").is_empty());
        assert_eq!(framer.push(b"\r\n"), vec![r#"{"choices":[]}"#.to_string()]);
    }

    #[test]
    fn keeps_a_multi_byte_character_whole() {
        let mut framer = SseFramer::default();
        let line = "data: 你好\n\n".as_bytes();
        // Split inside the first character's three-byte encoding.
        assert!(framer.push(&line[..7]).is_empty());
        assert_eq!(framer.push(&line[7..]), vec!["你好".to_string()]);
    }

    #[test]
    fn drops_comments_and_ignores_other_fields() {
        let mut framer = SseFramer::default();
        let events = framer.push(b": keep-alive\nevent: message\nid: 7\nretry: 10\ndata: x\n\n");
        assert_eq!(events, vec!["x".to_string()]);
    }

    #[test]
    fn joins_multiple_data_lines_of_one_event() {
        let mut framer = SseFramer::default();
        assert_eq!(
            framer.push(b"data: a\ndata:b\ndata\n\n"),
            vec!["a\nb\n".to_string()]
        );
    }

    #[test]
    fn never_delivers_an_unterminated_tail() {
        let mut framer = SseFramer::default();
        assert!(framer.push(b"data: {\"partial\":").is_empty());
        // Still no line terminator, so nothing is decodable yet.
        assert!(framer.push(b"1}").is_empty());
    }
}
