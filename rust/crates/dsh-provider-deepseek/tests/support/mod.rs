//! A hand-rolled HTTP/1.1 server for the provider's integration tests.
//!
//! The workspace has no `hyper` or `wiremock`, so the fake DeepSeek endpoint is
//! a `TcpListener` that reads one request, records it, and replies with a
//! scripted body. The four body-ending modes exist because the provider treats
//! them differently: a terminated chunked body completes, an unterminated one
//! is a transport failure, a clean EOF without `[DONE]` is a closed stream, and
//! a fixed length is a plain non-streaming reply.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// One request the server received.
#[derive(Clone, Debug)]
pub(crate) struct Recorded {
    /// Request line plus headers, verbatim.
    pub head: String,
    /// Request body, decoded as UTF-8.
    pub body: String,
}

impl Recorded {
    /// A header's value, case-insensitively.
    pub fn header(&self, name: &str) -> Option<String> {
        self.head.lines().skip(1).find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case(name)
                .then(|| value.trim().to_string())
        })
    }

    /// The request line, e.g. `POST /v1/chat/completions HTTP/1.1`.
    pub fn request_line(&self) -> &str {
        self.head.lines().next().unwrap_or_default()
    }

    /// The body parsed as JSON.
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).expect("request body is JSON")
    }
}

/// How the response body is delimited and terminated.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Ending {
    /// `Content-Length` over every frame.
    Length,
    /// Chunked, with the terminating zero chunk.
    Chunked,
    /// Chunked, but the socket dies mid-chunk.
    Truncated,
    /// No length, no chunking: the body ends when the socket closes.
    UntilClose,
}

/// A scripted response.
#[derive(Clone, Debug)]
pub(crate) struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub frames: Vec<String>,
    pub ending: Ending,
}

impl Response {
    /// A JSON body with a fixed length.
    pub fn json(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            frames: vec![body.into()],
            ending: Ending::Length,
        }
    }

    /// A complete Server-Sent Events body.
    pub fn sse(frames: Vec<String>) -> Self {
        Self {
            status: 200,
            headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
            frames,
            ending: Ending::Chunked,
        }
    }

    /// An SSE body whose socket dies before the terminating chunk.
    pub fn truncated_sse(frames: Vec<String>) -> Self {
        Self {
            ending: Ending::Truncated,
            ..Self::sse(frames)
        }
    }

    /// An SSE body that ends by closing the connection, with no `[DONE]`.
    pub fn sse_until_close(frames: Vec<String>) -> Self {
        Self {
            ending: Ending::UntilClose,
            ..Self::sse(frames)
        }
    }
}

/// A running fake server plus the requests it recorded.
pub(crate) struct MockServer {
    pub base_url: String,
    recorded: Arc<Mutex<Vec<Recorded>>>,
}

impl MockServer {
    /// Every request received so far, in order.
    pub fn requests(&self) -> Vec<Recorded> {
        self.recorded.lock().expect("lock").clone()
    }

    /// The single request received, failing loudly when there is not exactly one.
    pub fn only_request(&self) -> Recorded {
        let requests = self.requests();
        assert_eq!(requests.len(), 1, "expected exactly one request");
        requests.into_iter().next().expect("one request")
    }
}

/// Start a server that answers one connection with `response`.
pub(crate) async fn serve(response: Response) -> MockServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("local address");
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&recorded);
    tokio::spawn(async move {
        let Ok((mut socket, _)) = listener.accept().await else {
            return;
        };
        let Some(request) = read_request(&mut socket).await else {
            return;
        };
        log.lock().expect("lock").push(request);
        write_response(&mut socket, response).await;
    });
    MockServer {
        base_url: format!("http://{address}"),
        recorded,
    }
}

/// Read one request: the head, then exactly `Content-Length` body bytes.
async fn read_request(socket: &mut TcpStream) -> Option<Recorded> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        let read = socket.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        let Some(end) = position(&buffer, b"\r\n\r\n") else {
            continue;
        };
        let head = String::from_utf8_lossy(&buffer[..end]).into_owned();
        let body_start = end + 4;
        let length = content_length(&head);
        while buffer.len() < body_start + length {
            let read = socket.read(&mut chunk).await.ok()?;
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);
        }
        let body = String::from_utf8_lossy(&buffer[body_start..]).into_owned();
        return Some(Recorded { head, body });
    }
}

/// Write `response` and let the socket close.
async fn write_response(socket: &mut TcpStream, response: Response) {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nconnection: close\r\n",
        response.status,
        reason(response.status)
    );
    for (name, value) in &response.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    match response.ending {
        Ending::Length => {
            let body = response.frames.concat();
            head.push_str(&format!("content-length: {}\r\n\r\n", body.len()));
            let _ = socket.write_all(head.as_bytes()).await;
            let _ = socket.write_all(body.as_bytes()).await;
        }
        Ending::Chunked | Ending::Truncated => {
            head.push_str("transfer-encoding: chunked\r\n\r\n");
            let _ = socket.write_all(head.as_bytes()).await;
            for frame in &response.frames {
                let chunk = format!("{:x}\r\n{frame}\r\n", frame.len());
                if socket.write_all(chunk.as_bytes()).await.is_err() {
                    return;
                }
                let _ = socket.flush().await;
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            match response.ending {
                Ending::Chunked => {
                    let _ = socket.write_all(b"0\r\n\r\n").await;
                }
                // A chunk header promising 64 bytes, followed by far fewer and
                // then EOF: the client must see a broken body, not a short one.
                _ => {
                    let _ = socket.write_all(b"40\r\npartial").await;
                }
            }
        }
        Ending::UntilClose => {
            head.push_str("\r\n");
            let _ = socket.write_all(head.as_bytes()).await;
            for frame in &response.frames {
                if socket.write_all(frame.as_bytes()).await.is_err() {
                    return;
                }
                let _ = socket.flush().await;
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        }
    }
    let _ = socket.flush().await;
    let _ = socket.shutdown().await;
}

/// `Content-Length` from a request head, defaulting to zero.
fn content_length(head: &str) -> usize {
    head.lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(key, _)| key.trim().eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse().ok())
        .unwrap_or(0)
}

/// Index of `needle` inside `haystack`.
fn position(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Reason phrase for the statuses the tests use.
fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Error",
    }
}
