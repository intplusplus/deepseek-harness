//! Translation between the canonical vocabulary and the OpenAI-compatible
//! `chat/completions` wire format.
//!
//! Both directions are pure: no I/O happens here, so every mapping decision is
//! unit-testable without a socket.
//!
//! ## Request direction
//!
//! `LlmRequest` carries five message roles and structured content blocks; the
//! wire format has three roles and one string per message. The flattening
//! rules are:
//!
//! - every `system` message collapses into one leading `system` message, joined
//!   with a blank line (the TypeScript adapter joins its system prompt and its
//!   in-history system text the same way);
//! - a `developer` message becomes an in-place `system` message carrying its
//!   text; its `tool-addition`/`tool-removal` blocks are dropped because this
//!   endpoint re-declares the complete tool list on every request;
//! - text blocks inside one message are joined with `\n`, and adjacent messages
//!   of the same wire role are joined with a blank line, because string content
//!   cannot express separate parts;
//! - `image` and `file` blocks are rendered as placeholders: this transport has
//!   no attachment store, so it can never send the bytes;
//! - assistant `reasoning` blocks are dropped: the OpenAI surface has no signed
//!   thinking replay, and DeepSeek rejects `reasoning_content` in input;
//! - assistant `tool-call` blocks become `tool_calls`, tool results become
//!   `tool` messages paired by `tool_call_id`.
//!
//! ## Response direction
//!
//! A completion becomes reasoning → text → tool calls, in that order, and a
//! streamed delta run is accumulated into the same shape while the exact timed
//! record is preserved for replay.

#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use dsh_llm::{
    AssistantMessage, AssistantStreamRecord, ContentBlock, CoreContentBlock, FileAttachmentRef,
    FinishReason, ImageAttachmentRef, LlmFailure, Message, MessageId, StreamChunk, TokenUsage,
    ToolCallId, ToolSchema,
};
use dsh_provider::{LlmRequest, LlmResponse};
use serde_json::Value;

use crate::error;
use crate::wire::{
    WireChatCompletion, WireChunk, WireDelta, WireMessage, WireRequest, WireStreamOptions,
    WireTool, WireToolCallDelta, WireToolCallFunction, WireToolCallRequest, WireToolCallResponse,
    WireToolCallResponseFunction, WireUsage,
};

/// Translate one canonical request into a `chat/completions` body.
pub(crate) fn build_request(request: &LlmRequest, stream: bool) -> Result<WireRequest, LlmFailure> {
    let mut prompt = Vec::new();
    let system = system_text(&request.messages)?;
    if !system.is_empty() {
        prompt.push(system);
    }
    // A developer message is an in-place system update in the Messages
    // transport. This endpoint expects one prompt at the head, so a developer
    // message's text is appended to it in encounter order; its tool changes are
    // dropped because `tools` already declares the complete list.
    for message in &request.messages {
        if let Message::Developer(developer) = message {
            let text = developer_text(&developer.content)?;
            if !text.is_empty() {
                prompt.push(text);
            }
        }
    }
    let mut messages = Vec::new();
    if !prompt.is_empty() {
        messages.push(text_message("system", prompt.join("\n\n")));
    }
    // Call ids whose results have not appeared yet. A result that answers
    // nothing, or a call answered twice, would be rejected by the server with a
    // 400 and no useful detail, so the history is checked here instead.
    let mut pending: HashSet<String> = HashSet::new();
    for message in &request.messages {
        match message {
            // Folded into the leading system message above.
            Message::System(_) | Message::Developer(_) => {}
            Message::User(user) => {
                let text = content_text(&user.content)?;
                if !text.is_empty() {
                    push_user_text(&mut messages, text);
                }
            }
            Message::Assistant(assistant) => {
                let text = assistant_text(&assistant.content);
                let calls: Vec<WireToolCallRequest> = assistant
                    .content
                    .iter()
                    .filter_map(ContentBlock::as_tool_call)
                    .map(|call| WireToolCallRequest {
                        id: call.id.as_str().to_string(),
                        kind: "function".to_string(),
                        function: WireToolCallFunction {
                            name: call.name.to_string(),
                            arguments: call.arguments.to_string(),
                        },
                    })
                    .collect();
                if text.is_empty() && calls.is_empty() {
                    // An assistant message with nothing model-visible left is
                    // omitted rather than sent empty, which some servers reject.
                    continue;
                }
                for call in &calls {
                    if !pending.insert(call.id.clone()) {
                        return Err(error::invalid_request(format!(
                            "duplicate tool call id `{}`",
                            call.id
                        )));
                    }
                }
                messages.push(WireMessage {
                    role: "assistant".to_string(),
                    content: (!text.is_empty()).then(|| Value::String(text)),
                    tool_calls: (!calls.is_empty()).then_some(calls),
                    tool_call_id: None,
                });
            }
            Message::Tool(tool) => {
                // A tool result is flattened exactly like user content, so an
                // attachment inside one degrades to the same placeholder
                // instead of vanishing from the request.
                let text = content_text(&tool.content)?;
                if !pending.remove(tool.tool_call_id.as_str()) {
                    return Err(error::invalid_request(format!(
                        "tool result `{}` answers no pending tool call",
                        tool.tool_call_id
                    )));
                }
                messages.push(WireMessage {
                    role: "tool".to_string(),
                    content: Some(Value::String(text)),
                    tool_calls: None,
                    tool_call_id: Some(tool.tool_call_id.as_str().to_string()),
                });
            }
        }
    }
    if !pending.is_empty() {
        return Err(error::invalid_request(
            "history ends with tool calls that have no result",
        ));
    }
    Ok(WireRequest {
        model: request.config.model.clone(),
        messages,
        tools: (!request.tools.is_empty()).then(|| request.tools.iter().map(wire_tool).collect()),
        stream,
        stream_options: stream.then_some(WireStreamOptions {
            include_usage: true,
        }),
        reasoning_effort: reasoning_effort(request.config.reasoning_effort.as_deref()),
        temperature: request.config.temperature,
        max_tokens: request.config.max_tokens,
        stop: request.config.stop.clone(),
    })
}

/// Translate a non-streaming completion into a canonical response.
pub(crate) fn completion_response(
    completion: WireChatCompletion,
    provider: &str,
    model: &str,
) -> Result<LlmResponse, LlmFailure> {
    // `n` is never requested, so the first choice is ours; selecting by index
    // keeps a gateway that returns index 1 first from shifting the answer.
    let choice = completion
        .choices
        .into_iter()
        .find(|choice| choice.index == 0)
        .ok_or_else(|| error::malformed("completion carried no choice with index 0"))?;
    let mut blocks = Vec::new();
    if let Some(reasoning) = choice
        .message
        .reasoning_content
        .filter(|text| !text.is_empty())
    {
        blocks.push(ContentBlock::reasoning(reasoning));
    }
    if let Some(text) = choice.message.content.filter(|text| !text.is_empty()) {
        blocks.push(ContentBlock::text(text));
    }
    for (position, call) in choice
        .message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .enumerate()
    {
        blocks.push(tool_call_block(call, position)?);
    }
    let has_calls = blocks.iter().any(|block| block.as_tool_call().is_some());
    let reason = map_finish_reason(choice.finish_reason.as_deref(), has_calls);
    if blocks.is_empty() && reason == FinishReason::Stop {
        return Err(error::empty_response());
    }
    let usage = completion.usage.as_ref().map(token_usage);
    let message = AssistantMessage::new(
        MessageId::random(),
        blocks,
        provider.to_string(),
        model.to_string(),
    );
    // A non-streaming call delivers no timed stream; `stream` stays empty so a
    // replay reproduces "one whole response" rather than an invented cadence.
    Ok(LlmResponse {
        message,
        stream: Vec::new(),
        usage,
        finish_reason: reason,
    })
}

/// Which canonical block a streamed delta belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKind {
    Text,
    Reasoning,
    ToolCall,
}

impl BlockKind {
    /// The canonical block-type tag carried by `block-start`.
    fn tag(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Reasoning => "reasoning",
            Self::ToolCall => "tool-call",
        }
    }
}

/// One block being accumulated from a stream.
struct StreamBlock {
    /// Canonical index: the order the block first appeared.
    index: u64,
    kind: BlockKind,
    text: String,
    call_id: String,
    name: String,
    arguments: String,
    /// Index of the newest packed run for this block inside `records`.
    record: Option<usize>,
}

impl StreamBlock {
    fn new(index: u64, kind: BlockKind) -> Self {
        Self {
            index,
            kind,
            text: String::new(),
            call_id: String::new(),
            name: String::new(),
            arguments: String::new(),
            record: None,
        }
    }

    fn content_block(&self) -> ContentBlock {
        match self.kind {
            BlockKind::Text => ContentBlock::text(self.text.clone()),
            BlockKind::Reasoning => ContentBlock::reasoning(self.text.clone()),
            BlockKind::ToolCall => ContentBlock::tool_call(
                // A gateway that never sent an id still needs a stable pairing
                // key; the block index is unique within the response.
                ToolCallId::new(if self.call_id.is_empty() {
                    format!("call_{}", self.index)
                } else {
                    self.call_id.clone()
                }),
                self.name.clone(),
                // Empty arguments are `{}` rather than the empty string, so a
                // downstream JSON parse of a zero-argument call succeeds.
                if self.arguments.is_empty() {
                    "{}".to_string()
                } else {
                    self.arguments.clone()
                },
            ),
        }
    }
}

/// Accumulates streamed chunks into one canonical response.
///
/// The wire format has no block boundaries: a block is inferred from the delta
/// kind and, for tool calls, from the provider's `index`. Deltas are packed
/// into the same `*Chunks` records the session log stores, so replay keeps each
/// delta boundary and its original offset.
pub(crate) struct StreamAssembler {
    provider: String,
    model: String,
    records: Vec<AssistantStreamRecord>,
    blocks: Vec<StreamBlock>,
    /// Provider tool index → position in `blocks`. Parallel calls interleave.
    tools: HashMap<u64, usize>,
    last_tool_index: Option<u64>,
    usage: Option<TokenUsage>,
    finish: Option<String>,
    done: bool,
}

impl StreamAssembler {
    pub(crate) fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            records: Vec::new(),
            blocks: Vec::new(),
            tools: HashMap::new(),
            last_tool_index: None,
            usage: None,
            finish: None,
            done: false,
        }
    }

    /// Whether the terminal `[DONE]` frame has arrived.
    pub(crate) fn is_done(&self) -> bool {
        self.done
    }

    /// Decode one complete SSE `data` payload.
    pub(crate) fn push_frame(&mut self, data: &str) -> Result<(), LlmFailure> {
        if data.trim() == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        let payload: Value =
            serde_json::from_str(data).map_err(|_| error::malformed("stream frame is not JSON"))?;
        if payload.get("error").is_some() {
            return Err(error::map_in_band_error(&payload));
        }
        let chunk: WireChunk = serde_json::from_value(payload)
            .map_err(|failure| error::malformed(format!("unexpected stream frame: {failure}")))?;
        if let Some(usage) = chunk.usage.as_ref() {
            self.usage = Some(token_usage(usage));
        }
        let now = now_ms();
        for choice in chunk.choices {
            // `n` is never sent, so only the first choice is ours. A gateway
            // that multiplexes extras must not corrupt the block order.
            if choice.index != 0 {
                continue;
            }
            if let Some(reason) = choice.finish_reason {
                self.finish = Some(reason);
            }
            let WireDelta {
                content,
                reasoning_content,
                tool_calls,
                ..
            } = choice.delta;
            if let Some(text) = reasoning_content.filter(|text| !text.is_empty()) {
                self.push_text(BlockKind::Reasoning, text, now);
            }
            if let Some(text) = content.filter(|text| !text.is_empty()) {
                self.push_text(BlockKind::Text, text, now);
            }
            for call in tool_calls.unwrap_or_default() {
                self.push_tool_delta(call, now);
            }
        }
        Ok(())
    }

    /// Settle the accumulation into a response.
    ///
    /// A stream that ended after a `finish_reason` but without `[DONE]` is
    /// accepted: the response is complete by the provider's own statement, and
    /// a gateway that omits the sentinel must not lose a finished turn. A
    /// stream with neither is a truncated read and fails as `STREAM_CLOSED`.
    pub(crate) fn finish(mut self) -> Result<LlmResponse, LlmFailure> {
        if !self.done && self.finish.is_none() {
            return Err(error::stream_closed());
        }
        let rendered: Vec<(u64, ContentBlock)> = self
            .blocks
            .iter()
            .map(|block| (block.index, block.content_block()))
            .collect();
        let content: Vec<ContentBlock> = rendered.iter().map(|(_, block)| block.clone()).collect();
        let has_calls = self
            .blocks
            .iter()
            .any(|block| block.kind == BlockKind::ToolCall);
        let reason = map_finish_reason(self.finish.as_deref(), has_calls);
        if content.is_empty() && reason == FinishReason::Stop {
            return Err(error::empty_response());
        }
        let now = now_ms();
        for (index, block) in &rendered {
            self.records.push(AssistantStreamRecord::Chunk {
                time: now,
                chunk: Box::new(StreamChunk::BlockEnd {
                    index: *index,
                    block: block.clone(),
                }),
            });
        }
        if let Some(usage) = self.usage.clone() {
            self.records.push(AssistantStreamRecord::Chunk {
                time: now,
                chunk: Box::new(StreamChunk::Usage { usage }),
            });
        }
        self.records.push(AssistantStreamRecord::Chunk {
            time: now,
            chunk: Box::new(StreamChunk::Finish {
                reason: reason.clone(),
                // No adapter-private replay state exists: this wire format
                // needs no signature to rebuild the history.
                replay_state: None,
            }),
        });
        let message =
            AssistantMessage::new(MessageId::random(), content, self.provider, self.model);
        Ok(LlmResponse {
            message,
            stream: self.records,
            usage: self.usage,
            finish_reason: reason,
        })
    }

    /// Append a text or reasoning delta, opening the block on first use.
    fn push_text(&mut self, kind: BlockKind, text: String, now: u64) {
        let position = match self.blocks.last() {
            Some(block) if block.kind == kind => self.blocks.len() - 1,
            _ => {
                let index = self.blocks.len() as u64;
                self.blocks.push(StreamBlock::new(index, kind));
                self.records.push(AssistantStreamRecord::Chunk {
                    time: now,
                    chunk: Box::new(StreamChunk::BlockStart {
                        index,
                        block_type: kind.tag().to_string(),
                    }),
                });
                self.blocks.len() - 1
            }
        };
        self.blocks[position].text.push_str(&text);
        self.append_text_run(position, &text, now);
    }

    /// Append one tool-call delta, correlating parallel calls by provider index.
    fn push_tool_delta(&mut self, call: WireToolCallDelta, now: u64) {
        // Some gateways omit `index`; the previous call is then the only
        // sensible target, and a first delta without one starts at zero.
        let wire_index = call.index.or(self.last_tool_index).unwrap_or(0);
        self.last_tool_index = Some(wire_index);
        let position = match self.tools.get(&wire_index).copied() {
            Some(position) => position,
            None => {
                let index = self.blocks.len() as u64;
                self.blocks
                    .push(StreamBlock::new(index, BlockKind::ToolCall));
                self.records.push(AssistantStreamRecord::Chunk {
                    time: now,
                    chunk: Box::new(StreamChunk::BlockStart {
                        index,
                        block_type: BlockKind::ToolCall.tag().to_string(),
                    }),
                });
                let position = self.blocks.len() - 1;
                self.tools.insert(wire_index, position);
                position
            }
        };
        if let Some(id) = call.id.filter(|id| !id.is_empty()) {
            self.blocks[position].call_id = id;
        }
        if let Some(function) = call.function {
            if let Some(name) = function.name.filter(|name| !name.is_empty()) {
                self.blocks[position].name = name;
            }
            if let Some(arguments) = function.arguments.filter(|arguments| !arguments.is_empty()) {
                self.blocks[position].arguments.push_str(&arguments);
                self.append_tool_run(position, &arguments, now);
            }
        }
    }

    /// Extend the block's trailing text run, or start a new one.
    fn append_text_run(&mut self, position: usize, text: &str, now: u64) {
        let (index, kind, run) = {
            let block = &self.blocks[position];
            (block.index, block.kind, block.record)
        };
        if let Some(run) = run {
            // Only the record just written can be extended; a delta that
            // resumes after another block's record starts a new run.
            if run + 1 == self.records.len() {
                if let Some(time0) = text_run_time0(&self.records[run]) {
                    push_text_into(&mut self.records[run], text, now.saturating_sub(time0));
                    return;
                }
            }
        }
        let run = self.records.len();
        self.records.push(match kind {
            BlockKind::Reasoning => AssistantStreamRecord::ReasoningChunks {
                time0: now,
                index,
                dt: vec![0],
                texts: vec![text.to_string()],
            },
            _ => AssistantStreamRecord::TextChunks {
                time0: now,
                index,
                dt: vec![0],
                texts: vec![text.to_string()],
            },
        });
        self.blocks[position].record = Some(run);
    }

    /// Extend the block's trailing tool-argument run, or start a new one.
    fn append_tool_run(&mut self, position: usize, delta: &str, now: u64) {
        let (index, run, id, name) = {
            let block = &self.blocks[position];
            (
                block.index,
                block.record,
                block.call_id.clone(),
                block.name.clone(),
            )
        };
        if let Some(run) = run {
            if run + 1 == self.records.len() {
                if let AssistantStreamRecord::ToolCallChunks {
                    time0,
                    dt,
                    id: record_id,
                    name: record_name,
                    args,
                    ..
                } = &mut self.records[run]
                {
                    dt.push(now.saturating_sub(*time0));
                    args.push(delta.to_string());
                    // Identity can be announced on a later delta than the first
                    // argument fragment; keep the record's copy current.
                    if record_id.as_str().is_empty() && !id.is_empty() {
                        *record_id = ToolCallId::new(id);
                    }
                    if record_name.is_none() && !name.is_empty() {
                        *record_name = Some(name);
                    }
                    return;
                }
            }
        }
        let run = self.records.len();
        self.records.push(AssistantStreamRecord::ToolCallChunks {
            time0: now,
            index,
            dt: vec![0],
            id: ToolCallId::new(id),
            name: (!name.is_empty()).then_some(name),
            args: vec![delta.to_string()],
        });
        self.blocks[position].record = Some(run);
    }
}

/// Timestamp of a packed text or reasoning run, when `record` is one.
fn text_run_time0(record: &AssistantStreamRecord) -> Option<u64> {
    match record {
        AssistantStreamRecord::TextChunks { time0, .. }
        | AssistantStreamRecord::ReasoningChunks { time0, .. } => Some(*time0),
        _ => None,
    }
}

/// Append one delta to a packed text or reasoning run. Returns whether it did.
fn push_text_into(record: &mut AssistantStreamRecord, text: &str, dt: u64) -> bool {
    match record {
        AssistantStreamRecord::TextChunks {
            dt: deltas, texts, ..
        }
        | AssistantStreamRecord::ReasoningChunks {
            dt: deltas, texts, ..
        } => {
            deltas.push(dt);
            texts.push(text.to_string());
            true
        }
        _ => false,
    }
}

/// One completion tool call as a canonical block.
fn tool_call_block(
    call: WireToolCallResponse,
    position: usize,
) -> Result<ContentBlock, LlmFailure> {
    let WireToolCallResponseFunction { name, arguments } = call.function;
    let name = name
        .filter(|name| !name.is_empty())
        .ok_or_else(|| error::malformed("tool call carried no function name"))?;
    let id = call
        .id
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| format!("call_{position}"));
    Ok(ContentBlock::tool_call(
        ToolCallId::new(id),
        name,
        if arguments.is_empty() {
            "{}".to_string()
        } else {
            arguments
        },
    ))
}

/// Map a provider finish reason onto the canonical one.
///
/// Unknown reasons (a gateway's `content_filter`, a vendor extension) have no
/// canonical counterpart; the presence of tool calls is the only durable fact
/// they share, so that decides between `tool-calls` and `stop`. A reason the
/// provider *did* state is never overridden: a `stop` that happens to carry
/// tool calls stays `stop`, because upgrading it would invent an intent the
/// provider did not express.
fn map_finish_reason(raw: Option<&str>, has_tool_calls: bool) -> FinishReason {
    match raw {
        Some("stop" | "end_turn" | "stop_sequence") => FinishReason::Stop,
        Some("tool_calls" | "function_call" | "tool_use") => FinishReason::ToolCalls,
        Some("length" | "max_tokens") => FinishReason::MaxTokens,
        Some(_) | None if has_tool_calls => FinishReason::ToolCalls,
        Some(_) | None => FinishReason::Stop,
    }
}

/// Map wire token accounting onto the canonical, disjoint-count form.
fn token_usage(usage: &WireUsage) -> TokenUsage {
    let cached = usage
        .prompt_cache_hit_tokens
        .or_else(|| {
            usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
        })
        .unwrap_or(0);
    let prompt = if usage.prompt_tokens == 0 {
        // A gateway that reports only the cache buckets still yields a total.
        cached + usage.prompt_cache_miss_tokens.unwrap_or(0)
    } else {
        usage.prompt_tokens
    };
    TokenUsage {
        // `TokenUsage` counts are disjoint: `input_tokens` excludes cached
        // input, so the cache hit is subtracted rather than counted twice.
        input_tokens: prompt.saturating_sub(cached),
        output_tokens: usage.completion_tokens,
        total_tokens: Some(
            usage
                .total_tokens
                .unwrap_or(prompt + usage.completion_tokens),
        ),
        cache_read_tokens: (cached > 0).then_some(cached),
        // DeepSeek bills cache writes inside the prompt and reports no separate
        // bucket, so the canonical field stays absent rather than zero.
        cache_write_tokens: None,
        reasoning_tokens: usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
    }
}

/// Every system message's text, folded into one prompt.
fn system_text(messages: &[Message]) -> Result<String, LlmFailure> {
    let mut messages_text = Vec::new();
    for message in messages {
        let Message::System(system) = message else {
            continue;
        };
        let mut parts = Vec::new();
        for block in &system.content {
            match block.as_text() {
                Some("") => {}
                Some(text) => parts.push(text.to_string()),
                None => {
                    return Err(error::unsupported(format!(
                        "system content `{}`",
                        block.block_type()
                    )))
                }
            }
        }
        let text = parts.join("\n");
        if !text.is_empty() {
            messages_text.push(text);
        }
    }
    Ok(messages_text.join("\n\n"))
}

/// Text visible in a `developer` message, with its tool changes dropped.
fn developer_text(blocks: &[ContentBlock]) -> Result<String, LlmFailure> {
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Core(CoreContentBlock::Text { text }) => {
                if !text.is_empty() {
                    parts.push(text.clone());
                }
            }
            // `tools` already carries the complete list, so a dynamic
            // addition/removal needs no wire representation.
            ContentBlock::Core(
                CoreContentBlock::ToolAddition { .. } | CoreContentBlock::ToolRemoval { .. },
            ) => {}
            other => {
                return Err(error::unsupported(format!(
                    "developer content `{}`",
                    other.block_type()
                )))
            }
        }
    }
    Ok(parts.join("\n"))
}

/// Text visible in a `user` or `tool` message, with image and file references
/// flattened to a placeholder.
fn content_text(blocks: &[ContentBlock]) -> Result<String, LlmFailure> {
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Core(CoreContentBlock::Text { text }) => {
                if !text.is_empty() {
                    parts.push(text.clone());
                }
            }
            // Reasoning and tool calls are assistant facts; a user turn that
            // carries them has no faithful wire form, so they are skipped the
            // way the TypeScript adapter skips them.
            ContentBlock::Core(
                CoreContentBlock::Reasoning { .. } | CoreContentBlock::ToolCall { .. },
            ) => {}
            ContentBlock::Core(CoreContentBlock::Image { attachment, .. }) => {
                parts.push(image_placeholder(attachment));
            }
            ContentBlock::Core(CoreContentBlock::File { attachment }) => {
                parts.push(file_placeholder(attachment));
            }
            ContentBlock::Core(
                CoreContentBlock::ToolAddition { .. } | CoreContentBlock::ToolRemoval { .. },
            ) => {
                return Err(error::unsupported(
                    "tool-change block outside a developer message",
                ))
            }
            ContentBlock::Extension(block) => {
                return Err(error::unsupported(format!(
                    "user content `{}`",
                    block.kind()
                )))
            }
        }
    }
    Ok(parts.join("\n"))
}

/// Text of an assistant message, ignoring everything that is not visible text.
fn assistant_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(ContentBlock::as_text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// A placeholder stating plainly that the image bytes are absent.
fn image_placeholder(attachment: &ImageAttachmentRef) -> String {
    format!(
        "[image attachment \"{}\" ({}; {}x{}px, {} bytes) is not attached: this transport cannot send image bytes]",
        attachment.attachment_id,
        attachment.media_type,
        attachment.width,
        attachment.height,
        attachment.bytes,
    )
}

/// A placeholder stating plainly that the file bytes are absent.
fn file_placeholder(attachment: &FileAttachmentRef) -> String {
    format!(
        "[file attachment \"{}\" ({} bytes) is not attached: this transport cannot send file bytes]",
        attachment.name, attachment.bytes,
    )
}

/// The provider's accepted reasoning efforts, flattened onto the OpenAI field.
///
/// `off` disables the field entirely, `max` saturates at `high` (the surface's
/// ceiling), and an unrecognized effort is dropped rather than rejected: the
/// accepted set is server-dependent, and a rejected request is worse than a
/// request without the hint.
fn reasoning_effort(effort: Option<&str>) -> Option<String> {
    let effort = effort?;
    match effort {
        "off" => None,
        "low" | "medium" | "high" => Some(effort.to_string()),
        "max" => Some("high".to_string()),
        _ => None,
    }
}

/// A `function`-typed tool declaration.
fn wire_tool(tool: &ToolSchema) -> WireTool {
    WireTool {
        kind: "function".to_string(),
        function: crate::wire::WireFunction {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        },
    }
}

/// A message whose content is one string.
fn text_message(role: &str, text: String) -> WireMessage {
    WireMessage {
        role: role.to_string(),
        content: Some(Value::String(text)),
        tool_calls: None,
        tool_call_id: None,
    }
}

/// Push a user message, folding it into the previous one when that is also a
/// user message: string content cannot express separate parts, and some servers
/// reject two consecutive messages from the same role.
fn push_user_text(messages: &mut Vec<WireMessage>, text: String) {
    if let Some(previous) = messages.last_mut() {
        if previous.role == "user" && previous.tool_calls.is_none() {
            if let Some(Value::String(existing)) = previous.content.as_mut() {
                existing.push_str("\n\n");
                existing.push_str(&text);
                return;
            }
        }
    }
    messages.push(text_message("user", text));
}

/// Milliseconds since the Unix epoch, the unit `AssistantStreamRecord` uses.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsh_llm::{
        DeveloperMessage, FileAttachmentRef, LlmCallConfig, MessageSource, SystemMessage,
        ToolResultMessage, UserMessage,
    };
    use serde_json::json;

    fn call_config() -> LlmCallConfig {
        LlmCallConfig {
            provider: "deepseek".to_string(),
            model: "deepseek-chat".to_string(),
            reasoning_effort: None,
            temperature: None,
            max_tokens: None,
            stop: None,
        }
    }

    fn system(text: &str) -> Message {
        Message::System(SystemMessage::new(
            MessageId::new("sys"),
            vec![ContentBlock::text(text)],
        ))
    }

    fn developer(blocks: Vec<ContentBlock>) -> Message {
        Message::Developer(DeveloperMessage::new(
            MessageId::new("dev"),
            blocks,
            MessageSource::User,
        ))
    }

    fn user(text: &str) -> Message {
        Message::User(UserMessage::new(
            MessageId::new("user"),
            vec![ContentBlock::text(text)],
        ))
    }

    fn assistant(blocks: Vec<ContentBlock>) -> Message {
        Message::Assistant(AssistantMessage::new(
            MessageId::new("assistant"),
            blocks,
            "deepseek",
            "deepseek-chat",
        ))
    }

    fn tool_result(call_id: &str, text: &str) -> Message {
        Message::Tool(ToolResultMessage::new(
            MessageId::new("tool"),
            vec![ContentBlock::text(text)],
            ToolCallId::new(call_id),
            false,
        ))
    }

    fn completion(body: Value) -> WireChatCompletion {
        serde_json::from_value(body).expect("valid completion")
    }

    #[test]
    fn request_carries_every_role_and_its_sampling_fields() {
        let request = LlmRequest {
            config: LlmCallConfig {
                reasoning_effort: Some("high".to_string()),
                temperature: Some(0.2),
                max_tokens: Some(64),
                stop: Some(vec!["STOP".to_string()]),
                ..call_config()
            },
            messages: vec![
                system("be brief"),
                developer(vec![
                    ContentBlock::text("toolset updated"),
                    ContentBlock::tool_addition("lookup"),
                ]),
                user("hi"),
                assistant(vec![
                    ContentBlock::reasoning("thinking"),
                    ContentBlock::text("calling"),
                    ContentBlock::tool_call(ToolCallId::new("call_1"), "lookup", r#"{"key":"a"}"#),
                ]),
                tool_result("call_1", "42"),
            ],
            tools: vec![ToolSchema::new(
                "lookup",
                "Find a value",
                json!({"type": "object"}),
            )],
        };
        let wire = build_request(&request, true).expect("translated");
        assert_eq!(
            serde_json::to_value(&wire).expect("serializable"),
            json!({
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "be brief\n\ntoolset updated"},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "calling",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{\"key\":\"a\"}"},
                        }],
                    },
                    {"role": "tool", "content": "42", "tool_call_id": "call_1"},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Find a value",
                        "parameters": {"type": "object"},
                    },
                }],
                "stream": true,
                "stream_options": {"include_usage": true},
                "reasoning_effort": "high",
                "temperature": 0.2,
                "max_tokens": 64,
                "stop": ["STOP"],
            })
        );
    }

    #[test]
    fn request_omits_the_stream_options_only_when_not_streaming() {
        let request = LlmRequest {
            config: call_config(),
            messages: vec![user("hi")],
            tools: Vec::new(),
        };
        let wire = build_request(&request, false).expect("translated");
        let value = serde_json::to_value(&wire).expect("serializable");
        assert_eq!(
            value,
            json!({
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false,
            })
        );
        assert!(value.get("tools").is_none());
        assert!(value.get("temperature").is_none());
    }

    #[test]
    fn request_merges_adjacent_user_turns_and_skips_empty_content() {
        let request = LlmRequest {
            config: call_config(),
            messages: vec![
                user("first"),
                user("second"),
                assistant(vec![ContentBlock::text("")]),
                system(""),
            ],
            tools: Vec::new(),
        };
        let wire = build_request(&request, false).expect("translated");
        assert_eq!(
            serde_json::to_value(&wire).expect("serializable")["messages"],
            json!([{"role": "user", "content": "first\n\nsecond"}])
        );
    }

    #[test]
    fn request_rejects_an_unanswered_tool_call() {
        let request = LlmRequest {
            config: call_config(),
            messages: vec![
                user("go"),
                assistant(vec![ContentBlock::tool_call(
                    ToolCallId::new("call_1"),
                    "lookup",
                    "{}",
                )]),
            ],
            tools: Vec::new(),
        };
        // A history that ends on an unanswered call is refused, which also
        // proves the assistant message itself was kept.
        let failure = build_request(&request, false).expect_err("unanswered call");
        assert_eq!(failure.code, error::CODE_INVALID_REQUEST);
    }

    #[test]
    fn request_rejects_a_tool_result_with_no_call() {
        let request = LlmRequest {
            config: call_config(),
            messages: vec![user("go"), tool_result("call_missing", "42")],
            tools: Vec::new(),
        };
        let failure = build_request(&request, false).expect_err("orphan result");
        assert_eq!(failure.code, error::CODE_INVALID_REQUEST);
        assert!(failure.message.contains("call_missing"));
    }

    #[test]
    fn request_flattens_an_image_to_a_placeholder_naming_the_attachment() {
        let image = ImageAttachmentRef {
            attachment_id: "att_1".to_string(),
            media_type: "image/png".to_string(),
            bytes: 2048,
            width: 64,
            height: 32,
            name: None,
            original_dimensions: None,
        };
        let file = FileAttachmentRef {
            attachment_id: "att_2".to_string(),
            name: "notes.txt".to_string(),
            bytes: 12,
        };
        let request = LlmRequest {
            config: call_config(),
            messages: vec![Message::User(UserMessage::new(
                MessageId::new("user"),
                vec![
                    ContentBlock::Core(CoreContentBlock::Image {
                        attachment: image,
                        offloaded: false,
                    }),
                    ContentBlock::Core(CoreContentBlock::File { attachment: file }),
                ],
            ))],
            tools: Vec::new(),
        };
        let wire = build_request(&request, false).expect("translated");
        let content = serde_json::to_value(&wire).expect("serializable")["messages"][0]["content"]
            .as_str()
            .expect("content is a string")
            .to_string();
        assert!(content.contains("att_1"), "{content}");
        assert!(content.contains("64x32px"), "{content}");
        assert!(content.contains("notes.txt"), "{content}");
        assert!(content.contains("is not attached"), "{content}");
    }

    #[test]
    fn request_rejects_a_plugin_block_it_cannot_express() {
        let request = LlmRequest {
            config: call_config(),
            messages: vec![Message::User(UserMessage::new(
                MessageId::new("user"),
                vec![ContentBlock::extension("citation", serde_json::Map::new())],
            ))],
            tools: Vec::new(),
        };
        let failure = build_request(&request, false).expect_err("unsupported block");
        assert_eq!(failure.code, error::CODE_UNSUPPORTED);
    }

    #[test]
    fn reasoning_effort_saturates_and_drops_unknown_values() {
        assert_eq!(reasoning_effort(Some("off")), None);
        assert_eq!(reasoning_effort(Some("low")), Some("low".to_string()));
        assert_eq!(reasoning_effort(Some("max")), Some("high".to_string()));
        assert_eq!(reasoning_effort(Some("extreme")), None);
        assert_eq!(reasoning_effort(None), None);
    }

    #[test]
    fn completion_maps_reasoning_text_and_parallel_tool_calls() {
        let response = completion_response(
            completion(json!({
                "id": "chatcmpl-1",
                "model": "deepseek-reasoner",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "why",
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{\"x\":1}"}},
                            {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            })),
            "deepseek",
            "deepseek-reasoner",
        )
        .expect("translated");

        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        assert_eq!(
            response.message.content,
            vec![
                ContentBlock::reasoning("why"),
                ContentBlock::text("answer"),
                ContentBlock::tool_call(ToolCallId::new("call_1"), "a", "{\"x\":1}"),
                ContentBlock::tool_call(ToolCallId::new("call_2"), "b", "{}"),
            ]
        );
        assert_eq!(
            response.usage,
            Some(TokenUsage {
                input_tokens: 10,
                output_tokens: 4,
                total_tokens: Some(14),
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
            })
        );
        assert!(response.stream.is_empty());
        assert_eq!(
            response.message.source,
            MessageSource::Model {
                provider: "deepseek".to_string(),
                model: "deepseek-reasoner".to_string(),
                replay_state: None,
            }
        );
    }

    #[test]
    fn completion_accepts_empty_content_with_only_tool_calls() {
        let response = completion_response(
            completion(json!({
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": null, "tool_calls": [
                        {"id": "call_9", "type": "function", "function": {"name": "a", "arguments": ""}}
                    ]},
                    "finish_reason": "tool_calls",
                }],
            })),
            "deepseek",
            "deepseek-chat",
        )
        .expect("translated");
        assert_eq!(response.message.content.len(), 1);
        assert_eq!(
            response.message.content[0],
            // An absent argument string becomes `{}` so a downstream parse works.
            ContentBlock::tool_call(ToolCallId::new("call_9"), "a", "{}")
        );
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
    }

    #[test]
    fn completion_accepts_a_reasoning_only_response() {
        let response = completion_response(
            completion(json!({
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "reasoning_content": "hmm"},
                    "finish_reason": "length",
                }],
            })),
            "deepseek",
            "deepseek-reasoner",
        )
        .expect("translated");
        assert_eq!(
            response.message.content,
            vec![ContentBlock::reasoning("hmm")]
        );
        assert_eq!(response.finish_reason, FinishReason::MaxTokens);
        assert!(response.usage.is_none());
    }

    #[test]
    fn completion_with_no_content_at_all_is_an_empty_response() {
        let failure = completion_response(
            completion(json!({
                "choices": [{"index": 0, "message": {"role": "assistant"}, "finish_reason": "stop"}],
            })),
            "deepseek",
            "deepseek-chat",
        )
        .expect_err("empty response");
        assert_eq!(failure.code, dsh_llm::EMPTY_RESPONSE_CODE);
    }

    #[test]
    fn completion_with_no_choice_at_index_zero_is_malformed() {
        let failure = completion_response(
            completion(json!({"choices": []})),
            "deepseek",
            "deepseek-chat",
        )
        .expect_err("no choice");
        assert_eq!(failure.code, error::CODE_MALFORMED);
    }

    #[test]
    fn usage_splits_cached_input_and_keeps_the_reasoning_bucket() {
        let response = completion_response(
            completion(json!({
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_cache_hit_tokens": 40,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                },
            })),
            "deepseek",
            "deepseek-chat",
        )
        .expect("translated");
        let usage = response.usage.expect("usage");
        assert_eq!(usage.input_tokens, 60);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.cache_read_tokens, Some(40));
        assert_eq!(usage.reasoning_tokens, Some(7));
        // Total is recomputed from the wire prompt count when absent.
        assert_eq!(usage.total_tokens, Some(120));
    }

    #[test]
    fn finish_reasons_map_and_unknown_ones_fall_back_to_the_tool_fact() {
        assert_eq!(map_finish_reason(Some("stop"), false), FinishReason::Stop);
        assert_eq!(
            map_finish_reason(Some("length"), false),
            FinishReason::MaxTokens
        );
        assert_eq!(
            map_finish_reason(Some("tool_calls"), false),
            FinishReason::ToolCalls
        );
        assert_eq!(
            map_finish_reason(Some("content_filter"), true),
            FinishReason::ToolCalls
        );
        assert_eq!(
            map_finish_reason(Some("content_filter"), false),
            FinishReason::Stop
        );
        assert_eq!(map_finish_reason(None, false), FinishReason::Stop);
    }

    /// Feed a scripted stream and return the settled response.
    fn assemble(frames: &[&str]) -> Result<LlmResponse, LlmFailure> {
        let mut assembler = StreamAssembler::new("deepseek", "deepseek-reasoner");
        for frame in frames {
            assembler.push_frame(frame)?;
        }
        assembler.finish()
    }

    #[test]
    fn stream_accumulates_reasoning_text_usage_and_parallel_tool_calls() {
        let response = assemble(&[
            r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"reasoning_content":"think "}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"reasoning_content":"more"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"hel"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"lo"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"a","arguments":"{\"x\":"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"b","arguments":"{\"y\":"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":9,"total_tokens":21}}"#,
            "[DONE]",
        ])
        .expect("settled");

        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        assert_eq!(
            response.message.content,
            vec![
                ContentBlock::reasoning("think more"),
                ContentBlock::text("hello"),
                ContentBlock::tool_call(ToolCallId::new("call_1"), "a", "{\"x\":1}"),
                ContentBlock::tool_call(ToolCallId::new("call_2"), "b", "{\"y\":2}"),
            ]
        );
        assert_eq!(
            response.usage,
            Some(TokenUsage {
                input_tokens: 12,
                output_tokens: 9,
                total_tokens: Some(21),
                ..TokenUsage::default()
            })
        );

        // Block order is the order the blocks first appeared.
        let starts: Vec<(u64, String)> = response
            .stream
            .iter()
            .filter_map(|record| match record {
                AssistantStreamRecord::Chunk { chunk, .. } => match chunk.as_ref() {
                    StreamChunk::BlockStart { index, block_type } => {
                        Some((*index, block_type.clone()))
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(
            starts,
            vec![
                (0, "reasoning".to_string()),
                (1, "text".to_string()),
                (2, "tool-call".to_string()),
                (3, "tool-call".to_string()),
            ]
        );

        // Argument fragments keep their boundaries: interleaved parallel calls
        // open a new run each time they resume, so no two calls are merged.
        let runs: Vec<(String, Vec<String>)> = response
            .stream
            .iter()
            .filter_map(|record| match record {
                AssistantStreamRecord::ToolCallChunks { id, args, .. } => {
                    Some((id.as_str().to_string(), args.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            runs,
            vec![
                ("call_1".to_string(), vec!["{\"x\":".to_string()]),
                ("call_2".to_string(), vec!["{\"y\":".to_string()]),
                ("call_1".to_string(), vec!["1}".to_string()]),
                ("call_2".to_string(), vec!["2}".to_string()]),
            ]
        );

        // Usage precedes the single terminal finish.
        assert!(matches!(
            response.stream.last(),
            Some(AssistantStreamRecord::Chunk { chunk, .. })
                if matches!(chunk.as_ref(), StreamChunk::Finish { .. })
        ));
        let usage_at = response
            .stream
            .iter()
            .position(|record| matches!(record, AssistantStreamRecord::Chunk { chunk, .. } if matches!(chunk.as_ref(), StreamChunk::Usage { .. })))
            .expect("usage record");
        assert_eq!(usage_at, response.stream.len() - 2);
    }

    #[test]
    fn stream_accepts_a_finish_reason_without_the_done_sentinel() {
        let response = assemble(&[
            r#"{"choices":[{"index":0,"delta":{"content":"done"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ])
        .expect("settled");
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.message.content, vec![ContentBlock::text("done")]);
    }

    #[test]
    fn stream_without_a_finish_or_done_is_closed() {
        let failure = assemble(&[r#"{"choices":[{"index":0,"delta":{"content":"trunc"}}]}"#])
            .expect_err("truncated stream");
        assert_eq!(failure.code, error::CODE_STREAM_CLOSED);
    }

    #[test]
    fn stream_with_no_content_is_an_empty_response() {
        let failure = assemble(&["[DONE]"]).expect_err("empty stream");
        assert_eq!(failure.code, dsh_llm::EMPTY_RESPONSE_CODE);
    }

    #[test]
    fn stream_rejects_a_malformed_frame() {
        let failure = assemble(&["{not json"]).expect_err("malformed frame");
        assert_eq!(failure.code, error::CODE_MALFORMED);
    }

    #[test]
    fn stream_reports_an_in_band_error_frame() {
        let failure = assemble(&[
            r#"{"error":{"message":"Insufficient balance","type":"insufficient_quota"}}"#,
        ])
        .expect_err("in-band error");
        assert_eq!(failure.code, error::CODE_QUOTA);
        assert_eq!(failure.message, "Insufficient balance");
    }

    #[test]
    fn stream_ignores_choices_other_than_index_zero() {
        let response = assemble(&[
            r#"{"choices":[{"index":1,"delta":{"content":"wrong"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"right"}}]}"#,
            "[DONE]",
        ])
        .expect("settled");
        assert_eq!(response.message.content, vec![ContentBlock::text("right")]);
    }
}
