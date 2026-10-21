/**
 * Regenerate the v4 golden fixtures by running the shipped TypeScript
 * encoder and decoder (no install required).
 *
 *   node --experimental-transform-types rust/fixtures/generate.mjs
 *
 * Writes:
 *   v4-physical/<name>.v4.jsonl   real v4 physical rows (header + events)
 *   v4-event-data/<name>.jsonl    {"type": ..., ...data} projection of every row
 *
 * The projection is not a TypeScript artifact: it removes the v4 envelope
 * (`seq`, `time`, `data`, surface metadata) so the payload bytes Rust models
 * can be compared directly. Every physical row is re-decoded by the shipped
 * decoder before anything is written.
 */
import { register } from 'node:module'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

register('./resolve-hook.mjs', import.meta.url)
const { releasedV4SessionFormatCodec } = await import('@deepseek-ai/dsh-session-format-v3-to-v4')

const HERE = dirname(fileURLToPath(import.meta.url))
const PHYSICAL = resolve(HERE, 'v4-physical')
const EVENT_DATA = resolve(HERE, 'v4-event-data')

const text = value => ({ type: 'text', text: value })
const finish = (time, reason) => ({ type: 'chunk', time, chunk: { type: 'finish', reason } })
const userMessage = (id, body) => ({ id, role: 'user', content: [text(body)], source: { kind: 'user' } })

const turnWithToolCall = {
  header: {
    version: 4,
    id: 'fixture-turn-tool-call',
    createdAt: 1760000000000,
    cwd: '/workspace/dsh-fixtures',
    isSeeded: false,
    delegationDepth: 0,
  },
  rows: [
    { type: 'turn/start', data: { turn: 1 } },
    {
      type: 'user/message',
      data: userMessage('user-1', 'Read README.md and summarize its architecture section.'),
      surfaceOp: 'append',
    },
    { type: 'step/start', data: { turn: 1, step: 1 } },
    {
      type: 'request/header',
      data: {
        header: {
          config: {
            provider: 'deepseek',
            model: 'deepseek-chat',
            reasoningEffort: 'medium',
            temperature: 1,
            maxTokens: 8192,
            stop: ['\n\n'],
          },
          adapterDefaults: { reasoningEffort: true },
          tools: [{
            name: 'fs_read',
            description: 'Read a UTF-8 file from the workspace.',
            parameters: {
              type: 'object',
              properties: { path: { type: 'string', description: 'Workspace-relative path.' } },
              required: ['path'],
              additionalProperties: false,
            },
          }],
        },
        reason: 'initial',
      },
    },
    {
      type: 'assistant/message',
      data: {
        turn: 1,
        step: 1,
        message: {
          id: 'assistant-1',
          role: 'assistant',
          content: [{ type: 'tool-call', id: 'call_1', name: 'fs_read', arguments: '{"path":"README.md"}' }],
          source: { kind: 'model', provider: 'deepseek', model: 'deepseek-chat' },
        },
        stream: [
          { type: 'chunk', time: 0, chunk: { type: 'block-start', index: 0, blockType: 'tool-call' } },
          { type: 'chunk', time: 1, chunk: { type: 'tool-call-delta', index: 0, id: 'call_1', name: 'fs_read', argumentsDelta: '{"path":"README.md"}' } },
          { type: 'chunk', time: 2, chunk: { type: 'block-end', index: 0, block: { type: 'tool-call', id: 'call_1', name: 'fs_read', arguments: '{"path":"README.md"}' } } },
          finish(3, { kind: 'tool-calls' }),
        ],
        usage: { inputTokens: 42, outputTokens: 17, totalTokens: 59 },
      },
      surfaceOp: 'append',
    },
    { type: 'tool/call', data: { turn: 1, step: 1, callId: 'call_1', name: 'fs_read', arguments: '{"path":"README.md"}' } },
    {
      type: 'tool/result',
      data: {
        turn: 1,
        step: 1,
        message: {
          id: 'tool-1',
          role: 'tool',
          content: [text('# dsh — Rust foundation\n\n§3.12 线格式保真: `preserve_order` 已打开。')],
          source: { kind: 'tool', callId: 'call_1' },
          toolCallId: 'call_1',
          isError: false,
        },
        meta: { path: 'README.md', bytes: 2048, truncated: false },
      },
      surfaceOp: 'append',
    },
    { type: 'step/end', data: { turn: 1, step: 1 } },
    { type: 'step/start', data: { turn: 1, step: 2 } },
    {
      type: 'assistant/message',
      data: {
        turn: 1,
        step: 2,
        message: {
          id: 'assistant-2',
          role: 'assistant',
          content: [text('The README describes the Rust foundation and its wire-format decisions.')],
          source: { kind: 'model', provider: 'deepseek', model: 'deepseek-chat' },
        },
        stream: [
          { type: 'text-chunks', time0: 4, index: 0, dt: [1, 1], texts: ['The README describes ', 'the Rust foundation and its wire-format decisions.'] },
          finish(6, { kind: 'stop' }),
        ],
        usage: { inputTokens: 96, outputTokens: 12, totalTokens: 108, cacheReadTokens: 64 },
      },
      surfaceOp: 'append',
    },
    { type: 'step/end', data: { turn: 1, step: 2 } },
    { type: 'turn/end', data: { turn: 1, reason: { kind: 'completed' } } },
  ],
}

const retryFailure = {
  message: 'provider rate limited',
  code: 'RATE_LIMIT',
  status: 429,
  providerRetryAfterMs: 2000,
}

const retry = {
  header: {
    version: 4,
    id: 'fixture-retry',
    createdAt: 1760000001000,
    cwd: '/workspace/dsh-fixtures',
    isSeeded: false,
    delegationDepth: 0,
    agentPreset: 'standard',
  },
  rows: [
    { type: 'turn/start', data: { turn: 1 } },
    { type: 'user/message', data: userMessage('user-2', 'Retry the failing request.'), surfaceOp: 'append' },
    { type: 'step/start', data: { turn: 1, step: 1 } },
    {
      type: 'request/header',
      data: {
        header: { config: { provider: 'mock', model: 'mock' } },
        reason: 'initial',
      },
    },
    {
      type: 'assistant/attempt',
      data: {
        turn: 1,
        step: 1,
        stream: [finish(0, { kind: 'error', failure: retryFailure })],
      },
    },
    {
      type: 'llm/retry',
      data: {
        retryId: 'retry-1',
        turn: 1,
        step: 1,
        provider: 'mock',
        mode: 'normal',
        policyKey: '["normal",2,["RATE_LIMIT","SERVER"],1.5,4,0]',
        retry: 1,
        maxRetries: 2,
        delayMs: 1.5,
        failure: retryFailure,
      },
    },
    { type: 'llm/retry-started', data: { retryId: 'retry-1', turn: 1, step: 1, retry: 1 } },
    {
      type: 'assistant/message',
      data: {
        turn: 1,
        step: 1,
        message: {
          id: 'assistant-3',
          role: 'assistant',
          content: [text('Recovered after one retry.')],
          source: { kind: 'model', provider: 'mock', model: 'mock' },
        },
        stream: [finish(2, { kind: 'stop' })],
        usage: { inputTokens: 10, outputTokens: 4 },
      },
      surfaceOp: 'append',
    },
    { type: 'step/end', data: { turn: 1, step: 1 } },
    { type: 'turn/end', data: { turn: 1, reason: { kind: 'completed' } } },
  ],
}

const toolError = {
  header: {
    version: 4,
    id: 'fixture-tool-error',
    createdAt: 1760000003000,
    cwd: '/workspace/dsh-fixtures',
    isSeeded: false,
    delegationDepth: 0,
  },
  rows: [
    { type: 'turn/start', data: { turn: 1 } },
    { type: 'user/message', data: userMessage('user-5', 'Open the missing file.'), surfaceOp: 'append' },
    { type: 'step/start', data: { turn: 1, step: 1 } },
    { type: 'tool/call', data: { turn: 1, step: 1, callId: 'call_2', name: 'fs_read', arguments: '{"path":"MISSING.md"}' } },
    {
      type: 'tool/result',
      data: {
        turn: 1,
        step: 1,
        message: {
          id: 'tool-2',
          role: 'tool',
          content: [text('ENOENT: no such file or directory, open MISSING.md')],
          source: { kind: 'tool', callId: 'call_2' },
          toolCallId: 'call_2',
          isError: true,
        },
        error: { name: 'ToolError', code: 'ENOENT', reason: 'MISSING.md does not exist' },
        meta: { path: 'MISSING.md', bytes: 0, truncated: false },
      },
      surfaceOp: 'append',
    },
    { type: 'step/end', data: { turn: 1, step: 1 } },
    { type: 'turn/end', data: { turn: 1, reason: { kind: 'completed' } } },
  ],
}

const unicode = '中文 + emoji 🧪 + "quoted" + backslash \\ + newline\n + tab\t + control\u0001'

const extensionAndUnicode = {
  header: {
    version: 4,
    id: 'fixture-extension-unicode',
    createdAt: 1760000002000,
    cwd: '/workspace/dsh-fixtures',
    isSeeded: false,
    delegationDepth: 1,
  },
  rows: [
    { type: 'turn/start', data: { turn: 1 } },
    { type: 'user/message', data: userMessage('user-3', unicode), surfaceOp: 'append' },
    { type: 'step/start', data: { turn: 1, step: 1 } },
    { type: 'compaction/start', data: { compactionId: 'compaction-1', turn: 1 } },
    {
      type: 'compaction/summary',
      data: {
        compactionId: 'compaction-1',
        summary: [text('折叠摘要 ✅ 中文保留')],
        shadowedRange: { start: 0, end: 2 },
        shadowedSeqs: [0, 1, 2],
        shadowedTokenCount: 7,
        provider: 'deepseek',
        model: 'deepseek-chat',
      },
    },
    { type: 'compaction/end', data: { compactionId: 'compaction-1', turn: 1 } },
    {
      type: 'plugin:fixture/acme',
      // A downstream plugin event is outside KNOWN_SESSION_EVENT_TYPES; the
      // reader admits it only with the envelope's ignorable marker.
      ignorable: true,
      data: {
        label: '插件扩展 · π≈3.14159',
        ratio: 0.1,
        huge: 1e21,
        tiny: 1e-7,
        negativeZero: -0,
        maxSafe: 9007199254740991,
        half: 1.5,
        escaped: 'line\nbreak\t"quoted"\\backslash',
      },
    },
    {
      type: 'user/message',
      data: {
        id: 'user-4',
        role: 'user',
        content: [
          text('Skill catalog renewed for 工作区.'),
          {
            type: 'image',
            attachment: {
              attachmentId: 'sha256:0123456789abcdef',
              mediaType: 'image/png',
              bytes: 1024,
              width: 32,
              height: 16,
              name: '预览.png',
            },
            offloaded: true,
          },
        ],
        source: { kind: 'plugin:@deepseek-ai/dsh-skill', form: 'catalog', revision: 3 },
      },
      surfaceOp: 'append',
    },
    { type: 'step/end', data: { turn: 1, step: 1 } },
    { type: 'turn/end', data: { turn: 1, reason: { kind: 'completed' } } },
  ],
}

const scenarios = [
  { name: 'turn-tool-call', ...turnWithToolCall },
  { name: 'retry', ...retry },
  { name: 'tool-error', ...toolError },
  { name: 'extension-unicode', ...extensionAndUnicode },
]

mkdirSync(PHYSICAL, { recursive: true })
mkdirSync(EVENT_DATA, { recursive: true })

for (const scenario of scenarios) {
  const header = releasedV4SessionFormatCodec.encodeHeader(scenario.header, 0)
  const physicalHeader = JSON.stringify(header)
  if (physicalHeader !== JSON.stringify({ type: 'session', ...scenario.header })) {
    throw new Error(`${scenario.name}: encoded header differs from the declared row`)
  }

  const physical = [physicalHeader]
  const eventData = []
  for (const [seq, row] of scenario.rows.entries()) {
    // Same key order Session.append builds: type, seq, time, data, then the
    // optional envelope markers (ignorable, sourceEventSeqs, surfaceOp).
    // `encodeEvent` is identity.
    const physicalRow = { type: row.type, seq, time: scenario.header.createdAt + seq, data: row.data }
    if (row.ignorable !== undefined) physicalRow.ignorable = row.ignorable
    if (row.sourceEventSeqs !== undefined) physicalRow.sourceEventSeqs = row.sourceEventSeqs
    if (row.surfaceOp !== undefined) physicalRow.surfaceOp = row.surfaceOp
    const encoded = releasedV4SessionFormatCodec.encodeEvent(physicalRow)
    const line = JSON.stringify(encoded)
    physical.push(line)
    eventData.push(JSON.stringify({ type: row.type, ...row.data }))
  }

  // Round-trip through the shipped decoder: strict registry, no recovery.
  const decoder = releasedV4SessionFormatCodec.createDecoder(header, 'strict')
  const decoded = []
  const context = {
    emitEvent: event => decoded.push(event),
    emitRun: run => decoded.push(...run.expand()),
  }
  for (const line of physical.slice(1)) decoder.decodeRow(JSON.parse(line), context)
  const inherited = decoder.finish(context)
  if (decoded.length !== scenario.rows.length) {
    throw new Error(`${scenario.name}: decoder emitted ${decoded.length} of ${scenario.rows.length} events`)
  }
  for (const [index, event] of decoded.entries()) {
    if (event.type !== scenario.rows[index].type) {
      throw new Error(`${scenario.name}: row ${index} decoded as ${event.type}`)
    }
    if (JSON.stringify(event.data) !== JSON.stringify(scenario.rows[index].data)) {
      throw new Error(`${scenario.name}: row ${index} data changed on decode`)
    }
  }
  if (inherited !== 0) throw new Error(`${scenario.name}: unexpected inherited cut ${inherited}`)

  writeFileSync(resolve(PHYSICAL, `${scenario.name}.v4.jsonl`), `${physical.join('\n')}\n`)
  writeFileSync(resolve(EVENT_DATA, `${scenario.name}.jsonl`), `${eventData.join('\n')}\n`)
  console.log(`${scenario.name}: ${scenario.rows.length} events, ${physical.length} lines`)
}
