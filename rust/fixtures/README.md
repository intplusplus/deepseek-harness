# v4 session fixtures

Golden data for comparing the Rust session wire format with the shipped
TypeScript v4 JSONL format. The fixtures are generated from the TypeScript
sources at commit `687f282e8625c6b8d11ff58211a90524be2e4317` (2026-10-17).
Nothing here is generated during a test run; `rust/target/` stays ignored.

## Provenance

The generator imports the shipped codec directly:

- `packages/session/session-format-v3-to-v4/src/codec.ts` — the released v4
  codec; it delegates to the v2 physical codec and stamps `version: 4`.
- `packages/session/session-format-v1-to-v2/src/codec.ts` — header and row
  validation, canonical header order, and the v4 row envelope.
- `packages/session/session-format-v1-to-v2/src/validation.ts` — current-mode
  admission of unknown event types.
- `packages/session/session-format-v3-to-v4/src/tool-role.ts` — v4
  first-class tool-role messages.
- `packages/core/session/src/types.ts` and `packages/core/session/src/index.ts`
  — event payload types and the `Session.append` row key order.
- `packages/llm/llm/src/message.ts`, `packages/llm/llm/src/types.ts`, and
  `packages/llm/llm-retry/src/types.ts` — message, stream, and retry payloads.

`generate.mjs` runs on Node 22 without `pnpm install`:
`node --experimental-transform-types` loads the TypeScript sources, and
`resolve-hook.mjs` maps workspace package names to `src`. `v1-to-v2-codec.ts`
is a small shim that avoids loading the decorator-bearing `dsh-llm` package.

### Evidence levels

1. **Physical rows (`v4-physical/<scenario>.v4.jsonl`).** The header is written
   by `releasedV4SessionFormatCodec.encodeHeader`; every event row is written
   by `encodeEvent`. Each row is then decoded again with the shipped strict
   decoder, and the generator asserts that every decoded event has the same
   `type` and JSON-equal `data` as the source row. This is encoder-level
   evidence, not an end-to-end `Session.append` run: the generator builds the
   envelope itself, uses `createdAt + seq` for `time`, and does not exercise
   `Session.append` runtime validation.
2. **Payload projections (`v4-event-data/<scenario>.jsonl`).** These are
   derived, not TypeScript artifacts. Each line is `{"type": ..., ...data}`
   with the v4 envelope (`seq`, `time`, `data`, `ignorable`,
   `sourceEventSeqs`, `surfaceOp`) removed, because that is the flat shape Rust
   models. Payload field order is the producer's insertion order, not a
   schema-defined order; the comparison is against this recorded producer.
3. **Corroboration only.** The shipped artifact
   `packages/experimental/webworker-runtime/tests/fixtures/vfs-example/home/sessions/--dsh-workspace--/preview-showcase/session.v4.jsonl`
   confirms the v4 header fields and the row envelope on a real file. It is not
   a fixture source, and its row key order differs from `Session.append`
   (`type,data,seq,time` versus `type,seq,time,data`), so row key order is
   producer-specific rather than canonical.

No fixture comes from a live model or tool run, and no API key is required.

## Scenarios

| Fixture | Events | Physical lines | Covers |
| --- | ---: | ---: | --- |
| `turn-tool-call` | 12 | 13 | Full turn, assistant tool call, successful tool result, two steps, usage, timed stream records |
| `retry` | 10 | 11 | Failed attempt, `llm/retry`, `llm/retry-started`, recovery |
| `tool-error` | 7 | 8 | Failed tool result with `isError: true` and `error.reason` |
| `extension-unicode` | 10 | 11 | Compaction, unknown `plugin:fixture/acme`, Chinese/emoji text, `1e+21`, `1e-7`, max safe integer, escapes, image with `offloaded: true` |

The physical header uses the canonical order from
`session-format-v1-to-v2/src/codec.ts:164-175`:
`type`, `version`, `id`, `createdAt`, optional `cwd`/`parentSession`,
`isSeeded`, optional `origin`, `delegationDepth`, optional `agentPreset`.
`type` and `delegationDepth` are required by TypeScript
(`codec.ts:19`, `codec.ts:45`); `SESSION_FORMAT_VERSION` is `4` in both
implementations.

Rows use `type`, `seq`, `time`, `data` plus optional `ignorable`,
`sourceEventSeqs`, and `surfaceOp` (`codec.ts:21-22`). `seq` is dense from
zero and `time` is a safe integer. An unknown plugin event carries
`ignorable: true` because current-mode admission requires it when the type is
not in `KNOWN_SESSION_EVENT_TYPES` (`validation.ts:69-76`).

`offloaded` and `interrupted` are `true`-only markers in TypeScript
(`llm/src/types.ts:88`, `core/session/src/types.ts:349`). The fixtures contain
`offloaded: true`; Rust emits neither field when false, and a fixture test
removes the marker and checks that re-encoding does not add it back.

## Regeneration

```sh
node --experimental-transform-types --no-warnings rust/fixtures/generate.mjs
```

The generator rewrites all four scenarios and prints one line per scenario.
It is deterministic; running it again produces identical files.

## Comparison

```sh
cargo test -p dsh-session -p dsh-retry --target-dir target/agent-fixtures
cargo clippy -p dsh-session -p dsh-retry --all-targets --target-dir target/agent-fixtures -- -D warnings
node --experimental-transform-types --no-warnings rust/fixtures/compare-rust.mjs
```

`dsh-session/tests/fixtures.rs` pins every projection row as an exact match, a
documented byte difference, or a rejection. `compare-rust.mjs` runs that test
with `DSH_FIXTURE_REPORT`, wraps each Rust flat row back into a physical v4 row
at `seq: 0`, and feeds it through the shipped v4 decoder.

Expected summary:

```text
bytes:     28 match, 6 documented differences, 0 undocumented
ts-decode: 32 pass, 2 fail, 5 skipped
result:    PASS (every difference is documented; every decodable row is accepted)
```

The two decoder failures are the same expected gap: Rust omits the message
`role`, so the v4 decoder rejects both `tool/result` rows. They are documented
differences, not undocumented failures.

## Divergences

| Area | TypeScript evidence | Rust behavior | Status |
| --- | --- | --- | --- |
| Row envelope | `v1-to-v2/src/codec.ts:21-23,132-153`; `core/session/src/index.ts:817-823` | `jsonl::decode` reads a flat event per line; physical rows fail with `missing field \`turn\`` | Open, structural |
| Header discriminator and depth | `v1-to-v2/src/codec.ts:19,45,50,164-175`; `v3-to-v4/src/codec.ts:42-44` | `SessionHeader` has no `type` and `delegation_depth` is optional | Open, structural |
| `user/message` payload | `core/session/src/types.ts:310` | Rust nests the payload under `message` (`event.rs:73-75`) | Open, structural |
| Message `role` | `llm/src/message.ts:162-179` | Rust message structs derive the role from the event tag and omit it | Open |
| `isError: false` | `llm/src/message.ts:179` | Serialized only when true (`message.rs:280-282`) | Open |
| Integer `f64` | JavaScript `JSON.stringify(1)` is `1` | `serde_json` writes `1.0` (for example `temperature`) | Open, documented |
| Unknown event admission | `v1-to-v2/src/validation.ts:69-76` requires `ignorable: true` or an installed type | Rust accepts any unknown type as an extension without an envelope marker | Open |
| `surfaceOp` / `sourceEventSeqs` | `v1-to-v2/src/codec.ts:22` | Not modeled; dropped by the projection | Open, envelope |

Confirmed matching behavior:

- `1e21` is `1e+21` in both `JSON.stringify` and `serde_json`; `1e-7`,
  `9007199254740991`, `0.1`, and `1.5` also match byte for byte.
- `JSON.stringify(-0)` is `0`; the fixture records `0`. TypeScript rejects
  negative zero for safe-integer fields (`session-format/src/json.ts:18-24`),
  so negative zero is not observable on this wire.
- `tool/result.error` keeps `name`, `code`, and optional `reason` outside the
  message (`core/session/src/types.ts:376-385`); `tool-error` matches after the
  documented `role` difference.
- Both retry events match exactly, including `delayMs: 1.5` and the policy-key
  string.
- `StreamChunk` and `CoreContentBlock` now use camelCase fields
  (`blockType`, `argumentsDelta`, `toolName`) as TypeScript does
  (`llm/src/types.ts:471-474`); this was the only source fix needed for the
  fixture comparison.
