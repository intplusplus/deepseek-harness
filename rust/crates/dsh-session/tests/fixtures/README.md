# Session format fixtures

## `released-v0-real-shapes.jsonl`

Verbatim copy of
`origin/packages/session/session-persistence-jsonl/tests/fixtures/released-v0-real-shapes.jsonl`
— the only real released-v0 physical log in the repository.

It is **refused on purpose**, and the test suite asserts that refusal
(`tests/migration_chain.rs`):

```
format v2 surface before first step cannot acquire a system head without changing chronology
```

The TypeScript reader rejects it for exactly the same reason
(`session-persistence-jsonl/tests/jsonl.spec.ts`:
*"refuses the frozen pre-step V0 fixture on read/write open without publishing a
successor"*). The log opens a surface (`user/message` rows) before any
`step/start`, so promoting the streamed system prompt into a `system/message`
would have to rewrite chronology. This is an `UnsupportedMigrationError` in both
implementations, not a gap in the chain: a session whose stored shape cannot be
migrated is refused rather than silently reordered.

Keep this file byte-identical to the upstream fixture. If a future change makes
the chain accept it, the change is a bug: it means the chain started inventing
chronology.

## `v0-synthetic-normalized.jsonl` / `.v4.jsonl`

`v0-synthetic-normalized.jsonl` is the same scenario with the one structural
edit that makes it migratable: `step/start` moves ahead of the four queued
`user/message` rows, and the two compaction `sourceEventSeqs` / `shadowedSeqs`
/ `shadowedRange` references shift by one to match. Nothing else changes.

It exists to pin the **successful** edge of the chain on a realistic v0 payload
(queued user messages, an `llm/retry` pair, streamed assistant chunks that must
be folded into `assistant/attempt`, a completed compaction bracket, a title
request, and a trailing late user message).

`v0-synthetic-normalized.v4.jsonl` is the chain's own output, captured as a
regression lock. It is **implementation-produced evidence** — it proves the
output stopped changing, not that the output is correct. Correctness is asserted
separately and structurally in `tests/migration_chain.rs` (header normalization,
`type`-first flat rows, dense seq, system-head promotion, `role` stripping,
`code`→`ptc` renames, retry field order, stream folding).

Regenerate with:

```powershell
$env:CARGO_TARGET_DIR = "$env:USERPROFILE\.cache\dsh-target"
$env:DSH_UPDATE_GOLDEN = "1"
cargo test -p dsh-session --test migration_chain
```

Regenerating blindly defeats the purpose. The golden caught two real drifts
since it was first written — `system/message.message` had `source` before
`content`, and `assistant/message` had `usage` before `stream`. Both were fixed
to follow the TypeScript declaration order
(`packages/llm/llm/src/message.ts`, `packages/core/session/src/types.ts`), and
`tests/migration_chain.rs` now asserts those key sequences directly, so the
ordering claims no longer rest on the golden alone.
