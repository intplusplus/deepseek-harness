/**
 * Feed Rust's rendering of every fixture row back through the shipped v4
 * decoder and compare bytes with the TypeScript row.
 *
 *   node --experimental-transform-types rust/fixtures/compare-rust.mjs
 *
 * The script runs `cargo test -p dsh-session --test fixtures` with
 * `DSH_FIXTURE_REPORT` set, so the Rust bytes always come from the current
 * source. Each Rust flat row is wrapped into a physical v4 row
 * (`type`, `seq`, `time`, `data`) and decoded in isolation at `seq: 0`; the
 * wrapper drops `surfaceOp` and `sourceEventSeqs`, which Rust does not model.
 *
 * Exit status is non-zero when the Rust test fails, when a byte difference is
 * not explained by a documented gap, or when the decoder rejects a row whose
 * bytes matched the TypeScript row.
 */
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync, rmSync } from 'node:fs'
import { register } from 'node:module'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'

const USERPROFILE = homedir()

register('./resolve-hook.mjs', import.meta.url)
const { releasedV4SessionFormatCodec } = await import('@deepseek-ai/dsh-session-format-v3-to-v4')

const HERE = dirname(fileURLToPath(import.meta.url))
const RUST = resolve(HERE, '..')
// Build products must live outside the repository: the host backs up every
// modified file, and a project-local `target/` turns each `.o`/`.rlib`/`.pdb`
// into a backup copy. See rust/README.md §9.4.
const TARGET = process.env.CARGO_TARGET_DIR
  ? resolve(process.env.CARGO_TARGET_DIR, 'agent-fixtures')
  : resolve(USERPROFILE, '.cache', 'dsh-target', 'agent-fixtures')
const REPORT = resolve(TARGET, 'fixtures-report.json')
const SCENARIOS = ['turn-tool-call', 'retry', 'tool-error', 'extension-unicode']

/** Byte differences the Rust test also pins, applied in this order. */
const GAPS = [
  {
    name: 'message-role',
    apply: line => line.replace(/"role":"(?:assistant|tool)",/, ''),
  },
  {
    name: 'isError-false',
    apply: line => line.replace(/,"isError":false/, ''),
  },
  {
    name: 'integer-f64',
    apply: line => line.replace(/"temperature":1,/, '"temperature":1.0,'),
  },
]

function readLines(path) {
  return readFileSync(path, 'utf8').split('\n').filter(line => line.length > 0)
}

function classify(tsLine, rustLine) {
  if (rustLine === tsLine) return { kind: 'MATCH' }
  let expected = tsLine
  const applied = []
  for (const gap of GAPS) {
    const next = gap.apply(expected)
    if (next !== expected) {
      expected = next
      applied.push(gap.name)
    }
  }
  if (expected === rustLine) return { kind: 'DIFF', gaps: applied }
  return { kind: 'UNDOCUMENTED' }
}

/** Decode one Rust flat row as the single row of a fresh v4 artifact. */
function decodeRustRow(header, type, data) {
  const decoder = releasedV4SessionFormatCodec.createDecoder(header, 'strict')
  const events = []
  const context = {
    emitEvent: event => events.push(event),
    emitRun: run => events.push(...run.expand()),
  }
  try {
    decoder.decodeRow({ type, seq: 0, time: header.createdAt, data }, context)
    decoder.finish(context)
    return { ok: true, events: events.length }
  } catch (error) {
    return { ok: false, reason: error instanceof Error ? error.message : String(error) }
  }
}

function firstDifference(left, right) {
  const limit = Math.min(left.length, right.length)
  let index = 0
  while (index < limit && left[index] === right[index]) index += 1
  return index
}

rmSync(REPORT, { force: true })
let cargoFailure
try {
  execFileSync(
    'cargo',
    ['test', '-p', 'dsh-session', '--test', 'fixtures', '--target-dir', TARGET],
    { cwd: RUST, stdio: 'inherit', env: { ...process.env, DSH_FIXTURE_REPORT: REPORT } },
  )
} catch (error) {
  cargoFailure = `cargo test failed (exit ${error.status ?? 'unknown'})`
}
if (!existsSync(REPORT)) {
  console.error(`no report at ${REPORT}; cargo test did not reach write_comparison_report`)
  process.exit(1)
}

const report = JSON.parse(readFileSync(REPORT, 'utf8'))
const summary = { match: 0, documented: 0, undocumented: 0, pass: 0, fail: 0, skipped: 0 }
let hardFailure = cargoFailure !== undefined

console.log('')
console.log('TS <-> Rust v4 fixture comparison')
console.log(cargoFailure ?? 'cargo test: ok')
console.log('')

for (const scenario of SCENARIOS) {
  const physical = readLines(resolve(HERE, 'v4-physical', `${scenario}.v4.jsonl`))
  const projection = readLines(resolve(HERE, 'v4-event-data', `${scenario}.jsonl`))
  const header = JSON.parse(physical[0])
  const recorded = report.scenarios.find(entry => entry.name === scenario)
  if (recorded === undefined || recorded.rows.length !== projection.length) {
    console.error(`${scenario}: report does not cover every projection row`)
    hardFailure = true
    continue
  }

  console.log(scenario)
  console.log(`  ${'seq'.padStart(3)}  ${'event'.padEnd(24)} ${'bytes'.padEnd(34)} ts-decode`)
  for (const [index, tsLine] of projection.entries()) {
    const row = recorded.rows[index]
    if (row.seq !== index) {
      console.error(`  ${index}: report row has seq ${row.seq}`)
      hardFailure = true
      continue
    }
    if (row.rust.status === 'rejected') {
      summary.skipped += 1
      console.log(`  ${String(index).padStart(3)}  ${row.type.padEnd(24)} ${'SKIP (Rust rejects the row)'.padEnd(34)} -`)
      continue
    }

    const rustLine = row.rust.line
    const bytes = classify(tsLine, rustLine)
    let byteLabel
    if (bytes.kind === 'MATCH') {
      byteLabel = 'MATCH'
      summary.match += 1
    } else if (bytes.kind === 'DIFF') {
      byteLabel = `DIFF ${bytes.gaps.join('+')}`
      summary.documented += 1
    } else {
      byteLabel = `UNDOCUMENTED at char ${firstDifference(tsLine, rustLine)}`
      summary.undocumented += 1
      hardFailure = true
    }

    const { type, ...data } = JSON.parse(rustLine)
    const decoded = decodeRustRow(header, type, data)
    if (decoded.ok) {
      summary.pass += 1
      console.log(`  ${String(index).padStart(3)}  ${row.type.padEnd(24)} ${byteLabel.padEnd(34)} PASS`)
    } else {
      summary.fail += 1
      const expected = bytes.kind === 'DIFF' ? ` (expected: ${bytes.gaps.join('+')})` : ''
      console.log(`  ${String(index).padStart(3)}  ${row.type.padEnd(24)} ${byteLabel.padEnd(34)} FAIL${expected}`)
      console.log(`       ${decoded.reason}`)
      if (bytes.kind === 'MATCH') hardFailure = true
    }
  }
  console.log('')
}

console.log('summary')
console.log(`  bytes:     ${summary.match} match, ${summary.documented} documented differences, ${summary.undocumented} undocumented`)
console.log(`  ts-decode: ${summary.pass} pass, ${summary.fail} fail, ${summary.skipped} skipped`)
if (hardFailure) {
  console.log('  result:    FAIL')
  process.exit(1)
}
console.log('  result:    PASS (every difference is documented; every decodable row is accepted)')
