// Resolve the shipped Session-format packages to their TypeScript sources so
// the fixture generator runs the real encoder/decoder without an install.
// Only the physical codec path is loaded; `migration.ts` (which imports the
// decorator-bearing dsh-llm entrypoint) is deliberately not reachable here.
import { fileURLToPath, pathToFileURL } from 'node:url'
import { dirname, resolve as resolvePath } from 'node:path'

const HERE = dirname(fileURLToPath(import.meta.url))
// The TypeScript monorepo lives under `origin/`; the Rust rewrite lives
// beside it under `rust/`. Both are subdirectories of the repository root,
// so the sources are two levels up from `rust/fixtures/` and then into
// `origin/`.
const REPO = resolvePath(HERE, '../../origin')
const SESSION = resolvePath(REPO, 'packages/session')

const MAP = new Map([
  ['@deepseek-ai/dsh-session-format', resolvePath(SESSION, 'session-format/src/index.ts')],
  ['@deepseek-ai/dsh-session-format-v0-to-v1', resolvePath(SESSION, 'session-format-v0-to-v1/src/index.ts')],
  // The published v1-to-v2 index re-exports migration.ts, whose dsh-llm import
  // cannot load under Node's type transform (decorators). The codec subset is
  // all the v4 physical encoder needs.
  ['@deepseek-ai/dsh-session-format-v1-to-v2', resolvePath(HERE, 'v1-to-v2-codec.ts')],
  ['@deepseek-ai/dsh-session-format-v2-to-v3', resolvePath(SESSION, 'session-format-v2-to-v3/src/index.ts')],
  ['@deepseek-ai/dsh-session-format-v3-to-v4', resolvePath(SESSION, 'session-format-v3-to-v4/src/index.ts')],
  ['@deepseek-ai/dsh-util-values', resolvePath(REPO, 'packages/util/values/src/index.ts')],
])

export async function resolve(specifier, context, nextResolve) {
  const target = MAP.get(specifier)
  if (target !== undefined) return { url: pathToFileURL(target).href, shortCircuit: true }
  return nextResolve(specifier, context)
}
