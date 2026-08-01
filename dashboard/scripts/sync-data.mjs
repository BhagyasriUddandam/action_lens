// Copies the ML pipeline's output into public/, where Vite serves it as static
// assets. Run this after any `python src/evaluate.py` rerun (via `bun run sync`).
//
// Frame images live in ../data/frames/{label}/{clip_id}/frame_NN.jpg and are
// gitignored -- too many (2560 files) to ship whole. Instead we export just
// the first and last frame of each curated failure-case clip as a small
// thumbnail pair, showing the direction of motion the VLM's prompt keys on.

import { readFileSync, copyFileSync, mkdirSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const repoRoot = join(here, '..', '..')
const resultsSrc = join(repoRoot, 'results', 'results.json')
const resultsDest = join(here, '..', 'public', 'results.json')
const clipsDest = join(here, '..', 'public', 'clips')

if (!existsSync(resultsSrc)) {
  console.error(`ERROR: ${resultsSrc} not found. Run: python src/evaluate.py`)
  process.exit(1)
}

mkdirSync(clipsDest, { recursive: true })
copyFileSync(resultsSrc, resultsDest)
console.log(`copied results.json -> public/results.json`)

const results = JSON.parse(readFileSync(resultsSrc, 'utf-8'))
let copied = 0
let missing = 0

for (const fc of results.failure_cases ?? []) {
  for (const frame of ['00', '15']) {
    const src = join(repoRoot, 'data', 'frames', fc.label, fc.clip_id, `frame_${frame}.jpg`)
    const dest = join(clipsDest, `${fc.clip_id}_${frame}.jpg`)
    if (existsSync(src)) {
      copyFileSync(src, dest)
      copied++
    } else {
      missing++
      console.warn(`  missing frame: ${src}`)
    }
  }
}

console.log(`copied ${copied} clip thumbnails${missing ? ` (${missing} missing -- run data_prep.py to regenerate)` : ''}`)
