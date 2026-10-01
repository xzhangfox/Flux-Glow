// Copies the MediaPipe tasks-vision WASM runtime into public/ so it's
// self-hosted instead of fetched from jsdelivr at runtime (see
// src/lib/faceLandmarker.ts). Runs on postinstall so a fresh `npm install`
// (including Vercel's build) always has a copy matching the installed
// package version.
import { cpSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const src = join(root, 'node_modules/@mediapipe/tasks-vision/wasm')
const dest = join(root, 'public/mediapipe/wasm')

if (!existsSync(src)) {
  console.warn('[copy-mediapipe-wasm] source not found, skipping:', src)
  process.exit(0)
}
mkdirSync(dest, { recursive: true })
cpSync(src, dest, { recursive: true })
console.log('[copy-mediapipe-wasm] copied to', dest)
