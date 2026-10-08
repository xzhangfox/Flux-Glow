import type { Classifications } from '@mediapipe/tasks-vision'

// What the page's face landmarker and the live tracker's worker
// (trackerWorker.ts) share — kept in one place so the two never drift.

// Self-hosted (copied from node_modules/@mediapipe/tasks-vision/wasm at
// install time — see scripts/copy-mediapipe-wasm.js) rather than pulled
// from jsdelivr at runtime: one less third-party dependency for a page
// whose whole pitch is "nothing leaves your device," and it has to match
// the installed npm package's own version exactly (the WASM binary and
// the JS API driving it are version-locked to each other).
export const WASM_BASE_URL = '/mediapipe/wasm'
// Also self-hosted (public/mediapipe/face_landmarker.task, ~3.6MB, fetched
// once from Google's own model repo and committed here) for the same
// reason as the WASM runtime above.
export const MODEL_URL = '/mediapipe/face_landmarker.task'

/** The camera's landmarker settings. Lenient (defaults are 0.5): a face
 *  half hidden behind the phone in a mirror selfie, or small and far away,
 *  still counts, and the tracker holds on to a face it has found through
 *  partial cover. */
export const LIVE_OPTIONS = {
  numFaces: 1,
  outputFaceBlendshapes: true,
  minFaceDetectionConfidence: 0.3,
  minFacePresenceConfidence: 0.35,
  minTrackingConfidence: 0.3,
}

/** The eye closure in a landmarker result's blendshapes for face `i`. */
export function closureFrom(shapes: Classifications[] | undefined, i: number): { r: number; l: number } | undefined {
  const cats = shapes?.[i]?.categories
  if (!cats?.length) return undefined
  const get = (name: string) => cats.find((c) => c.categoryName === name)?.score ?? 0
  return { r: get('eyeBlinkRight'), l: get('eyeBlinkLeft') }
}
