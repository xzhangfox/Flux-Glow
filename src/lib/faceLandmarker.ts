import { FaceLandmarker, FilesetResolver, type NormalizedLandmark } from '@mediapipe/tasks-vision'

// Self-hosted (copied from node_modules/@mediapipe/tasks-vision/wasm at
// install time — see scripts/copy-mediapipe-wasm.js) rather than pulled
// from jsdelivr at runtime: one less third-party dependency for a page
// whose whole pitch is "nothing leaves your device," and it has to match
// the installed npm package's own version exactly (the WASM binary and
// the JS API driving it are version-locked to each other).
const WASM_BASE_URL = '/mediapipe/wasm'
// Also self-hosted (public/mediapipe/face_landmarker.task, ~3.6MB, fetched
// once from Google's own model repo and committed here) for the same
// reason as the WASM runtime above.
const MODEL_URL = '/mediapipe/face_landmarker.task'

let landmarkerPromise: Promise<FaceLandmarker> | null = null
let currentMode: 'IMAGE' | 'VIDEO' = 'IMAGE'

function getLandmarker(): Promise<FaceLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const filesetResolver = await FilesetResolver.forVisionTasks(WASM_BASE_URL)
      return FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
        runningMode: 'IMAGE',
        numFaces: 1,
      })
    })()
  }
  return landmarkerPromise
}

export async function detectFaceLandmarks(image: HTMLImageElement | HTMLCanvasElement): Promise<NormalizedLandmark[] | null> {
  const landmarker = await getLandmarker()
  if (currentMode !== 'IMAGE') {
    await landmarker.setOptions({ runningMode: 'IMAGE' })
    currentMode = 'IMAGE'
  }
  const result = landmarker.detect(image)
  return result.faceLandmarks[0] ?? null
}

/** For a live `<video>` stream — tracking mode, called once per frame with
 *  a monotonically increasing timestamp (performance.now() is fine). */
export async function detectFaceLandmarksForVideo(video: HTMLVideoElement, timestampMs: number): Promise<NormalizedLandmark[] | null> {
  const landmarker = await getLandmarker()
  if (currentMode !== 'VIDEO') {
    await landmarker.setOptions({ runningMode: 'VIDEO' })
    currentMode = 'VIDEO'
  }
  const result = landmarker.detectForVideo(video, timestampMs)
  return result.faceLandmarks[0] ?? null
}
