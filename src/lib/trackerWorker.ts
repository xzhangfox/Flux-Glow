// The live camera's face tracking, off the page's main thread (see
// LiveFaceTracker in faceLandmarker.ts): the page hands over a small
// picture of each frame and gets the landmarks back, while it retouches
// the frame before — the two run side by side instead of one after the
// other, and taps never wait on the tracking.
import { FaceLandmarker, FilesetResolver, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { closureFrom, LIVE_OPTIONS, MODEL_URL, WASM_BASE_URL } from './landmarkerConfig'

let landmarker: FaceLandmarker | null = null
const post = (m: unknown) => (self as unknown as Worker).postMessage(m)

async function init() {
  // (the ES-module build of the runtime: a module worker can't load the
  // classic one)
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE_URL, true)
  const make = (delegate: 'GPU' | 'CPU') => FaceLandmarker.createFromOptions(fileset, { ...LIVE_OPTIONS, baseOptions: { modelAssetPath: MODEL_URL, delegate }, runningMode: 'VIDEO' })
  // On the CPU: then it runs truly alongside the page, on a core of its
  // own. (On the GPU it queued behind the page's own GPU work — the
  // retouch, the 3D effects — and gained nothing.)
  try {
    landmarker = await make('CPU')
  } catch {
    landmarker = await make('GPU')
  }
}

self.onmessage = async (e: MessageEvent<{ type: 'init' } | { type: 'detect'; id: number; image: ImageBitmap; ts: number }>) => {
  const m = e.data
  if (m.type === 'init') {
    try {
      await init()
      post({ type: 'ready' })
    } catch (err) {
      post({ type: 'failed', error: String(err) })
    }
    return
  }
  let out: { lm: NormalizedLandmark[]; eyes?: { r: number; l: number } } | null = null
  try {
    const res = landmarker!.detectForVideo(m.image, m.ts)
    const f = res.faceLandmarks[0]
    if (f) out = { lm: f.map((p) => ({ x: p.x, y: p.y, z: p.z, visibility: p.visibility })), eyes: closureFrom(res.faceBlendshapes, 0) }
  } catch {
    // (a frame lost)
  } finally {
    m.image.close()
  }
  post({ type: 'result', id: m.id, out })
}
