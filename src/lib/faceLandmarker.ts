import { FaceLandmarker, FilesetResolver, type Classifications, type NormalizedLandmark } from '@mediapipe/tasks-vision'

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

// ---- Eye closure ----
//
// The face model also scores expressions ("blendshapes"); its eyeBlink
// scores say how shut each eye is far more reliably than the gap between
// lid landmarks (which shifts with eye size, gaze and perspective). They
// ride along on the landmark array as a non-enumerable property, so every
// copy step (cropping, mirroring, reshaping) can carry them across.

/** How shut each eye is, 0..1, by anatomy: `r` the eye at landmarks 33 /
 *  133 (the subject's right), `l` the one at 362 / 263 (their left). */
export interface EyeClosure {
  r: number
  l: number
}

export function eyeClosureOf(lm: readonly unknown[] | null | undefined): EyeClosure | undefined {
  return (lm as { eyes?: EyeClosure } | null | undefined)?.eyes
}

/** Tags `lm` with `eyes` (returns `lm`). */
export function withEyeClosure<T extends object>(lm: T, eyes: EyeClosure | undefined): T {
  if (eyes) Object.defineProperty(lm, 'eyes', { value: eyes, enumerable: false, configurable: true })
  return lm
}

/** The eye closure in a landmarker result's blendshapes for face `i`. */
export function closureFrom(shapes: Classifications[] | undefined, i: number): EyeClosure | undefined {
  const cats = shapes?.[i]?.categories
  if (!cats?.length) return undefined
  const get = (name: string) => cats.find((c) => c.categoryName === name)?.score ?? 0
  return { r: get('eyeBlinkRight'), l: get('eyeBlinkLeft') }
}

let landmarkerPromise: Promise<FaceLandmarker> | null = null
let currentMode: 'IMAGE' | 'VIDEO' = 'IMAGE'

let filesetPromise: ReturnType<typeof FilesetResolver.forVisionTasks> | null = null
/** The vision WASM runtime (loaded once, shared by every task). */
export const visionFileset = () => (filesetPromise ??= FilesetResolver.forVisionTasks(WASM_BASE_URL))

/** A landmarker of its own (the video editor's, tracking several faces),
 *  as lenient as the camera's. */
export async function createLandmarker(runningMode: 'IMAGE' | 'VIDEO', numFaces: number): Promise<FaceLandmarker> {
  return FaceLandmarker.createFromOptions(await visionFileset(), {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode,
    numFaces,
    outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.35,
    minFacePresenceConfidence: 0.4,
    minTrackingConfidence: 0.35,
  })
}

function getLandmarker(): Promise<FaceLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      return FaceLandmarker.createFromOptions(await visionFileset(), {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
        runningMode: 'IMAGE',
        numFaces: 1,
        outputFaceBlendshapes: true,
        // Lenient (defaults are 0.5): a face half hidden behind the phone
        // in a mirror selfie, or small and far away, still counts, and the
        // tracker holds on to a face it has found through partial cover.
        minFaceDetectionConfidence: 0.3,
        minFacePresenceConfidence: 0.35,
        minTrackingConfidence: 0.3,
      })
    })()
  }
  return landmarkerPromise
}

async function inMode(mode: 'IMAGE' | 'VIDEO') {
  const landmarker = await getLandmarker()
  if (currentMode !== mode) {
    await landmarker.setOptions({ runningMode: mode })
    currentMode = mode
  }
  return landmarker
}

// ---- Regions of interest ----
//
// The face detector looks at the whole frame shrunk to 128px, so a face
// much under a fifth of the frame (a mirror selfie from a step back) is
// too small for it to see. When the whole frame finds nothing, we also
// try zoomed-in windows of it — where a face that small is plenty big —
// and map the landmarks back.

/** A region of a frame, in fractions of its width and height. */
export interface Roi {
  x: number
  y: number
  w: number
  h: number
}

const FULL: Roi = { x: 0, y: 0, w: 1, h: 1 }
const isFull = (r: Roi) => r.w > 0.98 && r.h > 0.98
const ROI_MAX = 640
/** The whole frame is handed to the tracker at up to this size. */
const FULL_MAX = 1920

/** A square window `side` px wide, centred on (cx, cy) and kept inside
 *  the frame. */
function windowAt(cx: number, cy: number, side: number, W: number, H: number): Roi {
  const w = Math.min(1, side / W)
  const h = Math.min(1, side / H)
  return { x: Math.min(1 - w, Math.max(0, cx - w / 2)), y: Math.min(1 - h, Math.max(0, cy - h / 2)), w, h }
}

// [centre x, centre y, side as a fraction of the frame's short side]:
// where faces sit in selfies and mirror shots — upper middle first.
const SEARCH: [number, number, number][] = [
  [0.5, 0.4, 0.6],
  [0.5, 0.3, 0.4],
  [0.3, 0.35, 0.5],
  [0.7, 0.35, 0.5],
  [0.5, 0.6, 0.6],
  [0.3, 0.3, 0.35],
  [0.7, 0.3, 0.35],
  [0.5, 0.5, 0.35],
]

function searchWindows(W: number, H: number) {
  const m = Math.min(W, H)
  return SEARCH.map(([cx, cy, s]) => windowAt(cx, cy, s * m, W, H))
}

/** For a still: a grid of overlapping half-size windows over the whole
 *  frame, after the likely spots. */
function gridWindows(W: number, H: number) {
  const m = Math.min(W, H)
  const out: Roi[] = []
  for (const s of [0.5, 0.33]) {
    const side = s * m
    const nx = Math.max(1, Math.ceil((W - side) / (side / 2))) + 1
    const ny = Math.max(1, Math.ceil((H - side) / (side / 2))) + 1
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) out.push(windowAt((side / 2 + ((W - side) * i) / Math.max(1, nx - 1)) / W, (side / 2 + ((H - side) * j) / Math.max(1, ny - 1)) / H, side, W, H))
  }
  return out
}

function faceBox(lm: NormalizedLandmark[]) {
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0
  for (const p of lm) {
    x0 = Math.min(x0, p.x)
    y0 = Math.min(y0, p.y)
    x1 = Math.max(x1, p.x)
    y1 = Math.max(y1, p.y)
  }
  return { x0, y0, x1, y1 }
}

/** A window around a known face, roomy enough to keep it while it moves. */
function windowAround(lm: NormalizedLandmark[], W: number, H: number): Roi {
  const b = faceBox(lm)
  const m = Math.min(W, H)
  const face = Math.max((b.x1 - b.x0) * W, (b.y1 - b.y0) * H)
  return windowAt((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, Math.min(m, Math.max(0.3 * m, face * 3)), W, H)
}

let roiCanvas: HTMLCanvasElement | null = null

/** `r` of `source`, drawn into a reused canvas. */
function cropOf(source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement, W: number, H: number, r: Roi, max = ROI_MAX) {
  roiCanvas ??= document.createElement('canvas')
  const sw = r.w * W
  const sh = r.h * H
  const k = Math.min(1, max / Math.max(sw, sh))
  const cw = Math.max(1, Math.round(sw * k))
  const ch = Math.max(1, Math.round(sh * k))
  if (roiCanvas.width !== cw || roiCanvas.height !== ch) {
    roiCanvas.width = cw
    roiCanvas.height = ch
  }
  roiCanvas.getContext('2d')!.drawImage(source, r.x * W, r.y * H, sw, sh, 0, 0, cw, ch)
  return roiCanvas
}

/** Landmarks found in `r` back in the whole frame's coordinates (`z` is in
 *  units of the input's width, so it scales with x). */
const fromRoi = (lm: NormalizedLandmark[], r: Roi) => (isFull(r) ? lm : lm.map((p) => ({ ...p, x: r.x + p.x * r.w, y: r.y + p.y * r.h, z: p.z * r.w })))

const sizeOf = (src: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement) =>
  src instanceof HTMLVideoElement ? [src.videoWidth, src.videoHeight] : src instanceof HTMLImageElement ? [src.naturalWidth, src.naturalHeight] : [src.width, src.height]

/** Landmarks for a still. `hint`: where the face was last seen (e.g. the
 *  live preview's landmarks for a captured frame), tried right after the
 *  whole frame. */
export async function detectFaceLandmarks(image: HTMLImageElement | HTMLCanvasElement, hint?: NormalizedLandmark[] | null): Promise<NormalizedLandmark[] | null> {
  const landmarker = await inMode('IMAGE')
  const res = landmarker.detect(image)
  const found = res.faceLandmarks[0]
  if (found) return withEyeClosure(found, closureFrom(res.faceBlendshapes, 0))
  const [W, H] = sizeOf(image)
  const tries = [...(hint ? [windowAround(hint, W, H)] : []), ...searchWindows(W, H), ...gridWindows(W, H)]
  for (const r of tries) {
    const res = landmarker.detect(cropOf(image, W, H, r))
    const lm = res.faceLandmarks[0]
    if (lm) return withEyeClosure(fromRoi(lm, r), closureFrom(res.faceBlendshapes, 0))
  }
  return null
}

// A face lost for a moment (a hand or the phone sweeping past) keeps its
// last landmarks this long, so effects don't flicker off and on.
const HOLD_MS = 300

/** Face tracking for a live `<video>`, called once per frame with a
 *  monotonically increasing timestamp (performance.now() is fine).
 *
 *  Tracks on the whole frame, or on a window around a face too small for
 *  the whole frame. While no face is found it alternates the whole frame
 *  with zoomed-in windows: the part of the frame on screen (`view`, when
 *  zoomed in), then the likely spots. */
export class LiveFaceTracker {
  private roi: Roi = FULL
  private misses = 0
  private step = 0
  private last: NormalizedLandmark[] | null = null
  private lastAt = -Infinity

  async detect(video: HTMLVideoElement, timestampMs: number, view: Roi = FULL): Promise<NormalizedLandmark[] | null> {
    const landmarker = await inMode('VIDEO')
    const W = video.videoWidth
    const H = video.videoHeight
    let roi = this.roi
    if (this.misses >= 2) {
      const windows = [...(view.w < 0.8 || view.h < 0.8 ? [view] : []), ...searchWindows(W, H)]
      roi = this.step % 2 === 0 ? FULL : windows[(this.step >> 1) % windows.length]
      this.step++
    }
    // (The whole frame as it comes, unless the camera's is a photo-sized
    // one: then a copy at a size the models gain nothing beyond.)
    const input = !isFull(roi) ? cropOf(video, W, H, roi) : Math.max(W, H) > FULL_MAX ? cropOf(video, W, H, FULL, FULL_MAX) : video
    const res = landmarker.detectForVideo(input, timestampMs)
    const raw = res.faceLandmarks[0]
    if (!raw) {
      this.misses++
      return timestampMs - this.lastAt < HOLD_MS ? this.last : null
    }
    const lm = withEyeClosure(fromRoi(raw, roi), closureFrom(res.faceBlendshapes, 0))
    this.roi = this.follow(lm, roi, W, H)
    this.misses = 0
    this.step = 0
    this.last = lm
    this.lastAt = timestampMs
    return lm
  }

  /** The region to track in next: the same one while the face sits well
   *  inside it; otherwise re-centred on the face, or the whole frame once
   *  the face is big enough for it. */
  private follow(lm: NormalizedLandmark[], roi: Roi, W: number, H: number): Roi {
    if (isFull(roi)) return roi
    const b = faceBox(lm)
    const m = Math.min(W, H)
    const face = Math.max((b.x1 - b.x0) * W, (b.y1 - b.y0) * H)
    const side = Math.min(roi.w * W, roi.h * H)
    const cx = (b.x0 + b.x1) / 2
    const cy = (b.y0 + b.y1) / 2
    const centred = cx > roi.x + roi.w * 0.2 && cx < roi.x + roi.w * 0.8 && cy > roi.y + roi.h * 0.2 && cy < roi.y + roi.h * 0.8
    if (centred && face < side * 0.45 && face > side * 0.15) return roi
    if (face > m * 0.3) return FULL
    return windowAround(lm, W, H)
  }
}
