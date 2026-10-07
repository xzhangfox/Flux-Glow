import { ImageEmbedder, type FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { ALL_FORMATS, BlobSource, CanvasSink, Input } from 'mediabunny'
import { closureFrom, createLandmarker, visionFileset, withEyeClosure, type EyeClosure } from '../faceLandmarker'

// Video analysis for the video editor: every face in every frame, who each
// one is, and a steady track of each person's face mesh through the clip —
// so each person keeps their own retouch and effects, and those stay locked
// to their face, frame after frame.
//
// One pass over the decoded frames (at a working size), in order:
//
//  1. Detect. The landmarker in VIDEO mode finds and tracks up to six faces
//     on the whole frame. Faces too small for that (a group shot, someone
//     at the back) are found by a periodic scan of zoomed-in tiles, and a
//     face the whole-frame pass loses for a moment (turned away, half
//     covered, motion blur) is looked for again in a window around where
//     it's heading.
//  2. Associate. Each detection joins the track whose predicted box
//     (constant velocity) it overlaps best; the rest start new tracks. A
//     track survives short misses and ends after a longer one.
//  3. Identify. Each track gathers appearance embeddings (an image
//     embedder on the face, aligned upright, hair included) and a shape
//     signature. After the pass, tracks are joined into people: two tracks
//     that never share a frame and look alike (or one picks up where the
//     other left off) are the same person — so someone who leaves and comes
//     back, or turns away for a while, keeps their identity and settings.
//  4. Smooth. Each person's mesh is filled across short gaps and smoothed
//     in time with a speed-adaptive (One Euro) filter run forwards and
//     backwards, which removes jitter without lag — possible because the
//     whole clip is known in advance, unlike a live camera.

const WORK_MAX = 960
/** Landmarks per face (the 468-point mesh and the irises). */
export const N_LM = 478
const STRIDE = N_LM * 3
/** Per face per frame: the landmarks, then how shut each eye is (r, l). */
const FRAME = STRIDE + 2
const MAX_FACES = 6
/** Most of a second's absence still continues a track. */
const TRACK_GAP_S = 0.8
/** Gaps up to this long inside a person's presence are filled in. */
const FILL_GAP_S = 0.6
const TILE_EVERY = 10
const EMBED_EVERY = 12
export const MAX_DURATION_S = 180

const EMBEDDER_URL = '/mediapipe/image_embedder.tflite'

export interface PersonInfo {
  id: number
  /** A face picture (data URL) for the people strip. */
  thumb: string
  /** Seconds in the clip where this person is on screen. */
  spans: [number, number][]
  /** Their clearest frame, and their landmarks in it (for effect and
   *  look previews). */
  best: { frame: HTMLCanvasElement; landmarks: NormalizedLandmark[] }
  /** Share of the clip they're in (0..1). */
  presence: number
}

export interface VideoAnalysis {
  width: number
  height: number
  duration: number
  /** Presentation time of each analysed frame, seconds, ascending. */
  times: Float64Array
  people: PersonInfo[]
  /** Smoothed landmarks of `personId` at time `t` (interpolated between
   *  frames), or null where they're not on screen. */
  landmarksAt(personId: number, t: number): NormalizedLandmark[] | null
  /** Every person on screen at `t`. */
  facesAt(t: number): { id: number; landmarks: NormalizedLandmark[] }[]
  /** Joins `from` into `into` (the user says they're the same person). */
  merge(from: number, into: number): void
}

export interface AnalyzeProgress {
  /** 0..1 */
  progress: number
  people: number
}

// ---- Geometry ---------------------------------------------------------------------

interface Box {
  x0: number
  y0: number
  x1: number
  y1: number
}

function boxOf(lm: Float32Array, W: number, H: number): Box {
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (let i = 0; i < 468; i++) {
    const x = lm[i * 3] * W
    const y = lm[i * 3 + 1] * H
    if (x < x0) x0 = x
    if (x > x1) x1 = x
    if (y < y0) y0 = y
    if (y > y1) y1 = y
  }
  return { x0, y0, x1, y1 }
}

const area = (b: Box) => Math.max(0, b.x1 - b.x0) * Math.max(0, b.y1 - b.y0)
function iou(a: Box, b: Box) {
  const i = area({ x0: Math.max(a.x0, b.x0), y0: Math.max(a.y0, b.y0), x1: Math.min(a.x1, b.x1), y1: Math.min(a.y1, b.y1) })
  return i / Math.max(1e-6, area(a) + area(b) - i)
}
const sizeOf = (b: Box) => Math.max(b.x1 - b.x0, b.y1 - b.y0)
const centreOf = (b: Box) => ({ x: (b.x0 + b.x1) / 2, y: (b.y0 + b.y1) / 2 })
const shift = (b: Box, dx: number, dy: number): Box => ({ x0: b.x0 + dx, y0: b.y0 + dy, x1: b.x1 + dx, y1: b.y1 + dy })

function pack(lm: NormalizedLandmark[], eyes: EyeClosure | undefined, roi?: { x: number; y: number; w: number; h: number }): Float32Array {
  const out = new Float32Array(FRAME)
  out[STRIDE] = eyes?.r ?? 0
  out[STRIDE + 1] = eyes?.l ?? 0
  const n = Math.min(N_LM, lm.length)
  for (let i = 0; i < n; i++) {
    const p = lm[i]
    out[i * 3] = roi ? roi.x + p.x * roi.w : p.x
    out[i * 3 + 1] = roi ? roi.y + p.y * roi.h : p.y
    out[i * 3 + 2] = roi ? p.z * roi.w : p.z
  }
  return out
}

function unpack(a: Float32Array): NormalizedLandmark[] {
  const out: NormalizedLandmark[] = new Array(N_LM)
  for (let i = 0; i < N_LM; i++) out[i] = { x: a[i * 3], y: a[i * 3 + 1], z: a[i * 3 + 2], visibility: 1 }
  return withEyeClosure(out, { r: a[STRIDE], l: a[STRIDE + 1] })
}

/** How squarely the face looks at the camera (1 frontal, 0 profile). */
function frontal(lm: Float32Array) {
  const eyeDx = Math.abs(lm[263 * 3] - lm[33 * 3])
  const mid = (lm[263 * 3] + lm[33 * 3]) / 2
  return Math.max(0, 1 - Math.abs(lm[4 * 3] - mid) / Math.max(1e-6, eyeDx))
}

/** Proportions that belong to a face rather than to its pose or
 *  expression: distances between rigid points, over the eye spacing. */
const SIG_PAIRS: [number, number][] = [
  [33, 133], [362, 263], [168, 4], [4, 2], [61, 291], [234, 454], [10, 152], [168, 152], [133, 362], [98, 327], [70, 300], [172, 397],
]
function signature(lm: Float32Array): Float32Array {
  const d = (a: number, b: number) => Math.hypot(lm[a * 3] - lm[b * 3], lm[a * 3 + 1] - lm[b * 3 + 1], lm[a * 3 + 2] - lm[b * 3 + 2])
  const e = Math.max(1e-6, d(33, 263))
  return Float32Array.from(SIG_PAIRS.map(([a, b]) => d(a, b) / e))
}

// ---- Models ------------------------------------------------------------------------

let models: Promise<{ video: FaceLandmarker; image: FaceLandmarker; embedder: ImageEmbedder | null }> | null = null
function loadModels() {
  models ??= (async () => {
    const [video, image] = await Promise.all([createLandmarker('VIDEO', MAX_FACES), createLandmarker('IMAGE', 3)])
    let embedder: ImageEmbedder | null = null
    try {
      embedder = await ImageEmbedder.createFromOptions(await visionFileset(), {
        baseOptions: { modelAssetPath: EMBEDDER_URL, delegate: 'GPU' },
        runningMode: 'IMAGE',
        l2Normalize: true,
        quantize: false,
      })
    } catch {
      // Without it, identity rests on face shape and on position over time.
      embedder = null
    }
    return { video, image, embedder }
  })()
  return models
}

/** Warm the models up ahead of time (e.g. when the editor opens). */
export const preloadVideoModels = () => loadModels().then(() => undefined)

const scratch = { roi: null as HTMLCanvasElement | null, face: null as HTMLCanvasElement | null }

function crop(src: HTMLCanvasElement, r: { x: number; y: number; w: number; h: number }, max = 640) {
  const c = (scratch.roi ??= document.createElement('canvas'))
  const sw = r.w * src.width
  const sh = r.h * src.height
  const k = Math.min(1, max / Math.max(sw, sh))
  const cw = Math.max(1, Math.round(sw * k))
  const ch = Math.max(1, Math.round(sh * k))
  if (c.width !== cw || c.height !== ch) {
    c.width = cw
    c.height = ch
  }
  c.getContext('2d')!.drawImage(src, r.x * src.width, r.y * src.height, sw, sh, 0, 0, cw, ch)
  return c
}

/** A square region (fractions of the frame) of `side` px round (cx, cy). */
function windowAt(cx: number, cy: number, side: number, W: number, H: number) {
  const w = Math.min(1, side / W)
  const h = Math.min(1, side / H)
  return { x: Math.min(1 - w, Math.max(0, cx / W - w / 2)), y: Math.min(1 - h, Math.max(0, cy / H - h / 2)), w, h }
}

/** The face upright (eyes level), roughly a head and shoulders wide: hair
 *  and outline say as much about who someone is as the features do. */
function alignedFace(frame: HTMLCanvasElement, lm: Float32Array) {
  const W = frame.width
  const H = frame.height
  const c = (scratch.face ??= document.createElement('canvas'))
  c.width = c.height = 160
  const g = c.getContext('2d')!
  const lx = lm[33 * 3] * W
  const ly = lm[33 * 3 + 1] * H
  const rx = lm[263 * 3] * W
  const ry = lm[263 * 3 + 1] * H
  const ang = Math.atan2(ry - ly, rx - lx)
  const e = Math.hypot(rx - lx, ry - ly)
  const cx = (lx + rx) / 2 + (lm[13 * 3] * W - (lx + rx) / 2) * 0.35
  const cy = (ly + ry) / 2 + (lm[13 * 3 + 1] * H - (ly + ry) / 2) * 0.35
  const side = e * 3.4
  g.setTransform(1, 0, 0, 1, 0, 0)
  g.fillStyle = '#808080'
  g.fillRect(0, 0, 160, 160)
  g.translate(80, 80)
  g.scale(160 / side, 160 / side)
  g.rotate(-ang)
  g.translate(-cx, -cy)
  g.drawImage(frame, 0, 0)
  g.setTransform(1, 0, 0, 1, 0, 0)
  return c
}

// ---- Tracks ------------------------------------------------------------------------

interface Track {
  id: number
  first: number
  last: number
  lm: Map<number, Float32Array>
  box: Box
  vx: number
  vy: number
  embeds: Float32Array[]
  sigs: Float32Array[]
  best: { score: number; frame: HTMLCanvasElement | null; lm: Float32Array | null }
}

interface Det {
  lm: Float32Array
  box: Box
}

/** Drops detections that are the same face found twice (whole frame and a
 *  tile, say), keeping the larger. */
function dedupe(dets: Det[]): Det[] {
  dets.sort((a, b) => area(b.box) - area(a.box))
  const out: Det[] = []
  for (const d of dets) if (!out.some((o) => iou(o.box, d.box) > 0.35)) out.push(d)
  return out
}

function snapshot(frame: HTMLCanvasElement) {
  const k = Math.min(1, 640 / Math.max(frame.width, frame.height))
  const c = document.createElement('canvas')
  c.width = Math.round(frame.width * k)
  c.height = Math.round(frame.height * k)
  c.getContext('2d')!.drawImage(frame, 0, 0, c.width, c.height)
  return c
}

function cosine(a: Float32Array, b: Float32Array) {
  let d = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  return d / Math.max(1e-9, Math.sqrt(na * nb))
}

function mean(vs: Float32Array[]): Float32Array | null {
  if (!vs.length) return null
  const out = new Float32Array(vs[0].length)
  for (const v of vs) for (let i = 0; i < v.length; i++) out[i] += v[i]
  for (let i = 0; i < out.length; i++) out[i] /= vs.length
  return out
}

/** Median signature: robust to the odd frame caught mid-blink or mid-turn. */
function medianSig(vs: Float32Array[]): Float32Array | null {
  if (!vs.length) return null
  const out = new Float32Array(vs[0].length)
  for (let i = 0; i < out.length; i++) {
    const col = vs.map((v) => v[i]).sort((a, b) => a - b)
    out[i] = col[col.length >> 1]
  }
  return out
}

// ---- Smoothing ----------------------------------------------------------------------

/** One Euro filter over one coordinate series (in place into `out`):
 *  heavy smoothing while still, light while moving fast. */
function oneEuro(xs: Float32Array, ts: Float64Array, out: Float32Array, reverse: boolean, minCutoff: number, beta: number) {
  const n = xs.length
  const alpha = (cutoff: number, dt: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt))
  let prev = 0
  let dPrev = 0
  for (let k = 0; k < n; k++) {
    const i = reverse ? n - 1 - k : k
    const x = xs[i]
    if (k === 0) {
      prev = x
      dPrev = 0
      out[i] = x
      continue
    }
    const j = reverse ? i + 1 : i - 1
    const dt = Math.max(1e-3, Math.abs(ts[i] - ts[j]))
    const dx = (x - prev) / dt
    const ad = alpha(1, dt)
    dPrev = dPrev + ad * (dx - dPrev)
    const a = alpha(minCutoff + beta * Math.abs(dPrev), dt)
    prev = prev + a * (x - prev)
    out[i] = prev
  }
}

/** Zero-lag smoothing of a run of frames: the forward and backward passes
 *  lag in opposite directions, so their average doesn't. */
function smoothRun(frames: Float32Array[], ts: Float64Array) {
  const n = frames.length
  if (n < 3) return frames
  const series = new Float32Array(n)
  const fwd = new Float32Array(n)
  const bwd = new Float32Array(n)
  const out = frames.map(() => new Float32Array(FRAME))
  for (let c = 0; c < FRAME; c++) {
    for (let i = 0; i < n; i++) series[i] = frames[i][c]
    // Eye closure as found: a blink is over in a few frames, and smoothing
    // would wash it out.
    if (c >= STRIDE) {
      for (let i = 0; i < n; i++) out[i][c] = series[i]
      continue
    }
    // z is noisier and matters less: smoothed harder.
    const isZ = c % 3 === 2
    oneEuro(series, ts, fwd, false, isZ ? 0.8 : 1.4, isZ ? 3 : 9)
    oneEuro(series, ts, bwd, true, isZ ? 0.8 : 1.4, isZ ? 3 : 9)
    for (let i = 0; i < n; i++) out[i][c] = (fwd[i] + bwd[i]) / 2
  }
  return out
}

// ---- People -----------------------------------------------------------------------

interface Person {
  id: number
  tracks: Track[]
  /** Smoothed landmarks by frame index (absent where off screen). */
  frames: (Float32Array | undefined)[]
}

/** Builds each person's smoothed, gap-filled mesh from their tracks. */
function buildFrames(p: Person, times: Float64Array) {
  const raw: (Float32Array | undefined)[] = new Array(times.length)
  for (const t of p.tracks) for (const [i, lm] of t.lm) raw[i] = lm
  // Fill short gaps by interpolation.
  let lastIdx = -1
  for (let i = 0; i < raw.length; i++) {
    if (!raw[i]) continue
    if (lastIdx >= 0 && i - lastIdx > 1 && times[i] - times[lastIdx] <= FILL_GAP_S) {
      const a = raw[lastIdx]!
      const b = raw[i]!
      for (let k = lastIdx + 1; k < i; k++) {
        const f = (times[k] - times[lastIdx]) / (times[i] - times[lastIdx])
        const m = new Float32Array(FRAME)
        for (let c = 0; c < FRAME; c++) m[c] = a[c] + (b[c] - a[c]) * f
        raw[k] = m
      }
    }
    lastIdx = i
  }
  // Smooth each contiguous run.
  const out: (Float32Array | undefined)[] = new Array(times.length)
  let i = 0
  while (i < raw.length) {
    if (!raw[i]) {
      i++
      continue
    }
    let j = i
    while (j < raw.length && raw[j]) j++
    const run = raw.slice(i, j) as Float32Array[]
    const sm = smoothRun(run, times.slice(i, j))
    for (let k = 0; k < sm.length; k++) out[i + k] = sm[k]
    i = j
  }
  p.frames = out
}

function spansOf(p: Person, times: Float64Array): [number, number][] {
  const spans: [number, number][] = []
  let start = -1
  for (let i = 0; i <= p.frames.length; i++) {
    const on = i < p.frames.length && !!p.frames[i]
    if (on && start < 0) start = i
    if (!on && start >= 0) {
      spans.push([times[start], times[i - 1]])
      start = -1
    }
  }
  return spans
}

function faceThumb(frame: HTMLCanvasElement, lm: Float32Array) {
  const W = frame.width
  const H = frame.height
  const b = boxOf(lm, W, H)
  const side = sizeOf(b) * 1.45
  const c = centreOf(b)
  const out = document.createElement('canvas')
  out.width = out.height = 112
  out.getContext('2d')!.drawImage(frame, c.x - side / 2, c.y - side / 2 - side * 0.04, side, side, 0, 0, 112, 112)
  return out.toDataURL('image/jpeg', 0.85)
}

// ---- The pass --------------------------------------------------------------------

/** Analyses `file`. Rejects with an AbortError if `signal` aborts. */
export async function analyzeVideo(file: Blob, onProgress: (p: AnalyzeProgress) => void, signal?: AbortSignal): Promise<VideoAnalysis> {
  const { video: vLm, image: iLm, embedder } = await loadModels()
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
  const track = await input.getPrimaryVideoTrack()
  if (!track) throw new Error('This file has no video.')
  if (!(await track.canDecode())) throw new Error("This browser can't decode this video's format.")
  const duration = await input.computeDuration()
  if (duration > MAX_DURATION_S + 1) throw new Error(`Videos up to ${MAX_DURATION_S / 60} minutes can be edited.`)
  const dw = track.displayWidth
  const dh = track.displayHeight
  const k = Math.min(1, WORK_MAX / Math.max(dw, dh))
  const W = Math.max(2, Math.round(dw * k))
  const H = Math.max(2, Math.round(dh * k))
  const sink = new CanvasSink(track, { width: W, height: H, fit: 'fill', poolSize: 2 })

  const times: number[] = []
  const live: Track[] = []
  const done: Track[] = []
  let nextId = 1
  let lastTs = -1
  // Start the VIDEO-mode clock beyond anything a previous clip used.
  const clockBase = performance.now()
  let frameIdx = 0
  const fps = (await track.computePacketStats(60).catch(() => null))?.averagePacketRate || 30
  const gapFrames = Math.max(3, Math.round(TRACK_GAP_S * fps))

  const detectIn = (frame: HTMLCanvasElement, r: { x: number; y: number; w: number; h: number }): Det[] => {
    const c = crop(frame, r)
    const res = iLm.detect(c)
    return res.faceLandmarks.map((lm, i) => {
      const a = pack(lm, closureFrom(res.faceBlendshapes, i), r)
      return { lm: a, box: boxOf(a, W, H) }
    })
  }

  for await (const { canvas, timestamp } of sink.canvases()) {
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError')
    const frame = canvas as HTMLCanvasElement
    if (timestamp <= lastTs) continue
    lastTs = timestamp
    const fi = frameIdx++
    times.push(timestamp)
    const ms = clockBase + timestamp * 1000

    // 1. Whole frame.
    const vres = vLm.detectForVideo(frame, ms)
    let dets: Det[] = vres.faceLandmarks.map((lm, i) => {
      const a = pack(lm, closureFrom(vres.faceBlendshapes, i))
      return { lm: a, box: boxOf(a, W, H) }
    })
    // Small faces: tiles, now and then (and on the first frame).
    if (fi % TILE_EVERY === 0) {
      const m = Math.min(W, H)
      const side = m * 0.62
      const tiles = [
        windowAt(W * 0.3, H * 0.3, side, W, H),
        windowAt(W * 0.7, H * 0.3, side, W, H),
        windowAt(W * 0.3, H * 0.7, side, W, H),
        windowAt(W * 0.7, H * 0.7, side, W, H),
        windowAt(W * 0.5, H * 0.45, side, W, H),
      ]
      for (const r of tiles) dets.push(...detectIn(frame, r))
    }
    dets = dedupe(dets)

    // 2. Associate with live tracks (best overlap first).
    const pairs: [number, number, number][] = []
    live.forEach((t, ti) => {
      const pred = shift(t.box, t.vx, t.vy)
      dets.forEach((d, di) => {
        const o = iou(pred, d.box)
        const cd = Math.hypot(centreOf(pred).x - centreOf(d.box).x, centreOf(pred).y - centreOf(d.box).y) / Math.max(1, sizeOf(pred))
        const score = o > 0.15 ? o : cd < 0.5 ? 0.15 * (1 - cd) : 0
        if (score > 0) pairs.push([score, ti, di])
      })
    })
    pairs.sort((a, b) => b[0] - a[0])
    const tUsed = new Set<number>()
    const dUsed = new Set<number>()
    const matched: [Track, Det][] = []
    for (const [, ti, di] of pairs) {
      if (tUsed.has(ti) || dUsed.has(di)) continue
      tUsed.add(ti)
      dUsed.add(di)
      matched.push([live[ti], dets[di]])
    }
    // A live track the whole frame lost: look again around where it's going.
    live.forEach((t, ti) => {
      if (tUsed.has(ti) || fi - t.last > gapFrames) return
      const pred = shift(t.box, t.vx * (fi - t.last), t.vy * (fi - t.last))
      const c = centreOf(pred)
      const r = windowAt(c.x, c.y, Math.min(Math.min(W, H), Math.max(sizeOf(pred) * 2.6, 96)), W, H)
      const found = detectIn(frame, r).filter((d) => !dets.some((o) => iou(o.box, d.box) > 0.35))
      let best: Det | null = null
      let bo = 0.12
      for (const d of found) {
        const o = iou(pred, d.box)
        if (o > bo) {
          bo = o
          best = d
        }
      }
      if (best) {
        tUsed.add(ti)
        matched.push([t, best])
      }
    })

    for (const [t, d] of matched) {
      const dtf = Math.max(1, fi - t.last)
      const c0 = centreOf(t.box)
      const c1 = centreOf(d.box)
      // Velocity per frame, eased so one jumpy detection doesn't fling it.
      t.vx = t.vx * 0.5 + ((c1.x - c0.x) / dtf) * 0.5
      t.vy = t.vy * 0.5 + ((c1.y - c0.y) / dtf) * 0.5
      t.box = d.box
      t.last = fi
      t.lm.set(fi, d.lm)
    }
    // New tracks for the rest.
    dets.forEach((d, di) => {
      if (dUsed.has(di) || matched.some(([, m]) => m === d)) return
      if (sizeOf(d.box) < 14) return
      const t: Track = { id: nextId++, first: fi, last: fi, lm: new Map([[fi, d.lm]]), box: d.box, vx: 0, vy: 0, embeds: [], sigs: [], best: { score: 0, frame: null, lm: null } }
      live.push(t)
      matched.push([t, d])
    })

    // 3. Appearance, shape and best view, for tracks seen this frame.
    for (const [t, d] of matched) {
      const fr = frontal(d.lm)
      const sz = sizeOf(d.box)
      if (fr > 0.55) t.sigs.push(signature(d.lm))
      if (embedder && fr > 0.45 && sz > 28 && (t.embeds.length === 0 || (fi - t.first) % EMBED_EVERY === 0)) {
        const e = embedder.embed(alignedFace(frame, d.lm)).embeddings[0]?.floatEmbedding
        if (e) t.embeds.push(Float32Array.from(e))
      }
      const score = sz * (0.35 + fr)
      if (score > t.best.score * 1.15) t.best = { score, frame: snapshot(frame), lm: d.lm }
    }

    // Retire tracks gone too long.
    for (let i = live.length - 1; i >= 0; i--) if (fi - live[i].last > gapFrames) done.push(...live.splice(i, 1))
    onProgress({ progress: duration > 0 ? Math.min(0.99, timestamp / duration) : 0, people: live.length + done.filter((t) => t.lm.size > 8).length })
    // Let the page breathe now and then.
    if (fi % 6 === 0) await new Promise((r) => setTimeout(r, 0))
  }
  done.push(...live)
  input.dispose()

  if (!times.length) throw new Error("Couldn't read any frames from this video.")
  const T = Float64Array.from(times)
  // Noise: a "face" seen for a handful of frames and nowhere near the camera.
  const tracks = done.filter((t) => t.lm.size >= Math.max(4, Math.round(fps * 0.15)) && t.best.frame)
  const people = groupIntoPeople(tracks, T, W, H)
  for (const p of people) buildFrames(p, T)
  return makeAnalysis(people, T, W, H, duration)
}

/** Joins tracks into people. */
function groupIntoPeople(tracks: Track[], times: Float64Array, W: number, H: number): Person[] {
  const info = tracks.map((t) => ({ t, emb: mean(t.embeds), sig: medianSig(t.sigs) }))
  // Longest first: the best-known faces anchor the people.
  info.sort((a, b) => b.t.lm.size - a.t.lm.size)
  const people: { p: Person; embs: Float32Array[]; sigs: Float32Array[] }[] = []
  for (const it of info) {
    let best: (typeof people)[number] | null = null
    let bestScore = 0
    for (const cand of people) {
      // Never two tracks of one person in the same frame.
      if (cand.p.tracks.some((o) => overlap(o, it.t) > 2)) continue
      const s = sameness(cand, it, times, W, H)
      if (s > bestScore) {
        bestScore = s
        best = cand
      }
    }
    if (best && bestScore >= 0.5) {
      best.p.tracks.push(it.t)
      if (it.emb) best.embs.push(it.emb)
      if (it.sig) best.sigs.push(it.sig)
    } else people.push({ p: { id: 0, tracks: [it.t], frames: [] }, embs: it.emb ? [it.emb] : [], sigs: it.sig ? [it.sig] : [] })
  }
  // Number people by when they first appear, largest first among ties.
  const out = people.map((x) => x.p)
  out.sort((a, b) => Math.min(...a.tracks.map((t) => t.first)) - Math.min(...b.tracks.map((t) => t.first)))
  out.forEach((p, i) => (p.id = i + 1))
  return out
}

function overlap(a: Track, b: Track) {
  const lo = Math.max(a.first, b.first)
  const hi = Math.min(a.last, b.last)
  if (hi < lo) return 0
  let n = 0
  for (let i = lo; i <= hi; i++) if (a.lm.has(i) && b.lm.has(i)) n++
  return n
}

/** 0..1: how sure we are that track `it` is person `cand`. Appearance
 *  carries most of it; face shape backs it up; and a track that starts
 *  right where and when the person's last one ended is very likely them. */
function sameness(cand: { p: Person; embs: Float32Array[]; sigs: Float32Array[] }, it: { t: Track; emb: Float32Array | null; sig: Float32Array | null }, times: Float64Array, W: number, H: number) {
  let s = 0
  let wsum = 0
  if (it.emb && cand.embs.length) {
    const sim = Math.max(...cand.embs.map((e) => cosine(e, it.emb!)))
    // mobilenet embeddings of two crops of one face sit around 0.8-0.95;
    // of two different people around 0.4-0.7.
    s += 3 * Math.min(1, Math.max(0, (sim - 0.62) / 0.2))
    wsum += 3
  }
  if (it.sig && cand.sigs.length) {
    const sig = it.sig
    const d = Math.min(...cand.sigs.map((c) => Math.sqrt(c.reduce((acc, v, i) => acc + (v - sig[i]) ** 2, 0) / c.length)))
    s += 1.5 * Math.min(1, Math.max(0, 1 - d / 0.09))
    wsum += 1.5
  }
  // Continuity: the end of a track of theirs just before this one began.
  let cont = 0
  for (const o of cand.p.tracks) {
    if (o.last >= it.t.first) continue
    const dt = times[it.t.first] - times[o.last]
    if (dt > 2.5) continue
    const a = boxOf(o.lm.get(o.last)!, W, H)
    const b = boxOf(it.t.lm.get(it.t.first)!, W, H)
    const dc = Math.hypot(centreOf(a).x - centreOf(b).x, centreOf(a).y - centreOf(b).y) / Math.max(1, sizeOf(a))
    cont = Math.max(cont, Math.max(0, 1 - dc / 1.5) * Math.max(0, 1 - dt / 2.5))
  }
  s += 2 * cont
  wsum += 2 * (cont > 0 ? 1 : 0.25)
  return wsum ? s / wsum : 0
}

function makeAnalysis(people: Person[], times: Float64Array, W: number, H: number, duration: number): VideoAnalysis {
  const byId = new Map(people.map((p) => [p.id, p]))
  const info = (): PersonInfo[] =>
    people.map((p) => {
      const best = p.tracks.reduce((a, b) => (b.best.score > a.best.score ? b : a)).best
      const spans = spansOf(p, times)
      const on = p.frames.reduce((n, f) => n + (f ? 1 : 0), 0)
      const frame = best.frame!
      return { id: p.id, thumb: faceThumb(frame, best.lm!), spans, best: { frame, landmarks: unpack(best.lm!) }, presence: on / Math.max(1, times.length) }
    })
  let cached = info()

  const indexAt = (t: number) => {
    // Last frame at or before t.
    let lo = 0
    let hi = times.length - 1
    if (t <= times[0]) return 0
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if (times[mid] <= t + 1e-6) lo = mid
      else hi = mid - 1
    }
    return lo
  }
  const at = (p: Person, t: number) => {
    const i = indexAt(t)
    const a = p.frames[i]
    if (!a) {
      // Just before their first frame of a run, show it rather than nothing.
      const b = p.frames[i + 1]
      return b && times[i + 1] - t < 0.05 ? unpack(b) : null
    }
    const b = p.frames[i + 1]
    if (!b || times[i + 1] <= times[i]) return unpack(a)
    const f = Math.min(1, Math.max(0, (t - times[i]) / (times[i + 1] - times[i])))
    const m = new Float32Array(FRAME)
    for (let c = 0; c < FRAME; c++) m[c] = a[c] + (b[c] - a[c]) * f
    return unpack(m)
  }

  const analysis: VideoAnalysis = {
    width: W,
    height: H,
    duration,
    times,
    get people() {
      return cached
    },
    landmarksAt(id, t) {
      const p = byId.get(id)
      return p ? at(p, t) : null
    },
    facesAt(t) {
      const out: { id: number; landmarks: NormalizedLandmark[] }[] = []
      for (const p of people) {
        const lm = at(p, t)
        if (lm) out.push({ id: p.id, landmarks: lm })
      }
      return out
    },
    merge(from, into) {
      const a = byId.get(from)
      const b = byId.get(into)
      if (!a || !b || a === b) return
      b.tracks.push(...a.tracks)
      people.splice(people.indexOf(a), 1)
      byId.delete(from)
      buildFrames(b, times)
      cached = info()
    },
  }
  return analysis
}
