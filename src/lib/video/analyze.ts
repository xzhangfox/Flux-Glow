import { ImageEmbedder, type FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { ALL_FORMATS, BlobSource, Input, VideoSampleSink } from 'mediabunny'
import { closureFrom, createLandmarker, visionFileset } from '../faceLandmarker'
import { nextTask } from '../schedule'
import { area, boxOf, centreOf, cropOf, fitScore, iou, pack, shift, sizeOf, stamp, unpack, windowAt, type Box, type Region } from './faces'

// Video analysis for the video editor: who is in the clip, and where each
// of them is through it — so each person keeps their own retouch and
// effects. The face meshes themselves aren't kept: the preview and the
// export find them again, frame by frame, at the size they work at (see
// video/follow.ts), using what this pass learned to tell people apart. That
// keeps memory flat however long the clip — a three-hour video is the same
// handful of megabytes per person as a short one.
//
// One pass over the decoded frames (at a working size; a long clip at a
// reduced rate — enough to know who is where), in order:
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
//
// What's kept per person: their box at each analysed frame (24 bytes), a
// picture of their clearest view, and the spans they're on screen.

const WORK_MAX = 960
const MAX_FACES = 6
/** Most of a second's absence still continues a track. */
const TRACK_GAP_S = 0.8
/** Gaps up to this long inside a person's presence are bridged. */
export const FILL_GAP_S = 0.6
/** The tile scan for small faces, every this many analysed frames at most
 *  — and at least twice a second, whatever the rate. */
const TILE_EVERY = 10
const TILE_EVERY_S = 0.5
const EMBED_EVERY = 12
const EMBED_EVERY_S = 0.8
/** Shape signatures kept per track (a sample: they go into a median). */
const MAX_SIGS = 64
/** Appearances kept per person to compare new tracks with. */
const MAX_LOOKS = 48
/** Analysis rate, frames a second: every frame of a short clip (up to
 *  30 fps), fewer for a long one — identity needs far less than the
 *  per-frame meshes, which are found again when they're needed. */
const RATE_SHORT = 30
const RATE_LONG = 15
const LONG_CLIP_S = 10 * 60
/** Videos up to this long can be edited. */
export const MAX_DURATION_S = 4 * 60 * 60
/** A fixed camera (an interview, a talk show): people stay in their seats,
 *  so far fewer analysed frames tell who is where. */
const RATE_FIXED = 8
/** …and someone in their seat is still there across a gap this long (their
 *  face turned away, a hand over it). */
const FIXED_BRIDGE_S = 4
/** The clearest view of each face is kept as a crop this big (px). */
const BEST_PX = 288

const EMBEDDER_URL = '/mediapipe/image_embedder.tflite'

export interface PersonInfo {
  id: number
  /** A face picture (data URL) for the people strip. */
  thumb: string
  /** Seconds in the clip where this person is on screen. */
  spans: [number, number][]
  /** Their clearest view (a crop round their face) and their landmarks
   *  in it (for effect and look previews). */
  best: { frame: HTMLCanvasElement; landmarks: NormalizedLandmark[] }
  /** Share of the clip they're in (0..1). */
  presence: number
}

export interface VideoAnalysis {
  /** The working size the clip was analysed at. */
  width: number
  height: number
  duration: number
  /** The clip's frame rate (average). */
  fps: number
  /** Seconds between analysed frames. */
  interval: number
  /** Analysed as a fixed camera (people in fixed seats). */
  fixed: boolean
  people: PersonInfo[]
  /** Where each person on screen at `t` is (their face's box, in fractions
   *  of the frame), from the analysed frames either side. */
  boxesAt(t: number): { id: number; box: Box }[]
  /** Joins `from` into `into` (the user says they're the same person). */
  merge(from: number, into: number): void
}

export interface AnalyzeProgress {
  /** 0..1 */
  progress: number
  people: number
}

// ---- Models ------------------------------------------------------------------------

let models: Promise<{ video: FaceLandmarker; image: FaceLandmarker; lenient: FaceLandmarker; embedder: ImageEmbedder | null }> | null = null
let delegate: 'GPU' | 'CPU' = 'GPU'
/** Dev tests only (a machine whose GPU is software): run the models on the
 *  CPU. Before anything loads them. */
export const runModelsOnCpu = () => void (delegate = 'CPU')
/** The video editor's models: a landmarker tracking faces through a clip,
 *  one looking at single frames and crops, and the identity embedder. */
export function loadVideoModels() {
  models ??= (async () => {
    // (`lenient`: for a seat someone is known to be in — a face turned half
    // away or partly covered still counts there)
    const [video, image, lenient] = await Promise.all([createLandmarker('VIDEO', MAX_FACES, delegate), createLandmarker('IMAGE', MAX_FACES, delegate), createLandmarker('IMAGE', 2, delegate, 0.2)])
    let embedder: ImageEmbedder | null = null
    try {
      embedder = await ImageEmbedder.createFromOptions(await visionFileset(), {
        baseOptions: { modelAssetPath: EMBEDDER_URL, delegate },
        runningMode: 'IMAGE',
        l2Normalize: true,
        quantize: false,
      })
    } catch {
      // Without it, identity rests on face shape and on position over time.
      embedder = null
    }
    return { video, image, lenient, embedder }
  })()
  return models
}

/** Warm the models up ahead of time (e.g. when the editor opens). */
export const preloadVideoModels = () => loadVideoModels().then(() => undefined)

// ---- Identity cues ------------------------------------------------------------------

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

let faceCanvas: HTMLCanvasElement | null = null
/** The face upright (eyes level), roughly a head and shoulders wide: hair
 *  and outline say as much about who someone is as the features do. */
function alignedFace(frame: HTMLCanvasElement, lm: Float32Array) {
  const W = frame.width
  const H = frame.height
  const c = (faceCanvas ??= document.createElement('canvas'))
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

/** A square crop round the face (head and some shoulder), and the face's
 *  landmarks in it. */
function bestView(frame: HTMLCanvasElement, lm: Float32Array, into: HTMLCanvasElement | null) {
  const W = frame.width
  const H = frame.height
  const b = boxOf(lm, W, H)
  const side = Math.max(8, sizeOf(b) * 2.3)
  const c0 = centreOf(b)
  const x0 = c0.x - side / 2
  const y0 = c0.y - side / 2 - side * 0.04
  const px = Math.min(BEST_PX, Math.round(side))
  const c = into ?? document.createElement('canvas')
  c.width = c.height = px
  const g = c.getContext('2d')!
  g.fillStyle = '#3a3d42'
  g.fillRect(0, 0, px, px)
  g.drawImage(frame, x0, y0, side, side, 0, 0, px, px)
  const out = new Float32Array(lm.length)
  out.set(lm)
  for (let i = 0; i < 478; i++) {
    out[i * 3] = (lm[i * 3] * W - x0) / side
    out[i * 3 + 1] = (lm[i * 3 + 1] * H - y0) / side
    out[i * 3 + 2] = (lm[i * 3 + 2] * W) / side
  }
  return { canvas: c, lm: out }
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

// ---- Timelines ---------------------------------------------------------------------

/** Where a face is, frame after frame: analysed-frame indices and boxes
 *  (fractions of the frame), in growing typed arrays. */
class Timeline {
  idx = new Int32Array(64)
  box = new Float32Array(256)
  n = 0

  push(i: number, b: Box) {
    if (this.n === this.idx.length) {
      const idx = new Int32Array(this.n * 2)
      idx.set(this.idx)
      this.idx = idx
      const box = new Float32Array(this.n * 8)
      box.set(this.box)
      this.box = box
    }
    this.idx[this.n] = i
    this.box.set([b.x0, b.y0, b.x1, b.y1], this.n * 4)
    this.n++
  }

  boxAt(k: number): Box {
    const o = k * 4
    return { x0: this.box[o], y0: this.box[o + 1], x1: this.box[o + 2], y1: this.box[o + 3] }
  }

  get first() {
    return this.idx[0]
  }
  get last() {
    return this.idx[this.n - 1]
  }

  /** Frames both timelines have. */
  shared(o: Timeline) {
    if (this.last < o.first || o.last < this.first) return 0
    let i = 0
    let j = 0
    let n = 0
    while (i < this.n && j < o.n) {
      const a = this.idx[i]
      const b = o.idx[j]
      if (a === b) {
        n++
        i++
        j++
      } else if (a < b) i++
      else j++
    }
    return n
  }

  static merge(parts: Timeline[]): Timeline {
    const out = new Timeline()
    parts = parts.filter((p) => p.n).sort((a, b) => a.first - b.first)
    // One after another (a person's tracks never share a frame, as a rule):
    // end to end. A long clip has thousands of them.
    if (parts.every((p, i) => i === 0 || p.first > parts[i - 1].last)) {
      for (const p of parts) for (let k = 0; k < p.n; k++) out.push(p.idx[k], p.boxAt(k))
      return out
    }
    const at = parts.map(() => 0)
    for (;;) {
      let pick = -1
      for (let p = 0; p < parts.length; p++) if (at[p] < parts[p].n && (pick < 0 || parts[p].idx[at[p]] < parts[pick].idx[at[pick]])) pick = p
      if (pick < 0) break
      const k = at[pick]++
      // (one frame twice — two merged people both there: keep the first)
      if (out.n && out.idx[out.n - 1] === parts[pick].idx[k]) continue
      out.push(parts[pick].idx[k], parts[pick].boxAt(k))
    }
    return out
  }
}

// ---- Tracks ------------------------------------------------------------------------

interface Track {
  id: number
  tl: Timeline
  /** Last box, px of the working frame, and its velocity per analysed frame. */
  box: Box
  vx: number
  vy: number
  embSum: Float32Array | null
  embN: number
  sigs: Float32Array[]
  sigSeen: number
  best: { score: number; canvas: HTMLCanvasElement | null; url: string | null; lm: Float32Array | null }
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

// ---- The pass --------------------------------------------------------------------

/** Analyses `file`. Rejects with an AbortError if `signal` aborts. */
export async function analyzeVideo(file: Blob, onProgress: (p: AnalyzeProgress) => void, signal?: AbortSignal, opts: { fixed?: boolean } = {}): Promise<VideoAnalysis> {
  const fixed = !!opts.fixed
  const { video: vLm, image: iLm, embedder } = await loadVideoModels()
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
  try {
    const track = await input.getPrimaryVideoTrack()
    if (!track) throw new Error('This file has no video.')
    if (!(await track.canDecode())) throw new Error("This browser can't decode this video's format.")
    const duration = await input.computeDuration()
    if (duration > MAX_DURATION_S + 1) throw new Error(`Videos up to ${MAX_DURATION_S / 3600} hours can be edited.`)
    const dw = track.displayWidth
    const dh = track.displayHeight
    const k = Math.min(1, WORK_MAX / Math.max(dw, dh))
    const W = Math.max(2, Math.round(dw * k))
    const H = Math.max(2, Math.round(dh * k))
    const fps = (await track.computePacketStats(90).catch(() => null))?.averagePacketRate || 30
    const step = Math.max(1, Math.round(fps / (fixed ? RATE_FIXED : duration > LONG_CLIP_S ? RATE_LONG : RATE_SHORT)))
    const rate = fps / step
    const gapFrames = Math.max(3, Math.round(TRACK_GAP_S * rate))
    const tileEvery = Math.max(1, Math.min(TILE_EVERY, Math.round(TILE_EVERY_S * rate)))
    const embedEvery = Math.max(1, Math.min(EMBED_EVERY, Math.round(EMBED_EVERY_S * rate)))
    const minFrames = Math.max(3, Math.round(rate * 0.15))

    const frame = document.createElement('canvas')
    frame.width = W
    frame.height = H
    const fctx = frame.getContext('2d')!
    const times: number[] = []
    const live: Track[] = []
    const done: Track[] = []
    let nextId = 1
    let lastTs = -Infinity
    let srcIdx = 0
    // (a fixed sequence: the same clip is always analysed the same way)
    let seed = 0x9e3779b9
    const random = () => {
      seed ^= seed << 13
      seed ^= seed >>> 17
      seed ^= seed << 5
      return (seed >>> 0) / 4294967296
    }

    const detectIn = (r: Region): Det[] => {
      const res = iLm.detect(cropOf(frame, r))
      return res.faceLandmarks.map((lm, i) => {
        const a = pack(lm, closureFrom(res.faceBlendshapes, i), r)
        return { lm: a, box: boxOf(a, W, H) }
      })
    }
    // A track that's ended: the noise (a "face" for a handful of frames)
    // goes; the rest keep their best view as a small JPEG, not a canvas.
    // A running count of people for the progress line: a track that ends
    // looking like someone already counted (someone who left and came
    // back) isn't counted again. (Only a guess; who is who is settled
    // after the pass.)
    const counted: Float32Array[] = []
    const knownFace = (t: Track) => !!t.embSum && counted.some((c) => cosine(c, t.embSum!) > 0.82)
    let faceless = 0
    const retire = (t: Track) => {
      if (t.tl.n < minFrames || !t.best.canvas) return
      t.best.url = t.best.canvas.toDataURL('image/jpeg', 0.9)
      t.best.canvas = null
      done.push(t)
      if (t.embSum && !knownFace(t)) counted.push(t.embSum)
      if (!t.embSum) faceless++
    }

    const sink = new VideoSampleSink(track)
    let breathed = performance.now()
    for await (const sample of sink.samples()) {
      let use = false
      let ts = 0
      try {
        if (signal?.aborted) throw new DOMException('Canceled', 'AbortError')
        ts = sample.timestamp
        if (ts <= lastTs) continue
        lastTs = ts
        use = srcIdx++ % step === 0
        if (use) sample.draw(fctx, 0, 0, W, H)
      } finally {
        sample.close()
      }
      if (!use) continue
      const fi = times.length
      times.push(ts)

      // 1. Whole frame.
      const vres = vLm.detectForVideo(frame, stamp(vLm, ts))
      let dets: Det[] = vres.faceLandmarks.map((lm, i) => {
        const a = pack(lm, closureFrom(vres.faceBlendshapes, i))
        return { lm: a, box: boxOf(a, W, H) }
      })
      // Small faces: tiles, now and then (and on the first frame).
      if (fi % tileEvery === 0) {
        const m = Math.min(W, H)
        const side = m * 0.62
        const tiles = [
          windowAt(W * 0.3, H * 0.3, side, W, H),
          windowAt(W * 0.7, H * 0.3, side, W, H),
          windowAt(W * 0.3, H * 0.7, side, W, H),
          windowAt(W * 0.7, H * 0.7, side, W, H),
          windowAt(W * 0.5, H * 0.45, side, W, H),
        ]
        for (const r of tiles) dets.push(...detectIn(r))
      }
      dets = dedupe(dets)

      // 2. Associate with live tracks (best fit first).
      const pairs: [number, number, number][] = []
      live.forEach((t, ti) => {
        const pred = shift(t.box, t.vx * (fi - t.tl.last), t.vy * (fi - t.tl.last))
        dets.forEach((d, di) => {
          const s = fitScore(pred, d.box)
          if (s > 0) pairs.push([s, ti, di])
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
        if (tUsed.has(ti) || fi - t.tl.last > gapFrames) return
        const pred = shift(t.box, t.vx * (fi - t.tl.last), t.vy * (fi - t.tl.last))
        const c = centreOf(pred)
        const r = windowAt(c.x, c.y, Math.min(Math.min(W, H), Math.max(sizeOf(pred) * 2.6, 96)), W, H)
        const found = detectIn(r).filter((d) => !dets.some((o) => iou(o.box, d.box) > 0.35))
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
        const dtf = Math.max(1, fi - t.tl.last)
        const c0 = centreOf(t.box)
        const c1 = centreOf(d.box)
        // Velocity per frame, eased so one jumpy detection doesn't fling it.
        t.vx = t.vx * 0.5 + ((c1.x - c0.x) / dtf) * 0.5
        t.vy = t.vy * 0.5 + ((c1.y - c0.y) / dtf) * 0.5
        t.box = d.box
        t.tl.push(fi, { x0: d.box.x0 / W, y0: d.box.y0 / H, x1: d.box.x1 / W, y1: d.box.y1 / H })
      }
      // New tracks for the rest.
      dets.forEach((d, di) => {
        if (dUsed.has(di) || matched.some(([, m]) => m === d)) return
        if (sizeOf(d.box) < 14) return
        const tl = new Timeline()
        tl.push(fi, { x0: d.box.x0 / W, y0: d.box.y0 / H, x1: d.box.x1 / W, y1: d.box.y1 / H })
        const t: Track = { id: nextId++, tl, box: d.box, vx: 0, vy: 0, embSum: null, embN: 0, sigs: [], sigSeen: 0, best: { score: 0, canvas: null, url: null, lm: null } }
        live.push(t)
        matched.push([t, d])
      })

      // 3. Appearance, shape and best view, for tracks seen this frame.
      for (const [t, d] of matched) {
        const fr = frontal(d.lm)
        const sz = sizeOf(d.box)
        if (fr > 0.55) {
          // (a fair sample of them, however long the track)
          const s = signature(d.lm)
          const seen = t.sigSeen++
          if (t.sigs.length < MAX_SIGS) t.sigs.push(s)
          else {
            const j = Math.floor(random() * (seen + 1))
            if (j < MAX_SIGS) t.sigs[j] = s
          }
        }
        if (embedder && fr > 0.45 && sz > 28 && (t.embN === 0 || (fi - t.tl.first) % embedEvery === 0)) {
          const e = embedder.embed(alignedFace(frame, d.lm)).embeddings[0]?.floatEmbedding
          if (e) {
            t.embSum ??= new Float32Array(e.length)
            for (let i = 0; i < e.length; i++) t.embSum[i] += e[i]
            t.embN++
          }
        }
        const score = sz * (0.35 + fr)
        if (score > t.best.score * 1.15) {
          const v = bestView(frame, d.lm, t.best.canvas)
          t.best = { score, canvas: v.canvas, url: null, lm: v.lm }
        }
      }

      // Retire tracks gone too long.
      for (let i = live.length - 1; i >= 0; i--) if (fi - live[i].tl.last > gapFrames) retire(live.splice(i, 1)[0])
      if (fi % 15 === 0) onProgress({ progress: duration > 0 ? Math.min(0.99, ts / duration) : 0, people: faceless + counted.length + live.filter((t) => t.tl.n >= minFrames && !knownFace(t)).length })
      // Let the page breathe now and then (a task, not a timer: timers in a
      // hidden tab are slowed to a crawl).
      if (performance.now() - breathed > 50) {
        await nextTask()
        breathed = performance.now()
      }
    }
    for (const t of live.splice(0)) retire(t)

    if (!times.length) throw new Error("Couldn't read any frames from this video.")
    const T = Float64Array.from(times)
    const people = groupIntoPeople(done, T, W, H, fixed)
    return await makeAnalysis(people, T, W, H, duration, fps, step / fps, tileEvery, fixed)
  } finally {
    input.dispose()
  }
}

// ---- People -----------------------------------------------------------------------

interface Person {
  id: number
  tracks: Track[]
  /** All their tracks' boxes, in frame order. */
  tl: Timeline
}

/** Where a face sits through its frames: the middle of its box, and its
 *  size (px of the working frame; medians, so a lean or a stretch doesn't
 *  move it). */
interface Seat {
  x: number
  y: number
  size: number
}
function seatOf(tl: Timeline, W: number, H: number): Seat {
  const xs: number[] = []
  const ys: number[] = []
  const ss: number[] = []
  const k = Math.max(1, Math.floor(tl.n / 200))
  for (let i = 0; i < tl.n; i += k) {
    const b = tl.boxAt(i)
    xs.push(((b.x0 + b.x1) / 2) * W)
    ys.push(((b.y0 + b.y1) / 2) * H)
    ss.push(Math.max((b.x1 - b.x0) * W, (b.y1 - b.y0) * H))
  }
  const med = (v: number[]) => v.sort((a, b) => a - b)[v.length >> 1]
  return { x: med(xs), y: med(ys), size: med(ss) }
}
/** 0..1: how surely two seats are the same place (and face size). */
function sameSeat(a: Seat, b: Seat) {
  const d = Math.hypot(a.x - b.x, a.y - b.y) / Math.max(1, a.size, b.size)
  const r = Math.max(a.size, b.size) / Math.max(1, Math.min(a.size, b.size))
  return Math.max(0, 1 - d / 1.1) * Math.max(0, Math.min(1, (2 - r) / 0.6))
}

interface Group {
  p: Person
  embs: Float32Array[]
  sigs: Float32Array[]
  seats: Seat[]
}
type TrackInfo = { t: Track; emb: Float32Array | null; sig: Float32Array | null; seat: Seat }

/** Joins tracks into people. A fixed camera adds where each one sits: a
 *  track in someone's seat is very likely them — and then there are only
 *  as many people as there are seats in use. */
function groupIntoPeople(tracks: Track[], times: Float64Array, W: number, H: number, fixed: boolean): Person[] {
  const info: TrackInfo[] = tracks.map((t) => ({ t, emb: t.embSum && t.embN ? t.embSum.map((v) => v / t.embN) : null, sig: medianSig(t.sigs), seat: seatOf(t.tl, W, H) }))
  // Longest first: the best-known faces anchor the people.
  info.sort((a, b) => b.t.tl.n - a.t.tl.n)
  const groups: Group[] = []
  for (const it of info) {
    let best: Group | null = null
    let bestScore = 0
    for (const cand of groups) {
      // Never two tracks of one person in the same frame.
      if (cand.p.tracks.some((o) => o.tl.shared(it.t.tl) > 2)) continue
      const s = sameness(cand, it, times, W, H, fixed)
      if (s > bestScore) {
        bestScore = s
        best = cand
      }
    }
    if (best && bestScore >= (fixed ? 0.45 : 0.5)) join(best, it)
    else groups.push({ p: { id: 0, tracks: [it.t], tl: it.t.tl }, embs: it.emb ? [it.emb] : [], sigs: it.sig ? [it.sig] : [], seats: [it.seat] })
  }
  for (const g of groups) g.p.tl = Timeline.merge(g.p.tracks.map((t) => t.tl))
  if (fixed) consolidate(groups, times.length)
  // Number people by when they first appear.
  const out = groups.map((x) => x.p)
  out.sort((a, b) => a.tl.first - b.tl.first)
  out.forEach((p, i) => (p.id = i + 1))
  return out
}

function join(g: Group, it: { t: Track; emb: Float32Array | null; sig: Float32Array | null; seat?: Seat }) {
  g.p.tracks.push(it.t)
  // (a sample is enough: tracks come longest first, so these are the
  // clearest looks at them)
  if (it.emb && g.embs.length < MAX_LOOKS) g.embs.push(it.emb)
  if (it.sig && g.sigs.length < MAX_LOOKS) g.sigs.push(it.sig)
  if (it.seat && g.seats.length < MAX_LOOKS) g.seats.push(it.seat)
}

/** A fixed camera: there are as many people as faces on screen together
 *  (most of the time — a few frames with an extra face don't count). Anyone
 *  beyond that who never shares a frame with someone else is folded into
 *  whoever they most likely are: the same seat, or the same look. */
function consolidate(groups: Group[], frames: number) {
  const together = new Uint8Array(frames)
  for (const g of groups) for (let k = 0; k < g.p.tl.n; k++) together[g.p.tl.idx[k]]++
  const atLeast = new Array(MAX_FACES + 2).fill(0)
  for (const n of together) for (let c = 1; c <= Math.min(n, MAX_FACES + 1); c++) atLeast[c]++
  let seats = 1
  for (let c = 2; c <= MAX_FACES + 1; c++) if (atLeast[c] >= frames * 0.03) seats = c
  for (;;) {
    if (groups.length <= seats) return
    // The least seen first.
    const order = [...groups].sort((a, b) => a.p.tl.n - b.p.tl.n)
    let merged = false
    for (const g of order) {
      let best: Group | null = null
      let bestScore = 0.15
      for (const o of groups) {
        if (o === g || o.p.tl.shared(g.p.tl) > Math.max(2, g.p.tl.n * 0.01)) continue
        const seat = Math.max(...o.seats.map((a) => Math.max(...g.seats.map((b) => sameSeat(a, b)))))
        const look = g.embs.length && o.embs.length ? Math.max(...g.embs.map((a) => Math.max(...o.embs.map((b) => cosine(a, b))))) : 0
        const score = 0.65 * seat + 0.35 * Math.min(1, Math.max(0, (look - 0.5) / 0.3))
        if (score > bestScore) {
          bestScore = score
          best = o
        }
      }
      if (!best) continue
      for (const t of g.p.tracks) join(best, { t, emb: null, sig: null })
      best.embs.push(...g.embs.slice(0, Math.max(0, MAX_LOOKS - best.embs.length)))
      best.seats.push(...g.seats.slice(0, Math.max(0, MAX_LOOKS - best.seats.length)))
      best.p.tl = Timeline.merge([best.p.tl, g.p.tl])
      groups.splice(groups.indexOf(g), 1)
      merged = true
      break
    }
    if (!merged) return
  }
}

/** 0..1: how sure we are that track `it` is person `cand`. Appearance
 *  carries most of it; face shape backs it up; a track that starts right
 *  where and when the person's last one ended is very likely them; and
 *  with a fixed camera, so is one in their seat. */
function sameness(cand: Group, it: TrackInfo, times: Float64Array, W: number, H: number, fixed: boolean) {
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
  if (fixed) {
    s += 3 * Math.max(...cand.seats.map((c) => sameSeat(c, it.seat)))
    wsum += 3
  }
  // Continuity: the end of a track of theirs just before this one began.
  let cont = 0
  const px = (b: Box): Box => ({ x0: b.x0 * W, y0: b.y0 * H, x1: b.x1 * W, y1: b.y1 * H })
  for (const o of cand.p.tracks) {
    if (o.tl.last >= it.t.tl.first) continue
    const dt = times[it.t.tl.first] - times[o.tl.last]
    if (dt > 2.5) continue
    const a = px(o.tl.boxAt(o.tl.n - 1))
    const b = px(it.t.tl.boxAt(0))
    const dc = Math.hypot(centreOf(a).x - centreOf(b).x, centreOf(a).y - centreOf(b).y) / Math.max(1, sizeOf(a))
    cont = Math.max(cont, Math.max(0, 1 - dc / 1.5) * Math.max(0, 1 - dt / 2.5))
  }
  s += 2 * cont
  wsum += 2 * (cont > 0 ? 1 : 0.25)
  return wsum ? s / wsum : 0
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

function loadCanvas(url: string): Promise<HTMLCanvasElement> {
  return new Promise((resolve, reject) => {
    const im = new Image()
    im.onload = () => {
      const c = document.createElement('canvas')
      c.width = im.naturalWidth
      c.height = im.naturalHeight
      c.getContext('2d')!.drawImage(im, 0, 0)
      resolve(c)
    }
    im.onerror = () => reject(new Error('Bad face picture'))
    im.src = url
  })
}

async function makeAnalysis(people: Person[], times: Float64Array, W: number, H: number, duration: number, fps: number, interval: number, tileEvery: number, fixed: boolean): Promise<VideoAnalysis> {
  const byId = new Map(people.map((p) => [p.id, p]))
  // A person is on screen this far either side of an analysed frame they're
  // in, and across gaps up to FILL_GAP_S. And they're looked for a little
  // before they were first seen: a small face is only found by the tile
  // scan, every `tileEvery` analysed frames, so it may have been there for
  // up to that long already.
  const reach = interval * 0.75
  const lead = Math.max(reach, tileEvery * interval)
  const bridge = fixed ? FIXED_BRIDGE_S : FILL_GAP_S + interval

  const bestOf = (p: Person) => p.tracks.reduce((a, b) => (b.best.score > a.best.score ? b : a))
  // Each person's clearest view, back from its JPEG. (Merging two people
  // keeps the better of their two, so these are all that's ever needed.)
  const views = new Map<Track, HTMLCanvasElement>()
  await Promise.all(people.map(async (p) => views.set(bestOf(p), await loadCanvas(bestOf(p).best.url!))))
  const spansOf = (p: Person): [number, number][] => {
    const spans: [number, number][] = []
    for (let k = 0; k < p.tl.n; k++) {
      const t = times[p.tl.idx[k]]
      const last = spans[spans.length - 1]
      if (last && t - last[1] <= bridge) last[1] = t
      else spans.push([t, t])
    }
    return spans.map(([a, b]) => [Math.max(0, a - reach), Math.min(duration, b + reach)])
  }
  const thumbs = new Map<Track, string>()
  const info = (): PersonInfo[] =>
    people.map((p) => {
      const best = bestOf(p)
      const frame = views.get(best)!
      let thumb = thumbs.get(best)
      if (!thumb) thumbs.set(best, (thumb = faceThumb(frame, best.best.lm!)))
      const spans = spansOf(p)
      const on = spans.reduce((n, [a, b]) => n + (b - a), 0)
      return { id: p.id, thumb, spans, best: { frame, landmarks: unpack(best.best.lm!) }, presence: Math.min(1, on / Math.max(1e-3, duration)) }
    })
  let cached = info()

  /** Index into `p.tl` of the last analysed frame at or before `t` (-1: none). */
  const lastAt = (p: Person, t: number) => {
    let lo = 0
    let hi = p.tl.n - 1
    if (hi < 0 || times[p.tl.idx[0]] > t + 1e-6) return -1
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if (times[p.tl.idx[mid]] <= t + 1e-6) lo = mid
      else hi = mid - 1
    }
    return lo
  }
  const boxAt = (p: Person, t: number): Box | null => {
    const i = lastAt(p, t)
    const ta = i >= 0 ? times[p.tl.idx[i]] : -Infinity
    const tb = i + 1 < p.tl.n ? times[p.tl.idx[i + 1]] : Infinity
    if (i >= 0 && i + 1 < p.tl.n && tb - ta <= bridge) {
      const a = p.tl.boxAt(i)
      const b = p.tl.boxAt(i + 1)
      const f = Math.min(1, Math.max(0, (t - ta) / Math.max(1e-6, tb - ta)))
      return { x0: a.x0 + (b.x0 - a.x0) * f, y0: a.y0 + (b.y0 - a.y0) * f, x1: a.x1 + (b.x1 - a.x1) * f, y1: a.y1 + (b.y1 - a.y1) * f }
    }
    if (i >= 0 && t - ta <= reach) return p.tl.boxAt(i)
    if (i + 1 < p.tl.n && tb - t <= lead) return p.tl.boxAt(i + 1)
    return null
  }

  return {
    width: W,
    height: H,
    duration,
    fps,
    interval,
    fixed,
    get people() {
      return cached
    },
    boxesAt(t) {
      // (a player paused at the very end reports the clip's end, a frame
      // past the last frame's start)
      t = Math.min(t, times[times.length - 1])
      const out: { id: number; box: Box }[] = []
      for (const p of people) {
        const box = boxAt(p, t)
        if (box) out.push({ id: p.id, box })
      }
      return out
    },
    merge(from, into) {
      const a = byId.get(from)
      const b = byId.get(into)
      if (!a || !b || a === b) return
      b.tracks.push(...a.tracks)
      b.tl = Timeline.merge([b.tl, a.tl])
      people.splice(people.indexOf(a), 1)
      byId.delete(from)
      cached = info()
    },
  }
}
