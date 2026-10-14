import type { ImageSegmenter, NormalizedLandmark } from '@mediapipe/tasks-vision'
import { withEyeClosure, type EyeClosure } from '../faceLandmarker'

// What the video analysis, the preview and the export share about faces in
// frames: a face's mesh packed into a flat array, its box, how two boxes
// overlap, windows to look for a face in, and the One Euro filter that
// steadies a mesh through time.

/** Landmarks per face (the 468-point mesh and the irises). */
export const N_LM = 478
export const STRIDE = N_LM * 3
/** Per face per frame: the landmarks (x, y in fractions of the frame, z),
 *  then how shut each eye is (r, l). */
export const FRAME = STRIDE + 2

export interface Box {
  x0: number
  y0: number
  x1: number
  y1: number
}

/** A region of a frame, in fractions of its width and height. */
export interface Region {
  x: number
  y: number
  w: number
  h: number
}

/** The box round a face's mesh, in px of a W × H frame. */
export function boxOf(lm: Float32Array, W: number, H: number): Box {
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

export const area = (b: Box) => Math.max(0, b.x1 - b.x0) * Math.max(0, b.y1 - b.y0)
export function iou(a: Box, b: Box) {
  const i = area({ x0: Math.max(a.x0, b.x0), y0: Math.max(a.y0, b.y0), x1: Math.min(a.x1, b.x1), y1: Math.min(a.y1, b.y1) })
  return i / Math.max(1e-9, area(a) + area(b) - i)
}
export const sizeOf = (b: Box) => Math.max(b.x1 - b.x0, b.y1 - b.y0)
export const centreOf = (b: Box) => ({ x: (b.x0 + b.x1) / 2, y: (b.y0 + b.y1) / 2 })
export const shift = (b: Box, dx: number, dy: number): Box => ({ x0: b.x0 + dx, y0: b.y0 + dy, x1: b.x1 + dx, y1: b.y1 + dy })
export const scaleBox = (b: Box, sx: number, sy: number): Box => ({ x0: b.x0 * sx, y0: b.y0 * sy, x1: b.x1 * sx, y1: b.y1 * sy })

/** How well detection box `d` fits where a face was expected (`pred`):
 *  their overlap, or — for a small fast face that moved off its predicted
 *  box — a little for being close by. 0: not them. */
export function fitScore(pred: Box, d: Box) {
  const o = iou(pred, d)
  if (o > 0.15) return o
  const cp = centreOf(pred)
  const cd = centreOf(d)
  const dist = Math.hypot(cp.x - cd.x, cp.y - cd.y) / Math.max(1, sizeOf(pred))
  const ratio = sizeOf(d) / Math.max(1e-6, sizeOf(pred))
  return dist < 0.5 && ratio > 0.5 && ratio < 2 ? 0.15 * (1 - dist) : 0
}

export function pack(lm: NormalizedLandmark[], eyes: EyeClosure | undefined, roi?: Region): Float32Array {
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

export function unpack(a: Float32Array): NormalizedLandmark[] {
  const out: NormalizedLandmark[] = new Array(N_LM)
  for (let i = 0; i < N_LM; i++) out[i] = { x: a[i * 3], y: a[i * 3 + 1], z: a[i * 3 + 2], visibility: 1 }
  return withEyeClosure(out, { r: a[STRIDE], l: a[STRIDE + 1] })
}

/** A square region (fractions of the frame) of `side` px round (cx, cy),
 *  kept inside a W × H frame. */
export function windowAt(cx: number, cy: number, side: number, W: number, H: number): Region {
  const w = Math.min(1, side / W)
  const h = Math.min(1, side / H)
  return { x: Math.min(1 - w, Math.max(0, cx / W - w / 2)), y: Math.min(1 - h, Math.max(0, cy / H - h / 2)), w, h }
}

let cropCanvas: HTMLCanvasElement | null = null
/** `r` of `src`, at most `max` px on its long side (one shared canvas:
 *  use it before the next call). */
export function cropOf(src: HTMLCanvasElement, r: Region, max = 640) {
  const c = (cropCanvas ??= document.createElement('canvas'))
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

// ---- Is someone still in the seat? ------------------------------------------------
//
// With a fixed camera, a face that can't be found (turned away, a hand over
// it) is usually still there — and its effect should stay. But the person
// may have got up and gone. The person segmenter tells: where their head
// was, it still sees a person — the back of a head, hair, a hand over the
// face are all a person — or the empty room behind. (It used to go by the
// colours under the face, which an arm across the chest, a gesture or a
// turn of the shoulders changed as much as leaving did: the effect went
// whenever a hand came up to the face.) Without the segmenter, that's
// still the fallback: a tiny picture of the area under the face, kept each
// time it's found, compared with the same area now.

let segCanvas: HTMLCanvasElement | null = null
/** How much of `head` (px box of a W × H frame) is a person, 0..1 — looked
 *  at in a window round it, about head and shoulders, the way the
 *  segmenter sees people best. */
export function personIn(seg: ImageSegmenter, frame: HTMLCanvasElement, head: Box): number {
  const W = frame.width
  const H = frame.height
  const c = centreOf(head)
  const r = windowAt(c.x, c.y + sizeOf(head) * 0.35, Math.min(Math.min(W, H), Math.max(sizeOf(head) * 2.6, 64)), W, H)
  const src = cropOf(frame, r, 256)
  const crop = (segCanvas ??= document.createElement('canvas'))
  if (crop.width !== src.width || crop.height !== src.height) {
    crop.width = src.width
    crop.height = src.height
  }
  crop.getContext('2d')!.drawImage(src, 0, 0)
  const res = seg.segment(crop)
  const masks = res.confidenceMasks
  const m = masks && masks[masks.length - 1]?.getAsFloat32Array()
  res.close()
  if (!m) return 1
  // The middle of the head's box, in the crop.
  const cw = crop.width
  const ch = crop.height
  const shrink = 0.18
  const bx0 = ((head.x0 + (head.x1 - head.x0) * shrink) / W - r.x) / r.w
  const bx1 = ((head.x1 - (head.x1 - head.x0) * shrink) / W - r.x) / r.w
  const by0 = ((head.y0 + (head.y1 - head.y0) * shrink) / H - r.y) / r.h
  const by1 = ((head.y1 - (head.y1 - head.y0) * shrink) / H - r.y) / r.h
  const x0 = Math.max(0, Math.floor(bx0 * cw))
  const x1 = Math.min(cw, Math.ceil(bx1 * cw))
  const y0 = Math.max(0, Math.floor(by0 * ch))
  const y1 = Math.min(ch, Math.ceil(by1 * ch))
  let sum = 0
  let n = 0
  for (let y = y0; y < y1; y++)
    for (let x = x0; x < x1; x++) {
      sum += m[y * cw + x]
      n++
    }
  return n ? sum / n : 1
}

const THUMB_W = 32
const THUMB_H = 18
let thumbCanvas: HTMLCanvasElement | null = null
/** The frame shrunk to a few grey values: compared frame to frame, it shows
 *  a cut to another shot. */
export function frameThumb(frame: HTMLCanvasElement): Float32Array {
  const c = (thumbCanvas ??= document.createElement('canvas'))
  c.width = THUMB_W
  c.height = THUMB_H
  const g = c.getContext('2d', { willReadFrequently: true })!
  g.drawImage(frame, 0, 0, THUMB_W, THUMB_H)
  const d = g.getImageData(0, 0, THUMB_W, THUMB_H).data
  const out = new Float32Array(THUMB_W * THUMB_H)
  for (let i = 0; i < out.length; i++) out[i] = (0.3 * d[i * 4] + 0.59 * d[i * 4 + 1] + 0.11 * d[i * 4 + 2]) / 255
  return out
}

/** A cut: most of the picture changed at once (not a hand passing, a head
 *  turning — those change a corner of it). */
export function isCut(a: Float32Array, b: Float32Array) {
  let changed = 0
  for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 0.12) changed++
  return changed / a.length > 0.55
}

/** The area under a face (px box of a W × H frame): shoulders and chest. */
export function bodyBoxOf(face: Box, W: number, H: number): Box | null {
  const w = face.x1 - face.x0
  const h = face.y1 - face.y0
  const b = { x0: Math.max(0, face.x0 - w * 0.5), x1: Math.min(W, face.x1 + w * 0.5), y0: Math.min(H, face.y1 + h * 0.15), y1: Math.min(H, face.y1 + h * 1.6) }
  // (mostly out of the picture — a close-up: nothing to go by)
  return b.y1 - b.y0 < h * 0.4 || b.x1 - b.x0 < w ? null : b
}

const SIG = 6
let sigCanvas: HTMLCanvasElement | null = null
/** That area, shrunk to a few averaged colours. */
export function bodySig(frame: HTMLCanvasElement, b: Box): Float32Array {
  const c = (sigCanvas ??= document.createElement('canvas'))
  c.width = c.height = SIG
  const g = c.getContext('2d', { willReadFrequently: true })!
  g.imageSmoothingQuality = 'high'
  g.drawImage(frame, b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0, 0, 0, SIG, SIG)
  const d = g.getImageData(0, 0, SIG, SIG).data
  const out = new Float32Array(SIG * SIG * 3)
  for (let i = 0; i < SIG * SIG; i++) for (let k = 0; k < 3; k++) out[i * 3 + k] = d[i * 4 + k] / 255
  return out
}

/** Still the same person sitting there (not an empty seat)? */
export function sameBody(a: Float32Array, b: Float32Array) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i])
  return s / a.length < 0.12
}

// ---- Which way a head faces -------------------------------------------------------
//
// When a seated person turns past profile, no face is left to find — and
// the landmark models, asked to look hard, may find one on the back of the
// head (so a nose and whiskers end up on the hair). So a fixed camera keeps
// an eye on each head's turn: a mesh that swings further than a head can
// turn between two frames is a mistake, and so is one that turns up after a
// head was lost, unless a face really is there. And a head lost while
// turning away is taken to have gone on round, out of sight — so the
// effect's front (a nose, glasses) goes behind the head with it, while what
// sits on top (ears, a hat) stays put.

type V3 = [number, number, number]
const pt = (lm: Float32Array, i: number, W: number, H: number): V3 => [lm[i * 3] * W, lm[i * 3 + 1] * H, lm[i * 3 + 2] * W]
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const len = (a: V3) => Math.hypot(a[0], a[1], a[2])
const unit = (a: V3): V3 => {
  const l = Math.max(1e-9, len(a))
  return [a[0] / l, a[1] / l, a[2] / l]
}

/** How far the head is turned, degrees (0 facing the camera; the sign
 *  says which way). From the eye corners' depth. */
export function yawOf(lm: Float32Array, W: number, H: number) {
  const r = sub(pt(lm, 263, W, H), pt(lm, 33, W, H))
  return (Math.atan2(r[2], r[0]) * 180) / Math.PI
}

/** `a − b`, degrees, the short way round. */
const angleDiff = (a: number, b: number) => ((((a - b) % 360) + 540) % 360) - 180

/** The face turned round its own head until its yaw is `yaw`. */
export function turnTo(lm: Float32Array, yaw: number, W: number, H: number): Float32Array {
  const from = yawOf(lm, W, H)
  const deg = angleDiff(yaw, from)
  if (Math.abs(deg) < 0.5) return lm
  const up = unit(sub(pt(lm, 10, W, H), pt(lm, 152, W, H)))
  const side = sub(pt(lm, 454, W, H), pt(lm, 234, W, H))
  let fwd = unit(cross(unit(side), up))
  // (out of the face, toward the camera: nearer is smaller z)
  const nose = pt(lm, 1, W, H)
  const mid: V3 = [(pt(lm, 234, W, H)[0] + pt(lm, 454, W, H)[0]) / 2, (pt(lm, 234, W, H)[1] + pt(lm, 454, W, H)[1]) / 2, (pt(lm, 234, W, H)[2] + pt(lm, 454, W, H)[2]) / 2]
  const toNose = sub(nose, mid)
  if (fwd[0] * toNose[0] + fwd[1] * toNose[1] + fwd[2] * toNose[2] < 0) fwd = [-fwd[0], -fwd[1], -fwd[2]]
  const pivot: V3 = [mid[0] - fwd[0] * len(side) * 0.25, mid[1] - fwd[1] * len(side) * 0.25, mid[2] - fwd[2] * len(side) * 0.25]
  const rotate = (rad: number) => {
    const c = Math.cos(rad)
    const s = Math.sin(rad)
    const out = new Float32Array(lm)
    for (let i = 0; i < N_LM; i++) {
      const v = sub(pt(lm, i, W, H), pivot)
      const kv = cross(up, v)
      const d = up[0] * v[0] + up[1] * v[1] + up[2] * v[2]
      const r: V3 = [v[0] * c + kv[0] * s + up[0] * d * (1 - c), v[1] * c + kv[1] * s + up[1] * d * (1 - c), v[2] * c + kv[2] * s + up[2] * d * (1 - c)]
      out[i * 3] = (r[0] + pivot[0]) / W
      out[i * 3 + 1] = (r[1] + pivot[1]) / H
      out[i * 3 + 2] = (r[2] + pivot[2]) / W
    }
    return out
  }
  // (whichever way round gets there)
  const rad = (deg * Math.PI) / 180
  const a = rotate(rad)
  return Math.abs(angleDiff(yawOf(a, W, H), yaw)) <= Math.abs(deg) / 2 ? a : rotate(-rad)
}

/** A head lost while turned this far is taken to be turning away… */
const AWAY_FROM_DEG = 55
/** …as is one lost turning (from at least TURNING_FROM_DEG) this fast,
 *  deg/s. */
const TURNING_DEG_S = 40
const TURNING_FROM_DEG = 15
/** It goes on round at least this fast, deg/s, to at most MAX_TURN_DEG
 *  (well past profile: the effect's front is behind the head). */
const AWAY_DEG_S = 90
const MAX_TURN_DEG = 120

/** Where a head lost at `yaw`, turning at `rate` deg/s, has got to `dt`
 *  seconds later: on round, away from the camera, if it was turning away
 *  (or already turned well away) — otherwise (a hand over the face, say)
 *  where it was. (Run backwards from where a head turns up again — `rate`
 *  then its turn reversed — it gives where it came back from.) */
export function lostYaw(yaw: number, rate: number, dt: number, max = MAX_TURN_DEG) {
  const s = Math.sign(yaw) || 1
  const a = Math.abs(yaw)
  const away = a >= AWAY_FROM_DEG || (a >= TURNING_FROM_DEG && rate * s >= TURNING_DEG_S)
  if (!away || a >= max) return yaw
  return s * Math.min(max, a + Math.max(AWAY_DEG_S, rate * s) * Math.max(0, dt))
}

/** A fixed camera holds a head turned at most side-on: seated in an
 *  interview, people turn to the side to face whoever they're talking to
 *  far more than they turn their back — and a head shown side-on while
 *  they're facing away looks far less wrong than one facing away while
 *  their profile is in view. */
export const HELD_MAX_DEG = 90

/** A head's yaw over the last moments: how fast it's turning. */
export class Turn {
  private h: { yaw: number; t: number }[] = []

  add(yaw: number, t: number) {
    const last = this.h[this.h.length - 1]
    // (a jump: start afresh)
    if (last && (t <= last.t || t - last.t > 0.25)) this.h = []
    this.h.push({ yaw, t })
    while (this.h.length > 2 && t - this.h[1].t >= 0.3) this.h.shift()
  }

  /** deg/s over the last ~0.3 s (0: too little to go by). */
  get rate() {
    const a = this.h[0]
    const b = this.h[this.h.length - 1]
    return b && b.t - a.t >= 0.1 ? (b.yaw - a.yaw) / (b.t - a.t) : 0
  }
}

/** Can a head turn from `from` to `to` degrees in `dt` seconds? (Fast
 *  as people turn, a whole head doesn't flip in a frame.) */
export const plausibleTurn = (from: number, to: number, dt: number) => Math.abs(to - from) <= 40 + 200 * Math.max(0, dt)

/** Frame-to-frame interpolation of two packed faces. */
export function lerpFace(a: Float32Array, b: Float32Array, f: number, out = new Float32Array(FRAME)) {
  for (let c = 0; c < FRAME; c++) out[c] = a[c] + (b[c] - a[c]) * f
  return out
}

// ---- Video-mode clocks ------------------------------------------------------------
//
// A VIDEO-mode landmarker needs ever-increasing timestamps over its whole
// life — the analysis, then the preview, then each export all feed the same
// one. Each gets a clip's time where it runs on, and a jump ahead wherever
// it starts again.

const clocks = new WeakMap<object, { base: number; last: number }>()
/** The timestamp (ms) to give `landmarker` for a frame at `t` s. */
export function stamp(landmarker: object, t: number) {
  let c = clocks.get(landmarker)
  if (!c) clocks.set(landmarker, (c = { base: 1000, last: 0 }))
  let ms = c.base + t * 1000
  if (ms <= c.last) {
    c.base = c.last + 1000 - t * 1000
    ms = c.last + 1000
  }
  c.last = ms
  return ms
}

// ---- One Euro ---------------------------------------------------------------------
//
// Heavy smoothing while a face is still, light while it moves fast — so a
// still face doesn't shimmer and a turning one doesn't lag. z is noisier
// and matters less, so it's held harder. Eye closure isn't smoothed at all:
// a blink is over in a few frames and smoothing would wash it out.

const MIN_CUTOFF_XY = 1.4
const BETA_XY = 9
const MIN_CUTOFF_Z = 0.8
const BETA_Z = 3
/** `calm` (a fixed camera, people sitting): held harder still at rest —
 *  moving, it follows as fast as ever. */
const CALM = 0.55
const alpha = (cutoff: number, dt: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt))

/** One Euro state over every coordinate of one face. */
export class EuroState {
  readonly value = new Float32Array(FRAME)
  readonly deriv = new Float32Array(STRIDE)
  t = 0
  started = false
  private rest: number

  constructor(calm = false) {
    this.rest = calm ? CALM : 1
  }

  reset() {
    this.started = false
  }

  /** Takes the face `x` at time `t` (seconds; going either way in time, as
   *  long as it keeps going the same way) and returns the filtered face. */
  step(x: Float32Array, t: number): Float32Array {
    const v = this.value
    if (!this.started) {
      v.set(x)
      this.deriv.fill(0)
      this.t = t
      this.started = true
      return v
    }
    const dt = Math.max(1e-3, Math.abs(t - this.t))
    this.t = t
    const ad = alpha(1, dt)
    for (let c = 0; c < STRIDE; c++) {
      const isZ = c % 3 === 2
      const dx = (x[c] - v[c]) / dt
      const d = (this.deriv[c] += ad * (dx - this.deriv[c]))
      const a = alpha((isZ ? MIN_CUTOFF_Z : MIN_CUTOFF_XY) * this.rest + (isZ ? BETA_Z : BETA_XY) * Math.abs(d), dt)
      v[c] += a * (x[c] - v[c])
    }
    v[STRIDE] = x[STRIDE]
    v[STRIDE + 1] = x[STRIDE + 1]
    return v
  }
}
