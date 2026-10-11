import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
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
