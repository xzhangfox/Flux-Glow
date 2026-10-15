import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { eyeClosureOf, withEyeClosure } from './faceLandmarker'

// The live preview shows every frame the camera gives, at the camera's own
// rate, while the face tracker (in its worker) keeps up as it can — on a
// slower phone it finishes every other frame or so. Each frame is drawn
// with the newest tracked face, moved on to where it will have got by then,
// and is retouched at a size the phone keeps up with.

/** A tracking result: the face's landmarks in the camera frame (null when
 *  there's none), and when that frame was taken (performance.now()). */
export interface Tracked {
  lm: NormalizedLandmark[] | null
  at: number
}

/** How far ahead of the newest tracked frame the face may be carried. */
const LEAD_MAX_MS = 70
/** Carry a little less than the full motion: overshooting a face that has
 *  just stopped shows more than lagging one that's moving. */
const DAMP = 0.75

/** The face at time `t`: the newest landmarks, moved on by the whole face's
 *  motion between the last two results (as one: carrying each point on by
 *  its own motion would magnify the tracker's jitter), at most
 *  LEAD_MAX_MS ahead. */
export function landmarksAt(newest: Tracked | null, before: Tracked | null, t: number): NormalizedLandmark[] | null {
  const lm = newest?.lm
  if (!newest || !lm) return null
  const prev = before?.lm
  const span = newest.at - (before?.at ?? 0)
  const lead = Math.min(LEAD_MAX_MS, t - newest.at)
  if (!prev || prev === lm || prev.length !== lm.length || span <= 0 || span > 250 || lead <= 0) return lm
  let dx = 0
  let dy = 0
  for (let i = 0; i < lm.length; i++) {
    dx += lm[i].x - prev[i].x
    dy += lm[i].y - prev[i].y
  }
  const k = (DAMP * lead) / span / lm.length
  dx *= k
  dy *= k
  // (a jump — the face found again elsewhere, or misread — isn't motion)
  if (Math.hypot(dx, dy) > 0.06) return lm
  return withEyeClosure(
    lm.map((p) => ({ ...p, x: p.x + dx, y: p.y + dy })),
    eyeClosureOf(lm),
  )
}

/** The live frame's working size, steps down from full sharpness. */
const SIZES = [1280, 1120, 960, 840, 720]

/** Picks the working size the phone can retouch a frame in, within the
 *  camera's frame interval: one step smaller as soon as frames run long,
 *  one step back up after a few seconds with plenty to spare. */
export class LiveSize {
  private i = 0
  private avg = 0
  private easy = 0

  get size(): number {
    return SIZES[this.i]
  }

  /** After each frame: how long it took (ms), and the camera's interval. */
  note(ms: number, interval: number): void {
    this.avg = this.avg ? this.avg * 0.8 + ms * 0.2 : ms
    if (this.avg > interval * 0.92) {
      if (this.i < SIZES.length - 1) {
        this.i++
        this.avg = 0
      }
      this.easy = 0
    } else if (this.avg < interval * 0.5) {
      if (++this.easy > 120 && this.i > 0) {
        this.i--
        this.avg = 0
        this.easy = 0
      }
    } else this.easy = 0
  }
}
