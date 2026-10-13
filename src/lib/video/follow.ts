import type { FaceLandmarker } from '@mediapipe/tasks-vision'
import { ALL_FORMATS, BlobSource, CanvasSink, Input } from 'mediabunny'
import { closureFrom } from '../faceLandmarker'
import { FILL_GAP_S, loadVideoModels, type VideoAnalysis } from './analyze'
import { EuroState, FRAME, STRIDE, Turn, bodyBoxOf, bodySig, boxOf, centreOf, cropOf, fitScore, iou, lerpFace, lostYaw, pack, plausibleTurn, sameBody, scaleBox, shift, sizeOf, stamp, turnTo, windowAt, yawOf, type Box } from './faces'

// Finding the analysed people in a frame, as it's shown or exported: the
// landmarker runs on the frame itself, and each face it finds is put to the
// person whose place it fits — where the analysis saw them about then, and
// (frame to frame) where they just were. Someone the whole-frame pass misses
// (small, half turned) is looked for again in a window round where they
// should be. So each frame's meshes are exact for that frame, however long
// the clip, and nothing about them has to be kept.
//
// The export reads the clip ahead of itself (FaceScout) to steady each mesh
// with what comes next as well as what came before — zero lag, as if the
// whole clip were known — and to bridge a face lost for a moment.

export interface Found {
  id: number
  /** Packed landmarks (see faces.ts), in fractions of the frame. */
  lm: Float32Array
}

/** Windows looked through for missing people, per frame, at most. */
const MAX_RETRIES = 4
const MAX_FACES = 6
/** How long a person's own last place still guides where they are now. */
const RECENT_S = 0.3
/** A fixed camera: someone still in their seat whose face can't be found
 *  (turned away, a hand over it) keeps their last mesh this long, so their
 *  effect stays on rather than blinking off. */
const HOLD_S = 6
/** …and the export bridges gaps up to this long (as it does 0.6 s for a
 *  moving camera): their effect rides through a turned head. */
const FIXED_FILL_S = 6
/** Someone who doesn't come back within that keeps it this long more. */
const FIXED_TAIL_S = 4
/** A missing person whose seat has looked different this long has got up
 *  and gone (until their face is found again). */
const LEAVE_S = 0.5
/** Frames this close (or a few frame times) follow on from each other… */
const TRACKED_S = 0.25
/** …and a head lost for longer counts as back once a face is really found
 *  there — or once it's been found again and again for this long. */
const COMEBACK_S = 1

export class FaceFinder {
  private recent = new Map<number, { box: Box; t: number; vx: number; vy: number }>()
  private held = new Map<number, { lm: Float32Array; t: number; yaw: number; rate: number }>()
  /** Fixed camera: each head's last trusted turn, and how it's turning… */
  private poses = new Map<number, { yaw: number; t: number; turn: Turn }>()
  /** …and since when a head lost is being found again (not yet trusted). */
  private comeback = new Map<number, { since: number; last: number }>()
  /** The time between frames, as they come (s). */
  private step = 1 / 30
  /** Fixed camera: under each face last found, its shoulders (see
   *  faces.ts) — and who, missing now, is still sitting there. */
  private bodies = new Map<number, { box: Box; sig: Float32Array }>()
  /** …since when a missing person's seat has looked different, and who
   *  has gone (so a seat that happens to look the same again later — the
   *  light, what's behind — doesn't bring them back). */
  private leaving = new Map<number, number>()
  private gone = new Set<number>()
  seated = new Set<number>()
  private lastT = -Infinity
  private an: VideoAnalysis
  private video: FaceLandmarker
  private image: FaceLandmarker
  /** For a seat someone is known to be in (fixed camera): looser. */
  private seat: FaceLandmarker
  private hold: boolean

  constructor(an: VideoAnalysis, video: FaceLandmarker, image: FaceLandmarker, seat: FaceLandmarker, hold: boolean) {
    this.an = an
    this.video = video
    this.image = image
    this.seat = seat
    this.hold = hold
  }

  /** `hold`: with a fixed camera, keep a seated person's last mesh while
   *  their face can't be found (the preview; the export bridges gaps
   *  itself, looking ahead). */
  static async create(an: VideoAnalysis, hold = an.fixed) {
    const { video, seated, image, lenient } = await loadVideoModels()
    return new FaceFinder(an, an.fixed ? seated : video, image, an.fixed ? lenient : image, hold)
  }

  /** A face found afresh (not tracked) round box `b` of `frame`. */
  private faceAt(frame: HTMLCanvasElement, b: Box) {
    const W = frame.width
    const H = frame.height
    const c = centreOf(b)
    const r = windowAt(c.x, c.y, Math.min(Math.min(W, H), Math.max(sizeOf(b) * 2.2, 96)), W, H)
    const res = this.image.detect(cropOf(frame, r))
    return res.faceLandmarks.some((lm) => iou(boxOf(pack(lm, undefined, r), W, H), b) > 0.3)
  }

  /** The people in `frame`, at time `t` in the clip. `sequential`: frames
   *  come one after another (playing, exporting) — they're tracked from
   *  frame to frame; otherwise each is looked at on its own (a still). */
  find(frame: HTMLCanvasElement, t: number, sequential: boolean): Found[] {
    const W = frame.width
    const H = frame.height
    if (!sequential || t < this.lastT || t - this.lastT > RECENT_S) this.recent.clear()
    if (sequential && t > this.lastT && t - this.lastT < 1) this.step = 0.8 * this.step + 0.2 * (t - this.lastT)
    // (a jump: nothing held from before it)
    else if (!sequential || t < this.lastT) {
      this.held.clear()
      this.comeback.clear()
      this.leaving.clear()
      this.gone.clear()
    }
    this.lastT = t
    const expected = this.an.boxesAt(t)
    this.seated.clear()
    // No one is meant to be here: nothing to look for.
    if (!expected.length) {
      this.recent.clear()
      return []
    }
    const res = sequential ? this.video.detectForVideo(frame, stamp(this.video, t)) : this.image.detect(frame)
    const dets = res.faceLandmarks.map((lm, i) => {
      const a = pack(lm, closureFrom(res.faceBlendshapes, i))
      return { lm: a, box: boxOf(a, W, H) }
    })

    // Where each person should be: where they just were (moving on as they
    // were), and where the analysis had them.
    const want = expected.map(({ id, box }) => {
      const where = scaleBox(box, W, H)
      const r = this.recent.get(id)
      const near = r ? scaleBox(shift(r.box, r.vx * (t - r.t), r.vy * (t - r.t)), W, H) : null
      return { id, where, near }
    })
    const score = (w: (typeof want)[number], d: Box) => (w.near ? 0.6 * fitScore(w.near, d) + 0.4 * fitScore(w.where, d) : fitScore(w.where, d))

    const pairs: [number, number, number][] = []
    want.forEach((w, wi) =>
      dets.forEach((d, di) => {
        const s = score(w, d.box)
        if (s > 0.06) pairs.push([s, wi, di])
      }),
    )
    pairs.sort((a, b) => b[0] - a[0])
    const wUsed = new Set<number>()
    const dUsed = new Set<number>()
    const out: Found[] = []
    const boxes: Box[] = []
    for (const [, wi, di] of pairs) {
      if (wUsed.has(wi) || dUsed.has(di)) continue
      wUsed.add(wi)
      dUsed.add(di)
      out.push({ id: want[wi].id, lm: dets[di].lm })
      boxes.push(dets[di].box)
    }

    // Look again for whoever's missing, in a window round where they
    // should be (zoomed in: a small face is plenty big there).
    let tries = 0
    want.forEach((w, wi) => {
      if (wUsed.has(wi) || tries >= (this.an.fixed ? MAX_FACES : MAX_RETRIES)) return
      tries++
      const p = w.near ?? w.where
      const c = centreOf(p)
      const r = windowAt(c.x, c.y, Math.min(Math.min(W, H), Math.max(sizeOf(p) * 2.6, 96)), W, H)
      const rr = this.seat.detect(cropOf(frame, r))
      let best: { lm: Float32Array; box: Box } | null = null
      let bs = 0.1
      rr.faceLandmarks.forEach((lm, i) => {
        const a = pack(lm, closureFrom(rr.faceBlendshapes, i), r)
        const b = boxOf(a, W, H)
        if (boxes.some((o) => iou(o, b) > 0.35)) return
        const s = score(w, b)
        if (s > bs) {
          bs = s
          best = { lm: a, box: b }
        }
      })
      if (best) {
        const { lm, box } = best
        out.push({ id: w.id, lm })
        boxes.push(box)
      }
    })

    // Fixed camera: a mesh that has swung further than a head turns between
    // frames — or one that turns up after a head was lost — may be the back
    // of a head the models took for a face. Kept only if a face really is
    // there (found afresh, strictly), or it keeps being found a while.
    if (this.an.fixed && sequential) {
      const tracked = Math.max(TRACKED_S, this.step * 2.5)
      for (let i = out.length - 1; i >= 0; i--) {
        const { id, lm } = out[i]
        const yaw = yawOf(lm, W, H)
        const p = this.poses.get(id)
        const dt = p ? t - p.t : Infinity
        let ok: boolean
        if (dt >= 0 && dt <= tracked) ok = plausibleTurn(p!.yaw, yaw, dt) || this.faceAt(frame, boxes[i])
        else {
          const c = this.comeback.get(id)
          const since = c && t >= c.last && t - c.last <= tracked ? c.since : t
          this.comeback.set(id, { since, last: t })
          ok = t - since >= COMEBACK_S || this.faceAt(frame, boxes[i])
        }
        if (ok) {
          this.comeback.delete(id)
          const turn = p?.turn ?? new Turn()
          turn.add(yaw, t)
          this.poses.set(id, { yaw, t, turn })
        } else {
          out.splice(i, 1)
          boxes.splice(i, 1)
        }
      }
    }

    // Remember where everyone found is, and how they're moving.
    const seen = new Set<number>()
    out.forEach((f, i) => {
      seen.add(f.id)
      const b = boxes[i]
      const box = { x0: b.x0 / W, y0: b.y0 / H, x1: b.x1 / W, y1: b.y1 / H }
      const r = this.recent.get(f.id)
      const dt = r ? t - r.t : 0
      let vx = 0
      let vy = 0
      if (r && dt > 1e-4) {
        const c0 = centreOf(r.box)
        const c1 = centreOf(box)
        vx = r.vx * 0.5 + ((c1.x - c0.x) / dt) * 0.5
        vy = r.vy * 0.5 + ((c1.y - c0.y) / dt) * 0.5
      }
      this.recent.set(f.id, { box, t, vx, vy })
    })
    for (const id of [...this.recent.keys()]) if (!seen.has(id) && t - this.recent.get(id)!.t > RECENT_S) this.recent.delete(id)

    // Fixed camera: who is still sitting there, found or not; and (the
    // preview) whoever is, but wasn't found, keeps their last mesh a while.
    if (this.an.fixed) {
      out.forEach((f, i) => {
        this.seated.add(f.id)
        this.leaving.delete(f.id)
        this.gone.delete(f.id)
        const body = bodyBoxOf(boxes[i], W, H)
        if (body) this.bodies.set(f.id, { box: { x0: body.x0 / W, y0: body.y0 / H, x1: body.x1 / W, y1: body.y1 / H }, sig: bodySig(frame, body) })
        else this.bodies.delete(f.id)
      })
      for (const w of want) {
        if (seen.has(w.id) || this.gone.has(w.id)) continue
        const b = this.bodies.get(w.id)
        // (nothing to go by: take the analysis's word that they're there)
        if (!b || sameBody(b.sig, bodySig(frame, scaleBox(b.box, W, H)))) {
          this.leaving.delete(w.id)
          this.seated.add(w.id)
          continue
        }
        const since = this.leaving.get(w.id) ?? t
        this.leaving.set(w.id, since)
        if (t - since >= LEAVE_S) this.gone.add(w.id)
      }
      if (this.hold) {
        for (const f of out) {
          const p = this.poses.get(f.id)
          this.held.set(f.id, { lm: f.lm, t, yaw: p?.t === t ? p.yaw : yawOf(f.lm, W, H), rate: p?.t === t ? p.turn.rate : 0 })
        }
        for (const w of want) {
          if (seen.has(w.id) || !this.seated.has(w.id)) continue
          const h = this.held.get(w.id)
          // (lost turning away: gone on round, out of sight)
          if (h && t >= h.t && t - h.t <= HOLD_S) out.push({ id: w.id, lm: turnTo(h.lm, lostYaw(h.yaw, h.rate, t - h.t), W, H) })
        }
      }
    }
    return out
  }
}

/** Steadies meshes as frames play (the preview): One Euro, looking back
 *  only — what a live picture can do. */
export class LiveSmoother {
  private states = new Map<number, EuroState>()
  private lastT = -Infinity
  /** (a fixed camera: steadier at rest) */
  calm = false

  apply(found: Found[], t: number): Found[] {
    // A jump (seek, loop): start afresh.
    if (t < this.lastT || t - this.lastT > RECENT_S) this.states.clear()
    this.lastT = t
    const ids = new Set(found.map((f) => f.id))
    for (const id of [...this.states.keys()]) if (!ids.has(id)) this.states.delete(id)
    return found.map((f) => {
      let s = this.states.get(f.id)
      if (!s) this.states.set(f.id, (s = new EuroState(this.calm)))
      return { id: f.id, lm: Float32Array.from(s.step(f.lm, t)) }
    })
  }

  reset() {
    this.states.clear()
    this.lastT = -Infinity
  }
}

/** How far ahead the smoothing looks (the backward pass's run-in). */
const SMOOTH_AHEAD_S = 0.3
/** Fixed camera: how long into a head's coming back its turn is read. */
const TURN_READ_S = 0.3

const smoothstep = (u: number) => u * u * (3 - 2 * u)

interface Scouted {
  ts: number
  raw: Map<number, Float32Array>
  /** Fixed camera: who is still sitting there (found or not). */
  seated?: Set<number>
  /** With short gaps bridged; then each face filtered forwards. */
  filled?: Map<number, Float32Array>
  fwd?: Map<number, Float32Array>
}

/**
 * Reads a clip ahead of the export, finding the people in each frame and
 * steadying their meshes with the frames either side. The export asks for
 * frames in order (`at`, by source timestamp); this keeps about a second of
 * meshes in hand, never the frames themselves.
 */
export class FaceScout {
  private frames: Scouted[] = []
  private filledTo = 0
  private ended = false
  private euro = new Map<number, EuroState>()
  /** Each face as last found (fixed camera: which way it faced, and how
   *  fast it was turning). */
  private lastSeen = new Map<number, { ts: number; lm: Float32Array; yaw: number; rate: number }>()
  private turns = new Map<number, Turn>()
  /** The frames' size, px (for which way heads face). */
  private W = 1
  private H = 1
  /** Read past each frame before it's bridged: a fixed camera reads how a
   *  head that comes back is turning. */
  private turnAhead: number
  private lastFilled: Map<number, Float32Array> | undefined
  private back: EuroState
  private warned = false
  private an: VideoAnalysis
  /** Gaps bridged (a fixed camera's are longer), and how long someone who
   *  doesn't come back keeps their last mesh. */
  private gap: number
  private tail: number
  private input: Input
  private iter: AsyncIterator<{ canvas: HTMLCanvasElement | OffscreenCanvas; timestamp: number }>
  private finder: FaceFinder
  /** Half the shortest frame time seen: how close a match must be. */
  private tol = 0.5 / 30

  private constructor(input: Input, iter: FaceScout['iter'], finder: FaceFinder, an: VideoAnalysis) {
    this.input = input
    this.iter = iter
    this.finder = finder
    this.an = an
    this.gap = an.fixed ? FIXED_FILL_S : FILL_GAP_S
    this.tail = an.fixed ? FIXED_TAIL_S : 0
    this.turnAhead = an.fixed ? TURN_READ_S : 0
    this.back = new EuroState(an.fixed)
  }

  /** `workMax`: the long side frames are looked at in. */
  static async open(file: Blob, an: VideoAnalysis, workMax: number): Promise<FaceScout> {
    const finder = await FaceFinder.create(an, false)
    const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
    const track = await input.getPrimaryVideoTrack()
    if (!track) throw new Error('This file has no video.')
    const k = Math.min(1, workMax / Math.max(track.displayWidth, track.displayHeight))
    const sink = new CanvasSink(track, { width: Math.max(2, Math.round(track.displayWidth * k)), height: Math.max(2, Math.round(track.displayHeight * k)), fit: 'fill', poolSize: 2 })
    return new FaceScout(input, sink.canvases()[Symbol.asyncIterator](), finder, an)
  }

  private async pull() {
    const r = await this.iter.next()
    if (r.done) {
      this.ended = true
      return
    }
    const { canvas, timestamp } = r.value
    this.W = canvas.width
    this.H = canvas.height
    const prev = this.frames[this.frames.length - 1]
    if (prev && timestamp <= prev.ts) return
    if (prev) this.tol = Math.min(this.tol, Math.max(1e-3, (timestamp - prev.ts) / 2))
    const found = this.finder.find(canvas as HTMLCanvasElement, timestamp, true)
    this.frames.push({ ts: timestamp, raw: new Map(found.map((f) => [f.id, f.lm])), seated: this.an.fixed ? new Set(this.finder.seated) : undefined })
  }

  /** How fast the head that turns up in frame `k` turns, run backwards
   *  (from its first moments back), deg/s. */
  private rateBack(k: number, id: number, yaw: number) {
    let m = -1
    for (let n = k + 1; n < this.frames.length && this.frames[n].ts - this.frames[k].ts <= TURN_READ_S; n++) if (this.frames[n].raw.has(id)) m = n
    if (m < 0 || this.frames[m].ts - this.frames[k].ts < 0.1) return 0
    return (yaw - yawOf(this.frames[m].raw.get(id)!, this.W, this.H)) / (this.frames[m].ts - this.frames[k].ts)
  }

  /** Bridges short gaps in frame `j` and filters it forwards. Needs the
   *  frames up to `gap` (and `turnAhead`) past it, or the end of the clip. */
  private fill(j: number) {
    const f = this.frames[j]
    const { W, H } = this
    const filled = new Map(f.raw)
    for (const [id, seen] of this.lastSeen) {
      const lost = f.ts - seen.ts
      if (filled.has(id) || lost > this.gap) continue
      // (fixed camera: only while they're still in their seat)
      if (f.seated && !f.seated.has(id)) continue
      // Back within the gap? Then they were here all along.
      for (let k = j + 1; k < this.frames.length && this.frames[k].ts - seen.ts <= this.gap; k++) {
        const b = this.frames[k].raw.get(id)
        if (!b) continue
        const u = lost / Math.max(1e-6, this.frames[k].ts - seen.ts)
        if (!this.an.fixed) filled.set(id, lerpFace(seen.lm, b, u))
        else {
          // (a head lost turning away went on round, out of sight, and
          // came back the way it's turning when it turns up)
          const yb = yawOf(b, W, H)
          const ya = lostYaw(seen.yaw, seen.rate, lost)
          const yc = lostYaw(yb, this.rateBack(k, id, yb), this.frames[k].ts - f.ts)
          const yaw = ya + (yc - ya) * smoothstep(u)
          filled.set(id, lerpFace(turnTo(seen.lm, yaw, W, H), turnTo(b, yaw, W, H), u))
        }
        break
      }
      // (a fixed camera: not back, but still in their seat — keep them a
      // little longer)
      if (!filled.has(id) && lost <= this.tail) filled.set(id, turnTo(seen.lm, lostYaw(seen.yaw, seen.rate, lost), W, H))
    }
    for (const [id, lm] of f.raw) {
      if (!this.an.fixed) {
        this.lastSeen.set(id, { ts: f.ts, lm, yaw: 0, rate: 0 })
        continue
      }
      const yaw = yawOf(lm, W, H)
      let turn = this.turns.get(id)
      if (!turn) this.turns.set(id, (turn = new Turn()))
      turn.add(yaw, f.ts)
      this.lastSeen.set(id, { ts: f.ts, lm, yaw, rate: turn.rate })
    }
    const prev = this.lastFilled
    const fwd = new Map<number, Float32Array>()
    for (const [id, lm] of filled) {
      let s = this.euro.get(id)
      if (!s) this.euro.set(id, (s = new EuroState(this.an.fixed)))
      if (!prev?.has(id)) s.reset()
      fwd.set(id, Float32Array.from(s.step(lm, f.ts)))
    }
    for (const id of [...this.euro.keys()]) if (!filled.has(id)) this.euro.delete(id)
    f.filled = filled
    f.fwd = fwd
    this.lastFilled = filled
  }

  /** The people in the source frame at `ts`, their meshes steadied. */
  async at(ts: number): Promise<Found[]> {
    // Read on until this frame's look-ahead is all in.
    const need = ts + SMOOTH_AHEAD_S + this.gap + this.turnAhead + this.tol
    while (!this.ended && (!this.frames.length || this.frames[this.frames.length - 1].ts < need)) await this.pull()
    if (!this.frames.length) return []
    // Bridge and filter everything whose own look-ahead is in.
    const end = this.frames[this.frames.length - 1].ts
    while (this.filledTo < this.frames.length && (this.ended || this.frames[this.filledTo].ts + this.gap + this.turnAhead <= end)) this.fill(this.filledTo++)

    // The frame: the nearest one (frames come in order, so from the front).
    let i = 0
    while (i + 1 < this.frames.length && Math.abs(this.frames[i + 1].ts - ts) <= Math.abs(this.frames[i].ts - ts)) i++
    if (Math.abs(this.frames[i].ts - ts) > this.tol * 1.01 && !this.warned) {
      this.warned = true
      console.warn(`Face scout: no frame at ${ts.toFixed(4)}s (nearest ${this.frames[i].ts.toFixed(4)}s)`)
    }
    // (done with everything before it)
    if (i > 0) {
      this.frames.splice(0, i)
      this.filledTo -= i
      i = 0
    }
    const f = this.frames[0]
    if (!f.filled || !f.fwd) return []
    const out: Found[] = []
    for (const [id, lm] of f.filled) {
      // Backwards from a little ahead (as far as they stay on screen), to
      // meet the forward pass here: their average lags neither way.
      let e = 0
      while (e + 1 < this.filledTo && this.frames[e + 1].ts - f.ts <= SMOOTH_AHEAD_S && this.frames[e + 1].filled!.has(id)) e++
      this.back.reset()
      let b: Float32Array = lm
      for (let k = e; k >= 0; k--) b = this.back.step(this.frames[k].filled!.get(id)!, this.frames[k].ts)
      const fw = f.fwd.get(id)!
      const m = new Float32Array(FRAME)
      for (let c = 0; c < STRIDE; c++) m[c] = (fw[c] + b[c]) / 2
      m[STRIDE] = lm[STRIDE]
      m[STRIDE + 1] = lm[STRIDE + 1]
      out.push({ id, lm: m })
    }
    return out
  }

  dispose() {
    void this.iter.return?.()
    this.input.dispose()
  }
}
