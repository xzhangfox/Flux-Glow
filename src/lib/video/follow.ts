import type { FaceLandmarker, ImageSegmenter } from '@mediapipe/tasks-vision'
import { ALL_FORMATS, BlobSource, CanvasSink, Input } from 'mediabunny'
import { closureFrom } from '../faceLandmarker'
import { FILL_GAP_S, loadPersonSegmenter, loadStrictLandmarker, loadVideoModels, type VideoAnalysis } from './analyze'
import { EuroState, FRAME, HELD_MAX_DEG, STRIDE, Turn, bodyBoxOf, bodySig, boxOf, centreOf, cropOf, fitScore, iou, lerpFace, lostYaw, pack, personIn, sameBody, scaleBox, shift, sizeOf, stamp, turnTo, windowAt, yawOf, type Box } from './faces'
import { UsualFace, averagePose, lerpLocal, localOf, meshOf, poseOf, quatAngle, shapeDistance, type Pose } from './pose'

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
//
// A fixed camera (an interview) knows more, and keeps each head on through
// everything an interview throws at it:
//  • a mesh is taken for someone's head only if their head could have made
//    it — no sudden swell, jump or swing, built like their face (a hand by
//    the face, a mesh collapsing on a profile, the back of a head are not);
//  • a head that can't be seen (turned to the side, a hand over it) is held
//    for as long as the person segmenter still sees someone in the seat —
//    in its last pose, easing into their usual face, eyes open; lost while
//    turning, it goes on round to side-on;
//  • a held head counts as back once their clean face keeps being found,
//    and eases onto it;
//  • a head's size (a seated head only changes size as they lean) and turn
//    are steadied over the frames round each one;
//  • an edited clip's cuts (see analyze.ts) start each shot afresh: no head
//    is held, steadied or gated across one — where it was in the wide shot
//    says nothing of where it is in a close-up.

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
/** A frame this much further on (or back) than the last is a jump — a seek
 *  — not the next frame: nothing held carries over it. (Less — a pause, a
 *  nudge of the playhead — and the held heads stay.) */
const JUMP_S = 1
/** Export, fixed camera: a head that comes back within this long is eased
 *  across the whole time it was gone (as 0.6 s for a moving camera)… */
const FIXED_FILL_S = 6
/** …and one that's been gone longer is held, then eased onto where it
 *  comes back over this long before it does. */
const RETURN_BLEND_S = 0.8
/** A seat the segmenter sees this much of a person in, where the head was:
 *  someone is still sitting there. (An empty one scores near 0; a head
 *  turned away, whose outline no longer fills the box its face did, can
 *  score as little as 0.3.) */
const PRESENT = 0.2
/** A seat that has looked empty this long: they've got up and gone (until
 *  their face is found again). Before then the head isn't shown — except
 *  the preview's, held through the first DOUBT_S of a doubt (it can't look
 *  ahead, as the export does, to see whether they're still there). */
const LEAVE_S = 0.6
const DOUBT_S = 0.25
/** Frames this close (or a few frame times) follow on from each other… */
const TRACKED_S = 0.25
/** …and a head held a while counts as back once the same clean face has
 *  been found there frame after frame this long — this much longer if it
 *  turns up turned another way than the head was held. (A face in profile
 *  now and then gives a mesh that looks clean but is collapsed on itself;
 *  it doesn't keep doing it. The export, reading ahead, eases the head
 *  onto the face from before it's taken.) */
const BACK_S = 0.2
const COMEBACK_S = 0.8

/** Fixed camera: how far a head can have got since its last trusted mesh.
 *  Within NEAR_S: the size (|log ratio|), place (eye distances) and turn
 *  (degrees) it can change by at once, and by per second. After longer
 *  (hidden a while, or a jump): near where it was and about as big, turned
 *  any way. And the build of the face (see pose.ts) has to be theirs. */
const NEAR_S = 0.5
const GATE = { scale: 0.16, scaleRate: 0.6, place: 0.7, placeRate: 6, turn: 32, turnRate: 260, farScale: 0.4, farPlace: 2.5, farTurn: 40, shape: 0.22 }
/** A face found again after a while has to be built cleanly like theirs
 *  (a mesh collapsing on a face in profile mostly scores 0.23–0.33; one
 *  turning away still tracked, up to 0.21; facing the camera, under 0.1). */
const CLEAN_SHAPE = 0.15
/** A head turned further than FRONT_DEG reads smaller than it is (the
 *  landmarker's depth falls short): by TRUE_SIZE_DEG it takes the size it
 *  last had facing the camera — a seated head's size only changes as they
 *  lean. */
const FRONT_DEG = 15
const TRUE_SIZE_DEG = 40

/** The preview: a head found again after being held eases from where it
 *  was held onto where it's found — over this long, or longer for a bigger
 *  turn (at most EASE_DEG_S), up to RETURN_EASE_MAX_S. (The export, reading
 *  ahead, eases it in before it's found.) */
const RETURN_EASE_S = 0.35
const RETURN_EASE_MAX_S = 0.8
const EASE_DEG_S = 120

const smoothstep = (u: number) => u * u * (3 - 2 * u)

/** `from` eased onto `to` by `u` (0..1): turned alike first (by yaw, the
 *  short way), so a head held facing away doesn't flatten on its way back. */
function easeFace(from: Float32Array, to: Float32Array, u: number, W: number, H: number) {
  const ya = yawOf(from, W, H)
  const yb = yawOf(to, W, H)
  const yaw = ya + (yb - ya) * u
  return lerpFace(turnTo(from, yaw, W, H), turnTo(to, yaw, W, H), u)
}

/** A person's head as last trusted, and how it's held while it can't be
 *  seen: in its pose then (steadied over the moments before — the last
 *  frame before a hand comes over is often crumpled by it), easing into
 *  their usual face with the eyes open, and — if it was lost turning away —
 *  gone on round, out of sight. */
class Head {
  lm: Float32Array = new Float32Array(FRAME)
  t = -Infinity
  pose: Pose = { o: [0, 0, 0], s: 1, q: [0, 0, 0, 1] }
  /** Its box, fractions of the frame. */
  box: Box = { x0: 0, y0: 0, x1: 0, y1: 0 }
  yaw = 0
  readonly turn = new Turn()
  readonly usual = new UsualFace()
  private recent: { t: number; p: Pose }[] = []
  private own: Float32Array | null = null
  /** Its size (log) when last facing the camera, and when. */
  private front = { ls: 0, t: -Infinity }
  /** The shot it was last seen in, and that shot's camera angle. */
  shot = -1
  private angle = -1

  /** It's seen in shot `shot` now (from camera angle `angle`): how it was
   *  held in another shot is no guide to how it's held in this one — nor,
   *  from another camera, how big it is. (Their usual face is theirs from
   *  any angle.) */
  inShot(shot: number, angle: number) {
    if (shot === this.shot) return
    this.shot = shot
    this.recent = []
    if (angle !== this.angle) this.front = { ls: 0, t: -Infinity }
    this.angle = angle
  }

  see(lm: Float32Array, t: number, W: number, H: number, pose = poseOf(lm, W, H)) {
    if (t < this.t || t - this.t > 0.5) this.recent = []
    this.lm = lm
    this.t = t
    this.pose = pose
    const b = boxOf(lm, W, H)
    this.box = { x0: b.x0 / W, y0: b.y0 / H, x1: b.x1 / W, y1: b.y1 / H }
    this.yaw = yawOf(lm, W, H)
    this.turn.add(this.yaw, t)
    if (Math.abs(this.yaw) < FRONT_DEG) this.front = { ls: t - this.front.t < 1 ? this.front.ls + (Math.log(pose.s) - this.front.ls) * 0.2 : Math.log(pose.s), t }
    this.recent.push({ t, p: pose })
    while (this.recent.length > 1 && t - this.recent[0].t > 0.3) this.recent.shift()
    const local = localOf(lm, pose, W, H)
    this.own = local
    // (their usual face from clear views: toward the camera, and like it)
    const usual = this.usual.face
    if (Math.abs(this.yaw) < 30 && (!usual || shapeDistance(local, usual) < GATE.shape * 0.6)) this.usual.add(local)
    return local
  }

  /** How big a head turned `yaw` degrees really is, given it reads as
   *  `s` (see TRUE_SIZE_DEG). */
  trueSize(s: number, yaw: number, t: number) {
    const w = smoothstep(Math.min(1, Math.max(0, (Math.abs(yaw) - FRONT_DEG) / (TRUE_SIZE_DEG - FRONT_DEG))))
    if (!w || t - this.front.t > 30) return s
    return Math.exp(Math.log(s) + (this.front.ls - Math.log(s)) * w)
  }

  /** Mesh `lm` (pose `p`), sized as the head really is. */
  sized(lm: Float32Array, p: Pose, t: number, W: number, H: number): Float32Array {
    const s = this.trueSize(p.s, yawOf(lm, W, H), t)
    return Math.abs(Math.log(s / p.s)) < 0.005 ? lm : meshOf({ ...p, s }, localOf(lm, p, W, H), W, H, [lm[STRIDE], lm[STRIDE + 1]])
  }

  /** The head held at `t` (some time after it was last seen). */
  at(t: number, W: number, H: number): Float32Array {
    const lost = Math.max(0, t - this.t)
    const avg = averagePose(this.recent.map((r) => ({ p: r.p, w: 1 }))) ?? this.pose
    const pose = { ...avg, s: this.trueSize(avg.s, this.yaw, t) }
    const own = this.own ?? localOf(this.lm, this.pose, W, H)
    const usual = this.usual.face
    const u = usual ? smoothstep(Math.min(1, lost / 0.5)) : 0
    const local = usual && u > 0 ? lerpLocal(own, usual, u) : own
    const eyes: [number, number] = lost > 0.25 ? [0, 0] : [this.lm[STRIDE], this.lm[STRIDE + 1]]
    return turnTo(meshOf(pose, local, W, H, eyes), lostYaw(this.yaw, this.turn.rate, lost, HELD_MAX_DEG), W, H)
  }
}

export class FaceFinder {
  private recent = new Map<number, { box: Box; t: number; vx: number; vy: number }>()
  /** Fixed camera: each head as last trusted (see Head)… */
  private heads = new Map<number, Head>()
  /** …since when a head lost is being found again turned another way (not
   *  yet trusted). */
  private comeback = new Map<number, { since: number; last: number }>()
  /** Where the current run of frames began (a jump or a cut starts one):
   *  only a head seen since then is held. */
  private runStart = -Infinity
  /** The shot the last frame was in, and its camera angle; and whether this
   *  frame is the first of another (the preview's smoothing starts afresh). */
  private shot = -1
  private angle = 0
  cut = false
  /** The preview: each head as last held, and a head found again easing
   *  in from there. */
  private heldOut = new Map<number, { lm: Float32Array; t: number }>()
  private easing = new Map<number, { from: Float32Array; t0: number; d: number }>()
  /** The time between frames, as they come (s). */
  private step = 1 / 30
  /** Fixed camera, without the segmenter: under each face last found, its
   *  shoulders (see faces.ts). */
  private bodies = new Map<number, { box: Box; sig: Float32Array }>()
  /** …since when a missing person's seat has looked empty, and who has
   *  gone (so a seat that happens to look full again later doesn't bring
   *  them back before their face does). */
  private leaving = new Map<number, number>()
  private gone = new Set<number>()
  seated = new Set<number>()
  /** Dev: every mesh weighed against a head, and what was made of it. */
  gateLog: { t: number; id: number; ds: number; dp: number; dr: number; shape: number; far: boolean; ok: boolean }[] | null = null
  private lastT = -Infinity
  private an: VideoAnalysis
  private video: FaceLandmarker
  private image: FaceLandmarker
  /** For a seat someone is known to be in (fixed camera): looser. */
  private seat: FaceLandmarker
  /** "Surely a face" (faceAt): a fixed camera asks a strict landmarker. */
  private sure: FaceLandmarker
  private person: ImageSegmenter | null
  private hold: boolean

  constructor(an: VideoAnalysis, video: FaceLandmarker, image: FaceLandmarker, seat: FaceLandmarker, sure: FaceLandmarker, person: ImageSegmenter | null, hold: boolean) {
    this.an = an
    this.video = video
    this.image = image
    this.seat = seat
    this.sure = sure
    this.person = person
    this.hold = hold
  }

  /** `hold`: with a fixed camera, keep a seated person's head on while
   *  their face can't be found (the preview; the export bridges gaps
   *  itself, looking ahead). */
  static async create(an: VideoAnalysis, hold = an.fixed) {
    const [{ video, seated, image, lenient }, person, strict] = await Promise.all([loadVideoModels(), an.fixed ? loadPersonSegmenter() : null, an.fixed ? loadStrictLandmarker() : null])
    return new FaceFinder(an, an.fixed ? seated : video, image, an.fixed ? lenient : image, strict ?? image, person, hold)
  }

  /** A face found afresh (not tracked) round box `b` of `frame`. */
  private faceAt(frame: HTMLCanvasElement, b: Box) {
    const W = frame.width
    const H = frame.height
    const c = centreOf(b)
    const r = windowAt(c.x, c.y, Math.min(Math.min(W, H), Math.max(sizeOf(b) * 2.2, 96)), W, H)
    const res = this.sure.detect(cropOf(frame, r))
    return res.faceLandmarks.some((lm) => iou(boxOf(pack(lm, undefined, r), W, H), b) > 0.3)
  }

  /** Fixed camera: could person `id`'s head have made mesh `lm` (box `b`)
   *  now, given where it last was? */
  private trust(frame: HTMLCanvasElement, id: number, lm: Float32Array, b: Box, t: number, pose: Pose) {
    const W = frame.width
    const H = frame.height
    const h = this.heads.get(id)
    // (first seen in this shot: nothing to hold it to)
    if (!h || h.shot !== this.shot) return true
    const dt = t - h.t
    const usual = h.usual.face
    const shape = usual ? shapeDistance(localOf(lm, pose, W, H), usual) : 0
    const ds = Math.abs(Math.log(pose.s / h.pose.s))
    const dp = Math.hypot(pose.o[0] - h.pose.o[0], pose.o[1] - h.pose.o[1]) / h.pose.s
    const dr = quatAngle(pose.q, h.pose.q)
    const far = !(dt >= 0 && dt <= NEAR_S && h.t >= this.runStart)
    let ok: boolean
    if (!far) {
      // Moments after it was seen: a mesh that swelled, jumped or isn't
      // built like their face is something else (a hand) or a bad read —
      // and one that swung further than a head turns may be the back of
      // their head, unless a face really is there.
      ok = shape <= GATE.shape && ds <= GATE.scale + GATE.scaleRate * dt && dp <= GATE.place + GATE.placeRate * dt && (dr <= GATE.turn + GATE.turnRate * dt || this.faceAt(frame, b))
    } else {
      // Back after a while: the same clean face, frame after frame, for a
      // moment — longer if it's turned another way than the head was held;
      // and somewhere else or another size (they moved while hidden), only
      // if a face is surely there too, not a hand.
      const tracked = Math.max(TRACKED_S, this.step * 2.5)
      const c = this.comeback.get(id)
      const clean = shape <= CLEAN_SHAPE
      const since = c && t >= c.last && t - c.last <= tracked ? c.since : t
      if (clean) this.comeback.set(id, { since, last: t })
      else this.comeback.delete(id)
      const heldYaw = lostYaw(h.yaw, h.turn.rate, Math.max(0, dt), HELD_MAX_DEG)
      const turned = Math.abs(yawOf(lm, W, H) - heldYaw) > GATE.farTurn
      const placed = ds <= GATE.farScale && dp <= GATE.farPlace
      ok = clean && t - since >= (turned || !placed ? COMEBACK_S : BACK_S) && (placed || (dt > 1 && this.faceAt(frame, b)))
    }
    this.gateLog?.push({ t, id, ds, dp, dr, shape, far, ok })
    return ok
  }

  /** The people in `frame`, at time `t` in the clip. `sequential`: frames
   *  come one after another (playing, exporting) — they're tracked from
   *  frame to frame; otherwise each is looked at on its own (a still). */
  find(frame: HTMLCanvasElement, t: number, sequential: boolean): Found[] {
    const W = frame.width
    const H = frame.height
    const jump = Math.abs(t - this.lastT) > JUMP_S || (sequential && t < this.lastT)
    if (!sequential || t < this.lastT || t - this.lastT > RECENT_S) this.recent.clear()
    if (sequential && t > this.lastT && t - this.lastT < 1) this.step = 0.8 * this.step + 0.2 * (t - this.lastT)
    if (jump) {
      this.runStart = t
      this.heldOut.clear()
      this.easing.clear()
      this.comeback.clear()
      this.leaving.clear()
      this.gone.clear()
    }
    this.lastT = t
    // A cut to another shot: no one held from the last one.
    const shot = this.an.shotAt(t)
    const cut = (this.cut = this.an.fixed && shot !== this.shot && !jump)
    if (cut) this.runStart = t
    this.shot = shot
    this.angle = this.an.angleAt(t)
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
    // were) — or, a fixed camera, where their head last was — and where the
    // analysis had them.
    const want = expected.map(({ id, box }) => {
      const where = scaleBox(box, W, H)
      const r = this.recent.get(id)
      const h = this.an.fixed ? this.heads.get(id) : undefined
      const near = r ? scaleBox(shift(r.box, r.vx * (t - r.t), r.vy * (t - r.t)), W, H) : h && h.t >= this.runStart ? scaleBox(h.box, W, H) : null
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

    // Fixed camera: only a mesh their head could have made (see trust()).
    if (this.an.fixed) {
      for (let i = out.length - 1; i >= 0; i--) {
        const { id, lm } = out[i]
        const pose = poseOf(lm, W, H)
        if (!sequential || this.trust(frame, id, lm, boxes[i], t, pose)) {
          this.comeback.delete(id)
          let h = this.heads.get(id)
          if (!h) this.heads.set(id, (h = new Head()))
          h.inShot(this.shot, this.angle)
          h.see(lm, t, W, H, pose)
          out[i] = { id, lm: h.sized(lm, pose, t, W, H) }
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
    // preview) whoever is, but wasn't found, keeps their head on.
    if (this.an.fixed) {
      out.forEach((f, i) => {
        this.seated.add(f.id)
        this.leaving.delete(f.id)
        this.gone.delete(f.id)
        if (this.person) return
        const body = bodyBoxOf(boxes[i], W, H)
        if (body) this.bodies.set(f.id, { box: { x0: body.x0 / W, y0: body.y0 / H, x1: body.x1 / W, y1: body.y1 / H }, sig: bodySig(frame, body) })
        else this.bodies.delete(f.id)
      })
      for (const w of want) {
        if (seen.has(w.id) || this.gone.has(w.id)) continue
        // (another shot: whoever isn't in it is gone, till they're seen)
        if (cut) {
          this.gone.add(w.id)
          continue
        }
        const h = this.heads.get(w.id)
        let there: boolean
        if (this.person && h) there = personIn(this.person, frame, scaleBox(h.box, W, H)) >= PRESENT
        else {
          // (nothing to go by: take the analysis's word that they're there)
          const b = this.bodies.get(w.id)
          there = !b || sameBody(b.sig, bodySig(frame, scaleBox(b.box, W, H)))
        }
        if (there) {
          this.leaving.delete(w.id)
          this.seated.add(w.id)
          continue
        }
        // (not shown while it looks empty; gone — till their face is seen
        // again — once it's looked empty a moment, not at the first doubt)
        const since = this.leaving.get(w.id) ?? t
        this.leaving.set(w.id, since)
        if (t - since >= LEAVE_S) this.gone.add(w.id)
        else if (this.hold && t - since < DOUBT_S) this.seated.add(w.id)
      }
      if (this.hold) {
        for (const w of want) {
          if (seen.has(w.id) || !this.seated.has(w.id)) continue
          const h = this.heads.get(w.id)
          if (!h || h.t < this.runStart || t < h.t) continue
          const lm = h.at(t, W, H)
          out.push({ id: w.id, lm })
          this.heldOut.set(w.id, { lm, t })
          this.easing.delete(w.id)
        }
        // A head held until a moment ago, found again: eased in.
        for (const f of out) {
          if (!seen.has(f.id)) continue
          const held = this.heldOut.get(f.id)
          if (held && t >= held.t && t - held.t <= 0.5) {
            const turn = Math.abs(yawOf(held.lm, W, H) - yawOf(f.lm, W, H))
            this.easing.set(f.id, { from: held.lm, t0: t, d: Math.min(RETURN_EASE_MAX_S, Math.max(RETURN_EASE_S, turn / EASE_DEG_S)) })
          }
          this.heldOut.delete(f.id)
          const e = this.easing.get(f.id)
          if (!e) continue
          const u = (t - e.t0) / e.d
          if (u >= 1 || u < 0) this.easing.delete(f.id)
          else f.lm = easeFace(e.from, f.lm, smoothstep(u), W, H)
        }
      }
    }
    return out
  }
}

/** Steadies meshes as frames play (the preview): One Euro, looking back
 *  only — what a live picture can do. A fixed camera's heads are held
 *  steadier at rest, and their size steadier still (a seated head only
 *  changes size as they lean; the rest is the mesh's own wobble). */
export class LiveSmoother {
  private states = new Map<number, EuroState>()
  private sizes = new Map<number, { v: number; d: number; t: number }>()
  private lastT = -Infinity
  /** (a fixed camera) */
  calm = false

  /** `W` × `H`: the frame's size (steadies the head's size with it). */
  apply(found: Found[], t: number, W?: number, H?: number): Found[] {
    // A jump (seek, loop): start afresh.
    if (t < this.lastT || t - this.lastT > RECENT_S) {
      this.states.clear()
      this.sizes.clear()
    }
    const dt = Math.max(1e-3, t - this.lastT)
    this.lastT = t
    const ids = new Set(found.map((f) => f.id))
    for (const id of [...this.states.keys()]) if (!ids.has(id)) this.states.delete(id)
    for (const id of [...this.sizes.keys()]) if (!ids.has(id)) this.sizes.delete(id)
    return found.map((f) => {
      let s = this.states.get(f.id)
      if (!s) this.states.set(f.id, (s = new EuroState(this.calm)))
      const m = Float32Array.from(s.step(f.lm, t))
      if (!this.calm || !W || !H) return { id: f.id, lm: m }
      const p = poseOf(m, W, H)
      const ls = Math.log(p.s)
      const z = this.sizes.get(f.id)
      if (!z) {
        this.sizes.set(f.id, { v: ls, d: 0, t })
        return { id: f.id, lm: m }
      }
      // (One Euro on the log of the size: slow at rest, quick to follow a
      // real lean)
      const ad = 1 / (1 + 1 / (2 * Math.PI * 1 * dt))
      z.d += ad * ((ls - z.v) / dt - z.d)
      const a = 1 / (1 + 1 / (2 * Math.PI * (0.35 + 6 * Math.abs(z.d)) * dt))
      z.v += a * (ls - z.v)
      z.t = t
      return { id: f.id, lm: meshOf({ ...p, s: Math.exp(z.v) }, localOf(m, p, W, H), W, H, [m[STRIDE], m[STRIDE + 1]]) }
    })
  }

  reset() {
    this.states.clear()
    this.sizes.clear()
    this.lastT = -Infinity
  }
}

/** How far ahead the smoothing looks (the backward pass's run-in). */
const SMOOTH_AHEAD_S = 0.3
/** Fixed camera: how long into a head's coming back its turn is read. */
const TURN_READ_S = 0.3

/** Fixed camera: how far either side of a frame its head's size and turn
 *  are steadied over (see steady()). */
const STEADY_S = 0.45

interface Scouted {
  ts: number
  /** The shot it's in (see analyze.ts). */
  shot: number
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
  /** Each face as last found (a moving camera)… */
  private lastSeen = new Map<number, { ts: number; shot: number; lm: Float32Array }>()
  /** …or each head as last found (a fixed one: see Head). */
  private heads = new Map<number, Head>()
  /** Fixed camera: when each person was last seen in their seat. */
  private lastSeated = new Map<number, number>()
  /** Fixed camera: each head's pose in the frames just served (steady()). */
  private past = new Map<number, { ts: number; shot: number; p: Pose }[]>()
  /** The frames' size, px (for which way heads face). */
  private W = 1
  private H = 1
  /** Read past each frame before it's bridged: a fixed camera reads how a
   *  head that comes back is turning. */
  private turnAhead: number
  private lastFilled: Map<number, Float32Array> | undefined
  private lastShot = 0
  private back: EuroState
  private warned = false
  private an: VideoAnalysis
  /** Gaps bridged across (a fixed camera's are longer; and it holds a head
   *  for as long as the person is in their seat). */
  private gap: number
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
    this.frames.push({ ts: timestamp, shot: this.an.shotAt(timestamp), raw: new Map(found.map((f) => [f.id, f.lm])), seated: this.an.fixed ? new Set(this.finder.seated) : undefined })
  }

  /** A fixed camera: is person `id` in their seat in frame `j` — or only
   *  doubted there for a moment (the segmenter unsure): seen sitting there
   *  both just before and just after? (Not someone coming back to their
   *  seat: they aren't shown before they're there.) */
  private inSeat(j: number, id: number) {
    const f = this.frames[j]
    if (f.seated?.has(id)) {
      this.lastSeated.set(id, f.ts)
      return true
    }
    const before = this.lastSeated.get(id) ?? -Infinity
    for (let k = j + 1; k < this.frames.length && this.frames[k].ts - before <= LEAVE_S && this.frames[k].shot === f.shot; k++) if (this.frames[k].seated?.has(id)) return true
    return false
  }

  /** How fast the head that turns up in frame `k` turns, run backwards
   *  (from its first moments back), deg/s. */
  private rateBack(k: number, id: number, yaw: number) {
    let m = -1
    for (let n = k + 1; n < this.frames.length && this.frames[n].ts - this.frames[k].ts <= TURN_READ_S && this.frames[n].shot === this.frames[k].shot; n++) if (this.frames[n].raw.has(id)) m = n
    if (m < 0 || this.frames[m].ts - this.frames[k].ts < 0.1) return 0
    return (yaw - yawOf(this.frames[m].raw.get(id)!, this.W, this.H)) / (this.frames[m].ts - this.frames[k].ts)
  }

  /** Bridges short gaps in frame `j` and filters it forwards. Needs the
   *  frames up to `gap` (and `turnAhead`) past it, or the end of the clip. */
  private fill(j: number) {
    const f = this.frames[j]
    const { W, H } = this
    const filled = new Map(f.raw)
    if (!this.an.fixed) {
      for (const [id, seen] of this.lastSeen) {
        const lost = f.ts - seen.ts
        if (filled.has(id) || lost > this.gap || seen.shot !== f.shot) continue
        // Back within the gap? Then they were here all along.
        for (let k = j + 1; k < this.frames.length && this.frames[k].ts - seen.ts <= this.gap && this.frames[k].shot === f.shot; k++) {
          const b = this.frames[k].raw.get(id)
          if (!b) continue
          filled.set(id, lerpFace(seen.lm, b, lost / Math.max(1e-6, this.frames[k].ts - seen.ts)))
          break
        }
      }
      for (const [id, lm] of f.raw) this.lastSeen.set(id, { ts: f.ts, shot: f.shot, lm })
    } else {
      // A fixed camera: whoever is still in their seat keeps their head on,
      // however long it's hidden (see Head). One that comes back soon is
      // eased across the whole gap; after longer, onto where it comes back
      // just before it does.
      for (const [id, h] of this.heads) {
        if (filled.has(id) || h.shot !== f.shot || !this.inSeat(j, id) || f.ts <= h.t) continue
        const held = h.at(f.ts, W, H)
        let k = j + 1
        while (k < this.frames.length && this.frames[k].ts - f.ts <= this.gap && this.frames[k].shot === f.shot && !this.frames[k].raw.has(id)) k++
        const b = k < this.frames.length && this.frames[k].ts - f.ts <= this.gap && this.frames[k].shot === f.shot ? this.frames[k].raw.get(id) : undefined
        const tk = b ? this.frames[k].ts : Infinity
        const from = tk - h.t <= this.gap ? h.t : tk - RETURN_BLEND_S
        if (!b || f.ts <= from) {
          filled.set(id, held)
          continue
        }
        const u = (f.ts - from) / Math.max(1e-6, tk - from)
        // (a head lost turning away went on round, out of sight, and came
        // back the way it's turning when it turns up)
        const yb = yawOf(b, W, H)
        const ya = yawOf(held, W, H)
        const yc = lostYaw(yb, this.rateBack(k, id, yb), tk - f.ts, HELD_MAX_DEG)
        const yaw = ya + (yc - ya) * smoothstep(u)
        filled.set(id, lerpFace(turnTo(held, yaw, W, H), turnTo(b, yaw, W, H), u))
      }
      for (const [id, lm] of f.raw) {
        let h = this.heads.get(id)
        if (!h) this.heads.set(id, (h = new Head()))
        h.inShot(f.shot, this.an.angleAt(f.ts))
        h.see(lm, f.ts, W, H)
        this.lastSeated.set(id, f.ts)
      }
    }
    const prev = this.lastFilled
    const cut = f.shot !== this.lastShot
    const fwd = new Map<number, Float32Array>()
    for (const [id, lm] of filled) {
      let s = this.euro.get(id)
      if (!s) this.euro.set(id, (s = new EuroState(this.an.fixed)))
      if (!prev?.has(id) || cut) s.reset()
      fwd.set(id, Float32Array.from(s.step(lm, f.ts)))
    }
    for (const id of [...this.euro.keys()]) if (!filled.has(id)) this.euro.delete(id)
    f.filled = filled
    f.fwd = fwd
    this.lastFilled = filled
    this.lastShot = f.shot
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
      while (e + 1 < this.filledTo && this.frames[e + 1].ts - f.ts <= SMOOTH_AHEAD_S && this.frames[e + 1].shot === f.shot && this.frames[e + 1].filled!.has(id)) e++
      this.back.reset()
      let b: Float32Array = lm
      for (let k = e; k >= 0; k--) b = this.back.step(this.frames[k].filled!.get(id)!, this.frames[k].ts)
      const fw = f.fwd.get(id)!
      const m = new Float32Array(FRAME)
      for (let c = 0; c < STRIDE; c++) m[c] = (fw[c] + b[c]) / 2
      m[STRIDE] = lm[STRIDE]
      m[STRIDE + 1] = lm[STRIDE + 1]
      out.push({ id, lm: this.an.fixed ? this.steady(id, f, m) : m })
    }
    return out
  }

  /** A fixed camera: the head's size and turn in frame `f` (the first in
   *  hand) taken from the frames either side as well — a median-led
   *  average for the size, which a seated head only changes by leaning; for
   *  the turn, an average close round the frame with any frame that turned
   *  far from the rest left out — and mesh `m` set in them. What a bad
   *  frame or two would make swell, shrink or swing, doesn't. */
  private steady(id: number, f: Scouted, m: Float32Array): Float32Array {
    const { W, H } = this
    const p0 = poseOf(f.filled!.get(id)!, W, H)
    let past = this.past.get(id)
    if (!past || (past.length && (f.ts - past[past.length - 1].ts > 0.15 || past[past.length - 1].shot !== f.shot))) this.past.set(id, (past = []))
    const pts: { dt: number; p: Pose }[] = past.filter((x) => f.ts - x.ts <= STEADY_S).map((x) => ({ dt: x.ts - f.ts, p: x.p }))
    pts.push({ dt: 0, p: p0 })
    for (let k = 1; k < this.filledTo && this.frames[k].ts - f.ts <= STEADY_S && this.frames[k].shot === f.shot; k++) {
      const lm = this.frames[k].filled?.get(id)
      if (!lm) break
      pts.push({ dt: this.frames[k].ts - f.ts, p: poseOf(lm, W, H) })
    }
    past.push({ ts: f.ts, shot: f.shot, p: p0 })
    while (past.length && f.ts - past[0].ts > STEADY_S) past.shift()
    if (pts.length < 3) return m
    const g = (dt: number, sd: number) => Math.exp(-0.5 * (dt / sd) ** 2)
    const ls = pts.map((x) => Math.log(x.p.s)).sort((a, b) => a - b)
    const med = ls[ls.length >> 1]
    let sw = 0
    let sv = 0
    for (const x of pts) {
      const v = Math.log(x.p.s)
      if (Math.abs(v - med) > 0.1) continue
      const w = g(x.dt, 0.18)
      sw += w
      sv += v * w
    }
    const near = pts.filter((x) => Math.abs(x.dt) <= 0.25)
    const q1 = averagePose(near.map((x) => ({ p: x.p, w: g(x.dt, 0.08) })))!.q
    const q = averagePose(near.filter((x) => quatAngle(x.p.q, q1) <= 20).map((x) => ({ p: x.p, w: g(x.dt, 0.08) })))?.q ?? q1
    const pm = poseOf(m, W, H)
    return meshOf({ o: pm.o, s: sw ? Math.exp(sv / sw) : pm.s, q }, localOf(m, pm, W, H), W, H, [m[STRIDE], m[STRIDE + 1]])
  }

  dispose() {
    void this.iter.return?.()
    this.input.dispose()
  }
}
