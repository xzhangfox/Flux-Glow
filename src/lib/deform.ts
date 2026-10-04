import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { connectorsToLoop } from './landmarks'

// Where every Shape control moves the 468 face-mesh vertices (meshWarp.ts
// then renders the moved mesh). Two rules keep every slider natural even
// pinned at ±100:
//
// 1. Nothing moves a bare landmark loop on its own. Each control is a
//    smooth LOCAL transform — scale, shift or rotate a feature — whose
//    weight is 1 over the feature itself and eases to 0 (cosine²) across a
//    band of surrounding skin, so the skin around an enlarged eye stretches
//    with it instead of a thin ring of triangles getting crushed.
// 2. Every amount is bounded so the mapping stays one-to-one: a falloff band
//    of width B can absorb a displacement of up to ~0.64·B before any
//    triangle could invert, and the maxima below stay under half of that.
//
// Everything is measured in the face's own frame — `r` across the face,
// `d` down the midline (forehead→chin) — so a tilted or turned head is
// reshaped along its own axes, not the photo's. Distances are in
// "y-units" (normalized x scaled by the frame's aspect), so regions are
// true circles/ellipses on any aspect ratio.

export interface ReshapeParams {
  // Face (轮廓)
  /** Slim cheeks→chin: + narrows, − widens. */
  face: number
  /** Jaw angle (下颌/V脸): + narrows. */
  vJaw: number
  /** Chin length (下巴): + longer, − shorter. */
  chin: number
  /** Forehead (额头): + taller hairline, − lower. */
  forehead: number
  /** Temple (太阳穴): + narrows, − fuller. */
  temple: number
  /** Cheekbone (颧骨): + narrows, − wider. */
  cheekbone: number
  // Eyes
  /** Eye size: + enlarges. */
  eyes: number
  /** Eye width: + longer. */
  eyeWidth: number
  /** Eye height (opening): + rounder/more open. */
  eyeHeight: number
  /** Outer-corner tilt (眼角): + lifts outer corners. */
  eyeTilt: number
  /** Eye spacing (眼距): + further apart. */
  eyeDistance: number
  /** Eye position: + higher. */
  eyePosition: number
  // Brows
  /** Brow height: + raises. */
  eyebrowHeight: number
  /** Brow tilt (眉形): + lifts the tails. */
  browTilt: number
  /** Brow spacing (眉距): + further apart. */
  browDistance: number
  // Nose
  /** Overall nose size: + smaller (handled in reshape.ts). */
  nose: number
  /** Nose wings (鼻翼): + narrower. */
  noseWings: number
  /** Nose tip (鼻头): + smaller. */
  noseTip: number
  /** Nose length (鼻长): + longer. */
  noseLength: number
  /** Nose bridge (山根): + raises. */
  noseBridge: number
  // Mouth
  /** Mouth size: + larger. */
  mouth: number
  /** Mouth width: + wider. */
  mouthWidth: number
  /** Upper lip: + fuller. */
  mouthUpperLip: number
  /** Lower lip: + fuller. */
  mouthLowerLip: number
  /** Corners (嘴角): + lifts into a smile. */
  mouthCorners: number
  /** Mouth position (人中): + higher (shorter philtrum). */
  mouthPosition: number
  /** 3D relighting intensity, 0..1 (rendered in meshWarp's shader). */
  fillLight: number
}

export const SHAPE_PARAMS = [
  'face', 'vJaw', 'chin', 'forehead', 'temple', 'cheekbone',
  'eyes', 'eyeWidth', 'eyeHeight', 'eyeTilt', 'eyeDistance', 'eyePosition',
  'eyebrowHeight', 'browTilt', 'browDistance',
  'nose', 'noseWings', 'noseTip', 'noseLength', 'noseBridge',
  'mouth', 'mouthWidth', 'mouthUpperLip', 'mouthLowerLip', 'mouthCorners', 'mouthPosition',
] as const satisfies readonly (keyof ReshapeParams)[]

export const OVAL_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
const LEFT_EYE_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
const RIGHT_EYE_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)
const LEFT_BROW_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW)
const RIGHT_BROW_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW)
export const LIPS_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LIPS)

// Indices verified by plotting them on a real photo (not taken on citation
// alone): 168 bridge between the eyes, 6 just below it, 4 nose tip,
// 2 under the nose, 98/327 outer nose wings, 0/17 lip top/bottom, 61/291
// mouth corners, 152 chin, 10 top of the forehead, 234/454 widest cheek.
export const LM = {
  bridgeTop: 168, bridge: 6, noseTip: 4, subnasale: 2, wingL: 98, wingR: 327,
  lipTop: 0, lipBottom: 17, mouthL: 61, mouthR: 291, chin: 152, forehead: 10,
  cheekL: 234, cheekR: 454,
} as const

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
/** Signed slider value → amount, with separate maxima for each direction. */
const amt = (p: number, posMax: number, negMax: number) => (p >= 0 ? Math.min(p, 1) * posMax : Math.max(p, -1) * negMax)
const smoothstep = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a), 0, 1)
  return t * t * (3 - 2 * t)
}
/** Cosine² bump: 1 at `c`, 0 at distance ≥ `hw`. */
const bump = (x: number, c: number, hw: number) => {
  const d = Math.abs(x - c) / hw
  return d >= 1 ? 0 : Math.cos(d * Math.PI / 2) ** 2
}

interface V2 {
  x: number
  y: number
}

export interface FaceFrame {
  /** Points in y-units (x·aspect, y). */
  P: V2[]
  /** Unit vector across the face (towards image right) and down the midline. */
  r: V2
  d: V2
  /** Midline origin (between the eyes) and its length to the chin. */
  M: V2
  L: number
  /** Distance between the eye centers. */
  E: number
  aspect: number
}

export function faceFrame(landmarks: NormalizedLandmark[], aspect: number): FaceFrame {
  const P = landmarks.map((l) => ({ x: l.x * aspect, y: l.y }))
  const M = P[LM.bridgeTop]
  const chin = P[LM.chin]
  const L = Math.hypot(chin.x - M.x, chin.y - M.y) || 1e-6
  const d = { x: (chin.x - M.x) / L, y: (chin.y - M.y) / L }
  const r = { x: d.y, y: -d.x }
  const le = centroid(LEFT_EYE_LOOP, P)
  const re = centroid(RIGHT_EYE_LOOP, P)
  return { P, r, d, M, L, E: Math.hypot(le.x - re.x, le.y - re.y), aspect }
}

function centroid(loop: number[], P: V2[]): V2 {
  let x = 0
  let y = 0
  for (const i of loop) {
    x += P[i].x
    y += P[i].y
  }
  return { x: x / loop.length, y: y / loop.length }
}

/** Returns each vertex's target position in normalized image coordinates. */
export function deformTargets(landmarks: NormalizedLandmark[], params: ReshapeParams, aspect: number): Float32Array {
  const F = faceFrame(landmarks, aspect)
  const { P, r, d, M, L, E } = F
  const n = P.length
  const du = new Float32Array(n) // displacement along r
  const dv = new Float32Array(n) // displacement along d
  const U = new Float32Array(n)
  const V = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const x = P[i].x - M.x
    const y = P[i].y - M.y
    U[i] = x * r.x + y * r.y
    V[i] = (x * d.x + y * d.y) / L // 0 between the eyes, 1 at the chin
  }
  const u = (p: V2) => (p.x - M.x) * r.x + (p.y - M.y) * r.y
  const v = (p: V2) => ((p.x - M.x) * d.x + (p.y - M.y) * d.y) / L
  const mid = (a: V2, b: V2) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 })

  // Ellipse-shaped region around `c` (semi-axes ax across, ay down), full
  // weight inside, cosine² falloff out to `outer`× the ellipse.
  const weight = (i: number, cu: number, cv: number, ax: number, ay: number, outer: number) => {
    const rho = Math.hypot((U[i] - cu) / ax, ((V[i] - cv) * L) / ay)
    if (rho <= 1) return 1
    if (rho >= outer) return 0
    return Math.cos(((rho - 1) / (outer - 1)) * (Math.PI / 2)) ** 2
  }
  // Scale (and optionally rotate) a region about its center.
  const scaleRegion = (c: V2, ax: number, ay: number, outer: number, sx: number, sy: number, rot = 0) => {
    if (Math.abs(sx - 1) < 1e-4 && Math.abs(sy - 1) < 1e-4 && Math.abs(rot) < 1e-4) return
    const cu = u(c)
    const cv = v(c)
    const cs = Math.cos(rot)
    const sn = Math.sin(rot)
    for (let i = 0; i < n; i++) {
      const w = weight(i, cu, cv, ax, ay, outer)
      if (!w) continue
      const lu = U[i] - cu
      const lv = (V[i] - cv) * L
      const su = lu * sx
      const sv = lv * sy
      du[i] += (su * cs - sv * sn - lu) * w
      dv[i] += (su * sn + sv * cs - lv) * w
    }
  }
  const shiftRegion = (c: V2, ax: number, ay: number, outer: number, mu: number, mv: number) => {
    if (Math.abs(mu) < 1e-6 && Math.abs(mv) < 1e-6) return
    const cu = u(c)
    const cv = v(c)
    for (let i = 0; i < n; i++) {
      const w = weight(i, cu, cv, ax, ay, outer)
      if (!w) continue
      du[i] += mu * w
      dv[i] += mv * w
    }
  }

  // ---- Face contour: row-wise horizontal compression toward the
  // midline, stronger the further a vertex sits from it (u' = u·(1 − k·w·t),
  // t = |u| / that side's half-width). Monotonic for k < 0.4, and each
  // side is normalized by its own half-width, so a turned face slims both
  // sides in proportion rather than over-pulling the near cheek.
  const hwL = Math.max(1e-6, -u(P[LM.cheekL]))
  const hwR = Math.max(1e-6, u(P[LM.cheekR]))
  // Fade horizontal contour edits as the head turns far from frontal —
  // past ~35% yaw the far side is too foreshortened to reshape cleanly.
  const yaw = Math.abs(hwL - hwR) / (hwL + hwR)
  const yawFalloff = 1 - smoothstep(0.35, 0.6, yaw)
  const contour = [
    { k: amt(params.face, 0.075, 0.05), profile: (vv: number) => smoothstep(-0.05, 0.5, vv) },
    { k: amt(params.vJaw, 0.07, 0.05), profile: (vv: number) => bump(vv, 0.8, 0.32) },
    { k: amt(params.cheekbone, 0.055, 0.045), profile: (vv: number) => bump(vv, 0.12, 0.3) },
    { k: amt(params.temple, 0.055, 0.045), profile: (vv: number) => bump(vv, -0.42, 0.3) },
  ]
  for (const { k, profile } of contour) {
    if (Math.abs(k) < 1e-4) continue
    for (let i = 0; i < n; i++) {
      const w = profile(V[i])
      if (!w) continue
      const t = Math.min(1.2, Math.abs(U[i]) / (U[i] < 0 ? hwL : hwR))
      du[i] -= U[i] * k * w * t * yawFalloff
    }
  }
  // Chin length and forehead height: vertical stretch of just that end.
  const chinK = amt(params.chin, 0.07, 0.05)
  const browK = amt(params.forehead, 0.06, 0.045)
  if (Math.abs(chinK) > 1e-4 || Math.abs(browK) > 1e-4) {
    for (let i = 0; i < n; i++) {
      dv[i] += (chinK * smoothstep(0.6, 1.0, V[i]) - browK * smoothstep(0.2, 0.8, -V[i])) * L
    }
  }

  // ---- Eyes (each eye in its own region; mirrored for tilt/spacing).
  for (const loop of [LEFT_EYE_LOOP, RIGHT_EYE_LOOP]) {
    const c = centroid(loop, P)
    const us = loop.map((i) => U[i])
    const ew = (Math.max(...us) - Math.min(...us)) / 2
    const side = Math.sign(u(c)) || 1 // −1 image-left eye, +1 image-right eye
    const ax = ew * 1.1
    const ay = ew * 0.62
    scaleRegion(c, ax, ay, 2.0, 1 + amt(params.eyes, 0.11, 0.08) + amt(params.eyeWidth, 0.08, 0.07), 1 + amt(params.eyes, 0.09, 0.07) + amt(params.eyeHeight, 0.12, 0.09), -side * amt(params.eyeTilt, 0.1, 0.08))
    shiftRegion(c, ax, ay, 2.3, side * amt(params.eyeDistance, 0.1, 0.08) * ew, -amt(params.eyePosition, 0.1, 0.1) * ew)
  }

  // ---- Brows.
  for (const loop of [LEFT_BROW_LOOP, RIGHT_BROW_LOOP]) {
    const c = centroid(loop, P)
    const us = loop.map((i) => U[i])
    const bw = (Math.max(...us) - Math.min(...us)) / 2
    const side = Math.sign(u(c)) || 1
    const ax = bw * 1.05
    const ay = bw * 0.3
    scaleRegion(c, ax, ay, 2.0, 1, 1, -side * amt(params.browTilt, 0.11, 0.09))
    shiftRegion(c, ax, ay, 2.0, side * amt(params.browDistance, 0.035, 0.03) * E, -amt(params.eyebrowHeight, 0.04, 0.035) * E)
  }

  // ---- Nose (overall size stays the CPU radial pass in reshape.ts).
  const wingL = P[LM.wingL]
  const wingR = P[LM.wingR]
  const aw = Math.hypot(wingL.x - wingR.x, wingL.y - wingR.y) / 2
  scaleRegion(mid(wingL, wingR), aw * 1.1, aw * 0.55, 2.0, 1 - amt(params.noseWings, 0.13, 0.1), 1)
  const tipScale = 1 - amt(params.noseTip, 0.12, 0.1)
  scaleRegion(P[LM.noseTip], aw * 0.45, aw * 0.45, 2.4, tipScale, tipScale)
  shiftRegion(mid(P[LM.noseTip], P[LM.subnasale]), aw * 1.1, aw * 0.6, 2.2, 0, amt(params.noseLength, 0.03, 0.025) * E)
  shiftRegion(P[LM.bridge], E * 0.1, E * 0.1, 2.2, 0, -amt(params.noseBridge, 0.03, 0.025) * E)

  // ---- Mouth.
  const mL = P[LM.mouthL]
  const mR = P[LM.mouthR]
  const mw = Math.hypot(mL.x - mR.x, mL.y - mR.y) / 2
  const lips = centroid(LIPS_LOOP, P)
  const mScale = amt(params.mouth, 0.1, 0.08)
  scaleRegion(lips, mw * 1.05, mw * 0.5, 1.9, 1 + mScale + amt(params.mouthWidth, 0.08, 0.07), 1 + mScale)
  shiftRegion(lips, mw * 1.1, mw * 0.55, 2.2, 0, -amt(params.mouthPosition, 0.025, 0.02) * E)
  const cornerLift = -amt(params.mouthCorners, 0.018, 0.015) * E
  for (const corner of [mL, mR]) shiftRegion(corner, mw * 0.22, mw * 0.22, 2.4, 0, cornerLift)
  // Lip fullness moves the outer lip contour only (the inner lip line
  // stays), which is what reads as fuller rather than shifted.
  const upper = amt(params.mouthUpperLip, 0.018, 0.015) * E
  const lower = amt(params.mouthLowerLip, 0.018, 0.015) * E
  if (Math.abs(upper) > 1e-6 || Math.abs(lower) > 1e-6) {
    const cv = v(lips)
    for (const i of LIPS_LOOP) {
      if (V[i] < cv) dv[i] -= upper
      else dv[i] += lower
    }
  }

  const out = new Float32Array(n * 2)
  for (let i = 0; i < n; i++) {
    out[i * 2] = (P[i].x + du[i] * r.x + dv[i] * d.x) / aspect
    out[i * 2 + 1] = P[i].y + du[i] * r.y + dv[i] * d.y
  }
  return out
}
