import { FRAME, N_LM, STRIDE } from './faces'

// A head's pose in a frame — where it is, how big, which way it faces — and
// its mesh in its own frame of reference (the face's shape and expression,
// with the pose taken out). A fixed camera works with heads this way: a
// mesh no head could have made (a hand taken for a face, a flip to the back
// of the head, a sudden swell) is caught by its pose; a head that can't be
// seen is held in its last pose with its usual face; and a pose is steadied
// on its own — size and turn more than place — then the mesh put back
// together from it.

export type V3 = [number, number, number]
/** x, y, z, w */
export type Quat = [number, number, number, number]
/** `o`: the bridge of the nose (px of the frame, z as x); `s`: the distance
 *  between the eyes (px); `q`: the head's turn. */
export interface Pose {
  o: V3
  s: number
  q: Quat
}

const pt = (lm: Float32Array, i: number, W: number, H: number): V3 => [lm[i * 3] * W, lm[i * 3 + 1] * H, lm[i * 3 + 2] * W]
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const len = (a: V3) => Math.hypot(a[0], a[1], a[2])
const unit = (a: V3): V3 => {
  const l = Math.max(1e-9, len(a))
  return [a[0] / l, a[1] / l, a[2] / l]
}
const mid = (a: V3, b: V3): V3 => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]

/** The rotation whose columns are r, u, f, as a quaternion. */
function quatFromBasis(r: V3, u: V3, f: V3): Quat {
  const [m00, m10, m20] = r
  const [m01, m11, m21] = u
  const [m02, m12, m22] = f
  const tr = m00 + m11 + m22
  let q: Quat
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2
    q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s / 4]
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2
    q = [s / 4, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2
    q = [(m01 + m10) / s, s / 4, (m12 + m21) / s, (m02 - m20) / s]
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2
    q = [(m02 + m20) / s, (m12 + m21) / s, s / 4, (m10 - m01) / s]
  }
  return normQ(q)
}

const normQ = (q: Quat): Quat => {
  const l = Math.max(1e-12, Math.hypot(q[0], q[1], q[2], q[3]))
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
}
const dotQ = (a: Quat, b: Quat) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

/** v rotated by q. */
function rot(q: Quat, v: V3): V3 {
  const [x, y, z, w] = q
  const t = cross([x, y, z], v).map((c) => c * 2) as V3
  const u = cross([x, y, z], t)
  return [v[0] + w * t[0] + u[0], v[1] + w * t[1] + u[1], v[2] + w * t[2] + u[2]]
}
const conj = (q: Quat): Quat => [-q[0], -q[1], -q[2], q[3]]

/** The angle between two turns, degrees. */
export const quatAngle = (a: Quat, b: Quat) => (2 * Math.acos(Math.min(1, Math.abs(dotQ(a, b)))) * 180) / Math.PI

/** The eyes (each the middle of its corners) and the face's up line. */
export function poseOf(lm: Float32Array, W: number, H: number): Pose {
  const eA = mid(pt(lm, 33, W, H), pt(lm, 133, W, H))
  const eB = mid(pt(lm, 263, W, H), pt(lm, 362, W, H))
  const across = sub(eB, eA)
  const s = Math.max(1e-3, len(across))
  const r = unit(across)
  const f = unit(cross(r, unit(sub(pt(lm, 10, W, H), pt(lm, 152, W, H)))))
  const u = unit(cross(f, r))
  return { o: pt(lm, 168, W, H), s, q: quatFromBasis(r, u, f) }
}

/** The mesh in its head's own frame, in eye distances. */
export function localOf(lm: Float32Array, p: Pose, W: number, H: number): Float32Array {
  const out = new Float32Array(STRIDE)
  const inv = conj(p.q)
  for (let i = 0; i < N_LM; i++) {
    const v = rot(inv, sub(pt(lm, i, W, H), p.o))
    out[i * 3] = v[0] / p.s
    out[i * 3 + 1] = v[1] / p.s
    out[i * 3 + 2] = v[2] / p.s
  }
  return out
}

/** A packed mesh (fractions of the frame) from a pose and a head-frame
 *  mesh; `eyes`: how shut each eye is. */
export function meshOf(p: Pose, local: Float32Array, W: number, H: number, eyes: [number, number] = [0, 0]): Float32Array {
  const out = new Float32Array(FRAME)
  for (let i = 0; i < N_LM; i++) {
    const v = rot(p.q, [local[i * 3] * p.s, local[i * 3 + 1] * p.s, local[i * 3 + 2] * p.s])
    out[i * 3] = (v[0] + p.o[0]) / W
    out[i * 3 + 1] = (v[1] + p.o[1]) / H
    out[i * 3 + 2] = (v[2] + p.o[2]) / W
  }
  out[STRIDE] = eyes[0]
  out[STRIDE + 1] = eyes[1]
  return out
}

/** Points that keep their place on a face whatever it does (forehead, nose,
 *  eye corners, cheekbones, temples): its build. */
const RIGID = [10, 151, 9, 168, 6, 197, 195, 5, 4, 33, 133, 263, 362, 234, 454, 93, 323, 127, 356, 116, 345]

/** How unlike each other two head-frame meshes are built (RMS over the
 *  rigid points, eye distances). A mesh laid over a hand, or crumpled by
 *  one, is far from the person's own. */
export function shapeDistance(a: Float32Array, b: Float32Array) {
  let sum = 0
  for (const i of RIGID) sum += (a[i * 3] - b[i * 3]) ** 2 + (a[i * 3 + 1] - b[i * 3 + 1]) ** 2 + (a[i * 3 + 2] - b[i * 3 + 2]) ** 2
  return Math.sqrt(sum / RIGID.length)
}

/** Weighted average of poses: place and (log) size averaged, turns averaged
 *  on the same side of the sphere. */
export function averagePose(items: { p: Pose; w: number }[]): Pose | null {
  let W = 0
  const o: V3 = [0, 0, 0]
  let ls = 0
  const q: Quat = [0, 0, 0, 0]
  const ref = items[0]?.p.q
  for (const { p, w } of items) {
    if (!(w > 0)) continue
    W += w
    for (let k = 0; k < 3; k++) o[k] += p.o[k] * w
    ls += Math.log(p.s) * w
    const sg = dotQ(p.q, ref!) < 0 ? -1 : 1
    for (let k = 0; k < 4; k++) q[k] += p.q[k] * w * sg
  }
  if (!W) return null
  return { o: [o[0] / W, o[1] / W, o[2] / W], s: Math.exp(ls / W), q: normQ(q) }
}

/** `a` toward `b` by `u` (0..1): place and size straight, turn the short way. */
export function lerpPose(a: Pose, b: Pose, u: number): Pose {
  let bq = b.q
  if (dotQ(a.q, bq) < 0) bq = [-bq[0], -bq[1], -bq[2], -bq[3]]
  const d = dotQ(a.q, bq)
  let q: Quat
  if (d > 0.9995) q = normQ([a.q[0] + (bq[0] - a.q[0]) * u, a.q[1] + (bq[1] - a.q[1]) * u, a.q[2] + (bq[2] - a.q[2]) * u, a.q[3] + (bq[3] - a.q[3]) * u])
  else {
    const th = Math.acos(d)
    const sa = Math.sin((1 - u) * th) / Math.sin(th)
    const sb = Math.sin(u * th) / Math.sin(th)
    q = normQ([a.q[0] * sa + bq[0] * sb, a.q[1] * sa + bq[1] * sb, a.q[2] * sa + bq[2] * sb, a.q[3] * sa + bq[3] * sb])
  }
  return { o: [a.o[0] + (b.o[0] - a.o[0]) * u, a.o[1] + (b.o[1] - a.o[1]) * u, a.o[2] + (b.o[2] - a.o[2]) * u], s: Math.exp(Math.log(a.s) + (Math.log(b.s) - Math.log(a.s)) * u), q }
}

/** Two head-frame meshes mixed (`u` of the way from a to b). */
export function lerpLocal(a: Float32Array, b: Float32Array, u: number, out = new Float32Array(STRIDE)) {
  for (let c = 0; c < STRIDE; c++) out[c] = a[c] + (b[c] - a[c]) * u
  return out
}

/** A face as the person usually looks: a slow running mean of their
 *  head-frame meshes. What a hidden head is held with — not the last frame
 *  before it went, often caught mid-word or crumpled by a hand. */
export class UsualFace {
  private sum: Float32Array | null = null
  private n = 0

  add(local: Float32Array) {
    if (!this.sum) this.sum = new Float32Array(STRIDE)
    // (a running mean of the first ~30, then slowly moving on)
    const k = 1 / Math.min(++this.n, 30)
    for (let c = 0; c < STRIDE; c++) this.sum[c] += (local[c] - this.sum[c]) * k
  }

  get face(): Float32Array | null {
    return this.n >= 3 ? this.sum : null
  }

  get count() {
    return this.n
  }
}
