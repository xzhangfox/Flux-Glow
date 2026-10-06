import * as THREE from 'three'
import { FaceLandmarker } from '@mediapipe/tasks-vision'
import { FACE_TRIANGULATION } from '../faceTriangulation'
import { connectorsToLoop } from '../landmarks'
import type { Model, Rig } from './scene'

// Head-covering masks (Spider's hood, Bat's cowl), built every frame to fit
// the face and head they're worn on:
//
// • The face part is the live 468-point face mesh, smoothed (stretched
//   fabric or a moulded shell bridges the eye sockets instead of following
//   every contour) and lifted off the skin by the mask's thickness.
// • Past the face's outline the mask continues as a hood: each outline
//   point sweeps back round the head along a path whose points are traced
//   onto a head shape — skull, jaw, neck and the ears under the mask — so
//   the hood hugs a real head, not the hair.
// • Hair the photo shows outside the mask is painted out (see hair.ts).
//
// All in the scene's head rig: origin between the eyes, 1 unit = the
// distance between the eyes, x to image right, y up the face, z out of it.

const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z)
const ss = THREE.MathUtils.smoothstep
type V3 = THREE.Vector3

const OVAL = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
const LEFT_EYE = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
const RIGHT_EYE = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)
const N = 468
const RINGS = 18
const VERTS = N + OVAL.length * RINGS

/** The axis "inside" the head and neck that every surface faces away from. */
const insideOf = (p: V3, out: V3) => out.set(0, Math.min(p.y, 0.3), -1.35)

// ---- The head under the mask (signed distance, rig units) -------------------

function smin(a: number, b: number, k: number) {
  const h = Math.max(k - Math.abs(a - b), 0) / k
  return Math.min(a, b) - h * h * k * 0.25
}

function ellipsoid(x: number, y: number, z: number, rx: number, ry: number, rz: number) {
  const k0 = Math.hypot(x / rx, y / ry, z / rz)
  const k1 = Math.hypot(x / (rx * rx), y / (ry * ry), z / (rz * rz))
  return k1 > 1e-9 ? (k0 * (k0 - 1)) / k1 : -Math.min(rx, ry, rz)
}

// Skull: an egg, rounder at the forehead than at the back.
interface Skull {
  y: number
  z: number
  rx: number
  ry: number
  front: number
  back: number
}
const NECK = { z: -1.35, rx: 0.82, rz: 0.74, bottom: -4.2 }
// The hood's paths end a little below the hem, which is then cut clean
// (antialiased) in the shader.
const HEM_Y = -3.05
// Spider ends at the jaw, leaving the wearer's own neck: just under the
// chin at the front, rising round under the ears to the nape.
const SPIDER_HEM = 'vLocal.y - (-1.98 + 0.63 * (1.0 - smoothstep(-1.8, -0.3, vLocal.z)))'

interface HoodSpec {
  /** The cranium under the mask: an egg, rounder at the forehead. Masks
   *  make it a little taller and rounder than a bare skull — the head a
   *  hood is pulled over, hair and all. */
  skull: Skull
  /** Thickness over the head. */
  lift: number
  /** How far the ears under the mask push it out. */
  ears: number
  /** Half-width of the jaw behind the face (default 0.98). */
  jaw?: number
}

function headSdf(x: number, y: number, z: number, spec: HoodSpec) {
  const S = spec.skull
  const zs = z - S.z
  let d = ellipsoid(x, y - S.y, zs, S.rx, S.ry, zs > 0 ? S.front : S.back)
  // Jaw and the underside of the head, behind the face.
  d = smin(d, ellipsoid(x, y + 0.85, z + 1.05, spec.jaw ?? 0.98, 0.92, 0.95), 0.45)
  // Neck (an elliptic column), stopping well below the hem.
  const neck = Math.max((Math.hypot(x / NECK.rx, (z - NECK.z) / NECK.rz) - 1) * NECK.rz, NECK.bottom - y, y + 0.3)
  d = smin(d, neck, 0.55)
  // The ears, flattened under the mask.
  if (spec.ears > 0) for (const s of [-1, 1]) d = smin(d, ellipsoid(x - s * (1.12 + spec.ears * 0.4), y + 0.38, z + 1.5, spec.ears, 0.5, 0.4), 0.28)
  return d - spec.lift
}

const ORIGIN = V(0, -0.3, -1.4)

/** The outermost point of the head along `dir` from ORIGIN. */
function trace(dir: V3, spec: HoodSpec, out: V3) {
  let t = 7
  for (let i = 0; i < 80; i++) {
    const d = headSdf(ORIGIN.x + dir.x * t, ORIGIN.y + dir.y * t, ORIGIN.z + dir.z * t, spec)
    if (d < 1e-3) break
    t -= Math.max(d, 1e-3)
  }
  return out.copy(dir).multiplyScalar(t).add(ORIGIN)
}

function slerpDir(a: V3, b: V3, s: number, out: V3) {
  const dot = THREE.MathUtils.clamp(a.dot(b), -1, 1)
  const ang = Math.acos(dot)
  if (ang < 1e-4) return out.copy(a)
  const sa = Math.sin(ang)
  return out
    .copy(a)
    .multiplyScalar(Math.sin((1 - s) * ang) / sa)
    .addScaledVector(b, Math.sin(s * ang) / sa)
    .normalize()
}

/** Where each outline point's hood path ends: u = 0 (top of the face) →
 *  over the crown to the back of the head; u = 0.5 (the side, at the
 *  ear) → the nape; then down the back of the neck and round the hem to
 *  the front of the neck (u = 1, under the chin). */
function hoodTarget(u: number, side: number, SKULL: Skull, out: V3) {
  if (u <= 0.5) {
    const th = THREE.MathUtils.lerp(0.15, -0.95, u / 0.5)
    return out.set(0, SKULL.y + SKULL.ry * Math.sin(th), SKULL.z - SKULL.back * Math.cos(th))
  }
  if (u <= 0.62) {
    const t = (u - 0.5) / 0.12
    return out.set(0, THREE.MathUtils.lerp(-0.78, HEM_Y, t), THREE.MathUtils.lerp(-2.4, NECK.z - NECK.rz, t))
  }
  const phi = Math.PI * (1 - (u - 0.62) / 0.38)
  return out.set(side * NECK.rx * Math.sin(phi), HEM_Y, NECK.z + NECK.rz * Math.cos(phi))
}

// ---- Geometry ----------------------------------------------------------------

let neighbours: number[][] | null = null
function faceNeighbours() {
  if (neighbours) return neighbours
  const sets = Array.from({ length: N }, () => new Set<number>())
  for (let t = 0; t < FACE_TRIANGULATION.length; t += 3) {
    const [a, b, c] = [FACE_TRIANGULATION[t], FACE_TRIANGULATION[t + 1], FACE_TRIANGULATION[t + 2]]
    sets[a].add(b).add(c)
    sets[b].add(a).add(c)
    sets[c].add(a).add(b)
  }
  neighbours = sets.map((s) => [...s])
  return neighbours
}

function hoodGeometry() {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(VERTS * 3), 3))
  g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(VERTS * 3), 3))
  g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(VERTS * 2), 2))
  g.setAttribute('aLocal', new THREE.BufferAttribute(new Float32Array(VERTS * 3), 3))
  const face: number[] = Array.from(FACE_TRIANGULATION)
  const ringIdx: number[] = []
  const ring = (k: number, o: number) => (k === 0 ? OVAL[o] : N + (k - 1) * OVAL.length + o)
  for (let k = 0; k < RINGS; k++)
    for (let o = 0; o < OVAL.length; o++) {
      const o2 = (o + 1) % OVAL.length
      ringIdx.push(ring(k, o), ring(k + 1, o), ring(k, o2), ring(k, o2), ring(k + 1, o), ring(k + 1, o2))
    }
  const flip = (a: number[]) => {
    const f = a.slice()
    for (let t = 0; t < f.length; t += 3) [f[t + 1], f[t + 2]] = [f[t + 2], f[t + 1]]
    return f
  }
  // The face mesh and the hood can each come out wound either way
  // (mirroring flips the face; the rings' order follows the outline's
  // direction), so all four combinations are prepared and picked per frame.
  g.userData.ringIdx = ringIdx
  g.userData.windings = new Map<string, THREE.BufferAttribute>()
  for (const [fk, fa] of [['f', face], ['F', flip(face)]] as const)
    for (const [rk, ra] of [['r', ringIdx], ['R', flip(ringIdx)]] as const) g.userData.windings.set(fk + rk, new THREE.Uint32BufferAttribute([...fa, ...ra], 1))
  g.setIndex(g.userData.windings.get('fr'))
  return g
}

/** Cylindrical UVs round the head and down the neck, mirrored about the
 *  face's centre line: the textures are symmetric, and this way the
 *  wrap-around has no seam at the back. */
function uvOf(p: V3): [number, number] {
  const a = Math.abs(Math.atan2(p.x, p.z - NECK.z))
  return [0.5 + (a / Math.PI) * 0.9, (p.y + 3.3) / 5.6]
}

interface Fit {
  /** The bare face (rig units). */
  L: V3[]
  /** The mask's surface: face part first (N points), then the hood rings. */
  P: V3[]
  /** Outward normals at P. */
  normals: V3[]
  /** The mask's triangles (indices into P). */
  tris: ArrayLike<number>
  /** Whether the face is mirror-imaged (a mirrored selfie). */
  mirrored: boolean
}

interface MaskShape extends HoodSpec {
  /** Lift off the face (rig units), optionally varying over it. */
  faceLift: (p: V3) => number
  /** Laplacian smoothing passes over the face before lifting. */
  smooth: number
}

/** Rebuilds the mask surface from this frame's landmarks. */
function fitMask(g: THREE.BufferGeometry, rig: Rig, shape: MaskShape): Fit {
  const raw: V3[] = Array.from({ length: N }, (_, i) => rig.local(i))
  let L = raw
  const nb = faceNeighbours()
  for (let it = 0; it < shape.smooth; it++) {
    L = L.map((p, i) => {
      const avg = nb[i].reduce((s, j) => s.add(L[j]), new THREE.Vector3()).divideScalar(nb[i].length || 1)
      return p.clone().lerp(avg, 0.5)
    })
  }
  // Normals of the (smoothed) face, oriented away from the head.
  const nrm = Array.from({ length: N }, () => new THREE.Vector3())
  const e1 = new THREE.Vector3()
  const e2 = new THREE.Vector3()
  const fn = new THREE.Vector3()
  const c = new THREE.Vector3()
  for (let t = 0; t < FACE_TRIANGULATION.length; t += 3) {
    const a = FACE_TRIANGULATION[t]
    const b = FACE_TRIANGULATION[t + 1]
    const cc = FACE_TRIANGULATION[t + 2]
    e1.subVectors(L[b], L[a])
    e2.subVectors(L[cc], L[a])
    fn.crossVectors(e1, e2)
    nrm[a].add(fn)
    nrm[b].add(fn)
    nrm[cc].add(fn)
  }
  for (let i = 0; i < N; i++) {
    nrm[i].normalize()
    if (nrm[i].dot(e1.subVectors(L[i], insideOf(L[i], c))) < 0) nrm[i].negate()
  }
  // Smoothing shrinks convex parts (forehead, cheekbones, nose) below the
  // real skin, which would then poke through. Push the smoothed surface
  // back out by however far the face sits above it, spread over a couple
  // of rings so the correction itself stays smooth.
  let push = new Float32Array(N)
  if (L !== raw) {
    for (let i = 0; i < N; i++) push[i] = Math.max(0, e1.subVectors(raw[i], L[i]).dot(nrm[i]))
    for (let it = 0; it < 3; it++) {
      const next = new Float32Array(N)
      for (let i = 0; i < N; i++) {
        let m = push[i]
        let s = push[i]
        for (const j of nb[i]) {
          m = Math.max(m, push[j])
          s += push[j]
        }
        next[i] = it < 2 ? m : s / (nb[i].length + 1)
      }
      push = next
    }
  }
  const P: V3[] = []
  for (let i = 0; i < N; i++) P.push(L[i].clone().addScaledVector(nrm[i], shape.faceLift(L[i]) + push[i]))

  // The hood: from each outline point round the head to its target, the
  // path's points traced onto the head. The first stretch eases from the
  // face's own edge onto the traced head, so they meet without a step.
  const d0 = new THREE.Vector3()
  const dT = new THREE.Vector3()
  const d = new THREE.Vector3()
  const q = new THREE.Vector3()
  const q0 = new THREE.Vector3()
  const off = new THREE.Vector3()
  const rings: V3[][] = Array.from({ length: RINGS }, () => [])
  for (const i of OVAL) {
    const e = P[i]
    const u = Math.atan2(Math.abs(raw[i].x), raw[i].y + 0.5) / Math.PI
    d0.subVectors(e, ORIGIN).normalize()
    dT.subVectors(hoodTarget(u, Math.sign(raw[i].x) || 1, shape.skull, dT), ORIGIN).normalize()
    trace(d0, shape, q0)
    off.subVectors(e, q0)
    for (let k = 1; k <= RINGS; k++) {
      const s = k / RINGS
      trace(slerpDir(d0, dT, s, d), shape, q)
      rings[k - 1].push(q.clone().addScaledVector(off, 1 - ss(s, 0, 0.4)))
    }
  }
  for (const r of rings) P.push(...r)

  const pos = g.getAttribute('position') as THREE.BufferAttribute
  const uv = g.getAttribute('uv') as THREE.BufferAttribute
  const local = g.getAttribute('aLocal') as THREE.BufferAttribute
  P.forEach((p, i) => {
    pos.setXYZ(i, p.x, p.y, p.z)
    const [uu, vv] = uvOf(p)
    uv.setXY(i, uu, vv)
    const ql = i < N ? raw[i] : p
    local.setXYZ(i, ql.x, ql.y, ql.z)
  })
  // A mirrored selfie reverses the face mesh's triangle winding, which makes
  // the renderer treat the outer surface as a back face and light it inside-
  // out. Pick whichever winding faces outward this frame.
  const facing = (idx: ArrayLike<number>) => {
    let sum = 0
    for (let t = 0; t < idx.length; t += 3) {
      const a = P[idx[t]]
      e1.subVectors(P[idx[t + 1]], a)
      e2.subVectors(P[idx[t + 2]], a)
      sum += fn.crossVectors(e1, e2).dot(e1.subVectors(a, insideOf(a, c)))
    }
    return sum
  }
  const key = (facing(FACE_TRIANGULATION) >= 0 ? 'f' : 'F') + (facing(g.userData.ringIdx) >= 0 ? 'r' : 'R')
  const winding = g.userData.windings.get(key)
  if (g.index !== winding) g.setIndex(winding)
  g.computeVertexNormals()
  // Belt and braces: keep normals pointing outward.
  const nAttr = g.getAttribute('normal') as THREE.BufferAttribute
  const normals: V3[] = []
  for (let i = 0; i < VERTS; i++) {
    const v = new THREE.Vector3().fromBufferAttribute(nAttr, i)
    if (v.dot(e1.subVectors(P[i], insideOf(P[i], c))) < 0) {
      v.negate()
      nAttr.setXYZ(i, v.x, v.y, v.z)
    }
    normals.push(v)
  }
  pos.needsUpdate = true
  uv.needsUpdate = true
  local.needsUpdate = true
  nAttr.needsUpdate = true
  g.computeBoundingSphere()
  return { L: raw, P, normals, tris: g.index!.array, mirrored: raw[33].x > raw[263].x }
}

/** Eye centres (bare face, rig units), left to right in the image. */
function eyeCentres(L: V3[]) {
  const eyes = [LEFT_EYE, RIGHT_EYE].map((loop) => loop.reduce((s, i) => s.add(L[i]), new THREE.Vector3()).divideScalar(loop.length))
  return eyes.sort((a, b) => a.x - b.x)
}

// ---- Decals bound to the mask -------------------------------------------------
//
// Parts that lie on the mask (Spider's lenses and their rims) are meshes
// whose every vertex is pinned to a point of the mask's face surface — a
// triangle and barycentric weights, found once — plus a height along its
// normal. They move and bend with the fabric exactly, at any angle.

interface Pin {
  v: [number, number, number]
  b: [number, number, number]
  h: number
}

class Decal {
  readonly geo = new THREE.BufferGeometry()
  private pins: Pin[] | null = null
  private boundMirrored = false
  private readonly at: (fit: Fit) => { x: number; y: number; h: number }[]
  constructor(at: (fit: Fit) => { x: number; y: number; h: number }[], index: number[], uv: number[], groups: [number, number, number][]) {
    this.at = at
    const n = uv.length / 2
    this.geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3))
    this.geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2))
    this.geo.setIndex(index)
    for (const [s, c, m] of groups) this.geo.addGroup(s, c, m)
  }

  private bind(fit: Fit) {
    const P = fit.P
    const T = fit.tris
    this.pins = this.at(fit).map(({ x, y, h }) => {
      let best: Pin = { v: [0, 0, 0], b: [1, 0, 0], h }
      let bestScore = -Infinity
      let bestZ = -Infinity
      for (let t = 0; t < T.length; t += 3) {
        const a = P[T[t]]
        const b = P[T[t + 1]]
        const c = P[T[t + 2]]
        const det = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)
        if (Math.abs(det) < 1e-9) continue
        const w0 = ((b.y - c.y) * (x - c.x) + (c.x - b.x) * (y - c.y)) / det
        const w1 = ((c.y - a.y) * (x - c.x) + (a.x - c.x) * (y - c.y)) / det
        const w2 = 1 - w0 - w1
        const score = Math.min(w0, w1, w2)
        const z = w0 * a.z + w1 * b.z + w2 * c.z
        // Inside a triangle: the front-most one (the face, not the back of
        // the head behind it) wins; otherwise the nearest.
        if ((score >= -1e-6 && (bestScore < -1e-6 || z > bestZ)) || (bestScore < -1e-6 && score > bestScore)) {
          bestScore = score
          bestZ = z
          best = { v: [T[t], T[t + 1], T[t + 2]], b: [w0, w1, w2], h }
        }
      }
      if (bestScore < 0) {
        // Outside the face: clamp onto the nearest triangle.
        const b = best.b.map((v) => Math.max(0, v))
        const s = b[0] + b[1] + b[2] || 1
        best.b = [b[0] / s, b[1] / s, b[2] / s]
      }
      return best
    })
  }

  update(fit: Fit) {
    // Bound to landmarks, a decal follows them into a mirror image —
    // where its shape would be the wrong way round. Bind afresh then.
    if (!this.pins || this.boundMirrored !== fit.mirrored) {
      this.bind(fit)
      this.boundMirrored = fit.mirrored
    }
    const pos = this.geo.getAttribute('position') as THREE.BufferAttribute
    const p = new THREE.Vector3()
    const n = new THREE.Vector3()
    this.pins!.forEach((pin, i) => {
      p.set(0, 0, 0)
      n.set(0, 0, 0)
      for (let k = 0; k < 3; k++) {
        p.addScaledVector(fit.P[pin.v[k]], pin.b[k])
        n.addScaledVector(fit.normals[pin.v[k]], pin.b[k])
      }
      p.addScaledVector(n.normalize(), pin.h)
      pos.setXYZ(i, p.x, p.y, p.z)
    })
    pos.needsUpdate = true
    // Keep the triangles wound to face outward (a mirror flips them).
    const idx = this.geo.index!
    const a = new THREE.Vector3().fromBufferAttribute(pos, idx.getX(0))
    const e1 = new THREE.Vector3().fromBufferAttribute(pos, idx.getX(1)).sub(a)
    const e2 = new THREE.Vector3().fromBufferAttribute(pos, idx.getX(2)).sub(a)
    let facing = 0
    for (let t = 0; t < idx.count; t += 3) {
      a.fromBufferAttribute(pos, idx.getX(t))
      e1.fromBufferAttribute(pos, idx.getX(t + 1)).sub(a)
      e2.fromBufferAttribute(pos, idx.getX(t + 2)).sub(a)
      const pin = this.pins![idx.getX(t)]
      n.set(0, 0, 0)
      for (let k = 0; k < 3; k++) n.addScaledVector(fit.normals[pin.v[k]], pin.b[k])
      facing += e1.cross(e2).dot(n)
    }
    if (facing < 0) {
      const arr = idx.array as Uint16Array | Uint32Array
      for (let t = 0; t < arr.length; t += 3) [arr[t + 1], arr[t + 2]] = [arr[t + 2], arr[t + 1]]
      idx.needsUpdate = true
    }
    this.geo.computeVertexNormals()
    this.geo.computeBoundingSphere()
  }
}

/** Points along a closed outline made of cubic Bézier segments
 *  [x0,y0, c1x,c1y, c2x,c2y] (each ending where the next begins). */
function bezierLoop(segs: number[][], per: number) {
  const pts: THREE.Vector2[] = []
  for (let s = 0; s < segs.length; s++) {
    const [x0, y0, ax, ay, bx, by] = segs[s]
    const [x1, y1] = segs[(s + 1) % segs.length]
    for (let i = 0; i < per; i++) {
      const t = i / per
      const u = 1 - t
      pts.push(
        new THREE.Vector2(
          u * u * u * x0 + 3 * u * u * t * ax + 3 * u * t * t * bx + t * t * t * x1,
          u * u * u * y0 + 3 * u * u * t * ay + 3 * u * t * t * by + t * t * t * y1,
        ),
      )
    }
  }
  return pts
}

/** Outward offset of a closed outline (counter-clockwise points). */
function offsetLoop(pts: THREE.Vector2[], by: number) {
  const n = pts.length
  return pts.map((p, i) => {
    const a = pts[(i - 1 + n) % n]
    const b = pts[(i + 1) % n]
    const t = new THREE.Vector2().subVectors(b, a).normalize()
    return new THREE.Vector2(p.x + t.y * by, p.y - t.x * by)
  })
}

const signedArea = (pts: THREE.Vector2[]) => pts.reduce((s, p, i) => s + p.x * pts[(i + 1) % pts.length].y - pts[(i + 1) % pts.length].x * p.y, 0) / 2

/** A raised rim round an opening, and the (domed) panel filling it, as one
 *  decal: material 0 = rim, 1 = panel. `outline`: the opening, in a frame
 *  placed per fit by `place` (2D point → rig x, y). */
function rimmedPanel(
  outline: THREE.Vector2[],
  rim: [number, number][],
  panelH: number,
  dome: number,
  place: (fit: Fit, p: THREE.Vector2) => { x: number; y: number },
  mirror: boolean,
) {
  const ccw = signedArea(outline) > 0 ? outline : outline.slice().reverse()
  const M = ccw.length
  const centre = ccw.reduce((s, p) => s.add(p), new THREE.Vector2()).divideScalar(M)
  // Rim rings: [outward offset, height] from the outer edge inward.
  const rings = rim.map(([o, h]) => ({ pts: o > 0 ? offsetLoop(ccw, o) : ccw.map((p) => p.clone()), h }))
  // Panel rings: the outline shrunk toward its centre.
  const panelT = [1, 0.9, 0.78, 0.64, 0.5, 0.36, 0.22, 0.1]
  const panel = panelT.map((t) => ({ pts: ccw.map((p) => centre.clone().lerp(p, t)), h: panelH + dome * (1 - t * t) }))
  const verts: { p: THREE.Vector2; h: number }[] = []
  const uv: number[] = []
  const index: number[] = []
  const strip = (a0: number, b0: number) => {
    for (let i = 0; i < M; i++) {
      const i2 = (i + 1) % M
      index.push(a0 + i, a0 + i2, b0 + i, b0 + i, a0 + i2, b0 + i2)
    }
  }
  const shade: number[] = []
  const addRing = (r: { pts: THREE.Vector2[]; h: number }, s = 1) => {
    const start = verts.length
    for (const p of r.pts) {
      verts.push({ p, h: r.h })
      uv.push(p.x * 6, p.y * 6)
      shade.push(s)
    }
    return start
  }
  const rimStarts = rings.map((r) => addRing(r))
  for (let r = 0; r + 1 < rimStarts.length; r++) strip(rimStarts[r], rimStarts[r + 1])
  const rimCount = index.length
  // The panel is shaded darker toward the rim, as if recessed in it.
  const panelStarts = panel.map((r, i) => addRing(r, 0.62 + 0.38 * Math.min(1, i / 3)))
  for (let r = 0; r + 1 < panelStarts.length; r++) strip(panelStarts[r], panelStarts[r + 1])
  const c = verts.length
  verts.push({ p: centre, h: panelH + dome })
  uv.push(centre.x * 6, centre.y * 6)
  shade.push(1)
  const last = panelStarts[panelStarts.length - 1]
  for (let i = 0; i < M; i++) index.push(last + i, last + ((i + 1) % M), c)
  // Placed mirrored, the triangles' winding flips; flip it back so they
  // still face outward.
  if (mirror) for (let t = 0; t < index.length; t += 3) [index[t + 1], index[t + 2]] = [index[t + 2], index[t + 1]]
  const decal = new Decal(
    (fit) =>
      verts.map((v) => {
        const q = place(fit, v.p)
        return { x: q.x, y: q.y, h: v.h }
      }),
    index,
    uv,
    [
      [0, rimCount, 0],
      [rimCount, index.length - rimCount, 1],
    ],
  )
  decal.geo.setAttribute('color', new THREE.Float32BufferAttribute(shade.flatMap((s) => [s, s, s]), 3))
  return decal
}

// ---- Materials and textures ---------------------------------------------------

function canvas(size: number) {
  const c = document.createElement('canvas')
  c.width = c.height = size
  return [c, c.getContext('2d')!] as const
}

/** A tangent-space normal map from a height canvas (Sobel). */
function normalMapFrom(height: HTMLCanvasElement, strength: number) {
  const w = height.width
  const h = height.height
  const src = height.getContext('2d')!.getImageData(0, 0, w, h).data
  const [c, ctx] = canvas(w)
  const out = ctx.createImageData(w, h)
  const H = (x: number, y: number) => src[(((y + h) % h) * w + ((x + w) % w)) * 4] / 255
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const dx = (H(x + 1, y - 1) + 2 * H(x + 1, y) + H(x + 1, y + 1) - H(x - 1, y - 1) - 2 * H(x - 1, y) - H(x - 1, y + 1)) * strength
      const dy = (H(x - 1, y + 1) + 2 * H(x, y + 1) + H(x + 1, y + 1) - H(x - 1, y - 1) - 2 * H(x, y - 1) - H(x + 1, y - 1)) * strength
      const len = Math.hypot(dx, dy, 1)
      const k = (y * w + x) * 4
      out.data[k] = ((-dx / len) * 0.5 + 0.5) * 255
      out.data[k + 1] = ((dy / len) * 0.5 + 0.5) * 255
      out.data[k + 2] = ((1 / len) * 0.5 + 0.5) * 255
      out.data[k + 3] = 255
    }
  ctx.putImageData(out, 0, 0)
  const tex = new THREE.CanvasTexture(c)
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping
  return tex
}

/** The web in UV space: spokes from between the eyes, sagging rings. */
function drawWeb(ctx: CanvasRenderingContext2D, size: number, color: string, width: number) {
  const cx = size * 0.5
  const cy = size * (1 - (-0.08 + 3.3) / 5.6)
  ctx.strokeStyle = color
  ctx.lineWidth = width
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  const spokes = 28
  const R = size * 0.62
  const ends: [number, number][] = []
  for (let i = 0; i < spokes; i++) {
    const a = (i / spokes) * Math.PI * 2
    ends.push([cx + Math.cos(a) * R * 1.1, cy + Math.sin(a) * R])
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.lineTo(cx + (ends[i][0] - cx) * 1.8, cy + (ends[i][1] - cy) * 1.8)
    ctx.stroke()
  }
  for (let k = 1; k <= 16; k++) {
    const r = Math.pow(k / 11, 1.12) * 0.98
    ctx.beginPath()
    for (let i = 0; i <= spokes; i++) {
      const [ax, ay] = ends[i % spokes]
      const [bx, by] = ends[(i + 1) % spokes]
      const pa = [cx + (ax - cx) * r, cy + (ay - cy) * r]
      const pb = [cx + (bx - cx) * r, cy + (by - cy) * r]
      // Each strand sags toward the centre between two spokes.
      const m = [cx + ((pa[0] + pb[0]) / 2 - cx) * 0.9, cy + ((pa[1] + pb[1]) / 2 - cy) * 0.9]
      if (i === 0) ctx.moveTo(pa[0], pa[1])
      ctx.quadraticCurveTo(m[0], m[1], pb[0], pb[1])
    }
    ctx.stroke()
  }
}

function spiderTextures() {
  const size = 1024
  // Colour: near-black stretch fabric, grey raised web.
  const [col, c] = canvas(size)
  c.fillStyle = '#121316'
  c.fillRect(0, 0, size, size)
  drawWeb(c, size, '#3a3c42', 9)
  drawWeb(c, size, '#6d7179', 5)
  const map = new THREE.CanvasTexture(col)
  map.colorSpace = THREE.SRGBColorSpace
  map.anisotropy = 8
  map.wrapS = THREE.RepeatWrapping
  // Roughness: matte fabric, satin rubber web.
  const [rc, r] = canvas(size)
  r.fillStyle = '#e6e6e6'
  r.fillRect(0, 0, size, size)
  drawWeb(r, size, '#707070', 8)
  const roughnessMap = new THREE.CanvasTexture(rc)
  roughnessMap.wrapS = THREE.RepeatWrapping
  // Height: raised, rounded web piping over a fine knit.
  const [hc, h] = canvas(size)
  const img = h.createImageData(size, size)
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const v = 60 + 22 * Math.sin(x * 2.2) * Math.sin(y * 2.2) + Math.random() * 14
      const k = (y * size + x) * 4
      img.data[k] = img.data[k + 1] = img.data[k + 2] = v
      img.data[k + 3] = 255
    }
  h.putImageData(img, 0, 0)
  h.filter = 'blur(2px)'
  drawWeb(h, size, 'rgba(255,255,255,0.55)', 11)
  drawWeb(h, size, '#ffffff', 6)
  h.filter = 'none'
  return { map, roughnessMap, normalMap: normalMapFrom(hc, 2.6) }
}

/** Spider's lens: fine white mesh, see-through-dark where it's thinnest. */
function meshTexture() {
  const [c, x] = canvas(128)
  x.fillStyle = '#d9dce1'
  x.fillRect(0, 0, 128, 128)
  x.fillStyle = 'rgba(52,56,64,0.75)'
  for (let j = 0; j < 16; j++)
    for (let i = 0; i < 16; i++) {
      x.beginPath()
      x.arc(i * 8 + (j % 2) * 4 + 2, j * 8 + 4, 2.3, 0, Math.PI * 2)
      x.fill()
    }
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  t.wrapS = t.wrapT = THREE.RepeatWrapping
  t.repeat.set(2.2, 2.2)
  t.anisotropy = 8
  return t
}

// ---- Spider -------------------------------------------------------------------

// The lens opening (right eye; mirrored for the left), relative to the eye
// centre: a big swept teardrop, its point low by the nose, its top edge
// rising outward, the outer end broad and round — after the reference mask.
const LENS = bezierLoop(
  [
    [-0.43, -0.2, -0.33, 0.04, -0.1, 0.26],
    [0.28, 0.36, 0.56, 0.42, 0.7, 0.12],
    [0.52, -0.2, 0.32, -0.42, -0.14, -0.38],
  ],
  24,
)

export function spiderMask(): Model {
  const geo = hoodGeometry()
  const { map, roughnessMap, normalMap } = spiderTextures()
  const fabric = new THREE.MeshPhysicalMaterial({
    map,
    roughnessMap,
    normalMap,
    normalScale: new THREE.Vector2(1.2, 1.2),
    roughness: 1,
    sheen: 0.6,
    sheenRoughness: 0.5,
    sheenColor: new THREE.Color(0x6a6e78),
    envMapIntensity: 0.8,
  })
  fabric.alphaToCoverage = true
  fabric.onBeforeCompile = (shader) => {
    shader.vertexShader = 'attribute vec3 aLocal;\nvarying vec3 vLocal;\n' + shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n  vLocal = aLocal;')
    shader.fragmentShader =
      'varying vec3 vLocal;\n' +
      shader.fragmentShader
        .replace('void main() {', `void main() {\n  float keepV = ${SPIDER_HEM};\n  float cutA = clamp(keepV / max(fwidth(keepV), 1e-5) + 0.5, 0.0, 1.0);\n  if (cutA <= 0.0) discard;`)
        .replace('#include <dithering_fragment>', '#include <dithering_fragment>\n  gl_FragColor.a *= cutA;')
  }
  fabric.customProgramCacheKey = () => 'spider-hood'
  const mesh = new THREE.Mesh(geo, fabric)
  mesh.frustumCulled = false
  // Pulled toward the camera in depth so the fabric never shows through.
  const onTop = { polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -8 }
  const rimMat = new THREE.MeshPhysicalMaterial({ color: 0x0b0b0d, roughness: 0.32, clearcoat: 0.7, clearcoatRoughness: 0.35, envMapIntensity: 0.9, ...onTop })
  const lensMat = new THREE.MeshPhysicalMaterial({ color: 0xffffff, map: meshTexture(), vertexColors: true, roughness: 0.42, metalness: 0.1, clearcoat: 0.4, clearcoatRoughness: 0.4, envMapIntensity: 1.2, ...onTop })
  // The rim: a thick, rounded rubber frame standing proud of the fabric;
  // the mesh lens sits a little below its top.
  const rim: [number, number][] = [
    [0.15, 0.006],
    [0.142, 0.034],
    [0.122, 0.056],
    [0.07, 0.066],
    [0.022, 0.056],
    [0, 0.04],
  ]
  const lenses = [-1, 1].map((side) =>
    rimmedPanel(LENS, rim, 0.04, 0.02, (fit, p) => {
      const e = eyeCentres(fit.L)[side < 0 ? 0 : 1]
      // Set wide, as on the reference: a clear band of fabric between
      // the rims over the bridge of the nose.
      return { x: e.x + side * (p.x * 0.98 + 0.15), y: e.y + 0.08 + p.y * 1.28 }
    }, side < 0),
  )
  const root = new THREE.Group()
  root.add(mesh)
  for (const l of lenses) {
    const m = new THREE.Mesh(l.geo, [rimMat, lensMat])
    m.frustumCulled = false
    root.add(m)
  }
  // Stretch fabric: close over the face (smoothed only enough to bridge
  // the eye sockets and soften the lips), snug over the head and neck.
  // A high, round crown: the hood wraps the face and rises well above it.
  const shape: MaskShape = { faceLift: () => 0.03, smooth: 3, lift: 0.04, ears: 0.2, skull: { y: 0.55, z: -1.35, rx: 1.3, ry: 1.62, front: 1.3, back: 1.66 } }
  return {
    root,
    fullHead: true,
    // Ears and the sides of the head showing past the hood go too.
    hidesHead: -1.3,
    update(rig) {
      const fit = fitMask(geo, rig, shape)
      for (const l of lenses) l.update(fit)
    },
  }
}

// ---- Bat ----------------------------------------------------------------------

// Per-pixel cut of the cowl's outline and eye holes (in face space: vLocal
// is the bare face for face vertices, the hood's own position beyond), with
// an antialiased edge; and fine sculpted creases as a bump on the shading
// normal, so they stay crisp whatever the mesh density.
const BAT_GLSL = `
varying vec3 vLocal;
uniform vec4 uEyeA;
uniform vec4 uEyeB;
uniform mat3 uRigView;
// Eye opening (x, y, rx, ry), after the reference: a level almond whose
// inner corner dips toward the nose.
float eyeHole(vec3 p, vec4 e) {
  float side = e.x < 0.0 ? -1.0 : 1.0;
  vec2 d = p.xy - e.xy;
  float u = d.x * side;
  float hw = e.z * 0.98;
  float hh = e.z * 0.6;
  float dip = 0.07 * max(0.0, -u) / hw;
  float almond = pow(abs(u) / hw, 1.7) + pow(abs(d.y + dip - 0.01) / hh, 2.0) - 1.0;
  return almond * hh;
}
float batKeep(vec3 p) {
  float ax = abs(p.x);
  // The mouth-and-chin opening: square-shouldered under the nose guard,
  // out to the inner edges of the cheek pieces, which run down to the jaw.
  float inner = 0.25 + 0.48 * (1.0 - smoothstep(-0.92, -0.72, p.y)) + 0.08 * (1.0 - smoothstep(-1.44, -0.92, p.y));
  float opening = min(-0.71 - p.y, inner - ax);
  // Lower edge round the back: from the cheek points back under the ears.
  float hem = p.y - (-1.44 + 0.2 * (1.0 - smoothstep(-1.9, -0.7, p.z)));
  float keep = min(-opening, hem);
  return min(keep, min(eyeHole(p, uEyeA), eyeHole(p, uEyeB)));
}
float segDist(vec2 q, vec2 a, vec2 b) {
  vec2 ab = b - a;
  float t = clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0);
  return length(q - a - ab * t);
}
// Signed side of q against the line a→b (positive to its right).
float side2(vec2 q, vec2 a, vec2 b) {
  vec2 ab = normalize(b - a);
  return (q.x - a.x) * ab.y - (q.y - a.y) * ab.x;
}
// The moulding of the reference mask (rig units, on the front of the
// face), as a height over the shell. Sharp, angular profiles — blades,
// V-grooves and stepped planes — so every line catches the light:
// • a blade ridge up the middle of the forehead;
// • two grooves sweeping up from the inner brows, bowing outward;
// • the brow: a raised plane over each eye, its edge angled down to the nose;
// • the nose guard: a raised, flat-fronted trapezoid;
// • the face plate's bevels: a vertical edge at each temple and a line
//   down each cheekbone, where the front plane turns to the side.
float batHeight(vec3 p) {
  float front = smoothstep(-0.75, -0.25, p.z);
  float ax = abs(p.x);
  vec2 q = vec2(ax, p.y);
  float h = 0.0;
  // Blade.
  h += 0.028 * max(0.0, 1.0 - ax / 0.065) * smoothstep(0.25, 0.45, p.y) * (1.0 - smoothstep(1.15, 1.45, p.y));
  // Bowed grooves: two segments, inner brow → mid-forehead → top.
  float g = min(segDist(q, vec2(0.17, 0.34), vec2(0.33, 0.78)), segDist(q, vec2(0.33, 0.78), vec2(0.4, 1.3)));
  h -= 0.02 * max(0.0, 1.0 - g / 0.05);
  // Brow plane: a step up above a line from the nose bridge (low) to the
  // outer temple (higher).
  float browLine = 0.14 + 0.16 * ax;
  h += 0.024 * smoothstep(-0.025, 0.025, p.y - browLine) * (1.0 - smoothstep(0.9, 1.05, ax)) * (1.0 - smoothstep(0.9, 1.3, p.y));
  // Nose guard: raised between edges that splay from the bridge to the tip.
  float t = clamp((0.15 - p.y) / 0.86, 0.0, 1.0);
  float nw = 0.1 + 0.15 * t;
  h += 0.022 * (1.0 - smoothstep(nw - 0.02, nw + 0.02, ax)) * smoothstep(-0.74, -0.68, p.y) * (1.0 - smoothstep(0.1, 0.25, p.y));
  // Temple bevel and cheekbone line: the front plane steps down to the sides.
  h -= 0.022 * smoothstep(0.78, 0.92, ax) * (1.0 - smoothstep(0.6, 1.1, p.y));
  h -= 0.02 * smoothstep(-0.03, 0.05, side2(q, vec2(0.6, -0.22), vec2(0.86, -1.3))) * (1.0 - smoothstep(-0.24, -0.14, p.y)) * (1.0 - smoothstep(0.78, 0.92, ax));
  return h * front;
}
// Tilts the shading normal by the creases' slope (their gradient, taken
// in rig space and turned into view space), smooth at any resolution.
vec3 bumpNormal(vec3 n, vec3 p) {
  const float e = 0.006;
  vec3 g = vec3(batHeight(p + vec3(e, 0.0, 0.0)) - batHeight(p - vec3(e, 0.0, 0.0)), batHeight(p + vec3(0.0, e, 0.0)) - batHeight(p - vec3(0.0, e, 0.0)), batHeight(p + vec3(0.0, 0.0, e)) - batHeight(p - vec3(0.0, 0.0, e))) / (2.0 * e);
  g = uRigView * g;
  return normalize(n - (g - dot(g, n) * n));
}
`

// The cowl is hard plastic, so unlike Spider's fabric it doesn't follow
// the face: it's a rigid moulding, built once from flat facets (as the
// reference's are) with slightly rounded edges, then sized to each face
// and carried by the head rig. The facets' offsets were set to clear a
// face with room to spare; the face itself hides the cowl's inside.

type Facet = { n: V3; d: number }
/** [normal x, y, z, offset] or [normal x, y, z, point x, y, z]. */
const facets = (raw: number[][]): Facet[] =>
  raw.map((r) => {
    const n = V(r[0], r[1], r[2]).normalize()
    return { n, d: r.length > 4 ? n.dot(V(r[3], r[4], r[5])) : r[3] }
  })
// For the right half (x ≥ 0), mirrored for the left; the centre line,
// where the two halves meet, becomes a ridge.
const BAT_HEAD = facets([
  [0.32, 0.28, 1, 0.22], // forehead, either side of the centre ridge
  [0.18, 0.75, 0.65, 0.66], // forehead turning to the top
  [0, 1, -0.12, 1.5], // top, falling away to the back
  [0.75, 0.75, 0.1, 1.34], // top corners
  [1, 0.4, 0.2, 1.18], // upper sides, tapering to the top
  [1, 0.08, 0.32, 0.96], // temples
  [1, 0.05, -0.4, 1.1, 0.2, -1.8], // sides, behind
  [0.35, 0.4, -1, 0, 0.4, -2.5], // back
  [0.4, -0.2, 1, 0.38], // cheeks
  [1, -0.35, 0.4, 1.0], // cheek pieces, narrowing toward the jaw
  [0, -1, -0.25, 0, -1.55, -1.2], // underneath
])
// The nose guard: a wedge from the brow down to the tip.
const BAT_NOSE = facets([
  [0.9, 0.15, 0.6, 0.27], // flanks
  [0, 0.3, 1, 0.31], // front
  [0, 1, 0.3, 0, 0.2, 0], // top, into the brow
  [0, -1, 0, 0, -0.76, 0], // bottom
  [0, 0, -1, 0, 0, -0.4], // back
])
const BAT_ORIGIN = V(0, -0.1, -1.0)

function smaxk(a: number, b: number, k: number) {
  const h = Math.max(k - Math.abs(a - b), 0) / k
  return Math.max(a, b) + h * h * k * 0.25
}
function facetSdf(f: Facet[], x: number, y: number, z: number) {
  let d = f[0].n.x * x + f[0].n.y * y + f[0].n.z * z - f[0].d
  for (let i = 1; i < f.length; i++) d = smaxk(d, f[i].n.x * x + f[i].n.y * y + f[i].n.z * z - f[i].d, 0.05)
  return d
}
function batSdf(x: number, y: number, z: number) {
  const ax = Math.abs(x)
  return smin(facetSdf(BAT_HEAD, ax, y, z), facetSdf(BAT_NOSE, ax, y, z), 0.05)
}

function traceBat(dir: V3, out: V3) {
  let t = 5
  for (let i = 0; i < 90; i++) {
    const d = batSdf(BAT_ORIGIN.x + dir.x * t, BAT_ORIGIN.y + dir.y * t, BAT_ORIGIN.z + dir.z * t)
    if (Math.abs(d) < 5e-4) break
    t -= d
  }
  return out.copy(dir).multiplyScalar(t).add(BAT_ORIGIN)
}

/** The moulding, meshed: a dense sphere's directions traced onto it, with
 *  normals from the distance field (flat on facets, rounded at edges). */
function batGeometry() {
  const g = new THREE.SphereGeometry(1, 200, 150)
  const pos = g.getAttribute('position') as THREE.BufferAttribute
  const nrm = g.getAttribute('normal') as THREE.BufferAttribute
  const d = new THREE.Vector3()
  const p = new THREE.Vector3()
  const e = 0.003
  for (let i = 0; i < pos.count; i++) {
    traceBat(d.fromBufferAttribute(pos, i).normalize(), p)
    pos.setXYZ(i, p.x, p.y, p.z)
    const n = V(batSdf(p.x + e, p.y, p.z) - batSdf(p.x - e, p.y, p.z), batSdf(p.x, p.y + e, p.z) - batSdf(p.x, p.y - e, p.z), batSdf(p.x, p.y, p.z + e) - batSdf(p.x, p.y, p.z - e)).normalize()
    nrm.setXYZ(i, n.x, n.y, n.z)
  }
  // The cut and the moulded lines are laid out on the moulding itself.
  g.setAttribute('aLocal', pos.clone())
  g.computeBoundingSphere()
  return g
}

export function batCowl(): Model {
  const geo = batGeometry()
  const uniforms = { uEyeA: { value: new THREE.Vector4() }, uEyeB: { value: new THREE.Vector4() }, uRigView: { value: new THREE.Matrix3() } }
  // Moulded, polished black: broad soft highlights, like the reference's
  // plastic, rather than a pin-point CG glint.
  const shell = () => new THREE.MeshPhysicalMaterial({ color: 0x0c0d10, roughness: 0.36, metalness: 0, clearcoat: 0.6, clearcoatRoughness: 0.25, envMapIntensity: 0.55 })
  const material = shell()
  material.side = THREE.DoubleSide
  material.alphaToCoverage = true
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms)
    shader.vertexShader = 'attribute vec3 aLocal;\nvarying vec3 vLocal;\n' + shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n  vLocal = aLocal;')
    shader.fragmentShader =
      BAT_GLSL +
      shader.fragmentShader
        .replace('void main() {', 'void main() {\n  float keepV = batKeep(vLocal);\n  float cutA = clamp(keepV / max(fwidth(keepV), 1e-5) + 0.5, 0.0, 1.0);\n  if (cutA <= 0.0) discard;')
        .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\n  normal = bumpNormal(normal, vLocal);')
        .replace('#include <dithering_fragment>', '#include <dithering_fragment>\n  gl_FragColor.a *= cutA;')
  }
  material.customProgramCacheKey = () => 'bat-cowl'
  const mesh = new THREE.Mesh(geo, material)
  mesh.frustumCulled = false

  // The ears stand up off the top corners.
  const earMat = shell()
  const cowl = new THREE.Group()
  cowl.add(mesh)
  const base = new THREE.Vector3()
  for (const side of [-1, 1]) {
    const ear = new THREE.Mesh(batEar(), earMat)
    ear.frustumCulled = false
    traceBat(V(side * 0.72, 1.6, 0.5).normalize(), base)
    ear.position.copy(base).add(V(0, -0.1, 0))
    ear.scale.set(side, 1, 1)
    cowl.add(ear)
  }
  const root = new THREE.Group()
  root.add(cowl)
  const scale = new THREE.Vector3(1, 1, 1)
  return {
    root,
    // Worn on the face: the face hides the inside of the cowl, which
    // otherwise shows through the eye holes and round the mouth.
    occludeFace: true,
    // Ears and the sides of the head, down to the cowl's lower edge.
    hidesHead: -1.25,
    update(rig) {
      const L = Array.from({ length: N }, (_, i) => rig.local(i))
      // Sized to this face: its width at the cheeks, its eye-to-chin length
      // (the moulding's own are those of the face it was made round).
      const eyeY = (eyeCentres(L)[0].y + eyeCentres(L)[1].y) / 2
      const sx = THREE.MathUtils.clamp(Math.abs(L[454].x - L[234].x) / 2 / 1.06, 0.85, 1.25)
      const sy = THREE.MathUtils.clamp((eyeY - L[152].y) / 1.71, 0.85, 1.25)
      scale.set(sx, sy, sx)
      cowl.scale.copy(scale)
      // The camera looks straight down -z, so view space turns like world
      // space: the rig's own rotation, unscaled.
      uniforms.uRigView.value.setFromMatrix4(rig.matrix).multiplyScalar(1 / rig.E)
      // Eye holes over this face's eyes, in the moulding's own space.
      const eyes = [LEFT_EYE, RIGHT_EYE].map((loop) => {
        const xs = loop.map((i) => L[i].x)
        return { c: loop.reduce((s, i) => s.add(L[i]), new THREE.Vector3()).divideScalar(loop.length), rx: (Math.max(...xs) - Math.min(...xs)) * 0.8 }
      })
      eyes.sort((a, b) => a.c.x - b.c.x)
      eyes.forEach((e, k) => (k ? uniforms.uEyeB : uniforms.uEyeA).value.set(e.c.x / sx, e.c.y / sy + 0.02, e.rx / sx, (e.rx * 0.56) / sy))
    },
  }
}

/** A bat ear: a tall, slightly flattened pyramid with rounded edges, its
 *  base running down into the head. Local frame: y up, z toward the face. */
function batEar() {
  // Cross-section (x, z) at the base: front-inner, front-outer, back.
  const sec: [number, number][] = [
    [-0.32, 0.08],
    [0.0, 0.11],
    [0.3, 0.1],
    [0.3, -0.06],
    [0.0, -0.14],
    [-0.3, -0.05],
  ]
  const levels = 14
  const H = 0.9
  const pos: number[] = []
  const idx: number[] = []
  const M = sec.length * 4
  for (let l = 0; l <= levels; l++) {
    const t = l / levels
    // From 0.3 inside the head up to the tip; the profile narrows faster
    // toward the top, with the outer edge staying near vertical.
    const y = -0.3 + t * (H + 0.3)
    const k = y < 0 ? 1 : 1 - y / H
    for (let i = 0; i < M; i++) {
      const f = i / 4
      const a = sec[Math.floor(f) % sec.length]
      const b = sec[(Math.floor(f) + 1) % sec.length]
      const s = f - Math.floor(f)
      const x = a[0] + (b[0] - a[0]) * s
      const z = a[1] + (b[1] - a[1]) * s
      // The tip sits over the outer half, so the outer edge stays near
      // upright and the inner one slopes.
      pos.push(x * k - 0.02 * Math.max(0, y), y, z * k - 0.03 * Math.max(0, y))
    }
  }
  for (let l = 0; l < levels; l++)
    for (let i = 0; i < M; i++) {
      const a = l * M + i
      const b = l * M + ((i + 1) % M)
      idx.push(a, b, a + M, b, b + M, a + M)
    }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setIndex(idx)
  g.computeVertexNormals()
  return g
}
