import * as THREE from 'three'
import { plushCard, smooth, type PlushSpec } from './plush'
import type { Model, Rig } from './scene'

// Full fursuit heads that replace the wearer's head entirely, in the cute
// "kemono" fursuit style: a broad, fluffy head with wide cheeks narrowing to
// a small V chin, a short little muzzle with a button nose and a tiny mouth,
// big glossy anime eyes, huge long-furred ears, and long fur everywhere —
// fluff all round the silhouette and big white cheek fluff sticking out at
// the sides.
//
// The head is a smooth union of forms (cranium, cheeks, V-shaped lower
// face, muzzle, chin) as a signed distance field, meshed by casting rays
// out from inside it. Fur colour follows the 3D shape, occlusion comes from
// the field, and the eyes and mouth are decals conformed to the surface.
//
// Everything is in rig units (1 = eye distance; origin between the wearer's
// eyes; +y up, +z out of the face). The head is sized to cover the
// wearer's whole head and hair.

type V3 = THREE.Vector3
const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z)
const ss = THREE.MathUtils.smoothstep
const C = (h: string) => new THREE.Color(h)

interface SuitSpec {
  /** Fur colour at a point on the head (rig space). */
  fur: (p: V3) => THREE.Color
  ear: PlushSpec
  iris: [string, string, string]
  /** Fur and ear length (default 1). */
  furLength?: number
  earLength?: number
}

// ---- The sculpt (signed distance field) ---------------------------------------

function smin(a: number, b: number, k: number) {
  const h = Math.max(k - Math.abs(a - b), 0) / k
  return Math.min(a, b) - h * h * k * 0.25
}

const smax = (a: number, b: number, k: number) => -smin(-a, -b, k)

function ellipsoid(p: V3, cx: number, cy: number, cz: number, rx: number, ry: number, rz: number) {
  const x = p.x - cx
  const y = p.y - cy
  const z = p.z - cz
  const k0 = Math.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
  const k1 = Math.sqrt((x / (rx * rx)) ** 2 + (y / (ry * ry)) ** 2 + (z / (rz * rz)) ** 2)
  return k1 > 0 ? (k0 * (k0 - 1)) / k1 : -Math.min(rx, ry, rz)
}

/** A cone with rounded ends between a and b, squashed vertically by
 *  `flat` (< 1 = flatter on top and underneath). */
function cone(p: V3, a: V3, b: V3, ra: number, rb: number, flat: number) {
  // (Allocation-free: this runs millions of times while the head is built.)
  const abx = b.x - a.x
  const aby = b.y - a.y
  const abz = b.z - a.z
  const t = Math.min(1, Math.max(0, ((p.x - a.x) * abx + (p.y - a.y) * aby + (p.z - a.z) * abz) / (abx * abx + aby * aby + abz * abz)))
  const dx = p.x - (a.x + abx * t)
  const dy = (p.y - (a.y + aby * t)) / flat
  const dz = p.z - (a.z + abz * t)
  return Math.sqrt(dx * dx + dy * dy + dz * dz) - (ra + (rb - ra) * t)
}

// The snout runs from between the eyes, forward and gently down, to a
// broad nose; the lower jaw sits under it with a mouth line between them.
const MUZZLE_A = V(0, -0.56, 0.45)
const MUZZLE_B = V(0, -0.62, 1.62)
const BRIDGE_A = V(0, -0.3, 0.65)
const BRIDGE_B = V(0, -0.42, 1.6)
const JAW_A = V(0, -1.0, 0.4)
const JAW_B = V(0, -1.02, 1.45)
const EYE_X = 0.92
const EYE_Y = -0.06
const EYE_W = 1.24
const EYE_H = 1.06

/** A box with rounded edges: half-size (bx, by, bz) to the start of the
 *  rounding, radius r on top. */
function roundBox(p: V3, cx: number, cy: number, cz: number, bx: number, by: number, bz: number, r: number) {
  const qx = Math.abs(p.x - cx) - bx
  const qy = Math.abs(p.y - cy) - by
  const qz = Math.abs(p.z - cz) - bz
  const ox = Math.max(qx, 0)
  const oy = Math.max(qy, 0)
  const oz = Math.max(qz, 0)
  const out = Math.sqrt(ox * ox + oy * oy + oz * oz)
  return out + Math.min(Math.max(qx, qy, qz), 0) - r
}

/** The head without its snout and jaw: the face the eyes sit on. */
function sdfFace(p: V3) {
  // A head, not a ball: a rounded-box skull (flatter crown and sides,
  // rounded back), cut in front by a face plane that leans back a little
  // toward the forehead.
  let d = roundBox(p, 0, 0.62, -1.0, 0.68, 0.48, 0.52, 1.24)
  d = smax(d, p.z - (0.72 - 0.16 * p.y), 0.5)
  // Cheekbones: the head is widest at eye level.
  for (const s of [-1, 1]) d = smin(d, ellipsoid(p, s * 1.1, -0.35, -0.25, 0.82, 0.7, 0.78), 0.55)
  // The jaw tapers from the cheeks to a small chin: the V face.
  // A broad lower face trimmed by two side planes that run straight from
  // the cheekbones to a small chin.
  const lower = smax(ellipsoid(p, 0, -0.7, -0.45, 1.85, 1.25, 1.2), (Math.abs(p.x) - (0.42 + (p.y + 1.75) * 1.0)) * 0.7, 0.3)
  return smin(d, lower, 0.45)
}

/** Face, jaw and snout, before the mouth is carved in. */
function sdfShape(p: V3) {
  // The lower jaw grows out of the lower face…
  const face = smin(sdfFace(p), cone(p, JAW_A, JAW_B, 0.52, 0.32, 0.72), 0.45)
  // …and the snout out of the middle of the face, from a modest root below
  // the eyes (they stay clear): a bridge on top carries the line to the
  // nose, the muzzle under it ends round and blunt in a full pad.
  const muzzle = smin(cone(p, MUZZLE_A, MUZZLE_B, 0.6, 0.44, 0.88), ellipsoid(p, 0, -0.66, 1.56, 0.52, 0.4, 0.46), 0.25)
  const snout = smin(muzzle, cone(p, BRIDGE_A, BRIDGE_B, 0.3, 0.3, 1), 0.3)
  // Blended softly at the root, crisply toward the front, where the seam
  // between snout and jaw is the mouth line.
  const k = 0.4 - 0.32 * ss(p.z, 0.9, 1.55)
  return smin(face, snout, k)
}

// The mouth, modelled: a short groove down from the nose and a small "ω"
// smile, traced onto the snout and carved in.
interface Mouth {
  /** The mouth's strokes (philtrum, left and right half of the "ω"). */
  lines: V3[][]
  segs: [V3, V3][]
  nose: V3
  box: { x: number; y0: number; y1: number; z: number }
}
let mouth: Mouth | null = null

/** The surface point at (x, y), marching in from the front. */
function traceFront(f: (p: V3) => number, x: number, y: number) {
  const q = V(x, y, 4)
  for (let k = 0; k < 300; k++) {
    const dist = f(q)
    if (dist < 5e-4) break
    q.z -= Math.max(dist * 0.9, 1e-3)
  }
  return q
}

function buildMouth(): Mouth {
  // The nose sits on the most forward point of the snout…
  let best = traceFront(sdfShape, 0, -0.1)
  for (let y = -0.75; y <= -0.1; y += 0.01) {
    const q = traceFront(sdfShape, 0, y)
    if (q.z > best.z) best = q
  }
  const nose = best
  // …and the mouth just below it, on the front of the muzzle pad.
  const yM = nose.y - 0.3
  const at = (x: number, y: number) => traceFront(sdfShape, x, y).add(V(0, 0, -0.012))
  const top = at(0, nose.y - 0.17)
  const mid = at(0, yM)
  const half = (s: number) => [mid, at(s * 0.06, yM - 0.045), at(s * 0.13, yM - 0.065), at(s * 0.2, yM - 0.045), at(s * 0.26, yM + 0.005)]
  const lines = [[top, mid], half(-1), half(1)]
  const segs: [V3, V3][] = []
  for (const pts of lines) for (let i = 1; i < pts.length; i++) segs.push([pts[i - 1], pts[i]])
  return { lines, segs, nose, box: { x: 0.4, y0: yM - 0.2, y1: nose.y, z: nose.z - 0.9 } }
}

function segDist(p: V3, a: V3, b: V3) {
  const abx = b.x - a.x
  const aby = b.y - a.y
  const abz = b.z - a.z
  const t = Math.min(1, Math.max(0, ((p.x - a.x) * abx + (p.y - a.y) * aby + (p.z - a.z) * abz) / (abx * abx + aby * aby + abz * abz || 1)))
  const dx = p.x - a.x - abx * t
  const dy = p.y - a.y - aby * t
  const dz = p.z - a.z - abz * t
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

/** Distance to the mouth groove's centre line (Infinity far from it). */
function mouthDist(p: V3) {
  if (!mouth) return Infinity
  const b = mouth.box
  if (Math.abs(p.x) > b.x || p.y < b.y0 || p.y > b.y1 || p.z < b.z) return Infinity
  let d = Infinity
  for (const [a, c] of mouth.segs) d = Math.min(d, segDist(p, a, c))
  return d
}

function sdf(p: V3) {
  const d = sdfShape(p)
  const m = mouthDist(p)
  return m < 0.2 ? smax(d, 0.042 - m, 0.02) : d
}

function normalAt(p: V3) {
  const e = 0.01
  return V(
    sdf(V(p.x + e, p.y, p.z)) - sdf(V(p.x - e, p.y, p.z)),
    sdf(V(p.x, p.y + e, p.z)) - sdf(V(p.x, p.y - e, p.z)),
    sdf(V(p.x, p.y, p.z + e)) - sdf(V(p.x, p.y, p.z - e)),
  ).normalize()
}

/** Ambient occlusion from the field: how much the surface is hemmed in. */
function occlusion(p: V3, n: V3) {
  let occ = 0
  let w = 1
  for (let i = 1; i <= 4; i++) {
    const h = 0.09 * i
    occ += w * Math.max(0, h - sdf(p.clone().addScaledVector(n, h)))
    w *= 0.6
  }
  return THREE.MathUtils.clamp(1 - occ * 1.5, 0.62, 1)
}

// Rays start inside the head, behind the face, so the snout (the part
// furthest out) still gets plenty of the sphere's directions.
const ORIGIN = V(0, -0.3, -0.2)

/** Canvas pixel ↔ ray direction (front at the canvas centre, seam behind). */
function dirAt(u: number, v: number) {
  const phi = u * Math.PI * 2
  const th = v * Math.PI
  return V(-Math.sin(phi) * Math.sin(th), Math.cos(th), -Math.cos(phi) * Math.sin(th))
}
function uvOf(s: V3): [number, number] {
  const th = Math.acos(THREE.MathUtils.clamp(s.y, -1, 1))
  let phi = Math.atan2(-s.x, -s.z)
  if (phi < 0) phi += Math.PI * 2
  return [phi / (Math.PI * 2), th / Math.PI]
}

/** The outermost surface point along a ray from ORIGIN: sphere-traced in
 *  from outside, then bisected to precision. */
function surfaceAlong(d: V3) {
  const q = V(0, 0, 0)
  const at = (t: number) => sdf(q.copy(ORIGIN).addScaledVector(d, t))
  let hi = 5.5
  let lo = hi
  for (let i = 0; i < 80; i++) {
    const dist = at(lo)
    if (dist < 0) break
    if (dist < 2e-4) return q.clone()
    hi = lo
    lo -= Math.max(dist * 0.9, 2e-3)
    if (lo <= 0) {
      lo = 0
      break
    }
  }
  for (let i = 0; i < 12; i++) {
    const m = (lo + hi) / 2
    if (at(m) < 0) lo = m
    else hi = m
  }
  return ORIGIN.clone().addScaledVector(d, (lo + hi) / 2)
}

let seed = 11
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647

/** Neutral fur: grey strands combed back from the nose, multiplied over
 *  the head's own colours (vertex colours). */
function furTexture(W: number, H: number, strands: number) {
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const ctx = c.getContext('2d')!
  ctx.fillStyle = '#c4c4c4'
  ctx.fillRect(0, 0, W, H)
  ctx.lineCap = 'round'
  const src = MUZZLE_B.clone().sub(ORIGIN).normalize()
  const tmp = V(0, 0, 0)
  for (let i = 0; i < strands; i++) {
    const u = rnd()
    const v = Math.acos(1 - 2 * rnd()) / Math.PI
    const s = dirAt(u, v)
    const f = tmp.copy(s).sub(src).addScaledVector(s, -s.clone().sub(src).dot(s)).add(V(0, -0.12, 0))
    if (f.lengthSq() < 1e-4) f.set(0, -1, 0)
    f.normalize()
    const [u2, v2] = uvOf(s.clone().addScaledVector(f, 0.02).normalize())
    let dx = (u2 - u) * W
    if (dx > W / 2) dx -= W
    if (dx < -W / 2) dx += W
    const dy = (v2 - v) * H
    const l = Math.hypot(dx, dy) || 1
    const len = (5 + rnd() * 9) * (W / 1024)
    const g = Math.round(140 + rnd() * 115)
    ctx.strokeStyle = `rgb(${g},${g},${g})`
    ctx.globalAlpha = 0.25 + rnd() * 0.4
    ctx.lineWidth = (0.6 + rnd()) * (W / 1024)
    const x0 = u * W
    const y0 = v * H
    const bend = (rnd() - 0.5) * len * 0.4
    ctx.beginPath()
    ctx.moveTo(x0, y0)
    ctx.quadraticCurveTo(x0 + (dx / l) * len * 0.5 - (dy / l) * bend, y0 + (dy / l) * len * 0.5 + (dx / l) * bend, x0 + (dx / l) * len, y0 + (dy / l) * len)
    ctx.stroke()
  }
  ctx.globalAlpha = 0.6
  ctx.filter = 'blur(1px)'
  ctx.drawImage(c, 0, 0)
  ctx.filter = 'none'
  ctx.globalAlpha = 1
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.wrapS = THREE.RepeatWrapping
  tex.anisotropy = 8
  return tex
}
// The texture's average, in linear terms: vertex colours are divided by it
// so the head comes out at its intended colours.
const FUR_GREY = new THREE.Color('#c4c4c4').r

/** One lock of long-pile faux fur (neutral, tinted per card): fine
 *  strands from a broad root converging to a soft point, darker at the
 *  roots and lighter toward the tips, the root fading in. */
function lockTexture() {
  const W = 160
  const H = 320
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const ctx = c.getContext('2d')!
  ctx.lineCap = 'round'
  for (let i = 0; i < 650; i++) {
    const x0 = W / 2 + (rnd() - 0.5) * W * 0.86
    const y0 = H * (0.9 + rnd() * 0.1)
    const reach = 0.55 + rnd() * 0.45
    const x1 = W / 2 + (x0 - W / 2) * (0.35 + rnd() * 0.45) + (rnd() - 0.5) * W * 0.18
    const y1 = H - (H - 8) * reach * (1 - 0.8 * Math.abs(x0 - W / 2) / W)
    const cx = (x0 + x1) / 2 + (x0 - W / 2) * 0.35 + (rnd() - 0.5) * 10
    const cy = (y0 + y1) / 2
    const g = ctx.createLinearGradient(0, y0, 0, y1)
    const r = 215 + rnd() * 25
    g.addColorStop(0, `rgb(${r},${r},${r})`)
    g.addColorStop(1, 'rgb(255,255,255)')
    ctx.strokeStyle = g
    ctx.globalAlpha = 0.1 + rnd() * 0.3
    ctx.lineWidth = 0.7 + rnd() * 1.3
    ctx.beginPath()
    ctx.moveTo(x0, y0)
    ctx.quadraticCurveTo(cx, cy, x1, y1)
    ctx.stroke()
  }
  ctx.globalAlpha = 1
  ctx.globalCompositeOperation = 'destination-in'
  const fade = ctx.createLinearGradient(0, H, 0, H * 0.72)
  fade.addColorStop(0, 'rgba(0,0,0,0)')
  fade.addColorStop(1, 'rgba(0,0,0,1)')
  ctx.fillStyle = fade
  ctx.fillRect(0, 0, W, H)
  ctx.globalCompositeOperation = 'source-over'
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 4
  return tex
}

/** Lighting as if every fur card were the head's own surface: cards keep
 *  the head's normal on both sides (no dark backs where locks lift off). */
function furLit<T extends THREE.Material>(m: T): T {
  m.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <normal_fragment_begin>',
        THREE.ShaderChunk.normal_fragment_begin.replace('float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;', 'float faceDirection = 1.0;'),
      )
      // A soft glow where the coat turns away toward the outline, as
      // backlit plush does — the fluffy, toy-like edge of a fursuit.
      .replace('#include <opaque_fragment>', 'outgoingLight += diffuseColor.rgb * pow(1.0 - clamp(abs(normal.z), 0.0, 1.0), 2.5) * 0.35;\n#include <opaque_fragment>')
  }
  m.customProgramCacheKey = () => 'fur-lit'
  return m
}

/** Big glossy anime eye (the right eye; the left is mirrored): dark lash
 *  line, gradient iris, pupil, star sparkle and catch-lights. */
function eyeTexture(iris: [string, string, string]) {
  const W = 512
  const H = 440
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const ctx = c.getContext('2d')!
  const shape = () => {
    ctx.beginPath()
    ctx.moveTo(40, 250)
    ctx.bezierCurveTo(60, 70, 380, 25, 476, 150)
    ctx.bezierCurveTo(505, 280, 420, 420, 262, 418)
    ctx.bezierCurveTo(120, 416, 28, 360, 40, 250)
    ctx.closePath()
  }
  ctx.save()
  shape()
  ctx.fillStyle = '#fbf8f4'
  ctx.fill()
  ctx.clip()
  const g = ctx.createLinearGradient(0, 80, 0, 410)
  g.addColorStop(0, iris[0])
  g.addColorStop(0.5, iris[1])
  g.addColorStop(1, iris[2])
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.ellipse(258, 250, 170, 185, 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = 'rgba(20,8,4,0.55)'
  ctx.beginPath()
  ctx.ellipse(258, 285, 92, 100, 0, 0, Math.PI * 2)
  ctx.fill()
  // Lower-lid shadow across the top of the iris.
  const sh = ctx.createLinearGradient(0, 60, 0, 200)
  sh.addColorStop(0, 'rgba(10,5,5,0.55)')
  sh.addColorStop(1, 'rgba(10,5,5,0)')
  ctx.fillStyle = sh
  ctx.fillRect(0, 0, W, 220)
  const star = (x: number, y: number, r: number) => {
    ctx.beginPath()
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI) / 4 - Math.PI / 2
      const rr = i % 2 ? r * 0.28 : r
      ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr)
    }
    ctx.closePath()
    ctx.fill()
  }
  ctx.fillStyle = 'rgba(255,236,190,0.95)'
  star(205, 230, 50)
  ctx.fillStyle = 'rgba(255,255,255,0.95)'
  ctx.beginPath()
  ctx.ellipse(320, 160, 52, 40, -0.3, 0, Math.PI * 2)
  ctx.fill()
  ctx.beginPath()
  ctx.ellipse(190, 350, 20, 15, 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
  // Heavy upper lash line with a flick at the outer corner; fine lower line.
  ctx.strokeStyle = '#0d0a0a'
  ctx.fillStyle = '#0d0a0a'
  ctx.lineCap = 'round'
  ctx.lineWidth = 30
  ctx.beginPath()
  ctx.moveTo(40, 250)
  ctx.bezierCurveTo(60, 70, 380, 25, 476, 150)
  ctx.stroke()
  ctx.lineWidth = 9
  shape()
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(445, 112)
  ctx.lineTo(506, 118)
  ctx.lineTo(488, 175)
  ctx.closePath()
  ctx.fill()
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 8
  return tex
}

/** The head's coat: ~1,500 locks of long fur rooted all over its surface,
 *  each lying back at an angle along a grooming direction — combed out
 *  from the muzzle (up over the brow, out across the cheeks, down the
 *  chin), down the back of the head — short round the eyes and muzzle,
 *  long and swept sideways on the cheeks. All lit with the head's own
 *  normals, as one coat. */
function furCoat(head: THREE.Mesh, fur: (p: V3) => THREE.Color, nose: V3, lock: THREE.Texture, furLength: number) {
  const g = head.geometry
  const pos = g.getAttribute('position') as THREE.BufferAttribute
  const index = g.getIndex()!
  const cum: number[] = []
  const tris: number[] = []
  let total = 0
  const a = V(0, 0, 0)
  const b = V(0, 0, 0)
  const c = V(0, 0, 0)
  for (let t = 0; t < index.count; t += 3) {
    a.fromBufferAttribute(pos, index.getX(t))
    b.fromBufferAttribute(pos, index.getX(t + 1))
    c.fromBufferAttribute(pos, index.getX(t + 2))
    const area = b.clone().sub(a).cross(c.clone().sub(a)).length() / 2
    if (area < 1e-7) continue
    total += area
    cum.push(total)
    tris.push(t)
  }
  const P: number[] = []
  const Nn: number[] = []
  const Cc: number[] = []
  const UV: number[] = []
  const I: number[] = []
  const centers: number[] = []
  const back = V(0, -0.6, -1).normalize()
  const SEG = 3
  for (let k = 0; k < 3000; k++) {
    // Area-weighted random point on the head.
    const r = rnd() * total
    let lo = 0
    let hi = cum.length - 1
    while (lo < hi) {
      const m = (lo + hi) >> 1
      if (cum[m] < r) lo = m + 1
      else hi = m
    }
    const t = tris[lo]
    a.fromBufferAttribute(pos, index.getX(t))
    b.fromBufferAttribute(pos, index.getX(t + 1))
    c.fromBufferAttribute(pos, index.getX(t + 2))
    let u = rnd()
    let v = rnd()
    if (u + v > 1) {
      u = 1 - u
      v = 1 - v
    }
    const p = a.clone().addScaledVector(b.clone().sub(a), u).addScaledVector(c.clone().sub(a), v)
    const ax = Math.abs(p.x)
    // Keep the eyes, nose and mouth clear.
    // (Fur grows right up to the eyes' rims: a bare ring reads as a sticker.)
    if (p.z > 0 && ((ax - EYE_X) / (EYE_W * 0.47)) ** 2 + ((p.y - EYE_Y) / (EYE_H * 0.47)) ** 2 < 1 && p.z < 1) continue
    if (p.distanceTo(nose) < 0.36) continue
    if (mouthDist(p) < 0.1) continue
    const n = normalAt(p)
    // Grooming direction.
    const front = ss(p.z, -0.7, 0.2)
    const f = p.clone().sub(nose).multiplyScalar(front).addScaledVector(back, (1 - front) * 1.5)
    const cheek = ss(ax, 0.55, 1.1) * Math.exp(-(((p.y + 0.45) / 0.5) ** 2)) * ss(p.z, -0.7, 0.1)
    f.x += Math.sign(p.x) * cheek * 1.5
    f.addScaledVector(n, -f.dot(n))
    if (f.lengthSq() < 1e-5) f.set(0, -1, 0).addScaledVector(n, -n.y)
    f.normalize()
    const side = n.clone().cross(f).normalize()
    // Length and lift by region.
    const nearFace = p.z > 0.1 ? Math.exp(-((p.distanceTo(nose) / 0.9) ** 2)) : 0
    let len = THREE.MathUtils.lerp(0.55, 0.24, nearFace)
    len = THREE.MathUtils.lerp(len, 0.85, cheek)
    len *= (0.8 + rnd() * 0.4) * furLength
    const lift = THREE.MathUtils.lerp(0.28, 0.5, cheek) + (rnd() - 0.5) * 0.12
    // Narrow locks, many of them: broad cards overlap into a pattern of
    // scallops, like feathers.
    const w = 0.075 + len * 0.22
    const axis = f.clone().multiplyScalar(Math.cos(lift)).addScaledVector(n, Math.sin(lift))
    const col = fur(p).multiplyScalar(0.92 + rnd() * 0.14)
    const base = P.length / 3
    const root = p.clone().addScaledVector(n, -0.02)
    for (let sIdx = 0; sIdx <= SEG; sIdx++) {
      const sv = sIdx / SEG
      // Each lock droops back toward the coat along its length.
      const q = root.clone().addScaledVector(axis, len * sv).addScaledVector(n, -len * 0.18 * sv * sv)
      for (const e of [-0.5, 0.5]) {
        const qq = q.clone().addScaledVector(side, e * w)
        P.push(qq.x, qq.y, qq.z)
        Nn.push(n.x, n.y, n.z)
        Cc.push(col.r, col.g, col.b)
        UV.push(e + 0.5, sv)
      }
      if (sIdx < SEG) {
        const i0 = base + sIdx * 2
        I.push(i0, i0 + 2, i0 + 1, i0 + 1, i0 + 2, i0 + 3)
      }
    }
    const mid = root.clone().addScaledVector(axis, len * 0.5)
    centers.push(mid.x, mid.y, mid.z)
  }
  const cg = new THREE.BufferGeometry()
  cg.setAttribute('position', new THREE.Float32BufferAttribute(P, 3))
  cg.setAttribute('normal', new THREE.Float32BufferAttribute(Nn, 3))
  cg.setAttribute('color', new THREE.Float32BufferAttribute(Cc, 3))
  cg.setAttribute('uv', new THREE.Float32BufferAttribute(UV, 2))
  cg.setIndex(I)
  // Soft, blended locks, drawn back to front: re-sorted for each view.
  // A touch translucent, so overlapping locks blend instead of stacking
  // into hard-edged shards.
  const mat = furLit(new THREE.MeshStandardMaterial({ map: lock, vertexColors: true, transparent: true, opacity: 0.86, depthWrite: false, alphaTest: 0.01, side: THREE.DoubleSide, roughness: 1 }))
  const coat = new THREE.Mesh(cg, mat)
  coat.frustumCulled = false
  coat.renderOrder = 2
  const cards = centers.length / 3
  const order = Array.from({ length: cards }, (_, i) => i)
  const depth = new Float32Array(cards)
  const quad = I.slice(0, SEG * 6)
  const sorted = new Uint32Array(I.length)
  const sort = (toCam: V3) => {
    for (let i = 0; i < cards; i++) depth[i] = centers[i * 3] * toCam.x + centers[i * 3 + 1] * toCam.y + centers[i * 3 + 2] * toCam.z
    order.sort((x, y) => depth[x] - depth[y])
    let o = 0
    for (const card of order) {
      const base = card * (SEG + 1) * 2
      for (const q of quad) sorted[o++] = base + q
    }
    cg.setIndex(new THREE.BufferAttribute(sorted, 1))
  }
  return { coat, sort }
}

function fursuit(spec: SuitSpec): Model {
  seed = 11
  mouth ??= buildMouth()
  const root = new THREE.Group()

  // ---- Head mesh from the field ----
  const g = new THREE.SphereGeometry(1, 240, 180)
  g.rotateY(-Math.PI / 2)
  const pos = g.getAttribute('position') as THREE.BufferAttribute
  const uv = g.getAttribute('uv') as THREE.BufferAttribute
  const col = new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3)
  g.setAttribute('color', col)
  const nrm = g.getAttribute('normal') as THREE.BufferAttribute
  const d = V(0, 0, 0)
  for (let i = 0; i < pos.count; i++) {
    d.fromBufferAttribute(pos, i).normalize()
    const p = surfaceAlong(d)
    pos.setXYZ(i, p.x, p.y, p.z)
    const [u, w] = uvOf(d)
    // The seam sits at the back, out of sight; keep the UV's own u there.
    uv.setXY(i, Math.abs(u - uv.getX(i)) > 0.5 ? uv.getX(i) : u, 1 - w)
    // Normals from the sculpt itself, not the mesh's facets: smooth shading
    // and a smooth outline however the samples fall.
    const n = normalAt(p)
    nrm.setXYZ(i, n.x, n.y, n.z)
    const c = spec.fur(p).multiplyScalar(occlusion(p, n) / FUR_GREY)
    // The eyes sit in shallow hollows: a soft shadow round each.
    if (p.z > 0) {
      const e = Math.sqrt(((Math.abs(p.x) - EYE_X) / (EYE_W * 0.5)) ** 2 + ((p.y - EYE_Y) / (EYE_H * 0.5)) ** 2)
      c.multiplyScalar(1 - 0.28 * Math.exp(-(((e - 1.02) / 0.16) ** 2)))
    }
    // A soft shadow along the carved mouth groove.
    const md = mouthDist(p)
    if (md < 0.1) c.multiplyScalar(0.75 + 0.25 * (md / 0.1))
    col.setXYZ(i, c.r, c.g, c.b)
  }
  const furTex = furTexture(1024, 512, 120000)
  const furMat = new THREE.MeshPhysicalMaterial({ vertexColors: true, map: furTex, bumpMap: furTex, bumpScale: 0.2, roughness: 1, sheen: 0.3, sheenRoughness: 0.8, sheenColor: new THREE.Color(0x6a5a4a) })
  const head = new THREE.Mesh(g, furMat)
  head.castShadow = true
  head.receiveShadow = true
  root.add(head)

  // Features sit on the sculpted surface, found by casting rays at it.
  head.updateMatrixWorld()
  const ray = new THREE.Raycaster()
  const hitFront = (x: number, y: number) => {
    ray.set(V(x, y, 10), V(0, 0, -1))
    return ray.intersectObject(head)[0]?.point ?? V(x, y, 1)
  }

  // The face under the snout, found by marching in from the front.
  const onFace = (x: number, y: number) => {
    const q = V(x, y, 3)
    for (let k = 0; k < 200; k++) {
      const dist = sdfFace(q)
      if (dist < 1e-3) break
      q.z -= Math.max(dist, 2e-3)
    }
    return q
  }

  // A decal conformed to the head: a grid laid over (x, y) on the face,
  // each vertex dropped onto the surface just in front of it (`face`: the
  // face under the snout, so eyes sit beside the bridge, not on it).
  const decal = (cx: number, cy: number, w: number, h: number, rot: number, mirror: boolean, mat: THREE.Material, lift: number, face = false, dome = 0) => {
    const N = 20
    const pts: number[] = []
    const uvs: number[] = []
    const idx: number[] = []
    for (let j = 0; j <= N; j++)
      for (let i = 0; i <= N; i++) {
        const u = i / N - 0.5
        const v = j / N - 0.5
        const x = cx + Math.cos(rot) * u * w - Math.sin(rot) * v * h
        const y = cy + Math.sin(rot) * u * w + Math.cos(rot) * v * h
        const q = face ? onFace(x, y) : hitFront(x, y)
        // `dome`: bulged out in the middle, like a moulded acrylic eye.
        pts.push(q.x, q.y, q.z + lift + dome * Math.max(0, 1 - 4 * (u * u + v * v)))
        uvs.push(mirror ? 0.5 - u : u + 0.5, v + 0.5)
        if (i < N && j < N) {
          const a = j * (N + 1) + i
          idx.push(a, a + 1, a + N + 1, a + 1, a + N + 2, a + N + 1)
        }
      }
    const dg = new THREE.BufferGeometry()
    dg.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
    dg.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
    dg.setIndex(idx)
    dg.computeVertexNormals()
    const m = new THREE.Mesh(dg, mat)
    root.add(m)
    return m
  }

  // ---- Eyes: big glossy anime eyes, outer corners lifted ----
  const eyeMat = new THREE.MeshPhysicalMaterial({ map: eyeTexture(spec.iris), transparent: true, alphaTest: 0.3, roughness: 0.15, clearcoat: 1, clearcoatRoughness: 0.03, side: THREE.DoubleSide })
  // Domed, so the catch-lights and reflections move over them with the
  // head, as on a fursuit's acrylic eyes.
  for (const side of [-1, 1]) decal(side * EYE_X, EYE_Y, EYE_W, EYE_H, side * 0.1, side < 0, eyeMat, 0.05, true, 0.1)

  // ---- A big, glossy nose on the tip of the snout (the mouth is carved) ----
  const nose = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), new THREE.MeshPhysicalMaterial({ color: 0x141011, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.12 }))
  {
    const np = nose.geometry.getAttribute('position') as THREE.BufferAttribute
    for (let i = 0; i < np.count; i++) np.setX(i, np.getX(i) * (1 + 0.3 * np.getY(i)))
    nose.geometry.computeVertexNormals()
  }
  // A big, broad nose capping the snout.
  nose.scale.set(0.36, 0.23, 0.22)
  nose.position.copy(mouth.nose).add(V(0, 0.03, -0.08))
  nose.rotation.x = 0.3
  nose.castShadow = true
  root.add(nose)

  // The mouth's line itself: a thin dark cord laid in the carved groove, so
  // it's crisp and smooth whatever the mesh density.
  const lipMat = new THREE.MeshStandardMaterial({ color: 0x1c110d, roughness: 0.55 })
  for (const pts of mouth.lines) {
    const curve = new THREE.CatmullRomCurve3(pts.map((q) => q.clone().add(V(0, 0, 0.012))))
    const line = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, 0.02, 8, false), lipMat)
    root.add(line)
    for (const end of [pts[0], pts[pts.length - 1]]) {
      const cap = new THREE.Mesh(new THREE.SphereGeometry(0.02, 8, 6), lipMat)
      cap.position.copy(end).add(V(0, 0, 0.012))
      root.add(cap)
    }
  }

  // ---- Ears: huge, long-furred, set high on the head ----
  const earCard = plushCard(spec.ear)
  for (const side of [-1, 1]) {
    const ear = earCard.clone()
    ear.position.set(side * 1.12, 1.95, -0.85)
    ear.rotation.set(-0.06, side * -0.25, side * -0.3)
    ear.scale.set(side, spec.earLength ?? 1, 1)
    root.add(ear)
  }

  // ---- Fur coat: locks of long-pile fur rooted all over the head ----
  const lock = lockTexture()
  const coat = furCoat(head, spec.fur, nose.position, lock, spec.furLength ?? 1)
  root.add(coat.coat)

  // ---- Silhouette fluff and cheek locks, aimed at the camera ----
  const tuft = lock
  const tuftGeo = new THREE.PlaneGeometry(1, 1)
  tuftGeo.translate(0, 0.5, 0)
  const tuftMat = (color: THREE.Color) => furLit(new THREE.MeshStandardMaterial({ map: tuft, color, transparent: true, depthWrite: false, alphaTest: 0.01, side: THREE.DoubleSide, roughness: 1 }))
  interface Tuft { mesh: THREE.Mesh; a: number; inset: number; size: number; ruff: boolean; shade: number }
  const tufts: Tuft[] = []
  const addTuft = (a: number, inset: number, size: number, ruff: boolean, shade: number, layer: number) => {
    const mesh = new THREE.Mesh(tuftGeo, tuftMat(new THREE.Color()))
    mesh.renderOrder = 3 + layer
    root.add(mesh)
    tufts.push({ mesh, a, inset, size, ruff, shade })
  }
  // Fur in layers, not one fringe: inner layers sit over the head itself,
  // shorter and a little shaded (the fur under the fur), and the outer
  // ones reach past it — so the outline has depth and grows out of the
  // coat instead of ringing it.
  const LAYERS = [
    { inset: 0.78, size: 0.7, shade: 0.88 },
    { inset: 0.88, size: 0.85, shade: 0.94 },
    { inset: 0.97, size: 1, shade: 1 },
  ]
  LAYERS.forEach((L, layer) => {
    for (let i = 0; i < 60; i++) addTuft(((i + rnd()) / 60) * Math.PI * 2, L.inset, (0.45 + rnd() * 0.35) * L.size, false, L.shade, layer)
    // Big cheek fluff, sweeping out sideways from the cheeks (angle 0 =
    // the head's left in view, π = its right; a little below level)…
    for (let i = 0; i < 16; i++) {
      const t = (i + rnd()) / 16
      addTuft(-0.02 - t * 0.6, L.inset - 0.08, (0.9 + rnd() * 0.9) * L.size, true, L.shade, layer)
      addTuft(Math.PI + 0.02 + t * 0.6, L.inset - 0.08, (0.9 + rnd() * 0.9) * L.size, true, L.shade, layer)
    }
    // …and a ruff under the jaw, joining them.
    for (let i = 0; i < 24; i++) addTuft(-Math.PI / 2 + ((i + rnd()) / 24 - 0.5) * 2.2, L.inset - 0.06, (0.7 + rnd() * 0.6) * L.size, true, L.shade, layer)
  })
  const SIL_C = V(0, 0.3, -0.95)
  const SIL_R = V(2.05, 2.15, 1.95)
  const toCam = V(0, 0, 1)
  const e1 = V(0, 0, 0)
  const e2 = V(0, 0, 0)
  const m4 = new THREE.Matrix4()
  return {
    root,
    fullHead: true,
    update(rig: Rig) {
      // Silhouette of the head for this view (as an ellipsoid): s ⟂ R⁻¹·d.
      toCam.set(0, 0, 1).transformDirection(rig.inverse)
      coat.sort(toCam)
      const m = V(toCam.x / SIL_R.x, toCam.y / SIL_R.y, toCam.z / SIL_R.z).normalize()
      e1.set(0, 1, 0).addScaledVector(m, -m.y)
      if (e1.lengthSq() < 1e-4) e1.set(1, 0, 0)
      e1.normalize()
      e2.crossVectors(m, e1).normalize()
      for (const t of tufts) {
        const s = e1.clone().multiplyScalar(Math.sin(t.a)).addScaledVector(e2, Math.cos(t.a))
        const p = V(s.x * SIL_R.x, s.y * SIL_R.y, s.z * SIL_R.z)
        const outward = p.clone().addScaledVector(toCam, -p.dot(toCam)).normalize()
        if (t.ruff) outward.add(V(0, -0.15, 0)).normalize()
        // Rooted inside the rim and brought forward to the visible surface,
        // so they soften the edge while the snout can still pass in front.
        t.mesh.position.copy(SIL_C).addScaledVector(p, t.inset).addScaledVector(toCam, 0.8)
        const x = V(0, 0, 0).crossVectors(outward, toCam).normalize()
        const z = V(0, 0, 0).crossVectors(x, outward).normalize()
        m4.makeBasis(x, outward, z)
        t.mesh.quaternion.setFromRotationMatrix(m4)
        const len = t.size * (spec.furLength ?? 1)
        t.mesh.scale.set(len * 0.6, len, 1)
        ;(t.mesh.material as THREE.MeshStandardMaterial).color.copy(spec.fur(t.mesh.position)).multiplyScalar(1.1 * t.shade)
      }
    },
  }
}

// ---- Designs ----------------------------------------------------------------

const near = (p: V3, x: number, y: number, rx: number, ry: number) => Math.exp(-(((Math.abs(p.x) - x) / rx) ** 2) - ((p.y - y) / ry) ** 2)

/** How white the face is at p: cheeks from eye level down, everything
 *  below the nose, and under the chin — but not the stripe down the
 *  bridge of the nose, and not the back of the head. */
function whiteFace(p: V3) {
  const ax = Math.abs(p.x)
  const front = ss(p.z, -1.0, -0.3)
  // (Not on the snout: its top and sides stay coloured down to the mouth.)
  // Crisp edges: markings read as graphic shapes, as in anime.
  const cheeks = ss(0.12 - p.y, -0.03, 0.03) * ss(ax, 0.26, 0.34) * (1 - ss(p.z, 0.86, 1.0))
  const lower = ss(-0.6 - p.y, -0.03, 0.03)
  let w = Math.max(cheeks, lower) * front
  w = Math.max(w, ss(-p.y, 1.55, 1.8))
  // Pale brow spots above the inner corners of the eyes.
  w = Math.max(w, near(p, 0.42, 0.66, 0.09, 0.13) * ss(p.z, 0, 0.4))
  return THREE.MathUtils.clamp(w, 0, 1)
}

function foxFur(p: V3) {
  // Pastel peach, a touch deeper down the nose bridge; cream white.
  const bridge = Math.exp(-((p.x / 0.3) ** 2)) * ss(p.y, -0.6, -0.1) * (1 - ss(p.y, 0.4, 0.9)) * ss(p.z, 0, 0.5)
  const peach = C('#f2a465').lerp(C('#f8c08c'), ss(p.y, 0.2, 1.8) * 0.5).lerp(C('#ec9152'), bridge * 0.7)
  return peach.lerp(C('#fbf6ee'), whiteFace(p))
}

function huskyFur(p: V3) {
  const grey = C('#666b75').lerp(C('#40444c'), ss(p.y, 0.5, 2) * 0.6)
  // Huskies' white mask reaches up round the eyes too.
  const ax = Math.abs(p.x)
  const mask = ss(0.5 - p.y, -0.05, 0.05) * ss(ax, 0.2, 0.36) * ss(p.z, -0.6, 0)
  return grey.lerp(C('#f6f6f4'), Math.max(whiteFace(p), mask))
}

const EAR_OUTLINE = smooth([[22, 890], [40, 570], [118, 280], [250, 20], [392, 260], [478, 550], [494, 890], [262, 900]])
const EAR_INNER = smooth([[118, 850], [124, 590], [192, 330], [254, 150], [326, 330], [392, 590], [400, 850], [262, 862]])

export function foxHead(): Model {
  return fursuit({
    fur: foxFur,
    ear: {
      w: 512, h: 900, cardW: 2.3,
      outline: EAR_OUTLINE, inner: EAR_INNER, tip: [255, -40],
      root: '#e48a4c', mid: '#f3a66c', light: '#ffd2a8',
      skin: '#e88a50', plush: '#ee9a62', tuft: '#fff3e4',
      fan: 0.35,
      strand: [28, 70],
    },
    iris: ['#3b1d10', '#a35a26', '#f5c46a'],
  })
}

export function huskyHead(): Model {
  return fursuit({
    fur: huskyFur,
    ear: {
      w: 512, h: 900, cardW: 2.2,
      fan: 0.35,
      outline: EAR_OUTLINE, inner: EAR_INNER, tip: [255, -40],
      root: '#33373e', mid: '#60656f', light: '#a0a5ad',
      skin: '#dad6d4', plush: '#f4f2f0', tuft: '#ffffff',
      strand: [28, 70],
    },
    iris: ['#0b2440', '#2e78c0', '#a8e4ff'],
    // A husky's coat is short and dense, its ears short and upright.
    furLength: 0.68,
    earLength: 0.74,
  })
}
