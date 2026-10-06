import * as THREE from 'three'
import type { Model, Rig } from './scene'

// A stylised Shiba head, after a toy-like reference model (front, 3/4, side,
// back, top and bottom views): smooth and matte, not furry — a round head
// with full cheeks, chunky conical spikes of "fur" round the cheeks, the
// back and the crown, big thick triangular ears with cream hollows, a short
// cream muzzle, cream dot brows, big blue cartoon eyes, a black nose and a
// wide smiling mouth line.
//
// Built as one signed distance field (ears and spikes included, blended
// in), meshed with surface nets on a grid — which, unlike casting rays from
// a centre, gets the ears' hollows and the spikes right — coloured per
// vertex. Eyes are decals conformed to the surface; the nose and the mouth
// line are their own meshes. Rig units (1 = eye distance; origin between
// the wearer's eyes; +y up, +z out of the face); sized to cover the
// wearer's whole head.

type V3 = THREE.Vector3
const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z)
const ss = THREE.MathUtils.smoothstep

function smin(a: number, b: number, k: number) {
  const h = Math.max(k - Math.abs(a - b), 0) / k
  return Math.min(a, b) - h * h * k * 0.25
}
const smax = (a: number, b: number, k: number) => -smin(-a, -b, k)

function ellipsoid(x: number, y: number, z: number, rx: number, ry: number, rz: number) {
  const k0 = Math.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
  const k1 = Math.sqrt((x / (rx * rx)) ** 2 + (y / (ry * ry)) ** 2 + (z / (rz * rz)) ** 2)
  return k1 > 0 ? (k0 * (k0 - 1)) / k1 : -Math.min(rx, ry, rz)
}

/** Tapered capsule from a (radius ra) to b (radius rb). */
function cone(px: number, py: number, pz: number, a: V3, b: V3, ra: number, rb: number) {
  const abx = b.x - a.x
  const aby = b.y - a.y
  const abz = b.z - a.z
  const t = Math.min(1, Math.max(0, ((px - a.x) * abx + (py - a.y) * aby + (pz - a.z) * abz) / (abx * abx + aby * aby + abz * abz)))
  const dx = px - (a.x + abx * t)
  const dy = py - (a.y + aby * t)
  const dz = pz - (a.z + abz * t)
  return Math.sqrt(dx * dx + dy * dy + dz * dz) - (ra + (rb - ra) * t)
}

// ---- The sculpt ---------------------------------------------------------------

/** Head without ears and spikes. */
function headBody(x: number, y: number, z: number) {
  // Cranium: round, a little wider than tall.
  let d = ellipsoid(x, y - 0.3, z + 0.8, 1.85, 1.9, 1.85)
  // Full, round cheeks, widest just below the eyes, tapering to the jaw.
  for (const s of [-1, 1]) d = smin(d, ellipsoid(x - s * 0.95, y + 0.5, z + 0.05, 0.98, 0.9, 0.95), 0.55)
  // A short, round muzzle and the chin under it.
  d = smin(d, ellipsoid(x, y + 0.7, z - 1.0, 0.7, 0.5, 0.62), 0.4)
  d = smin(d, ellipsoid(x, y + 1.12, z - 0.68, 0.6, 0.34, 0.5), 0.35)
  // A gentle brow over each eye.
  for (const s of [-1, 1]) d = smin(d, ellipsoid(x - s * 0.62, y - 0.52, z - 0.72, 0.42, 0.18, 0.28), 0.25)
  return d
}

interface Ear {
  base: V3
  /** Up the ear, across it, and out of its face (unit vectors). */
  u: V3
  r: V3
  f: V3
  h: number
  w: number
  t: number
}

function makeEar(side: number): Ear {
  const base = V(side * 1.0, 1.45, -0.6)
  const u = V(side * 0.42, 1, -0.08).normalize()
  const f0 = V(side * 0.32, 0.05, 1).normalize()
  const r = new THREE.Vector3().crossVectors(u, f0).normalize()
  const f = new THREE.Vector3().crossVectors(r, u).normalize()
  return { base, u, r, f, h: 1.95, w: 0.82, t: 0.36 }
}
const EARS = [makeEar(-1), makeEar(1)]

/** Ear-local coordinates: a (up), s (across), t (out of the face). */
function earLocal(e: Ear, x: number, y: number, z: number) {
  const qx = x - e.base.x
  const qy = y - e.base.y
  const qz = z - e.base.z
  return {
    a: qx * e.u.x + qy * e.u.y + qz * e.u.z,
    s: qx * e.r.x + qy * e.r.y + qz * e.r.z,
    t: qx * e.f.x + qy * e.f.y + qz * e.f.z,
  }
}

/** A thick triangular ear, rounded, with a deep hollow in its front. */
function earSdf(e: Ear, x: number, y: number, z: number) {
  const { a, s, t } = earLocal(e, x, y, z)
  const k = Math.min(1, Math.max(0, a / e.h))
  // Pointed: width and thickness both run out at the tip; an ellipse in
  // section, so the edges are round, not boxy.
  const w = e.w * (1 - k) ** 0.8 + 0.02
  const th = e.t * (1 - k) ** 0.6 + 0.02
  let d = Math.max((Math.hypot(s / w, t / th) - 1) * Math.min(w, th), -a - 0.5, a - e.h - 0.02)
  // The hollow: the same triangle, smaller, pressed into the front.
  const kc = Math.min(1, Math.max(0, (a - 0.12) / (e.h * 0.85)))
  const wc = e.w * 0.66 * (1 - kc) ** 0.85
  const cav = Math.max(Math.abs(s) - wc, -(t - th * 0.1), 0.12 - a, a - e.h * 0.85) * 0.85
  d = smax(d, -cav, 0.08)
  return d
}

/** Inside the ear's hollow (for colour). */
function inEarHollow(e: Ear, p: V3) {
  const { a, s, t } = earLocal(e, p.x, p.y, p.z)
  const kc = Math.min(1, Math.max(0, (a - 0.08) / (e.h * 0.88)))
  return a > 0.05 && a < e.h * 0.92 && Math.abs(s) < e.w * 0.74 * (1 - kc) ** 0.9 + 0.03 && t > -0.02
}

interface Spike {
  a: V3
  b: V3
  ra: number
  /** For the bounding test. */
  c: V3
  reach: number
  cream: boolean
}
const spikes: Spike[] = []

/** The head's surface along a direction from its centre. */
function surfaceDir(dir: V3) {
  const c = V(0, -0.1, -0.7)
  let t = 4
  for (let i = 0; i < 80; i++) {
    const p = c.clone().addScaledVector(dir, t)
    const d = headBody(p.x, p.y, p.z)
    if (Math.abs(d) < 1e-3) break
    t -= d
  }
  return c.clone().addScaledVector(dir, t)
}

function addSpike(dir: V3, len: number, ra: number, tilt: V3, cream = false) {
  const at = surfaceDir(dir.clone().normalize())
  // Clumps lie back along the head rather than sticking straight out.
  const out = dir.clone().normalize().add(tilt).add(V(0, -0.25, -0.25)).normalize()
  const a = at.clone().addScaledVector(out, -0.12)
  const b = at.clone().addScaledVector(out, len)
  spikes.push({ a, b, ra, c: a.clone().lerp(b, 0.5), reach: len / 2 + ra + 0.25, cream })
}

let srnd = 7
const sr = () => (srnd = (srnd * 16807) % 2147483647) / 2147483647

function buildSpikes() {
  if (spikes.length) return
  // Cheeks: a row of chunky points sweeping out and down on each side.
  for (const s of [-1, 1]) {
    const rows: [number, number, number, number][] = [
      // y, z, length, radius
      [0.3, -0.45, 0.5, 0.3],
      [-0.15, -0.2, 0.62, 0.34],
      [-0.6, -0.05, 0.62, 0.34],
      [-1.0, 0.05, 0.5, 0.3],
    ]
    for (const [y, z, len, ra] of rows) addSpike(V(s, y * 0.9, z), len, ra, V(0, -0.45 - y * 0.15, -0.3), y < -0.5)
    // Up the sides of the head.
    addSpike(V(s * 0.95, 0.85, -0.7), 0.42, 0.26, V(0, 0.2, -0.35))
    // Cream tufts in the ears' hollows, at their inner base.
    const e = EARS[s < 0 ? 0 : 1]
    const root = e.base.clone().addScaledVector(e.u, 0.25).addScaledVector(e.r, -s * 0.12).addScaledVector(e.f, 0.05)
    for (const [du, dr] of [[0.0, 0], [0.18, 0.12]]) {
      const a = root.clone().addScaledVector(e.u, du).addScaledVector(e.r, dr * -s)
      const dir = e.u.clone().multiplyScalar(0.6).addScaledVector(e.f, 0.8).addScaledVector(e.r, -s * 0.3).normalize()
      const b = a.clone().addScaledVector(dir, 0.35)
      spikes.push({ a, b, ra: 0.1, c: a.clone().lerp(b, 0.5), reach: 0.5, cream: true })
    }
  }
  // Crown: a small tuft at the front of the head and a ridge down the back.
  addSpike(V(0, 1, 0.1), 0.34, 0.2, V(0, 0, 0.5))
  addSpike(V(0, 0.95, -0.5), 0.36, 0.22, V(0, 0.2, -0.4))
  addSpike(V(0, 0.75, -1.0), 0.36, 0.22, V(0, 0.1, -0.4))
  // The back of the head: soft points, scattered, longer toward the
  // bottom where the head meets the neck.
  srnd = 7
  for (let ring = 0; ring < 3; ring++) {
    const n = 5 + ring
    for (let i = 0; i < n; i++) {
      const a = Math.PI * (0.15 + (0.7 * (i + 0.3 + sr() * 0.4)) / n)
      const elev = 0.5 - ring * 0.6 + (sr() - 0.5) * 0.2
      const dir = V(Math.cos(a), elev, -Math.sin(a) * 1.1)
      addSpike(dir, 0.42 + ring * 0.1 + sr() * 0.1, 0.28 + ring * 0.03, V(0, -0.15 * ring, -0.15))
    }
  }
  // Under the chin and jaw: cream points hanging down.
  for (const [x, z, len] of [[0, 0.15, 0.5], [0.45, -0.05, 0.45], [-0.45, -0.05, 0.45], [0.85, -0.45, 0.4], [-0.85, -0.45, 0.4]] as const)
    addSpike(V(x, -1, z), len, 0.28, V(x * 0.3, -0.4, -0.2), true)
}

function sdf(x: number, y: number, z: number) {
  let d = headBody(x, y, z)
  for (const e of EARS) d = smin(d, earSdf(e, x, y, z), 0.28)
  for (const s of spikes) {
    const dx = x - s.c.x
    const dy = y - s.c.y
    const dz = z - s.c.z
    if (dx * dx + dy * dy + dz * dz > s.reach * s.reach) continue
    d = smin(d, cone(x, y, z, s.a, s.b, s.ra, 0.06), 0.22)
  }
  return d
}

// ---- Colour ---------------------------------------------------------------------

const ORANGE = new THREE.Color('#e2883d')
const ORANGE_DEEP = new THREE.Color('#cf7330')
const CREAM = new THREE.Color('#f5ead9')
const MARK = new THREE.Color('#8a4a24')

function nearSpike(p: V3) {
  let best: Spike | null = null
  let bd = 0.12
  for (const s of spikes) {
    const d = cone(p.x, p.y, p.z, s.a, s.b, s.ra, 0.06)
    if (d < bd) {
      bd = d
      best = s
    }
  }
  return best
}

function colorAt(p: V3): THREE.Color {
  const ax = Math.abs(p.x)
  for (const e of EARS) if (inEarHollow(e, p)) return CREAM.clone()
  const sp = nearSpike(p)
  if (sp?.cream) return CREAM.clone()
  // Cream: the muzzle, the cheeks below the eyes, the chin and throat.
  const front = ss(p.z, -0.75, -0.25)
  const boundary = -0.3 - 0.35 * Math.max(0, ax - 0.5)
  let cream = ss(boundary - p.y, -0.04, 0.04) * front
  cream = Math.max(cream, ss(-1.2 - p.y, -0.05, 0.05))
  // The cream dot brows.
  for (const s of [-1, 1]) {
    const q = ((p.x - s * 0.55) / 0.2) ** 2 + ((p.y - 0.78) / 0.12) ** 2
    if (q < 1 && p.z > 0) cream = 1
  }
  // A deeper orange over the crown and down the back, lighter at the sides.
  const c = ORANGE.clone().lerp(ORANGE_DEEP, ss(-p.z, 0.5, 2.2) * 0.7)
  c.lerp(CREAM, cream)
  // Small dark marks under the outer corners of the eyes.
  for (const s of [-1, 1]) {
    const q = ((p.x - s * 1.22) / 0.05) ** 2 + ((p.y + 0.12) / 0.12) ** 2
    if (q < 1 && p.z > -0.2) c.copy(MARK)
  }
  return c
}

// ---- Meshing: surface nets ---------------------------------------------------

function surfaceNets(f: (x: number, y: number, z: number) => number, min: V3, max: V3, cell: number) {
  const nx = Math.ceil((max.x - min.x) / cell) + 1
  const ny = Math.ceil((max.y - min.y) / cell) + 1
  const nz = Math.ceil((max.z - min.z) / cell) + 1
  // Coarse pass first; exact values only near the surface.
  const C = 4
  const cnx = Math.ceil(nx / C) + 1
  const cny = Math.ceil(ny / C) + 1
  const cnz = Math.ceil(nz / C) + 1
  const coarse = new Float32Array(cnx * cny * cnz)
  for (let k = 0; k < cnz; k++)
    for (let j = 0; j < cny; j++)
      for (let i = 0; i < cnx; i++) coarse[(k * cny + j) * cnx + i] = f(min.x + i * C * cell, min.y + j * C * cell, min.z + k * C * cell)
  const band = cell * C * 1.2
  const vals = new Float32Array(nx * ny * nz)
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const ci = Math.min(cnx - 2, Math.floor(i / C))
        const cj = Math.min(cny - 2, Math.floor(j / C))
        const ck = Math.min(cnz - 2, Math.floor(k / C))
        const tx = i / C - ci
        const ty = j / C - cj
        const tz = k / C - ck
        const g = (a: number, b: number, c: number) => coarse[((ck + c) * cny + (cj + b)) * cnx + (ci + a)]
        const lerp = (a: number, b: number, t: number) => a + (b - a) * t
        const est = lerp(lerp(lerp(g(0, 0, 0), g(1, 0, 0), tx), lerp(g(0, 1, 0), g(1, 1, 0), tx), ty), lerp(lerp(g(0, 0, 1), g(1, 0, 1), tx), lerp(g(0, 1, 1), g(1, 1, 1), tx), ty), tz)
        vals[(k * ny + j) * nx + i] = Math.abs(est) > band ? est : f(min.x + i * cell, min.y + j * cell, min.z + k * cell)
      }
  const idx = (i: number, j: number, k: number) => (k * ny + j) * nx + i
  const cellVert = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1)
  const cidx = (i: number, j: number, k: number) => (k * (ny - 1) + j) * (nx - 1) + i
  const pos: number[] = []
  const corners = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
  ]
  const edges = [
    [0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7],
  ]
  const cv = new Float32Array(8)
  for (let k = 0; k < nz - 1; k++)
    for (let j = 0; j < ny - 1; j++)
      for (let i = 0; i < nx - 1; i++) {
        let neg = 0
        for (let c = 0; c < 8; c++) {
          cv[c] = vals[idx(i + corners[c][0], j + corners[c][1], k + corners[c][2])]
          if (cv[c] < 0) neg++
        }
        if (neg === 0 || neg === 8) continue
        let sx = 0
        let sy = 0
        let sz = 0
        let n = 0
        for (const [a, b] of edges) {
          if (cv[a] < 0 === cv[b] < 0) continue
          const t = cv[a] / (cv[a] - cv[b])
          sx += corners[a][0] + (corners[b][0] - corners[a][0]) * t
          sy += corners[a][1] + (corners[b][1] - corners[a][1]) * t
          sz += corners[a][2] + (corners[b][2] - corners[a][2]) * t
          n++
        }
        cellVert[cidx(i, j, k)] = pos.length / 3
        pos.push(min.x + (i + sx / n) * cell, min.y + (j + sy / n) * cell, min.z + (k + sz / n) * cell)
      }
  const index: number[] = []
  const quad = (a: number, b: number, c: number, d: number, flip: boolean) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return
    if (flip) index.push(a, c, b, a, d, c)
    else index.push(a, b, c, a, c, d)
  }
  for (let k = 1; k < nz - 1; k++)
    for (let j = 1; j < ny - 1; j++)
      for (let i = 1; i < nx - 1; i++) {
        const v0 = vals[idx(i, j, k)] < 0
        // Edge along x to (i+1, j, k): the four cells round it.
        if (i < nx - 1 && v0 !== vals[idx(i + 1, j, k)] < 0)
          quad(cellVert[cidx(i, j - 1, k - 1)], cellVert[cidx(i, j, k - 1)], cellVert[cidx(i, j, k)], cellVert[cidx(i, j - 1, k)], v0)
        if (j < ny - 1 && v0 !== vals[idx(i, j + 1, k)] < 0)
          quad(cellVert[cidx(i - 1, j, k - 1)], cellVert[cidx(i - 1, j, k)], cellVert[cidx(i, j, k)], cellVert[cidx(i, j, k - 1)], v0)
        if (k < nz - 1 && v0 !== vals[idx(i, j, k + 1)] < 0)
          quad(cellVert[cidx(i - 1, j - 1, k)], cellVert[cidx(i, j - 1, k)], cellVert[cidx(i, j, k)], cellVert[cidx(i - 1, j, k)], v0)
      }
  return { pos, index }
}

// ---- Face features ------------------------------------------------------------

/** The surface point at (x, y), marching in from the front. */
function onFront(x: number, y: number) {
  const q = V(x, y, 4)
  for (let i = 0; i < 300; i++) {
    const d = sdf(q.x, q.y, q.z)
    if (d < 1e-3) break
    q.z -= Math.max(d * 0.8, 2e-3)
  }
  return q
}

function eyeTexture() {
  const W = 256
  const H = 300
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const g = c.getContext('2d')!
  const cx = W / 2
  const cy = H / 2
  // Dark outline, blue iris graded darker at the top, big black pupil.
  g.fillStyle = '#0d0f14'
  g.beginPath()
  g.ellipse(cx, cy, W * 0.48, H * 0.48, 0, 0, Math.PI * 2)
  g.fill()
  const ir = g.createLinearGradient(0, cy - H * 0.42, 0, cy + H * 0.42)
  ir.addColorStop(0, '#1d4f9a')
  ir.addColorStop(0.55, '#2f7fd2')
  ir.addColorStop(1, '#6fb6f0')
  g.fillStyle = ir
  g.beginPath()
  g.ellipse(cx, cy + 4, W * 0.41, H * 0.42, 0, 0, Math.PI * 2)
  g.fill()
  g.fillStyle = '#07080b'
  g.beginPath()
  g.ellipse(cx, cy + 10, W * 0.22, H * 0.25, 0, 0, Math.PI * 2)
  g.fill()
  // Catch-lights: a big one up and to one side, a small one opposite.
  g.fillStyle = '#ffffff'
  g.beginPath()
  g.ellipse(cx + W * 0.12, cy - H * 0.17, W * 0.13, H * 0.12, -0.4, 0, Math.PI * 2)
  g.fill()
  g.beginPath()
  g.ellipse(cx - W * 0.13, cy + H * 0.18, W * 0.05, H * 0.045, 0, 0, Math.PI * 2)
  g.fill()
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  t.anisotropy = 8
  return t
}

/** A decal over (x, y) conformed to the head's front. */
function decal(cx: number, cy: number, w: number, h: number, mirror: boolean, mat: THREE.Material, lift: number) {
  const N = 18
  const pts: number[] = []
  const uvs: number[] = []
  const idx: number[] = []
  for (let j = 0; j <= N; j++)
    for (let i = 0; i <= N; i++) {
      const u = i / N - 0.5
      const v = j / N - 0.5
      const q = onFront(cx + u * w, cy + v * h)
      pts.push(q.x, q.y, q.z + lift)
      uvs.push(mirror ? 0.5 - u : u + 0.5, v + 0.5)
      if (i < N && j < N) {
        const a = j * (N + 1) + i
        idx.push(a, a + 1, a + N + 1, a + 1, a + N + 2, a + N + 1)
      }
    }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
  g.setIndex(idx)
  g.computeVertexNormals()
  return new THREE.Mesh(g, mat)
}

// ---- The model ------------------------------------------------------------------

let built: { geo: THREE.BufferGeometry } | null = null

function headGeometry() {
  if (built) return built.geo
  buildSpikes()
  const { pos, index } = surfaceNets(sdf, V(-3.0, -2.6, -3.1), V(3.0, 3.4, 2.2), 0.045)
  const geo = new THREE.BufferGeometry()
  const n = pos.length / 3
  const nrm = new Float32Array(n * 3)
  const col = new Float32Array(n * 3)
  const p = V(0, 0, 0)
  const e = 0.01
  for (let i = 0; i < n; i++) {
    p.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2])
    const g = V(sdf(p.x + e, p.y, p.z) - sdf(p.x - e, p.y, p.z), sdf(p.x, p.y + e, p.z) - sdf(p.x, p.y - e, p.z), sdf(p.x, p.y, p.z + e) - sdf(p.x, p.y, p.z - e)).normalize()
    nrm.set([g.x, g.y, g.z], i * 3)
    // A little soft occlusion in creases (where the field is hemmed in).
    const occ = Math.max(0.72, Math.min(1, 1 - 3 * Math.max(0, 0.08 - sdf(p.x + g.x * 0.08, p.y + g.y * 0.08, p.z + g.z * 0.08))))
    const c = colorAt(p).multiplyScalar(occ)
    col.set([c.r, c.g, c.b], i * 3)
  }
  // Wind every triangle to face along the field's normal (outward).
  for (let t = 0; t < index.length; t += 3) {
    const [a, b2, c] = [index[t], index[t + 1], index[t + 2]]
    const ux = pos[b2 * 3] - pos[a * 3]
    const uy = pos[b2 * 3 + 1] - pos[a * 3 + 1]
    const uz = pos[b2 * 3 + 2] - pos[a * 3 + 2]
    const vx = pos[c * 3] - pos[a * 3]
    const vy = pos[c * 3 + 1] - pos[a * 3 + 1]
    const vz = pos[c * 3 + 2] - pos[a * 3 + 2]
    const fx = uy * vz - uz * vy
    const fy = uz * vx - ux * vz
    const fz = ux * vy - uy * vx
    if (fx * (nrm[a * 3] + nrm[b2 * 3] + nrm[c * 3]) + fy * (nrm[a * 3 + 1] + nrm[b2 * 3 + 1] + nrm[c * 3 + 1]) + fz * (nrm[a * 3 + 2] + nrm[b2 * 3 + 2] + nrm[c * 3 + 2]) < 0) {
      index[t + 1] = c
      index[t + 2] = b2
    }
  }
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3))
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
  geo.setIndex(index)
  geo.computeBoundingSphere()
  built = { geo }
  return geo
}

export function shibaHead(): Model {
  const root = new THREE.Group()
  // Matte, soft-touch: like a vinyl toy or a smooth plush.
  const skin = new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.72, sheen: 0.4, sheenRoughness: 0.6, sheenColor: new THREE.Color(0xffe2c4), envMapIntensity: 0.6 })
  const head = new THREE.Mesh(headGeometry(), skin)
  head.castShadow = true
  head.receiveShadow = true
  root.add(head)

  // Eyes: big, tall ovals, glossy.
  const eyeMat = new THREE.MeshPhysicalMaterial({ map: eyeTexture(), transparent: true, alphaTest: 0.4, roughness: 0.12, clearcoat: 1, clearcoatRoughness: 0.03 })
  for (const s of [-1, 1]) root.add(decal(s * 0.68, 0.05, 0.86, 1.0, s < 0, eyeMat, 0.012))

  // Nose: a glossy rounded triangle on the tip of the muzzle.
  const tip = onFront(0, -0.45)
  const ng = new THREE.SphereGeometry(1, 32, 24)
  const np = ng.getAttribute('position') as THREE.BufferAttribute
  for (let i = 0; i < np.count; i++) np.setX(i, np.getX(i) * (1 + 0.35 * np.getY(i)))
  ng.computeVertexNormals()
  const nose = new THREE.Mesh(ng, new THREE.MeshPhysicalMaterial({ color: 0x121012, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.1 }))
  nose.scale.set(0.27, 0.17, 0.15)
  nose.position.set(0, tip.y + 0.02, tip.z - 0.03)
  nose.rotation.x = 0.35
  root.add(nose)

  // The mouth: a short line down from the nose and a wide "w" smile whose
  // ends sweep up toward the cheeks.
  const line = new THREE.MeshStandardMaterial({ color: 0x2a1a14, roughness: 0.6 })
  const y0 = tip.y - 0.13
  const strokes: [number, number][][] = [
    [[0, y0], [0, y0 - 0.12]],
    [[0, y0 - 0.12], [-0.1, y0 - 0.19], [-0.22, y0 - 0.2], [-0.36, y0 - 0.14], [-0.48, y0 - 0.03]],
    [[0, y0 - 0.12], [0.1, y0 - 0.19], [0.22, y0 - 0.2], [0.36, y0 - 0.14], [0.48, y0 - 0.03]],
  ]
  for (const st of strokes) {
    const pts = st.map(([x, y]) => onFront(x, y).add(V(0, 0, 0.008)))
    const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 40, 0.016, 8, false), line)
    root.add(tube)
    for (const end of [pts[0], pts[pts.length - 1]]) {
      const cap = new THREE.Mesh(new THREE.SphereGeometry(0.016, 8, 6), line)
      cap.position.copy(end)
      root.add(cap)
    }
  }
  return {
    root,
    fullHead: true,
    update(_rig: Rig) {},
  }
}
