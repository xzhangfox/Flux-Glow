import * as THREE from 'three'
import type { Model, Rig } from './scene'

// Stylised animal heads after a toy-like Shiba reference model (front, 3/4,
// side, back, top and bottom views): smooth and matte, not furry — a round
// head with full cheeks, chunky conical spikes of "fur" round the cheeks,
// the back and the crown, big thick triangular ears with pale hollows, a
// short muzzle, dot brows, big cartoon eyes, a nose and a smiling mouth.
// The Shiba is the reference; the fox and the husky are derived from it,
// changing the ears, fur, eyes, muzzle, mouth, markings and colours (see
// DESIGNS).
//
// Built as one signed distance field (ears and spikes included, blended
// in), meshed with surface nets on a grid — which, unlike casting rays from
// a centre, gets the ears' hollows and the spikes right — coloured per
// vertex. Eyes and face markings are decals conformed to the surface; the
// nose and the mouth line are their own meshes. Sculpted in the reference
// sheet's units and matched to it view by view; scaled up to cover the
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

// ---- Designs ------------------------------------------------------------------

type Markings = 'shiba' | 'fox' | 'husky'

interface Design {
  id: string
  /** Muzzle: extra length forward, and width (1 = the Shiba's). */
  muzzle: { len: number; width: number }
  /** Ear root (x, y, z), lean out (u.x) and forward (u.z), how far it turns
   *  out (f.x), and its height, half-width and half-thickness. */
  ear: { x: number; y: number; z: number; lean: number; fwd: number; out: number; h: number; w: number; t: number }
  /** Fur clumps: length and girth scale, spacing round the head (degrees),
   *  and big pale tufts sweeping out of the cheeks (the fox's ruff). */
  fur: { len: number; girth: number; step: number; cheekTufts: boolean; earRootTufts: boolean }
  markings: Markings
  colors: { main: string; deep: string; crown: string; cream: string; earIn: string; earTip?: string; brow: string }
  /** Brows: centre, half-axes, tilt (outer end up). */
  brow: { x: number; y: number; rx: number; ry: number; tilt: number }
  /** The Shiba's brown marks beside the nose. */
  noseMarks: boolean
  /** Eyes: size (1 = the Shiba's), outer corner raised (radians), iris
   *  colours (rim, top, middle, bottom) and pupil. */
  eye: {
    size: number
    tilt: number
    rim: string
    top: string
    mid: string
    bottom: string
    pupil: string
    /** Pupil width and height (1 = the Shiba's round, open look). */
    pupilW: number
    pupilH: number
    catchLight: number
    /** An upper lid in the face's colour drawn down over the eye: how far
     *  down its edge sits at the eye's middle (0 = none, 0..1 of the eye's
     *  height) and how much it drops toward the nose. */
    lid: { cover: number; slope: number } | null
  }
  nose: { y: number; scale: number }
  mouth: 'closed' | 'open'
}

const DESIGNS: Record<string, Design> = {
  shiba: {
    id: 'shiba',
    muzzle: { len: 0, width: 1 },
    ear: { x: 0.56, y: 0.62, z: -0.4, lean: 0.36, fwd: 0.16, out: 0.75, h: 0.94, w: 0.39, t: 0.42 },
    fur: { len: 1, girth: 1, step: 23, cheekTufts: false, earRootTufts: false },
    markings: 'shiba',
    colors: { main: '#d6803c', deep: '#c46e2e', crown: '#e9a062', cream: '#f1e3cf', earIn: '#f6eee2', brow: '#f5e9d6' },
    brow: { x: 0.32, y: 0.42, rx: 0.135, ry: 0.082, tilt: 0.42 },
    noseMarks: true,
    eye: { size: 1, tilt: 0, rim: '#3f86d6', top: '#163c7c', mid: '#2a63b0', bottom: '#3478c8', pupil: '#0b1530', pupilW: 1, pupilH: 1, catchLight: 1, lid: null },
    nose: { y: -0.38, scale: 1 },
    mouth: 'closed',
  },
  // A red fox: taller ears with dark backs and tips, a longer, narrower
  // muzzle, longer fur with a white ruff sweeping out of the cheeks, round
  // dot brows, amber-orange eyes tipped up at the outer corners, mouth shut.
  fox: {
    id: 'fox',
    muzzle: { len: 0.13, width: 0.86 },
    ear: { x: 0.54, y: 0.62, z: -0.4, lean: 0.4, fwd: 0.16, out: 0.75, h: 1.12, w: 0.43, t: 0.42 },
    fur: { len: 1.22, girth: 0.95, step: 22, cheekTufts: true, earRootTufts: false },
    markings: 'fox',
    colors: { main: '#d8662a', deep: '#bb5220', crown: '#e7823f', cream: '#f5eee4', earIn: '#f7f1e8', earTip: '#2e211c', brow: '#f6efe4' },
    brow: { x: 0.3, y: 0.42, rx: 0.085, ry: 0.075, tilt: 0 },
    noseMarks: false,
    eye: { size: 0.97, tilt: 0.08, rim: '#f08a3a', top: '#7a2c10', mid: '#c9521c', bottom: '#e9792f', pupil: '#24100a', pupilW: 0.62, pupilH: 0.95, catchLight: 0.8, lid: { cover: 0.34, slope: -0.08 } },
    nose: { y: -0.4, scale: 0.9 },
    mouth: 'closed',
  },
  // A husky: grey cap and back over a white face, a dark stripe down the
  // forehead, white fluffy brows, shorter, broader ears set wide, shorter,
  // denser fur, bright pale-blue eyes tipped up, and an open smile.
  husky: {
    id: 'husky',
    muzzle: { len: 0.06, width: 1.02 },
    ear: { x: 0.62, y: 0.6, z: -0.42, lean: 0.46, fwd: 0.12, out: 0.75, h: 0.8, w: 0.44, t: 0.42 },
    fur: { len: 0.82, girth: 0.92, step: 19, cheekTufts: false, earRootTufts: true },
    markings: 'husky',
    colors: { main: '#62666d', deep: '#474a50', crown: '#6c7077', cream: '#f2f1ee', earIn: '#f4f2ef', brow: '#f7f6f3' },
    brow: { x: 0.31, y: 0.56, rx: 0.14, ry: 0.065, tilt: 0.38 },
    noseMarks: false,
    eye: { size: 0.98, tilt: 0.12, rim: '#7fdcff', top: '#1677c2', mid: '#2fa3ea', bottom: '#6ad3ff', pupil: '#0b2340', pupilW: 0.62, pupilH: 0.6, catchLight: 1.1, lid: { cover: 0.2, slope: 0.12 } },
    nose: { y: -0.4, scale: 1.02 },
    mouth: 'open',
  },
}

/** The design being built. */
let D: Design = DESIGNS.shiba

// ---- The sculpt ---------------------------------------------------------------
//
// Modelled in the reference sheet's own units: 1 = the distance between the
// eye centres, origin midway between them, +y up, +z out of the face. All
// proportions are measured off the six views; shibaHead() scales the result
// up to cover the wearer's head.

/** Head without ears and spikes. */
function headBody(x: number, y: number, z: number) {
  // Cranium: a round dome, a little shorter front to back than across.
  // Narrowing toward the top, so the crown domes up between the ears.
  const tx = x * (1 + 0.32 * ss(y, 0.1, 1.0))
  let d = ellipsoid(tx, y - 0.1, z + 0.3, 0.82, 0.92, 0.62)
  // A full, round back of the skull, bulging out behind the ears.
  d = smin(d, ellipsoid(x * (1 + 0.15 * ss(y, 0.1, 1.0)), y - 0.02, z + 0.56, 0.74, 0.8, 0.56), 0.3)
  // Under each ear, filling out the head's upper sides that the ears grow from.
  for (const s of [-1, 1]) d = smin(d, ellipsoid(x - s * 0.64, y - 0.4, z + 0.42, 0.3, 0.32, 0.34), 0.22)
  // Full jowls, the head's widest at the eyes' height...
  for (const s of [-1, 1]) d = smin(d, ellipsoid(x - s * 0.4, y + 0.1, z + 0.36, 0.48, 0.4, 0.46), 0.25)
  // ...tapering to a round jaw.
  d = smin(d, ellipsoid(x, y + 0.45, z + 0.12, 0.56, 0.34, 0.48), 0.25)
  // A short, round muzzle that the nose sits on, puffed out either side
  // where the mouth curls up, and the chin under it.
  const ml = D.muzzle.len
  const mw = D.muzzle.width
  const xm = x / mw
  d = smin(d, ellipsoid(xm, y + 0.4, z - 0.26 - ml, 0.3, 0.24, 0.3 + ml * 0.5), 0.18)
  for (const s of [-1, 1]) d = smin(d, ellipsoid(xm - s * 0.24, y + 0.5, z - 0.14 - ml * 0.6, 0.26, 0.21, 0.26), 0.14)
  // Round, puffed cheeks below the eyes.
  for (const s of [-1, 1]) d = smin(d, ellipsoid(x - s * 0.46, y + 0.38, z - 0.0, 0.36, 0.32, 0.3), 0.2)
  d = smin(d, ellipsoid(xm, y + 0.7, z - 0.06 - ml * 0.5, 0.26, 0.14, 0.24), 0.14)
  // The two round pads under the nose that the mouth curls round: made a
  // little too big, then cut off along the mouth line, so their lower edge
  // is exactly where the line is drawn — line and lip are one edge.
  let pads = Infinity
  for (const s of [-1, 1]) pads = smin(pads, ellipsoid(xm - s * 0.15, y + 0.6, z - 0.4 - ml, 0.23, 0.18, 0.16), 0.035)
  const ax = Math.abs(x)
  const lip = lipSegment(ax)
  // ...below along the line, and at the sides where it turns up at the
  // corners (the line's last point), the pads ending in that upturn.
  const [cx] = LIP[LIP.length - 1]
  // (the height gap over the line's slope ≈ the true distance to it, which
  // keeps the cut clean where the line turns steeply up)
  if (lip) pads = smax(pads, (lip.y - y) / Math.sqrt(1 + lip.slope * lip.slope), 0.012)
  // (the side softly rounded, and just past the line, so the upturn sits
  // on the pad's front rather than on a sharp wall)
  pads = smax(pads, ax - cx - 0.02, 0.045)
  d = smin(d, pads, 0.03)
  // ...with a shallow groove down between them from the nose.
  d = smax(d, -(Math.hypot(x / 0.5, (z - 0.58 - ml) / 0.5) * 0.5 - 0.018 + Math.max(0, y + 0.47) * 2 + Math.max(0, -0.68 - y) * 2), 0.02)
  // A short neck at the back.
  d = smin(d, ellipsoid(x, y + 0.34, z + 0.52, 0.45, 0.3, 0.38), 0.3)
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
  const E = D.ear
  const base = V(side * E.x, E.y, E.z)
  // Seen from the side it leans a little forward (see earLocal too).
  const u = V(side * E.lean, 1, E.fwd).normalize()
  // Facing forward and out, so the hollow shows from the front and side.
  const f0 = V(side * E.out, 0, 1).normalize()
  const r = new THREE.Vector3().crossVectors(u, f0).normalize()
  const f = new THREE.Vector3().crossVectors(r, u).normalize()
  return { base, u, r, f, h: E.h, w: E.w, t: E.t }
}
let EARS: Ear[] = []

/** Ear-local coordinates: a (up), s (across), t (out of the face). */
function earLocal(e: Ear, x: number, y: number, z: number) {
  const qx = x - e.base.x
  const qy = y - e.base.y
  const qz = z - e.base.z
  const a = qx * e.u.x + qy * e.u.y + qz * e.u.z
  const k = Math.min(1, Math.max(0, a / e.h))
  return {
    a,
    s: qx * e.r.x + qy * e.r.y + qz * e.r.z,
    // The ear curves a little forward: its section slides forward as it
    // rises, the back of it bowed.
    t: qx * e.f.x + qy * e.f.y + qz * e.f.z - 0.1 * e.h * k * k,
  }
}

/** Ear half-width and half-thickness at height a: a leaf, its sides bowed
 *  out, running to a softly rounded tip. */
function earSection(e: Ear, a: number) {
  const k = Math.min(1, Math.max(0, a / e.h))
  // Below its root the ear narrows away inside the head, so it never
  // breaks out of the side of the head as a ledge.
  const root = 1 - 0.9 * ss(-a, -0.08, 0.36)
  return { w: (e.w * (1 - k ** 1.3) ** 0.8 + 0.022) * root, th: (e.t * (1 - k) ** 0.5 + 0.055) * root }
}

/** A thick ear with a deep hollow in its front and a rolled rim round it. */
function earSdf(e: Ear, x: number, y: number, z: number) {
  const { a, s, t } = earLocal(e, x, y, z)
  const { w, th } = earSection(e, a)
  // The back of the ear is rounder than the front.
  const tt = t < 0 ? t * 1.35 : t * 1.25
  let d = Math.max((Math.hypot(s / w, tt / th) - 1) * Math.min(w, th), -a - 0.4, a - e.h)
  // The hollow: the same leaf, narrower, pressed deep into the front.
  // A rounded bowl: an ellipse in section, centred out in front of the
  // ear so its floor curves and a thick rim rolls round it.
  const kc = Math.min(1, Math.max(0, (a - 0.02) / (e.h * 0.86)))
  // (closing over at the bottom, so the hollow's floor curves into the head)
  // (closing up smoothly short of the tip, so it leaves no seam there)
  const wc = (e.w * 0.62 * (1 - kc) ** 0.6 + 0.005) * Math.sqrt(ss(a, -0.06, 0.22)) * (1 - ss(a, e.h * 0.66, e.h * 0.88)) + 1e-3
  const dc = th * 1.2 + 0.01
  const cav = Math.max((Math.hypot(s / wc, (t - th * 0.85) / dc) - 1) * Math.min(wc, dc), 0.02 - a)
  d = smax(d, -cav, 0.04)
  return d
}

/** Inside the ear's hollow (for colour). */
function earHollow(e: Ear, p: V3) {
  const { a, s, t } = earLocal(e, p.x, p.y, p.z)
  const kc = Math.min(1, Math.max(0, (a - 0.02) / (e.h * 0.86)))
  const wc = e.w * 0.5 * (1 - kc) ** 0.6 + 0.005
  return ss(wc - Math.abs(s), -0.012, 0.012) * ss(a, 0.0, 0.04) * ss(e.h * 0.84 - a, -0.01, 0.03) * ss(t, -earSection(e, a).th * 0.6, -earSection(e, a).th * 0.3)
}

interface Spike {
  a: V3
  b: V3
  ra: number
  /** For the bounding test. */
  c: V3
  reach: number
  cream: boolean
  /** Tip radius: small for a crisp point, larger for a soft, rounded lock. */
  rb: number
}
let spikes: Spike[] = []
const CENTRE = V(0, -0.05, -0.4)

/** The head's surface along a direction from its centre. */
function surfaceDir(dir: V3) {
  let t = 2
  for (let i = 0; i < 80; i++) {
    const p = CENTRE.clone().addScaledVector(dir, t)
    const d = headBody(p.x, p.y, p.z)
    if (Math.abs(d) < 1e-4) break
    t -= d
  }
  return CENTRE.clone().addScaledVector(dir, t)
}

function surfaceNormal(p: V3) {
  const e = 0.004
  return V(headBody(p.x + e, p.y, p.z) - headBody(p.x - e, p.y, p.z), headBody(p.x, p.y + e, p.z) - headBody(p.x, p.y - e, p.z), headBody(p.x, p.y, p.z + e) - headBody(p.x, p.y, p.z - e)).normalize()
}

function pushSpike(a: V3, out: V3, len: number, ra: number, cream: boolean, rb = 0.012) {
  // Rooted a little below the surface, so the cone's full base shows.
  const a0 = a.clone().addScaledVector(out, -0.05)
  const b = a.clone().addScaledVector(out, len)
  spikes.push({ a: a0, b, ra, c: a0.clone().lerp(b, 0.5), reach: (len + 0.05) / 2 + ra + 0.08, cream, rb })
}

/** A cone of fur: full at the base, its sides bowed out a little, to a
 *  softly rounded point. */
function spikeSdf(px: number, py: number, pz: number, s: Spike) {
  const abx = s.b.x - s.a.x
  const aby = s.b.y - s.a.y
  const abz = s.b.z - s.a.z
  const l2 = abx * abx + aby * aby + abz * abz
  const h = ((px - s.a.x) * abx + (py - s.a.y) * aby + (pz - s.a.z) * abz) / l2
  const t = Math.min(1, Math.max(0, h))
  const dx = px - (s.a.x + abx * t)
  const dy = py - (s.a.y + aby * t)
  const dz = pz - (s.a.z + abz * t)
  const r = s.rb + (s.ra - s.rb) * (1 - t) ** 0.8
  // (a cone's distance is under-estimated by its slope; good enough here)
  return (Math.sqrt(dx * dx + dy * dy + dz * dz) - r) * 0.85 + Math.max(0, h - 1) * Math.sqrt(l2) * 0.15
}

let srnd = 11
const sr = () => (srnd = (srnd * 16807) % 2147483647) / 2147483647

function buildSpikes() {
  spikes = []
  srnd = 11
  const F = D.fur
  // The ruff: crisp cones of fur in staggered columns, starting at the
  // edge of the face (just ahead of the ears) and running back round the
  // sides and the edge of the back. High up they point back and up, at
  // the middle straight back, low down back and down; each stands well off
  // the head. The face, the crown and the middle of the back stay smooth.
  const rows: [number, number, number][] = [
    // height (of the direction from the head's centre), first and last
    // angle round from the front (180 = the middle of the back)
    [0.6, 104, 140],
    [0.32, 96, 145],
    [0.04, 92, 150],
    [-0.24, 88, 155],
    [-0.5, 82, 165],
    [-0.7, 74, 180],
  ]
  for (const s of [-1, 1])
    rows.forEach(([y, from, to], row) => {
      const step = F.step
      for (let ang = from + (row % 2) * step * 0.5; ang <= to + 0.1; ang += step) {
        if (ang > 179 && s > 0) continue
        const phi = THREE.MathUtils.degToRad(ang + (sr() - 0.5) * 5)
        const yy = y + (sr() - 0.5) * 0.05
        const p = surfaceDir(V(s * Math.sin(phi), yy, Math.cos(phi)).normalize())
        const n = surfaceNormal(p)
        // The lie: back along the head, lifting up high and drooping low.
        const flowDir = V(0, 0.55 * y - 0.15, -1)
        flowDir.addScaledVector(n, -flowDir.dot(n)).normalize()
        const out = n.clone().multiplyScalar(1.35).add(flowDir).normalize()
        const big = 1 + 0.15 * ss(-y, -0.2, 0.5)
        const cream = creamAt(p) > 0.5 && ang < 120
        pushSpike(p, out, (0.24 + sr() * 0.04) * big * F.len, (0.13 + sr() * 0.015) * big * F.girth, cream)
      }
    })
  // The fox's ruff: big pale tufts sweeping out and down from the cheeks.
  if (F.cheekTufts)
    for (const s of [-1, 1])
      // Soft, rounded locks in two layers, the outer ones longer.
      for (const [y, z, len, ra, oy] of [
        [-0.08, -0.12, 0.24, 0.11, -0.1],
        [-0.26, -0.02, 0.32, 0.13, -0.3],
        [-0.44, -0.06, 0.34, 0.13, -0.5],
        [-0.6, -0.16, 0.3, 0.12, -0.7],
        [-0.18, -0.3, 0.26, 0.12, -0.2],
        [-0.38, -0.3, 0.3, 0.12, -0.45],
      ])
        pushSpike(surfaceDir(V(s * 0.95, y, z - CENTRE.z).normalize()), V(s, oy, -0.3).normalize(), len, ra, true, 0.045)
  // The nape: a point hanging down at the back, and cream fluff in a V
  // down the throat.
  pushSpike(surfaceDir(V(0, -0.8, -0.75).normalize()), V(0, -1, -0.45).normalize(), 0.32, 0.16, false)
  for (const [x, z, len] of [[0, -0.55, 0.22], [0.2, -0.42, 0.18], [-0.2, -0.42, 0.18], [0.1, -0.75, 0.2], [-0.1, -0.75, 0.2]] as const)
    pushSpike(surfaceDir(V(x, -0.9, z - CENTRE.z).normalize()), V(x * 0.4, -0.5, -1).normalize(), len, 0.12, true)
  // A small tuft on the crown, and a ridge of it running back down the head.
  for (const [x, y, z, ox, oy, oz, len, ra] of [
    [-0.07, 0.96, -0.25, -0.15, 1, 0.1, 0.13, 0.06],
    [0.07, 0.96, -0.25, 0.15, 1, 0.1, 0.11, 0.055],
    [0, 0.93, -0.55, 0, 0.6, -1, 0.2, 0.07],
    [0, 0.78, -0.9, 0, 0.4, -1, 0.18, 0.07],
  ])
    pushSpike(surfaceDir(V(x, y, z).sub(CENTRE).normalize()), V(ox, oy, oz).normalize(), len, ra, false)
  for (const s of [-1, 1]) {
    // Points up the sides of the head, in front of each ear's base.
    for (const [x, y, z, len, ra] of [
      [0.42, 0.84, -0.12, 0.22, 0.1],
      [0.66, 0.62, -0.18, 0.22, 0.1],
      [0.86, 0.36, -0.3, 0.22, 0.105],
    ])
      pushSpike(surfaceDir(V(s * x, y, z).sub(CENTRE).normalize()), V(s * 0.55, 0.75, 0.3).normalize(), len, ra, false)
    // A pale tuft out of each ear's hollow, at its outer base, poking out
    // past the rim.
    const e = EARS[s < 0 ? 0 : 1]
    const out = e.r.clone().multiplyScalar(Math.sign(e.r.x * s))
    const a = e.base.clone().addScaledVector(e.u, 0.08).addScaledVector(out, e.w * 0.55).addScaledVector(e.f, 0.12)
    pushSpike(a, out.clone().addScaledVector(e.u, 0.05).addScaledVector(e.f, 0.45).normalize(), 0.26, 0.1, true)
    // The husky's white fluff round the outside of each ear's root.
    if (F.earRootTufts)
      for (const [du, len, ra, up] of [
        [-0.05, 0.22, 0.1, -0.1],
        [0.08, 0.2, 0.09, 0.25],
        [0.2, 0.17, 0.08, 0.5],
      ]) {
        const at = e.base.clone().addScaledVector(e.u, du).addScaledVector(out, e.w * 0.85)
        pushSpike(at, out.clone().addScaledVector(e.u, up).addScaledVector(e.f, 0.2).normalize(), len, ra, true, 0.03)
      }
  }
}

function sdf(x: number, y: number, z: number) {
  let d = headBody(x, y, z)
  for (const e of EARS) d = smin(d, earSdf(e, x, y, z), 0.18)
  for (const s of spikes) {
    const dx = x - s.c.x
    const dy = y - s.c.y
    const dz = z - s.c.z
    if (dx * dx + dy * dy + dz * dz > s.reach * s.reach) continue
    d = smin(d, spikeSdf(x, y, z, s), 0.035)
  }
  return d
}

// ---- Colour ---------------------------------------------------------------------


function nearSpike(p: V3) {
  let best: Spike | null = null
  let bd = 0.02
  for (const s of spikes) {
    const d = spikeSdf(p.x, p.y, p.z, s)
    if (d < bd) {
      bd = d
      best = s
    }
  }
  return best
}

/** How cream a point of the head is (0 coat .. 1 cream). */
function creamAt(p: V3) {
  return D.markings === 'husky' ? huskyMask(p) : shibaMask(p)
}

/** The husky's white face: everything in front below a line just over the
 *  eyes, but for a dark stripe down the middle of the forehead narrowing
 *  between the eyes; round the sides and down the throat as on the Shiba. */
function huskyMask(p: V3) {
  const ax = Math.abs(p.x)
  const stripe = 0.07 + 0.2 * ss(p.y, -0.12, 0.3)
  // (arching up over each eye, dipping to the stripe and down the sides)
  // (well clear of the eyes' tops, which sit at about 0.3)
  const line = 0.44 - 0.45 * ss(ax, 0.72, 1.05) - 0.14 * ss(0.3, 0.1, ax)
  let c = ss(line - p.y, -0.02, 0.02) * (1 - ss(stripe - ax, -0.015, 0.015) * ss(p.y, -0.2, -0.12))
  c *= ss(p.z, -0.65 - 0.3 * ss(-p.y, 0.1, 0.6), -0.42 - 0.3 * ss(-p.y, 0.1, 0.6))
  return Math.max(c, shibaMask(p))
}

function shibaMask(p: V3) {
  const ax = Math.abs(p.x)
  // The face: cream from just under the eyes down, the orange of the nose
  // bridge running down in a soft point to the nose.
  const line = -0.21 - 0.1 * ss(0.24, 0.02, ax) - 0.16 * ss(ax, 0.6, 1.0)
  let c = ss(line - p.y, -0.015, 0.015)
  // ...round the sides only as far back as the ears, sweeping down, and
  // not under the jaw...
  c *= ss(p.z, -0.62 - 0.3 * ss(-p.y, 0.3, 0.7), -0.4 - 0.3 * ss(-p.y, 0.3, 0.7)) * ss(p.y, -0.8, -0.7)
  // ...where it runs back down the throat in a narrowing V.
  const hw = 0.08 + 0.55 * ss(p.z, -1.15, -0.1)
  c = Math.max(c, ss(-p.y, 0.6, 0.7) * ss(hw - ax, -0.02, 0.02))
  return c
}

function colorAt(p: V3): THREE.Color {
  const C = D.colors
  const CREAM = new THREE.Color(C.cream)
  const EAR_CREAM = new THREE.Color(C.earIn)
  const sp = nearSpike(p)
  if (sp?.cream) return CREAM
  const c = new THREE.Color(C.main).lerp(new THREE.Color(C.deep), ss(-p.z, 0.3, 1.2) * 0.6)
  // A paler, sunlit crown and forehead.
  c.lerp(new THREE.Color(C.crown), ss(p.y, 0.3, 0.9) * ss(p.z, -0.6, 0.1) * 0.4)
  // The fox's ears: dark on the back and toward the tip.
  if (C.earTip)
    for (const e of EARS) {
      const { a, t } = earLocal(e, p.x, p.y, p.z)
      if (a > 0) c.lerp(new THREE.Color(C.earTip), Math.max(ss(a, e.h * 0.5, e.h * 0.7), ss(-t, 0.0, 0.06) * ss(a, 0.1, 0.3)) * (earSdf(e, p.x, p.y, p.z) < 0.01 ? 1 : 0))
    }
  c.lerp(CREAM, creamAt(p))
  // The ears' hollows: cream, a little deeper toward their roots.
  for (const e of EARS) {
    const h = earHollow(e, p)
    if (h <= 0) continue
    const { a } = earLocal(e, p.x, p.y, p.z)
    const shade = 0.93 + 0.07 * ss(a, 0.02, 0.4)
    c.lerp(EAR_CREAM.clone().multiplyScalar(shade), h)
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
  const band = cell * C * 2.5
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
  const q = V(x, y, 2)
  for (let i = 0; i < 300; i++) {
    const d = sdf(q.x, q.y, q.z)
    if (d < 5e-4) break
    q.z -= Math.max(d * 0.8, 1e-3)
  }
  return q
}

function canvasTexture(W: number, H: number, draw: (g: CanvasRenderingContext2D) => void) {
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  draw(c.getContext('2d')!)
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  t.anisotropy = 8
  return t
}

/** The eye's outline in a unit box (x right, y down), for the wearer's
 *  right eye (on the viewer's left), traced off the sheet: an egg, taller
 *  than wide, its top highest toward the nose and its outer side fuller
 *  and lower. */
const EYE_PTS: [number, number][] = [
  [0.07, 0.21], [0.26, 0.08], [0.59, 0.01], [0.85, 0.06], [0.96, 0.23], [0.99, 0.47],
  [0.93, 0.75], [0.8, 0.94], [0.5, 1.0], [0.2, 0.9], [0.04, 0.7], [0.0, 0.45],
]
const EYE_MARGIN = 0.12

/** A smooth closed curve through EYE_PTS (midpoint quadratics), scaled by
 *  `grow` about the eye's middle and shifted by (dx, dy). */
function eyePath(g: CanvasRenderingContext2D, W: number, H: number, grow = 0, dx = 0, dy = 0) {
  const k = 1 - 2 * EYE_MARGIN
  const P = ([x, y]: [number, number]): [number, number] => [W * (EYE_MARGIN + (0.5 + (x - 0.5) * (1 + grow) + dx) * k), H * (EYE_MARGIN + (0.5 + (y - 0.5) * (1 + grow) + dy) * k)]
  const n = EYE_PTS.length
  const mid = (i: number): [number, number] => {
    const a = EYE_PTS[i % n]
    const b = EYE_PTS[(i + 1) % n]
    return P([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
  }
  g.beginPath()
  g.moveTo(...mid(0))
  for (let i = 1; i <= n; i++) g.quadraticCurveTo(...P(EYE_PTS[i % n]), ...mid(i))
  g.closePath()
}

/** A cartoon eye after the sheet: white on the outer side; a big blue iris
 *  and a big navy pupil set toward the nose, cut off by the eye's inner
 *  edge; a heavy lid line over the top and down the outer side, thickest
 *  at the upper outer corner, none on the inner side; a catch-light on the
 *  upper outer edge of the pupil. */
function eyeTexture(right: boolean) {
  const W = 256
  const H = 256
  const k = 1 - 2 * EYE_MARGIN
  const X = (x: number) => W * (EYE_MARGIN + x * k)
  const Y = (y: number) => H * (EYE_MARGIN + y * k)
  const E = D.eye
  return canvasTexture(W, H, (g) => {
    if (!right) {
      g.translate(W, 0)
      g.scale(-1, 1)
    }
    // Tip the outer corner (the left, here) up.
    g.translate(W / 2, H / 2)
    g.rotate(E.tilt)
    g.translate(-W / 2, -H / 2)
    // The lid line: the outline grown up and out, clipped to the top and
    // the outer side.
    g.save()
    g.beginPath()
    g.moveTo(X(1.2), Y(-0.2))
    g.lineTo(X(1.0), Y(0.22))
    g.lineTo(X(0.5), Y(0.5))
    g.lineTo(X(0.15), Y(1.2))
    g.lineTo(X(-0.3), Y(1.2))
    g.lineTo(X(-0.3), Y(-0.2))
    g.closePath()
    g.clip()
    g.fillStyle = '#121216'
    eyePath(g, W, H, 0.07, -0.035, -0.04)
    g.fill()
    g.restore()
    g.save()
    eyePath(g, W, H)
    g.clip()
    const sc = g.createLinearGradient(X(0), 0, X(0.6), 0)
    sc.addColorStop(0, '#e9e9ee')
    sc.addColorStop(1, '#fbfbf9')
    g.fillStyle = sc
    g.fillRect(0, 0, W, H)
    // Iris, a lighter rim round a deeper middle.
    g.fillStyle = E.rim
    g.beginPath()
    g.ellipse(X(0.74), Y(0.55), W * 0.33 * k, H * 0.46 * k, 0, 0, Math.PI * 2)
    g.fill()
    const ir = g.createLinearGradient(0, Y(0.1), 0, Y(1))
    ir.addColorStop(0, E.top)
    ir.addColorStop(0.6, E.mid)
    ir.addColorStop(1, E.bottom)
    g.fillStyle = ir
    g.beginPath()
    g.ellipse(X(0.75), Y(0.55), W * 0.29 * k, H * 0.42 * k, 0, 0, Math.PI * 2)
    g.fill()
    g.fillStyle = E.pupil
    g.beginPath()
    g.ellipse(X(0.8 - 0.05 * (1 - E.pupilW)), Y(0.53), W * 0.17 * k * E.pupilW, H * 0.32 * k * E.pupilH, 0, 0, Math.PI * 2)
    g.fill()
    // The lid's soft shadow across the top.
    const sh = g.createLinearGradient(0, Y(0), 0, Y(0.25))
    sh.addColorStop(0, 'rgba(12,20,40,0.4)')
    sh.addColorStop(1, 'rgba(12,20,40,0)')
    g.fillStyle = sh
    g.fillRect(0, 0, W, H)
    g.fillStyle = '#ffffff'
    g.beginPath()
    g.arc(X(0.6), Y(0.42), W * 0.07 * k * E.catchLight, 0, Math.PI * 2)
    g.fill()
    g.restore()
    if (E.lid) {
      // The upper lid, come down over the top of the eye: everything above
      // its edge is cut away so the face itself shows there, and its edge
      // is a dark line; x = 1 is the nose side.
      const edge = (x: number) => Y(E.lid!.cover + E.lid!.slope * (x - 0.5))
      g.save()
      g.globalCompositeOperation = 'destination-out'
      g.fillStyle = '#000'
      g.beginPath()
      g.moveTo(X(-0.3), edge(-0.3))
      g.lineTo(X(1.3), edge(1.3))
      g.lineTo(X(1.3), Y(-0.4))
      g.lineTo(X(-0.3), Y(-0.4))
      g.closePath()
      g.fill()
      g.restore()
      g.save()
      eyePath(g, W, H, 0.04)
      g.clip()
      g.strokeStyle = '#121216'
      g.lineWidth = W * 0.06 * k
      g.beginPath()
      g.moveTo(X(-0.1), edge(-0.1))
      g.lineTo(X(1.1), edge(1.1))
      g.stroke()
      // A soft shadow under the lid.
      const sh2 = g.createLinearGradient(0, edge(0.5), 0, edge(0.5) + H * 0.12 * k)
      sh2.addColorStop(0, 'rgba(12,20,40,0.35)')
      sh2.addColorStop(1, 'rgba(12,20,40,0)')
      g.fillStyle = sh2
      g.fillRect(0, edge(0.5) - H * 0.1, W, H * 0.3)
      g.restore()
    }
  })
}

/** Face paint over the front: the dot brows, the Shiba's little brown
 *  marks beside the nose, lash ticks, and the husky's open mouth. Covers
 *  x, y in [-0.8, 0.8]. */
const PAINT = 0.8
function paintTexture() {
  const W = 512
  const px = (x: number) => ((x + PAINT) / (2 * PAINT)) * W
  const py = (y: number) => ((PAINT - y) / (2 * PAINT)) * W
  const k = W / (2 * PAINT)
  return canvasTexture(W, W, (g) => {
    const B = D.brow
    if (D.mouth === 'open') {
      // The open smile: dark, deepest at the back, between the lips.
      const m = g.createLinearGradient(0, py(-0.64), 0, py(-0.76))
      m.addColorStop(0, '#1c0d0d')
      m.addColorStop(1, '#4a2024')
      g.fillStyle = m
      g.beginPath()
      const [upper, lower] = openMouth()
      const pts = [...upper.slice().reverse(), ...upper.map(([x, y]) => [-x, y] as [number, number]).slice(1), ...lower.slice().reverse()]
      pts.forEach(([x, y], i) => (i ? g.lineTo(px(x), py(y + D.nose.y + 0.38)) : g.moveTo(px(x), py(y + D.nose.y + 0.38))))
      g.closePath()
      g.fill()
    }
    for (const s of [-1, 1]) {
      g.fillStyle = D.colors.brow
      g.beginPath()
      // (tilted, the outer end raised)
      g.ellipse(px(s * B.x), py(B.y), B.rx * k, B.ry * k, -s * B.tilt, 0, Math.PI * 2)
      g.fill()
      g.lineCap = 'round'
      if (D.noseMarks) {
        // The brown marks beside the bridge of the nose, slanting out.
        g.strokeStyle = 'rgba(122,62,28,0.9)'
        g.lineWidth = 0.03 * k
        g.beginPath()
        g.moveTo(px(s * 0.215), py(-0.2))
        g.lineTo(px(s * 0.33), py(-0.295))
        g.stroke()
      }
      // A little lash tick over each eye.
      g.strokeStyle = 'rgba(70,36,16,0.85)'
      g.lineWidth = 0.012 * k
      g.beginPath()
      g.moveTo(px(s * 0.33), py(0.29))
      g.lineTo(px(s * 0.38), py(0.265))
      g.stroke()
    }
  })
}

/** The open mouth's lips: the upper lip's left half (from the middle out to
 *  its corner) and the lower lip (corner to corner). */
/** The upper lip's line from the middle out to the corner (x >= 0), as
 *  drawn: the closed "ω" (to where it starts to hook up), or the open
 *  smile's upper lip. In the head's coordinates (muzzle width and nose
 *  height applied). */
function upperLipLine(): [number, number][] {
  const half: [number, number][] =
    D.mouth === 'open'
      ? openMouth()[0].map(([x, y]) => [-x, y])
      : [[0, -0.64], [0.06, -0.69], [0.16, -0.715], [0.26, -0.69], [0.33, -0.61], [0.365, -0.52]]
  const dy = D.nose.y + 0.38
  return half.map(([x, y]) => [x * D.muzzle.width, y + dy])
}
let LIP: [number, number][] = []

/** The upper lip line's height at |x| = `ax`, or null beyond its corner. */
function upperLipY(ax: number): number | null {
  const s = lipSegment(ax)
  return s ? s.y : null
}

/** Height and slope of the upper lip line at |x| = `ax`. */
function lipSegment(ax: number): { y: number; slope: number } | null {
  for (let i = 1; i < LIP.length; i++) {
    const [x0, y0] = LIP[i - 1]
    const [x1, y1] = LIP[i]
    // (the last segment carries on a little past the corner, so the pads'
    // cut has no gap there)
    if (ax <= x1 || (i === LIP.length - 1 && ax <= x1 + 0.06)) {
      const slope = (y1 - y0) / Math.max(1e-6, x1 - x0)
      return { y: y0 + slope * (ax - x0), slope }
    }
  }
  return null
}

function openMouth(): [[number, number][], [number, number][]] {
  const upper: [number, number][] = [[0, -0.645], [-0.07, -0.672], [-0.14, -0.682], [-0.21, -0.665], [-0.27, -0.62], [-0.3, -0.57]]
  const lower: [number, number][] = [[-0.27, -0.62], [-0.22, -0.69], [-0.13, -0.735], [0, -0.75], [0.13, -0.735], [0.22, -0.69], [0.27, -0.62]]
  return [upper, lower]
}

/** The surface point marching in from `from` along `dir`, or null if the
 *  ray misses the head (it would otherwise run off to infinity). */
function march(from: V3, dir: V3): V3 | null {
  const q = from.clone()
  let travelled = 0
  for (let i = 0; i < 300; i++) {
    const d = sdf(q.x, q.y, q.z)
    if (d < 5e-4) return q
    const step = Math.max(d * 0.8, 1e-3)
    travelled += step
    if (travelled > 4) return null
    q.addScaledVector(dir, step)
  }
  return null
}

/** A decal over (cx, cy) conformed to the head, projected along the
 *  surface's normal there (or straight in, for `straight`), so features on
 *  the curve of the face sit on it rather than smear across it. */
function decal(cx: number, cy: number, w: number, h: number, mat: THREE.Material, lift: number, N = 18, straight = false) {
  const pts: number[] = []
  const uvs: number[] = []
  const idx: number[] = []
  const c = onFront(cx, cy)
  const e = 0.004
  const n = straight ? V(0, 0, 1) : V(sdf(c.x + e, c.y, c.z) - sdf(c.x - e, c.y, c.z), sdf(c.x, c.y + e, c.z) - sdf(c.x, c.y - e, c.z), sdf(c.x, c.y, c.z + e) - sdf(c.x, c.y, c.z - e)).normalize()
  // Keep it upright: tilt only sideways.
  n.y = 0
  n.normalize()
  const ux = new THREE.Vector3().crossVectors(V(0, 1, 0), n).normalize()
  const back = n.clone().negate()
  const miss = new Set<number>()
  for (let j = 0; j <= N; j++)
    for (let i = 0; i <= N; i++) {
      const u = i / N - 0.5
      const v = j / N - 0.5
      const q = march(c.clone().addScaledVector(n, 1).addScaledVector(ux, u * w).add(V(0, v * h, 0)), back)
      // (a miss: parked at the centre, and its cells dropped below)
      if (!q) miss.add(pts.length / 3)
      const at = q ?? c
      pts.push(at.x + n.x * lift, at.y, at.z + n.z * lift)
      uvs.push(u + 0.5, v + 0.5)
    }
  // Skip cells stretched across a fold (where the march slid off the side).
  const step = (Math.max(w, h) / N) * 2.5
  const far = (a: number, b: number) => miss.has(a) || miss.has(b) || Math.hypot(pts[a * 3] - pts[b * 3], pts[a * 3 + 1] - pts[b * 3 + 1], pts[a * 3 + 2] - pts[b * 3 + 2]) > step
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const a = j * (N + 1) + i
      if (far(a, a + 1) || far(a, a + N + 1) || far(a + 1, a + N + 2) || far(a + N + 1, a + N + 2)) continue
      idx.push(a, a + 1, a + N + 1, a + 1, a + N + 2, a + N + 1)
    }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
  g.setIndex(idx)
  g.computeVertexNormals()
  return new THREE.Mesh(g, mat)
}

// ---- The model ------------------------------------------------------------------

const built = new Map<string, THREE.BufferGeometry>()

/** Make `d` the design being built (the field, ears and fur follow it). */
function selectDesign(d: Design) {
  D = d
  LIP = upperLipLine()
  EARS = [makeEar(-1), makeEar(1)]
  buildSpikes()
}

function headGeometry() {
  const cached = built.get(D.id)
  if (cached) return cached
  const { pos, index } = surfaceNets(sdf, V(-1.45, -1.15, -1.55), V(1.45, 1.75, 0.85), 0.021)
  const geo = new THREE.BufferGeometry()
  const n = pos.length / 3
  const nrm = new Float32Array(n * 3)
  const col = new Float32Array(n * 3)
  const p = V(0, 0, 0)
  const e = 0.004
  for (let i = 0; i < n; i++) {
    p.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2])
    const g = V(sdf(p.x + e, p.y, p.z) - sdf(p.x - e, p.y, p.z), sdf(p.x, p.y + e, p.z) - sdf(p.x, p.y - e, p.z), sdf(p.x, p.y, p.z + e) - sdf(p.x, p.y, p.z - e)).normalize()
    nrm.set([g.x, g.y, g.z], i * 3)
    // Soft occlusion in the creases between spikes and in the ear hollows.
    const o = 0.05
    const occ = Math.max(0.68, Math.min(1, 1 - 4 * Math.max(0, o - sdf(p.x + g.x * o, p.y + g.y * o, p.z + g.z * o))))
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
  built.set(D.id, geo)
  return geo
}

/** Reference units to rig units, and where the sculpt sits on the wearer:
 *  big enough to swallow their head, its eyes a little below theirs. */
export const SHIBA_SCALE = 2.15
export const SHIBA_OFFSET = new THREE.Vector3(0, -0.3, 0.3)

export const shibaHead = () => animalHead(DESIGNS.shiba)
export const foxHead = () => animalHead(DESIGNS.fox)
export const huskyHead = () => animalHead(DESIGNS.husky)

function animalHead(design: Design): Model {
  selectDesign(design)
  const root = new THREE.Group()
  const sculpt = new THREE.Group()
  sculpt.scale.setScalar(SHIBA_SCALE)
  sculpt.position.copy(SHIBA_OFFSET)
  root.add(sculpt)
  // Matte, soft-touch: like a vinyl toy.
  const skin = new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.62, sheen: 0.35, sheenRoughness: 0.5, sheenColor: new THREE.Color(0xffe6cc), envMapIntensity: 0.6 })
  const head = new THREE.Mesh(headGeometry(), skin)
  head.castShadow = true
  head.receiveShadow = true
  sculpt.add(head)

  const paint = new THREE.MeshStandardMaterial({ map: paintTexture(), transparent: true, roughness: 0.65, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2 })
  sculpt.add(decal(0, 0, 2 * PAINT, 2 * PAINT, paint, 0.003, 40, true))

  // Eyes: big, a little inset under the brow.
  for (const s of [-1, 1]) {
    const eyeMat = new THREE.MeshStandardMaterial({ map: eyeTexture(s < 0), transparent: true, alphaTest: 0.4, roughness: 0.7 })
    sculpt.add(decal(s * 0.48, -0.02, 0.68 * D.eye.size, 0.76 * D.eye.size, eyeMat, 0.006))
  }

  // Nose: a matte rounded triangle, broad on top, on the tip of the muzzle.
  const tip = onFront(0, D.nose.y)
  const ng = new THREE.SphereGeometry(1, 40, 28)
  const np = ng.getAttribute('position') as THREE.BufferAttribute
  for (let i = 0; i < np.count; i++) {
    const y = np.getY(i)
    np.setX(i, np.getX(i) * (0.72 + 0.42 * y))
    np.setZ(i, np.getZ(i) * (1 - 0.15 * Math.max(0, -y)))
  }
  ng.computeVertexNormals()
  const nose = new THREE.Mesh(ng, new THREE.MeshStandardMaterial({ color: 0x1a1718, roughness: 0.85 }))
  nose.scale.set(0.23, 0.13, 0.12).multiplyScalar(D.nose.scale)
  nose.position.set(0, tip.y, tip.z - 0.02)
  nose.rotation.x = -0.25
  sculpt.add(nose)

  // The mouth: a line down from the nose into a wide, smiling "ω" whose
  // ends hook up into the cheeks.
  const line = new THREE.MeshStandardMaterial({ color: 0x24160f, roughness: 0.6 })
  const dy = D.nose.y + 0.38
  const mw = D.muzzle.width
  const strokes: [number, number][][] = (
    D.mouth === 'open'
      ? [
          // The upper lip over the open mouth, its corners curling up, and
          // the lower lip round under it.
          [[0, -0.5], [0, -0.645]],
          openMouth()[0],
          openMouth()[0].map(([x, y]): [number, number] => [-x, y]),
          openMouth()[1],
        ]
      : [
          [[0, -0.5], [0, -0.64]],
          [[0, -0.64], [-0.06, -0.69], [-0.16, -0.715], [-0.26, -0.69], [-0.33, -0.61], [-0.365, -0.52], [-0.35, -0.48]],
          [[0, -0.64], [0.06, -0.69], [0.16, -0.715], [0.26, -0.69], [0.33, -0.61], [0.365, -0.52], [0.35, -0.48]],
        ]
  ).map((st) => st.map(([x, y]): [number, number] => [x * mw, y + dy]))
  for (const st of strokes) {
    const pts = st.map(([x, y]) => {
      // Points on the upper lip sit on the pads' cut edge, a hair above it.
      const lip = upperLipY(Math.abs(x))
      const onLip = lip !== null && Math.abs(y - lip) < 0.004
      return onFront(x, onLip ? y + 0.006 : y).add(V(0, onLip ? -0.004 : 0, 0.004))
    })
    sculpt.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 48, 0.0085, 8, false), line))
    for (const end of [pts[0], pts[pts.length - 1]]) {
      const cap = new THREE.Mesh(new THREE.SphereGeometry(0.0085, 8, 6), line)
      cap.position.copy(end)
      sculpt.add(cap)
    }
  }
  return {
    root,
    fullHead: true,
    update(_rig: Rig) {},
  }
}
