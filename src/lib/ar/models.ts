import * as THREE from 'three'
import { plushCard, smooth, type PlushSpec } from './plush'
import { FACE_TRIANGULATION } from '../faceTriangulation'
import type { Model, Rig } from './scene'

// Every 3D effect, modeled procedurally in the head rig's units (1 = the
// distance between the eye centers; origin between the eyes; +y up the
// face, +z out of it). Anchors are re-read from the live landmarks every
// frame — ears from the forehead-top and temple points, glasses from the
// eyes and nose bridge, noses from the nose tip — so each prop sits on
// this particular head rather than a generic one.

const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z)

interface Anchors {
  top: THREE.Vector3
  templeL: THREE.Vector3
  templeR: THREE.Vector3
  eyeL: THREE.Vector3
  eyeR: THREE.Vector3
  bridge: THREE.Vector3
  noseTip: THREE.Vector3
  lipUpperInner: THREE.Vector3
  lipLowerInner: THREE.Vector3
  wingL: THREE.Vector3
  wingR: THREE.Vector3
}

const EYE_A = [33, 133, 159, 145]
const EYE_B = [263, 362, 386, 374]
const avg = (rig: Rig, idx: number[]) => idx.reduce((s, i) => s.add(rig.local(i)), new THREE.Vector3()).divideScalar(idx.length)

function anchors(rig: Rig): Anchors {
  const sortX = (a: THREE.Vector3, b: THREE.Vector3) => (a.x <= b.x ? [a, b] : [b, a])
  const [templeL, templeR] = sortX(rig.local(21), rig.local(251))
  const [eyeL, eyeR] = sortX(avg(rig, EYE_A), avg(rig, EYE_B))
  const [wingL, wingR] = sortX(rig.local(98), rig.local(327))
  return {
    top: rig.local(10),
    templeL,
    templeR,
    eyeL,
    eyeR,
    bridge: rig.local(6),
    noseTip: rig.local(4),
    lipUpperInner: rig.local(13),
    lipLowerInner: rig.local(14),
    wingL,
    wingR,
  }
}

// ---- Materials -------------------------------------------------------------

const physical = (o: THREE.MeshPhysicalMaterialParameters) => new THREE.MeshPhysicalMaterial(o)
const gold = () => new THREE.MeshStandardMaterial({ color: 0xe0b04a, metalness: 1, roughness: 0.22 })
const shadowed = <T extends THREE.Object3D>(o: T) => {
  o.traverse((c) => {
    if ((c as THREE.Mesh).isMesh) (c as THREE.Mesh).castShadow = true
  })
  return o
}

// ---- Geometry helpers --------------------------------------------------------

/** A tube whose radius tapers from r0 to r1 along the curve (horns, tongue
 *  tips, whiskers). */
function taperTube(points: THREE.Vector3[], r0: number, r1: number, radial = 20, segs = 60) {
  const curve = new THREE.CatmullRomCurve3(points)
  const frames = curve.computeFrenetFrames(segs, false)
  const pos: number[] = []
  const idx: number[] = []
  for (let i = 0; i <= segs; i++) {
    const t = i / segs
    const p = curve.getPointAt(t)
    const r = THREE.MathUtils.lerp(r0, r1, Math.pow(t, 0.85))
    for (let j = 0; j <= radial; j++) {
      const a = (j / radial) * Math.PI * 2
      const n = frames.normals[i].clone().multiplyScalar(Math.cos(a)).add(frames.binormals[i].clone().multiplyScalar(Math.sin(a)))
      pos.push(p.x + n.x * r, p.y + n.y * r, p.z + n.z * r)
    }
  }
  for (let i = 0; i < segs; i++)
    for (let j = 0; j < radial; j++) {
      const a = i * (radial + 1) + j
      const b = a + radial + 1
      idx.push(a, b, a + 1, a + 1, b, b + 1)
    }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setIndex(idx)
  g.computeVertexNormals()
  return g
}

function blob(sx: number, sy: number, sz: number, mat: THREE.Material, shape?: (v: THREE.Vector3) => void) {
  const g = new THREE.SphereGeometry(1, 40, 28)
  if (shape) {
    const p = g.getAttribute('position') as THREE.BufferAttribute
    const v = new THREE.Vector3()
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i)
      shape(v)
      p.setXYZ(i, v.x, v.y, v.z)
    }
    g.computeVertexNormals()
  }
  const m = new THREE.Mesh(g, mat)
  m.scale.set(sx, sy, sz)
  return m
}

// ---- Fur ears -----------------------------------------------------------------

interface EarSpec {
  card: PlushSpec
  /** Where it sits: t along forehead-top→temple, then lifted/pushed. */
  t: number
  lift: THREE.Vector3
  rot: THREE.Euler
  scale: number
}

function earPair(spec: EarSpec) {
  const card = plushCard(spec.card)
  const make = () => {
    const pivot = new THREE.Group()
    pivot.add(card.clone())
    return pivot
  }
  const L = make()
  const R = make()
  R.scale.set(spec.scale, spec.scale, spec.scale)
  L.scale.set(-spec.scale, spec.scale, spec.scale)
  const root = new THREE.Group()
  root.add(L, R)
  const place = (a: Anchors) => {
    for (const [ear, temple, side] of [[L, a.templeL, -1], [R, a.templeR, 1]] as const) {
      ear.position.copy(a.top).lerp(temple, spec.t).add(V(spec.lift.x * side, spec.lift.y, spec.lift.z))
      ear.rotation.set(spec.rot.x, spec.rot.y * side, spec.rot.z * side)
    }
  }
  return { root, place }
}


function animalModel(spec: EarSpec, extras: (root: THREE.Group) => (a: Anchors, rig: Rig, t: number) => void, band?: string): Model {
  const root = new THREE.Group()
  const pair = earPair(spec)
  root.add(pair.root)
  // The ears sit on a slim velvet headband, like the real accessory — it
  // gives them something to be attached to instead of floating on the hair.
  const bandMat = band ? physical({ color: band, roughness: 0.85, sheen: 1, sheenRoughness: 0.5, sheenColor: new THREE.Color(band).lerp(new THREE.Color(0xffffff), 0.35) }) : null
  const bandMesh = bandMat ? new THREE.Mesh(new THREE.BufferGeometry(), bandMat) : null
  if (bandMesh) {
    bandMesh.castShadow = true
    root.add(bandMesh)
  }
  const extra = extras(root)
  return {
    root,
    update(rig, t) {
      const a = anchors(rig)
      pair.place(a)
      if (bandMesh) {
        const [L, R] = pair.root.children
        const mid = a.top.clone().add(V(0, 0.62, -0.78))
        const pts = [
          L.position.clone().add(V(-0.18, -0.55, -0.25)),
          L.position.clone().add(V(0, 0.02, -0.06)),
          mid,
          R.position.clone().add(V(0, 0.02, -0.06)),
          R.position.clone().add(V(0.18, -0.55, -0.25)),
        ]
        bandMesh.geometry.dispose()
        bandMesh.geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 64, 0.04, 10, false)
      }
      extra(a, rig, t)
    },
  }
}

// Painted ear designs (canvas px; the right ear, outward = +x, front = the
// viewer). Left ears are the same card mirrored.
const CAT: PlushSpec = {
  w: 512, h: 600, cardW: 1.0,
  outline: smooth([[50, 590], [80, 330], [190, 120], [300, 18], [370, 120], [455, 340], [478, 590], [260, 600]]),
  inner: smooth([[140, 565], [170, 350], [260, 160], [320, 110], [375, 330], [395, 565], [265, 575]]),
  tip: [305, 0],
  root: '#060608', mid: '#18181d', light: '#7a7a86',
  skin: '#d98ea3', plush: '#f2b3c4', tuft: '#ffffff',
  strand: [14, 34],
}
const FOX: PlushSpec = {
  ...CAT, h: 660,
  outline: smooth([[50, 650], [85, 360], [200, 120], [295, 15], [365, 130], [455, 370], [478, 650], [260, 660]]),
  inner: smooth([[140, 625], [170, 390], [265, 175], [318, 125], [375, 365], [395, 625], [265, 635]]),
  tip: [298, 0],
  root: '#6e2a0e', mid: '#c35d1d', light: '#f7ad62',
  earTip: ['#120c0a', 190],
  skin: '#e7a998', plush: '#f8e2d8', tuft: '#ffffff',
}
const BUNNY: PlushSpec = {
  w: 320, h: 900, cardW: 0.5,
  outline: smooth([[95, 890], [62, 600], [72, 260], [138, 40], [200, 28], [262, 200], [272, 600], [238, 890], [165, 900]]),
  inner: smooth([[122, 860], [104, 560], [124, 240], [170, 110], [216, 250], [226, 560], [206, 860]]),
  tip: [172, 0],
  root: '#bdb7c2', mid: '#ebe8ee', light: '#ffffff',
  skin: '#e598ad', plush: '#f7c1d0', tuft: '#ffffff',
  strand: [10, 24],
}
const BEAR: PlushSpec = {
  w: 520, h: 440, cardW: 0.74,
  outline: smooth([[40, 430], [42, 250], [120, 80], [260, 28], [400, 80], [478, 250], [480, 430], [260, 440]]),
  inner: smooth([[130, 420], [128, 270], [190, 160], [260, 130], [330, 160], [392, 270], [390, 420]]),
  tip: [260, -260],
  root: '#21130a', mid: '#5b3820', light: '#b0805a',
  skin: '#b98a68', plush: '#d6ad8a', tuft: '#f0dcc2',
  strand: [14, 32],
}
function nose(mat: THREE.Material, sx: number, sy: number, sz: number, tri = 0.5) {
  // A soft inverted-triangle nose: narrower toward the bottom.
  return blob(sx, sy, sz, mat, (v) => {
    v.x *= 1 + v.y * tri
  })
}

function kitty(): Model {
  // Plush black cosplay ears with a pink inner ear and white tufts.
  return animalModel({ card: CAT, t: 0.68, lift: V(0.06, 0.24, -0.4), rot: new THREE.Euler(-0.15, -0.22, -0.4), scale: 1 }, (root) => {
    const n = nose(physical({ color: 0xf28aa6, roughness: 0.32, clearcoat: 0.7, sheen: 0.3 }), 0.085, 0.06, 0.055)
    root.add(n)
    // Whiskers: thin translucent-white tapered tubes fanning from the muzzle.
    const whiskerMat = physical({ color: 0xf6f2ea, roughness: 0.35, clearcoat: 0.5 })
    const whiskers = new THREE.Group()
    for (const side of [-1, 1])
      for (const a of [-0.13, 0.02, 0.17]) {
        const g = taperTube([V(0, 0, 0), V(side * 0.32, a * 0.6 + 0.02, 0.06), V(side * 0.72, a * 1.5 - 0.02, -0.06)], 0.012, 0.003, 6, 24)
        const m = new THREE.Mesh(g, whiskerMat)
        m.userData.side = side
        whiskers.add(m)
      }
    root.add(whiskers)
    return (a) => {
      n.position.copy(a.noseTip).add(V(0, 0.03, 0.04))
      for (const w of whiskers.children) {
        const wing = w.userData.side < 0 ? a.wingL : a.wingR
        w.position.copy(wing).add(V(w.userData.side * 0.06, -0.04, 0.05))
      }
    }
  })
}

function fox(): Model {
  // Same plush style, fox colouring: russet fur, black tips, white tufts.
  return animalModel({ card: FOX, t: 0.66, lift: V(0.06, 0.22, -0.4), rot: new THREE.Euler(-0.15, -0.22, -0.36), scale: 1 }, (root) => {
    const n = nose(physical({ color: 0x1a1414, roughness: 0.3, clearcoat: 1 }), 0.08, 0.055, 0.05)
    root.add(n)
    return (a) => n.position.copy(a.noseTip).add(V(0, 0.03, 0.04))
  })
}

function bunny(): Model {
  return animalModel({ card: BUNNY, t: 0.4, lift: V(0.02, 0.3, -0.42), rot: new THREE.Euler(-0.2, -0.12, -0.14), scale: 1 }, (root) => {
    const n = nose(physical({ color: 0xff8fab, roughness: 0.35, clearcoat: 0.6 }), 0.055, 0.04, 0.035)
    root.add(n)
    const teethMat = physical({ color: 0xfbfbf6, roughness: 0.18, clearcoat: 1 })
    const teeth = [-1, 1].map((s) => {
      const g = new THREE.BoxGeometry(0.075, 0.12, 0.03, 4, 4, 2)
      const p = g.getAttribute('position') as THREE.BufferAttribute
      for (let i = 0; i < p.count; i++) if (p.getY(i) < 0) p.setX(i, p.getX(i) * 0.92)
      g.computeVertexNormals()
      const m = new THREE.Mesh(g, teethMat)
      m.userData.s = s
      root.add(m)
      return m
    })
    return (a) => {
      n.position.copy(a.noseTip).add(V(0, 0.02, 0.035))
      for (const t of teeth) t.position.copy(a.lipUpperInner).add(V(t.userData.s * 0.04, -0.06, 0.04))
    }
  })
}

function bear(): Model {
  return animalModel({ card: BEAR, t: 0.8, lift: V(0.06, 0.22, -0.45), rot: new THREE.Euler(-0.1, -0.2, -0.3), scale: 1 }, (root) => {
    const n = nose(physical({ color: 0x2a1a10, roughness: 0.3, clearcoat: 1 }), 0.12, 0.08, 0.08, 0.4)
    root.add(n)
    return (a) => n.position.copy(a.noseTip).add(V(0, 0.03, 0.05))
  })
}

// ---- Eyewear ------------------------------------------------------------------
//
// Two sunglasses, each modelled after a product photo. All sizes in rig
// units (1 = distance between the eye centres ≈ 63 mm): a 55 mm lens is
// ≈ 0.87 wide, so a front spans a little over 2 — the face's width.

/** Closed Catmull-Rom spline through control points, as a polygon. */
function closedSpline(ctrl: [number, number][], per = 12) {
  const out: THREE.Vector2[] = []
  const n = ctrl.length
  for (let i = 0; i < n; i++) {
    const [p0, p1, p2, p3] = [ctrl[(i - 1 + n) % n], ctrl[i], ctrl[(i + 1) % n], ctrl[(i + 2) % n]]
    for (let k = 0; k < per; k++) {
      const t = k / per
      const f = (a: number, b: number, c: number, d: number) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t * t * t)
      out.push(new THREE.Vector2(f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])))
    }
  }
  return out
}

const mirrorX = (pts: [number, number][]) => pts.map(([x, y]) => [-x, y] as [number, number]).reverse()

/** Bends a flat front (built in x, y, extruded along z) round the face. */
function bendGeometry(g: THREE.BufferGeometry, z: (x: number, y: number) => number) {
  const p = g.getAttribute('position') as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) p.setZ(i, p.getZ(i) + z(p.getX(i), p.getY(i)))
  g.computeVertexNormals()
  return g
}

/** A temple: a side profile (tall at the hinge, tapering, curving down
 *  behind the ear), extruded to its thickness, running back along -z from
 *  the origin. `paint(along, up, outer)` gives each vertex's colour. */
function templeArm(profile: (s: THREE.Shape) => void, thickness: number, side: number, paint?: (along: number, up: number, outer: boolean) => THREE.Color) {
  const s = new THREE.Shape()
  profile(s)
  const g = new THREE.ExtrudeGeometry(s, { depth: thickness, bevelEnabled: true, bevelThickness: thickness * 0.3, bevelSize: thickness * 0.25, bevelSegments: 3, curveSegments: 28 })
  g.rotateY(Math.PI / 2)
  g.translate(-thickness / 2, 0, 0)
  if (paint) {
    const p = g.getAttribute('position') as THREE.BufferAttribute
    const col: number[] = []
    for (let i = 0; i < p.count; i++) {
      const c = paint(-p.getZ(i), p.getY(i), p.getX(i) * side > thickness * 0.35)
      col.push(c.r, c.g, c.b)
    }
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3))
  }
  return g
}

// What eyewear reflects: a photo studio, the way product shots light it —
// a bright sky band over a sharp horizon, a dark floor, a big overhead
// softbox and two tall window strips. Lenses and gloss pick up its
// gradient and crisp edge highlights, which is most of what makes
// sunglasses read as real (the frame's own environment is just its
// average colours: fine for skin and fur, flat on a lens).
let studio: THREE.Texture | null = null
function studioEnv() {
  if (studio) return studio
  const c = document.createElement('canvas')
  c.width = 512
  c.height = 256
  const g = c.getContext('2d')!
  // Dark all round the horizon — where a surface facing the camera looks —
  // so black stays black face-on and only edges and curves turned up or
  // aside catch the light.
  const sky = g.createLinearGradient(0, 0, 0, 128)
  sky.addColorStop(0, '#eef1f5')
  sky.addColorStop(0.3, '#b4bac2')
  sky.addColorStop(0.5, '#3b3f45')
  sky.addColorStop(0.75, '#16181b')
  sky.addColorStop(1, '#0c0d0e')
  g.fillStyle = sky
  g.fillRect(0, 0, 512, 128)
  const floor = g.createLinearGradient(0, 128, 0, 256)
  floor.addColorStop(0, '#0c0c0c')
  floor.addColorStop(0.4, '#1d1c1b')
  floor.addColorStop(1, '#0a0a0a')
  g.fillStyle = floor
  g.fillRect(0, 128, 512, 128)
  g.filter = 'blur(2px)'
  g.fillStyle = '#ffffff'
  // A long overhead softbox all the way round (a bright band on every top
  // edge and the top of each lens)…
  g.fillRect(0, 14, 512, 14)
  // …a big key softbox above, and window strips high on either side.
  g.fillRect(196, 4, 120, 40)
  g.globalAlpha = 0.8
  g.fillRect(76, 26, 16, 54)
  g.fillRect(420, 26, 16, 54)
  g.globalAlpha = 1
  g.filter = 'none'
  const t = new THREE.CanvasTexture(c)
  t.mapping = THREE.EquirectangularReflectionMapping
  t.colorSpace = THREE.SRGBColorSpace
  return (studio = t)
}

// What a lens reflects: what's in front of the wearer — a bright sky
// above the horizon fading to the dark ground below — so the top of each
// curved lens shows the classic bright sheen and the bottom stays dark.
let sky: THREE.Texture | null = null
function lensEnv() {
  if (sky) return sky
  const c = document.createElement('canvas')
  c.width = 512
  c.height = 256
  const g = c.getContext('2d')!
  const grad = g.createLinearGradient(0, 0, 0, 256)
  // (The skyline sits a little below straight ahead: glasses tip forward
  // on the nose, so a lens facing the camera looks slightly down.)
  grad.addColorStop(0, '#f2f5f9')
  grad.addColorStop(0.35, '#d0d6de')
  grad.addColorStop(0.5, '#a3abb5')
  grad.addColorStop(0.57, '#68707a')
  grad.addColorStop(0.61, '#2b2a28')
  grad.addColorStop(0.75, '#141312')
  grad.addColorStop(1, '#070707')
  g.fillStyle = grad
  g.fillRect(0, 0, 512, 256)
  // A few soft shapes on the skyline (trees, buildings), so the sheen
  // isn't a perfect gradient.
  g.filter = 'blur(4px)'
  g.fillStyle = 'rgba(40,44,48,0.55)'
  for (const [x, w, h] of [[40, 60, 18], [150, 30, 30], [300, 80, 14], [420, 40, 24]]) g.fillRect(x, 154 - h, w, h)
  g.filter = 'none'
  const t = new THREE.CanvasTexture(c)
  t.mapping = THREE.EquirectangularReflectionMapping
  t.colorSpace = THREE.SRGBColorSpace
  return (sky = t)
}

/** A lens: a fan of rings from its centre out to `outline`, so it has
 *  vertices all over and really follows the curve (a flat triangulation
 *  only bends at its edge). `tint(y)`: rgba by height, for gradient
 *  lenses. */
function lensGeometry(outline: THREE.Vector2[], bend: (x: number, y: number) => number, lift: number, tint: (y: number) => [number, number, number, number]) {
  const c = outline.reduce((a, p) => a.add(p), new THREE.Vector2()).divideScalar(outline.length)
  const R = 10
  const n = outline.length
  // A real lens is curved both ways (its base curve), not just round the
  // face: it's what spreads the sky's reflection down it.
  const curve = (x: number, y: number) => bend(x, y) - 0.45 * (y - c.y) ** 2 - 0.1 * (x - c.x) ** 2
  const pos: number[] = []
  const col: number[] = []
  const idx: number[] = []
  pos.push(c.x, c.y, curve(c.x, c.y) + lift)
  col.push(...tint(c.y))
  for (let r = 1; r <= R; r++)
    for (let i = 0; i < n; i++) {
      const p = c.clone().lerp(outline[i], r / R)
      pos.push(p.x, p.y, curve(p.x, p.y) + lift)
      col.push(...tint(p.y))
    }
  const at = (r: number, i: number) => (r === 0 ? 0 : 1 + (r - 1) * n + (i % n))
  for (let i = 0; i < n; i++) idx.push(0, at(1, i), at(1, i + 1))
  for (let r = 1; r < R; r++)
    for (let i = 0; i < n; i++) idx.push(at(r, i), at(r + 1, i), at(r + 1, i + 1), at(r, i), at(r + 1, i + 1), at(r, i + 1))
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 4))
  g.setIndex(idx)
  g.computeVertexNormals()
  // (Facing the camera whichever way the outline ran.)
  const nz = g.getAttribute('normal').getZ(0)
  if (nz < 0) {
    g.setIndex(idx.map((_, k) => idx[k - (k % 3) + 2 - (k % 3)]))
    g.computeVertexNormals()
  }
  return g
}

/** Coated sunglass lens: the glass itself nearly opaque (the vertex
 *  alpha), its surface a smooth grey reflector — the coating sends back
 *  about a quarter of the light, which is what puts the sky's sheen on
 *  real sunglasses (plain glass, 4%, barely shows; the colour is sRGB, so
 *  0x8a8f96 is ~25% linear) — and a clear coat for
 *  the white Fresnel rim at grazing angles. */
const lensGlass = (o: THREE.MeshPhysicalMaterialParameters = {}) =>
  physical({
    color: 0x8a8f96,
    vertexColors: true,
    metalness: 1,
    roughness: 0.035,
    clearcoat: 1,
    clearcoatRoughness: 0.02,
    envMap: lensEnv(),
    envMapIntensity: 1.1,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    ...o,
  })

/** A one-piece front from a right-half outline (top centre round to the
 *  bridge) and the right lens opening, mirrored, extruded and bent. */
function frontGeometry(outerR: [number, number][], holeR: [number, number][], depth: number, bevel: number, bend: (x: number, y: number) => number) {
  const half = outerR.filter(([x]) => x > 1e-6)
  const outer = closedSpline([[0, outerR[0][1]], ...half, [0, outerR[outerR.length - 1][1]], ...mirrorX(half)], 10)
  const hole = closedSpline(holeR, 12)
  const holeL = hole.map((v) => new THREE.Vector2(-v.x, v.y)).reverse()
  const shape = new THREE.Shape(outer)
  // Holes wind against the outline.
  const area = (pts: THREE.Vector2[]) => pts.reduce((s, p, i) => s + p.x * pts[(i + 1) % pts.length].y - pts[(i + 1) % pts.length].x * p.y, 0)
  const against = (pts: THREE.Vector2[]) => (Math.sign(area(pts)) === Math.sign(area(outer)) ? pts.slice().reverse() : pts)
  shape.holes.push(new THREE.Path(against(hole)), new THREE.Path(against(holeL)))
  const g = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel * 0.85, bevelSegments: 6, curveSegments: 14 })
  g.translate(0, 0, -depth)
  bendGeometry(g, bend)
  return { geometry: g, hole, holeL }
}

/** Sport wraparound, after the first photo: one continuous matte-black
 *  front — a heavy straight brow bar, a slim rim under the lens — wrapping
 *  hard round to the temples; big dark polarised lenses whose lower edge
 *  sweeps up to a narrow bridge; tall temples with grey rubber inlays
 *  along their lower outer side and grey tips. */
function sportGlasses(): Model {
  const outerR: [number, number][] = [
    [0, 0.31], [0.5, 0.35], [0.98, 0.34], [1.15, 0.25], [1.19, 0.04], [1.12, -0.19],
    [0.93, -0.35], [0.64, -0.43], [0.38, -0.42], [0.21, -0.33], [0.12, -0.15], [0.08, 0.02], [0.04, 0.08], [0, 0.09],
  ]
  const holeR: [number, number][] = [
    [0.16, 0.19], [0.55, 0.215], [0.95, 0.2], [1.09, 0.11], [1.12, -0.04], [1.05, -0.18],
    [0.88, -0.31], [0.63, -0.385], [0.39, -0.375], [0.25, -0.29], [0.18, -0.13], [0.155, 0.04],
  ]
  // An 8-base wrap: curved all the way across, sweeping back at the ends.
  const bend = (x: number, y: number) => -0.2 * x * x - 1.25 * Math.max(0, Math.abs(x) - 0.72) ** 2 - 0.05 * y * y
  const { geometry, hole, holeL } = frontGeometry(outerR, holeR, 0.075, 0.016, bend)
  const env = studioEnv()
  const matte = physical({ color: 0x18191c, roughness: 0.42, metalness: 0, clearcoat: 0.35, clearcoatRoughness: 0.35, envMap: env, envMapIntensity: 2 })
  const glasses = new THREE.Group()
  glasses.add(new THREE.Mesh(geometry, matte))
  // Solid dark polarised lenses, a shade lighter low down.
  const lensMat = lensGlass()
  for (const h of [hole, holeL]) {
    const lens = new THREE.Mesh(lensGeometry(h, bend, -0.03, (y) => [1, 1, 1, 0.9 + 0.06 * THREE.MathUtils.smoothstep(y, -0.35, 0.2)]), lensMat)
    lens.renderOrder = 5
    glasses.add(lens)
  }
  // Temples: tall at the hinge (as deep as the front's end), tapering back.
  const black = new THREE.Color(0x18191c)
  const grey = new THREE.Color(0x9a9da2)
  const templeMat = physical({ vertexColors: true, roughness: 0.45, clearcoat: 0.3, clearcoatRoughness: 0.35, envMap: env, envMapIntensity: 2 })
  for (const side of [-1, 1]) {
    const g = templeArm(
      (s) => {
        s.moveTo(0, 0.16)
        s.bezierCurveTo(0.35, 0.14, 0.7, 0.07, 1.0, 0.05)
        s.quadraticCurveTo(1.14, 0.04, 1.2, -0.04)
        s.lineTo(1.16, -0.08)
        s.quadraticCurveTo(1.1, -0.02, 0.98, -0.02)
        s.bezierCurveTo(0.7, -0.03, 0.35, -0.11, 0, -0.16)
        s.lineTo(0, 0.16)
      },
      0.055,
      side,
      (along, up, outerFace) => {
        if (along > 1.02) return grey
        if (outerFace && along > 0.12 && along < 0.9 && up < 0.02 - 0.06 * along) return grey
        return black
      },
    )
    const t = new THREE.Mesh(g, templeMat)
    const hx = 1.14
    t.position.set(side * hx, 0.05, bend(hx, 0.05) - 0.06)
    t.rotation.y = side * 0.06
    glasses.add(t)
  }
  return glassesModel(glasses, 0.16)
}

/** Square smart glasses, after the second photo: thick glossy black
 *  acetate with softly rounded edges, a heavy straight brow, broad lenses
 *  a little wider at the top, a keyhole bridge, chunky end pieces with a
 *  camera lens at each top outer corner, and thick flat temples; grey
 *  lenses that darken towards the top. */
function wayfarerGlasses(): Model {
  const outerR: [number, number][] = [
    [0, 0.355], [0.5, 0.375], [0.95, 0.38], [1.05, 0.34], [1.08, 0.22], [1.07, -0.08], [1.03, -0.27],
    [0.93, -0.37], [0.74, -0.41], [0.5, -0.415], [0.3, -0.38], [0.18, -0.29], [0.13, -0.13], [0.11, 0.05], [0.07, 0.15], [0.03, 0.18], [0, 0.185],
  ]
  const holeR: [number, number][] = [
    [0.2, 0.25], [0.56, 0.275], [0.9, 0.27], [0.97, 0.21], [0.99, 0.08], [0.98, -0.1], [0.94, -0.24],
    [0.84, -0.32], [0.62, -0.345], [0.4, -0.33], [0.27, -0.27], [0.21, -0.15], [0.195, 0.02], [0.2, 0.15], [0.235, 0.23],
  ]
  const bend = (x: number, y: number) => -0.075 * x * x - 0.03 * Math.max(0, Math.abs(x) - 0.9) - 0.02 * y * y
  const { geometry, hole, holeL } = frontGeometry(outerR, holeR, 0.105, 0.03, bend)
  const env = studioEnv()
  // (The studio is a plain picture, no brighter than white; real softboxes
  // are many times that, which is why gloss shows such crisp highlights.)
  const gloss = physical({ color: 0x060607, roughness: 0.16, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.03, envMap: env, envMapIntensity: 3.2 })
  const glasses = new THREE.Group()
  glasses.add(new THREE.Mesh(geometry, gloss))
  // Grey gradient lenses: dark at the top, the eyes faintly through below.
  const lensMat = lensGlass({ color: 0x868a90 })
  for (const h of [hole, holeL]) {
    const lens = new THREE.Mesh(lensGeometry(h, bend, -0.05, (y) => [1, 1, 1, 0.8 + 0.16 * THREE.MathUtils.smoothstep(y, -0.3, 0.25)]), lensMat)
    lens.renderOrder = 5
    glasses.add(lens)
  }
  // The cameras: a dark glass disc in a satin ring, top outer corners.
  for (const side of [-1, 1]) {
    const cam = new THREE.Group()
    const ring = new THREE.Mesh(new THREE.CylinderGeometry(0.034, 0.034, 0.012, 28), physical({ color: 0x3a3b3f, metalness: 0.9, roughness: 0.3, envMap: env }))
    const glass = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.016, 28), physical({ color: 0x030405, roughness: 0.02, clearcoat: 1, envMap: env, envMapIntensity: 2 }))
    ring.rotation.x = glass.rotation.x = Math.PI / 2
    cam.add(ring, glass)
    const x = side * 1.04
    cam.position.set(x, 0.33, bend(x, 0.33) + 0.045)
    glasses.add(cam)
  }
  // Thick, flat temples, hooking down at the end.
  for (const side of [-1, 1]) {
    const g = templeArm(
      (s) => {
        s.moveTo(0, 0.08)
        s.lineTo(0.95, 0.06)
        s.quadraticCurveTo(1.12, 0.055, 1.2, -0.08)
        s.lineTo(1.15, -0.11)
        s.quadraticCurveTo(1.08, -0.02, 0.95, -0.03)
        s.lineTo(0, -0.075)
        s.lineTo(0, 0.08)
      },
      0.07,
      side,
    )
    const t = new THREE.Mesh(g, gloss)
    t.position.set(side * 1.06, 0.2, bend(1.06, 0.2) - 0.08)
    t.rotation.y = side * -0.04
    glasses.add(t)
  }
  return glassesModel(glasses, 0.14)
}

function glassesModel(glasses: THREE.Group, forward: number): Model {
  shadowed(glasses)
  glasses.traverse((o) => {
    const m = o as THREE.Mesh
    if (m.isMesh && m.renderOrder === 5) m.castShadow = false
  })
  const root = new THREE.Group()
  root.add(glasses)
  return {
    root,
    // Thin frames and tinted glass: a light, close shadow, not a band.
    shadow: 0.38,
    update(rig) {
      const a = anchors(rig)
      // Glasses ride on the nose: lenses centred a little below the eyes.
      const y = (a.eyeL.y + a.eyeR.y) / 2 - 0.04
      glasses.position.set(0, y, a.bridge.z + forward)
      glasses.rotation.set(-0.08, 0, 0)
    },
  }
}

// ---- Props --------------------------------------------------------------------

/** The top of the head, centred over the skull: landmark 10 is the top of
 *  the forehead, at the front; the crown of the head (with hair) is higher
 *  and well behind it. */
function crownOf(a: Anchors) {
  return V(a.top.x, a.top.y + 0.62, -1.15)
}

function crown(): Model {
  const g = new THREE.CylinderGeometry(0.6, 0.64, 0.36, 160, 8, true)
  const p = g.getAttribute('position') as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) {
    const y = p.getY(i)
    const k = Math.max(0, (y + 0.18) / 0.36)
    const a = Math.atan2(p.getX(i), p.getZ(i))
    const spike = Math.pow((Math.cos(a * 5) + 1) / 2, 4) * 0.42
    p.setY(i, y + spike * Math.pow(k, 1.5))
  }
  // A touch of perspective (the scene's camera is orthographic): the far
  // side of the ring drawn a little narrower than the near side.
  const taper = (v: THREE.Vector3) => v.setX(v.x * (1 + 0.07 * (v.z / 0.64)))
  const tv = new THREE.Vector3()
  for (let i = 0; i < p.count; i++) {
    taper(tv.fromBufferAttribute(p, i))
    p.setX(i, tv.x)
  }
  g.computeVertexNormals()
  const goldMat = gold()
  const crownMesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0xe8b84a, metalness: 1, roughness: 0.2, side: THREE.DoubleSide }))
  const band = new THREE.Mesh(new THREE.TorusGeometry(0.645, 0.045, 16, 120), goldMat)
  band.rotation.x = Math.PI / 2
  band.position.y = -0.17
  const grp = new THREE.Group()
  grp.add(crownMesh, band)
  const gem = (color: number, r: number) => new THREE.Mesh(new THREE.OctahedronGeometry(r, 1), physical({ color, roughness: 0.02, clearcoat: 1, specularIntensity: 1, ior: 2.2, emissive: color, emissiveIntensity: 0.15 }))
  for (let k = 0; k < 5; k++) {
    const a = (k / 5) * Math.PI * 2
    const tip = gem(k % 2 ? 0x2f9bff : 0xff2d55, 0.055)
    tip.position.set(Math.sin(a) * 0.62, 0.18 + 0.45, Math.cos(a) * 0.62)
    const set = gem(k % 2 ? 0xff2d55 : 0x9b4dff, 0.05)
    set.position.set(Math.sin(a + Math.PI / 5) * 0.67, -0.05, Math.cos(a + Math.PI / 5) * 0.67)
    grp.add(tip, set)
  }
  shadowed(grp)
  const root = new THREE.Group()
  root.add(grp)
  return {
    root,
    update(rig) {
      const a = anchors(rig)
      // Raised clear of the head, so no part of the ring sinks into it,
      // and tipped toward the camera, so it reads as seen from in front
      // and a little above — the far rim higher than the near one.
      grp.position.copy(crownOf(a)).add(V(0, 0.48, 0.05))
      grp.rotation.set(0.16, 0, 0)
    },
  }
}

// ---- Faun (after Pan's Labyrinth) ------------------------------------------------

/** A ram's horn, after the faun's: from a broad root on top of the head
 *  it grows back, then coils — down behind, round under and up in front,
 *  about two turns, each smaller than the last and drifting outward so the
 *  coils lie side by side — and the tip flicks forward and up. The horn
 *  itself twists as it grows: its rounded-triangular section, its deep,
 *  uneven growth rings and the keel along it turn twice round from root to
 *  tip. Rig units, right side (mirrored for the left). */
function ramHorn(side: number) {
  // The coil: a conical spiral in a plane facing out and forward.
  const u = V(0, 1, 0)
  const w = V(side * 0.7, 0, -0.7).normalize() // back and out
  const n = V(side * 0.7, 0, 0.7).normalize() // out and forward: the coil's axis
  const C = V(side * 1.5, 0.55, -0.95)
  const TURNS = 1.45
  const pts: THREE.Vector3[] = [V(side * 0.52, 0.95, -0.42), V(side * 0.95, 1.45, -0.62)]
  for (let k = 0; k <= 28; k++) {
    const f = k / 28
    const th = f * TURNS * Math.PI * 2
    const r = 1.0 * (1 - 0.38 * f)
    pts.push(C.clone().addScaledVector(u, r * Math.cos(th)).addScaledVector(w, r * Math.sin(th)).addScaledVector(n, 0.95 * f))
  }
  // The tip flicks forward and up.
  const end = pts[pts.length - 1]
  pts.push(end.clone().add(V(side * 0.15, 0.12, 0.35)), end.clone().add(V(side * 0.3, 0.3, 0.6)))
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal')
  const N = 700
  const R = 28
  const frames = curve.computeFrenetFrames(N, false)
  const pos: number[] = []
  const col: number[] = []
  const idx: number[] = []
  const groove = new THREE.Color('#2e241a')
  const crest = new THREE.Color('#8b7558')
  const tipCol = new THREE.Color('#b9a582')
  const RINGS = 80
  const TWIST = 2 * Math.PI * 2
  for (let i = 0; i <= N; i++) {
    const t = i / N
    const p = curve.getPointAt(t)
    const radius = 0.33 * (1 - t) ** 0.6 + 0.035
    const ringAmt = 1 - THREE.MathUtils.smoothstep(t, 0.82, 0.97)
    for (let j = 0; j <= R; j++) {
      const a = (j / R) * Math.PI * 2
      // The section turns along the horn: everything below is measured in
      // the twisted angle.
      const at = a - t * TWIST
      const ph = t * RINGS + 2.2 * Math.sin(t * 9) + 0.9 * Math.sin(t * 23 + 1) + 0.45 * Math.sin(at * 2 + t * 17) + 0.25 * Math.sin(at * 5 + t * 40)
      const ridge = Math.pow(0.5 + 0.5 * Math.cos(ph * Math.PI * 2), 2.5)
      const tri = 1 + 0.14 * Math.cos(3 * at)
      const r = radius * tri * (1 - 0.09 * ringAmt * (1 - ridge))
      const nrm = frames.normals[i].clone().multiplyScalar(Math.cos(a)).add(frames.binormals[i].clone().multiplyScalar(Math.sin(a)))
      pos.push(p.x + nrm.x * r, p.y + nrm.y * r, p.z + nrm.z * r)
      // A darker keel line spiralling along the horn shows the twist.
      const keel = Math.exp(-(((Math.atan2(Math.sin(at), Math.cos(at)) - 0.0) / 0.22) ** 2))
      const c = groove
        .clone()
        .lerp(crest, 0.25 + 0.75 * ridge * ringAmt + (1 - ringAmt) * 0.5)
        .lerp(tipCol, THREE.MathUtils.smoothstep(t, 0.6, 1) * 0.8)
        .multiplyScalar(1 - 0.35 * keel)
      col.push(c.r, c.g, c.b)
    }
  }
  for (let i = 0; i < N; i++)
    for (let j = 0; j < R; j++) {
      const a = i * (R + 1) + j
      const b = a + R + 1
      idx.push(a, b, a + 1, a + 1, b, b + 1)
    }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3))
  g.setIndex(side > 0 ? idx : idx.map((_, k) => idx[k - (k % 3) + [0, 2, 1][k % 3]]))
  g.computeVertexNormals()
  return g
}

/** A faun's goat ear: a broad leaf, rounded and then drawn to a point,
 *  thick, with a rolled rim and a deep cup on its front. Local frame: x
 *  along the ear, z out of its face. */
function faunEarGeometry() {
  const g = new THREE.SphereGeometry(1, 64, 40)
  const p = g.getAttribute('position') as THREE.BufferAttribute
  const col: number[] = []
  const v = new THREE.Vector3()
  const outer = new THREE.Color(0x8a7058)
  const inner = new THREE.Color(0x6e4a3a)
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i)
    const t = (v.x + 1) / 2
    // Broad near the root, widest at 40%, drawn to a point at the tip.
    const w = Math.sin(Math.PI * Math.min(1, t * 1.05)) ** 0.7 * (1 - 0.45 * t ** 1.5)
    const hollow = v.z > 0 ? (1 - v.y * v.y) ** 1.5 * w * (1 - t * 0.5) : 0
    p.setXYZ(i, t * 1.1, v.y * 0.55 * w, (v.z * 0.15 - hollow * 0.24) * (0.35 + w))
    const c = outer.clone().lerp(inner, Math.min(1, hollow * 1.7))
    col.push(c.r, c.g, c.b)
  }
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3))
  g.computeVertexNormals()
  return g
}

/** Short soft hair for the ears' fuzz: fine strands, mostly transparent. */
function fuzzTexture() {
  const c = document.createElement('canvas')
  c.width = c.height = 256
  const ctx = c.getContext('2d')!
  let seed = 3
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647
  ctx.lineCap = 'round'
  for (let i = 0; i < 2600; i++) {
    const x = rnd() * 256
    const y = rnd() * 256
    const a = -Math.PI / 2 + (rnd() - 0.5) * 0.9
    const l = 4 + rnd() * 9
    ctx.strokeStyle = `rgba(${200 + rnd() * 40},${170 + rnd() * 30},${130 + rnd() * 30},${0.35 + rnd() * 0.45})`
    ctx.lineWidth = 0.6 + rnd() * 0.8
    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l)
    ctx.stroke()
  }
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  t.wrapS = t.wrapT = THREE.RepeatWrapping
  t.repeat.set(3, 2)
  return t
}

/** The faun's forehead: whorls in relief — broad raised coils, two big
 *  ones over the brows and a smaller one between and below them, with
 *  sweeping ridges joining them — in the skin itself, so they show only by
 *  their light and shade. Painted as a height map, then lit from above:
 *  highlights on the upper slopes, shadow under each coil. Laid on the
 *  face (u, v from the face's own x, y). */
function faunPattern() {
  const S = 512
  // Rig (x, y) → canvas: x ∈ [-0.9, 0.9], y ∈ [-0.45, 1.35].
  const X = (x: number) => (0.5 + x / 1.8) * S
  const Y = (y: number) => (1 - (0.5 + (y - 0.45) / 1.8)) * S
  const h = document.createElement('canvas')
  h.width = h.height = S
  // (in memory: it's read back below, and a GPU canvas stalls on that)
  const hx = h.getContext('2d', { willReadFrequently: true })!
  hx.fillStyle = '#000'
  hx.fillRect(0, 0, S, S)
  hx.lineCap = 'round'
  hx.lineJoin = 'round'
  const stroke = (pts: [number, number][], w: number) => {
    hx.lineWidth = w
    hx.beginPath()
    pts.forEach(([x, y], k) => (k ? hx.lineTo(X(x), Y(y)) : hx.moveTo(X(x), Y(y))))
    hx.stroke()
  }
  const coil = (cx: number, cy: number, r: number, turns: number, dir: number, rot: number) => {
    const pts: [number, number][] = []
    for (let k = 0; k <= 240; k++) {
      const t = k / 240
      const a = rot + dir * t * turns * Math.PI * 2
      pts.push([cx + Math.cos(a) * r * (0.12 + 0.88 * t), cy + Math.sin(a) * r * (0.12 + 0.88 * t)])
    }
    return pts
  }
  hx.filter = 'blur(5px)'
  hx.strokeStyle = '#fff'
  const W = 15
  stroke(coil(-0.3, 0.58, 0.27, 2.6, 1, -0.4), W)
  stroke(coil(0.3, 0.58, 0.27, 2.6, -1, Math.PI + 0.4), W)
  stroke(coil(0, 0.3, 0.13, 1.8, 1, Math.PI / 2), W * 0.85)
  // Ridges sweeping from the whorls up toward the hairline, and in over
  // the brows.
  stroke([[-0.57, 0.62], [-0.6, 0.82], [-0.48, 1.0]], W)
  stroke([[0.57, 0.62], [0.6, 0.82], [0.48, 1.0]], W)
  stroke([[-0.12, 0.3], [-0.2, 0.2], [-0.42, 0.22]], W * 0.8)
  stroke([[0.12, 0.3], [0.2, 0.2], [0.42, 0.22]], W * 0.8)
  const H = hx.getImageData(0, 0, S, S).data
  const c = document.createElement('canvas')
  c.width = c.height = S
  const cx = c.getContext('2d')!
  const out = cx.createImageData(S, S)
  const at = (x: number, y: number) => H[(Math.min(S - 1, Math.max(0, y)) * S + Math.min(S - 1, Math.max(0, x))) * 4] / 255
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++) {
      // Slope facing up (toward a light above) brightens; facing down darkens.
      const dy = at(x, y + 2) - at(x, y - 2)
      const dx = at(x + 2, y) - at(x - 2, y)
      const lit = dy * 1.6 + dx * 0.4
      const k = (y * S + x) * 4
      // Fade out at the hairline and below the brows.
      const v = 1 - y / S
      const fade = THREE.MathUtils.smoothstep(v, 0.2, 0.3) * (1 - THREE.MathUtils.smoothstep(v, 0.78, 0.86))
      // Only the coils' real slopes: no haze from the blur's faint tails.
      const m = Math.max(0, Math.abs(lit) - 0.035)
      if (lit > 0) {
        out.data[k] = 255
        out.data[k + 1] = 242
        out.data[k + 2] = 225
        out.data[k + 3] = Math.min(220, m * 380 * fade)
      } else {
        out.data[k] = 60
        out.data[k + 1] = 36
        out.data[k + 2] = 22
        out.data[k + 3] = Math.min(230, m * 360 * fade)
      }
    }
  cx.putImageData(out, 0, 0)
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 8
  return tex
}

function faun(): Model {
  const hornMat = physical({ vertexColors: true, roughness: 0.68, clearcoat: 0.15, clearcoatRoughness: 0.6 })
  const horns = [-1, 1].map((side) => {
    const m = new THREE.Mesh(ramHorn(side), hornMat)
    m.castShadow = true
    return m
  })
  // Bark-brown skin, like the faun's; leathery, with a soft sheen.
  const earMat = physical({ vertexColors: true, roughness: 0.7, sheen: 0.5, sheenColor: new THREE.Color(0xd8b48a), side: THREE.DoubleSide })
  const earGeo = faunEarGeometry()
  // Fuzz: a slightly larger shell of short soft hair round each ear.
  const fuzzMat = new THREE.MeshStandardMaterial({ map: fuzzTexture(), transparent: true, depthWrite: false, roughness: 1, side: THREE.DoubleSide })
  const ears = [-1, 1].map((side) => {
    const m = new THREE.Group()
    const ear = new THREE.Mesh(earGeo, earMat)
    ear.castShadow = true
    const fuzz = new THREE.Mesh(earGeo, fuzzMat)
    fuzz.scale.set(1.03, 1.12, 1.25)
    fuzz.renderOrder = 3
    m.add(ear, fuzz)
    m.scale.set(side, 1, 1)
    return m
  })
  // The forehead pattern, laid on the live face mesh.
  const N = 468
  const decalGeo = new THREE.BufferGeometry()
  decalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3))
  decalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(N * 2), 2))
  decalGeo.setIndex(Array.from(FACE_TRIANGULATION))
  const decal = new THREE.Mesh(
    decalGeo,
    new THREE.MeshStandardMaterial({ map: faunPattern(), transparent: true, roughness: 0.8, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -8, side: THREE.DoubleSide }),
  )
  decal.frustumCulled = false
  decal.renderOrder = 4
  const root = new THREE.Group()
  root.add(...horns, ...ears, decal)
  return {
    root,
    update(rig) {
      const a = anchors(rig)
      // The horns' paths start from a forehead top at (0, 0.8, -0.14);
      // shift them onto this one.
      for (const h of horns) h.position.set(0, a.top.y - 0.8, a.top.z + 0.14)
      // Ears stand out from the sides of the head, below the horns.
      ears.forEach((e, k) => {
        const side = k ? 1 : -1
        const temple = side < 0 ? a.templeL : a.templeR
        // Tucked up under the horns, sticking straight out to the side.
        // Out sideways under the horns, angled down and a little back.
        e.position.set(temple.x - side * 0.02, temple.y - 0.45, temple.z - 0.3)
        e.rotation.set(0.15, side * 0.35, side * -0.5)
      })
      const pos = decalGeo.getAttribute('position') as THREE.BufferAttribute
      const uv = decalGeo.getAttribute('uv') as THREE.BufferAttribute
      for (let i = 0; i < N; i++) {
        const p = rig.local(i)
        pos.setXYZ(i, p.x, p.y, p.z + 0.008)
        uv.setXY(i, 0.5 + p.x / 1.8, 0.5 + (p.y - 0.45) / 1.8)
      }
      pos.needsUpdate = true
      uv.needsUpdate = true
      decalGeo.computeVertexNormals()
      decalGeo.computeBoundingSphere()
    },
  }
}

function glowSprite(color: string) {
  const c = document.createElement('canvas')
  c.width = c.height = 128
  const ctx = c.getContext('2d')!
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64)
  g.addColorStop(0, color)
  g.addColorStop(1, 'rgba(255,220,120,0)')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, 128, 128)
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  return new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }))
}

function angel(): Model {
  const halo = new THREE.Mesh(new THREE.TorusGeometry(0.58, 0.055, 24, 140), new THREE.MeshStandardMaterial({ color: 0xffe08a, emissive: 0xffc94a, emissiveIntensity: 1.6, metalness: 0.6, roughness: 0.25 }))
  const glow = glowSprite('rgba(255,220,120,0.55)')
  glow.scale.set(2.1, 1.1, 1)
  const grp = new THREE.Group()
  grp.add(halo)
  const root = new THREE.Group()
  root.add(grp, glow)
  return {
    root,
    update(rig, t) {
      const a = anchors(rig)
      const bob = Math.sin(t * 2.2) * 0.05
      grp.position.copy(crownOf(a)).add(V(0, 0.75 + bob, 0))
      grp.rotation.set(Math.PI / 2 - 0.12, 0, 0)
      glow.position.copy(grp.position)
    },
  }
}

function stars(): Model {
  const shape = new THREE.Shape()
  for (let i = 0; i < 10; i++) {
    const a = (i / 10) * Math.PI * 2 + Math.PI / 2
    const r = i % 2 ? 0.065 : 0.16
    if (i) shape.lineTo(Math.cos(a) * r, Math.sin(a) * r)
    else shape.moveTo(Math.cos(a) * r, Math.sin(a) * r)
  }
  const geo = new THREE.ExtrudeGeometry(shape, { depth: 0.04, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.015, bevelSegments: 3 })
  geo.center()
  const mats = [gold(), physical({ color: 0xfff6e0, roughness: 0.15, metalness: 0.3, clearcoat: 1, iridescence: 0.6 })]
  const orbit = new THREE.Group()
  const list = Array.from({ length: 9 }, (_, i) => {
    const m = new THREE.Mesh(geo, mats[i % 3 === 0 ? 1 : 0])
    m.castShadow = true
    orbit.add(m)
    return m
  })
  const root = new THREE.Group()
  root.add(orbit)
  return {
    root,
    update(rig, t) {
      const a = anchors(rig)
      orbit.position.copy(crownOf(a)).add(V(0, 0.2, 0))
      orbit.rotation.set(0.1, 0, 0)
      list.forEach((m, i) => {
        const ang = t * 0.9 + (i / list.length) * Math.PI * 2
        m.position.set(Math.sin(ang) * 1.5, Math.sin(ang * 2 + i) * 0.06, Math.cos(ang) * 1.5)
        m.rotation.set(0, ang * 2.5, Math.sin(t + i) * 0.3)
        m.scale.setScalar(0.85 + 0.25 * Math.sin(i * 1.7))
      })
    },
  }
}

// ---- Custom face sticker ------------------------------------------------------

/** The user's own square picture worn over the face: a gently curved card
 *  in front of it, turning with the head. Picks up a new picture whenever
 *  the source's version changes. */
export function faceSticker(source: () => { image: HTMLCanvasElement | null; version: number }): Model {
  const g = new THREE.PlaneGeometry(1, 1, 16, 16)
  const p = g.getAttribute('position') as THREE.BufferAttribute
  // Curved round the face a little, like a card held to it.
  for (let i = 0; i < p.count; i++) p.setZ(i, -0.22 * p.getX(i) ** 2 - 0.06 * p.getY(i) ** 2)
  g.computeVertexNormals()
  const tex = new THREE.CanvasTexture(document.createElement('canvas'))
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 8
  const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, toneMapped: false })
  const card = new THREE.Mesh(g, mat)
  card.castShadow = true
  const root = new THREE.Group()
  root.add(card)
  let version = -1
  return {
    root,
    update(rig) {
      const src = source()
      card.visible = !!src.image
      if (src.image && src.version !== version) {
        version = src.version
        tex.image = src.image
        tex.needsUpdate = true
      }
      // Covers the face from brow to chin, centred on it, in front of the
      // nose.
      const a = anchors(rig)
      const size = 2.9
      card.scale.set(size, size, size)
      card.position.set(0, (a.top.y + rig.local(152).y) / 2 - 0.05, a.noseTip.z + 0.35)
    },
  }
}

export function buildModel(id: string): Model {
  switch (id) {
    case 'kitty': return kitty()
    case 'fox': return fox()
    case 'bunny': return bunny()
    case 'bear': return bear()
    case 'sport': return sportGlasses()
    case 'wayfarer': return wayfarerGlasses()
    case 'crown': return crown()
    case 'faun': return faun()
    case 'angel': return angel()
    case 'stars': return stars()
    default: return { root: new THREE.Group() }
  }
}
