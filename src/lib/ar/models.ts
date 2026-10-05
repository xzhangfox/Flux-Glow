import * as THREE from 'three'
import { plushCard, smooth, type PlushSpec } from './plush'
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
const DOG: PlushSpec = {
  w: 400, h: 760, cardW: 0.66, hang: true,
  outline: smooth([[110, 10], [300, 10], [360, 200], [372, 480], [322, 700], [220, 755], [118, 700], [58, 480], [62, 200]]),
  tip: [212, 800],
  root: '#3c2112', mid: '#9a6238', light: '#e2ad72',
  strand: [18, 40],
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

function puppy(): Model {
  return animalModel({ card: DOG, t: 1, lift: V(0.2, 0.24, -0.45), rot: new THREE.Euler(0.1, -0.35, 0.22), scale: 1 }, (root) => {
    const black = physical({ color: 0x141416, roughness: 0.28, clearcoat: 1, clearcoatRoughness: 0.15 })
    const n = nose(black, 0.2, 0.13, 0.13, 0.35)
    const nostrilMat = new THREE.MeshStandardMaterial({ color: 0x050505, roughness: 0.9 })
    const nostrils = [-1, 1].map((s) => {
      const m = blob(0.045, 0.028, 0.03, nostrilMat)
      m.position.set(s * 0.075, -0.035, 0.115)
      n.add(m)
      return m
    })
    void nostrils
    root.add(n)
    // Tongue: shown when the mouth opens, longer the wider it opens.
    const tongue = shadowed(
      new THREE.Mesh(
        new THREE.CapsuleGeometry(0.13, 0.32, 8, 24),
        physical({ color: 0xe55d7c, roughness: 0.42, clearcoat: 0.6, clearcoatRoughness: 0.2, sheen: 0.4, sheenColor: new THREE.Color(0xffb0c0) }),
      ),
    )
    tongue.scale.set(1, 1, 0.38)
    root.add(tongue)
    return (a, rig) => {
      n.position.copy(a.noseTip).add(V(0, 0.04, 0.07))
      const open = THREE.MathUtils.clamp((rig.mouthOpen - 0.12) / 0.3, 0, 1)
      tongue.visible = open > 0.02
      const len = 0.35 + 0.65 * open
      tongue.scale.set(0.9, len, 0.38)
      tongue.position.copy(a.lipLowerInner).add(V(0, -0.2 * len, 0.09))
      tongue.rotation.set(0.35, 0, 0)
    }
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

type LensStyle = 'square' | 'oval' | 'cateye' | 'shield' | 'tinted' | 'aviator' | 'hearts'

// All sizes in rig units (1 = distance between the eye centres ≈ 63 mm),
// taken from real frames: a 55 mm sunglass lens is ≈ 0.87 wide, a bridge
// ≈ 18 mm ≈ 0.29, so the whole front spans a little over 2 — as wide as
// the face at the temples.
function lensShape(style: LensStyle): THREE.Shape {
  const s = new THREE.Shape()
  switch (style) {
    case 'square':
      // Gentle Monster-style: big, flat-topped, softly squared.
      s.moveTo(-0.4, 0.33)
      s.lineTo(0.38, 0.36)
      s.quadraticCurveTo(0.45, 0.36, 0.45, 0.27)
      s.lineTo(0.42, -0.2)
      s.quadraticCurveTo(0.39, -0.34, 0.24, -0.34)
      s.lineTo(-0.22, -0.33)
      s.quadraticCurveTo(-0.38, -0.32, -0.4, -0.18)
      s.lineTo(-0.43, 0.24)
      s.quadraticCurveTo(-0.44, 0.33, -0.4, 0.33)
      break
    case 'oval':
      // Slim, long, swept up toward the outer edge.
      s.moveTo(-0.4, 0.0)
      s.bezierCurveTo(-0.4, 0.15, -0.14, 0.18, 0.16, 0.17)
      s.bezierCurveTo(0.4, 0.16, 0.46, 0.11, 0.46, 0.03)
      s.bezierCurveTo(0.46, -0.11, 0.24, -0.16, 0.0, -0.16)
      s.bezierCurveTo(-0.28, -0.16, -0.4, -0.11, -0.4, 0.0)
      break
    case 'cateye':
      // Upswept outer corners, a flatter inner edge.
      s.moveTo(-0.4, 0.17)
      s.bezierCurveTo(-0.2, 0.25, 0.2, 0.26, 0.47, 0.34)
      s.bezierCurveTo(0.5, 0.2, 0.44, -0.05, 0.32, -0.2)
      s.bezierCurveTo(0.18, -0.33, -0.18, -0.33, -0.32, -0.2)
      s.bezierCurveTo(-0.42, -0.1, -0.44, 0.08, -0.4, 0.17)
      break
    case 'tinted':
      // Soft octagon, the 'kpop tinted' shape.
      s.moveTo(-0.24, 0.33)
      s.lineTo(0.24, 0.33)
      s.quadraticCurveTo(0.4, 0.31, 0.41, 0.14)
      s.lineTo(0.41, -0.1)
      s.quadraticCurveTo(0.4, -0.3, 0.22, -0.33)
      s.lineTo(-0.22, -0.33)
      s.quadraticCurveTo(-0.4, -0.3, -0.41, -0.1)
      s.lineTo(-0.41, 0.14)
      s.quadraticCurveTo(-0.4, 0.31, -0.24, 0.33)
      break
    case 'aviator':
      s.moveTo(-0.38, 0.22)
      s.bezierCurveTo(-0.16, 0.32, 0.26, 0.31, 0.41, 0.17)
      s.bezierCurveTo(0.5, 0.0, 0.42, -0.36, 0.1, -0.4)
      s.bezierCurveTo(-0.2, -0.43, -0.42, -0.18, -0.42, 0.04)
      s.bezierCurveTo(-0.42, 0.13, -0.41, 0.19, -0.38, 0.22)
      break
    case 'hearts':
      s.moveTo(0, -0.36)
      s.bezierCurveTo(-0.5, -0.03, -0.47, 0.37, -0.17, 0.35)
      s.bezierCurveTo(-0.06, 0.34, 0, 0.24, 0, 0.18)
      s.bezierCurveTo(0, 0.24, 0.06, 0.34, 0.17, 0.35)
      s.bezierCurveTo(0.47, 0.37, 0.5, -0.03, 0, -0.36)
      break
    case 'shield':
      // One wraparound visor across both eyes, notched for the nose.
      s.moveTo(-1.08, 0.18)
      s.bezierCurveTo(-0.7, 0.34, 0.7, 0.34, 1.08, 0.18)
      s.bezierCurveTo(1.16, 0.0, 1.06, -0.26, 0.82, -0.3)
      s.bezierCurveTo(0.5, -0.33, 0.2, -0.26, 0.12, -0.12)
      s.quadraticCurveTo(0, -0.02, -0.12, -0.12)
      s.bezierCurveTo(-0.2, -0.26, -0.5, -0.33, -0.82, -0.3)
      s.bezierCurveTo(-1.06, -0.26, -1.16, 0.0, -1.08, 0.18)
      break
  }
  return s
}

/** The lens outline pushed outward by a thickness that can vary around it
 *  (a real acetate rim is thicker along the top), as an even offset rather
 *  than a scale, so the rim width is what it says everywhere. */
function offsetOutline(pts: THREE.Vector2[], th: (p: THREE.Vector2) => number) {
  const n = pts.length
  let area = 0
  for (let i = 0; i < n; i++) area += pts[i].x * pts[(i + 1) % n].y - pts[(i + 1) % n].x * pts[i].y
  const sign = area > 0 ? 1 : -1
  return pts.map((p, i) => {
    const a = pts[(i - 1 + n) % n]
    const b = pts[(i + 1) % n]
    const t = new THREE.Vector2(b.x - a.x, b.y - a.y).normalize()
    const out = new THREE.Vector2(t.y * sign, -t.x * sign)
    return p.clone().addScaledVector(out, th(p))
  })
}

/** Lens/frame curvature: real fronts bow back toward the temples; a shield
 *  wraps hard around the face. */
const bow = (style: LensStyle) => (style === 'shield' ? 0.2 : 0.16)
const lensZ = (style: LensStyle, x: number, y: number) => (style === 'shield' ? -bow(style) * x * x - 0.06 * y * y : 0.01 - bow(style) * (x * x + y * y))

function curvedLens(shape: THREE.Shape, style: LensStyle) {
  const g = new THREE.ShapeGeometry(shape, 48)
  const p = g.getAttribute('position') as THREE.BufferAttribute
  g.computeBoundingBox()
  const bb = g.boundingBox!
  const uv = g.getAttribute('uv') as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) {
    p.setZ(i, lensZ(style, p.getX(i), p.getY(i)))
    // UVs across the lens's own box, so tint gradients run top → bottom.
    uv.setXY(i, (p.getX(i) - bb.min.x) / (bb.max.x - bb.min.x), (p.getY(i) - bb.min.y) / (bb.max.y - bb.min.y))
  }
  g.computeVertexNormals()
  return g
}

/** A thick acetate temple: a side profile (deep at the hinge, tapering,
 *  hooked down behind the ear), extruded to its thickness. */
function acetateTemple(depthFront: number, thickness: number) {
  const s = new THREE.Shape()
  const d = depthFront
  s.moveTo(0, d / 2)
  s.bezierCurveTo(0.4, d / 2, 0.7, d * 0.32, 0.98, d * 0.26)
  s.quadraticCurveTo(1.1, d * 0.2, 1.18, -0.07)
  s.quadraticCurveTo(1.19, -0.11, 1.15, -0.1)
  s.quadraticCurveTo(1.1, -0.02, 0.97, -d * 0.22)
  s.bezierCurveTo(0.7, -d * 0.28, 0.4, -d / 2, 0, -d / 2)
  s.lineTo(0, d / 2)
  const g = new THREE.ExtrudeGeometry(s, { depth: thickness, bevelEnabled: true, bevelThickness: thickness * 0.3, bevelSize: thickness * 0.3, bevelSegments: 3, curveSegments: 24 })
  g.rotateY(Math.PI / 2)
  g.translate(-thickness / 2, 0, 0)
  return g
}

let tortoiseTex: THREE.CanvasTexture | null = null
/** Tortoiseshell acetate: amber with soft dark-brown mottling. */
function tortoise() {
  if (tortoiseTex) return tortoiseTex
  const c = document.createElement('canvas')
  c.width = c.height = 256
  const ctx = c.getContext('2d')!
  ctx.fillStyle = '#8a4e1c'
  ctx.fillRect(0, 0, 256, 256)
  let seed = 5
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647
  for (let i = 0; i < 160; i++) {
    const x = rnd() * 256
    const y = rnd() * 256
    const r = 6 + rnd() * 26
    const g = ctx.createRadialGradient(x, y, 0, x, y, r)
    const dark = rnd() < 0.65
    g.addColorStop(0, dark ? 'rgba(40,18,6,0.85)' : 'rgba(214,150,70,0.6)')
    g.addColorStop(1, dark ? 'rgba(40,18,6,0)' : 'rgba(214,150,70,0)')
    ctx.fillStyle = g
    ctx.beginPath()
    ctx.ellipse(x, y, r, r * (0.5 + rnd()), rnd() * Math.PI, 0, Math.PI * 2)
    ctx.fill()
  }
  tortoiseTex = new THREE.CanvasTexture(c)
  tortoiseTex.colorSpace = THREE.SRGBColorSpace
  tortoiseTex.wrapS = tortoiseTex.wrapT = THREE.RepeatWrapping
  tortoiseTex.repeat.set(2.2, 2.2)
  return tortoiseTex
}

function gradientLens(top: string, bottom: string, opacity: number) {
  const c = document.createElement('canvas')
  c.width = 4
  c.height = 64
  const ctx = c.getContext('2d')!
  const g = ctx.createLinearGradient(0, 0, 0, 64)
  g.addColorStop(0, top)
  g.addColorStop(1, bottom)
  ctx.fillStyle = g
  ctx.fillRect(0, 0, 4, 64)
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  return physical({ map: tex, roughness: 0.03, transparent: true, opacity, clearcoat: 1, envMapIntensity: 1.3, side: THREE.DoubleSide, depthWrite: false })
}

interface Style {
  frame: 'acetate' | 'wire' | 'shield'
  frameMat: () => THREE.Material
  lensMat: () => THREE.Material
  rim?: (p: THREE.Vector2) => number
  depth?: number
  pins?: boolean
  temple?: [number, number]
}

const acetate = (o: THREE.MeshPhysicalMaterialParameters) => physical({ roughness: 0.16, clearcoat: 1, clearcoatRoughness: 0.05, envMapIntensity: 0.9, ...o })
const smoke = () => physical({ color: 0x2a2c2f, roughness: 0.03, transparent: true, opacity: 0.92, clearcoat: 1, envMapIntensity: 1.3, side: THREE.DoubleSide, depthWrite: false })

const STYLES: Record<LensStyle, Style> = {
  square: {
    frame: 'acetate',
    frameMat: () => acetate({ color: 0x0b0b0d }),
    lensMat: smoke,
    rim: (p) => 0.075 + 0.045 * THREE.MathUtils.smoothstep(p.y, 0.05, 0.3) + 0.02 * THREE.MathUtils.smoothstep(p.x, 0.25, 0.45),
    depth: 0.08,
    pins: true,
    temple: [0.11, 0.045],
  },
  oval: {
    frame: 'acetate',
    frameMat: () => acetate({ color: 0x0b0b0d }),
    lensMat: smoke,
    rim: (p) => 0.055 + 0.05 * THREE.MathUtils.smoothstep(p.x, 0.2, 0.46),
    depth: 0.065,
    temple: [0.07, 0.035],
  },
  cateye: {
    frame: 'acetate',
    frameMat: () => acetate({ color: 0xffffff, map: tortoise() }),
    lensMat: () => gradientLens('rgba(70,40,20,0.95)', 'rgba(190,140,90,0.55)', 0.92),
    rim: (p) => 0.06 + 0.05 * THREE.MathUtils.smoothstep(p.x, 0.25, 0.47) * THREE.MathUtils.smoothstep(p.y, 0.0, 0.3),
    depth: 0.07,
    temple: [0.09, 0.04],
  },
  tinted: {
    frame: 'wire',
    frameMat: () => new THREE.MeshStandardMaterial({ color: 0xd8b26a, metalness: 1, roughness: 0.16 }),
    lensMat: () => gradientLens('rgba(255,120,150,0.75)', 'rgba(255,200,210,0.25)', 0.75),
  },
  aviator: {
    frame: 'wire',
    frameMat: gold,
    lensMat: () => gradientLens('rgba(40,52,60,0.97)', 'rgba(120,135,140,0.6)', 0.9),
  },
  hearts: {
    frame: 'acetate',
    frameMat: () => acetate({ color: 0xff3d85, roughness: 0.22 }),
    lensMat: () => gradientLens('rgba(255,70,140,0.7)', 'rgba(255,170,205,0.4)', 0.75),
    depth: 0.07,
    temple: [0.07, 0.035],
  },
  shield: {
    frame: 'shield',
    frameMat: () => new THREE.MeshStandardMaterial({ color: 0xd9dde3, metalness: 1, roughness: 0.12 }),
    // Mirrored, iridescent visor: blue-violet sheen that shifts with angle.
    lensMat: () => {
      // Y2K oil-slick mirror: blue through violet and pink to a gold top edge.
      const c = document.createElement('canvas')
      c.width = 4
      c.height = 128
      const ctx = c.getContext('2d')!
      const g = ctx.createLinearGradient(0, 0, 0, 128)
      g.addColorStop(0, '#f2c14e')
      g.addColorStop(0.3, '#f15bb5')
      g.addColorStop(0.6, '#7b5cf0')
      g.addColorStop(1, '#2f6fe0')
      ctx.fillStyle = g
      ctx.fillRect(0, 0, 4, 128)
      const map = new THREE.CanvasTexture(c)
      map.colorSpace = THREE.SRGBColorSpace
      return physical({ map, metalness: 0.55, roughness: 0.06, iridescence: 0.8, iridescenceIOR: 1.5, clearcoat: 1, transparent: true, opacity: 0.92, envMapIntensity: 1.8, side: THREE.DoubleSide, depthWrite: false })
    },
  },
}

function eyewear(style: LensStyle): Model {
  const S = STYLES[style]
  const shape = lensShape(style)
  const pts = shape.getPoints(140)
  const frameMat = S.frameMat()
  const lensMat = S.lensMat()
  const silver = new THREE.MeshStandardMaterial({ color: 0xe8e8ec, metalness: 1, roughness: 0.15 })
  const glasses = new THREE.Group()
  const halfW = Math.max(...pts.map((p) => Math.abs(p.x)))
  let hinge: number
  let hingeY = 0.12

  if (S.frame === 'shield') {
    const lens = new THREE.Mesh(curvedLens(shape, style), lensMat)
    lens.renderOrder = 5
    glasses.add(lens)
    // A slim metal top bar following the visor's top edge.
    const top = pts.filter((p) => p.y > 0.12 && Math.abs(p.x) < halfW - 0.04).sort((a, b) => a.x - b.x)
    const curve = new THREE.CatmullRomCurve3(top.map((p) => V(p.x, p.y + 0.02, lensZ(style, p.x, p.y) + 0.01)))
    glasses.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 120, 0.018, 10, false), frameMat))
    hinge = halfW - 0.03
    hingeY = 0.12
    for (const side of [-1, 1]) {
      const z0 = lensZ(style, hinge, 0.12)
      glasses.add(new THREE.Mesh(taperTube([V(side * hinge, hingeY, z0), V(side * (hinge + 0.02), hingeY, z0 - 0.3), V(side * 1.04, hingeY, -0.95), V(side * 1.0, -0.02, -1.3)], 0.016, 0.013, 10, 48), frameMat))
    }
  } else {
    const cx = 0.15 + halfW
    const outerPts = S.frame === 'acetate' ? (style === 'hearts' ? pts.map((p) => new THREE.Vector2(p.x * 1.2, p.y * 1.24)) : offsetOutline(pts, S.rim ?? (() => 0.07))) : pts
    const depth = S.depth ?? 0.06
    const lensFor = (side: -1 | 1) => {
      const grp = new THREE.Group()
      const lens = new THREE.Mesh(curvedLens(shape, style), lensMat)
      lens.renderOrder = 5
      grp.add(lens)
      if (S.frame === 'wire') {
        const curve = new THREE.CatmullRomCurve3(pts.map((p) => V(p.x, p.y, lensZ(style, p.x, p.y))), true)
        grp.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 200, 0.017, 12, true), frameMat))
      } else {
        const outer = new THREE.Shape(outerPts)
        outer.holes.push(new THREE.Path(pts.slice().reverse()))
        const g = new THREE.ExtrudeGeometry(outer, { depth, bevelEnabled: true, bevelThickness: 0.024, bevelSize: 0.02, bevelSegments: 5, curveSegments: 64 })
        g.translate(0, 0, -depth + 0.01)
        // Bow the rim with the lens.
        const p = g.getAttribute('position') as THREE.BufferAttribute
        for (let i = 0; i < p.count; i++) p.setZ(i, p.getZ(i) + lensZ(style, p.getX(i), p.getY(i)) - 0.01)
        g.computeVertexNormals()
        grp.add(new THREE.Mesh(g, frameMat))
        if (S.pins) {
          const top = outerPts.reduce((a, b) => (b.x > 0.25 && b.y > a.y ? b : a), outerPts[0])
          for (const dy of [0, -0.06]) {
            const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.012, 16), silver)
            pin.rotation.x = Math.PI / 2
            pin.position.set(top.x - 0.06, top.y - 0.07 + dy, lensZ(style, top.x, top.y) + 0.045)
            grp.add(pin)
          }
        }
      }
      grp.position.x = side * cx
      grp.scale.x = side
      return grp
    }
    glasses.add(lensFor(-1), lensFor(1))
    const outerW = Math.max(...outerPts.map((p) => p.x))
    const innerW = Math.max(...outerPts.map((p) => -p.x))
    hinge = cx + (S.frame === 'wire' ? halfW : outerW - 0.03)
    hingeY = S.frame === 'wire' ? 0.12 : Math.max(...outerPts.map((p) => p.y)) - 0.1
    if (S.frame === 'wire') {
      const inner = cx - halfW
      glasses.add(new THREE.Mesh(taperTube([V(-inner, 0.12, 0), V(0, 0.18, 0.03), V(inner, 0.12, 0)], 0.016, 0.016, 10, 24), frameMat))
      if (style === 'aviator') glasses.add(new THREE.Mesh(taperTube([V(-inner - 0.04, 0.27, -0.01), V(0, 0.29, 0.0), V(inner + 0.04, 0.27, -0.01)], 0.014, 0.014, 10, 24), frameMat))
    } else {
      // Keyhole bridge: a moulded bar joining the two rims.
      const gap = cx - innerW + 0.04
      const by = style === 'oval' ? 0.04 : 0.13
      const b = new THREE.Shape()
      b.moveTo(-gap, by + 0.06)
      b.lineTo(gap, by + 0.06)
      b.lineTo(gap, by - 0.045)
      b.quadraticCurveTo(0, by + 0.0, -gap, by - 0.045)
      b.lineTo(-gap, by + 0.06)
      const bg = new THREE.ExtrudeGeometry(b, { depth, bevelEnabled: true, bevelThickness: 0.024, bevelSize: 0.014, bevelSegments: 4 })
      bg.translate(0, 0, -depth + 0.02)
      glasses.add(new THREE.Mesh(bg, frameMat))
    }
    for (const side of [-1, 1]) {
      if (S.frame === 'wire' || !S.temple) {
        glasses.add(new THREE.Mesh(taperTube([V(side * hinge, hingeY, -0.03), V(side * (hinge + 0.04), hingeY, -0.28), V(side * 1.06, hingeY, -0.95), V(side * 1.02, -0.02, -1.3)], 0.016, 0.013, 10, 48), frameMat))
      } else {
        const temple = new THREE.Mesh(acetateTemple(S.temple[0], S.temple[1]), frameMat)
        temple.position.set(side * hinge, hingeY, lensZ(style, outerW, 0) - 0.04)
        temple.rotation.y = side * -0.04
        glasses.add(temple)
        if (S.pins) {
          const plate = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.045, 0.09), silver)
          plate.position.set(side * (hinge + S.temple[1] * 0.7), hingeY, lensZ(style, outerW, 0) - 0.2)
          glasses.add(plate)
        }
      }
    }
  }

  shadowed(glasses)
  // Lenses don't cast hard shadows (they're glass) — just the frame does.
  glasses.traverse((o) => {
    if ((o as THREE.Mesh).material === lensMat) (o as THREE.Mesh).castShadow = false
  })
  const root = new THREE.Group()
  root.add(glasses)
  return {
    root,
    update(rig) {
      const a = anchors(rig)
      // Glasses ride on the nose: lenses centred a little below the eyes.
      const y = (a.eyeL.y + a.eyeR.y) / 2 - 0.05
      glasses.position.set(0, y, a.bridge.z + (style === 'shield' ? 0.2 : 0.13))
      glasses.rotation.set(-0.08, 0, 0)
    },
  }
}

// ---- Props --------------------------------------------------------------------

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
      grp.position.copy(a.top).add(V(0, 0.82, -0.62))
      grp.rotation.set(-0.32, 0, 0)
    },
  }
}

function devil(): Model {
  const mat = physical({ color: 0xc8141f, roughness: 0.28, clearcoat: 1, clearcoatRoughness: 0.1, sheen: 0.3, sheenColor: new THREE.Color(0xff6040) })
  const horn = (side: number) => {
    const m = new THREE.Mesh(taperTube([V(0, 0, 0), V(side * 0.05, 0.25, 0.03), V(side * 0.18, 0.5, 0.06), V(side * 0.38, 0.68, 0.02)], 0.12, 0.004, 24, 64), mat)
    m.castShadow = true
    return m
  }
  const L = horn(-1)
  const R = horn(1)
  const root = new THREE.Group()
  root.add(L, R)
  return {
    root,
    update(rig) {
      const a = anchors(rig)
      L.position.copy(a.top).lerp(a.templeL, 0.48).add(V(0, 0.12, -0.12))
      R.position.copy(a.top).lerp(a.templeR, 0.48).add(V(0, 0.12, -0.12))
      L.rotation.set(-0.15, 0, 0.12)
      R.rotation.set(-0.15, 0, -0.12)
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
      grp.position.copy(a.top).add(V(0, 1.3 + bob, -0.55))
      grp.rotation.set(Math.PI / 2 - 0.42, 0, 0)
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
      orbit.position.copy(a.top).add(V(0, 0.75, -1.1))
      orbit.rotation.set(0.18, 0, 0)
      list.forEach((m, i) => {
        const ang = t * 0.9 + (i / list.length) * Math.PI * 2
        m.position.set(Math.sin(ang) * 1.5, Math.sin(ang * 2 + i) * 0.06, Math.cos(ang) * 1.5)
        m.rotation.set(0, ang * 2.5, Math.sin(t + i) * 0.3)
        m.scale.setScalar(0.85 + 0.25 * Math.sin(i * 1.7))
      })
    },
  }
}

export function buildModel(id: string): Model {
  switch (id) {
    case 'kitty': return kitty()
    case 'fox': return fox()
    case 'puppy': return puppy()
    case 'bunny': return bunny()
    case 'bear': return bear()
    case 'aviator': return eyewear('aviator')
    case 'square': return eyewear('square')
    case 'oval': return eyewear('oval')
    case 'cateye': return eyewear('cateye')
    case 'shield': return eyewear('shield')
    case 'tinted': return eyewear('tinted')
    case 'hearts': return eyewear('hearts')
    case 'crown': return crown()
    case 'devil': return devil()
    case 'angel': return angel()
    case 'stars': return stars()
    default: return { root: new THREE.Group() }
  }
}
