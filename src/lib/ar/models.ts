import * as THREE from 'three'
import { earGeometry, furMesh, type FurLook } from './fur'
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
const gloss = (color: THREE.ColorRepresentation, roughness = 0.3) => physical({ color, roughness, clearcoat: 1, clearcoatRoughness: 0.12 })
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
  look: FurLook
  geo: THREE.BufferGeometry
  /** Where it sits: t along forehead-top→temple, then lifted/pushed. */
  t: number
  lift: THREE.Vector3
  rot: THREE.Euler
  scale: number
}

function earPair(spec: EarSpec, shells: number) {
  const make = () => {
    const pivot = new THREE.Group()
    pivot.add(furMesh(spec.geo, spec.look, shells))
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

const SHELLS_STILL = 40
const SHELLS_LIVE = 18

function animalModel(spec: EarSpec, extras: (root: THREE.Group) => (a: Anchors, rig: Rig, t: number) => void): Model {
  const root = new THREE.Group()
  // Two fur LODs: dense for stills, lighter for the 30fps viewfinder.
  const still = earPair(spec, SHELLS_STILL)
  const live = earPair(spec, SHELLS_LIVE)
  root.add(still.root, live.root)
  const extra = extras(root)
  return {
    root,
    update(rig, t) {
      const a = anchors(rig)
      still.root.visible = !rig.live
      live.root.visible = rig.live
      ;(rig.live ? live : still).place(a)
      extra(a, rig, t)
    },
  }
}

const catEar = earGeometry(
  0.9,
  // Rounded triangle: full at the base, softly domed tip.
  (v) => 0.45 * Math.pow(1 - v, 0.72) * (1 - 0.15 * v),
  (v) => 0.26 * Math.pow(1 - v, 0.75) + 0.02,
  (v) => [0, -0.16 * v * v],
  0.6,
)
const bunnyEar = earGeometry(
  1.75,
  (v) => 0.24 * Math.pow(Math.sin(Math.PI * (0.16 + 0.84 * v)), 0.55),
  (v) => 0.11 * Math.pow(Math.sin(Math.PI * (0.16 + 0.84 * v)), 0.5) + 0.015,
  (v) => [0.02 * v, -0.22 * v * v],
  0.7,
)
const dogEar = earGeometry(
  1.3,
  (v) => 0.36 * Math.pow(Math.sin(Math.PI * (0.1 + 0.9 * v)), 0.55),
  // A curled leaf rather than a flat flap: deep crescent, so the drape
  // catches light across its width.
  (v) => 0.13 * Math.pow(Math.sin(Math.PI * (0.1 + 0.9 * v)), 0.4) + 0.02,
  (v) => [0.18 * v * v, 0.1 * v],
  0.85,
)
const bearEar = earGeometry(
  0.6,
  (v) => 0.36 * Math.sqrt(Math.max(0, 1 - Math.pow((v - 0.22) / 0.78, 2))),
  (v) => 0.2 * Math.sqrt(Math.max(0, 1 - Math.pow((v - 0.22) / 0.78, 2))) + 0.02,
  () => [0, 0],
  0.6,
)

function nose(mat: THREE.Material, sx: number, sy: number, sz: number, tri = 0.5) {
  // A soft inverted-triangle nose: narrower toward the bottom.
  return blob(sx, sy, sz, mat, (v) => {
    v.x *= 1 + v.y * tri
  })
}

function kitty(): Model {
  const look: FurLook = { root: '#d8c6b4', tip: '#fffaf3', skin: '#ffbccb', length: 0.12, density: 38, comb: [0, 0.6, 0], innerBare: 0.55 }
  return animalModel({ look, geo: catEar, t: 0.6, lift: V(0.05, 0.42, -0.3), rot: new THREE.Euler(-0.18, -0.35, -0.38), scale: 0.95 }, (root) => {
    const n = shadowed(nose(physical({ color: 0xf28aa6, roughness: 0.32, clearcoat: 0.7, sheen: 0.3 }), 0.085, 0.06, 0.055))
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

function puppy(): Model {
  const look: FurLook = { root: '#6a4126', tip: '#c08a5a', skin: '#5b3a2a', length: 0.13, density: 32, comb: [0, 0.9, 0.05], innerBare: 0.2 }
  return animalModel({ look, geo: dogEar, t: 0.95, lift: V(0.18, 0.28, -0.42), rot: new THREE.Euler(0.15, -0.5, -(Math.PI - 0.5)), scale: 1 }, (root) => {
    const black = physical({ color: 0x141416, roughness: 0.28, clearcoat: 1, clearcoatRoughness: 0.15 })
    const n = shadowed(nose(black, 0.2, 0.13, 0.13, 0.35))
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
  const look: FurLook = { root: '#e4dfe6', tip: '#ffffff', skin: '#ffc2d2', length: 0.09, density: 46, comb: [0, 0.5, 0], innerBare: 0.75 }
  return animalModel({ look, geo: bunnyEar, t: 0.38, lift: V(0.02, 0.5, -0.25), rot: new THREE.Euler(-0.22, -0.2, -0.14), scale: 0.95 }, (root) => {
    const n = shadowed(nose(physical({ color: 0xff8fab, roughness: 0.35, clearcoat: 0.6 }), 0.055, 0.04, 0.035))
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
  const look: FurLook = { root: '#4a2c19', tip: '#8f6141', skin: '#d2a882', length: 0.13, density: 34, comb: [0, 0.3, 0.1], innerBare: 0.4 }
  return animalModel({ look, geo: bearEar, t: 0.78, lift: V(0.08, 0.36, -0.35), rot: new THREE.Euler(-0.1, -0.25, -0.32), scale: 1 }, (root) => {
    const n = shadowed(nose(physical({ color: 0x2a1a10, roughness: 0.3, clearcoat: 1 }), 0.12, 0.08, 0.08, 0.4))
    root.add(n)
    return (a) => n.position.copy(a.noseTip).add(V(0, 0.03, 0.05))
  })
}

// ---- Eyewear ------------------------------------------------------------------

type LensStyle = 'aviator' | 'wayfarer' | 'round' | 'hearts'

function lensShape(style: LensStyle): THREE.Shape {
  const s = new THREE.Shape()
  switch (style) {
    case 'aviator':
      // Teardrop: flat-ish top, deep drop toward the outer-lower corner.
      s.moveTo(-0.27, 0.13)
      s.bezierCurveTo(-0.12, 0.2, 0.18, 0.19, 0.29, 0.1)
      s.bezierCurveTo(0.36, 0.0, 0.3, -0.24, 0.08, -0.27)
      s.bezierCurveTo(-0.14, -0.29, -0.3, -0.12, -0.3, 0.02)
      s.bezierCurveTo(-0.3, 0.08, -0.29, 0.11, -0.27, 0.13)
      break
    case 'wayfarer':
      s.moveTo(-0.29, 0.15)
      s.lineTo(0.31, 0.17)
      s.quadraticCurveTo(0.33, 0.17, 0.32, 0.12)
      s.quadraticCurveTo(0.29, -0.15, 0.12, -0.19)
      s.lineTo(-0.16, -0.18)
      s.quadraticCurveTo(-0.29, -0.14, -0.31, 0.1)
      s.quadraticCurveTo(-0.31, 0.15, -0.29, 0.15)
      break
    case 'round':
      s.absarc(0, 0, 0.25, 0, Math.PI * 2, false)
      break
    case 'hearts':
      s.moveTo(0, -0.26)
      s.bezierCurveTo(-0.36, -0.02, -0.34, 0.27, -0.12, 0.25)
      s.bezierCurveTo(-0.04, 0.24, 0, 0.17, 0, 0.13)
      s.bezierCurveTo(0, 0.17, 0.04, 0.24, 0.12, 0.25)
      s.bezierCurveTo(0.34, 0.27, 0.36, -0.02, 0, -0.26)
      break
  }
  return s
}

/** Lens surfaces bow slightly outward, like real ground lenses. */
function curvedLens(shape: THREE.Shape) {
  const g = new THREE.ShapeGeometry(shape, 32)
  const p = g.getAttribute('position') as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) p.setZ(i, -0.35 * (p.getX(i) ** 2 + p.getY(i) ** 2))
  g.computeVertexNormals()
  return g
}

function eyewear(style: LensStyle): Model {
  const shape = lensShape(style)
  const pts = shape.getPoints(96)
  const wire = style === 'aviator' || style === 'round'
  const frameMat =
    style === 'aviator' ? gold() : style === 'round' ? new THREE.MeshStandardMaterial({ color: 0xc9a25b, metalness: 1, roughness: 0.18 }) : style === 'hearts' ? gloss(0xff3d85, 0.22) : gloss(0x101012, 0.18)
  const lensMat =
    style === 'aviator'
      ? physical({ color: 0x3c4a52, roughness: 0.04, metalness: 0.2, transparent: true, opacity: 0.86, clearcoat: 1, envMapIntensity: 1.8, side: THREE.DoubleSide, depthWrite: false, vertexColors: true })
      : style === 'wayfarer'
        ? physical({ color: 0x1c2a22, roughness: 0.03, transparent: true, opacity: 0.9, clearcoat: 1, envMapIntensity: 1.6, side: THREE.DoubleSide, depthWrite: false })
        : style === 'round'
          ? physical({ color: 0xffffff, roughness: 0.02, transparent: true, opacity: 0.14, clearcoat: 1, envMapIntensity: 2.2, side: THREE.DoubleSide, depthWrite: false })
          : physical({ color: 0xff5c9e, roughness: 0.04, transparent: true, opacity: 0.55, clearcoat: 1, envMapIntensity: 1.6, side: THREE.DoubleSide, depthWrite: false })

  const lensFor = (side: -1 | 1) => {
    const grp = new THREE.Group()
    const lensGeo = curvedLens(shape)
    if (style === 'aviator') {
      // Gradient tint: darker at the top, as on real aviators.
      const p = lensGeo.getAttribute('position') as THREE.BufferAttribute
      const col: number[] = []
      for (let i = 0; i < p.count; i++) {
        const k = THREE.MathUtils.smoothstep(p.getY(i), -0.25, 0.15)
        col.push(0.55 + 0.45 * (1 - k), 0.5 + 0.4 * (1 - k), 0.45 + 0.35 * (1 - k))
      }
      lensGeo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3))
    }
    const lens = new THREE.Mesh(lensGeo, lensMat)
    lens.renderOrder = 5
    grp.add(lens)
    if (wire) {
      const curve = new THREE.CatmullRomCurve3(pts.map((p) => V(p.x, p.y, -0.35 * (p.x * p.x + p.y * p.y))), true)
      grp.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 160, style === 'round' ? 0.013 : 0.016, 12, true), frameMat))
    } else {
      const outer = new THREE.Shape(pts.map((p) => new THREE.Vector2(p.x * 1.22, p.y * 1.3 + (style === 'wayfarer' ? 0.02 : 0))))
      outer.holes.push(new THREE.Path(pts.slice().reverse()))
      const g = new THREE.ExtrudeGeometry(outer, { depth: 0.05, bevelEnabled: true, bevelThickness: 0.018, bevelSize: 0.014, bevelSegments: 4, curveSegments: 48 })
      g.translate(0, 0, -0.04)
      grp.add(new THREE.Mesh(g, frameMat))
    }
    grp.scale.x = side
    return grp
  }

  const glasses = new THREE.Group()
  const L = lensFor(-1)
  const R = lensFor(1)
  glasses.add(L, R)
  const halfW = Math.max(...pts.map((p) => Math.abs(p.x)))
  const cx = 0.52
  L.position.x = -cx
  R.position.x = cx
  const armR = wire ? 0.014 : 0.026
  // Bridge(s) and temple arms.
  const bridgeY = style === 'round' ? 0.05 : 0.1
  const inner = cx - halfW * (wire ? 1 : 1.18)
  glasses.add(new THREE.Mesh(taperTube([V(-inner, bridgeY, 0), V(0, bridgeY + 0.06, 0.03), V(inner, bridgeY, 0)], armR, armR, 10, 24), frameMat))
  if (style === 'aviator') glasses.add(new THREE.Mesh(taperTube([V(-inner - 0.03, 0.17, 0), V(0, 0.19, 0.01), V(inner + 0.03, 0.17, 0)], 0.012, 0.012, 10, 24), frameMat))
  const ear = cx + halfW * (wire ? 1 : 1.22)
  for (const side of [-1, 1]) {
    const arm = taperTube([V(side * ear, 0.1, -0.02), V(side * (ear + 0.06), 0.1, -0.25), V(side * 1.02, 0.11, -0.95), V(side * 1.0, -0.02, -1.3)], armR, armR * 0.8, 10, 48)
    glasses.add(new THREE.Mesh(arm, frameMat))
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
      const y = (a.eyeL.y + a.eyeR.y) / 2 + 0.01
      glasses.position.set(0, y, a.bridge.z + 0.1)
      glasses.rotation.set(-0.08, 0, 0)
      const s = 1.12 * Math.max(0.9, Math.min(1.15, (a.eyeR.x - a.eyeL.x) / 1.0))
      glasses.scale.setScalar(s)
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
      grp.position.copy(a.top).add(V(0, 0.42, -0.62))
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
      grp.position.copy(a.top).add(V(0, 0.82 + bob, -0.55))
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
      orbit.position.copy(a.top).add(V(0, 0.15, -1.1))
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
    case 'puppy': return puppy()
    case 'bunny': return bunny()
    case 'bear': return bear()
    case 'aviator': return eyewear('aviator')
    case 'wayfarer': return eyewear('wayfarer')
    case 'round': return eyewear('round')
    case 'hearts': return eyewear('hearts')
    case 'crown': return crown()
    case 'devil': return devil()
    case 'angel': return angel()
    case 'stars': return stars()
    default: return { root: new THREE.Group() }
  }
}
