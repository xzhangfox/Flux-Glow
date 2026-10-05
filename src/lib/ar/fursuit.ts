import * as THREE from 'three'
import { plushCard, smooth, type PlushSpec } from './plush'
import type { Model, Rig } from './scene'

// Full fursuit heads that replace the wearer's head entirely. Sculpted like
// a real fursuit head (or a character sculpt), not a deformed ball: the head
// is a smooth union of anatomical forms — rounded cranium, lower head and
// jaw, cheeks, a long tapering snout with a flat top, a separate lower jaw
// that leaves a mouth crease under it, brow ridges over eye sockets — as a
// signed distance field, turned into a mesh by casting rays out from inside
// it. Fur colour follows that 3D shape (orange snout top, white lower jaw
// and cheeks), occlusion comes from the same field, and on top: glossy
// eyeballs set in the sockets, a broad leather nose, tall plush ears, a
// forehead tuft, and fluffy tufts round the silhouette and cheeks.
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
  iris: [string, string]
  tuft: string
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
  const k0 = Math.hypot(x / rx, y / ry, z / rz)
  const k1 = Math.hypot(x / (rx * rx), y / (ry * ry), z / (rz * rz))
  return k1 > 0 ? (k0 * (k0 - 1)) / k1 : -Math.min(rx, ry, rz)
}

/** A cone with rounded ends between a and b, squashed vertically by
 *  `flat` (< 1 = flatter on top and underneath). */
function cone(p: V3, a: V3, b: V3, ra: number, rb: number, flat: number) {
  const ab = b.clone().sub(a)
  const t = THREE.MathUtils.clamp(p.clone().sub(a).dot(ab) / ab.lengthSq(), 0, 1)
  const c = a.clone().addScaledVector(ab, t)
  const dx = p.x - c.x
  const dy = (p.y - c.y) / flat
  const dz = p.z - c.z
  return Math.hypot(dx, dy, dz) - THREE.MathUtils.lerp(ra, rb, t)
}

const SNOUT_A = V(0, -0.22, 0.3)
const SNOUT_B = V(0, -0.52, 1.72)
const JAW_A = V(0, -0.82, 0.25)
const JAW_B = V(0, -0.93, 1.32)
const EYE_X = 0.56
const EYE_Y = 0.16
const EYE_Z = 0.5
const EYE_R = 0.37

function sdf(p: V3) {
  // Cranium, lower head, cheeks.
  let d = ellipsoid(p, 0, 0.35, -0.95, 1.85, 2.0, 1.75)
  d = smin(d, ellipsoid(p, 0, -1.0, -0.5, 1.42, 1.1, 1.35), 0.5)
  for (const s of [-1, 1]) d = smin(d, ellipsoid(p, s * 1.0, -0.72, -0.15, 0.76, 0.7, 0.78), 0.4)
  // Snout, then the lower jaw beneath it: blended tightly, so a mouth
  // crease runs between them.
  d = smin(d, cone(p, SNOUT_A, SNOUT_B, 0.66, 0.31, 0.8), 0.45)
  d = smin(d, cone(p, JAW_A, JAW_B, 0.5, 0.24, 0.72), 0.12)
  // Brow ridges, and the sockets the eyeballs sit in.
  for (const s of [-1, 1]) d = smin(d, ellipsoid(p, s * 0.54, 0.52, 0.62, 0.44, 0.18, 0.3), 0.25)
  for (const s of [-1, 1]) d = smax(d, -(p.clone().sub(V(s * EYE_X, EYE_Y, EYE_Z + 0.2)).length() - EYE_R * 1.02), 0.12)
  return d
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
const ORIGIN = V(0, -0.25, -0.55)

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

function surfaceAlong(d: V3) {
  let lo = 0
  let hi = 0.08
  while (hi < 6 && sdf(ORIGIN.clone().addScaledVector(d, hi)) < 0) {
    lo = hi
    hi += 0.08
  }
  for (let i = 0; i < 14; i++) {
    const m = (lo + hi) / 2
    if (sdf(ORIGIN.clone().addScaledVector(d, m)) < 0) lo = m
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
  const src = SNOUT_B.clone().sub(ORIGIN).normalize()
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
    const len = (3 + rnd() * 6) * (W / 1024)
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

/** A neutral (near-white) tuft: strands fanning from the base to a soft
 *  point; tinted per card through the material colour. */
function tuftTexture() {
  const W = 256
  const H = 320
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const ctx = c.getContext('2d')!
  ctx.lineCap = 'round'
  for (let i = 0; i < 2600; i++) {
    const k = i / 2600
    const x0 = W / 2 + (rnd() - 0.5) * W * 0.62
    const y0 = H - 10 - rnd() * H * 0.3
    const ang = -Math.PI / 2 + ((x0 - W / 2) / W) * 1.6 + (rnd() - 0.5) * 0.5
    const len = H * (0.2 + rnd() * 0.55) * Math.sqrt(Math.max(0, 1 - ((x0 - W / 2) / (W * 0.31)) ** 2))
    const g = Math.round(150 + 105 * (0.4 + 0.6 * k) * (0.75 + 0.25 * rnd()))
    ctx.strokeStyle = `rgb(${g},${g},${g})`
    ctx.globalAlpha = 0.18 + rnd() * 0.4
    ctx.lineWidth = 0.8 + rnd() * 1.6
    const ex = x0 + Math.cos(ang) * len
    const ey = y0 + Math.sin(ang) * len
    const bend = (rnd() - 0.5) * len * 0.3
    ctx.beginPath()
    ctx.moveTo(x0, y0)
    ctx.quadraticCurveTo((x0 + ex) / 2 + bend, (y0 + ey) / 2, ex, ey)
    ctx.stroke()
  }
  // Roots fade in, so the tufts grow out of the head's fur without a line.
  ctx.globalAlpha = 1
  ctx.globalCompositeOperation = 'destination-in'
  const fade = ctx.createLinearGradient(0, H, 0, H * 0.55)
  fade.addColorStop(0, 'rgba(0,0,0,0)')
  fade.addColorStop(1, 'rgba(0,0,0,1)')
  ctx.fillStyle = fade
  ctx.fillRect(0, 0, W, H)
  ctx.globalCompositeOperation = 'source-over'
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

/** Eyeball texture (sphere UVs; the iris faces +z): a large dark iris
 *  glowing toward the bottom, pupil, catch-lights; a sliver of white. */
function eyeTexture(iris: [string, string]) {
  const W = 512
  const H = 256
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const ctx = c.getContext('2d')!
  const img = ctx.createImageData(W, H)
  const dark = C(iris[0])
  const glow = C(iris[1])
  const white = C('#f3f0ea')
  const col = new THREE.Color()
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const phi = (x / W) * Math.PI * 2
      const th = (y / H) * Math.PI
      // three.js SphereGeometry vertex direction for this uv.
      const dx = -Math.cos(phi) * Math.sin(th)
      const dy = Math.cos(th)
      const dz = Math.sin(phi) * Math.sin(th)
      const a = Math.acos(THREE.MathUtils.clamp(dz, -1, 1))
      if (a < 0.34) col.set('#07080c')
      else if (a < 0.95) {
        col.copy(dark).lerp(glow, ss(-dy, -0.3, 0.55))
        col.multiplyScalar(1 - 0.55 * ss(a, 0.82, 0.95))
      } else col.copy(white)
      // Catch-lights: a big one upper-left, a small one lower-right.
      const h1 = Math.hypot(dx + 0.28, dy - 0.3)
      const h2 = Math.hypot(dx - 0.2, dy + 0.2)
      if (dz > 0 && (h1 < 0.13 || h2 < 0.06)) col.set('#ffffff')
      const k = (y * W + x) * 4
      img.data[k] = col.r * 255
      img.data[k + 1] = col.g * 255
      img.data[k + 2] = col.b * 255
      img.data[k + 3] = 255
    }
  ctx.putImageData(img, 0, 0)
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

function fursuit(spec: SuitSpec): Model {
  seed = 11
  const root = new THREE.Group()

  // ---- Head mesh from the field ----
  const g = new THREE.SphereGeometry(1, 220, 160)
  g.rotateY(-Math.PI / 2)
  const pos = g.getAttribute('position') as THREE.BufferAttribute
  const uv = g.getAttribute('uv') as THREE.BufferAttribute
  const col = new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3)
  g.setAttribute('color', col)
  const d = V(0, 0, 0)
  for (let i = 0; i < pos.count; i++) {
    d.fromBufferAttribute(pos, i).normalize()
    const p = surfaceAlong(d)
    pos.setXYZ(i, p.x, p.y, p.z)
    const [u, w] = uvOf(d)
    // The seam sits at the back, out of sight; keep the UV's own u there.
    uv.setXY(i, Math.abs(u - uv.getX(i)) > 0.5 ? uv.getX(i) : u, 1 - w)
    const c = spec.fur(p).multiplyScalar(occlusion(p, normalAt(p)) / FUR_GREY)
    col.setXYZ(i, c.r, c.g, c.b)
  }
  g.computeVertexNormals()
  const furTex = furTexture(1024, 512, 120000)
  const furMat = new THREE.MeshPhysicalMaterial({ vertexColors: true, map: furTex, bumpMap: furTex, bumpScale: 0.3, roughness: 0.95, sheen: 0.7, sheenRoughness: 0.6, sheenColor: new THREE.Color(0x6a5a4a) })
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

  // ---- Nose: broad, glossy leather, wider on top ----
  const noseGeo = new THREE.SphereGeometry(1, 40, 28)
  {
    const np = noseGeo.getAttribute('position') as THREE.BufferAttribute
    for (let i = 0; i < np.count; i++) np.setX(i, np.getX(i) * (1 + 0.28 * np.getY(i)))
    noseGeo.computeVertexNormals()
  }
  const nose = new THREE.Mesh(noseGeo, new THREE.MeshPhysicalMaterial({ color: 0x141011, roughness: 0.38, clearcoat: 0.9, clearcoatRoughness: 0.2 }))
  nose.scale.set(0.3, 0.19, 0.2)
  nose.position.copy(hitFront(0, -0.48)).add(V(0, 0.02, -0.07))
  nose.rotation.x = 0.25
  nose.castShadow = true
  root.add(nose)

  // ---- Eyes: glossy eyeballs in the sockets, looking ahead ----
  const eyeMat = new THREE.MeshPhysicalMaterial({ map: eyeTexture(spec.iris), roughness: 0.12, clearcoat: 1, clearcoatRoughness: 0.03 })
  const eyeGeo = new THREE.SphereGeometry(EYE_R, 48, 32)
  for (const side of [-1, 1]) {
    const e = new THREE.Mesh(eyeGeo, eyeMat)
    e.position.set(side * EYE_X, EYE_Y, EYE_Z)
    e.rotation.set(0.05, side * 0.1, 0)
    root.add(e)
  }

  // ---- Ears: tall plush ears on the top corners of the head ----
  const earCard = plushCard(spec.ear)
  for (const side of [-1, 1]) {
    const ear = earCard.clone()
    ear.position.set(side * 0.95, 1.95, -0.85)
    ear.rotation.set(-0.12, side * -0.15, side * -0.3)
    ear.scale.set(side, 1, 1)
    root.add(ear)
  }

  // ---- Tufts: forehead tuft, silhouette fluff, cheek ruffs ----
  const tuft = tuftTexture()
  const tuftGeo = new THREE.PlaneGeometry(1, 1)
  tuftGeo.translate(0, 0.5, 0)
  const tuftMat = (color: THREE.Color) => new THREE.MeshStandardMaterial({ map: tuft, color, transparent: true, alphaTest: 0.04, depthWrite: false, side: THREE.DoubleSide, roughness: 1 })
  for (let i = 0; i < 6; i++) {
    const x = (i / 5 - 0.5) * 0.55
    const m = new THREE.Mesh(tuftGeo, tuftMat(C(spec.tuft)))
    m.position.copy(hitFront(x, 1.15)).add(V(0, 0, -0.08))
    m.rotation.set(-0.5, 0, -x * 1.4 + (rnd() - 0.5) * 0.3)
    m.scale.set(0.5, 0.45 + rnd() * 0.2, 1)
    m.renderOrder = 3
    root.add(m)
  }

  interface Tuft { mesh: THREE.Mesh; a: number; inset: number; size: number; ruff: boolean }
  const tufts: Tuft[] = []
  const addTuft = (a: number, inset: number, size: number, ruff: boolean) => {
    const mesh = new THREE.Mesh(tuftGeo, tuftMat(new THREE.Color()))
    mesh.renderOrder = 3
    root.add(mesh)
    tufts.push({ mesh, a, inset, size, ruff })
  }
  for (let i = 0; i < 70; i++) addTuft((i / 70) * Math.PI * 2 + rnd() * 0.05, 0.92, 0.28 + rnd() * 0.18, false)
  // Cheek ruffs: white fluff sweeping out from the cheeks, as on the sculpt.
  for (const side of [-1, 1])
    for (let i = 0; i < 14; i++) addTuft(Math.PI / 2 - side * (Math.PI / 2 + 0.25 + (i / 13) * 0.55), 0.86, 0.6 + rnd() * 0.35, true)

  const SIL_C = V(0, 0.1, -0.85)
  const SIL_R = V(2.0, 2.25, 1.8)
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
      const m = V(toCam.x / SIL_R.x, toCam.y / SIL_R.y, toCam.z / SIL_R.z).normalize()
      e1.set(0, 1, 0).addScaledVector(m, -m.y)
      if (e1.lengthSq() < 1e-4) e1.set(1, 0, 0)
      e1.normalize()
      e2.crossVectors(m, e1).normalize()
      for (const t of tufts) {
        const s = e1.clone().multiplyScalar(Math.sin(t.a)).addScaledVector(e2, Math.cos(t.a))
        const p = V(s.x * SIL_R.x, s.y * SIL_R.y, s.z * SIL_R.z)
        const outward = p.clone().addScaledVector(toCam, -p.dot(toCam)).normalize()
        if (t.ruff) outward.add(V(0, -0.35, 0)).normalize()
        // Rooted inside the rim and brought forward to the visible surface,
        // so they soften the edge while the snout can still pass in front.
        t.mesh.position.copy(SIL_C).addScaledVector(p, t.inset).addScaledVector(toCam, 0.8)
        const x = V(0, 0, 0).crossVectors(outward, toCam).normalize()
        const z = V(0, 0, 0).crossVectors(x, outward).normalize()
        m4.makeBasis(x, outward, z)
        t.mesh.quaternion.setFromRotationMatrix(m4)
        t.mesh.scale.set(t.size * 1.1, t.size, 1)
        ;(t.mesh.material as THREE.MeshStandardMaterial).color.copy(spec.fur(t.mesh.position))
      }
    },
  }
}

// ---- Designs ----------------------------------------------------------------

/** 0 above the mouth line, 1 below it — the line rises toward the cheeks. */
function lowerFace(p: V3) {
  const ax = Math.abs(p.x)
  const line = -0.86 + 0.42 * ss(ax, 0.35, 1.15)
  const back = ss(p.z, -1.3, -0.5)
  return Math.max(ss(line - p.y, -0.05, 0.05) * back, ss(-p.y, 1.65, 1.85))
}

function foxFur(p: V3) {
  const orange = C('#e88634').lerp(C('#f6a85c'), ss(p.y, -0.6, 0.6) * 0.3 * ss(p.z, 0, 1))
  return orange.lerp(C('#f8f4ee'), lowerFace(p))
}

function huskyFur(p: V3) {
  const grey = C('#5a5f69').lerp(C('#3b3f47'), ss(p.y, 0.6, 2) * 0.6)
  const ax = Math.abs(p.x)
  const front = ss(p.z, -0.2, 0.4)
  // White face below the brows, except a grey stripe down the nose bridge;
  // white eyebrow dots.
  let w = ss(0.4 - p.y, -0.05, 0.08) * front * (1 - ss(0.26 - ax, -0.04, 0.04) * ss(p.y, -0.5, -0.35))
  w = Math.max(w, lowerFace(p), front * Math.exp(-(((ax - 0.5) / 0.1) ** 2) - ((p.y - 0.78) / 0.07) ** 2))
  return grey.lerp(C('#f5f5f3'), THREE.MathUtils.clamp(w, 0, 1))
}

const EAR_OUTLINE = smooth([[60, 890], [70, 520], [150, 230], [262, 20], [372, 230], [452, 520], [462, 890], [262, 900]])
const EAR_INNER = smooth([[150, 860], [160, 560], [222, 300], [262, 170], [302, 300], [362, 560], [372, 860], [262, 872]])

export function foxHead(): Model {
  return fursuit({
    fur: foxFur,
    ear: {
      w: 512, h: 900, cardW: 1.75,
      outline: EAR_OUTLINE, inner: EAR_INNER, tip: [262, 0],
      root: '#b0601f', mid: '#e88634', light: '#ffc07a',
      earTip: ['#3a1d10', 380],
      skin: '#e98f48', plush: '#f6b277', tuft: '#fff2e2',
      strand: [16, 40],
    },
    iris: ['#0b1b38', '#2f6fb4'],
    tuft: '#e47f2e',
  })
}

export function huskyHead(): Model {
  return fursuit({
    fur: huskyFur,
    ear: {
      w: 512, h: 900, cardW: 1.65,
      outline: EAR_OUTLINE, inner: EAR_INNER, tip: [262, 0],
      root: '#2a2e35', mid: '#565b65', light: '#8e949d',
      skin: '#d8d4d2', plush: '#f4f2f0', tuft: '#ffffff',
      strand: [16, 40],
    },
    iris: ['#0b2a4a', '#5ab8f0'],
    tuft: '#4d525b',
  })
}
