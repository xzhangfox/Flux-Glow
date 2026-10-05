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
}

// ---- The sculpt (signed distance field) ---------------------------------------

function smin(a: number, b: number, k: number) {
  const h = Math.max(k - Math.abs(a - b), 0) / k
  return Math.min(a, b) - h * h * k * 0.25
}

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

const MUZZLE_A = V(0, -0.36, 0.3)
const MUZZLE_B = V(0, -0.56, 0.95)
const EYE_X = 0.7
const EYE_Y = -0.08

function sdf(p: V3) {
  // Broad cranium, full cheeks, and a narrower lower face to a small chin:
  // the V face.
  let d = ellipsoid(p, 0, 0.45, -0.95, 1.8, 1.95, 1.7)
  for (const s of [-1, 1]) d = smin(d, ellipsoid(p, s * 0.95, -0.35, -0.25, 0.95, 0.82, 0.88), 0.5)
  d = smin(d, ellipsoid(p, 0, -1.05, -0.38, 1.0, 1.05, 1.15), 0.55)
  // Short, soft muzzle with a little chin under it.
  d = smin(d, cone(p, MUZZLE_A, MUZZLE_B, 0.5, 0.22, 0.85), 0.35)
  d = smin(d, ellipsoid(p, 0, -0.86, 0.55, 0.3, 0.17, 0.3), 0.18)
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

function mouthTexture() {
  const c = document.createElement('canvas')
  c.width = 256
  c.height = 160
  const ctx = c.getContext('2d')!
  ctx.strokeStyle = '#1a1212'
  ctx.lineWidth = 9
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(128, 6)
  ctx.lineTo(128, 42)
  ctx.moveTo(44, 30)
  ctx.quadraticCurveTo(84, 78, 128, 42)
  ctx.quadraticCurveTo(172, 78, 212, 30)
  ctx.stroke()
  // Small open smile with a pink tongue.
  ctx.fillStyle = '#2a1414'
  ctx.beginPath()
  ctx.moveTo(84, 70)
  ctx.quadraticCurveTo(128, 150, 172, 70)
  ctx.quadraticCurveTo(128, 92, 84, 70)
  ctx.fill()
  ctx.fillStyle = '#e8828c'
  ctx.beginPath()
  ctx.ellipse(128, 106, 24, 12, 0, 0, Math.PI * 2)
  ctx.fill()
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

  // A decal conformed to the head: a grid laid over (x, y) on the face,
  // each vertex dropped onto the surface just in front of it.
  const decal = (cx: number, cy: number, w: number, h: number, rot: number, mirror: boolean, mat: THREE.Material) => {
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
        const q = hitFront(x, y)
        pts.push(q.x, q.y, q.z + 0.02)
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
  for (const side of [-1, 1]) decal(side * EYE_X, EYE_Y, 1.24, 1.06, side * 0.1, side < 0, eyeMat)

  // ---- Button nose and a tiny smile ----
  const nose = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), new THREE.MeshPhysicalMaterial({ color: 0x141011, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.12 }))
  {
    const np = nose.geometry.getAttribute('position') as THREE.BufferAttribute
    for (let i = 0; i < np.count; i++) np.setX(i, np.getX(i) * (1 + 0.3 * np.getY(i)))
    nose.geometry.computeVertexNormals()
  }
  nose.scale.set(0.15, 0.1, 0.1)
  nose.position.copy(hitFront(0, -0.5)).add(V(0, 0, -0.02))
  nose.castShadow = true
  root.add(nose)
  decal(0, -0.72, 0.46, 0.29, 0, false, new THREE.MeshStandardMaterial({ map: mouthTexture(), transparent: true, alphaTest: 0.25, roughness: 0.7, side: THREE.DoubleSide }))

  // ---- Ears: huge, long-furred, set high on the head ----
  const earCard = plushCard(spec.ear)
  for (const side of [-1, 1]) {
    const ear = earCard.clone()
    ear.position.set(side * 0.95, 1.55, -0.7)
    ear.rotation.set(-0.08, side * -0.28, side * -0.3)
    ear.scale.set(side, 1, 1)
    root.add(ear)
  }

  // ---- Fluff ----
  const tuft = tuftTexture()
  const tuftGeo = new THREE.PlaneGeometry(1, 1)
  tuftGeo.translate(0, 0.5, 0)
  const tuftMat = (color: THREE.Color) => new THREE.MeshStandardMaterial({ map: tuft, color, transparent: true, alphaTest: 0.04, depthWrite: false, side: THREE.DoubleSide, roughness: 1 })
  interface Tuft { mesh: THREE.Mesh; a: number; inset: number; size: number; ruff: boolean }
  const tufts: Tuft[] = []
  const addTuft = (a: number, inset: number, size: number, ruff: boolean) => {
    const mesh = new THREE.Mesh(tuftGeo, tuftMat(new THREE.Color()))
    mesh.renderOrder = 3
    root.add(mesh)
    tufts.push({ mesh, a, inset, size, ruff })
  }
  for (let i = 0; i < 110; i++) addTuft((i / 110) * Math.PI * 2 + rnd() * 0.05, 0.92, 0.42 + rnd() * 0.32, false)
  // Big white cheek fluff, sweeping out sideways from the cheeks (angle
  // 0 = the head's left in view, π = its right; a little below level).
  for (let i = 0; i < 22; i++) {
    const t = i / 21
    addTuft(-0.08 - t * 0.75, 0.84, 0.9 + rnd() * 0.75, true)
    addTuft(Math.PI + 0.08 + t * 0.75, 0.84, 0.9 + rnd() * 0.75, true)
  }
  const SIL_C = V(0, 0.15, -0.85)
  const SIL_R = V(2.0, 2.2, 1.75)
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
        if (t.ruff) outward.add(V(0, -0.15, 0)).normalize()
        // Rooted inside the rim and brought forward to the visible surface,
        // so they soften the edge while the snout can still pass in front.
        t.mesh.position.copy(SIL_C).addScaledVector(p, t.inset).addScaledVector(toCam, 0.8)
        const x = V(0, 0, 0).crossVectors(outward, toCam).normalize()
        const z = V(0, 0, 0).crossVectors(x, outward).normalize()
        m4.makeBasis(x, outward, z)
        t.mesh.quaternion.setFromRotationMatrix(m4)
        t.mesh.scale.set(t.size * 1.1, t.size, 1)
        ;(t.mesh.material as THREE.MeshStandardMaterial).color.copy(spec.fur(t.mesh.position)).multiplyScalar(1.25)
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
  const cheeks = ss(0.12 - p.y, -0.06, 0.06) * ss(ax, 0.22, 0.4)
  const lower = ss(-0.6 - p.y, -0.05, 0.05)
  let w = Math.max(cheeks, lower) * front
  w = Math.max(w, ss(-p.y, 1.55, 1.8))
  // Pale brow spots above the inner corners of the eyes.
  w = Math.max(w, near(p, 0.42, 0.66, 0.09, 0.13) * ss(p.z, 0, 0.4))
  return THREE.MathUtils.clamp(w, 0, 1)
}

function foxFur(p: V3) {
  const peach = C('#f09b55').lerp(C('#f8bb84'), ss(p.y, -0.5, 1.2) * 0.45 * ss(p.z, -0.2, 0.8))
  return peach.lerp(C('#fbf7f1'), whiteFace(p))
}

function huskyFur(p: V3) {
  const grey = C('#666b75').lerp(C('#40444c'), ss(p.y, 0.5, 2) * 0.6)
  // Huskies' white mask reaches up round the eyes too.
  const ax = Math.abs(p.x)
  const mask = ss(0.5 - p.y, -0.05, 0.05) * ss(ax, 0.2, 0.36) * ss(p.z, -0.6, 0)
  return grey.lerp(C('#f6f6f4'), Math.max(whiteFace(p), mask))
}

const EAR_OUTLINE = smooth([[60, 890], [60, 560], [130, 270], [250, 20], [380, 250], [460, 540], [462, 890], [262, 900]])
const EAR_INNER = smooth([[150, 850], [150, 590], [205, 330], [255, 170], [318, 330], [370, 590], [372, 850], [262, 862]])

export function foxHead(): Model {
  return fursuit({
    fur: foxFur,
    ear: {
      w: 512, h: 900, cardW: 2.5,
      outline: EAR_OUTLINE, inner: EAR_INNER, tip: [255, -40],
      root: '#d77a3a', mid: '#f3a462', light: '#ffd0a2',
      skin: '#f2a073', plush: '#f6b58a', tuft: '#fff4e8',
      strand: [28, 70],
    },
    iris: ['#3b1d10', '#a35a26', '#f5c46a'],
  })
}

export function huskyHead(): Model {
  return fursuit({
    fur: huskyFur,
    ear: {
      w: 512, h: 900, cardW: 2.35,
      outline: EAR_OUTLINE, inner: EAR_INNER, tip: [255, -40],
      root: '#33373e', mid: '#60656f', light: '#a0a5ad',
      skin: '#dad6d4', plush: '#f4f2f0', tuft: '#ffffff',
      strand: [28, 70],
    },
    iris: ['#0b2440', '#2e78c0', '#a8e4ff'],
  })
}
