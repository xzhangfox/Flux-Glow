import * as THREE from 'three'
import { plushCard, smooth, type PlushSpec } from './plush'
import type { Model, Rig } from './scene'

// Full fursuit heads that replace the wearer's head entirely: an oversized
// plush head (bigger than head + hair, so nothing of the real head shows)
// sculpted into a canine face — snout, nose bridge, eye sockets, brow,
// cheeks — with painted fur, glossy resin anime eyes, big plush ears, and — what makes it read as faux fur rather than a smooth ball —
// fluffy tufts all round the silhouette, re-aimed at the camera every frame
// so the outline is always soft, with long cheek ruffs at the sides.

type Dir = THREE.Vector3

interface SuitSpec {
  /** Fur colour at a direction on the head (unit vector, +z = front). */
  fur: (s: Dir) => THREE.Color
  ear: PlushSpec
  iris: [string, string, string]
  outline: string
}

const C = (h: string) => new THREE.Color(h)
const ss = THREE.MathUtils.smoothstep

// The head shell, in rig units (1 = eye distance; origin between the eyes).
const CENTER = new THREE.Vector3(0, 0.05, -0.85)
const RX = 1.8
const RZ = 1.7
const ry = (y: number) => (y > 0 ? 2.25 : 2.15)
const cheekBulge = (y: number) => 1 + 0.1 * Math.exp(-(((y + 0.35) / 0.35) ** 2))

// The face is a flatter plane than the back of the head, so the snout
// stands out from it.
const rz = (z: number) => (z > 0 ? RZ * 0.86 : RZ)
function shellPoint(s: Dir) {
  return new THREE.Vector3(s.x * RX * cheekBulge(s.y), s.y * ry(s.y), s.z * rz(s.z)).add(CENTER)
}

// Sculpting on top of the base volume, in rig units on the face (x across,
// y up, origin between the wearer's eyes): a canine face rather than a
// ball — a snout that rises from the stop between the eyes to a blunt nose
// and curls back under into the mouth and chin, eye sockets under a brow
// ridge, puffed cheeks. Faded out toward the sides of the head.
const EYE_X = 0.56
const EYE_Y = 0.14
const gauss = (dx: number, dy: number, rx: number, ry: number) => Math.exp(-((dx / rx) ** 2) - (dy / ry) ** 2)
function sculpt(x: number, y: number) {
  const ax = Math.abs(x)
  // Snout: length along the face, rising from the stop to the nose tip,
  // then curling back under into the chin. Narrow on the bridge, broad at
  // the mouth, flat-topped across.
  const len = 1.25 * ss(-y, -0.05, 0.62) * (1 - ss(-y, 0.76, 1.3))
  const halfW = 0.26 + 0.26 * ss(-y, 0.0, 0.85)
  const snout = len * Math.exp(-Math.pow(ax / halfW, 3.2))
  // Stop: a slight dip at the root of the nose, between the eyes.
  const stop = -0.06 * gauss(x, y - 0.0, 0.16, 0.14)
  const sockets = -0.07 * gauss(ax - EYE_X, y - EYE_Y, 0.36, 0.3)
  const brow = 0.09 * gauss(ax - 0.42, y - 0.5, 0.3, 0.1)
  const cheeks = 0.22 * gauss(ax - 0.8, y + 0.6, 0.34, 0.34)
  const forehead = 0.06 * gauss(x, y - 0.75, 0.6, 0.35)
  return snout + stop + sockets + brow + cheeks + forehead
}

function shellNormal(s: Dir) {
  return new THREE.Vector3(s.x / RX, s.y / ry(s.y), s.z / RZ).normalize()
}

/** Canvas pixel → direction (matches the rotated SphereGeometry's UVs:
 *  front at the canvas centre, seam at the back). */
function dirAt(u: number, v: number) {
  const phi = u * Math.PI * 2
  const th = v * Math.PI
  return new THREE.Vector3(-Math.sin(phi) * Math.sin(th), Math.cos(th), -Math.cos(phi) * Math.sin(th))
}
function uvOf(s: Dir): [number, number] {
  const th = Math.acos(THREE.MathUtils.clamp(s.y, -1, 1))
  let phi = Math.atan2(-s.x, -s.z)
  if (phi < 0) phi += Math.PI * 2
  return [phi / (Math.PI * 2), th / Math.PI]
}

let seed = 11
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647

/** Fur painted in the shell's UV space: a soft base, then tens of thousands
 *  of strands flowing back and down from the face. */
function paintFur(W: number, H: number, fur: (s: Dir) => THREE.Color, strands: number) {
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const ctx = c.getContext('2d')!
  const img = ctx.createImageData(W, H)
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const col = fur(dirAt(x / W, y / H)).multiplyScalar(0.78)
      const k = (y * W + x) * 4
      img.data[k] = col.r * 255
      img.data[k + 1] = col.g * 255
      img.data[k + 2] = col.b * 255
      img.data[k + 3] = 255
    }
  ctx.putImageData(img, 0, 0)
  ctx.lineCap = 'round'
  // Plush fur radiates from the muzzle, back over the head and down.
  const src = new THREE.Vector3(0, -0.4, 0.92).normalize()
  const tmp = new THREE.Vector3()
  for (let i = 0; i < strands; i++) {
    const u = rnd()
    const v = Math.acos(1 - 2 * rnd()) / Math.PI
    const s = dirAt(u, v)
    const f = tmp.copy(s).sub(src).addScaledVector(s, -s.clone().sub(src).dot(s)).add(new THREE.Vector3(0, -0.15, 0))
    if (f.lengthSq() < 1e-4) f.set(0, -1, 0)
    f.normalize()
    const [u2, v2] = uvOf(s.clone().addScaledVector(f, 0.02).normalize())
    let dx = (u2 - u) * W
    if (dx > W / 2) dx -= W
    if (dx < -W / 2) dx += W
    const dy = (v2 - v) * H
    const l = Math.hypot(dx, dy) || 1
    const len = (4 + rnd() * 7) * (W / 1024)
    const j = s.clone().add(new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).multiplyScalar(0.05)).normalize()
    const col = fur(j).multiplyScalar(0.7 + rnd() * 0.45)
    ctx.strokeStyle = '#' + col.getHexString()
    ctx.globalAlpha = 0.3 + rnd() * 0.4
    ctx.lineWidth = (0.7 + rnd() * 1.1) * (W / 1024)
    const x0 = u * W
    const y0 = v * H
    const bend = (rnd() - 0.5) * len * 0.4
    ctx.beginPath()
    ctx.moveTo(x0, y0)
    ctx.quadraticCurveTo(x0 + (dx / l) * len * 0.5 - (dy / l) * bend, y0 + (dy / l) * len * 0.5 + (dx / l) * bend, x0 + (dx / l) * len, y0 + (dy / l) * len)
    ctx.stroke()
  }
  // A touch of softness: plush, not bristle.
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
  // Roots fade in, so the tufts grow out of the shell's fur without a line.
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

/** Big glossy anime eye (the right eye; the left is mirrored). */
function eyeTexture(iris: [string, string, string], outline: string) {
  const W = 512
  const H = 420
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const ctx = c.getContext('2d')!
  const shape = () => {
    ctx.beginPath()
    ctx.moveTo(40, 230)
    ctx.bezierCurveTo(70, 60, 380, 20, 478, 150)
    ctx.bezierCurveTo(500, 260, 420, 390, 270, 395)
    ctx.bezierCurveTo(140, 398, 30, 340, 40, 230)
    ctx.closePath()
  }
  ctx.save()
  shape()
  ctx.fillStyle = '#fbf8f4'
  ctx.fill()
  ctx.clip()
  // Iris: dark at the top (under the lid's shadow), glowing at the bottom.
  const g = ctx.createLinearGradient(0, 70, 0, 380)
  g.addColorStop(0, iris[0])
  g.addColorStop(0.55, iris[1])
  g.addColorStop(1, iris[2])
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.ellipse(262, 235, 150, 168, 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.strokeStyle = 'rgba(30,15,10,0.6)'
  ctx.lineWidth = 8
  ctx.stroke()
  ctx.fillStyle = 'rgba(25,10,5,0.75)'
  ctx.beginPath()
  ctx.ellipse(262, 255, 66, 82, 0, 0, Math.PI * 2)
  ctx.fill()
  // Sparkle star and highlights.
  const star = (x: number, y: number, r: number) => {
    ctx.beginPath()
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI) / 4 - Math.PI / 2
      const rr = i % 2 ? r * 0.3 : r
      ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr)
    }
    ctx.closePath()
    ctx.fill()
  }
  ctx.fillStyle = 'rgba(255,240,200,0.95)'
  star(230, 200, 46)
  ctx.fillStyle = 'rgba(255,255,255,0.92)'
  ctx.beginPath()
  ctx.ellipse(330, 150, 44, 34, -0.3, 0, Math.PI * 2)
  ctx.fill()
  ctx.beginPath()
  ctx.ellipse(205, 320, 18, 14, 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
  // Heavy upper lash line, thin lower line, a flick at the outer corner.
  ctx.strokeStyle = outline
  ctx.lineCap = 'round'
  ctx.lineWidth = 34
  ctx.beginPath()
  ctx.moveTo(40, 230)
  ctx.bezierCurveTo(70, 60, 380, 20, 478, 150)
  ctx.stroke()
  ctx.lineWidth = 10
  shape()
  ctx.stroke()
  ctx.fillStyle = outline
  ctx.beginPath()
  ctx.moveTo(440, 110)
  ctx.lineTo(505, 120)
  ctx.lineTo(482, 175)
  ctx.closePath()
  ctx.fill()
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

function mouthTexture(color: string) {
  const c = document.createElement('canvas')
  c.width = 256
  c.height = 128
  const ctx = c.getContext('2d')!
  ctx.strokeStyle = color
  ctx.lineWidth = 9
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(128, 8)
  ctx.lineTo(128, 40)
  ctx.moveTo(40, 34)
  ctx.quadraticCurveTo(80, 78, 128, 40)
  ctx.quadraticCurveTo(176, 78, 216, 34)
  ctx.stroke()
  // Small open mouth with a pink tongue.
  ctx.fillStyle = '#2a1414'
  ctx.beginPath()
  ctx.ellipse(128, 70, 38, 22, 0, 0, Math.PI)
  ctx.fill()
  ctx.fillStyle = '#e8828c'
  ctx.beginPath()
  ctx.ellipse(128, 84, 22, 9, 0, 0, Math.PI)
  ctx.fill()
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

function curvedPlane(w: number, h: number, bulge: number) {
  const g = new THREE.PlaneGeometry(w, h, 16, 16)
  const p = g.getAttribute('position') as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) p.setZ(i, bulge * (1 - (p.getX(i) / (w / 2)) ** 2 - (p.getY(i) / (h / 2)) ** 2))
  g.computeVertexNormals()
  return g
}

function fursuit(spec: SuitSpec): Model {
  seed = 11
  const root = new THREE.Group()

  // Shell.
  const g = new THREE.SphereGeometry(1, 180, 120)
  g.rotateY(-Math.PI / 2)
  const pos = g.getAttribute('position') as THREE.BufferAttribute
  const uv = g.getAttribute('uv') as THREE.BufferAttribute
  const col = new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3)
  g.setAttribute('color', col)
  const v = new THREE.Vector3()
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).normalize()
    const q = shellPoint(v)
    const front = ss(v.z, 0.25, 0.65)
    q.z += sculpt(q.x, q.y) * front
    pos.setXYZ(i, q.x, q.y, q.z)
    // Baked occlusion: darker where the surface sits below its
    // surroundings — beside and under the snout, in the eye sockets.
    let around = 0
    for (let k = 0; k < 8; k++) {
      const a = (k / 8) * Math.PI * 2
      around += sculpt(q.x + Math.cos(a) * 0.3, q.y + Math.sin(a) * 0.3)
    }
    const occ = Math.max(0, around / 8 - sculpt(q.x, q.y)) * front
    const shade = 1 - Math.min(0.45, occ * 1.6)
    col.setXYZ(i, shade, shade, shade)
    // The seam sits at the back, out of sight; keep the UV's own u there.
    const [u, w] = uvOf(v)
    uv.setXY(i, Math.abs(u - uv.getX(i)) > 0.5 ? uv.getX(i) : u, 1 - w)
  }
  g.computeVertexNormals()
  const furTex = paintFur(1024, 512, spec.fur, 110000)
  const furMat = new THREE.MeshPhysicalMaterial({ vertexColors: true, map: furTex, bumpMap: furTex, bumpScale: 0.35, roughness: 0.95, sheen: 0.8, sheenRoughness: 0.6, sheenColor: new THREE.Color(0x8a7a6a) })
  const shell = new THREE.Mesh(g, furMat)
  shell.castShadow = true
  shell.receiveShadow = true
  root.add(shell)

  // Features sit on the sculpted surface: found by casting rays at it.
  shell.updateMatrixWorld()
  const ray = new THREE.Raycaster()
  const surface = (x: number, y: number) => {
    ray.set(new THREE.Vector3(x, y, 10), new THREE.Vector3(0, 0, -1))
    const hit = ray.intersectObject(shell)[0]
    const n = hit?.face?.normal.clone() ?? new THREE.Vector3(0, 0, 1)
    if (n.z < 0) n.negate()
    return { p: hit?.point ?? new THREE.Vector3(x, y, 1), n }
  }
  const place = (o: THREE.Object3D, x: number, y: number, out: number, tiltTo = 0) => {
    const { p, n } = surface(x, y)
    n.lerp(new THREE.Vector3(0, 0, 1), tiltTo).normalize()
    o.position.copy(p).addScaledVector(n, out)
    o.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n)
  }

  // Nose: a broad, glossy leather nose capping the snout.
  const noseGeo = new THREE.SphereGeometry(1, 40, 28)
  {
    const np = noseGeo.getAttribute('position') as THREE.BufferAttribute
    for (let i = 0; i < np.count; i++) {
      const y = np.getY(i)
      // Wider on top, tapering underneath into the philtrum.
      np.setX(i, np.getX(i) * (1 + 0.25 * y))
    }
    noseGeo.computeVertexNormals()
  }
  const nose = new THREE.Mesh(noseGeo, new THREE.MeshPhysicalMaterial({ color: 0x131012, roughness: 0.35, clearcoat: 1, clearcoatRoughness: 0.15 }))
  nose.scale.set(0.2, 0.13, 0.12)
  place(nose, 0, -0.5, 0.02, 0.4)
  const mouth = new THREE.Mesh(curvedPlane(0.62, 0.31, 0.04), new THREE.MeshStandardMaterial({ map: mouthTexture('#1a1212'), transparent: true, alphaTest: 0.2, roughness: 0.6 }))
  place(mouth, 0, -0.78, 0.015)
  root.add(nose, mouth)

  // Eyes: glossy domed resin, set into the sockets under the brow.
  const eyeTex = eyeTexture(spec.iris, spec.outline)
  const eyeMat = new THREE.MeshPhysicalMaterial({ map: eyeTex, transparent: true, alphaTest: 0.3, roughness: 0.18, clearcoat: 1, clearcoatRoughness: 0.04 })
  for (const side of [-1, 1]) {
    const e = new THREE.Mesh(curvedPlane(0.92, 0.75, 0.08), eyeMat)
    place(e, side * EYE_X, EYE_Y - 0.02, 0.08, 0.45)
    e.rotateZ(side * -0.18)
    if (side < 0) e.scale.x = -1
    root.add(e)
  }

  // Ears.
  const earCard = plushCard(spec.ear)
  for (const side of [-1, 1]) {
    const ear = earCard.clone()
    const s = new THREE.Vector3(side * 0.46, 0.86, 0.02).normalize()
    ear.position.copy(shellPoint(s)).addScaledVector(shellNormal(s), -0.35)
    ear.rotation.set(-0.08, side * -0.2, side * -0.38)
    ear.scale.set(side, 1, 1)
    root.add(ear)
  }

  // Silhouette fluff, re-aimed at the camera each frame.
  const tuft = tuftTexture()
  const tuftGeo = new THREE.PlaneGeometry(1, 1)
  tuftGeo.translate(0, 0.5, 0)
  interface Tuft { mesh: THREE.Mesh; a: number; inset: number; size: number; ruff: boolean }
  const tufts: Tuft[] = []
  const addTuft = (a: number, inset: number, size: number, ruff: boolean) => {
    // Drawn over the shell (they only sit on its rim): their soft roots
    // blend the edge into fluff.
    const mat = new THREE.MeshStandardMaterial({ map: tuft, transparent: true, alphaTest: 0.04, depthWrite: false, depthTest: false, side: THREE.DoubleSide, roughness: 1 })
    const mesh = new THREE.Mesh(tuftGeo, mat)
    mesh.renderOrder = 3
    root.add(mesh)
    tufts.push({ mesh, a, inset, size, ruff })
  }
  for (let i = 0; i < 90; i++) addTuft((i / 90) * Math.PI * 2 + rnd() * 0.05, 0.88, 0.34 + rnd() * 0.18, false)
  for (let i = 0; i < 50; i++) addTuft(rnd() * Math.PI * 2, 0.9, 0.45 + rnd() * 0.25, false)
  // Cheek ruffs: long tufts sweeping out and down at both sides.
  for (const side of [-1, 1])
    for (let i = 0; i < 14; i++) addTuft(Math.PI / 2 - side * (Math.PI / 2 + 0.1 + (i / 13) * 0.8), 0.86, 0.8 + rnd() * 0.5, true)

  const toCam = new THREE.Vector3()
  const A = new THREE.Vector3(RX * 1.05, 2.2, RZ)
  const e1 = new THREE.Vector3()
  const e2 = new THREE.Vector3()
  const up = new THREE.Vector3()
  const m4 = new THREE.Matrix4()
  return {
    root,
    fullHead: true,
    update(rig: Rig) {
      // Camera direction in rig space.
      toCam.set(0, 0, 1).transformDirection(rig.inverse)
      // Ellipsoid silhouette for that view: s ⟂ A⁻¹·d, point = c + A·s.
      const m = new THREE.Vector3(toCam.x / A.x, toCam.y / A.y, toCam.z / A.z).normalize()
      e1.set(0, 1, 0).addScaledVector(m, -m.y)
      if (e1.lengthSq() < 1e-4) e1.set(1, 0, 0)
      e1.normalize()
      e2.crossVectors(m, e1).normalize()
      for (const t of tufts) {
        const s = e1.clone().multiplyScalar(Math.sin(t.a)).addScaledVector(e2, Math.cos(t.a))
        const p = new THREE.Vector3(s.x * A.x, s.y * (s.y > 0 ? 2.25 : 2.15), s.z * A.z)
        const outward = p.clone().addScaledVector(toCam, -p.dot(toCam)).normalize()
        if (t.ruff) outward.add(new THREE.Vector3(0, -0.45, 0)).normalize()
        // Rooted just inside the rim, so the tufts overlap the shell's edge
        // and soften it rather than ring it.
        t.mesh.position.copy(CENTER).addScaledVector(p, t.inset)
        up.copy(outward)
        const x = new THREE.Vector3().crossVectors(up, toCam).normalize()
        const z = new THREE.Vector3().crossVectors(x, up).normalize()
        m4.makeBasis(x, up, z)
        t.mesh.quaternion.setFromRotationMatrix(m4)
        const len = t.size * (t.ruff ? 1 : 1)
        t.mesh.scale.set(len * 1.1, len, 1)
        ;(t.mesh.material as THREE.MeshStandardMaterial).color.copy(spec.fur(s.clone().normalize()))
      }
    },
  }
}

// ---- Designs ----------------------------------------------------------------

const near = (s: Dir, x: number, y: number, r: number) => Math.exp(-(((s.x - x) ** 2 + (s.y - y) ** 2) / (r * r)))

function foxFur(s: Dir) {
  const orange = C('#f4a463').lerp(C('#ffcf9c'), ss(s.y, 0.2, 0.9) * 0.35 * Math.max(0, s.z))
  const white = C('#fbf5ec')
  const ax = Math.abs(s.x)
  const front = ss(s.z, -0.25, 0.15)
  // White cheeks and chin; the nose bridge stays orange.
  let w = ss(-s.y + (ax - 0.3) * 0.3, 0.06, 0.16) * front
  w = Math.max(w, ss(-s.y, 0.42, 0.55))
  w *= 1 - Math.exp(-((s.x / 0.16) ** 2)) * ss(s.y, -0.42, -0.3) * front
  // Two pale brow spots.
  w = Math.max(w, (near(s, -0.21, 0.36, 0.06) + near(s, 0.21, 0.36, 0.06)) * front)
  return orange.lerp(white, THREE.MathUtils.clamp(w, 0, 1))
}

function huskyFur(s: Dir) {
  const grey = C('#5d626c').lerp(C('#3a3e46'), ss(s.y, 0.3, 1) * 0.6)
  const white = C('#f6f6f4')
  const ax = Math.abs(s.x)
  const front = ss(s.z, -0.1, 0.3)
  // White face mask with a dark widow's peak down to between the eyes.
  let w = ss(0.28 - s.y + (ax - 0.1) * 0.3, 0, 0.12) * front
  w *= 1 - Math.exp(-((s.x / 0.1) ** 2)) * ss(s.y, 0.12, 0.25)
  w = Math.max(w, ss(-s.y, 0.35, 0.5))
  w = Math.max(w, (near(s, -0.22, 0.32, 0.05) + near(s, 0.22, 0.32, 0.05)) * front)
  return grey.lerp(white, THREE.MathUtils.clamp(w, 0, 1))
}

const SUIT_EAR_OUTLINE = smooth([[50, 690], [70, 380], [175, 130], [290, 15], [380, 140], [462, 390], [478, 690], [260, 700]])
const SUIT_EAR_INNER = smooth([[150, 660], [165, 420], [250, 210], [300, 140], [365, 380], [385, 660], [265, 672]])

export function foxHead(): Model {
  return fursuit({
    fur: foxFur,
    ear: {
      w: 512, h: 700, cardW: 1.9,
      outline: SUIT_EAR_OUTLINE, inner: SUIT_EAR_INNER, tip: [292, 0],
      root: '#c9763a', mid: '#f2a25f', light: '#ffd2a0',
      skin: '#f0a27a', plush: '#f7b48c', tuft: '#fff6ea',
      strand: [16, 40],
    },
    iris: ['#3d2010', '#b45a1e', '#f3c060'],
    outline: '#120c0c',
  })
}

export function huskyHead(): Model {
  return fursuit({
    fur: huskyFur,
    ear: {
      w: 512, h: 700, cardW: 1.8,
      outline: SUIT_EAR_OUTLINE, inner: SUIT_EAR_INNER, tip: [292, 0],
      root: '#2c3036', mid: '#555a63', light: '#8d939c',
      skin: '#d9d6d6', plush: '#f4f2f0', tuft: '#ffffff',
      strand: [16, 40],
    },
    iris: ['#0d2a4a', '#2d7fc4', '#a6e3ff'],
    outline: '#0c0e12',
  })
}
