import * as THREE from 'three'
import { FaceLandmarker } from '@mediapipe/tasks-vision'
import { FACE_TRIANGULATION } from '../faceTriangulation'
import { connectorsToLoop } from '../landmarks'
import { HEAD_RINGS, headRing, type Model, type Rig } from './scene'

// Masks that are actually worn: built from the live 468-point face mesh,
// lifted off the skin along its surface normals, and extended past the
// face's outline with extra rings that sweep outward and back so the mask
// wraps over the forehead, temples and around the head. Lit by the same
// estimated light and reflections as every other model, so the fabric and
// rubber read with real volume — cheekbones catch light, eye sockets fall
// into shade.

const OVAL = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
const LEFT_EYE = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
const RIGHT_EYE = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)
const RINGS = HEAD_RINGS
const N = 468
const VERTS = N + OVAL.length * RINGS
const HEAD_CENTER = new THREE.Vector3(0, 0.25, -1.3)

/** Cylindrical UVs around the head — the web/texture wraps the sides
 *  instead of smearing like a flat front projection would. */
function uvOf(p: THREE.Vector3): [number, number] {
  const a = Math.atan2(p.x, p.z - HEAD_CENTER.z)
  return [0.5 + (a / Math.PI) * 0.9, 0.5 + (p.y - 0.05) / 3.4]
}

interface MaskShape {
  /** Lift off the skin (rig units), optionally varying over the face. */
  lift: (p: THREE.Vector3) => number
  /** Extra height over the crown, to cover the hair. */
  hair: number
  /** Signed "keep" value per vertex (negative = cut away). Interpolated
   *  across triangles and cut in the shader, so openings and edges come
   *  out as smooth curves rather than the mesh's triangle staircase. */
  cut?: (p: THREE.Vector3, eyes: EyeHole[]) => number
}

interface EyeHole {
  c: THREE.Vector3
  rx: number
  ry: number
}

function maskGeometry() {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(VERTS * 3), 3))
  g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(VERTS * 3), 3))
  g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(VERTS * 2), 2))
  g.setAttribute('aCut', new THREE.BufferAttribute(new Float32Array(VERTS).fill(1), 1))
  const idx: number[] = Array.from(FACE_TRIANGULATION)
  const ring = (k: number, o: number) => (k === 0 ? OVAL[o] : N + (k - 1) * OVAL.length + o)
  for (let k = 0; k < RINGS; k++)
    for (let o = 0; o < OVAL.length; o++) {
      const o2 = (o + 1) % OVAL.length
      idx.push(ring(k, o), ring(k + 1, o), ring(k, o2), ring(k, o2), ring(k + 1, o), ring(k + 1, o2))
    }
  // Both windings, chosen per frame (see updateMask).
  const flipped = idx.slice()
  for (let t = 0; t < flipped.length; t += 3) [flipped[t + 1], flipped[t + 2]] = [flipped[t + 2], flipped[t + 1]]
  g.userData.windings = [new THREE.Uint16BufferAttribute(idx, 1), new THREE.Uint16BufferAttribute(flipped, 1)]
  g.setIndex(g.userData.windings[0])
  return g
}

/** Adds the per-vertex cut to a standard material. */
function withCut<T extends THREE.Material>(m: T): T {
  m.onBeforeCompile = (shader) => {
    shader.vertexShader = 'attribute float aCut;\nvarying float vCut;\n' + shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n  vCut = aCut;')
    shader.fragmentShader = 'varying float vCut;\n' + shader.fragmentShader.replace('void main() {', 'void main() {\n  if (vCut < 0.0) discard;')
  }
  return m
}

/** Rebuilds the mask surface from this frame's landmarks. */
function updateMask(g: THREE.BufferGeometry, rig: Rig, shape: MaskShape) {
  const L: THREE.Vector3[] = Array.from({ length: N }, (_, i) => rig.local(i))
  // Per-vertex normals of the bare face, oriented away from the head.
  const nrm = Array.from({ length: N }, () => new THREE.Vector3())
  const e1 = new THREE.Vector3()
  const e2 = new THREE.Vector3()
  const fn = new THREE.Vector3()
  for (let t = 0; t < FACE_TRIANGULATION.length; t += 3) {
    const a = FACE_TRIANGULATION[t]
    const b = FACE_TRIANGULATION[t + 1]
    const c = FACE_TRIANGULATION[t + 2]
    e1.subVectors(L[b], L[a])
    e2.subVectors(L[c], L[a])
    fn.crossVectors(e1, e2)
    nrm[a].add(fn)
    nrm[b].add(fn)
    nrm[c].add(fn)
  }
  for (let i = 0; i < N; i++) {
    nrm[i].normalize()
    if (nrm[i].dot(e1.subVectors(L[i], HEAD_CENTER)) < 0) nrm[i].negate()
  }
  const P: THREE.Vector3[] = []
  for (let i = 0; i < N; i++) P.push(L[i].clone().addScaledVector(nrm[i], shape.lift(L[i])))

  // Past the outline the mask continues around the head: each ring point
  // lies on a curve that leaves the face outline along the face's own
  // surface direction (so there's no fold at the hairline or jaw) and bends
  // round to the skull the scene uses as its head occluder, lifted by the
  // mask's thickness and any hair allowance.
  const ringPts: THREE.Vector3[][] = Array.from({ length: RINGS }, () => [])
  for (const i of OVAL) {
    const base = P[i]
    const n = nrm[i]
    const out = new THREE.Vector3(base.x, base.y + 0.15, 0).normalize()
    const tangent = out.clone().addScaledVector(n, -out.dot(n)).normalize()
    const end = headRing(L[i], RINGS, shape.lift(L[i]), shape.hair)
    const ctrl = base.clone().addScaledVector(tangent, base.distanceTo(end) * 0.55)
    for (let k = 1; k <= RINGS; k++) {
      const s = k / RINGS
      ringPts[k - 1].push(base.clone().multiplyScalar((1 - s) * (1 - s)).addScaledVector(ctrl, 2 * s * (1 - s)).addScaledVector(end, s * s))
    }
  }
  for (const r of ringPts) P.push(...r)

  const eyes: EyeHole[] = [LEFT_EYE, RIGHT_EYE].map((loop) => {
    const pts = loop.map((i) => L[i])
    const c = pts.reduce((s, p) => s.add(p), new THREE.Vector3()).divideScalar(pts.length)
    const rx = (Math.max(...pts.map((p) => p.x)) - Math.min(...pts.map((p) => p.x))) * 0.72
    return { c: c.add(new THREE.Vector3(0, 0.03, 0)), rx, ry: rx * 0.6 }
  })
  const pos = g.getAttribute('position') as THREE.BufferAttribute
  const uv = g.getAttribute('uv') as THREE.BufferAttribute
  const cut = g.getAttribute('aCut') as THREE.BufferAttribute
  P.forEach((p, i) => {
    pos.setXYZ(i, p.x, p.y, p.z)
    const [u, v] = uvOf(p)
    uv.setXY(i, u, v)
    cut.setX(i, shape.cut ? shape.cut(i < N ? L[i] : p, eyes) : 1)
  })
  // A mirrored selfie reverses the face mesh's triangle winding, which makes
  // the renderer treat the outer surface as a back face and light it inside-
  // out. Pick whichever winding faces outward this frame.
  let facing = 0
  for (let t = 0; t < FACE_TRIANGULATION.length; t += 3) {
    const a = P[FACE_TRIANGULATION[t]]
    e1.subVectors(P[FACE_TRIANGULATION[t + 1]], a)
    e2.subVectors(P[FACE_TRIANGULATION[t + 2]], a)
    facing += fn.crossVectors(e1, e2).dot(e1.subVectors(a, HEAD_CENTER))
  }
  const winding = g.userData.windings[facing >= 0 ? 0 : 1]
  if (g.index !== winding) g.setIndex(winding)
  g.computeVertexNormals()
  // Belt and braces: keep normals pointing outward.
  const nAttr = g.getAttribute('normal') as THREE.BufferAttribute
  const v = new THREE.Vector3()
  const pv = new THREE.Vector3()
  for (let i = 0; i < VERTS; i++) {
    v.fromBufferAttribute(nAttr, i)
    pv.fromBufferAttribute(pos, i).sub(HEAD_CENTER)
    if (v.dot(pv) < 0) nAttr.setXYZ(i, -v.x, -v.y, -v.z)
  }
  pos.needsUpdate = true
  uv.needsUpdate = true
  cut.needsUpdate = true
  nAttr.needsUpdate = true
  g.computeBoundingSphere()
}

// ---- Procedural textures -----------------------------------------------------

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

/** Draws the web in UV space: spokes from between the eyes, sagging rings. */
function drawWeb(ctx: CanvasRenderingContext2D, size: number, color: string, width: number) {
  const cx = size * 0.5
  const cy = size * (1 - (0.5 + (0 - 0.05) / 3.4))
  ctx.strokeStyle = color
  ctx.lineWidth = width
  ctx.lineCap = 'round'
  const spokes = 26
  const R = size * 0.75
  const ends: [number, number][] = []
  for (let i = 0; i < spokes; i++) {
    const a = (i / spokes) * Math.PI * 2
    ends.push([cx + Math.cos(a) * R, cy + Math.sin(a) * R * 1.05])
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.lineTo(ends[i][0], ends[i][1])
    ctx.stroke()
  }
  for (let k = 1; k <= 13; k++) {
    const r = Math.pow(k / 13, 1.25) * 0.98
    ctx.beginPath()
    for (let i = 0; i <= spokes; i++) {
      const [ax, ay] = ends[i % spokes]
      const [bx, by] = ends[(i + 1) % spokes]
      const pa = [cx + (ax - cx) * r, cy + (ay - cy) * r]
      const pb = [cx + (bx - cx) * r, cy + (by - cy) * r]
      const m = [cx + ((pa[0] + pb[0]) / 2 - cx) * 0.92, cy + ((pa[1] + pb[1]) / 2 - cy) * 0.92]
      if (i === 0) ctx.moveTo(pa[0], pa[1])
      ctx.quadraticCurveTo(m[0], m[1], pb[0], pb[1])
    }
    ctx.stroke()
  }
}

function spiderTextures() {
  const size = 1024
  const [col, c] = canvas(size)
  const g = c.createLinearGradient(0, 0, 0, size)
  g.addColorStop(0, '#d01722')
  g.addColorStop(1, '#a50f18')
  c.fillStyle = g
  c.fillRect(0, 0, size, size)
  drawWeb(c, size, 'rgba(14,10,12,0.92)', 4)
  const map = new THREE.CanvasTexture(col)
  map.colorSpace = THREE.SRGBColorSpace
  map.anisotropy = 8
  // Height: raised web lines plus a fine fabric weave.
  const [hc, h] = canvas(size)
  h.fillStyle = '#000'
  h.fillRect(0, 0, size, size)
  const img = h.getImageData(0, 0, size, size)
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const v = 40 + 30 * (Math.sin(x * 1.9) * Math.sin(y * 1.9)) + Math.random() * 18
      const k = (y * size + x) * 4
      img.data[k] = img.data[k + 1] = img.data[k + 2] = v
    }
  h.putImageData(img, 0, 0)
  h.filter = 'blur(1.5px)'
  drawWeb(h, size, '#ffffff', 6)
  h.filter = 'none'
  return { map, normal: normalMapFrom(hc, 2.2) }
}

function pebbleNormal() {
  const size = 512
  const [hc, h] = canvas(size)
  h.fillStyle = '#808080'
  h.fillRect(0, 0, size, size)
  for (let i = 0; i < 9000; i++) {
    const x = Math.random() * size
    const y = Math.random() * size
    const r = 1.5 + Math.random() * 3
    const g = h.createRadialGradient(x, y, 0, x, y, r)
    g.addColorStop(0, 'rgba(255,255,255,0.5)')
    g.addColorStop(1, 'rgba(255,255,255,0)')
    h.fillStyle = g
    h.fillRect(x - r, y - r, r * 2, r * 2)
  }
  const t = normalMapFrom(hc, 1.4)
  t.repeat.set(3, 3)
  return t
}

// ---- Spider ------------------------------------------------------------------

function spiderLens(side: number) {
  // The classic swept lens: pointed toward the nose, sweeping up and out.
  const s = new THREE.Shape()
  s.moveTo(-0.26, -0.02)
  s.bezierCurveTo(-0.22, 0.12, 0.05, 0.2, 0.3, 0.19)
  s.bezierCurveTo(0.3, 0.02, 0.12, -0.16, -0.26, -0.02)
  const outer = new THREE.Shape(s.getPoints(64).map((p) => new THREE.Vector2(p.x * 1.18 + 0.01, p.y * 1.35 + 0.005)))
  outer.holes.push(new THREE.Path(s.getPoints(64).reverse()))
  const frame = new THREE.Mesh(
    new THREE.ExtrudeGeometry(outer, { depth: 0.03, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.015, bevelSegments: 4, curveSegments: 40 }),
    new THREE.MeshPhysicalMaterial({ color: 0x0c0c0e, roughness: 0.3, clearcoat: 0.8 }),
  )
  const lensGeo = new THREE.ShapeGeometry(s, 32)
  const p = lensGeo.getAttribute('position') as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) p.setZ(i, 0.03 - 0.25 * (p.getX(i) ** 2 + p.getY(i) ** 2))
  lensGeo.computeVertexNormals()
  const lens = new THREE.Mesh(lensGeo, new THREE.MeshPhysicalMaterial({ color: 0xf2f4f7, roughness: 0.12, metalness: 0.25, clearcoat: 1, envMapIntensity: 1.6 }))
  const grp = new THREE.Group()
  grp.add(frame, lens)
  grp.scale.set(side, 1, 1)
  grp.traverse((o) => ((o as THREE.Mesh).castShadow = true))
  return grp
}

export function spiderMask(): Model {
  const geo = maskGeometry()
  const { map, normal } = spiderTextures()
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshPhysicalMaterial({ map, normalMap: normal, normalScale: new THREE.Vector2(1.1, 1.1), roughness: 0.62, sheen: 0.7, sheenRoughness: 0.45, sheenColor: new THREE.Color(0xff5a5a), side: THREE.DoubleSide }),
  )
  mesh.frustumCulled = false
  const lenses = [spiderLens(-1), spiderLens(1)]
  const root = new THREE.Group()
  root.add(mesh, ...lenses)
  const shape: MaskShape = { lift: () => 0.03, hair: 0.45 }
  return {
    root,
    update(rig) {
      updateMask(geo, rig, shape)
      const eyes = [LEFT_EYE, RIGHT_EYE].map((loop) => loop.reduce((s, i) => s.add(rig.local(i)), new THREE.Vector3()).divideScalar(loop.length))
      eyes.sort((a, b) => a.x - b.x)
      eyes.forEach((e, k) => {
        const side = k ? 1 : -1
        lenses[k].position.set(e.x + side * 0.04, e.y + 0.05, e.z + 0.1)
        lenses[k].rotation.set(-0.05, side * 0.32, side * 0.12)
      })
    },
  }
}

// ---- Bat ----------------------------------------------------------------------

export function batCowl(): Model {
  const geo = maskGeometry()
  const mat = withCut(new THREE.MeshPhysicalMaterial({ color: 0x1a1b20, roughness: 0.42, metalness: 0.15, clearcoat: 0.55, clearcoatRoughness: 0.35, normalMap: pebbleNormal(), normalScale: new THREE.Vector2(0.5, 0.5), side: THREE.DoubleSide }))
  const earMat = new THREE.MeshPhysicalMaterial({ color: 0x1a1b20, roughness: 0.42, metalness: 0.15, clearcoat: 0.55, clearcoatRoughness: 0.35 })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.castShadow = true
  mesh.frustumCulled = false
  // Ears: tall faceted spikes rising from the crown.
  const earGeo = new THREE.ConeGeometry(0.17, 0.85, 4, 1)
  earGeo.translate(0, 0.42, 0)
  const ears = [-1, 1].map((s) => {
    const m = new THREE.Mesh(earGeo, earMat)
    m.castShadow = true
    m.userData.s = s
    return m
  })
  const root = new THREE.Group()
  root.add(mesh, ...ears)
  let cutY = -0.9
  const lowerEdge = (x: number) => cutY - 0.55 * Math.max(0, Math.abs(x) - 0.22)
  const shape: MaskShape = {
    // Sculpted: thicker over the brow ridge and down the nose for the
    // cowl's heroic brow and moulded nose guard (all smooth functions, so
    // the surface stays continuous).
    lift: (p) =>
      0.045 +
      0.06 * Math.exp(-(((p.y - 0.12) / 0.16) ** 2)) * Math.max(0, 1 - Math.abs(p.x) / 0.9) +
      0.03 * Math.exp(-((p.x / 0.18) ** 2)) * THREE.MathUtils.smoothstep(-p.y, -0.15, 0.05),
    hair: 0.35,
    cut: (p, eyes) => {
      const below = (p.y - lowerEdge(p.x)) * 6
      const hole = Math.min(...eyes.map((e) => Math.hypot((p.x - e.c.x) / e.rx, (p.y - e.c.y) / e.ry) - 1))
      return Math.min(below, hole)
    },
  }
  return {
    root,
    update(rig) {
      cutY = rig.local(2).y - 0.04
      updateMask(geo, rig, shape)
      const top = rig.local(10)
      const tl = rig.local(21)
      const tr = rig.local(251)
      for (const e of ears) {
        const temple = e.userData.s < 0 ? (tl.x < tr.x ? tl : tr) : tl.x < tr.x ? tr : tl
        e.position.copy(top).lerp(temple, 0.58).add(new THREE.Vector3(0, 0.62, -0.42))
        e.rotation.set(-0.12, Math.PI / 4, e.userData.s * -0.12)
      }
    },
  }
}
