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
  const face: number[] = Array.from(FACE_TRIANGULATION)
  const ringIdx: number[] = []
  const ring = (k: number, o: number) => (k === 0 ? OVAL[o] : N + (k - 1) * OVAL.length + o)
  for (let k = 0; k < RINGS; k++)
    for (let o = 0; o < OVAL.length; o++) {
      const o2 = (o + 1) % OVAL.length
      ringIdx.push(ring(k, o), ring(k + 1, o), ring(k, o2), ring(k, o2), ring(k + 1, o), ring(k + 1, o2))
    }
  const flip = (a: number[]) => {
    const f = a.slice()
    for (let t = 0; t < f.length; t += 3) [f[t + 1], f[t + 2]] = [f[t + 2], f[t + 1]]
    return f
  }
  // The face mesh and the hood rings can each come out wound either way
  // (mirroring flips the face; the rings' order depends on the outline's
  // direction), so all four combinations are prepared and picked per frame.
  g.userData.ringIdx = ringIdx
  g.userData.windings = new Map<string, THREE.BufferAttribute>()
  for (const [fk, fa] of [['f', face], ['F', flip(face)]] as const)
    for (const [rk, ra] of [['r', ringIdx], ['R', flip(ringIdx)]] as const) g.userData.windings.set(fk + rk, new THREE.Uint16BufferAttribute([...fa, ...ra], 1))
  g.setIndex(g.userData.windings.get('fr'))
  return g
}

/** Adds the per-vertex cut to a standard material. */
function withCut<T extends THREE.Material>(m: T): T {
  // Antialiased: coverage ramps over one pixel of the cut function, resolved
  // smooth by alpha-to-coverage instead of a hard, stair-stepped discard.
  m.alphaToCoverage = true
  m.onBeforeCompile = (shader) => {
    shader.vertexShader = 'attribute float aCut;\nvarying float vCut;\n' + shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n  vCut = aCut;')
    shader.fragmentShader =
      'varying float vCut;\n' +
      shader.fragmentShader
        .replace('void main() {', 'void main() {\n  float cutA = clamp(vCut / max(fwidth(vCut), 1e-4) + 0.5, 0.0, 1.0);\n  if (cutA <= 0.0) discard;')
        .replace('#include <dithering_fragment>', '#include <dithering_fragment>\n  gl_FragColor.a *= cutA;')
  }
  m.customProgramCacheKey = () => 'mask-cut'
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
  const facing = (idx: ArrayLike<number>) => {
    let sum = 0
    for (let t = 0; t < idx.length; t += 3) {
      const a = P[idx[t]]
      e1.subVectors(P[idx[t + 1]], a)
      e2.subVectors(P[idx[t + 2]], a)
      sum += fn.crossVectors(e1, e2).dot(e1.subVectors(a, HEAD_CENTER))
    }
    return sum
  }
  const key = (facing(FACE_TRIANGULATION) >= 0 ? 'f' : 'F') + (facing(g.userData.ringIdx) >= 0 ? 'r' : 'R')
  const winding = g.userData.windings.get(key)
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
  // Moulded blue-black plastic: smooth and glossy, so the sculpting reads
  // through sharp highlights rather than texture.
  const plastic = () => new THREE.MeshPhysicalMaterial({ color: 0x10141c, roughness: 0.42, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.06, envMapIntensity: 0.55, side: THREE.DoubleSide })
  const mesh = new THREE.Mesh(geo, withCut(plastic()))
  mesh.castShadow = true
  mesh.frustumCulled = false
  // Ears: tall, thin fins with slightly concave sides, bevelled edges.
  const earShape = new THREE.Shape()
  earShape.moveTo(-0.3, 0)
  earShape.quadraticCurveTo(-0.1, 0.5, -0.03, 1.2)
  earShape.lineTo(0.03, 1.2)
  earShape.quadraticCurveTo(0.14, 0.55, 0.34, 0)
  earShape.lineTo(-0.3, 0)
  const earGeo = new THREE.ExtrudeGeometry(earShape, { depth: 0.07, bevelEnabled: true, bevelThickness: 0.035, bevelSize: 0.03, bevelSegments: 5, curveSegments: 24 })
  earGeo.translate(0, -0.12, -0.035)
  const earMat = plastic()
  const ears = [-1, 1].map((s) => {
    const m = new THREE.Mesh(earGeo, earMat)
    m.castShadow = true
    m.userData.s = s
    return m
  })
  const root = new THREE.Group()
  root.add(mesh, ...ears)
  let cutY = -0.9
  let jawY = -1.6
  // Straight across under the nose, then down the cheek flaps to the jaw.
  const lowerEdge = (x: number) => {
    const ax = Math.abs(x)
    const drop = THREE.MathUtils.smoothstep(ax, 0.5, 0.78)
    return THREE.MathUtils.lerp(cutY, jawY, drop)
  }
  const shape: MaskShape = {
    // Sculpted: an angled V brow ridge with a furrow at its centre,
    // cheekbone ridges and a moulded nose guard (all smooth functions, so
    // the surface stays continuous).
    lift: (p) => {
      const ax = Math.abs(p.x)
      const brow = 0.07 * Math.exp(-(((p.y - (0.16 + 0.32 * ax)) / 0.09) ** 2)) * THREE.MathUtils.smoothstep(ax, 0.04, 0.16) * (1 - THREE.MathUtils.smoothstep(ax, 0.7, 0.95))
      const furrow = 0.035 * Math.exp(-((p.x / 0.05) ** 2)) * Math.exp(-(((p.y - 0.3) / 0.18) ** 2))
      const cheek = 0.035 * Math.exp(-(((ax - 0.58) / 0.16) ** 2) - (((p.y + 0.48) / 0.2) ** 2))
      const nose = 0.035 * Math.exp(-((p.x / 0.16) ** 2)) * THREE.MathUtils.smoothstep(-p.y, -0.15, 0.05)
      return 0.05 + brow + furrow + cheek + nose
    },
    hair: 0.4,
    cut: (p, eyes) => {
      const below = (p.y - lowerEdge(p.x)) * 6
      // Big almond openings, outer corners swept up.
      const hole = Math.min(
        ...eyes.map((e) => {
          const side = Math.sign(e.c.x) || 1
          const dx = p.x - e.c.x
          const dy = p.y - e.c.y
          const a = -side * 0.22
          const u = (dx * Math.cos(a) - dy * Math.sin(a)) / (e.rx * 1.18)
          const v = (dx * Math.sin(a) + dy * Math.cos(a)) / (e.ry * 1.25)
          return Math.pow(Math.abs(u), 1.7) + Math.pow(Math.abs(v), 2.2) - 1
        }),
      )
      return Math.min(below, hole)
    },
  }
  return {
    root,
    update(rig) {
      cutY = rig.local(2).y - 0.04
      jawY = rig.local(152).y + 0.45
      updateMask(geo, rig, shape)
      const top = rig.local(10)
      const tl = rig.local(21)
      const tr = rig.local(251)
      for (const e of ears) {
        const temple = e.userData.s < 0 ? (tl.x < tr.x ? tl : tr) : tl.x < tr.x ? tr : tl
        e.position.copy(top).lerp(temple, 0.66).add(new THREE.Vector3(e.userData.s * 0.08, 0.62, -0.36))
        // Broad side to the front, as on the real cowl; a slight outward lean.
        e.rotation.set(-0.12, e.userData.s * 0.12, e.userData.s * -0.12)
      }
    },
  }
}
