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
  /** Laplacian smoothing passes over the face before lifting: a moulded
   *  shell shouldn't follow every contour of the face underneath it. */
  smooth?: number
}

let neighbours: number[][] | null = null
function faceNeighbours() {
  if (neighbours) return neighbours
  const sets = Array.from({ length: N }, () => new Set<number>())
  for (let t = 0; t < FACE_TRIANGULATION.length; t += 3) {
    const [a, b, c] = [FACE_TRIANGULATION[t], FACE_TRIANGULATION[t + 1], FACE_TRIANGULATION[t + 2]]
    sets[a].add(b).add(c)
    sets[b].add(a).add(c)
    sets[c].add(a).add(b)
  }
  neighbours = sets.map((s) => [...s])
  return neighbours
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
  g.setAttribute('aLocal', new THREE.BufferAttribute(new Float32Array(VERTS * 3), 3))
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

// Eye masks are cut out of the full mask surface per pixel: each fragment
// gets its face-space position (vLocal) and a GLSL "keep" function decides
// (negative = cut away), so the outline and eye openings are exact smooth
// curves rather than following the face mesh's coarse triangles.
const CUT_GLSL = `
varying vec3 vLocal;
uniform vec4 uEyeA;
uniform vec4 uEyeB;
uniform float uEyeY;
// A band across the eyes between top and bottom (relative to eye height),
// narrowing into ties past the temples, ending before the back of the head.
float domino(vec3 p, float top, float bottom) {
  float ax = abs(p.x);
  float taper = max(0.0, ax - 0.95) * 0.55 + max(0.0, -0.2 - p.z) * 0.22;
  return min(min(uEyeY + top - taper - p.y, p.y - (uEyeY + bottom + taper)), p.z + 1.05);
}
// Almond opening around an eye (x, y, rx, ry), outer corner swept up.
float almond(vec3 p, vec4 e) {
  float side = e.x < 0.0 ? -1.0 : 1.0;
  vec2 d = p.xy - e.xy;
  float a = -side * 0.22;
  float u = (d.x * cos(a) - d.y * sin(a)) / (e.z * 1.15);
  float v = (d.x * sin(a) + d.y * cos(a)) / (e.w * 1.3);
  return (pow(abs(u), 1.7) + pow(abs(v), 2.2) - 1.0) * e.z * 0.5;
}
`

interface CutUniforms {
  uEyeA: { value: THREE.Vector4 }
  uEyeB: { value: THREE.Vector4 }
  uEyeY: { value: number }
}

/** Adds a per-pixel cut (`keep`: GLSL body of float maskKeep(vec3 p)). */
function withCut<T extends THREE.Material>(m: T, name: string, keep: string): { material: T; uniforms: CutUniforms } {
  const uniforms: CutUniforms = { uEyeA: { value: new THREE.Vector4() }, uEyeB: { value: new THREE.Vector4() }, uEyeY: { value: 0 } }
  // Antialiased: coverage ramps over one pixel of the cut function, resolved
  // by alpha-to-coverage instead of a hard, stair-stepped discard.
  m.alphaToCoverage = true
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms)
    shader.vertexShader = 'attribute vec3 aLocal;\nvarying vec3 vLocal;\n' + shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n  vLocal = aLocal;')
    shader.fragmentShader =
      CUT_GLSL +
      `float maskKeep(vec3 p) {\n${keep}\n}\n` +
      shader.fragmentShader
        .replace('void main() {', 'void main() {\n  float keepV = maskKeep(vLocal);\n  float cutA = clamp(keepV / max(fwidth(keepV), 1e-5) + 0.5, 0.0, 1.0);\n  if (cutA <= 0.0) discard;')
        .replace('#include <dithering_fragment>', '#include <dithering_fragment>\n  gl_FragColor.a *= cutA;')
  }
  m.customProgramCacheKey = () => 'mask-cut-' + name
  return { material: m, uniforms }
}

function setEyes(u: CutUniforms, eyes: EyeHole[]) {
  const [a, b] = eyes
  u.uEyeA.value.set(a.c.x, a.c.y, a.rx, a.ry)
  u.uEyeB.value.set(b.c.x, b.c.y, b.rx, b.ry)
  u.uEyeY.value = (a.c.y + b.c.y) / 2
}

/** Rebuilds the mask surface from this frame's landmarks. */
function updateMask(g: THREE.BufferGeometry, rig: Rig, shape: MaskShape): EyeHole[] {
  const raw: THREE.Vector3[] = Array.from({ length: N }, (_, i) => rig.local(i))
  let L = raw
  if (shape.smooth) {
    const nb = faceNeighbours()
    for (let it = 0; it < shape.smooth; it++) {
      L = L.map((p, i) => {
        const avg = nb[i].reduce((s, j) => s.add(L[j]), new THREE.Vector3()).divideScalar(nb[i].length || 1)
        return p.clone().lerp(avg, 0.5)
      })
    }
  }
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
  // Smoothing shrinks convex parts (forehead, cheekbones, nose) below the
  // real skin, which would then poke through the shell. Push the smoothed
  // surface back out by however far the face sits above it, spread over a
  // couple of rings so the correction itself stays smooth.
  let push = new Float32Array(N)
  if (L !== raw) {
    for (let i = 0; i < N; i++) push[i] = Math.max(0, e1.subVectors(raw[i], L[i]).dot(nrm[i]))
    const nb = faceNeighbours()
    for (let it = 0; it < 3; it++) {
      const next = new Float32Array(N)
      for (let i = 0; i < N; i++) {
        let m = push[i]
        let s = push[i]
        for (const j of nb[i]) {
          m = Math.max(m, push[j])
          s += push[j]
        }
        next[i] = it < 2 ? m : s / (nb[i].length + 1)
      }
      push = next
    }
  }
  const P: THREE.Vector3[] = []
  for (let i = 0; i < N; i++) P.push(L[i].clone().addScaledVector(nrm[i], shape.lift(L[i]) + push[i]))

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
  const local = g.getAttribute('aLocal') as THREE.BufferAttribute
  P.forEach((p, i) => {
    pos.setXYZ(i, p.x, p.y, p.z)
    const [u, v] = uvOf(p)
    uv.setXY(i, u, v)
    const q = i < N ? L[i] : p
    local.setXYZ(i, q.x, q.y, q.z)
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
  local.needsUpdate = true
  nAttr.needsUpdate = true
  g.computeBoundingSphere()
  return eyes
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
  const { material, uniforms } = withCut(
    new THREE.MeshPhysicalMaterial({ map, normalMap: normal, normalScale: new THREE.Vector2(1.1, 1.1), roughness: 0.62, sheen: 0.7, sheenRoughness: 0.45, sheenColor: new THREE.Color(0xff5a5a), side: THREE.DoubleSide }),
    'spider',
    `float ax = abs(p.x);
    float top = 0.44 - 0.07 * exp(-pow(ax / 0.2, 2.0));
    float bottom = -0.34 + 0.12 * exp(-pow(ax / 0.15, 2.0)) + 0.06 * smoothstep(0.55, 0.9, ax);
    return domino(p, top, bottom);`,
  )
  const mesh = new THREE.Mesh(geo, material)
  mesh.frustumCulled = false
  const lenses = [spiderLens(-1), spiderLens(1)]
  const root = new THREE.Group()
  root.add(mesh, ...lenses)
  // A stretch-fabric eye mask: it bridges the eye sockets (smoothed) and
  // the white lenses sit on it, covering the eyes.
  const shape: MaskShape = {
    lift: () => 0.035,
    hair: 0,
    smooth: 4,
  }
  return {
    root,
    update(rig) {
      setEyes(uniforms, updateMask(geo, rig, shape))
      const eyes = [LEFT_EYE, RIGHT_EYE].map((loop) => loop.reduce((s, i) => s.add(rig.local(i)), new THREE.Vector3()).divideScalar(loop.length))
      eyes.sort((a, b) => a.x - b.x)
      eyes.forEach((e, k) => {
        const side = k ? 1 : -1
        lenses[k].position.set(e.x + side * 0.04, e.y + 0.05, e.z + 0.17)
        lenses[k].rotation.set(-0.05, side * 0.32, side * 0.12)
      })
    },
  }
}

// ---- Bat ----------------------------------------------------------------------

export function batCowl(): Model {
  const geo = maskGeometry()
  // Satin-finish moulded black: soft, broad highlights rather than a
  // pin-point CG glint.
  const { material, uniforms } = withCut(
    new THREE.MeshPhysicalMaterial({ color: 0x101319, roughness: 0.62, metalness: 0, clearcoat: 0.2, clearcoatRoughness: 0.45, envMapIntensity: 0.7, side: THREE.DoubleSide }),
    'bat',
    // Top edge rising into two pointed bat ears over the brows and sweeping
    // up at the outer corners; almond eye openings.
    `float ax = abs(p.x);
    float ear = pow(max(0.0, 1.0 - abs(ax - 0.42) / 0.12), 1.5) * 0.34;
    float top = 0.32 + 0.2 * smoothstep(0.5, 0.95, ax) - 0.05 * exp(-pow(ax / 0.18, 2.0)) + ear;
    float bottom = -0.3 + 0.13 * exp(-pow(ax / 0.15, 2.0)) - 0.06 * smoothstep(0.55, 0.85, ax);
    return min(domino(p, top, bottom), min(almond(p, uEyeA), almond(p, uEyeB)));`,
  )
  const mesh = new THREE.Mesh(geo, material)
  mesh.frustumCulled = false
  const root = new THREE.Group()
  root.add(mesh)
  const shape: MaskShape = {
    lift: (p) => {
      const ax = Math.abs(p.x)
      const brow = 0.04 * Math.exp(-(((p.y - (0.16 + 0.25 * ax)) / 0.1) ** 2)) * THREE.MathUtils.smoothstep(ax, 0.04, 0.16)
      return 0.05 + brow
    },
    hair: 0,
    smooth: 6,
  }
  return {
    root,
    update(rig) {
      setEyes(uniforms, updateMask(geo, rig, shape))
    },
  }
}
