import * as THREE from 'three'
import { FaceLandmarker } from '@mediapipe/tasks-vision'
import { FACE_TRIANGULATION } from '../faceTriangulation'
import { connectorsToLoop } from '../landmarks'
import { compositeAR } from './composite'

// The 3D layer behind every modeled AR effect (fur ears, masks, glasses,
// crown…). One three.js renderer, reused for every frame and photo.
//
// Space: an orthographic camera whose units are the photo's own pixels
// (X right, Y up, Z toward the viewer), so MediaPipe's landmarks — x/y in
// the image, z a relative depth in the same scale as x — drop straight in
// and the 3D face mesh lines up with the photo exactly, no fitting.
//
// Each effect is authored in a head-local "rig" frame: origin between the
// eyes, 1 unit = the distance between the eye centers, x toward image
// right, y up the face, z out of the face. The rig is rebuilt every frame
// from the 3D landmarks, so models follow yaw, pitch and roll.
//
// Realism comes from the photo itself:
// • Occlusion — the real face mesh plus an ellipsoid stand-in for the rest
//   of the head write depth only, so a glasses arm disappears behind the
//   temple, ear bases sink into the hair, a halo's back half goes behind.
// • Lighting — the key light's direction and strength are estimated from
//   how the face is shaded (left vs right cheek, forehead vs chin), the
//   ambient light from the frame's own top/bottom colors, and reflections
//   from an environment map made of the frame itself.
// • Shadows — props cast soft shadows onto the face mesh, which renders as
//   a shadow-only catcher over the photo.

export interface P3 {
  x: number
  y: number
  z: number
}

export interface Rig {
  /** World (px) → rig and back. */
  matrix: THREE.Matrix4
  inverse: THREE.Matrix4
  E: number
  /** Landmark i in rig units. */
  local: (i: number) => THREE.Vector3
  world: THREE.Vector3[]
  /** Lip gap relative to mouth width. */
  mouthOpen: number
  live: boolean
}

export interface Model {
  root: THREE.Object3D
  update?: (rig: Rig, t: number) => void
}

interface State {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.OrthographicCamera
  key: THREE.DirectionalLight
  hemi: THREE.HemisphereLight
  rig: THREE.Group
  faceGeo: THREE.BufferGeometry
  headGeo: THREE.BufferGeometry
  pmrem: THREE.PMREMGenerator
  env: THREE.WebGLRenderTarget | null
  envTex: THREE.CanvasTexture
  envCanvas: HTMLCanvasElement
  envFrame: number
  sample: CanvasRenderingContext2D
  models: Map<string, Model>
}

let S: State | null = null

function init(): State {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: false })
  renderer.setPixelRatio(1)
  renderer.setClearColor(0x000000, 0)
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.05
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap

  const scene = new THREE.Scene()
  const camera = new THREE.OrthographicCamera(0, 1, 0, -1, -20000, 20000)
  camera.position.set(0, 0, 10000)

  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.7)
  scene.add(hemi)
  const key = new THREE.DirectionalLight(0xffffff, 2.2)
  key.castShadow = true
  key.shadow.mapSize.set(2048, 2048)
  key.shadow.radius = 7
  scene.add(key, key.target)

  // Face occluder (depth only) + shadow catcher, both on the live mesh.
  const faceGeo = new THREE.BufferGeometry()
  faceGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(468 * 3), 3))
  faceGeo.setIndex(Array.from(FACE_TRIANGULATION))
  const occluder = new THREE.Mesh(faceGeo, new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.DoubleSide }))
  occluder.renderOrder = -10
  occluder.frustumCulled = false
  const catcher = new THREE.Mesh(
    faceGeo,
    new THREE.ShadowMaterial({ opacity: 0.26, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8, side: THREE.DoubleSide }),
  )
  catcher.receiveShadow = true
  catcher.frustumCulled = false
  scene.add(occluder, catcher)

  const rig = new THREE.Group()
  rig.matrixAutoUpdate = false
  scene.add(rig)
  // The rest of the head (depth only): skull rings swept back from the
  // face outline, closed at the back. Rebuilt each frame from this face, so
  // unlike a stock ellipsoid it can never poke out in front of the face.
  const headGeo = new THREE.BufferGeometry()
  headGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((OVAL.length * (HEAD_RINGS + 1) + 1) * 3), 3))
  const hIdx: number[] = []
  const ringIdx = (k: number, o: number) => k * OVAL.length + (o % OVAL.length)
  for (let k = 0; k < HEAD_RINGS; k++)
    for (let o = 0; o < OVAL.length; o++) hIdx.push(ringIdx(k, o), ringIdx(k + 1, o), ringIdx(k, o + 1), ringIdx(k, o + 1), ringIdx(k + 1, o), ringIdx(k + 1, o + 1))
  const back = OVAL.length * (HEAD_RINGS + 1)
  for (let o = 0; o < OVAL.length; o++) hIdx.push(ringIdx(HEAD_RINGS, o), back, ringIdx(HEAD_RINGS, o + 1))
  headGeo.setIndex(hIdx)
  const head = new THREE.Mesh(headGeo, new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.DoubleSide }))
  head.renderOrder = -10
  head.frustumCulled = false
  // …and catches shadows, so ears and headbands sit on the head rather
  // than float over it.
  const headCatch = new THREE.Mesh(
    headGeo,
    new THREE.ShadowMaterial({ opacity: 0.34, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8, side: THREE.DoubleSide }),
  )
  headCatch.receiveShadow = true
  headCatch.frustumCulled = false
  rig.add(head, headCatch)

  const envCanvas = document.createElement('canvas')
  envCanvas.width = 256
  envCanvas.height = 128
  const envTex = new THREE.CanvasTexture(envCanvas)
  envTex.mapping = THREE.EquirectangularReflectionMapping
  envTex.colorSpace = THREE.SRGBColorSpace
  const sampleCanvas = document.createElement('canvas')
  sampleCanvas.width = 64
  sampleCanvas.height = 64

  return {
    renderer,
    scene,
    camera,
    key,
    hemi,
    rig,
    faceGeo,
    headGeo,
    pmrem: new THREE.PMREMGenerator(renderer),
    env: null,
    envTex,
    envCanvas,
    envFrame: 0,
    sample: sampleCanvas.getContext('2d', { willReadFrequently: true })!,
    models: new Map(),
  }
}

export const OVAL = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
export const HEAD_RINGS = 10

/** The skull behind the face, as rings swept back from the face outline
 *  (rig units): each oval point moves outward a little — more over the
 *  crown — and back around the head. `lift` pushes everything outward
 *  (masks sit on top of the head this describes). Ring 0 is the outline
 *  itself. */
export function headRing(o: THREE.Vector3, k: number, lift = 0, hair = 0): THREE.Vector3 {
  const dir = new THREE.Vector3(o.x, o.y + 0.15, 0).normalize()
  const up = Math.max(0, dir.y)
  const s = k / HEAD_RINGS
  // `hair`: extra room over the crown for something worn over the hair.
  // The hair allowance eases in (flat at the face outline) so a hood flows
  // up from the forehead instead of stepping out at the hairline.
  const out = (0.08 + 0.45 * up * up) * Math.sin((s * Math.PI) / 2) + hair * up * up * (1 - Math.cos(s * Math.PI)) * 0.5 + lift
  // A quarter-ellipse profile: the surface first continues up/out from the
  // face (no fold at the hairline), then curves back around the skull.
  const back = 1.35 * (1 - Math.cos((s * Math.PI) / 2))
  return new THREE.Vector3(o.x + dir.x * out, o.y + dir.y * out, o.z - back - lift * 0.3)
}
export const HEAD_BACK = new THREE.Vector3(0, 0.2, -2.35)

const LEFT_EYE = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
const RIGHT_EYE = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)

function centroid(idx: number[], P: THREE.Vector3[]) {
  const c = new THREE.Vector3()
  for (const i of idx) c.add(P[i])
  return c.divideScalar(idx.length)
}

function buildRig(P: THREE.Vector3[], live: boolean): Rig {
  let eyeL = centroid(LEFT_EYE, P)
  let eyeR = centroid(RIGHT_EYE, P)
  if (eyeL.x > eyeR.x) [eyeL, eyeR] = [eyeR, eyeL]
  const E = Math.max(1, eyeL.distanceTo(eyeR))
  const right = eyeR.clone().sub(eyeL).normalize()
  const upRaw = P[10].clone().sub(P[152]).normalize()
  const fwd = new THREE.Vector3().crossVectors(right, upRaw).normalize()
  const up = new THREE.Vector3().crossVectors(fwd, right).normalize()
  const matrix = new THREE.Matrix4().makeBasis(right.multiplyScalar(E), up.multiplyScalar(E), fwd.multiplyScalar(E)).setPosition(P[168])
  const inverse = matrix.clone().invert()
  const mouthW = Math.max(1, P[61].distanceTo(P[291]))
  return {
    matrix,
    inverse,
    E,
    local: (i) => P[i].clone().applyMatrix4(inverse),
    world: P,
    mouthOpen: P[13].distanceTo(P[14]) / mouthW,
    live,
  }
}

// Light from the photo: sample the face's shading at cheek, forehead and
// chin points on a small copy of the frame.
function estimateLight(st: State, frame: HTMLCanvasElement, P: THREE.Vector3[], rig: Rig) {
  const W = frame.width
  const H = frame.height
  const s = st.sample
  s.drawImage(frame, 0, 0, 64, 64)
  const img = s.getImageData(0, 0, 64, 64).data
  const lum = (p: THREE.Vector3) => {
    const x = Math.min(63, Math.max(0, Math.round((p.x / W) * 64)))
    const y = Math.min(63, Math.max(0, Math.round((-p.y / H) * 64)))
    let sum = 0
    let n = 0
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const xx = Math.min(63, Math.max(0, x + dx))
        const yy = Math.min(63, Math.max(0, y + dy))
        const k = (yy * 64 + xx) * 4
        sum += (0.3 * img[k] + 0.59 * img[k + 1] + 0.11 * img[k + 2]) / 255
        n++
      }
    return sum / n
  }
  const [cA, cB] = [P[50], P[280]]
  const [cL, cR] = cA.x < cB.x ? [cA, cB] : [cB, cA]
  const lL = lum(cL)
  const lR = lum(cR)
  const lT = lum(P[151])
  const lB = lum(P[199])
  const mean = (lL + lR + lT + lB) / 4
  const dx = (lR - lL) / Math.max(0.05, lR + lL)
  const dy = (lT - lB) / Math.max(0.05, lT + lB)
  // Kept fairly frontal: a phone selfie's light is mostly in front, and a
  // grazing estimate would throw long, unconvincing shadows across the face.
  const dir = new THREE.Vector3(THREE.MathUtils.clamp(dx * 2, -0.6, 0.6), THREE.MathUtils.clamp(dy * 2, -0.3, 0.6) + 0.3, 1.25).normalize()
  const head = new THREE.Vector3(0, 0.3, -1).applyMatrix4(rig.matrix)
  st.key.target.position.copy(head)
  st.key.position.copy(head).addScaledVector(dir, rig.E * 12)
  st.key.intensity = THREE.MathUtils.clamp(2.6 * (mean / 0.5), 1.2, 3.4)
  // Tint the key toward the face's average highlight colour.
  const avg = [0, 0, 0]
  for (let k = 0; k < img.length; k += 4) {
    avg[0] += img[k]
    avg[1] += img[k + 1]
    avg[2] += img[k + 2]
  }
  const m = Math.max(avg[0], avg[1], avg[2]) || 1
  st.key.color.setRGB(0.75 + 0.25 * (avg[0] / m), 0.75 + 0.25 * (avg[1] / m), 0.75 + 0.25 * (avg[2] / m), THREE.SRGBColorSpace)
  const band = (y0: number, y1: number) => {
    const c = [0, 0, 0]
    let n = 0
    for (let y = y0; y < y1; y++)
      for (let x = 0; x < 64; x++) {
        const k = (y * 64 + x) * 4
        c[0] += img[k]
        c[1] += img[k + 1]
        c[2] += img[k + 2]
        n++
      }
    return new THREE.Color().setRGB(c[0] / n / 255, c[1] / n / 255, c[2] / n / 255, THREE.SRGBColorSpace)
  }
  st.hemi.color.copy(band(0, 20)).lerp(new THREE.Color(1, 1, 1), 0.45)
  st.hemi.groundColor.copy(band(44, 64)).multiplyScalar(0.8)
  st.hemi.intensity = 0.55 + mean * 0.9

  const sc = st.key.shadow.camera
  const r = rig.E * 3.2
  sc.left = -r
  sc.right = r
  sc.top = r
  sc.bottom = -r
  sc.near = 1
  sc.far = rig.E * 30
  sc.updateProjectionMatrix()
  st.key.shadow.bias = -0.0006
  st.key.shadow.normalBias = rig.E * 0.01
}

// Reflections: the frame itself, softened, as an equirect environment.
function updateEnvironment(st: State, frame: HTMLCanvasElement, force: boolean) {
  if (!force && st.envFrame++ % 45 !== 0 && st.env) return
  const ctx = st.envCanvas.getContext('2d')!
  // Mirror the frame side-by-side so the wrap-around seam isn't a hard edge.
  // Only the room's colours should reflect, never an image of it — even a
  // tiny copy of the frame keeps its dark/bright layout, which on a glossy
  // dark surface reads as see-through. So the environment is a smooth
  // vertical gradient of the frame's top, middle and bottom colours, plus a
  // soft overhead softbox and two dim side fills for highlights.
  const s2 = st.sample
  s2.drawImage(frame, 0, 0, 64, 64)
  const px = s2.getImageData(0, 0, 64, 64).data
  const band = (y0: number, y1: number) => {
    let r = 0, g = 0, b = 0, n = 0
    for (let y = y0; y < y1; y++)
      for (let x = 0; x < 64; x++) {
        const k = (y * 64 + x) * 4
        r += px[k]
        g += px[k + 1]
        b += px[k + 2]
        n++
      }
    return `rgb(${Math.round(r / n)},${Math.round(g / n)},${Math.round(b / n)})`
  }
  const grad = ctx.createLinearGradient(0, 0, 0, 128)
  grad.addColorStop(0, band(0, 16))
  grad.addColorStop(0.5, band(24, 40))
  grad.addColorStop(1, band(48, 64))
  ctx.globalCompositeOperation = 'source-over'
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, 256, 128)
  ctx.save()
  ctx.globalCompositeOperation = 'screen'
  const soft = (x: number, y: number, r: number, a: number) => {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r)
    g.addColorStop(0, `rgba(255,250,240,${a})`)
    g.addColorStop(1, 'rgba(255,250,240,0)')
    ctx.fillStyle = g
    ctx.fillRect(0, 0, 256, 128)
  }
  soft(128, 26, 34, 0.75)
  soft(40, 60, 26, 0.25)
  soft(216, 60, 26, 0.25)
  ctx.restore()
  st.envTex.needsUpdate = true
  st.env?.dispose()
  st.env = st.pmrem.fromEquirectangular(st.envTex)
  st.scene.environment = st.env.texture
  st.scene.environmentIntensity = 0.9
}

/** Renders the 3D effect over `frame` (in place). `P` are landmark pixel
 *  positions with z in pixels (MediaPipe z × width, negative = nearer). */
export function renderAR(frame: HTMLCanvasElement, P2: P3[], effectId: string, build: () => Model, t: number, live: boolean) {
  S ??= init()
  const st = S
  const W = frame.width
  const H = frame.height
  // Rendered a little under full size and scaled up when composited: a
  // phone photo never resolves edges as crisply as a clean render does.
  const RS = 0.8
  const rw = Math.round(W * RS)
  const rh = Math.round(H * RS)
  if (st.renderer.domElement.width !== rw || st.renderer.domElement.height !== rh) st.renderer.setSize(rw, rh, false)
  st.camera.left = 0
  st.camera.right = W
  st.camera.top = 0
  st.camera.bottom = -H
  st.camera.updateProjectionMatrix()

  const P = P2.map((p) => new THREE.Vector3(p.x, -p.y, -p.z))
  const pos = st.faceGeo.getAttribute('position') as THREE.BufferAttribute
  for (let i = 0; i < 468; i++) pos.setXYZ(i, P[i].x, P[i].y, P[i].z)
  pos.needsUpdate = true
  st.faceGeo.computeBoundingSphere()

  const rig = buildRig(P, live)
  st.rig.matrix.copy(rig.matrix)
  st.rig.matrixWorldNeedsUpdate = true
  const hp = st.headGeo.getAttribute('position') as THREE.BufferAttribute
  const ovalLocal = OVAL.map((i) => rig.local(i))
  for (let k = 0; k <= HEAD_RINGS; k++)
    ovalLocal.forEach((o, j) => {
      const q = headRing(o, k)
      hp.setXYZ(k * OVAL.length + j, q.x, q.y, q.z)
    })
  hp.setXYZ(OVAL.length * (HEAD_RINGS + 1), HEAD_BACK.x, HEAD_BACK.y, HEAD_BACK.z)
  hp.needsUpdate = true

  let model = st.models.get(effectId)
  if (!model) {
    model = build()
    st.models.set(effectId, model)
    st.rig.add(model.root)
  }
  for (const [id, m] of st.models) m.root.visible = id === effectId
  model.update?.(rig, t)

  estimateLight(st, frame, P, rig)
  updateEnvironment(st, frame, !live)
  st.renderer.render(st.scene, st.camera)
  // Composite over a box around the head (ears, halo and hood included),
  // matched to the photo's tones and grain — see composite.ts.
  const c = new THREE.Vector3(0, 0.4, -0.6).applyMatrix4(rig.matrix)
  const r = rig.E * 3.4
  const x0 = Math.max(0, Math.floor(c.x - r))
  const y0 = Math.max(0, Math.floor(-c.y - r * 1.15))
  const x1 = Math.min(W, Math.ceil(c.x + r))
  const y1 = Math.min(H, Math.ceil(-c.y + r))
  compositeAR(frame, st.renderer.domElement, RS, { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }, live)
}
