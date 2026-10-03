import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { connectorsToLoop } from './landmarks'
import { FACE_TRIANGULATION } from './faceTriangulation'

// Why this replaced the old CPU per-pixel warp (MLS + radial zoom):
//
// That approach computed a deformation FIELD numerically — weighted by
// distance to a handful of control points — and however carefully the
// control points and anchor rings were tuned, the field between them had
// no hard guarantee of staying well-behaved. Three rounds of real-device
// testing kept finding new ways it broke (wavy backgrounds, eyes folding
// into solid-color smears, and still-reported distortion after both
// fixes on an extreme, close, poorly-lit angle where MediaPipe's own
// landmarks are noisier than on a calm test photo).
//
// A triangulated mesh warp — what Snapchat/Meitu/Instagram-style filters
// actually use — sidesteps this by construction instead of by tuning.
// Each of MediaPipe's 468 landmarks is a vertex in a fixed triangle mesh
// (FACE_TRIANGULATION, the published reference topology); warping is just
// moving vertices, and the GPU fills each triangle's interior with
// ordinary texture-mapped interpolation. A triangle can't "fold into a
// smear" the way a global point-cloud field can — it would have to
// literally invert (a vertex crossing through the opposite edge), which
// the displacement amounts here never come close to. And since the
// un-warped background is a completely separate full-image layer drawn
// first, with the face mesh (plus a small anchored "skirt" ring, see
// below) drawn on top of it, the background is never touched by the warp
// math at all — not "moved so little it's not visible," but literally
// never read as input to any displacement calculation.

const SKIRT_PAD_FRACTION = 0.28 // how far outside the oval the anchor ring sits, as a fraction of the oval's own radius at each point

const VERTEX_SRC = `
  attribute vec2 a_texCoord;
  attribute vec2 a_targetPos;
  attribute vec3 a_normal;
  varying vec2 v_texCoord;
  varying vec3 v_normal;
  void main() {
    v_texCoord = a_texCoord;
    v_normal = a_normal;
    vec2 clip = a_targetPos * 2.0 - 1.0;
    gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  }
`

// Single-light Lambertian + a touch of Blinn-Phong specular, driven by
// per-vertex normals derived from the face mesh's own 3D landmark
// geometry (see computeVertexNormals) and interpolated across each
// triangle by the rasterizer — real directional shading, not a 2D
// brightness map. `u_lightIntensity` is 0 for the background pass (see
// renderMeshWarp), which is what keeps the background completely unlit
// regardless of whatever normal data happens to be bound for that draw.
//
// The warm "terminator" term approximates subsurface scattering: real
// skin is translucent, so light grazing the boundary between a lit and
// shadowed area scatters inside the skin and re-emerges with a reddish
// cast, which is what keeps that transition from reading as a flat,
// plastic falloff. Applied only in a narrow band around n.l ~ 0 (the
// actual terminator line), not as a general warm tint.
const FRAGMENT_SRC = `
  precision mediump float;
  varying vec2 v_texCoord;
  varying vec3 v_normal;
  uniform sampler2D u_image;
  uniform vec3 u_lightDir;
  uniform float u_lightIntensity;
  void main() {
    vec4 color = texture2D(u_image, v_texCoord);
    vec3 n = length(v_normal) > 0.0001 ? normalize(v_normal) : vec3(0.0);
    float ndotl = max(dot(n, u_lightDir), 0.0);

    // Screen-blended lift (not an additive brighten) so a fully-lit
    // surface compresses toward white instead of blowing straight through
    // it, and an already-bright highlight moves less than a midtone does.
    float lift = ndotl * u_lightIntensity * 0.55;
    vec3 lit = 1.0 - (1.0 - color.rgb) * (1.0 - lift);

    // A modest Blinn-Phong specular highlight — the "water-light skin"
    // touch — using the same light direction and a camera-facing view
    // vector (the mesh is rendered near-orthographically, so (0,0,1) is a
    // reasonable stand-in for the eye direction here).
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    vec3 halfVec = normalize(u_lightDir + viewDir);
    float spec = pow(max(dot(n, halfVec), 0.0), 24.0);
    lit += vec3(spec) * u_lightIntensity * 0.12;

    float terminator = 1.0 - smoothstep(0.0, 0.3, abs(ndotl - 0.12));
    vec3 warm = vec3(0.14, 0.035, -0.02);
    lit += warm * terminator * u_lightIntensity * 0.5;

    gl_FragColor = vec4(clamp(lit, 0.0, 1.0), color.a);
  }
`

interface GLState {
  canvas: HTMLCanvasElement
  gl: WebGLRenderingContext
  program: WebGLProgram
  texCoordBuffer: WebGLBuffer
  targetPosBuffer: WebGLBuffer
  normalBuffer: WebGLBuffer
  indexBuffer: WebGLBuffer
  texCoordLoc: number
  targetPosLoc: number
  normalLoc: number
  texture: WebGLTexture
  backgroundIndexBuffer: WebGLBuffer
}

let state: GLState | null = null

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader)
    gl.deleteShader(shader)
    throw new Error('Shader compile failed: ' + info)
  }
  return shader
}

function getState(): GLState {
  if (state) return state
  const canvas = document.createElement('canvas')
  const gl = canvas.getContext('webgl', { premultipliedAlpha: false })
  if (!gl) throw new Error('WebGL unavailable')

  const program = gl.createProgram()!
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, VERTEX_SRC))
  gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SRC))
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error('Program link failed: ' + gl.getProgramInfoLog(program))
  }

  const texture = gl.createTexture()!
  gl.bindTexture(gl.TEXTURE_2D, texture)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  // Deliberately NOT setting UNPACK_FLIP_Y_WEBGL: texImage2D uploads a
  // canvas's rows in the order they're stored (row 0 = canvas top) with no
  // flip, so texCoord v=0 already samples the canvas's top row — exactly
  // matching MediaPipe's normalized-landmark convention (y=0 is the top of
  // the image), with nothing extra to correct. Flipping here would only
  // reintroduce a mismatch the vertex shader would then have to undo.

  // The background quad (two triangles covering the full frame, identity
  // mapping from texCoord to position) is static — build its index buffer
  // once. Its own texCoord/position vertices are written fresh each call
  // since they're trivial (just the four corners).
  const backgroundIndexBuffer = gl.createBuffer()!
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, backgroundIndexBuffer)
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 2, 1, 3]), gl.STATIC_DRAW)

  state = {
    canvas,
    gl,
    program,
    texCoordBuffer: gl.createBuffer()!,
    targetPosBuffer: gl.createBuffer()!,
    normalBuffer: gl.createBuffer()!,
    indexBuffer: gl.createBuffer()!,
    texCoordLoc: gl.getAttribLocation(program, 'a_texCoord'),
    targetPosLoc: gl.getAttribLocation(program, 'a_targetPos'),
    normalLoc: gl.getAttribLocation(program, 'a_normal'),
    texture,
    backgroundIndexBuffer,
  }
  return state
}

// Every field is bidirectional: negative/positive move the feature the two
// opposite ways, 0 is the untouched original — a drag-bar-from-the-middle
// control, not a one-directional intensity slider. All but `nose` are
// clamped to [-1, 1] right where they're consumed below.
export interface ReshapeParams {
  /** Jaw/cheek: negative widens, positive narrows. */
  face: number
  /** Eyes: negative shrinks, positive enlarges. */
  eyes: number
  /** Nose: negative widens, positive narrows (handled separately — see reshape.ts). */
  nose: number
  /** Mouth: negative shrinks, positive enlarges. */
  mouth: number
  /** Eyebrows: negative lowers, positive raises. */
  eyebrowHeight: number
  /** Nose bridge (山根): negative flattens, positive raises/sharpens. */
  noseBridge: number
  /** Temple (太阳穴/颞区): negative widens, positive narrows. */
  temple: number
  /** Cheekbone (颧骨): negative widens, positive narrows. */
  cheekbone: number
  /** Upper lip thickness: negative thins, positive thickens. */
  mouthUpperLip: number
  /** Lower lip thickness: negative thins, positive thickens. */
  mouthLowerLip: number
  /** Mouth corners: negative downturns, positive lifts. */
  mouthCorners: number
  /** 3D relighting intensity, 0-1 — not bidirectional like the rest (see MESH_FIELDS in reshape.ts, which this is also added to so the mesh pass still runs when this is the only active field). */
  fillLight: number
}

const OVAL_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
const LEFT_EYE_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
const RIGHT_EYE_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)
const LIPS_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LIPS)
const LEFT_EYEBROW_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW)
const RIGHT_EYEBROW_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW)
const LEFT_CHEEK = 234
const RIGHT_CHEEK = 454
// Nose bridge (山根): the well-known vertical nose centerline is landmarks
// 6 -> 197 -> 195 -> 5 -> 4 (bridge to tip) in every public MediaPipe face
// mesh reference; 6 sits right at the bridge itself. Verified by rendering
// all 468 indices on a real photo and visually confirming 6's position —
// see the session's landmark-crop-nosebridge.png.
const NOSE_BRIDGE_POINT = 6
// Mouth corners — not a guess either: verified by computing the lips
// loop's own leftmost/rightmost points programmatically on a real photo
// and confirming they land exactly on indices 61 and 291 (the same two
// indices nearly every public MediaPipe reference cites for this, but
// confirmed here directly rather than trusted on citation alone).
export const LEFT_MOUTH_CORNER = 61
export const RIGHT_MOUTH_CORNER = 291

// Temple and cheekbone are not separate landmark loops — they're specific
// points *on* the oval (also verified visually: 21/251 sit exactly at the
// hairline beside the eyebrow tail, 234/454 at the widest point of the
// cheek — see landmark-crop-temple.png / landmark-crop-cheekbone.png).
// Rather than hand-picking a few neighboring indices with a hard edge
// between "affected" and "not", each gets a smooth cosine-squared falloff
// in *oval arc position* around its center — zero past `ARC_HALF_WIDTH`
// indices away, full strength at the center. That's what lets temple and
// cheekbone be independent sliders from face/jaw without a visible seam
// where their influence ends: the weight itself tapers to zero smoothly,
// so there's nothing to cut off abruptly when it's layered additively on
// top of whatever face/jaw already did to that same point.
const ARC_HALF_WIDTH = 3
const OVAL_RIGHT_TEMPLE_CENTER = OVAL_LOOP.indexOf(251)
const OVAL_LEFT_TEMPLE_CENTER = OVAL_LOOP.indexOf(21)
const OVAL_RIGHT_CHEEKBONE_CENTER = OVAL_LOOP.indexOf(454)
const OVAL_LEFT_CHEEKBONE_CENTER = OVAL_LOOP.indexOf(234)

function ovalArcWeight(loopIndex: number, center: number): number {
  const loopLen = OVAL_LOOP.length
  let d = Math.abs(loopIndex - center)
  d = Math.min(d, loopLen - d)
  if (d >= ARC_HALF_WIDTH) return 0
  return Math.cos((d / ARC_HALF_WIDTH) * (Math.PI / 2)) ** 2
}

function clamp11(v: number): number {
  return Math.min(1, Math.max(-1, v))
}

function buildVertexBuffers(landmarks: NormalizedLandmark[], params: ReshapeParams, aspect: number) {
  const n = landmarks.length // 468
  const texCoord = new Float32Array((n + OVAL_LOOP.length) * 2)
  const targetPos = new Float32Array((n + OVAL_LOOP.length) * 2)

  for (let i = 0; i < n; i++) {
    texCoord[i * 2] = landmarks[i].x
    texCoord[i * 2 + 1] = landmarks[i].y
    targetPos[i * 2] = landmarks[i].x
    targetPos[i * 2 + 1] = landmarks[i].y
  }

  const leftEyeCenter = loopCenterNorm(LEFT_EYE_LOOP, landmarks)
  const rightEyeCenter = loopCenterNorm(RIGHT_EYE_LOOP, landmarks)
  const trueEyeLineY = (leftEyeCenter.y + rightEyeCenter.y) / 2
  const faceCenterX = (landmarks[LEFT_CHEEK].x + landmarks[RIGHT_CHEEK].x) / 2
  // Landmarks are normalized per-axis (x by width, y by height), so a raw
  // hypot of normalized dx/dy only measures true distance when the frame
  // is square — on a non-square frame one unit of normalized-x covers a
  // different real pixel distance than one unit of normalized-y. Scaling
  // dx by `aspect` (width/height) first converts it into the same
  // normalized-y-equivalent unit as dy, so this is a real-world distance
  // (expressed in units consistent with the Y-axis shifts it scales
  // below) regardless of aspect ratio — and reduces to the original
  // formula exactly when aspect === 1, so the square case is unchanged.
  const eyeSpan = Math.hypot((rightEyeCenter.x - leftEyeCenter.x) * aspect, rightEyeCenter.y - leftEyeCenter.y)

  // "Push toward centerX" assumes a roughly frontal face — the real slim
  // effect is narrowing width around the face's symmetry axis, and
  // centerX only approximates that axis when the face is facing the
  // camera. Turn the head and the two sides of the oval are at very
  // different depths (one near, one foreshortened far), so pushing both
  // toward the same centerX by the same fraction moves them by very
  // different real amounts — right where the near and far side's
  // triangles meet (around the nose bridge), that mismatch showed up as
  // a visible vertical seam, worst against a lighting gradient across
  // the face (the same "invisible on skin, obvious against a gradient"
  // lesson as the original wavy-background bug, just with the gradient
  // now a shadow instead of a door frame). Rather than modeling head
  // pose properly, fade every horizontal oval push out as the face turns
  // away from frontal: measured via how far the eye-corners' own midpoint
  // sits from the cheek-to-cheek midpoint, relative to how far apart the
  // cheeks are (0.07-0.14 on frontal/mildly-turned test photos, 0.36-0.57
  // on a turned photo that produced the seam — a clean separation). This
  // applies to face/temple/cheekbone alike since all three push the oval
  // horizontally toward the same centerX.
  const cheekSpan = Math.abs(landmarks[RIGHT_CHEEK].x - landmarks[LEFT_CHEEK].x)
  const eyesCenterX = (landmarks[33].x + landmarks[133].x + landmarks[362].x + landmarks[263].x) / 4
  const yawProxy = cheekSpan > 1e-5 ? Math.abs(eyesCenterX - faceCenterX) / cheekSpan : 0
  const yawFalloff = 1 - Math.min(1, Math.max(0, (yawProxy - 0.15) / 0.2))

  // Jaw/cheek: oval points below the eye-line get pushed toward (positive)
  // or away from (negative) the face's own horizontal center. Points
  // at/above the eye-line are simply never touched (they keep
  // target === original from the loop above) — no explicit "anchor"
  // bookkeeping needed, unlike the old MLS version, because leaving a
  // vertex alone IS the default state here.
  if (Math.abs(params.face) > 0.001) {
    const pushFraction = clamp11(params.face) * 0.14 * yawFalloff
    for (const idx of OVAL_LOOP) {
      if (landmarks[idx].y <= trueEyeLineY) continue
      targetPos[idx * 2] = landmarks[idx].x + (faceCenterX - landmarks[idx].x) * pushFraction
    }
  }

  // Temple and cheekbone: small additive bumps layered on top of
  // whatever face/jaw already did to the same oval point (see the arc
  // weight comment above for why this can't introduce a new seam).
  if (Math.abs(params.temple) > 0.001) {
    const pushFraction = clamp11(params.temple) * 0.09 * yawFalloff
    for (let i = 0; i < OVAL_LOOP.length; i++) {
      const w = Math.max(ovalArcWeight(i, OVAL_RIGHT_TEMPLE_CENTER), ovalArcWeight(i, OVAL_LEFT_TEMPLE_CENTER))
      if (w <= 0) continue
      const idx = OVAL_LOOP[i]
      targetPos[idx * 2] += (faceCenterX - landmarks[idx].x) * pushFraction * w
    }
  }
  if (Math.abs(params.cheekbone) > 0.001) {
    const pushFraction = clamp11(params.cheekbone) * 0.09 * yawFalloff
    for (let i = 0; i < OVAL_LOOP.length; i++) {
      const w = Math.max(ovalArcWeight(i, OVAL_RIGHT_CHEEKBONE_CENTER), ovalArcWeight(i, OVAL_LEFT_CHEEKBONE_CENTER))
      if (w <= 0) continue
      const idx = OVAL_LOOP[i]
      targetPos[idx * 2] += (faceCenterX - landmarks[idx].x) * pushFraction * w
    }
  }

  if (Math.abs(params.eyes) > 0.001) {
    // Widened more than it's heightened, not scaled uniformly: a glasses
    // frame's rim sits closest to the eye right at its top and bottom (a
    // lens is wider than the eye opening it surrounds, so there's more
    // real clearance at the sides) — moving the eye loop less vertically
    // there is a direct, low-cost way to leave a frame rim bending less,
    // not just a cosmetic side effect. It also happens to look more like
    // real "bigger eyes" filters (which bias toward widening) than a
    // uniform bulge does.
    const amount = clamp11(params.eyes)
    const scaleX = 1 + amount * 0.3
    const scaleY = 1 + amount * 0.14
    for (const [loop, center] of [
      [LEFT_EYE_LOOP, leftEyeCenter],
      [RIGHT_EYE_LOOP, rightEyeCenter],
    ] as const) {
      for (const idx of loop) {
        targetPos[idx * 2] = center.x + (landmarks[idx].x - center.x) * scaleX
        targetPos[idx * 2 + 1] = center.y + (landmarks[idx].y - center.y) * scaleY
      }
    }
  }

  if (Math.abs(params.mouth) > 0.001) {
    const lipsCenter = loopCenterNorm(LIPS_LOOP, landmarks)
    const scale = 1 + clamp11(params.mouth) * 0.2
    for (const idx of LIPS_LOOP) {
      targetPos[idx * 2] = lipsCenter.x + (landmarks[idx].x - lipsCenter.x) * scale
      targetPos[idx * 2 + 1] = lipsCenter.y + (landmarks[idx].y - lipsCenter.y) * scale
    }
  }

  // Finer mouth sub-controls, each layered additively on top of whatever
  // the overall mouth scale above already did to the same points — same
  // compositing pattern as temple/cheekbone on top of jaw. Upper/lower
  // lip split the lips loop by which half of it (above/below the loop's
  // own vertical center) a point falls in, rather than needing a separate
  // "which points are upper lip" list.
  if (Math.abs(params.mouthUpperLip) > 0.001 || Math.abs(params.mouthLowerLip) > 0.001) {
    const lipsCenter = loopCenterNorm(LIPS_LOOP, landmarks)
    const upperShift = clamp11(params.mouthUpperLip) * eyeSpan * 0.03
    const lowerShift = clamp11(params.mouthLowerLip) * eyeSpan * 0.03
    for (const idx of LIPS_LOOP) {
      const p = landmarks[idx]
      if (p.y < lipsCenter.y) {
        targetPos[idx * 2 + 1] -= upperShift // negative y = up = away from center = thicker
      } else {
        targetPos[idx * 2 + 1] += lowerShift // positive y = down = away from center = thicker
      }
    }
  }
  if (Math.abs(params.mouthCorners) > 0.001) {
    const shift = -clamp11(params.mouthCorners) * eyeSpan * 0.025 // negative y = up = lifted
    targetPos[LEFT_MOUTH_CORNER * 2 + 1] += shift
    targetPos[RIGHT_MOUTH_CORNER * 2 + 1] += shift
  }

  // Eyebrows: the official loop, shifted vertically. A small, purely
  // interior movement (no oval/background boundary involved), the same
  // category of change as the eye-enlarge above, so it needs no extra
  // anchoring beyond the surrounding mesh already being fixed elsewhere.
  if (Math.abs(params.eyebrowHeight) > 0.001) {
    const shift = -clamp11(params.eyebrowHeight) * eyeSpan * 0.1
    for (const idx of [...LEFT_EYEBROW_LOOP, ...RIGHT_EYEBROW_LOOP]) {
      targetPos[idx * 2 + 1] = landmarks[idx].y + shift
    }
  }

  // Nose bridge (山根): no official landmark loop exists for it, so
  // instead of moving one bare point (which would crease against its
  // fixed neighbors), every one of the 468 vertices within a small
  // radius of the bridge point gets a share of the same vertical shift,
  // weighted by the same cosine-squared falloff used for temple/
  // cheekbone — a smooth local bump, not a single displaced pin.
  if (Math.abs(params.noseBridge) > 0.001) {
    const center = landmarks[NOSE_BRIDGE_POINT]
    const radius = eyeSpan * 0.22
    const shift = -clamp11(params.noseBridge) * eyeSpan * 0.05
    for (let idx = 0; idx < n; idx++) {
      const dx = landmarks[idx].x - center.x
      const dy = landmarks[idx].y - center.y
      // Same normalized-x-to-y-equivalent scaling as eyeSpan above — without
      // it, this falloff region is a true circle only on a square frame and
      // an ellipse (stretched along whichever axis has more pixels per
      // normalized unit) on any other aspect ratio.
      const d = Math.hypot(dx * aspect, dy)
      if (d >= radius) continue
      const w = Math.cos((d / radius) * (Math.PI / 2)) ** 2
      targetPos[idx * 2 + 1] += shift * w
    }
  }

  // The skirt: one new vertex per oval point, pushed further outward
  // along the same direction from the face center — anchored (its
  // texCoord and targetPos are identical, so it shows undisturbed
  // background and never moves). Triangles connecting each oval point to
  // its skirt counterpart absorb the jaw's inward push into the
  // background over a short, smooth band instead of leaving a visible
  // step where the moved jaw boundary meets the static background layer.
  const faceCenterY = trueEyeLineY
  for (let i = 0; i < OVAL_LOOP.length; i++) {
    const idx = OVAL_LOOP[i]
    const p = landmarks[idx]
    const dx = p.x - faceCenterX
    const dy = p.y - faceCenterY
    const sx = p.x + dx * SKIRT_PAD_FRACTION
    const sy = p.y + dy * SKIRT_PAD_FRACTION
    const skirtIdx = n + i
    texCoord[skirtIdx * 2] = sx
    texCoord[skirtIdx * 2 + 1] = sy
    targetPos[skirtIdx * 2] = sx
    targetPos[skirtIdx * 2 + 1] = sy
  }

  return { texCoord, targetPos, vertexCount: n + OVAL_LOOP.length }
}

function loopCenterNorm(loop: number[], landmarks: NormalizedLandmark[]) {
  let x = 0
  let y = 0
  for (const idx of loop) {
    x += landmarks[idx].x
    y += landmarks[idx].y
  }
  return { x: x / loop.length, y: y / loop.length }
}

// Per-vertex surface normals derived from the face mesh's own 3D landmark
// geometry (MediaPipe gives x, y, and a rough relative depth z per point)
// and the same fixed triangulation the warp itself uses for rasterization
// — not a learned depth/normal map (this app has no trained model for
// that and no way to get one in this environment), but real surface-
// normal math over real mesh geometry: each triangle's own face normal
// (from its two edge vectors) is accumulated into its three vertices,
// then each vertex's accumulated normal is renormalized — the standard
// per-vertex-normal-from-a-mesh technique. Driven by only 478 points, it
// resolves the face's broad forms (forehead, cheek, nose bridge, jaw)
// correctly but not fine wrinkle-level surface detail, which would need a
// far denser mesh than MediaPipe provides.
//
// x is normalized by the frame's width, y by its height, and z (per
// MediaPipe's convention) is roughly in the same unit as x, more negative
// the closer a point is to the camera — so y needs the same aspect
// correction used elsewhere in this file (dividing by `aspect` converts a
// normalized-y delta into normalized-x-equivalent units) before cross
// products between the three axes produce a geometrically sensible
// direction instead of one stretched by the frame's own aspect ratio.
function computeVertexNormals(landmarks: NormalizedLandmark[], aspect: number): Float32Array {
  const n = landmarks.length
  const px = new Float32Array(n)
  const py = new Float32Array(n)
  const pz = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    px[i] = landmarks[i].x
    py[i] = landmarks[i].y / aspect
    pz[i] = landmarks[i].z
  }

  const normals = new Float32Array(n * 3)
  for (let t = 0; t < FACE_TRIANGULATION.length; t += 3) {
    const a = FACE_TRIANGULATION[t]
    const b = FACE_TRIANGULATION[t + 1]
    const c = FACE_TRIANGULATION[t + 2]

    const e1x = px[b] - px[a]
    const e1y = py[b] - py[a]
    const e1z = pz[b] - pz[a]
    const e2x = px[c] - px[a]
    const e2y = py[c] - py[a]
    const e2z = pz[c] - pz[a]

    // e1 x e2 — the resulting sign convention (which way "outward" points)
    // was verified empirically against real landmark data: the forehead,
    // nose bridge, and cheeks all come out pointing toward the camera (see
    // NORMAL_TOWARD_CAMERA_SIGN below, applied right after this loop).
    const nx = e1y * e2z - e1z * e2y
    const ny = e1z * e2x - e1x * e2z
    const nz = e1x * e2y - e1y * e2x

    normals[a * 3] += nx
    normals[a * 3 + 1] += ny
    normals[a * 3 + 2] += nz
    normals[b * 3] += nx
    normals[b * 3 + 1] += ny
    normals[b * 3 + 2] += nz
    normals[c * 3] += nx
    normals[c * 3 + 1] += ny
    normals[c * 3 + 2] += nz
  }

  for (let i = 0; i < n; i++) {
    const x = normals[i * 3] * NORMAL_TOWARD_CAMERA_SIGN
    const y = normals[i * 3 + 1] * NORMAL_TOWARD_CAMERA_SIGN
    const z = normals[i * 3 + 2] * NORMAL_TOWARD_CAMERA_SIGN
    const len = Math.hypot(x, y, z)
    if (len > 1e-6) {
      normals[i * 3] = x / len
      normals[i * 3 + 1] = y / len
      normals[i * 3 + 2] = z / len
    } else {
      normals[i * 3] = 0
      normals[i * 3 + 1] = 0
      normals[i * 3 + 2] = 0
    }
  }
  return normals
}
// Flips the raw e1 x e2 result if needed so normals point toward the
// camera (negative z in this function's own (x, y/aspect, z) space,
// matching MediaPipe's "more negative z = closer to camera" convention)
// rather than away from it — set from the empirical check described
// above, not a guess.
const NORMAL_TOWARD_CAMERA_SIGN = 1

// A fixed "loop lighting"-style default: mostly toward the camera (where
// a light near a phone during a selfie actually sits), a bit from above,
// a bit to one side — a flattering, classic portrait position rather than
// straight-on (which flattens the whole face evenly) or from the side
// (which can look harsh). In the same (x, y/aspect, z) space
// computeVertexNormals produces, so no extra conversion is needed before
// the dot product in the shader. Not yet user-adjustable — a movable
// light is a reasonable follow-up, scoped out of this pass.
function normalize3(v: [number, number, number]): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2])
  return len > 1e-6 ? [v[0] / len, v[1] / len, v[2] / len] : v
}
const DEFAULT_LIGHT_DIR = normalize3([0.28, -0.42, -0.85])

let cachedSkirtIndices: Uint16Array | null = null
function buildIndexArray(n: number): Uint16Array {
  if (cachedSkirtIndices) return cachedSkirtIndices
  const indices = [...FACE_TRIANGULATION]
  const loopLen = OVAL_LOOP.length
  for (let i = 0; i < loopLen; i++) {
    const a = OVAL_LOOP[i]
    const b = OVAL_LOOP[(i + 1) % loopLen]
    const sa = n + i
    const sb = n + ((i + 1) % loopLen)
    indices.push(a, b, sa, b, sb, sa)
  }
  cachedSkirtIndices = new Uint16Array(indices)
  return cachedSkirtIndices
}

export function renderMeshWarp(source: HTMLCanvasElement, landmarks: NormalizedLandmark[], params: ReshapeParams): HTMLCanvasElement {
  const s = getState()
  const { gl } = s
  s.canvas.width = source.width
  s.canvas.height = source.height
  gl.viewport(0, 0, source.width, source.height)
  gl.useProgram(s.program)

  gl.bindTexture(gl.TEXTURE_2D, s.texture)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source)
  gl.activeTexture(gl.TEXTURE0)
  gl.uniform1i(gl.getUniformLocation(s.program, 'u_image'), 0)

  gl.enableVertexAttribArray(s.texCoordLoc)
  gl.enableVertexAttribArray(s.targetPosLoc)
  gl.enableVertexAttribArray(s.normalLoc)

  const lightDirLoc = gl.getUniformLocation(s.program, 'u_lightDir')
  const lightIntensityLoc = gl.getUniformLocation(s.program, 'u_lightIntensity')
  gl.uniform3f(lightDirLoc, DEFAULT_LIGHT_DIR[0], DEFAULT_LIGHT_DIR[1], DEFAULT_LIGHT_DIR[2])

  // Pass 1: the untouched background — identity mapping, covers the
  // whole frame. Drawn first so the warped mesh (pass 2) sits on top of
  // it; anything outside both the mesh and its skirt keeps showing this
  // layer, completely unaffected by any reshape math. Lighting is forced
  // off for this pass (intensity 0) regardless of the normal data bound —
  // the background isn't part of the face and must never be relit.
  gl.uniform1f(lightIntensityLoc, 0)
  const bgTexCoord = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1])
  const bgNormal = new Float32Array(8 * 3) // 4 vertices, zero vectors — irrelevant anyway since intensity is 0
  gl.bindBuffer(gl.ARRAY_BUFFER, s.texCoordBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, bgTexCoord, gl.DYNAMIC_DRAW)
  gl.vertexAttribPointer(s.texCoordLoc, 2, gl.FLOAT, false, 0, 0)
  gl.bindBuffer(gl.ARRAY_BUFFER, s.targetPosBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, bgTexCoord, gl.DYNAMIC_DRAW)
  gl.vertexAttribPointer(s.targetPosLoc, 2, gl.FLOAT, false, 0, 0)
  gl.bindBuffer(gl.ARRAY_BUFFER, s.normalBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, bgNormal, gl.DYNAMIC_DRAW)
  gl.vertexAttribPointer(s.normalLoc, 3, gl.FLOAT, false, 0, 0)
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, s.backgroundIndexBuffer)
  gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)

  // Pass 2: the warped face mesh + anchored skirt on top, now relit.
  const aspect = source.width / source.height
  const { texCoord, targetPos, vertexCount } = buildVertexBuffers(landmarks, params, aspect)
  gl.bindBuffer(gl.ARRAY_BUFFER, s.texCoordBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, texCoord, gl.DYNAMIC_DRAW)
  gl.vertexAttribPointer(s.texCoordLoc, 2, gl.FLOAT, false, 0, 0)
  gl.bindBuffer(gl.ARRAY_BUFFER, s.targetPosBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, targetPos, gl.DYNAMIC_DRAW)
  gl.vertexAttribPointer(s.targetPosLoc, 2, gl.FLOAT, false, 0, 0)

  // Per-vertex normals for the 468 real landmarks, padded with zero
  // vectors for the skirt ring — a zero normal gives n.l = 0 regardless of
  // light direction, so the skirt (which shows undisturbed background
  // content) stays exactly as unlit as the true background just outside
  // it, avoiding a brightness seam at that boundary.
  const faceNormals = computeVertexNormals(landmarks, aspect)
  const normalData = new Float32Array(vertexCount * 3)
  normalData.set(faceNormals, 0)
  gl.bindBuffer(gl.ARRAY_BUFFER, s.normalBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, normalData, gl.DYNAMIC_DRAW)
  gl.vertexAttribPointer(s.normalLoc, 3, gl.FLOAT, false, 0, 0)

  gl.uniform1f(lightIntensityLoc, Math.max(0, Math.min(1, params.fillLight)))

  const indices = buildIndexArray(vertexCount - OVAL_LOOP.length)
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, s.indexBuffer)
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW)
  gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0)

  // Copy into a plain 2D canvas rather than returning the WebGL canvas
  // directly: downstream (the nose pass, filters, JPEG export) all use
  // getImageData/ctx.filter, and this avoids relying on
  // preserveDrawingBuffer (which has its own performance cost) just to
  // keep a read-back working.
  const out = document.createElement('canvas')
  out.width = source.width
  out.height = source.height
  out.getContext('2d')!.drawImage(s.canvas, 0, 0)
  return out
}
