import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { FACE_TRIANGULATION } from './faceTriangulation'
import { deformTargets, OVAL_LOOP, type ReshapeParams } from './deform'

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
    // The normal's length carries the edge fade (see computeVertexNormals).
    float edge = min(length(v_normal), 1.0);
    vec3 n = edge > 0.0001 ? normalize(v_normal) : vec3(0.0);
    float ndotl = max(dot(n, u_lightDir), 0.0) * edge;

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

export type { ReshapeParams }

// Mouth corners — verified by computing the lips loop's own leftmost/
// rightmost points programmatically on a real photo (61 and 291).
export const LEFT_MOUTH_CORNER = 61
export const RIGHT_MOUTH_CORNER = 291

function buildVertexBuffers(landmarks: NormalizedLandmark[], params: ReshapeParams, aspect: number) {
  const n = landmarks.length // 468
  const texCoord = new Float32Array((n + OVAL_LOOP.length) * 2)
  const targetPos = new Float32Array((n + OVAL_LOOP.length) * 2)
  for (let i = 0; i < n; i++) {
    texCoord[i * 2] = landmarks[i].x
    texCoord[i * 2 + 1] = landmarks[i].y
  }
  targetPos.set(deformTargets(landmarks, params, aspect), 0)

  // The skirt: one new vertex per oval point, pushed further outward
  // along the same direction from the face center — anchored (its
  // texCoord and targetPos are identical, so it shows undisturbed
  // background and never moves). Triangles connecting each oval point to
  // its skirt counterpart absorb any contour change into the background
  // over a short, smooth band instead of leaving a visible step where the
  // moved face boundary meets the static background layer.
  const faceCenterX = (landmarks[234].x + landmarks[454].x) / 2
  const faceCenterY = (landmarks[33].y + landmarks[263].y) / 2
  for (let i = 0; i < OVAL_LOOP.length; i++) {
    const p = landmarks[OVAL_LOOP[i]]
    const sx = p.x + (p.x - faceCenterX) * SKIRT_PAD_FRACTION
    const sy = p.y + (p.y - faceCenterY) * SKIRT_PAD_FRACTION
    const skirtIdx = n + i
    texCoord[skirtIdx * 2] = sx
    texCoord[skirtIdx * 2 + 1] = sy
    targetPos[skirtIdx * 2] = sx
    targetPos[skirtIdx * 2 + 1] = sy
  }

  return { texCoord, targetPos, vertexCount: n + OVAL_LOOP.length }
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
  // Fade relighting out toward the face contour. Each vertex's normal is
  // scaled by how far inside the oval it sits (1 in the middle of the face,
  // easing to 0 at the oval), and the shader reads that length back as a
  // weight — so the lit area dissolves into the hairline and jaw instead of
  // stopping at the mesh's edge (which showed as a hard band across the
  // top of the forehead).
  const cx = (landmarks[234].x + landmarks[454].x) / 2
  const cy = (landmarks[10].y + landmarks[152].y) / 2
  const oval = OVAL_LOOP.map((i) => {
    const dx = landmarks[i].x * aspect - cx * aspect
    const dy = landmarks[i].y - cy
    return { a: Math.atan2(dy, dx), r: Math.hypot(dx, dy) }
  })
  for (let i = 0; i < n; i++) {
    const dx = landmarks[i].x * aspect - cx * aspect
    const dy = landmarks[i].y - cy
    const a = Math.atan2(dy, dx)
    let best = oval[0]
    let bestD = Infinity
    for (const o of oval) {
      const d = Math.abs(Math.atan2(Math.sin(a - o.a), Math.cos(a - o.a)))
      if (d < bestD) {
        bestD = d
        best = o
      }
    }
    const rho = Math.hypot(dx, dy) / Math.max(best.r, 1e-6)
    const t = Math.min(1, Math.max(0, (0.95 - rho) / 0.4))
    const w = t * t * (3 - 2 * t)
    normals[i * 3] *= w
    normals[i * 3 + 1] *= w
    normals[i * 3 + 2] *= w
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
