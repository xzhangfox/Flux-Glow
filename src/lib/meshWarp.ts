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
  varying vec2 v_texCoord;
  void main() {
    v_texCoord = a_texCoord;
    vec2 clip = a_targetPos * 2.0 - 1.0;
    gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  }
`

const FRAGMENT_SRC = `
  precision mediump float;
  varying vec2 v_texCoord;
  uniform sampler2D u_image;
  void main() {
    gl_FragColor = texture2D(u_image, v_texCoord);
  }
`

interface GLState {
  canvas: HTMLCanvasElement
  gl: WebGLRenderingContext
  program: WebGLProgram
  texCoordBuffer: WebGLBuffer
  targetPosBuffer: WebGLBuffer
  indexBuffer: WebGLBuffer
  texCoordLoc: number
  targetPosLoc: number
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
    indexBuffer: gl.createBuffer()!,
    texCoordLoc: gl.getAttribLocation(program, 'a_texCoord'),
    targetPosLoc: gl.getAttribLocation(program, 'a_targetPos'),
    texture,
    backgroundIndexBuffer,
  }
  return state
}

export interface ReshapeParams {
  /** Jaw/cheek narrow. */
  face: number
  /** Eye enlarge. */
  eyes: number
  /** Nose narrow (handled separately — see reshape.ts). */
  nose: number
  /** Lip plump. */
  mouth: number
}

const OVAL_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
const LEFT_EYE_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
const RIGHT_EYE_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)
const LIPS_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LIPS)
const LEFT_CHEEK = 234
const RIGHT_CHEEK = 454

function buildVertexBuffers(landmarks: NormalizedLandmark[], params: ReshapeParams) {
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

  // Jaw/cheek: oval points below the eye-line get pushed toward the
  // face's own horizontal center. Points at/above the eye-line are simply
  // never touched (they keep target === original from the loop above) —
  // no explicit "anchor" bookkeeping needed, unlike the old MLS version,
  // because leaving a vertex alone IS the default state here.
  //
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
  // pose properly, fade the push out as the face turns away from
  // frontal: when the eye-line's own midpoint sits noticeably off from
  // the cheek-to-cheek midpoint (relative to how wide apart the cheeks
  // are), the face is turned enough that this formula's assumption no
  // longer holds, so ease off instead of forcing it anyway.
  if (params.face > 0.001) {
    const faceCenterX = (landmarks[LEFT_CHEEK].x + landmarks[RIGHT_CHEEK].x) / 2
    const cheekSpan = Math.abs(landmarks[RIGHT_CHEEK].x - landmarks[LEFT_CHEEK].x)
    // The four eye-corner indices (not the full eye loop) — empirically
    // measured at 0.07-0.14 for a frontal/mildly-turned face and 0.36-0.57
    // for the turned angle that produced the seam, a clean separation.
    const eyesCenterX = (landmarks[33].x + landmarks[133].x + landmarks[362].x + landmarks[263].x) / 4
    const yawProxy = cheekSpan > 1e-5 ? Math.abs(eyesCenterX - faceCenterX) / cheekSpan : 0
    const yawFalloff = 1 - Math.min(1, Math.max(0, (yawProxy - 0.15) / 0.2))
    const pushFraction = Math.min(params.face, 1) * 0.14 * yawFalloff
    for (const idx of OVAL_LOOP) {
      if (landmarks[idx].y <= trueEyeLineY) continue
      targetPos[idx * 2] = landmarks[idx].x + (faceCenterX - landmarks[idx].x) * pushFraction
    }
  }

  if (params.eyes > 0.001) {
    const scale = 1 + Math.min(params.eyes, 1) * 0.3
    for (const [loop, center] of [
      [LEFT_EYE_LOOP, leftEyeCenter],
      [RIGHT_EYE_LOOP, rightEyeCenter],
    ] as const) {
      for (const idx of loop) {
        targetPos[idx * 2] = center.x + (landmarks[idx].x - center.x) * scale
        targetPos[idx * 2 + 1] = center.y + (landmarks[idx].y - center.y) * scale
      }
    }
  }

  if (params.mouth > 0.001) {
    const lipsCenter = loopCenterNorm(LIPS_LOOP, landmarks)
    const scale = 1 + Math.min(params.mouth, 1) * 0.2
    for (const idx of LIPS_LOOP) {
      targetPos[idx * 2] = lipsCenter.x + (landmarks[idx].x - lipsCenter.x) * scale
      targetPos[idx * 2 + 1] = lipsCenter.y + (landmarks[idx].y - lipsCenter.y) * scale
    }
  }

  // The skirt: one new vertex per oval point, pushed further outward
  // along the same direction from the face center — anchored (its
  // texCoord and targetPos are identical, so it shows undisturbed
  // background and never moves). Triangles connecting each oval point to
  // its skirt counterpart absorb the jaw's inward push into the
  // background over a short, smooth band instead of leaving a visible
  // step where the moved jaw boundary meets the static background layer.
  const faceCenterX = (landmarks[LEFT_CHEEK].x + landmarks[RIGHT_CHEEK].x) / 2
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

  // Pass 1: the untouched background — identity mapping, covers the
  // whole frame. Drawn first so the warped mesh (pass 2) sits on top of
  // it; anything outside both the mesh and its skirt keeps showing this
  // layer, completely unaffected by any reshape math.
  const bgTexCoord = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1])
  gl.bindBuffer(gl.ARRAY_BUFFER, s.texCoordBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, bgTexCoord, gl.DYNAMIC_DRAW)
  gl.vertexAttribPointer(s.texCoordLoc, 2, gl.FLOAT, false, 0, 0)
  gl.bindBuffer(gl.ARRAY_BUFFER, s.targetPosBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, bgTexCoord, gl.DYNAMIC_DRAW)
  gl.vertexAttribPointer(s.targetPosLoc, 2, gl.FLOAT, false, 0, 0)
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, s.backgroundIndexBuffer)
  gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)

  // Pass 2: the warped face mesh + anchored skirt on top.
  const { texCoord, targetPos, vertexCount } = buildVertexBuffers(landmarks, params)
  gl.bindBuffer(gl.ARRAY_BUFFER, s.texCoordBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, texCoord, gl.DYNAMIC_DRAW)
  gl.vertexAttribPointer(s.texCoordLoc, 2, gl.FLOAT, false, 0, 0)
  gl.bindBuffer(gl.ARRAY_BUFFER, s.targetPosBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, targetPos, gl.DYNAMIC_DRAW)
  gl.vertexAttribPointer(s.targetPosLoc, 2, gl.FLOAT, false, 0, 0)
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
