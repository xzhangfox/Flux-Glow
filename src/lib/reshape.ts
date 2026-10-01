import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { connectorsToLoop, loopCenterPx, dist, lerp, type Px } from './landmarks'
import { warpRegion, type ControlPoint } from './mls'

// Canonical MediaPipe face-mesh indices for the widest point of each cheek
// at jaw height — verified against real detected faces (see landmarks.ts
// history), not a guess.
const LEFT_CHEEK = 234
const RIGHT_CHEEK = 454

function bilinearSample(data: Uint8ClampedArray, w: number, h: number, x: number, y: number, out: [number, number, number, number]) {
  const x0 = Math.floor(x)
  const y0 = Math.floor(y)
  const x1 = Math.min(x0 + 1, w - 1)
  const y1 = Math.min(y0 + 1, h - 1)
  const fx = x - x0
  const fy = y - y0
  const cx0 = Math.max(0, Math.min(w - 1, x0))
  const cy0 = Math.max(0, Math.min(h - 1, y0))
  const i00 = (cy0 * w + cx0) * 4
  const i10 = (cy0 * w + x1) * 4
  const i01 = (y1 * w + cx0) * 4
  const i11 = (y1 * w + x1) * 4
  for (let c = 0; c < 4; c++) {
    const top = data[i00 + c] * (1 - fx) + data[i10 + c] * fx
    const bottom = data[i01 + c] * (1 - fx) + data[i11 + c] * fx
    out[c] = top * (1 - fy) + bottom * fy
  }
}

/**
 * The nose has no official MediaPipe landmark loop to build real MLS
 * control points from, so it stays on the simpler isolated radial warp
 * (verified clean and tightly contained — see the accuracy check this
 * module's history was built from) rather than guessing raw landmark
 * indices for it.
 */
function radialWarpInPlace(srcData: Uint8ClampedArray, outData: Uint8ClampedArray, w: number, h: number, center: Px, radius: number, amount: number) {
  if (Math.abs(amount) < 0.001 || radius < 1) return
  const sample: [number, number, number, number] = [0, 0, 0, 0]
  const minX = Math.max(0, Math.floor(center.x - radius))
  const maxX = Math.min(w - 1, Math.ceil(center.x + radius))
  const minY = Math.max(0, Math.floor(center.y - radius))
  const maxY = Math.min(h - 1, Math.ceil(center.y + radius))

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const dx = x - center.x
      const dy = y - center.y
      const d = Math.sqrt(dx * dx + dy * dy)
      if (d >= radius || d < 0.0001) continue
      const normalized = d / radius
      const warped = Math.pow(normalized, 1 - amount)
      const factor = warped / normalized
      const sx = center.x + dx * factor
      const sy = center.y + dy * factor
      bilinearSample(srcData, w, h, Math.max(0, Math.min(w - 1, sx)), Math.max(0, Math.min(h - 1, sy)), sample)
      const outIdx = (y * w + x) * 4
      outData[outIdx] = sample[0]
      outData[outIdx + 1] = sample[1]
      outData[outIdx + 2] = sample[2]
      outData[outIdx + 3] = sample[3]
    }
  }
}

export interface ReshapeParams {
  /** Jaw/cheek narrow. */
  face: number
  /** Eye enlarge. */
  eyes: number
  /** Nose narrow. */
  nose: number
  /** Lip plump. */
  mouth: number
}

/**
 * Builds ONE combined set of MLS control points for the whole face rather
 * than independent per-feature regions. Every landmark group is always
 * included — as a moving point (pushed toward its target) when that
 * slider is active, as an anchor (q == p, zero displacement) when it
 * isn't. This is what actually fixes the smearing a lone circular region
 * was prone to: a "moving" jaw point right next to an "anchor" eyebrow
 * point pulls the deformation field taut between them instead of the two
 * regions warping independently and fighting (or overlapping) at their
 * boundary. A ring of anchors around the whole bounding box does the same
 * job at the outer edge, so the warped region blends into the untouched
 * rest of the photo instead of showing a seam.
 */
function buildFaceControls(landmarks: NormalizedLandmark[], w: number, h: number, params: ReshapeParams): { controls: ControlPoint[]; bounds: { minX: number; minY: number; maxX: number; maxY: number } } {
  const controls: ControlPoint[] = []
  const toPx = (idx: number): Px => ({ x: landmarks[idx].x * w, y: landmarks[idx].y * h })

  const leftCheek = toPx(LEFT_CHEEK)
  const rightCheek = toPx(RIGHT_CHEEK)
  const faceCenterX = (leftCheek.x + rightCheek.x) / 2

  const ovalLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
  const leftEyeLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
  const rightEyeLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)
  const leftBrowLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW)
  const rightBrowLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW)
  const lipsLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LIPS)

  const leftEye = loopCenterPx(leftEyeLoop, landmarks, w, h)
  const rightEye = loopCenterPx(rightEyeLoop, landmarks, w, h)
  const lips = loopCenterPx(lipsLoop, landmarks, w, h)
  const eyesCenter = lerp(leftEye, rightEye, 0.5)
  const eyeSpan = dist(leftEye, rightEye)

  // Face oval: points below the eye-line are the jaw/cheek/chin — the
  // part a slim-face filter should narrow. Points above it (forehead,
  // temple) always stay anchored so the warp has something stable to
  // blend into on its way up.
  const faceStrength = Math.min(Math.max(params.face, 0), 1)
  for (const idx of ovalLoop) {
    const p = toPx(idx)
    if (p.y > eyesCenter.y && faceStrength > 0.001) {
      const pushFraction = faceStrength * 0.14
      controls.push({ p, q: { x: p.x + (faceCenterX - p.x) * pushFraction, y: p.y } })
    } else {
      controls.push({ p, q: p })
    }
  }

  for (const idx of [...leftBrowLoop, ...rightBrowLoop]) {
    const p = toPx(idx)
    controls.push({ p, q: p })
  }

  const eyeStrength = Math.min(Math.max(params.eyes, 0), 1)
  for (const [loop, center] of [
    [leftEyeLoop, leftEye],
    [rightEyeLoop, rightEye],
  ] as const) {
    const scale = 1 + eyeStrength * 0.22
    for (const idx of loop) {
      const p = toPx(idx)
      if (eyeStrength > 0.001) {
        controls.push({ p, q: { x: center.x + (p.x - center.x) * scale, y: center.y + (p.y - center.y) * scale } })
      } else {
        controls.push({ p, q: p })
      }
    }
  }

  const mouthStrength = Math.min(Math.max(params.mouth, 0), 1)
  for (const idx of lipsLoop) {
    const p = toPx(idx)
    if (mouthStrength > 0.001) {
      const scale = 1 + mouthStrength * 0.18
      controls.push({ p, q: { x: lips.x + (p.x - lips.x) * scale, y: lips.y + (p.y - lips.y) * scale } })
    } else {
      controls.push({ p, q: p })
    }
  }

  // Bounding box around everything above, padded, then pinned with its
  // own ring of anchors so the field is faithful right at the edge —
  // nothing leaks out into hair or background past this box.
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const c of controls) {
    minX = Math.min(minX, c.p.x)
    minY = Math.min(minY, c.p.y)
    maxX = Math.max(maxX, c.p.x)
    maxY = Math.max(maxY, c.p.y)
  }
  const pad = eyeSpan * 0.35
  minX -= pad
  minY -= pad
  maxX += pad
  maxY += pad

  const ringSteps = 10
  for (let i = 0; i < ringSteps; i++) {
    const t = i / ringSteps
    const edgePoints: Px[] = [
      { x: minX + (maxX - minX) * t, y: minY },
      { x: minX + (maxX - minX) * t, y: maxY },
      { x: minX, y: minY + (maxY - minY) * t },
      { x: maxX, y: minY + (maxY - minY) * t },
    ]
    for (const p of edgePoints) controls.push({ p, q: p })
  }

  return { controls, bounds: { minX, minY, maxX, maxY } }
}

export function applyReshape(source: HTMLCanvasElement, landmarks: NormalizedLandmark[], params: ReshapeParams, grid = 1): HTMLCanvasElement {
  const w = source.width
  const h = source.height
  const srcCtx = source.getContext('2d')!
  const srcImageData = srcCtx.getImageData(0, 0, w, h)
  const outCanvas = document.createElement('canvas')
  outCanvas.width = w
  outCanvas.height = h
  const outCtx = outCanvas.getContext('2d')!
  outCtx.drawImage(source, 0, 0)
  const outImageData = outCtx.getImageData(0, 0, w, h)

  const anyFaceFeatureActive = params.face > 0.001 || params.eyes > 0.001 || params.mouth > 0.001
  if (anyFaceFeatureActive) {
    const { controls, bounds } = buildFaceControls(landmarks, w, h, params)
    warpRegion(srcImageData.data, outImageData.data, w, h, controls, bounds, grid)
  }

  if (Math.abs(params.nose) > 0.001) {
    const leftEye = loopCenterPx(connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE), landmarks, w, h)
    const rightEye = loopCenterPx(connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE), landmarks, w, h)
    const lips = loopCenterPx(connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LIPS), landmarks, w, h)
    const eyesCenter = lerp(leftEye, rightEye, 0.5)
    const eyeSpan = dist(leftEye, rightEye)
    const nose = lerp(eyesCenter, lips, 0.42)
    const noseRadius = eyeSpan * 0.24
    const noseStrength = Math.min(Math.abs(params.nose), 1) * 0.4 * Math.sign(params.nose)
    // Must read from a snapshot, not outImageData.data itself — this warp
    // samples from nearby already-written pixels as it scans, so reading
    // and writing the same live buffer would corrupt itself mid-pass.
    // Composes on top of whatever the MLS pass above already did rather
    // than overwriting it.
    const noseSrc = anyFaceFeatureActive ? new Uint8ClampedArray(outImageData.data) : srcImageData.data
    radialWarpInPlace(noseSrc, outImageData.data, w, h, nose, noseRadius, noseStrength)
  }

  outCtx.putImageData(outImageData, 0, 0)
  return outCanvas
}
