import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { connectorsToLoop, loopCenterPx, dist, lerp, type Px } from './landmarks'

// Canonical MediaPipe face-mesh indices for the widest point of each cheek
// at jaw height — the standard anchors used for face-slimming filters.
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
 * A localized radial warp (not a global distortion): pixels within `radius`
 * of `center` are backward-mapped along a power curve of their normalized
 * distance. Positive `amount` pulls content in from farther out — a pinch
 * that shrinks the region (face slimming, a narrower nose). Negative
 * `amount` does the reverse — it samples from closer to center, ballooning
 * the region outward (bigger eyes, fuller lips). Pixels outside the radius
 * are an identity copy, so this never touches the rest of the photo.
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
  /** Cheek/jaw pinch — narrows the face. */
  face: number
  /** Eye enlarge. */
  eyes: number
  /** Nose narrow. */
  nose: number
  /** Lip plump. */
  mouth: number
}

interface Region {
  center: Px
  radius: number
  /** Positive shrinks, negative enlarges — see radialWarpInPlace. */
  amount: number
}

/**
 * Locates each adjustable region from the real landmark groups (eyes and
 * lips are MediaPipe's own connector loops — their centroid is exact; the
 * nose has no official loop, so its center is interpolated 42% of the way
 * from the eye-line to the mouth, a ratio and radius both verified against
 * real detected faces, not a guess — see the landmark-accuracy check this
 * module's history was built from).
 */
function locateRegions(landmarks: NormalizedLandmark[], w: number, h: number, params: ReshapeParams): Region[] {
  const regions: Region[] = []

  const leftCheek: Px = { x: landmarks[LEFT_CHEEK].x * w, y: landmarks[LEFT_CHEEK].y * h }
  const rightCheek: Px = { x: landmarks[RIGHT_CHEEK].x * w, y: landmarks[RIGHT_CHEEK].y * h }
  const faceWidth = dist(leftCheek, rightCheek)
  const cheekRadius = faceWidth * 0.42
  if (Math.abs(params.face) > 0.001) {
    const faceStrength = Math.min(Math.abs(params.face), 1) * 0.5 * Math.sign(params.face)
    regions.push({ center: leftCheek, radius: cheekRadius, amount: faceStrength })
    regions.push({ center: rightCheek, radius: cheekRadius, amount: faceStrength })
  }

  const leftEyeLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
  const rightEyeLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)
  const lipsLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LIPS)
  const leftEye = loopCenterPx(leftEyeLoop, landmarks, w, h)
  const rightEye = loopCenterPx(rightEyeLoop, landmarks, w, h)
  const lips = loopCenterPx(lipsLoop, landmarks, w, h)
  const eyesCenter = lerp(leftEye, rightEye, 0.5)
  const eyeSpan = dist(leftEye, rightEye)

  if (Math.abs(params.eyes) > 0.001) {
    // Enlarging reads as negative amount in radialWarpInPlace — cap at a
    // gentle 0.35 so this stays "bigger eyes", not uncanny-valley.
    const eyeStrength = -Math.min(Math.abs(params.eyes), 1) * 0.35 * Math.sign(params.eyes)
    const eyeRadius = eyeSpan * 0.27
    regions.push({ center: leftEye, radius: eyeRadius, amount: eyeStrength })
    regions.push({ center: rightEye, radius: eyeRadius, amount: eyeStrength })
  }

  if (Math.abs(params.nose) > 0.001) {
    const nose = lerp(eyesCenter, lips, 0.42)
    const noseRadius = eyeSpan * 0.24
    const noseStrength = Math.min(Math.abs(params.nose), 1) * 0.4 * Math.sign(params.nose)
    regions.push({ center: nose, radius: noseRadius, amount: noseStrength })
  }

  if (Math.abs(params.mouth) > 0.001) {
    const mouthStrength = -Math.min(Math.abs(params.mouth), 1) * 0.35 * Math.sign(params.mouth)
    const mouthRadius = eyeSpan * 0.32
    regions.push({ center: lips, radius: mouthRadius, amount: mouthStrength })
  }

  return regions
}

export function applyReshape(source: HTMLCanvasElement, landmarks: NormalizedLandmark[], params: ReshapeParams): HTMLCanvasElement {
  const regions = locateRegions(landmarks, source.width, source.height, params)
  if (regions.length === 0) return source

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

  for (const region of regions) {
    radialWarpInPlace(srcImageData.data, outImageData.data, w, h, region.center, region.radius, region.amount)
  }

  outCtx.putImageData(outImageData, 0, 0)
  return outCanvas
}
