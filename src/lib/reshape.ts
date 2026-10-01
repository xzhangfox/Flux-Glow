import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { connectorsToLoop, loopCenterPx, dist, lerp, type Px } from './landmarks'
import { renderMeshWarp, type ReshapeParams } from './meshWarp'

export type { ReshapeParams }

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
 * Nose narrowing stays on this isolated, monotonic radial zoom — the GPU
 * mesh warp handles face/eyes/mouth (see meshWarp.ts), but the nose has no
 * official MediaPipe landmark loop to pick mesh vertices from without
 * guessing raw indices, and this CPU pass was already verified clean and
 * tightly contained, so there's no reason to touch it.
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

export function applyReshape(source: HTMLCanvasElement, landmarks: NormalizedLandmark[], params: ReshapeParams): HTMLCanvasElement {
  const meshed = params.face > 0.001 || params.eyes > 0.001 || params.mouth > 0.001 ? renderMeshWarp(source, landmarks, params) : source

  if (Math.abs(params.nose) <= 0.001) return meshed

  const w = meshed.width
  const h = meshed.height
  const srcData = meshed.getContext('2d')!.getImageData(0, 0, w, h)
  const outCanvas = document.createElement('canvas')
  outCanvas.width = w
  outCanvas.height = h
  const outCtx = outCanvas.getContext('2d')!
  outCtx.drawImage(meshed, 0, 0)
  const outImageData = outCtx.getImageData(0, 0, w, h)

  const leftEye = loopCenterPx(connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE), landmarks, w, h)
  const rightEye = loopCenterPx(connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE), landmarks, w, h)
  const lips = loopCenterPx(connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LIPS), landmarks, w, h)
  const eyesCenter = lerp(leftEye, rightEye, 0.5)
  const eyeSpan = dist(leftEye, rightEye)
  const nose = lerp(eyesCenter, lips, 0.42)
  const noseRadius = eyeSpan * 0.24
  const noseStrength = Math.min(Math.abs(params.nose), 1) * 0.4 * Math.sign(params.nose)
  radialWarpInPlace(srcData.data, outImageData.data, w, h, nose, noseRadius, noseStrength)

  outCtx.putImageData(outImageData, 0, 0)
  return outCanvas
}
