import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

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
 * A localized pinch (not a global warp): pixels within `radius` of `center`
 * sample from progressively farther out as `amount` increases, pulling the
 * cheek/jaw edge inward — the standard "pucker" distortion face-slimming
 * filters use, applied once per cheek. Pixels outside the radius are an
 * identity copy, so this never touches the rest of the photo.
 */
function pinchInPlace(srcData: Uint8ClampedArray, outData: Uint8ClampedArray, w: number, h: number, center: { x: number; y: number }, radius: number, amount: number) {
  const sample: [number, number, number, number] = [0, 0, 0, 0]
  const minX = Math.max(0, Math.floor(center.x - radius))
  const maxX = Math.min(w - 1, Math.ceil(center.x + radius))
  const minY = Math.max(0, Math.floor(center.y - radius))
  const maxY = Math.min(h - 1, Math.ceil(center.y + radius))

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const dx = x - center.x
      const dy = y - center.y
      const dist = Math.sqrt(dx * dx + dy * dy)
      if (dist >= radius || dist < 0.0001) continue
      const normalized = dist / radius
      // amount in (0, ~0.6]: exponent < 1 pulls content from farther out
      // inward to fill near the center, i.e. compresses this region.
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

export function applyContour(source: HTMLCanvasElement, landmarks: NormalizedLandmark[], amount: number): HTMLCanvasElement {
  if (amount <= 0.001) return source
  const w = source.width
  const h = source.height
  const left = landmarks[LEFT_CHEEK]
  const right = landmarks[RIGHT_CHEEK]
  const leftPx = { x: left.x * w, y: left.y * h }
  const rightPx = { x: right.x * w, y: right.y * h }
  const faceWidth = Math.hypot(rightPx.x - leftPx.x, rightPx.y - leftPx.y)
  const radius = faceWidth * 0.42

  const srcCtx = source.getContext('2d')!
  const srcImageData = srcCtx.getImageData(0, 0, w, h)
  const outCanvas = document.createElement('canvas')
  outCanvas.width = w
  outCanvas.height = h
  const outCtx = outCanvas.getContext('2d')!
  outCtx.drawImage(source, 0, 0)
  const outImageData = outCtx.getImageData(0, 0, w, h)

  // Cap well below a grotesque pinch — this is a subtle reshape, not a
  // cartoon squeeze.
  const strength = Math.min(amount, 1) * 0.5
  pinchInPlace(srcImageData.data, outImageData.data, w, h, leftPx, radius, strength)
  pinchInPlace(srcImageData.data, outImageData.data, w, h, rightPx, radius, strength)

  outCtx.putImageData(outImageData, 0, 0)
  return outCanvas
}
