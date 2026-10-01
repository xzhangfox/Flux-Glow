import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { connectorsToLoop, loopCenterPx, loopToPx, dist, lerp, type Px, type Bounds } from './landmarks'
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
 * A monotonic radial zoom — provably well-behaved by construction (every
 * output radius maps to exactly one source radius via a power curve, so it
 * can't fold or overshoot the way a sparse MLS control-point ring can).
 * Used for the nose (no official MediaPipe loop to build MLS controls
 * from) and also for eyes/mouth: an earlier version scaled those via MLS
 * (loop points pushed outward from center, pinned by a nearby anchor
 * ring), but with the ring close enough to be tight, the gap between
 * "scale outward" and "stay put" had no intermediate control points to
 * guide it, and MLS's closed-form solution isn't guaranteed monotonic in
 * that gap — it visibly folded the eyebrow/eye-socket and philtrum/chin
 * area into solid-color smears even at moderate strength. A simple radial
 * zoom has no such gap to fold in.
 */
function radialWarpInPlace(srcData: Uint8ClampedArray, outData: Uint8ClampedArray, w: number, h: number, center: Px, radiusX: number, radiusY: number, amount: number) {
  if (Math.abs(amount) < 0.001 || radiusX < 1 || radiusY < 1) return
  const sample: [number, number, number, number] = [0, 0, 0, 0]
  const minX = Math.max(0, Math.floor(center.x - radiusX))
  const maxX = Math.min(w - 1, Math.ceil(center.x + radiusX))
  const minY = Math.max(0, Math.floor(center.y - radiusY))
  const maxY = Math.min(h - 1, Math.ceil(center.y + radiusY))

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const dx = x - center.x
      const dy = y - center.y
      // Normalized into the ellipse's own coordinate frame so the warp
      // follows the feature's real aspect ratio (eyes and lips are both
      // much wider than tall) instead of a circle that reaches into
      // whatever anatomy happens to sit closest above or below.
      const nx = dx / radiusX
      const ny = dy / radiusY
      const normalized = Math.sqrt(nx * nx + ny * ny)
      if (normalized >= 1 || normalized < 0.0001) continue
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
 * A ring of anchors (q == p, zero displacement) around a box, spaced by a
 * FRACTION of the box's own size rather than a fixed count. That fraction
 * is what actually matters: a sparse ring on a large box leaves wide gaps
 * where the field isn't pinned to anything, and it visibly sags/bulges
 * between anchors — which is exactly what caused straight background
 * lines (a door frame, a mirror edge) to come out wavy the first time
 * this used one shared box for the whole face. Keeping each feature's box
 * tight AND its ring dense relative to that box is what actually fixes it
 * — not just adding more anchors in the abstract.
 */
function ringAnchors(bounds: Bounds, divisions: number): ControlPoint[] {
  const { minX, minY, maxX, maxY } = bounds
  const nx = Math.max(2, Math.round(divisions))
  const ny = Math.max(2, Math.round((divisions * (maxY - minY)) / Math.max(1, maxX - minX)))
  const anchors: ControlPoint[] = []
  for (let i = 0; i <= nx; i++) {
    const x = minX + ((maxX - minX) * i) / nx
    anchors.push({ p: { x, y: minY }, q: { x, y: minY } })
    anchors.push({ p: { x, y: maxY }, q: { x, y: maxY } })
  }
  for (let j = 0; j <= ny; j++) {
    const y = minY + ((maxY - minY) * j) / ny
    anchors.push({ p: { x: minX, y }, q: { x: minX, y } })
    anchors.push({ p: { x: maxX, y }, q: { x: maxX, y } })
  }
  return anchors
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

  // Each pass below composes on top of whatever the previous one did, so
  // every pass after the first must read from a snapshot — reading and
  // writing the same live buffer mid-warp would corrupt itself, since
  // these sample from nearby already-written pixels as they scan.
  let working = srcImageData.data
  const snapshotIfNeeded = () => (working === srcImageData.data ? outImageData.data.slice() : working)

  const toPx = (idx: number): Px => ({ x: landmarks[idx].x * w, y: landmarks[idx].y * h })
  const leftEyeLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE).map(toPx)
  const rightEyeLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE).map(toPx)
  const lipsLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LIPS).map(toPx)
  const leftEye = loopCenterPx(connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE), landmarks, w, h)
  const rightEye = loopCenterPx(connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE), landmarks, w, h)
  const lips = loopCenterPx(connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LIPS), landmarks, w, h)
  const eyesCenter = lerp(leftEye, rightEye, 0.5)
  const eyeSpan = dist(leftEye, rightEye)

  if (params.face > 0.001) {
    const leftCheek = toPx(LEFT_CHEEK)
    const rightCheek = toPx(RIGHT_CHEEK)
    const faceCenterX = (leftCheek.x + rightCheek.x) / 2
    const ovalPts = loopToPx(connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL), landmarks, w, h)
    // Only the jaw/cheek/chin half of the oval (below the eye-line) — the
    // part a slim filter should narrow. The box's own top edge sits right
    // at that line, so the dense ring anchored there is what actually
    // stops the forehead/temple from moving, not a separate anchor set.
    const jawPts = ovalPts.filter((p) => p.y > eyesCenter.y)
    const pushFraction = Math.min(params.face, 1) * 0.14
    const controls: ControlPoint[] = jawPts.map((p) => ({ p, q: { x: p.x + (faceCenterX - p.x) * pushFraction, y: p.y } }))
    const pad = eyeSpan * 0.18
    const bounds: Bounds = { minX: Math.min(...jawPts.map((p) => p.x)) - pad, minY: eyesCenter.y, maxX: Math.max(...jawPts.map((p) => p.x)) + pad, maxY: Math.max(...jawPts.map((p) => p.y)) + pad }
    controls.push(...ringAnchors(bounds, 14))
    const src = snapshotIfNeeded()
    warpRegion(src, outImageData.data, w, h, controls, bounds, grid)
    working = outImageData.data
  }

  if (params.eyes > 0.001) {
    const eyeAmount = Math.min(params.eyes, 1) * 0.35
    const eyeBoxWidth = (loop: Px[]) => Math.max(...loop.map((p) => p.x)) - Math.min(...loop.map((p) => p.x))
    const eyeBoxHeight = (loop: Px[]) => Math.max(...loop.map((p) => p.y)) - Math.min(...loop.map((p) => p.y))
    // Vertical radius is kept much tighter than horizontal (the eye itself
    // is a flat ellipse) specifically so the warp doesn't reach up into
    // the eyebrow or down into the cheek — reaching into a neighboring
    // straight-ish feature is what bent it into a visible wedge before.
    const src1 = snapshotIfNeeded()
    radialWarpInPlace(src1, outImageData.data, w, h, leftEye, eyeBoxWidth(leftEyeLoop) * 0.75, eyeBoxHeight(leftEyeLoop) * 0.9, eyeAmount)
    working = outImageData.data
    const src2 = snapshotIfNeeded()
    radialWarpInPlace(src2, outImageData.data, w, h, rightEye, eyeBoxWidth(rightEyeLoop) * 0.75, eyeBoxHeight(rightEyeLoop) * 0.9, eyeAmount)
    working = outImageData.data
  }

  if (params.mouth > 0.001) {
    const mouthAmount = Math.min(params.mouth, 1) * 0.3
    const lipWidth = Math.max(...lipsLoop.map((p) => p.x)) - Math.min(...lipsLoop.map((p) => p.x))
    const lipHeight = Math.max(...lipsLoop.map((p) => p.y)) - Math.min(...lipsLoop.map((p) => p.y))
    const src = snapshotIfNeeded()
    radialWarpInPlace(src, outImageData.data, w, h, lips, lipWidth * 0.65, lipHeight * 0.85, mouthAmount)
    working = outImageData.data
  }

  if (Math.abs(params.nose) > 0.001) {
    const nose = lerp(eyesCenter, lips, 0.42)
    const noseRadius = eyeSpan * 0.24
    const noseStrength = Math.min(Math.abs(params.nose), 1) * 0.4 * Math.sign(params.nose)
    const src = snapshotIfNeeded()
    radialWarpInPlace(src, outImageData.data, w, h, nose, noseRadius, noseRadius, noseStrength)
    working = outImageData.data
  }

  outCtx.putImageData(outImageData, 0, 0)
  return outCanvas
}
