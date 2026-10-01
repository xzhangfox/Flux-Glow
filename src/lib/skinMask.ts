import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'

// MediaPipe exposes each face region as an unordered set of edges, not an
// ordered polygon — walk the chain into a point loop so canvas can draw it
// as a single closed path.
function connectorsToLoop(connections: { start: number; end: number }[]): number[] {
  const adjacency = new Map<number, number[]>()
  for (const { start, end } of connections) {
    if (!adjacency.has(start)) adjacency.set(start, [])
    if (!adjacency.has(end)) adjacency.set(end, [])
    adjacency.get(start)!.push(end)
    adjacency.get(end)!.push(start)
  }
  const first = connections[0].start
  const loop = [first]
  const visited = new Set([first])
  let current = first
  for (let i = 0; i < connections.length; i++) {
    const next = (adjacency.get(current) ?? []).find((n) => !visited.has(n))
    if (next === undefined) break
    loop.push(next)
    visited.add(next)
    current = next
  }
  return loop
}

function fillLoop(ctx: CanvasRenderingContext2D, loop: number[], landmarks: NormalizedLandmark[], w: number, h: number) {
  ctx.beginPath()
  loop.forEach((idx, i) => {
    const p = landmarks[idx]
    const x = p.x * w
    const y = p.y * h
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.closePath()
  ctx.fill()
}

/**
 * A soft-edged mask covering the face's actual skin — the full face oval
 * minus the eyes, eyebrows, and lips — so smoothing never touches the
 * features that need to stay sharp for the result to look like a real
 * photo instead of a smeared one.
 */
export function buildSkinMask(landmarks: NormalizedLandmark[], w: number, h: number): HTMLCanvasElement {
  const oval = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
  const leftEye = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
  const rightEye = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)
  const leftBrow = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW)
  const rightBrow = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW)
  const lips = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LIPS)

  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!

  ctx.fillStyle = '#fff'
  fillLoop(ctx, oval, landmarks, w, h)

  ctx.globalCompositeOperation = 'destination-out'
  for (const loop of [leftEye, rightEye, leftBrow, rightBrow, lips]) {
    fillLoop(ctx, loop, landmarks, w, h)
  }
  ctx.globalCompositeOperation = 'source-over'

  // Feather the cutouts/outline so the smoothed region fades in rather
  // than ending in a visible hard edge.
  const feathered = document.createElement('canvas')
  feathered.width = w
  feathered.height = h
  const fctx = feathered.getContext('2d')!
  fctx.filter = `blur(${Math.max(3, Math.round(w / 170))}px)`
  fctx.drawImage(canvas, 0, 0)
  return feathered
}
