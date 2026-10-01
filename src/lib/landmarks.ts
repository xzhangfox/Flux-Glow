import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

// MediaPipe exposes each face region as an unordered set of edges, not an
// ordered polygon — walk the chain into a point loop so canvas can draw it
// as a single closed path, or so a region's pixel center/extent can be
// computed from it.
export function connectorsToLoop(connections: { start: number; end: number }[]): number[] {
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

export interface Px {
  x: number
  y: number
}

export function loopToPx(loop: number[], landmarks: NormalizedLandmark[], w: number, h: number): Px[] {
  return loop.map((idx) => ({ x: landmarks[idx].x * w, y: landmarks[idx].y * h }))
}

/** Centroid of a loop's points in pixel space — a region's real geometric
 *  center, not a single hand-picked landmark that might sit off-center. */
export function loopCenterPx(loop: number[], landmarks: NormalizedLandmark[], w: number, h: number): Px {
  const pts = loopToPx(loop, landmarks, w, h)
  const x = pts.reduce((s, p) => s + p.x, 0) / pts.length
  const y = pts.reduce((s, p) => s + p.y, 0) / pts.length
  return { x, y }
}

export function dist(a: Px, b: Px): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

export function lerp(a: Px, b: Px, t: number): Px {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }
}

export interface Bounds {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

/** Bounding box of a loop's points, padded and clamped to the canvas —
 *  lets per-pixel passes skip the (often majority) of the frame outside
 *  the face entirely instead of processing every pixel unconditionally. */
export function loopBoundsPx(loop: number[], landmarks: NormalizedLandmark[], w: number, h: number, pad: number): Bounds {
  const pts = loopToPx(loop, landmarks, w, h)
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const p of pts) {
    if (p.x < minX) minX = p.x
    if (p.y < minY) minY = p.y
    if (p.x > maxX) maxX = p.x
    if (p.y > maxY) maxY = p.y
  }
  return {
    minX: Math.max(0, Math.floor(minX - pad)),
    minY: Math.max(0, Math.floor(minY - pad)),
    maxX: Math.min(w, Math.ceil(maxX + pad)),
    maxY: Math.min(h, Math.ceil(maxY + pad)),
  }
}
