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
