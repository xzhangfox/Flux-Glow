// Image deformation using Moving Least Squares — Schaefer, McPhail, Warren,
// SIGGRAPH 2006, "Image Deformation Using Moving Least Squares". This is
// the real algorithm behind most professional face-warp tools (not an ad
// hoc radial pinch): instead of warping isolated circular regions one at a
// time, every "moving" control point (an eye-contour vertex pushed
// outward, a jaw-line vertex pushed toward the centerline, …) and every
// "anchor" control point (q == p — explicitly pinned, zero displacement)
// contribute to ONE smooth deformation field for the whole image at once.
// A pixel near a moving point follows it closely; a pixel near an anchor
// stays put; everything between blends continuously — which is exactly
// what stops one feature's warp from smearing into a neighboring one (the
// failure mode independent circular regions hit here).
//
// Ported faithfully from the similarity-transform case in the reference
// Rust implementation (github.com/mpizenberg/rust_mls, MPL-2.0 licensed),
// itself a direct transcription of the paper's equations — not a
// from-scratch reconstruction, since getting the weighting/centroid math
// subtly wrong is exactly the kind of bug that's hard to spot by eye.

export interface Px {
  x: number
  y: number
}

interface Mat2 {
  m11: number
  m21: number
  m12: number
  m22: number
}

function matMul(a: Mat2, b: Mat2): Mat2 {
  return {
    m11: a.m11 * b.m11 + a.m12 * b.m21,
    m21: a.m21 * b.m11 + a.m22 * b.m21,
    m12: a.m11 * b.m12 + a.m12 * b.m22,
    m22: a.m21 * b.m12 + a.m22 * b.m22,
  }
}

/** (point - origin) treated as a row vector, times mat: result = v^T · M. */
function transposeMul(v: Px, m: Mat2): Px {
  return { x: m.m11 * v.x + m.m21 * v.y, y: m.m12 * v.x + m.m22 * v.y }
}

/**
 * Where does `point` land under the similarity deformation (rotation +
 * uniform scale, no shear — the right amount of freedom for a face
 * feature: an affine transform can skew, a rigid one can't scale at all)
 * that moves each `controlsP[i]` to `controlsQ[i]`? Closed-form, no
 * iteration: a weighted least-squares fit, weight `1/|p_i - v|²` so
 * nearby control points dominate and far ones fade out.
 */
function deformSimilarity(controlsP: Px[], controlsQ: Px[], point: Px): Px {
  const n = controlsP.length
  const weights = new Array<number>(n)
  let wSum = 0
  for (let i = 0; i < n; i++) {
    const dx = controlsP[i].x - point.x
    const dy = controlsP[i].y - point.y
    const d2 = dx * dx + dy * dy
    if (d2 < 1e-6) return controlsQ[i] // point coincides with a control point
    const w = 1 / d2
    weights[i] = w
    wSum += w
  }

  let pStarX = 0
  let pStarY = 0
  let qStarX = 0
  let qStarY = 0
  for (let i = 0; i < n; i++) {
    pStarX += weights[i] * controlsP[i].x
    pStarY += weights[i] * controlsP[i].y
    qStarX += weights[i] * controlsQ[i].x
    qStarY += weights[i] * controlsQ[i].y
  }
  pStarX /= wSum
  pStarY /= wSum
  qStarX /= wSum
  qStarY /= wSum

  let muS = 0
  let m: Mat2 = { m11: 0, m21: 0, m12: 0, m22: 0 }
  for (let i = 0; i < n; i++) {
    const phx = controlsP[i].x - pStarX
    const phy = controlsP[i].y - pStarY
    const qhx = controlsQ[i].x - qStarX
    const qhy = controlsQ[i].y - qStarY
    const w = weights[i]
    muS += w * (phx * phx + phy * phy)
    const pMat: Mat2 = { m11: phx, m21: phy, m12: phy, m22: -phx }
    const qMat: Mat2 = { m11: qhx, m21: qhy, m12: qhy, m22: -qhx }
    const term = matMul(pMat, qMat)
    m = {
      m11: m.m11 + w * term.m11,
      m21: m.m21 + w * term.m21,
      m12: m.m12 + w * term.m12,
      m22: m.m22 + w * term.m22,
    }
  }
  m = { m11: m.m11 / muS, m21: m.m21 / muS, m12: m.m12 / muS, m22: m.m22 / muS }

  const projected = transposeMul({ x: point.x - pStarX, y: point.y - pStarY }, m)
  return { x: projected.x + qStarX, y: projected.y + qStarY }
}

export interface ControlPoint {
  p: Px // original landmark position
  q: Px // desired position — equal to p for an anchor (zero displacement)
}

function bilinearSample(data: Uint8ClampedArray, w: number, h: number, x: number, y: number, out: Float32Array) {
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
 * Warps the pixels inside [minX,maxX]×[minY,maxY] by backward-mapping each
 * output pixel through the MLS field (query with p/q swapped — "where in
 * the original image did this output pixel's content come from" — the
 * standard way to render a forward-specified deformation without holes).
 * Pixels outside the box are an untouched identity copy. `grid` controls
 * how many pixels apart the expensive MLS evaluation actually runs (1 =
 * every pixel; >1 evaluates on a coarser grid and bilinearly interpolates
 * the deformation field between grid points — the exact trick that keeps
 * this usable on a live video frame, since an N-landmark MLS query is
 * O(N) per evaluated point).
 */
export function warpRegion(
  srcData: Uint8ClampedArray,
  outData: Uint8ClampedArray,
  w: number,
  h: number,
  controls: ControlPoint[],
  bounds: { minX: number; minY: number; maxX: number; maxY: number },
  grid = 1
) {
  if (controls.length === 0) return
  const controlsQ = controls.map((c) => c.p) // swapped: deform FROM q TO p
  const controlsP = controls.map((c) => c.q)
  const minX = Math.max(0, Math.floor(bounds.minX))
  const maxX = Math.min(w - 1, Math.ceil(bounds.maxX))
  const minY = Math.max(0, Math.floor(bounds.minY))
  const maxY = Math.min(h - 1, Math.ceil(bounds.maxY))
  if (minX > maxX || minY > maxY) return

  const sample = new Float32Array(4)
  const g = Math.max(1, Math.round(grid))

  if (g === 1) {
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const src = deformSimilarity(controlsP, controlsQ, { x, y })
        bilinearSample(srcData, w, h, Math.max(0, Math.min(w - 1, src.x)), Math.max(0, Math.min(h - 1, src.y)), sample)
        const idx = (y * w + x) * 4
        outData[idx] = sample[0]
        outData[idx + 1] = sample[1]
        outData[idx + 2] = sample[2]
        outData[idx + 3] = sample[3]
      }
    }
    return
  }

  // Sparse grid: evaluate the real (expensive) MLS deformation only at
  // grid-cell corners, then bilinearly interpolate the *deformation*
  // between them for every pixel in between — the deformation field is
  // smooth (that's the whole point of MLS), so this is a close
  // approximation at a fraction of the cost.
  const cols = Math.ceil((maxX - minX) / g) + 2
  const rows = Math.ceil((maxY - minY) / g) + 2
  const anchors: Px[] = new Array(cols * rows)
  for (let gy = 0; gy < rows; gy++) {
    const y = Math.min(maxY, minY + gy * g)
    for (let gx = 0; gx < cols; gx++) {
      const x = Math.min(maxX, minX + gx * g)
      anchors[gy * cols + gx] = deformSimilarity(controlsP, controlsQ, { x, y })
    }
  }

  for (let y = minY; y <= maxY; y++) {
    const gy = Math.min(rows - 2, Math.floor((y - minY) / g))
    const fy = Math.min(1, (y - (minY + gy * g)) / g)
    for (let x = minX; x <= maxX; x++) {
      const gx = Math.min(cols - 2, Math.floor((x - minX) / g))
      const fx = Math.min(1, (x - (minX + gx * g)) / g)
      const tl = anchors[gy * cols + gx]
      const tr = anchors[gy * cols + gx + 1]
      const bl = anchors[(gy + 1) * cols + gx]
      const br = anchors[(gy + 1) * cols + gx + 1]
      const srcX = tl.x * (1 - fx) * (1 - fy) + tr.x * fx * (1 - fy) + bl.x * (1 - fx) * fy + br.x * fx * fy
      const srcY = tl.y * (1 - fx) * (1 - fy) + tr.y * fx * (1 - fy) + bl.y * (1 - fx) * fy + br.y * fx * fy
      bilinearSample(srcData, w, h, Math.max(0, Math.min(w - 1, srcX)), Math.max(0, Math.min(h - 1, srcY)), sample)
      const idx = (y * w + x) * 4
      outData[idx] = sample[0]
      outData[idx + 1] = sample[1]
      outData[idx + 2] = sample[2]
      outData[idx + 3] = sample[3]
    }
  }
}
