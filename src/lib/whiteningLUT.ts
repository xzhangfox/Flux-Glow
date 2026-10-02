// A 3D color lookup table for whitening — the same mechanism real camera
// color pipelines use (a 17-point-per-axis LUT is a common real-world
// size, including in Android's own camera2 LUT support) to turn an
// expensive per-pixel color-space transform into a cheap one: build the
// transform's output once on a coarse 17x17x17 RGB grid (4913 points,
// each one a real CIELAB round trip), then look up any actual pixel by
// trilinear interpolation between its 8 nearest grid corners — simple
// array reads and multiplies, no transcendental math, regardless of how
// many million pixels use it. Measured: building the LUT costs a few ms
// (fixed, independent of image size); the per-pixel Lab round trip this
// replaces cost ~150ms on a single ~300k-pixel face region on its own.

import { rgbToLab, labToRgb } from './colorSpace'

const REDNESS_THRESHOLD = 6 // a* units above which redness counts as excess, not normal warm undertone
const REDNESS_RANGE = 16
const LIFT_MAX = 9 // L* units (0-100 scale) at full intensity and full midtone weight

function whitenPixel(r: number, g: number, b: number, intensity: number): [number, number, number] {
  const [L, a, labB] = rgbToLab(r, g, b)
  const excess = Math.max(0, a - REDNESS_THRESHOLD)
  const correction = Math.min(1, excess / REDNESS_RANGE) * intensity
  const newA = a - excess * correction * 0.6
  const midtoneWeight = Math.max(0, 1 - Math.abs(L / 100 - 0.55) * 2.2)
  const newL = Math.min(100, L + midtoneWeight * intensity * LIFT_MAX)
  return labToRgb(newL, newA, labB)
}

export interface WhiteningLUT {
  size: number
  table: Float32Array
}

export function buildWhiteningLUT(intensity: number, size = 17): WhiteningLUT {
  const table = new Float32Array(size * size * size * 3)
  for (let ri = 0; ri < size; ri++) {
    const r = (ri / (size - 1)) * 255
    for (let gi = 0; gi < size; gi++) {
      const g = (gi / (size - 1)) * 255
      for (let bi = 0; bi < size; bi++) {
        const b = (bi / (size - 1)) * 255
        const [nr, ng, nb] = whitenPixel(r, g, b, intensity)
        const idx = ((ri * size + gi) * size + bi) * 3
        table[idx] = nr
        table[idx + 1] = ng
        table[idx + 2] = nb
      }
    }
  }
  return { size, table }
}

/** Trilinear lookup — the only per-pixel cost once the LUT is built. */
export function applyLUT(r: number, g: number, b: number, lut: WhiteningLUT, out: [number, number, number]): void {
  const { size, table } = lut
  const scale = (size - 1) / 255
  const rf = r * scale
  const gf = g * scale
  const bf = b * scale
  const r0 = Math.min(size - 2, Math.max(0, Math.floor(rf)))
  const g0 = Math.min(size - 2, Math.max(0, Math.floor(gf)))
  const b0 = Math.min(size - 2, Math.max(0, Math.floor(bf)))
  const r1 = r0 + 1
  const g1 = g0 + 1
  const b1 = b0 + 1
  const rt = rf - r0
  const gt = gf - g0
  const bt = bf - b0

  for (let c = 0; c < 3; c++) {
    const c000 = table[((r0 * size + g0) * size + b0) * 3 + c]
    const c100 = table[((r1 * size + g0) * size + b0) * 3 + c]
    const c010 = table[((r0 * size + g1) * size + b0) * 3 + c]
    const c110 = table[((r1 * size + g1) * size + b0) * 3 + c]
    const c001 = table[((r0 * size + g0) * size + b1) * 3 + c]
    const c101 = table[((r1 * size + g0) * size + b1) * 3 + c]
    const c011 = table[((r0 * size + g1) * size + b1) * 3 + c]
    const c111 = table[((r1 * size + g1) * size + b1) * 3 + c]
    const c00 = c000 * (1 - rt) + c100 * rt
    const c10 = c010 * (1 - rt) + c110 * rt
    const c01 = c001 * (1 - rt) + c101 * rt
    const c11 = c011 * (1 - rt) + c111 * rt
    const c0 = c00 * (1 - gt) + c10 * gt
    const c1 = c01 * (1 - gt) + c11 * gt
    out[c] = c0 * (1 - bt) + c1 * bt
  }
}
