export interface ZoomRange {
  min: number
  max: number
  mode: 'hardware' | 'digital'
}

// Digital zoom beyond ~4x is just cropping an already-modest live frame
// down to a quarter of its linear size — past where it still reads as a
// lens rather than a blown-up JPEG — so that's the fallback ceiling when
// the camera doesn't expose real optical/sensor zoom.
export const DIGITAL_ZOOM_RANGE: ZoomRange = { min: 1, max: 4, mode: 'digital' }

// Tap-to-cycle stops spanning the real available range, rounded to values
// someone actually reaches for (0.5x ultra-wide, 1x, 2x, 3x…).
export function buildZoomPresets({ min, max }: ZoomRange): number[] {
  const presets = new Set<number>()
  if (min < 1) presets.add(Math.round(min * 10) / 10)
  presets.add(1)
  for (const v of [2, 3, 5]) if (v > min && v <= max) presets.add(v)
  if (max > 1) presets.add(Math.round(max * 10) / 10)
  return Array.from(presets)
    .filter((v) => v >= min - 0.001 && v <= max + 0.001)
    .sort((a, b) => a - b)
}
