export interface ZoomRange {
  min: number
  max: number
  mode: 'hardware' | 'digital'
  /** The zoom showing the chosen frame shape uncropped: 1 for digital
   *  zoom, the camera's own minimum for hardware zoom. Below it the frame
   *  widens instead, uncovering the rest of the sensor (see `wideZoom`). */
  floor: number
}

// Digital zoom beyond ~4x is just cropping an already-modest live frame
// down to a quarter of its linear size — past where it still reads as a
// lens rather than a blown-up JPEG — so that's the fallback ceiling when
// the camera doesn't expose real optical/sensor zoom.
export const DIGITAL_ZOOM_RANGE: ZoomRange = { min: 1, max: 4, mode: 'digital', floor: 1 }

/** The zoom below 1 at which the frame shows the camera's whole field of
 *  view: a frame shape narrower than the sensor's (e.g. Full on a tall
 *  phone) crops the sides away, and zooming out brings them back. */
export function wideZoom(crop: { fw: number; fh: number }) {
  return Math.ceil(Math.min(1, crop.fw, crop.fh) * 100) / 100
}

/** The digital crop factor for a zoom value: hardware zoom handles
 *  everything from its floor up, the crop only what lies below it. */
export function cropZoom(value: number, { mode, floor }: ZoomRange) {
  return mode === 'hardware' ? Math.min(1, value / floor) : value
}

// Tap-to-cycle stops spanning the real available range, rounded to values
// someone actually reaches for (widest, 1x, 2x, 3x…).
export function buildZoomPresets({ min, max }: ZoomRange): number[] {
  const presets = new Set<number>()
  if (min < 0.95) presets.add(min)
  presets.add(1)
  for (const v of [2, 3, 5]) if (v > min && v <= max) presets.add(v)
  if (max > 1) presets.add(Math.round(max * 10) / 10)
  return Array.from(presets)
    .filter((v) => v >= min - 0.001 && v <= max + 0.001)
    .sort((a, b) => a - b)
}
