// Real color grading, not a stack of CSS filter() keywords. Chained CSS
// filters (saturate/sepia/hue-rotate/brightness) only expose a handful of
// *global* knobs — the same multiplier hits shadows and highlights alike,
// which is why they tend to look like a cheap tint rather than a graded
// photo. The presets below instead follow the actual film/photo grading
// model used by tools like Lightroom and DaVinci Resolve: an exposure
// curve, a white-balance shift, a shadow/highlight split-tone (different
// color recipes for dark vs. light areas — what gives a graded image its
// depth, e.g. teal shadows with warm skin highlights), a "fade" lift that
// raises the black point without touching the rest of the tone curve (the
// matte/faded-film look), then contrast and saturation on top.

export interface ColorGrade {
  exposure: number // -1..1, a gamma curve: lifts/lowers midtones while pinning pure black and white in place
  warmth: number // -1..1, global white-balance shift (warm/cool)
  fade: number // 0..1, lifts the black point toward gray without touching mid/highlight tone — the "faded film" look
  shadowTint: readonly [number, number, number] // color added to shadows, weighted by how dark the pixel is (split-toning)
  highlightTint: readonly [number, number, number] // color added to highlights, weighted by how bright the pixel is
  contrast: number // 1 = none
  saturation: number // 1 = none
  grayscale?: boolean
}

export interface FilterPreset {
  id: string
  label: string
  grade: ColorGrade
}

const NEUTRAL: ColorGrade = { exposure: 0, warmth: 0, fade: 0, shadowTint: [0, 0, 0], highlightTint: [0, 0, 0], contrast: 1, saturation: 1 }

export const FILTER_PRESETS: FilterPreset[] = [
  { id: 'none', label: 'Natural', grade: NEUTRAL },
  {
    // The pink-flushed, slightly brightened look that's the default
    // portrait filter in most Asian beauty cameras — the pink lives in
    // the highlights (cheeks, nose bridge) rather than as a flat tint.
    id: 'peach',
    label: 'Peach',
    grade: { exposure: 0.15, warmth: 0.08, fade: 0, shadowTint: [4, -2, 2], highlightTint: [12, 0, 5], contrast: 0.97, saturation: 1.06 },
  },
  {
    // Bright, low-contrast, slightly desaturated with warm milky blacks —
    // "cream skin" looks trade punch for softness on purpose.
    id: 'cream',
    label: 'Cream',
    grade: { exposure: 0.22, warmth: 0.12, fade: 0.25, shadowTint: [6, 4, 0], highlightTint: [8, 6, 0], contrast: 0.9, saturation: 0.92 },
  },
  {
    // Clean and airy: a touch cool, lifted midtones, cyan-leaning shadows.
    id: 'fresh',
    label: 'Fresh',
    grade: { exposure: 0.18, warmth: -0.08, fade: 0, shadowTint: [-4, 2, 6], highlightTint: [0, 4, 4], contrast: 1.02, saturation: 1.05 },
  },
  {
    id: 'warm',
    label: 'Warm',
    grade: { exposure: 0, warmth: 0.35, fade: 0, shadowTint: [6, 2, -4], highlightTint: [14, 7, -8], contrast: 1.03, saturation: 1.08 },
  },
  {
    id: 'cool',
    label: 'Cool',
    // Teal shadows + a touch of warmth left in the highlights (skin tones)
    // is the most common split-tone recipe in commercial color grading —
    // cooling the whole image uniformly instead just looks desaturated.
    grade: { exposure: 0, warmth: -0.25, fade: 0, shadowTint: [-10, 1, 12], highlightTint: [2, 2, 4], contrast: 1.05, saturation: 1.03 },
  },
  {
    id: 'soft',
    label: 'Soft Glow',
    grade: { exposure: 0.05, warmth: 0.1, fade: 0.45, shadowTint: [4, 2, -2], highlightTint: [8, 5, -2], contrast: 0.93, saturation: 1.05 },
  },
  {
    // Portra-style negative film: green-teal shadows, warm highlights,
    // faded blacks, restrained saturation.
    id: 'film',
    label: 'Film',
    grade: { exposure: 0, warmth: 0.15, fade: 0.3, shadowTint: [-6, 4, 6], highlightTint: [12, 6, -6], contrast: 1.04, saturation: 0.9 },
  },
  {
    id: 'vivid',
    label: 'Vivid',
    grade: { exposure: 0, warmth: 0.05, fade: 0, shadowTint: [-4, 0, 4], highlightTint: [6, 3, -4], contrast: 1.14, saturation: 1.45 },
  },
  {
    id: 'bw',
    label: 'Mono',
    // A flat desaturate reads as cheap; classic black-and-white film
    // stocks (and the presets modeled on them) split-tone too — warm
    // highlights, cool-leaning shadows — rather than neutral gray
    // throughout.
    grade: { exposure: 0, warmth: 0, fade: 0, shadowTint: [-6, -2, 4], highlightTint: [10, 6, -6], contrast: 1.1, saturation: 1, grayscale: true },
  },
]

export function findPreset(id: string): FilterPreset {
  return FILTER_PRESETS.find((p) => p.id === id) ?? FILTER_PRESETS[0]
}

function clamp255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v
}

// Per-channel 256-entry table for the exposure gamma — one pow() per
// entry instead of three per pixel.
function exposureLut(exposure: number): Uint8ClampedArray | null {
  if (Math.abs(exposure) < 0.001) return null
  const gamma = 1 / (1 + exposure * 0.6)
  const lut = new Uint8ClampedArray(256)
  for (let i = 0; i < 256; i++) lut[i] = Math.round(255 * Math.pow(i / 255, gamma))
  return lut
}

/** Grades `source` into a new canvas. `strength` (0..1) blends the graded
 *  result back toward the original per pixel — the "filter intensity"
 *  slider every commercial camera app puts under its filter strip, since a
 *  preset tuned to read clearly on its own is almost always too strong
 *  for at least some faces/lighting. */
export function applyFilter(source: HTMLCanvasElement, preset: FilterPreset, strength = 1): HTMLCanvasElement {
  const g = preset.grade
  const canvas = document.createElement('canvas')
  canvas.width = source.width
  canvas.height = source.height
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(source, 0, 0)
  const s = Math.max(0, Math.min(1, strength))
  if (g === NEUTRAL || s < 0.001) return canvas

  const img = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const data = img.data
  const lut = exposureLut(g.exposure)
  const warmShift = g.warmth * 18
  const fadeAmount = g.fade * 45

  for (let i = 0; i < data.length; i += 4) {
    const r0 = data[i]
    const g0 = data[i + 1]
    const b0 = data[i + 2]
    let r = lut ? lut[r0] : r0
    let gC = lut ? lut[g0] : g0
    let b = lut ? lut[b0] : b0

    if (g.grayscale) {
      const lum = r * 0.299 + gC * 0.587 + b * 0.114
      r = gC = b = lum
    }

    // White balance: a simple global push toward orange (warm) or blue
    // (cool), same idea as a camera's temperature slider.
    r += warmShift
    b -= warmShift

    // Luminance *after* white balance, used to weight both the fade lift
    // and the split-tone below so they respond to this pixel's actual
    // brightness rather than the pre-grade original.
    const lum01 = (r * 0.299 + gC * 0.587 + b * 0.114) / 255

    // Fade: raises the black point by lifting only genuinely dark pixels
    // toward gray — a flat brightness add would wash out the whole image
    // instead of giving the faded-film look's soft/milky blacks.
    if (fadeAmount > 0) {
      const lift = fadeAmount * Math.max(0, 1 - lum01 / 0.5)
      r += lift
      gC += lift
      b += lift
    }

    // Split-tone: shadows and highlights each get their own color recipe,
    // weighted smoothly by how dark/bright the pixel already is.
    const shadowW = Math.max(0, 1 - lum01 * 2)
    const highlightW = Math.max(0, (lum01 - 0.5) * 2)
    r += g.shadowTint[0] * shadowW + g.highlightTint[0] * highlightW
    gC += g.shadowTint[1] * shadowW + g.highlightTint[1] * highlightW
    b += g.shadowTint[2] * shadowW + g.highlightTint[2] * highlightW

    if (g.contrast !== 1) {
      r = (r - 128) * g.contrast + 128
      gC = (gC - 128) * g.contrast + 128
      b = (b - 128) * g.contrast + 128
    }

    if (g.saturation !== 1) {
      const lum = r * 0.299 + gC * 0.587 + b * 0.114
      r = lum + (r - lum) * g.saturation
      gC = lum + (gC - lum) * g.saturation
      b = lum + (b - lum) * g.saturation
    }

    data[i] = clamp255(r0 + (r - r0) * s)
    data[i + 1] = clamp255(g0 + (gC - g0) * s)
    data[i + 2] = clamp255(b0 + (b - b0) * s)
  }

  ctx.putImageData(img, 0, 0)
  return canvas
}

/** A small square preview of every preset, cropped around `focus` (the
 *  face, when there is one) — what the filter strip shows instead of
 *  text-only chips, so picking a look doesn't take trial and error. */
export function renderFilterThumbnails(source: HTMLCanvasElement, focus: { x: number; y: number; size: number } | null, px = 112): Map<string, string> {
  const side = focus ? Math.min(focus.size, source.width, source.height) : Math.min(source.width, source.height)
  const cx = focus ? focus.x : source.width / 2
  const cy = focus ? focus.y : source.height / 2
  const sx = Math.max(0, Math.min(source.width - side, cx - side / 2))
  const sy = Math.max(0, Math.min(source.height - side, cy - side / 2))
  const crop = document.createElement('canvas')
  crop.width = px
  crop.height = px
  crop.getContext('2d')!.drawImage(source, sx, sy, side, side, 0, 0, px, px)
  const out = new Map<string, string>()
  for (const preset of FILTER_PRESETS) out.set(preset.id, applyFilter(crop, preset).toDataURL('image/jpeg', 0.8))
  return out
}
