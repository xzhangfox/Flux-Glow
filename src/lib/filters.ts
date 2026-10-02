// Real color grading, not a stack of CSS filter() keywords. Chained CSS
// filters (saturate/sepia/hue-rotate/brightness) only expose a handful of
// *global* knobs — the same multiplier hits shadows and highlights alike,
// which is why they tend to look like a cheap tint rather than a graded
// photo. The presets below instead follow the actual film/photo grading
// model used by tools like Lightroom and DaVinci Resolve: a white-balance
// shift, a shadow/highlight split-tone (different colors recipe for dark
// vs. light areas — what gives a graded image its depth, e.g. teal
// shadows with warm skin highlights), a "fade" lift that raises the black
// point without touching the rest of the tone curve (the matte/faded-film
// look), then contrast and saturation on top.

export interface ColorGrade {
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

const NEUTRAL: ColorGrade = { warmth: 0, fade: 0, shadowTint: [0, 0, 0], highlightTint: [0, 0, 0], contrast: 1, saturation: 1 }

export const FILTER_PRESETS: FilterPreset[] = [
  { id: 'none', label: 'Natural', grade: NEUTRAL },
  {
    id: 'warm',
    label: 'Warm',
    grade: { warmth: 0.35, fade: 0, shadowTint: [6, 2, -4], highlightTint: [14, 7, -8], contrast: 1.03, saturation: 1.08 },
  },
  {
    id: 'cool',
    label: 'Cool',
    // Teal shadows + a touch of warmth left in the highlights (skin tones)
    // is the most common split-tone recipe in commercial color grading —
    // cooling the whole image uniformly instead just looks desaturated.
    grade: { warmth: -0.25, fade: 0, shadowTint: [-10, 1, 12], highlightTint: [2, 2, 4], contrast: 1.05, saturation: 1.03 },
  },
  {
    id: 'soft',
    label: 'Soft Glow',
    grade: { warmth: 0.1, fade: 0.45, shadowTint: [4, 2, -2], highlightTint: [8, 5, -2], contrast: 0.93, saturation: 1.05 },
  },
  {
    id: 'vivid',
    label: 'Vivid',
    grade: { warmth: 0.05, fade: 0, shadowTint: [-4, 0, 4], highlightTint: [6, 3, -4], contrast: 1.14, saturation: 1.45 },
  },
  {
    id: 'bw',
    label: 'Mono',
    // A flat desaturate reads as cheap; classic black-and-white film
    // stocks (and the presets modeled on them) split-tone too — warm
    // highlights, cool-leaning shadows — rather than neutral gray
    // throughout.
    grade: { warmth: 0, fade: 0, shadowTint: [-6, -2, 4], highlightTint: [10, 6, -6], contrast: 1.1, saturation: 1, grayscale: true },
  },
]

function clamp255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v
}

export function applyFilter(source: HTMLCanvasElement, preset: FilterPreset): HTMLCanvasElement {
  const g = preset.grade
  const canvas = document.createElement('canvas')
  canvas.width = source.width
  canvas.height = source.height
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(source, 0, 0)
  if (g === NEUTRAL) return canvas

  const img = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const data = img.data
  const warmShift = g.warmth * 18
  const fadeAmount = g.fade * 45

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i]
    let gC = data[i + 1]
    let b = data[i + 2]

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
    // toward gray, independent of the tone curve below — a flat
    // brightness add here would just wash out the whole image instead of
    // giving it the faded-film look's hallmark soft/milky blacks.
    if (fadeAmount > 0) {
      const shadowWeight = Math.max(0, 1 - lum01 / 0.5)
      const lift = fadeAmount * shadowWeight
      r += lift
      gC += lift
      b += lift
    }

    // Split-tone: shadows and highlights each get their own color recipe,
    // weighted smoothly by how dark/bright the pixel already is, with
    // midtones mostly untouched — this is what gives a graded image
    // depth instead of a single uniform tint.
    const shadowW = Math.max(0, 1 - lum01 * 2)
    const highlightW = Math.max(0, (lum01 - 0.5) * 2)
    r += g.shadowTint[0] * shadowW + g.highlightTint[0] * highlightW
    gC += g.shadowTint[1] * shadowW + g.highlightTint[1] * highlightW
    b += g.shadowTint[2] * shadowW + g.highlightTint[2] * highlightW

    // Contrast: an S-curve around mid-gray.
    if (g.contrast !== 1) {
      r = (r - 128) * g.contrast + 128
      gC = (gC - 128) * g.contrast + 128
      b = (b - 128) * g.contrast + 128
    }

    // Saturation: pull each channel toward (or push away from) this
    // pixel's own luminance, computed fresh after every adjustment above.
    if (g.saturation !== 1) {
      const lum = r * 0.299 + gC * 0.587 + b * 0.114
      r = lum + (r - lum) * g.saturation
      gC = lum + (gC - lum) * g.saturation
      b = lum + (b - lum) * g.saturation
    }

    data[i] = clamp255(r)
    data[i + 1] = clamp255(gC)
    data[i + 2] = clamp255(b)
  }

  ctx.putImageData(img, 0, 0)
  return canvas
}
