// Four independent skin/tone adjustments, separate from `smoothSkin`'s
// frequency-separation blemish/tone smoothing — each is mask-bounded the
// same way (only the face's own bounding box is ever touched, since the
// mask is zero everywhere else anyway) but does a genuinely different
// kind of correction rather than just being the same blur at another
// strength.

function blurredCopy(source: HTMLCanvasElement, radiusPx: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = source.width
  canvas.height = source.height
  const ctx = canvas.getContext('2d')!
  ctx.filter = `blur(${radiusPx}px)`
  ctx.drawImage(source, 0, 0)
  return canvas
}

export interface Bounds {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

function boundsRect(w: number, h: number, bounds?: Bounds) {
  const bx = bounds ? Math.max(0, Math.floor(bounds.minX)) : 0
  const by = bounds ? Math.max(0, Math.floor(bounds.minY)) : 0
  const bw = (bounds ? Math.min(w, Math.ceil(bounds.maxX)) : w) - bx
  const bh = (bounds ? Math.min(h, Math.ceil(bounds.maxY)) : h) - by
  return { bx, by, bw, bh }
}

/**
 * Fill light: lifts shadows without flattening the face into a uniform
 * pale "mask" — the failure mode of a flat screen blend applied to every
 * skin pixel alike, which brightens an already-lit cheek by nearly the
 * same amount as a genuinely shadowed jawline or nose side and so erases
 * the light/shadow modeling that makes a face read as three-dimensional
 * in the first place.
 *
 * This instead follows the same idea as Lightroom/Camera Raw's "Shadows"
 * recovery: derive a *local* exposure map by blurring at roughly the
 * scale real facial shadows occur at (nose bridge, under-eye, jawline) —
 * well above pore/texture scale — then only lift pixels whose surrounding
 * area actually reads as dark, with a smooth (smoothstep) falloff rather
 * than a hard cutoff. Already-bright regions stay close to untouched. The
 * lift is applied to each pixel's own full-resolution value, not the
 * blurred copy, so fine skin texture is preserved exactly — only the
 * broad tone balance changes, the way a reflector changes a photo.
 */
export function applyFillLight(source: HTMLCanvasElement, mask: HTMLCanvasElement, intensity: number, bounds?: Bounds): HTMLCanvasElement {
  const w = source.width
  const h = source.height
  const result = document.createElement('canvas')
  result.width = w
  result.height = h
  const rctx = result.getContext('2d')!
  rctx.drawImage(source, 0, 0)
  if (intensity <= 0.001) return result

  const { bx, by, bw, bh } = boundsRect(w, h, bounds)
  if (bw <= 0 || bh <= 0) return result

  // Blur radius tuned to the scale facial shadows actually occur at, not
  // pixel-level darkness — this is what makes the lift map read as "is
  // this part of the face in shadow" rather than "is this exact pixel
  // dark", which would just reintroduce a texture-flattening blur.
  const shadowScale = Math.max(6, bw * 0.12)
  const localExposure = blurredCopy(source, shadowScale)

  const orig = source.getContext('2d')!.getImageData(bx, by, bw, bh)
  const localData = localExposure.getContext('2d')!.getImageData(bx, by, bw, bh)
  const maskData = mask.getContext('2d')!.getImageData(bx, by, bw, bh)
  const out = new ImageData(bw, bh)

  const SHADOW_THRESHOLD = 0.6 // local luminance (0-1) below which an area counts as "shadowed"
  const MAX_LIFT = 0.55 // ceiling on the screen-blend amount even at intensity=1 in the darkest shadow

  for (let i = 0; i < orig.data.length; i += 4) {
    const maskAlpha = maskData.data[i + 3] / 255
    const localLum = (localData.data[i] * 0.299 + localData.data[i + 1] * 0.587 + localData.data[i + 2] * 0.114) / 255
    const t = Math.min(1, Math.max(0, (SHADOW_THRESHOLD - localLum) / SHADOW_THRESHOLD))
    const shadowWeight = t * t * (3 - 2 * t) // smoothstep: soft falloff, no visible "shadow / not-shadow" boundary
    const liftAmount = shadowWeight * intensity * MAX_LIFT
    for (let c = 0; c < 3; c++) {
      const v = orig.data[i + c]
      const screened = 255 - (255 - v) * (1 - liftAmount)
      out.data[i + c] = v + (screened - v) * maskAlpha
    }
    out.data[i + 3] = orig.data[i + 3]
  }
  rctx.putImageData(out, bx, by)
  return result
}

/**
 * Whitening/tone-evening: corrects the two things that actually read as
 * "uneven, ruddy skin" — localized redness around the nose and cheeks,
 * and dull midtones — instead of a flat screen blend plus a global
 * desaturate-toward-luminance. The old approach ignored *where* the
 * unevenness actually was and instead washed color out of the whole face
 * at once, which is indistinguishable from a cheap pale overlay once the
 * intensity went up.
 *
 * Redness is corrected as an *excess* above a normal baseline (the same
 * deviation-thresholding idea `applyAcneRemoval` below uses for spots),
 * so ordinary warm skin tones are left alone and only genuine
 * blotchiness/ruddiness gets pulled toward neutral — closer to how a
 * selective color correction targets a specific hue range than to a
 * blanket desaturate. Brightening is weighted toward midtones (a bump
 * centered on mid-gray, falling off toward both black and white) so dull
 * areas lift without blowing out highlights that a flat screen blend
 * would have brightened just as much.
 */
export function applyWhitening(source: HTMLCanvasElement, mask: HTMLCanvasElement, intensity: number, bounds?: Bounds): HTMLCanvasElement {
  const w = source.width
  const h = source.height
  const result = document.createElement('canvas')
  result.width = w
  result.height = h
  const rctx = result.getContext('2d')!
  rctx.drawImage(source, 0, 0)
  if (intensity <= 0.001) return result

  const { bx, by, bw, bh } = boundsRect(w, h, bounds)
  if (bw <= 0 || bh <= 0) return result

  const orig = source.getContext('2d')!.getImageData(bx, by, bw, bh)
  const maskData = mask.getContext('2d')!.getImageData(bx, by, bw, bh)
  const out = new ImageData(bw, bh)

  const REDNESS_THRESHOLD = 8
  const REDNESS_RANGE = 45
  const BRIGHTEN_MAX = 0.16

  for (let i = 0; i < orig.data.length; i += 4) {
    const maskAlpha = maskData.data[i + 3] / 255
    const r = orig.data[i]
    const g = orig.data[i + 1]
    const b = orig.data[i + 2]

    const avgGB = (g + b) / 2
    const redness = r - avgGB
    const excess = Math.max(0, redness - REDNESS_THRESHOLD)
    const correction = Math.min(1, excess / REDNESS_RANGE) * intensity
    const correctedR = r - excess * correction * 0.7

    const lum = (correctedR * 0.299 + g * 0.587 + b * 0.114) / 255
    const midtoneWeight = Math.max(0, 1 - Math.abs(lum - 0.55) * 2.2) // peaks near mid-gray, ~0 near black/white
    const brighten = midtoneWeight * intensity * BRIGHTEN_MAX * 255

    const finalR = correctedR + brighten
    const finalG = g + brighten
    const finalB = b + brighten

    out.data[i] = r + (finalR - r) * maskAlpha
    out.data[i + 1] = g + (finalG - g) * maskAlpha
    out.data[i + 2] = b + (finalB - b) * maskAlpha
    out.data[i + 3] = orig.data[i + 3]
  }
  rctx.putImageData(out, bx, by)
  return result
}

/**
 * Acne/blemish removal: unlike `smoothSkin` (which smooths tone
 * everywhere in the mask uniformly), this only pulls a pixel toward its
 * local blurred average once it deviates from that average by more than
 * a threshold — ordinary skin texture stays untouched, and only actual
 * outliers (a spot, a red mark) get corrected, which is what a real
 * spot-removal tool does instead of a blanket blur.
 */
export function applyAcneRemoval(source: HTMLCanvasElement, mask: HTMLCanvasElement, intensity: number, bounds?: Bounds): HTMLCanvasElement {
  const w = source.width
  const h = source.height
  const result = document.createElement('canvas')
  result.width = w
  result.height = h
  const rctx = result.getContext('2d')!
  rctx.drawImage(source, 0, 0)
  if (intensity <= 0.001) return result

  const { bx, by, bw, bh } = boundsRect(w, h, bounds)
  if (bw <= 0 || bh <= 0) return result

  const low = blurredCopy(source, 7)
  const orig = source.getContext('2d')!.getImageData(bx, by, bw, bh)
  const lowData = low.getContext('2d')!.getImageData(bx, by, bw, bh)
  const maskData = mask.getContext('2d')!.getImageData(bx, by, bw, bh)
  const out = new ImageData(bw, bh)
  const DEVIATION_THRESHOLD = 16
  const DEVIATION_RANGE = 40

  for (let i = 0; i < orig.data.length; i += 4) {
    const maskAlpha = maskData.data[i + 3] / 255
    const deviation = Math.abs(orig.data[i] - lowData.data[i]) + Math.abs(orig.data[i + 1] - lowData.data[i + 1]) + Math.abs(orig.data[i + 2] - lowData.data[i + 2])
    const excess = Math.max(0, deviation - DEVIATION_THRESHOLD)
    const pullAmount = Math.min(1, excess / DEVIATION_RANGE) * intensity * maskAlpha
    for (let c = 0; c < 3; c++) {
      out.data[i + c] = orig.data[i + c] + (lowData.data[i + c] - orig.data[i + c]) * pullAmount
    }
    out.data[i + 3] = orig.data[i + 3]
  }
  rctx.putImageData(out, bx, by)
  return result
}

/**
 * Wrinkle removal: a three-band split — fine texture (pores, a very light
 * blur's residual) is preserved completely; the medium band in between a
 * light and a medium blur, where wrinkles and fine lines actually live,
 * is the only thing reduced; the medium-blurred base itself (overall
 * tone) is left for `smoothSkin` to handle. Reducing a frequency band
 * instead of just blurring harder is what keeps pores visible while
 * wrinkles fade — a stronger flat blur would take both together and
 * start looking plastic.
 */
export function applyWrinkleRemoval(source: HTMLCanvasElement, mask: HTMLCanvasElement, intensity: number, bounds?: Bounds): HTMLCanvasElement {
  const w = source.width
  const h = source.height
  const result = document.createElement('canvas')
  result.width = w
  result.height = h
  const rctx = result.getContext('2d')!
  rctx.drawImage(source, 0, 0)
  if (intensity <= 0.001) return result

  const { bx, by, bw, bh } = boundsRect(w, h, bounds)
  if (bw <= 0 || bh <= 0) return result

  const fine = blurredCopy(source, 3)
  const medium = blurredCopy(source, 9)

  const orig = source.getContext('2d')!.getImageData(bx, by, bw, bh)
  const fineData = fine.getContext('2d')!.getImageData(bx, by, bw, bh)
  const mediumData = medium.getContext('2d')!.getImageData(bx, by, bw, bh)
  const maskData = mask.getContext('2d')!.getImageData(bx, by, bw, bh)
  const out = new ImageData(bw, bh)
  const reduction = intensity * 0.7

  for (let i = 0; i < orig.data.length; i += 4) {
    const maskAlpha = maskData.data[i + 3] / 255
    for (let c = 0; c < 3; c++) {
      const high = orig.data[i + c] - fineData.data[i + c]
      const mediumBand = fineData.data[i + c] - mediumData.data[i + c]
      const newFine = mediumData.data[i + c] + mediumBand * (1 - reduction)
      const corrected = newFine + high
      out.data[i + c] = orig.data[i + c] + (corrected - orig.data[i + c]) * maskAlpha
    }
    out.data[i + 3] = orig.data[i + 3]
  }
  rctx.putImageData(out, bx, by)
  return result
}

/**
 * Mouth-corner fold/pouch (口角囊袋) smoothing: unlike the four effects
 * above, this isn't masked by the general skin mask — it builds its own
 * small radial mask centered on each mouth corner, since the fold runs
 * from there down past the corner rather than being spread across the
 * whole face. Blends toward a modest local blur within that small region
 * only.
 */
export function applyMouthCornerSmoothing(source: HTMLCanvasElement, corners: { x: number; y: number }[], radius: number, intensity: number): HTMLCanvasElement {
  const w = source.width
  const h = source.height
  const result = document.createElement('canvas')
  result.width = w
  result.height = h
  const rctx = result.getContext('2d')!
  rctx.drawImage(source, 0, 0)
  if (intensity <= 0.001 || corners.length === 0) return result

  const minX = Math.max(0, Math.floor(Math.min(...corners.map((c) => c.x)) - radius))
  const maxX = Math.min(w, Math.ceil(Math.max(...corners.map((c) => c.x)) + radius))
  const minY = Math.max(0, Math.floor(Math.min(...corners.map((c) => c.y)) - radius))
  const maxY = Math.min(h, Math.ceil(Math.max(...corners.map((c) => c.y)) + radius))
  const bw = maxX - minX
  const bh = maxY - minY
  if (bw <= 0 || bh <= 0) return result

  const maskCanvas = document.createElement('canvas')
  maskCanvas.width = w
  maskCanvas.height = h
  const mctx = maskCanvas.getContext('2d')!
  for (const c of corners) {
    const gradient = mctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, radius)
    gradient.addColorStop(0, 'rgba(255,255,255,1)')
    gradient.addColorStop(1, 'rgba(255,255,255,0)')
    mctx.fillStyle = gradient
    mctx.beginPath()
    mctx.arc(c.x, c.y, radius, 0, Math.PI * 2)
    mctx.fill()
  }

  const blurred = blurredCopy(source, radius * 0.35)
  const orig = source.getContext('2d')!.getImageData(minX, minY, bw, bh)
  const blurData = blurred.getContext('2d')!.getImageData(minX, minY, bw, bh)
  const maskData = mctx.getImageData(minX, minY, bw, bh)
  const out = new ImageData(bw, bh)

  for (let i = 0; i < orig.data.length; i += 4) {
    const amount = (maskData.data[i + 3] / 255) * intensity * 0.8
    for (let c = 0; c < 3; c++) {
      out.data[i + c] = orig.data[i + c] + (blurData.data[i + c] - orig.data[i + c]) * amount
    }
    out.data[i + 3] = orig.data[i + 3]
  }
  rctx.putImageData(out, minX, minY)
  return result
}
