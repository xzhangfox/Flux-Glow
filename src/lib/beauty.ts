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
 * Fill light: a screen blend (`255 - (255-x)*(1-amount)`), not a flat
 * brightness add — it lifts shadows noticeably while leaving highlights
 * close to where they already were, which is what a reflector/fill light
 * does to a face and a flat additive brighten doesn't (it would wash out
 * highlights by the same amount as shadows, looking hazy rather than lit).
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

  const orig = source.getContext('2d')!.getImageData(bx, by, bw, bh)
  const maskData = mask.getContext('2d')!.getImageData(bx, by, bw, bh)
  const out = new ImageData(bw, bh)
  const screenAmount = intensity * 0.5

  for (let i = 0; i < orig.data.length; i += 4) {
    const maskAlpha = maskData.data[i + 3] / 255
    for (let c = 0; c < 3; c++) {
      const v = orig.data[i + c]
      const screened = 255 - (255 - v) * (1 - screenAmount)
      out.data[i + c] = v + (screened - v) * maskAlpha
    }
    out.data[i + 3] = orig.data[i + 3]
  }
  rctx.putImageData(out, bx, by)
  return result
}

/**
 * Whitening: a lighter screen blend than fill light, plus pulling color
 * toward its own luminance (desaturating) rather than toward a fixed pale
 * color — so it reduces ruddiness/sallowness generically instead of
 * pushing every skin tone toward the same target shade.
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
  const screenAmount = intensity * 0.22
  const desatAmount = intensity * 0.35

  for (let i = 0; i < orig.data.length; i += 4) {
    const maskAlpha = maskData.data[i + 3] / 255
    const r = orig.data[i]
    const g = orig.data[i + 1]
    const b = orig.data[i + 2]
    const screenedR = 255 - (255 - r) * (1 - screenAmount)
    const screenedG = 255 - (255 - g) * (1 - screenAmount)
    const screenedB = 255 - (255 - b) * (1 - screenAmount)
    const luminance = screenedR * 0.299 + screenedG * 0.587 + screenedB * 0.114
    const finalR = screenedR + (luminance - screenedR) * desatAmount
    const finalG = screenedG + (luminance - screenedG) * desatAmount
    const finalB = screenedB + (luminance - screenedB) * desatAmount
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
