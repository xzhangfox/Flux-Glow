// Three independent skin/tone adjustments, separate from `smoothSkin`'s
// frequency-separation blemish/tone smoothing — each is mask-bounded the
// same way (only the face's own bounding box is ever touched, since the
// mask is zero everywhere else anyway) but does a genuinely different
// kind of correction rather than just being the same blur at another
// strength. (Fill light used to live here too — it's now a GPU shader
// pass in meshWarp.ts, since real directional relighting needs the mesh's
// 3D vertex normals, which only exist in that pass.)
//
// Each function mutates `canvas` in place over just its bounds rect
// instead of allocating a full-frame copy and returning it. The pipeline
// chains several of these per frame; allocating and redrawing an entire
// 1280x1280 live frame for each one (when the face's own bounding box is
// often a fraction of that) was the dominant cost in the live preview
// loop, measured at 200-350ms for four effects together — this is what
// brings that down to something a 30fps loop can actually afford.

import { edgeAwareBlur, croppedBlur } from './guidedFilter'
import { buildWhiteningLUT, applyLUT } from './whiteningLUT'

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
 * Whitening/tone-evening: corrects the two things that actually read as
 * "uneven, ruddy skin" — localized redness around the nose and cheeks,
 * and dull midtones — instead of a flat screen blend plus a global
 * desaturate-toward-luminance. The old approach ignored *where* the
 * unevenness actually was and instead washed color out of the whole face
 * at once, which is indistinguishable from a cheap pale overlay once the
 * intensity went up.
 *
 * The color mapping itself (redness-excess correction on a*, midtone-
 * weighted lift on L*) runs through a 3D LUT (see whiteningLUT.ts), not a
 * live per-pixel Lab round trip — the same mechanism real camera color
 * pipelines use a LUT for: build the transform once on a coarse grid,
 * then every actual pixel is a cheap trilinear lookup, no transcendental
 * math per pixel. Decoupling L*, a*, and b* is still what makes the transform
 * itself correct (a lightness lift that can't help nudging hue the way
 * adjusting R/G/B independently would); the LUT is what makes doing that
 * at full image resolution affordable.
 *
 * The LUT only touches a *low-frequency* base layer, not the original
 * pixels directly — `edgeAwareBlur` (the guided filter already used
 * elsewhere in this file) splits the face into a low-frequency tone layer
 * and a high-frequency detail layer (pores, fine texture), the whitening
 * map applies to the low layer only, and the untouched detail recombines
 * on top at full strength afterward. Mapping the raw pixels directly
 * would run the same lightness/redness correction on pore-level noise
 * too, which is how aggressive whitening ends up looking waxy — flattened
 * micro-contrast, not just brighter, even though nothing here blurs the
 * final image (the detail layer is added back exactly, unblurred).
 */
export function applyWhitening(canvas: HTMLCanvasElement, mask: HTMLCanvasElement, intensity: number, bounds?: Bounds): void {
  if (intensity <= 0.001) return
  const { bx, by, bw, bh } = boundsRect(canvas.width, canvas.height, bounds)
  if (bw <= 0 || bh <= 0) return

  const lut = buildWhiteningLUT(intensity)
  const lowData = edgeAwareBlur(canvas, bx, by, bw, bh, 12, 700) // same tone-vs-edge eps as smoothing.ts's TONE_EDGE_EPS

  const ctx = canvas.getContext('2d')!
  const orig = ctx.getImageData(bx, by, bw, bh)
  const maskData = mask.getContext('2d')!.getImageData(bx, by, bw, bh)
  const out = new ImageData(bw, bh)
  const mapped: [number, number, number] = [0, 0, 0]

  for (let i = 0; i < orig.data.length; i += 4) {
    const maskAlpha = maskData.data[i + 3] / 255
    const r = orig.data[i]
    const g = orig.data[i + 1]
    const b = orig.data[i + 2]
    if (maskAlpha < 0.004) {
      out.data[i] = r
      out.data[i + 1] = g
      out.data[i + 2] = b
      out.data[i + 3] = orig.data[i + 3]
      continue
    }

    const lowR = lowData.data[i]
    const lowG = lowData.data[i + 1]
    const lowB = lowData.data[i + 2]
    applyLUT(lowR, lowG, lowB, lut, mapped)
    const finalR = mapped[0] + (r - lowR)
    const finalG = mapped[1] + (g - lowG)
    const finalB = mapped[2] + (b - lowB)

    out.data[i] = r + (finalR - r) * maskAlpha
    out.data[i + 1] = g + (finalG - g) * maskAlpha
    out.data[i + 2] = b + (finalB - b) * maskAlpha
    out.data[i + 3] = orig.data[i + 3]
  }
  ctx.putImageData(out, bx, by)
}

/**
 * Acne/blemish removal: unlike `smoothSkin` (which smooths tone
 * everywhere in the mask uniformly), this only pulls a pixel toward its
 * local average once it deviates from that average by more than a
 * threshold — ordinary skin texture stays untouched, and only actual
 * outliers (a spot, a red mark) get corrected. This is the same idea real
 * blemish-removal tools use (match the spot's luminosity/color to its
 * surroundings rather than blur it away), and the "local average" here is
 * a guided filter rather than a Gaussian blur specifically so a strong
 * real edge (nostril shadow, eyebrow) never reads as a false "deviation"
 * and gets incorrectly smoothed — a plain blur's average right next to an
 * edge is a muddy mix of both sides, which would otherwise flag the
 * boundary itself as a blemish.
 *
 * The radius (14px) is deliberately bigger than a typical blemish itself:
 * a filter radius comparable to or smaller than the blemish just averages
 * the blemish with itself and barely registers any deviation at all
 * (confirmed empirically — at radius 7 a clearly brighter ~8px spot barely
 * moved after "correction" because the local average was computed mostly
 * from the spot's own pixels). A wider radius means the average actually
 * reflects the surrounding normal skin the blemish sits on top of, while
 * the guided filter's edge-awareness still keeps it from reaching across
 * a real nearby feature boundary despite the larger radius.
 */
export function applyAcneRemoval(canvas: HTMLCanvasElement, mask: HTMLCanvasElement, intensity: number, bounds?: Bounds, highQuality = true): void {
  if (intensity <= 0.001) return
  const { bx, by, bw, bh } = boundsRect(canvas.width, canvas.height, bounds)
  if (bw <= 0 || bh <= 0) return

  const lowData = highQuality ? edgeAwareBlur(canvas, bx, by, bw, bh, 14, 300) : croppedBlur(canvas, bx, by, bw, bh, 14)
  const ctx = canvas.getContext('2d')!
  const orig = ctx.getImageData(bx, by, bw, bh)
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
  ctx.putImageData(out, bx, by)
}

/**
 * Wrinkle removal: a three-band split — fine texture (pores, a very light
 * blur's residual) is preserved completely; the medium band, where
 * wrinkles and fine lines actually live, is the only thing reduced; the
 * medium layer's own base tone is left for `smoothSkin` to handle.
 * Reducing a frequency band instead of just blurring harder is what keeps
 * pores visible while wrinkles fade.
 *
 * Both bands are extracted with a guided filter, not a Gaussian blur —
 * and deliberately the *same kind* of filter for both fine and medium,
 * not a Gaussian "fine" paired with a guided "medium". A Gaussian blur's
 * response right at a real edge is a difference-of-Gaussians edge
 * detector by construction, so a real boundary (eyelid crease, mouth
 * corner) shows up as a large medium-band value and gets its contrast
 * *reduced* like a wrinkle — and mixing filter types made it worse,
 * producing outright overshoot (confirmed empirically: a synthetic sharp
 * edge came out with MORE contrast after processing, not less, because
 * the Gaussian-blurred fine layer and the edge-preserving medium layer
 * disagreed about how sharp the edge should be and their difference
 * spiked right at the boundary). With both layers edge-aware, a real
 * edge makes both filters track the original value closely at roughly
 * the same radius, so their difference — the "medium band" — correctly
 * collapses to ~0 exactly where it shouldn't be touched, while still
 * responding normally to genuine wrinkle-scale contrast in between.
 */
export function applyWrinkleRemoval(canvas: HTMLCanvasElement, mask: HTMLCanvasElement, intensity: number, bounds?: Bounds, highQuality = true): void {
  if (intensity <= 0.001) return
  const { bx, by, bw, bh } = boundsRect(canvas.width, canvas.height, bounds)
  if (bw <= 0 || bh <= 0) return

  const fineData = highQuality ? edgeAwareBlur(canvas, bx, by, bw, bh, 3, 200) : croppedBlur(canvas, bx, by, bw, bh, 3)
  const mediumData = highQuality ? edgeAwareBlur(canvas, bx, by, bw, bh, 9, 500) : croppedBlur(canvas, bx, by, bw, bh, 9)

  const ctx = canvas.getContext('2d')!
  const orig = ctx.getImageData(bx, by, bw, bh)
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
  ctx.putImageData(out, bx, by)
}

/**
 * Mouth-corner fold/pouch (口角囊袋) smoothing: unlike the four effects
 * above, this isn't masked by the general skin mask — it builds its own
 * small radial mask centered on each mouth corner, since the fold runs
 * from there down past the corner rather than being spread across the
 * whole face. Blends toward a modest local blur within that small region
 * only.
 */
export function applyMouthCornerSmoothing(canvas: HTMLCanvasElement, corners: { x: number; y: number }[], radius: number, intensity: number): void {
  if (intensity <= 0.001 || corners.length === 0) return

  const minX = Math.max(0, Math.floor(Math.min(...corners.map((c) => c.x)) - radius))
  const maxX = Math.min(canvas.width, Math.ceil(Math.max(...corners.map((c) => c.x)) + radius))
  const minY = Math.max(0, Math.floor(Math.min(...corners.map((c) => c.y)) - radius))
  const maxY = Math.min(canvas.height, Math.ceil(Math.max(...corners.map((c) => c.y)) + radius))
  const bw = maxX - minX
  const bh = maxY - minY
  if (bw <= 0 || bh <= 0) return

  // Mask built directly at the small region's own size (not the full
  // frame) — the gradients just need shifting by (-minX, -minY).
  const maskCanvas = document.createElement('canvas')
  maskCanvas.width = bw
  maskCanvas.height = bh
  const mctx = maskCanvas.getContext('2d')!
  for (const c of corners) {
    const cx = c.x - minX
    const cy = c.y - minY
    const gradient = mctx.createRadialGradient(cx, cy, 0, cx, cy, radius)
    gradient.addColorStop(0, 'rgba(255,255,255,1)')
    gradient.addColorStop(1, 'rgba(255,255,255,0)')
    mctx.fillStyle = gradient
    mctx.beginPath()
    mctx.arc(cx, cy, radius, 0, Math.PI * 2)
    mctx.fill()
  }

  const blurData = croppedBlur(canvas, minX, minY, bw, bh, radius * 0.35)
  const ctx = canvas.getContext('2d')!
  const orig = ctx.getImageData(minX, minY, bw, bh)
  const maskData = mctx.getImageData(0, 0, bw, bh)
  const out = new ImageData(bw, bh)

  for (let i = 0; i < orig.data.length; i += 4) {
    const amount = (maskData.data[i + 3] / 255) * intensity * 0.8
    for (let c = 0; c < 3; c++) {
      out.data[i + c] = orig.data[i + c] + (blurData.data[i + c] - orig.data[i + c]) * amount
    }
    out.data[i + 3] = orig.data[i + 3]
  }
  ctx.putImageData(out, minX, minY)
}
