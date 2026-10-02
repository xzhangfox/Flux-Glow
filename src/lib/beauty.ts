// Four independent skin/tone adjustments, separate from `smoothSkin`'s
// frequency-separation blemish/tone smoothing — each is mask-bounded the
// same way (only the face's own bounding box is ever touched, since the
// mask is zero everywhere else anyway) but does a genuinely different
// kind of correction rather than just being the same blur at another
// strength.
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
 * Builds a 256-bucket luminance histogram over a canvas's pixels that fall
 * inside a mask (`maskAlpha > 0.1`), and returns the value at a given
 * percentile (0-1). Used to find where "shadow" actually starts *for this
 * specific face*, instead of guessing a fixed brightness number that only
 * fits one lighting condition.
 */
export function luminancePercentile(data: Uint8ClampedArray, maskData: Uint8ClampedArray, percentile: number): number {
  const buckets = new Uint32Array(256)
  let total = 0
  for (let i = 0; i < data.length; i += 4) {
    if (maskData[i + 3] < 26) continue // ~0.1 alpha — skip pixels outside the face
    const lum = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114
    buckets[Math.round(lum)]++
    total++
  }
  if (total === 0) return 128
  const target = total * percentile
  let cumulative = 0
  for (let v = 0; v < 256; v++) {
    cumulative += buckets[v]
    if (cumulative >= target) return v
  }
  return 255
}

/**
 * Fill light: lifts shadows without flattening the face into a uniform
 * pale "mask" — the failure mode of a flat screen blend applied to every
 * skin pixel alike, which brightens an already-lit cheek by nearly the
 * same amount as a genuinely shadowed jawline or nose side and so erases
 * the light/shadow modeling that makes a face read as three-dimensional
 * in the first place.
 *
 * The first version of this function fixed that with a *local* exposure
 * map (blur at the scale real facial shadows occur at) but still compared
 * it against a fixed absolute brightness — which fails under ordinary
 * indoor lighting, where a whole face can sit below that fixed number at
 * once, so "lift anything darker than X" ends up meaning "lift the whole
 * face" again, the exact symptom this was meant to fix.
 *
 * The actual fix: judge "shadow" relative to *this face's own* tonal
 * range (its 15th/85th luminance percentiles), the same way Lightroom/
 * Camera Raw derive Shadows/Highlights behavior from the image's own
 * histogram rather than a hardcoded brightness value. A face that's
 * evenly lit — even if the whole frame is dim — has a narrow range, so
 * almost nothing registers as "below the shadow point" and the effect
 * correctly stays close to a no-op (dim-but-even lighting is an exposure
 * problem, not a shadow problem, and isn't fill light's job to fix). A
 * face lit from one side has a wide range, so the genuinely dark side
 * gets lifted while the lit side doesn't.
 *
 * In high quality mode the local exposure estimate itself is multi-scale
 * — three blurs at different radii, blended — rather than one single
 * radius, the same idea Multi-Scale Retinex uses to estimate illumination
 * (a single "surround" scale is a tradeoff: small enough to find a tight
 * crease like an under-eye shadow and it's too twitchy to read a broad
 * one-sided key-light shadow correctly, and vice versa). Blending a small,
 * medium, and large surround catches both without having to pick one.
 */
export function applyFillLight(canvas: HTMLCanvasElement, mask: HTMLCanvasElement, intensity: number, bounds?: Bounds, highQuality = true): void {
  if (intensity <= 0.001) return
  const { bx, by, bw, bh } = boundsRect(canvas.width, canvas.height, bounds)
  if (bw <= 0 || bh <= 0) return

  // Radius tuned to the scale facial shadows actually occur at, not
  // pixel-level darkness — this is what makes the lift map read as "is
  // this part of the face in shadow" rather than "is this exact pixel
  // dark", which would just reintroduce a texture-flattening blur. A
  // guided filter rather than a plain blur here too, so a hard real
  // boundary (hairline, jaw against the background, a glasses rim) can't
  // bleed its brightness into the exposure estimate on the other side of
  // it — eps is looser than the other effects' since this estimate is
  // meant to be broad/regional, not responsive to skin-texture-scale
  // variance.
  const shadowScale = Math.max(6, Math.round(bw * 0.12))
  // Three scales blended (not averaged blindly — the medium scale, tuned
  // to where real facial shadows live, carries the most weight) only in
  // high quality mode: each extra scale is another full blur/getImageData
  // round trip, affordable once for a static photo but not worth tripling
  // the live path's per-frame cost for a refinement that mostly shows up
  // at scales a reduced-resolution live preview is already smoothing over.
  const localLayers = highQuality
    ? [
        { data: edgeAwareBlur(canvas, bx, by, bw, bh, Math.max(3, Math.round(shadowScale * 0.5)), 2000), weight: 0.25 },
        { data: edgeAwareBlur(canvas, bx, by, bw, bh, shadowScale, 2000), weight: 0.5 },
        { data: edgeAwareBlur(canvas, bx, by, bw, bh, Math.round(shadowScale * 1.8), 2000), weight: 0.25 },
      ]
    : [{ data: croppedBlur(canvas, bx, by, bw, bh, shadowScale), weight: 1 }]

  const ctx = canvas.getContext('2d')!
  const orig = ctx.getImageData(bx, by, bw, bh)
  const maskData = mask.getContext('2d')!.getImageData(bx, by, bw, bh)
  const out = new ImageData(bw, bh)

  // The shadow point sits at this face's own 35th percentile — below that,
  // an area counts as genuinely shadowed *for this photo*; above it, it
  // doesn't, no matter how dim the overall shot is. `span` is the gap back
  // to the darkest 10% of the face, giving the falloff room to be smooth
  // instead of a hard cutoff right at the shadow point; it's floored so a
  // near-flat histogram (very evenly lit face) can't produce a near-zero
  // span and make ordinary texture noise register as "deep shadow".
  const shadowPoint = luminancePercentile(orig.data, maskData.data, 0.35)
  const darkPoint = luminancePercentile(orig.data, maskData.data, 0.1)
  const span = Math.max(28, shadowPoint - darkPoint)
  const MAX_LIFT = 0.5 // ceiling on the screen-blend amount even at intensity=1 in the darkest shadow

  for (let i = 0; i < orig.data.length; i += 4) {
    const maskAlpha = maskData.data[i + 3] / 255
    let localLum = 0
    for (const layer of localLayers) {
      localLum += (layer.data.data[i] * 0.299 + layer.data.data[i + 1] * 0.587 + layer.data.data[i + 2] * 0.114) * layer.weight
    }
    const t = Math.min(1, Math.max(0, (shadowPoint - localLum) / span))
    const shadowWeight = t * t * (3 - 2 * t) // smoothstep: soft falloff, no visible "shadow / not-shadow" boundary
    const liftAmount = shadowWeight * intensity * MAX_LIFT
    for (let c = 0; c < 3; c++) {
      const v = orig.data[i + c]
      const screened = 255 - (255 - v) * (1 - liftAmount)
      out.data[i + c] = v + (screened - v) * maskAlpha
    }
    out.data[i + 3] = orig.data[i + 3]
  }
  ctx.putImageData(out, bx, by)
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
