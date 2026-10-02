import { edgeAwareBlur, croppedBlur } from './guidedFilter'

// Variance threshold (pixel-value-squared) for the guided filter below: a
// local neighborhood whose variance clears this is a real boundary
// (eyebrow/eyelid/nose/lip edge, roughly a 20+ gray-level jump) and stays
// close to untouched; well under it is ordinary blotchy skin tone and
// smooths toward the local mean. Tuned empirically against a synthetic
// sharp-edge test (see the direct-pipeline tests) rather than guessed.
const TONE_EDGE_EPS = 700

/**
 * True frequency separation, not a flat masked blur: `high = original -
 * lightlyBlurred` isolates fine texture (pores, fine lines), which is kept
 * untouched; a SECOND pass over that same low layer removes the blotchy
 * tone/color variation blemishes and redness actually are, and the two
 * recombine as `smoothedLow + high`.
 *
 * That second pass is a guided filter (see guidedFilter.ts), not another
 * Gaussian blur — a plain blur has no notion of a real edge and smooths
 * straight across eyebrows, eyelids, and the nose/lip boundary just as
 * readily as it smooths flat cheek skin, which is the actual mechanism
 * behind heavily-smoothed skin reading as flat/plastic (this is also
 * exactly how real "surface blur" skin-smoothing in shipped beauty camera
 * pipelines differs from a naive blur-based approach). The guided filter
 * keeps real boundaries sharp while still averaging away blotchy tone in
 * between them.
 *
 * `bounds`, when given, is the face's own bounding box (padded well past
 * the mask's blur radius) — background, hair, and everything else outside
 * it is mask-value zero anyway, i.e. a guaranteed no-op in the blend
 * formula below, so skipping it entirely (both the expensive
 * getImageData/putImageData calls and the per-pixel loop) changes nothing
 * about the result and is often a large chunk of the frame to not pay for.
 *
 * `highQuality` picks the tone pass above: the guided filter is
 * meaningfully better (see above) but, being a CPU per-pixel operation,
 * costs on the order of 100ms+ per call at a typical face-region size —
 * fine for a one-time static-photo edit, far too slow to run every frame
 * of a live 30fps preview alongside the other effects that also need it.
 * Live preview falls back to the original Gaussian blur for this step so
 * the viewfinder stays responsive; the captured photo gets the accurate
 * pass.
 *
 * Mutates `canvas` in place over just the bounds rect, like the beauty.ts
 * effects — see the note at the top of that file for why.
 */
export function smoothSkin(
  canvas: HTMLCanvasElement,
  mask: HTMLCanvasElement,
  intensity: number,
  bounds?: { minX: number; minY: number; maxX: number; maxY: number },
  highQuality = true,
): void {
  const w = canvas.width
  const h = canvas.height

  const bx = bounds ? Math.max(0, Math.floor(bounds.minX)) : 0
  const by = bounds ? Math.max(0, Math.floor(bounds.minY)) : 0
  const bw = (bounds ? Math.min(w, Math.ceil(bounds.maxX)) : w) - bx
  const bh = (bounds ? Math.min(h, Math.ceil(bounds.maxY)) : h) - by
  if (bw <= 0 || bh <= 0) return

  const lowData = croppedBlur(canvas, bx, by, bw, bh, 5)
  const lowCanvas = document.createElement('canvas')
  lowCanvas.width = bw
  lowCanvas.height = bh
  lowCanvas.getContext('2d')!.putImageData(lowData, 0, 0)

  const toneRadius = Math.round(7 + intensity * 10)
  const smoothedLowData = highQuality ? edgeAwareBlur(lowCanvas, 0, 0, bw, bh, toneRadius, TONE_EDGE_EPS) : croppedBlur(lowCanvas, 0, 0, bw, bh, toneRadius)

  const ctx = canvas.getContext('2d')!
  const origData = ctx.getImageData(bx, by, bw, bh)
  const maskData = mask.getContext('2d')!.getImageData(bx, by, bw, bh)

  const out = new ImageData(bw, bh)
  const orig = origData.data
  const lowArr = lowData.data
  const smoothArr = smoothedLowData.data
  const maskArr = maskData.data
  const outArr = out.data

  for (let i = 0; i < orig.length; i += 4) {
    const amount = (maskArr[i + 3] / 255) * intensity
    for (let c = 0; c < 3; c++) {
      const high = orig[i + c] - lowArr[i + c]
      const blendedLow = lowArr[i + c] * (1 - amount) + smoothArr[i + c] * amount
      outArr[i + c] = Math.max(0, Math.min(255, blendedLow + high))
    }
    outArr[i + 3] = orig[i + 3]
  }

  ctx.putImageData(out, bx, by)
}
