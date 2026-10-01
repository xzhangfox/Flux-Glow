function blurredCopy(source: HTMLCanvasElement, radiusPx: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = source.width
  canvas.height = source.height
  const ctx = canvas.getContext('2d')!
  ctx.filter = `blur(${radiusPx}px)`
  ctx.drawImage(source, 0, 0)
  return canvas
}

/**
 * True frequency separation, not a flat masked blur: `high = original -
 * lightlyBlurred` isolates fine texture (pores, fine lines), which is kept
 * untouched; a SECOND, heavier blur of that same low layer removes the
 * blotchy tone/color variation blemishes and redness actually are, and the
 * two recombine as `smoothedLow + high`. This is what keeps retouched skin
 * from reading as flat/plastic — the texture survives, only the blotchy
 * tone underneath it gets smoothed, which doubles as blemish reduction
 * without a separate spot-removal tool.
 */
/**
 * `bounds`, when given, is the face's own bounding box (padded well past
 * the mask's blur radius) — background, hair, and everything else outside
 * it is mask-value zero anyway, i.e. a guaranteed no-op in the blend
 * formula below, so skipping it entirely (both the expensive
 * getImageData/putImageData calls and the per-pixel loop) changes nothing
 * about the result and is often a large chunk of the frame to not pay for.
 */
export function smoothSkin(source: HTMLCanvasElement, mask: HTMLCanvasElement, intensity: number, bounds?: { minX: number; minY: number; maxX: number; maxY: number }): HTMLCanvasElement {
  const w = source.width
  const h = source.height
  const sctx = source.getContext('2d')!
  const mctx = mask.getContext('2d')!

  const result = document.createElement('canvas')
  result.width = w
  result.height = h
  const rctx = result.getContext('2d')!
  rctx.drawImage(source, 0, 0)

  const bx = bounds ? Math.max(0, Math.floor(bounds.minX)) : 0
  const by = bounds ? Math.max(0, Math.floor(bounds.minY)) : 0
  const bw = (bounds ? Math.min(w, Math.ceil(bounds.maxX)) : w) - bx
  const bh = (bounds ? Math.min(h, Math.ceil(bounds.maxY)) : h) - by
  if (bw <= 0 || bh <= 0) return result

  const low = blurredCopy(source, 5)
  const smoothedLow = blurredCopy(low, 7 + intensity * 10)

  const origData = sctx.getImageData(bx, by, bw, bh)
  const lowData = low.getContext('2d')!.getImageData(bx, by, bw, bh)
  const smoothedLowData = smoothedLow.getContext('2d')!.getImageData(bx, by, bw, bh)
  const maskData = mctx.getImageData(bx, by, bw, bh)

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

  rctx.putImageData(out, bx, by)
  return result
}
