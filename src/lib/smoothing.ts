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
export function smoothSkin(source: HTMLCanvasElement, mask: HTMLCanvasElement, intensity: number): HTMLCanvasElement {
  const w = source.width
  const h = source.height
  const sctx = source.getContext('2d')!
  const mctx = mask.getContext('2d')!

  const low = blurredCopy(source, 5)
  const smoothedLow = blurredCopy(low, 7 + intensity * 10)

  const origData = sctx.getImageData(0, 0, w, h)
  const lowData = low.getContext('2d')!.getImageData(0, 0, w, h)
  const smoothedLowData = smoothedLow.getContext('2d')!.getImageData(0, 0, w, h)
  const maskData = mctx.getImageData(0, 0, w, h)

  const out = new ImageData(w, h)
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

  const result = document.createElement('canvas')
  result.width = w
  result.height = h
  result.getContext('2d')!.putImageData(out, 0, 0)
  return result
}
