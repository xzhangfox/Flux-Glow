// A single consolidated pass for the live 30fps preview, covering the same
// four tone effects as whitening/acne/wrinkle/smoothSkin in beauty.ts and
// smoothing.ts, with the same math — but reading the face's pixel data and
// mask ONCE and writing the result ONCE, instead of each effect doing its
// own getImageData/putImageData pair. (Fill light isn't part of this pass
// — it's a GPU shader pass in meshWarp.ts now, using the mesh's own 3D
// vertex normals for real directional shading, equally cheap for live and
// static since it's a shader, not a CPU blur.)
//
// Fusing the four effects' own reads/writes turned out not to be the main
// win: measured, the dominant cost was the *blur* operations themselves
// (acne's local average, wrinkle's two bands, smoothing's two layers —
// five `croppedBlur` calls total, each a real getImageData/GPU-readback),
// not the handful of extra reads around them. Those on a ~500x600 face
// region still cost real time even after consolidating everything else,
// far past a 33ms budget.
//
// So this also downsamples the whole working region once before running
// any of the blurs, and upsamples the single corrected result back at the
// end — cutting every one of those readbacks' pixel counts by the square
// of the scale factor (4x at the default half-resolution), the same "work
// at reduced resolution for an edge-preserving/blur-based operation, then
// upsample" idea the guided filter's own fast variant uses (see
// guidedFilter.ts), applied here across the whole pass instead of inside
// a single filter. This only reduces the live *preview's* resolution for
// these effects — the captured/confirmed photo still runs the full-
// resolution sequential guided-filter pipeline.
//
// This path is deliberately lower-fidelity than the per-function static
// pipeline in one more way beyond the downsampling and the Gaussian-vs-
// guided-filter difference (see beauty.ts): each effect's blur layer is
// computed from the frame's original pixels rather than re-blurring after
// every prior effect in the chain (the sequential composition the static
// pipeline does). This only matters where an earlier effect meaningfully
// shifted the local tone before a later effect's blur was taken; at
// everyday slider settings the divergence from the static pipeline's
// result is small, and widens mainly when several effects are pushed
// toward their maximum simultaneously — an acceptable live-preview
// approximation given the captured photo is unaffected.

import { buildWhiteningLUT, applyLUT } from './whiteningLUT'

export interface LiveToneParams {
  whitening: number
  acneRemoval: number
  wrinkleRemoval: number
  smoothness: number
}

function blurSmall(source: HTMLCanvasElement, radiusPx: number): ImageData {
  const canvas = document.createElement('canvas')
  canvas.width = source.width
  canvas.height = source.height
  const ctx = canvas.getContext('2d')!
  ctx.filter = `blur(${radiusPx}px)`
  ctx.drawImage(source, 0, 0)
  return ctx.getImageData(0, 0, canvas.width, canvas.height)
}

/** The frame size (long side) the blur radii here were tuned at. */
const TUNED_SIZE = 1600

export function applyLiveTonePass(canvas: HTMLCanvasElement, mask: HTMLCanvasElement, params: LiveToneParams, bounds: { minX: number; minY: number; maxX: number; maxY: number }): void {
  const bx = Math.max(0, Math.floor(bounds.minX))
  const by = Math.max(0, Math.floor(bounds.minY))
  const bw = Math.min(canvas.width, Math.ceil(bounds.maxX)) - bx
  const bh = Math.min(canvas.height, Math.ceil(bounds.maxY)) - by
  if (bw <= 0 || bh <= 0) return

  const needWhitening = params.whitening > 0.001
  const needAcne = params.acneRemoval > 0.001
  const needWrinkle = params.wrinkleRemoval > 0.001
  const needSmooth = params.smoothness > 0.001
  if (!needWhitening && !needAcne && !needWrinkle && !needSmooth) return

  // Same CIELAB-based whitening transform the static pipeline uses (see
  // beauty.ts/whiteningLUT.ts), via the same LUT — building the LUT costs
  // a few ms regardless of image size (it's a fixed 17x17x17 grid, not
  // per-pixel), so it's affordable here too; what the live path skips is
  // the static path's extra frequency-separation blur, not the transform
  // itself, so live and captured whitening now actually look the same.
  const whiteningLut = needWhitening ? buildWhiteningLUT(params.whitening) : null
  const whitenedPixel: [number, number, number] = [0, 0, 0]

  // Half resolution once the region is big enough that it matters — for
  // an already-small bounds rect (e.g. editing a small/cropped photo) the
  // blurs are cheap regardless and downsampling would only cost quality
  // for no real speed gain.
  // And scaled with the frame beyond the size the radii were tuned at, so a
  // full-resolution photo gets the same look as its editing-size copy.
  const scale = (bw * bh > 120_000 ? 2 : 1) * Math.max(1, Math.max(canvas.width, canvas.height) / TUNED_SIZE)
  const sw = Math.max(1, Math.round(bw / scale))
  const sh = Math.max(1, Math.round(bh / scale))

  const small = document.createElement('canvas')
  small.width = sw
  small.height = sh
  const sctx = small.getContext('2d')!
  sctx.drawImage(canvas, bx, by, bw, bh, 0, 0, sw, sh)
  const orig = sctx.getImageData(0, 0, sw, sh)

  const smallMask = document.createElement('canvas')
  smallMask.width = sw
  smallMask.height = sh
  smallMask.getContext('2d')!.drawImage(mask, bx, by, bw, bh, 0, 0, sw, sh)
  const maskData = smallMask.getContext('2d')!.getImageData(0, 0, sw, sh)

  // Each effect's own blur layer(s), now at the reduced resolution — still
  // one canvas round trip per distinct radius (unavoidable: a CSS blur
  // needs an actual canvas), but exactly one per effect, not one per
  // effect per channel of math, and 1/scale^2 the pixels of before.
  const acneLocal = needAcne ? blurSmall(small, Math.max(1, Math.round(14 / scale))) : null

  let wrinkleFine: ImageData | null = null
  let wrinkleMedium: ImageData | null = null
  if (needWrinkle) {
    wrinkleFine = blurSmall(small, Math.max(1, Math.round(3 / scale)))
    wrinkleMedium = blurSmall(small, Math.max(1, Math.round(9 / scale)))
  }

  let smoothLow: ImageData | null = null
  let smoothTone: ImageData | null = null
  if (needSmooth) {
    smoothLow = blurSmall(small, Math.max(1, Math.round(5 / scale)))
    const lowCanvas = document.createElement('canvas')
    lowCanvas.width = sw
    lowCanvas.height = sh
    lowCanvas.getContext('2d')!.putImageData(smoothLow, 0, 0)
    const toneRadius = Math.max(1, Math.round((7 + params.smoothness * 10) / scale))
    smoothTone = blurSmall(lowCanvas, toneRadius)
  }

  const out = new ImageData(sw, sh)
  // (reduced resolution: the change instead, split by sign — see below)
  const neg = scale === 1 ? null : new ImageData(sw, sh)
  const DEVIATION_THRESHOLD = 16
  const DEVIATION_RANGE = 40
  const wrinkleReduction = params.wrinkleRemoval * 0.7

  for (let i = 0; i < orig.data.length; i += 4) {
    const maskAlpha = maskData.data[i + 3] / 255
    let r = orig.data[i]
    let g = orig.data[i + 1]
    let b = orig.data[i + 2]

    if (whiteningLut) {
      applyLUT(r, g, b, whiteningLut, whitenedPixel)
      r = r + (whitenedPixel[0] - r) * maskAlpha
      g = g + (whitenedPixel[1] - g) * maskAlpha
      b = b + (whitenedPixel[2] - b) * maskAlpha
    }

    if (acneLocal) {
      const avgR = acneLocal.data[i]
      const avgG = acneLocal.data[i + 1]
      const avgB = acneLocal.data[i + 2]
      const deviation = Math.abs(r - avgR) + Math.abs(g - avgG) + Math.abs(b - avgB)
      const excess = Math.max(0, deviation - DEVIATION_THRESHOLD)
      const pullAmount = Math.min(1, excess / DEVIATION_RANGE) * params.acneRemoval * maskAlpha
      r = r + (avgR - r) * pullAmount
      g = g + (avgG - g) * pullAmount
      b = b + (avgB - b) * pullAmount
    }

    if (wrinkleFine && wrinkleMedium) {
      const fineR = wrinkleFine.data[i]
      const fineG = wrinkleFine.data[i + 1]
      const fineB = wrinkleFine.data[i + 2]
      const medR = wrinkleMedium.data[i]
      const medG = wrinkleMedium.data[i + 1]
      const medB = wrinkleMedium.data[i + 2]
      const highR = r - fineR
      const highG = g - fineG
      const highB = b - fineB
      const correctedR = medR + (fineR - medR) * (1 - wrinkleReduction) + highR
      const correctedG = medG + (fineG - medG) * (1 - wrinkleReduction) + highG
      const correctedB = medB + (fineB - medB) * (1 - wrinkleReduction) + highB
      r = r + (correctedR - r) * maskAlpha
      g = g + (correctedG - g) * maskAlpha
      b = b + (correctedB - b) * maskAlpha
    }

    if (smoothLow && smoothTone) {
      const amount = maskAlpha * params.smoothness
      const lowR = smoothLow.data[i]
      const lowG = smoothLow.data[i + 1]
      const lowB = smoothLow.data[i + 2]
      const toneR = smoothTone.data[i]
      const toneG = smoothTone.data[i + 1]
      const toneB = smoothTone.data[i + 2]
      r = Math.max(0, Math.min(255, lowR * (1 - amount) + toneR * amount + (r - lowR)))
      g = Math.max(0, Math.min(255, lowG * (1 - amount) + toneG * amount + (g - lowG)))
      b = Math.max(0, Math.min(255, lowB * (1 - amount) + toneB * amount + (b - lowB)))
    }

    if (scale === 1) {
      out.data[i] = r
      out.data[i + 1] = g
      out.data[i + 2] = b
      out.data[i + 3] = orig.data[i + 3]
    } else {
      // The change, not the picture: what's added in `out`, what's taken
      // away in `neg` (clamped bytes can't go below 0).
      const dr = r - orig.data[i]
      const dg = g - orig.data[i + 1]
      const db = b - orig.data[i + 2]
      out.data[i] = dr
      out.data[i + 1] = dg
      out.data[i + 2] = db
      out.data[i + 3] = 255
      neg!.data[i] = -dr
      neg!.data[i + 1] = -dg
      neg!.data[i + 2] = -db
      neg!.data[i + 3] = 255
    }
  }

  const ctx = canvas.getContext('2d')!
  if (scale === 1) {
    ctx.putImageData(out, bx, by)
    return
  }
  // At reduced resolution it's the change that gets scaled back up and
  // applied to the full-resolution pixels — the smoothing and toning are
  // soft by nature, but eyes, brows, lashes and lips inside the face's box
  // keep every bit of their own detail. (Drawing the reduced picture itself
  // back over the box blurred all of them.) All on the GPU, no readback:
  // add with 'lighter'; subtract as invert ('difference' with white), add,
  // invert back.
  sctx.putImageData(out, 0, 0)
  const nctx = smallMask.getContext('2d')!
  nctx.putImageData(neg!, 0, 0)
  ctx.save()
  ctx.fillStyle = '#fff'
  ctx.globalCompositeOperation = 'difference'
  ctx.fillRect(bx, by, bw, bh)
  ctx.globalCompositeOperation = 'lighter'
  ctx.drawImage(smallMask, 0, 0, sw, sh, bx, by, bw, bh)
  ctx.globalCompositeOperation = 'difference'
  ctx.fillRect(bx, by, bw, bh)
  ctx.globalCompositeOperation = 'lighter'
  ctx.drawImage(small, 0, 0, sw, sh, bx, by, bw, bh)
  ctx.restore()
}
