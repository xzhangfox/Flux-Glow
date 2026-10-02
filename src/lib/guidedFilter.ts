// He, Sun & Tang, "Guided Image Filtering" (2013) — the standard fast
// edge-preserving smoothing filter real beauty-camera "surface blur"
// implementations use in place of a plain Gaussian blur. A Gaussian blur
// has no notion of "this is a real edge" and smooths straight across
// eyebrows, eyelids, and the nose/lip boundary exactly as readily as it
// smooths flat cheek skin — that indifference is what makes heavily
// smoothed skin read as flattened/plastic. The guided filter instead fits
// a local linear model (q = a*I + b) in a window around each pixel: where
// the window is near-uniform (flat skin, blotchy tone), it behaves like a
// mean filter; where the window straddles a real boundary (high local
// variance), `a` approaches 1 and the output stays close to the original
// pixel instead of blending across the edge. `eps` is the variance
// threshold between those two regimes.
//
// This also uses the "fast guided filter" variant from He & Sun's 2015
// follow-up paper: the linear-model coefficients a/b are solved at a
// reduced resolution and then upsampled, rather than at full resolution.
// a/b vary smoothly almost everywhere (they only need to react sharply
// right at genuine edges, and the downsampling step still resolves those
// at full-res boundary positions once applied back to the full-res pixel)
// so solving them at a fraction of the pixel count is a large speed win
// with little visible cost — necessary here because a naive full-res
// version was measured at several hundred milliseconds on a single
// region of a live video frame, far past a usable per-frame budget.

function boxBlurH(src: Float32Array, w: number, h: number, r: number, dst: Float32Array) {
  const windowSize = 2 * r + 1
  for (let y = 0; y < h; y++) {
    const row = y * w
    let sum = 0
    for (let k = -r; k <= r; k++) sum += src[row + Math.min(w - 1, Math.max(0, k))]
    dst[row] = sum / windowSize
    for (let x = 1; x < w; x++) {
      const addX = Math.min(w - 1, x + r)
      const subX = Math.max(0, x - r - 1)
      sum += src[row + addX] - src[row + subX]
      dst[row + x] = sum / windowSize
    }
  }
}

function boxBlurV(src: Float32Array, w: number, h: number, r: number, dst: Float32Array) {
  const windowSize = 2 * r + 1
  for (let x = 0; x < w; x++) {
    let sum = 0
    for (let k = -r; k <= r; k++) sum += src[Math.min(h - 1, Math.max(0, k)) * w + x]
    dst[x] = sum / windowSize
    for (let y = 1; y < h; y++) {
      const addY = Math.min(h - 1, y + r)
      const subY = Math.max(0, y - r - 1)
      sum += src[addY * w + x] - src[subY * w + x]
      dst[y * w + x] = sum / windowSize
    }
  }
}

function boxBlur2D(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(w * h)
  const dst = new Float32Array(w * h)
  boxBlurH(src, w, h, r, tmp)
  boxBlurV(tmp, w, h, r, dst)
  return dst
}

/** Area-average downsample of one channel straight out of image data — box
 * averaging is the right prefilter here (vs. nearest/bilinear) since it
 * anti-aliases instead of just discarding samples, which matters for the
 * variance computation downstream. */
function downsampleChannel(data: Uint8ClampedArray, w: number, h: number, channelOffset: number, dw: number, dh: number): Float32Array {
  const dst = new Float32Array(dw * dh)
  const counts = new Float32Array(dw * dh)
  const xRatio = dw / w
  const yRatio = dh / h
  for (let y = 0; y < h; y++) {
    const dy = Math.min(dh - 1, Math.floor(y * yRatio))
    for (let x = 0; x < w; x++) {
      const dx = Math.min(dw - 1, Math.floor(x * xRatio))
      const idx = dy * dw + dx
      dst[idx] += data[(y * w + x) * 4 + channelOffset]
      counts[idx]++
    }
  }
  for (let i = 0; i < dst.length; i++) dst[i] /= Math.max(1, counts[i])
  return dst
}

function bilinearUpsample(src: Float32Array, sw: number, sh: number, dw: number, dh: number): Float32Array {
  const dst = new Float32Array(dw * dh)
  const xRatio = sw / dw
  const yRatio = sh / dh
  for (let y = 0; y < dh; y++) {
    const sy = (y + 0.5) * yRatio - 0.5
    const sy0 = Math.max(0, Math.min(sh - 1, Math.floor(sy)))
    const sy1 = Math.min(sh - 1, sy0 + 1)
    const fy = Math.max(0, Math.min(1, sy - sy0))
    for (let x = 0; x < dw; x++) {
      const sx = (x + 0.5) * xRatio - 0.5
      const sx0 = Math.max(0, Math.min(sw - 1, Math.floor(sx)))
      const sx1 = Math.min(sw - 1, sx0 + 1)
      const fx = Math.max(0, Math.min(1, sx - sx0))
      const v00 = src[sy0 * sw + sx0]
      const v10 = src[sy0 * sw + sx1]
      const v01 = src[sy1 * sw + sx0]
      const v11 = src[sy1 * sw + sx1]
      const top = v00 + (v10 - v00) * fx
      const bot = v01 + (v11 - v01) * fx
      dst[y * dw + x] = top + (bot - top) * fy
    }
  }
  return dst
}

function linearModel(I: Float32Array, w: number, h: number, radius: number, eps: number): { a: Float32Array; b: Float32Array } {
  const n = w * h
  const I2 = new Float32Array(n)
  for (let i = 0; i < n; i++) I2[i] = I[i] * I[i]
  const meanI = boxBlur2D(I, w, h, radius)
  const meanI2 = boxBlur2D(I2, w, h, radius)
  const a = new Float32Array(n)
  const b = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const varI = meanI2[i] - meanI[i] * meanI[i]
    const ai = varI / (varI + eps)
    a[i] = ai
    b[i] = meanI[i] * (1 - ai)
  }
  return { a: boxBlur2D(a, w, h, radius), b: boxBlur2D(b, w, h, radius) }
}

/**
 * Runs the self-guided filter independently on R/G/B over a canvas
 * region, returning an `ImageData` of just that region — a drop-in
 * edge-preserving replacement for `blurredCopy(...).getImageData(...)` at
 * call sites that need a local *tone* estimate rather than a blind blur.
 * `eps` is in raw 0-255 pixel-value-squared units: a window whose local
 * variance clears it is treated as a real edge and mostly kept; well
 * below it, treated as flat/texture and smoothed toward the local mean.
 */
export function edgeAwareBlur(source: HTMLCanvasElement, bx: number, by: number, bw: number, bh: number, radius: number, eps: number): ImageData {
  const data = source.getContext('2d')!.getImageData(bx, by, bw, bh).data
  const out = new ImageData(bw, bh)

  // Only subsample for radii large enough that full-res box blurs would
  // actually be expensive — small radii (fine pore-scale texture) are
  // cheap at full res anyway and subsampling them would just throw away
  // precision for no speed benefit.
  const subsample = Math.max(1, Math.min(4, Math.round(radius / 4)))
  const sw = Math.max(1, Math.round(bw / subsample))
  const sh = Math.max(1, Math.round(bh / subsample))
  const smallRadius = Math.max(1, Math.round(radius / subsample))

  for (let ch = 0; ch < 3; ch++) {
    const fullRes = new Float32Array(bw * bh)
    for (let i = 0, p = 0; i < data.length; i += 4, p++) fullRes[p] = data[i + ch]

    let a: Float32Array
    let b: Float32Array
    if (subsample === 1) {
      ;({ a, b } = linearModel(fullRes, bw, bh, radius, eps))
    } else {
      const small = downsampleChannel(data, bw, bh, ch, sw, sh)
      const small2 = linearModel(small, sw, sh, smallRadius, eps)
      a = bilinearUpsample(small2.a, sw, sh, bw, bh)
      b = bilinearUpsample(small2.b, sw, sh, bw, bh)
    }

    for (let p = 0; p < fullRes.length; p++) {
      out.data[p * 4 + ch] = a[p] * fullRes[p] + b[p]
    }
  }
  for (let i = 0; i < data.length; i += 4) out.data[i + 3] = data[i + 3]
  return out
}
