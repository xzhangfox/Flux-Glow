// Makes the rendered 3D layer look photographed with the picture it sits in,
// rather than pasted over it. Clean CG gives itself away three ways: its
// blacks are pure black and its whites pure white (a photo's are lifted and
// tinted by the scene and the camera's processing), it has no sensor noise,
// and its edges are sharper than anything a phone lens resolves. So, over
// the head's bounding box:
//
// 1. Measure the photo: its black point and the colour of its shadows, its
//    white point and highlight colour, and its noise level.
// 2. Remap the layer's colours into that range — CG black becomes the
//    photo's own warm or cool shadow, highlights can't outshine the photo.
// 3. Add grain matched to the photo's noise.
// 4. Alpha-blend onto the photo (the layer arrives slightly softened, having
//    been rendered at reduced size and scaled up).

export interface Box {
  x: number
  y: number
  w: number
  h: number
}

let scratch: HTMLCanvasElement | null = null
// Live preview re-measures the photo only every few frames; the scene's
// tones don't change faster than that.
let cached: { stats: Stats; frames: number } | null = null

interface Stats {
  black: [number, number, number]
  white: [number, number, number]
  noise: number
}

function measure(photo: Uint8ClampedArray, w: number, h: number): Stats {
  // Luminance histogram over a subsample.
  const hist = new Uint32Array(256)
  let count = 0
  for (let y = 0; y < h; y += 3)
    for (let x = 0; x < w; x += 3) {
      const k = (y * w + x) * 4
      hist[(photo[k] * 77 + photo[k + 1] * 150 + photo[k + 2] * 29) >> 8]++
      count++
    }
  const pct = (q: number) => {
    let acc = 0
    for (let i = 0; i < 256; i++) {
      acc += hist[i]
      if (acc >= q * count) return i
    }
    return 255
  }
  const lo = pct(0.04)
  const hi = pct(0.985)
  // Average colour of the darkest and brightest pixels = the photo's shadow
  // and highlight tints.
  const b = [0, 0, 0]
  const wsum = [0, 0, 0]
  let nb = 0
  let nw = 0
  let noise = 0
  let nn = 0
  for (let y = 1; y < h - 1; y += 3)
    for (let x = 1; x < w - 1; x += 3) {
      const k = (y * w + x) * 4
      const l = (photo[k] * 77 + photo[k + 1] * 150 + photo[k + 2] * 29) >> 8
      if (l <= lo + 4) {
        b[0] += photo[k]
        b[1] += photo[k + 1]
        b[2] += photo[k + 2]
        nb++
      } else if (l >= hi - 4) {
        wsum[0] += photo[k]
        wsum[1] += photo[k + 1]
        wsum[2] += photo[k + 2]
        nw++
      }
      // Noise: deviation from the 4-neighbour mean, in mid-tones only
      // (edges and clipped areas would overstate it).
      if (l > 40 && l < 215) {
        const g = photo[k + 1]
        const m = (photo[k + 1 - 4] + photo[k + 1 + 4] + photo[k + 1 - w * 4] + photo[k + 1 + w * 4]) / 4
        const d = Math.abs(g - m)
        if (d < 18) {
          noise += d
          nn++
        }
      }
    }
  const avg = (s: number[], n: number, fallback: number): [number, number, number] =>
    n ? [s[0] / n / 255, s[1] / n / 255, s[2] / n / 255] : [fallback, fallback, fallback]
  const black = avg(b, nb, 0.03)
  const white = avg(wsum, nw, 0.97)
  return {
    // Don't let a very flat photo crush the layer: keep a sensible range.
    black: black.map((v) => Math.min(0.14, v)) as [number, number, number],
    white: white.map((v) => Math.max(0.78, v)) as [number, number, number],
    noise: nn ? Math.min(10, (noise / nn) * 1.1) : 2,
  }
}

/** Blends `layer` (same pixel size as `frame`) into `frame` within `box`. */
export function compositeAR(frame: HTMLCanvasElement, layer: CanvasImageSource, layerScale: number, box: Box, live = false) {
  const fctx = frame.getContext('2d', { willReadFrequently: true })!
  const { x, y, w, h } = box
  if (w < 2 || h < 2) return
  scratch ??= document.createElement('canvas')
  if (scratch.width !== w || scratch.height !== h) {
    scratch.width = w
    scratch.height = h
  }
  const sctx = scratch.getContext('2d', { willReadFrequently: true })!
  sctx.clearRect(0, 0, w, h)
  sctx.imageSmoothingEnabled = true
  sctx.imageSmoothingQuality = 'high'
  sctx.drawImage(layer, x * layerScale, y * layerScale, w * layerScale, h * layerScale, 0, 0, w, h)
  const L = sctx.getImageData(0, 0, w, h).data
  const img = fctx.getImageData(x, y, w, h)
  const P = img.data
  if (!live || !cached || ++cached.frames > 12) cached = { stats: measure(P, w, h), frames: 0 }
  const st = cached.stats
  const [br, bg, bb] = st.black
  const sr = st.white[0] - br
  const sg = st.white[1] - bg
  const sb = st.white[2] - bb
  const amp = st.noise * 1.7
  let seed = (Math.random() * 2 ** 31) | 0
  for (let i = 0; i < L.length; i += 4) {
    const a = L[i + 3] / 255
    if (a <= 0.003) continue
    // Remap into the photo's own range (partly — 60%, so props keep some
    // of their own contrast), then add matched grain.
    let r = L[i] / 255
    let g = L[i + 1] / 255
    let b = L[i + 2] / 255
    r = r + ((br + r * sr) - r) * 0.6
    g = g + ((bg + g * sg) - g) * 0.6
    b = b + ((bb + b * sb) - b) * 0.6
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    const n = ((seed / 0x7fffffff) - 0.5) * amp
    const o = 1 - a
    P[i] = P[i] * o + (r * 255 + n) * a
    P[i + 1] = P[i + 1] * o + (g * 255 + n) * a
    P[i + 2] = P[i + 2] * o + (b * 255 + n) * a
  }
  fctx.putImageData(img, x, y)
}
