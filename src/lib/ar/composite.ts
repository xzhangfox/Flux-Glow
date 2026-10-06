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
  /** Mean saturation of the photo (0–1). */
  sat: number
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
  let sat = 0
  let ns = 0
  for (let y = 1; y < h - 1; y += 3)
    for (let x = 1; x < w - 1; x += 3) {
      const k = (y * w + x) * 4
      const l = (photo[k] * 77 + photo[k + 1] * 150 + photo[k + 2] * 29) >> 8
      const mx = Math.max(photo[k], photo[k + 1], photo[k + 2])
      if (mx > 20) {
        sat += (mx - Math.min(photo[k], photo[k + 1], photo[k + 2])) / mx
        ns++
      }
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
  // The shadows' level, but only a hint of their hue: the darkest pixels
  // are often hair or a dark doorway, and their colour cast on a black
  // mask reads as that mask being made of them.
  const dark = avg(b, nb, 0.03)
  const dl = 0.3 * dark[0] + 0.59 * dark[1] + 0.11 * dark[2]
  const black = dark.map((v) => dl + (v - dl) * 0.35) as [number, number, number]
  const white = avg(wsum, nw, 0.97)
  return {
    // Don't let a very flat photo crush the layer: keep a sensible range.
    black: black.map((v) => Math.min(0.14, v)) as [number, number, number],
    white: white.map((v) => Math.max(0.78, v)) as [number, number, number],
    noise: nn ? Math.min(10, (noise / nn) * 1.1) : 2,
    sat: ns ? sat / ns : 0.3,
  }
}

/** The 3D layer (rendered at `layerScale` of the frame's size) over `box`,
 *  at the frame's own scale — read back from the GPU once a frame, for
 *  everything that needs its pixels. */
export function readLayer(layer: CanvasImageSource, layerScale: number, box: Box): HTMLCanvasElement {
  const { x, y, w, h } = box
  scratch ??= document.createElement('canvas')
  if (scratch.width !== w || scratch.height !== h) {
    scratch.width = Math.max(1, w)
    scratch.height = Math.max(1, h)
  }
  const sctx = scratch.getContext('2d', { willReadFrequently: true })!
  sctx.clearRect(0, 0, w, h)
  sctx.imageSmoothingEnabled = true
  sctx.imageSmoothingQuality = 'high'
  sctx.drawImage(layer, x * layerScale, y * layerScale, w * layerScale, h * layerScale, 0, 0, w, h)
  return scratch
}

/** Blends `layer` (from readLayer) into `frame` within `box`. */
export function compositeAR(frame: HTMLCanvasElement, layer: HTMLCanvasElement, box: Box, live = false) {
  const fctx = frame.getContext('2d', { willReadFrequently: true })!
  const { x, y, w, h } = box
  if (w < 2 || h < 2) return
  const L = layer.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, w, h).data
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
  // A muted photo mutes the props too (CG colour is cleaner than any
  // camera's): pull saturation toward the photo's own.
  const desat = Math.min(0.45, Math.max(0, 0.32 - st.sat) * 1.6)
  // Light wrap: near the layer's outline, the background's own light
  // bleeds over the edge, as it does round anything photographed against
  // it. Both at quarter resolution — it's all soft.
  const wrap = lightWrap(L, P, w, h)
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
    if (desat > 0) {
      const lum = 0.3 * r + 0.59 * g + 0.11 * b
      r += (lum - r) * desat
      g += (lum - g) * desat
      b += (lum - b) * desat
    }
    if (wrap) {
      const q = wrap.at(i >> 2)
      if (q.k > 0.01) {
        r += (q.r - r) * q.k
        g += (q.g - g) * q.k
        b += (q.b - b) * q.k
      }
    }
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    const n = ((seed / 0x7fffffff) - 0.5) * amp
    const o = 1 - a
    P[i] = P[i] * o + (r * 255 + n) * a
    P[i + 1] = P[i + 1] * o + (g * 255 + n) * a
    P[i + 2] = P[i + 2] * o + (b * 255 + n) * a
  }
  fctx.putImageData(img, x, y)
}

/** Light wrap for the layer L over photo P (both w×h RGBA): for each
 *  pixel, the blurred background colour and how much of it to mix in —
 *  strongest just inside the layer's outline, fading inward. */
function lightWrap(L: Uint8ClampedArray, P: Uint8ClampedArray, w: number, h: number) {
  const S = 4
  const sw = Math.max(2, Math.ceil(w / S))
  const sh = Math.max(2, Math.ceil(h / S))
  const alpha = new Float32Array(sw * sh)
  const bg = new Float32Array(sw * sh * 4)
  for (let y = 0; y < sh; y++)
    for (let x = 0; x < sw; x++) {
      const k = (Math.min(h - 1, y * S) * w + Math.min(w - 1, x * S)) * 4
      const a = L[k + 3] / 255
      const i = y * sw + x
      alpha[i] = a
      // Background only where the layer isn't.
      const wb = 1 - a
      bg[i * 4] = (P[k] / 255) * wb
      bg[i * 4 + 1] = (P[k + 1] / 255) * wb
      bg[i * 4 + 2] = (P[k + 2] / 255) * wb
      bg[i * 4 + 3] = wb
    }
  const blur = (src: Float32Array, ch: number, r: number) => {
    const tmp = new Float32Array(src.length)
    const out = new Float32Array(src.length)
    for (let y = 0; y < sh; y++)
      for (let x = 0; x < sw; x++)
        for (let c = 0; c < ch; c++) {
          let s = 0
          for (let d = -r; d <= r; d++) s += src[(y * sw + Math.min(sw - 1, Math.max(0, x + d))) * ch + c]
          tmp[(y * sw + x) * ch + c] = s / (2 * r + 1)
        }
    for (let y = 0; y < sh; y++)
      for (let x = 0; x < sw; x++)
        for (let c = 0; c < ch; c++) {
          let s = 0
          for (let d = -r; d <= r; d++) s += tmp[(Math.min(sh - 1, Math.max(0, y + d)) * sw + x) * ch + c]
          out[(y * sw + x) * ch + c] = s / (2 * r + 1)
        }
    return out
  }
  const R = Math.max(2, Math.round(Math.min(sw, sh) / 40))
  const bgB = blur(bg, 4, R)
  let any = false
  for (let i = 0; i < alpha.length; i++) if (alpha[i] > 0.5 && bgB[i * 4 + 3] > 0.05) any = true
  if (!any) return null
  const q = { r: 0, g: 0, b: 0, k: 0 }
  return {
    at(i: number) {
      const x = Math.min(sw - 1, Math.floor((i % w) / S))
      const y = Math.min(sh - 1, Math.floor(i / w / S))
      const j = y * sw + x
      const wgt = bgB[j * 4 + 3]
      if (wgt < 0.02) {
        q.k = 0
        return q
      }
      q.r = bgB[j * 4] / wgt
      q.g = bgB[j * 4 + 1] / wgt
      q.b = bgB[j * 4 + 2] / wgt
      q.k = Math.min(0.2, wgt * 0.45) * alpha[j]
      return q
    },
  }
}

let sample: HTMLCanvasElement | null = null

/** The live preview's composite: the same tone match as compositeAR, but
 *  done with the canvas's own (GPU) blending instead of reading pixels
 *  back — multiply by each channel's scale, add each channel's offset,
 *  re-masked to the layer's own coverage each time. No grain or light
 *  wrap (the camera's own noise and motion hide their absence). */
export function compositeARFast(frame: HTMLCanvasElement, layer: CanvasImageSource, layerScale: number, box: Box) {
  const { x, y, w, h } = box
  if (w < 2 || h < 2) return
  const fctx = frame.getContext('2d')!
  if (!cached || ++cached.frames > 12) {
    // Measure on a small copy of the box: a cheap read-back.
    sample ??= document.createElement('canvas')
    const k = Math.min(1, 160 / Math.max(w, h))
    sample.width = Math.max(2, Math.round(w * k))
    sample.height = Math.max(2, Math.round(h * k))
    const sc = sample.getContext('2d', { willReadFrequently: true })!
    sc.drawImage(frame, x, y, w, h, 0, 0, sample.width, sample.height)
    cached = { stats: measure(sc.getImageData(0, 0, sample.width, sample.height).data, sample.width, sample.height), frames: 0 }
  }
  const st = cached.stats
  scratch ??= document.createElement('canvas')
  if (scratch.width !== w || scratch.height !== h) {
    scratch.width = w
    scratch.height = h
  }
  const s = scratch.getContext('2d')!
  s.globalCompositeOperation = 'source-over'
  s.clearRect(0, 0, w, h)
  s.imageSmoothingEnabled = true
  s.drawImage(layer, x * layerScale, y * layerScale, w * layerScale, h * layerScale, 0, 0, w, h)
  // c' = c·(0.4 + 0.6·range) + 0.6·black, per channel.
  const to255 = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255)
  const mul = [0, 1, 2].map((c) => 0.4 + 0.6 * (st.white[c] - st.black[c]))
  const add = [0, 1, 2].map((c) => 0.6 * st.black[c])
  s.globalCompositeOperation = 'multiply'
  s.fillStyle = `rgb(${to255(mul[0])},${to255(mul[1])},${to255(mul[2])})`
  s.fillRect(0, 0, w, h)
  s.globalCompositeOperation = 'destination-in'
  s.drawImage(layer, x * layerScale, y * layerScale, w * layerScale, h * layerScale, 0, 0, w, h)
  s.globalCompositeOperation = 'lighter'
  s.fillStyle = `rgb(${to255(add[0])},${to255(add[1])},${to255(add[2])})`
  s.fillRect(0, 0, w, h)
  s.globalCompositeOperation = 'destination-in'
  s.drawImage(layer, x * layerScale, y * layerScale, w * layerScale, h * layerScale, 0, 0, w, h)
  s.globalCompositeOperation = 'source-over'
  fctx.drawImage(scratch, x, y)
}
