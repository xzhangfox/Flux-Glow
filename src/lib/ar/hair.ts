import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision'
import type { Box } from './composite'

// What a head-covering mask hides. A hood or cowl pulled over the head
// flattens the hair and covers the ears, so whatever of them the photo
// shows outside the mask's silhouette has to go, or it reads as poking
// through. Two of MediaPipe's segmenters (self-hosted, like the face model)
// find it:
// • hair, removed wherever it lies outside the mask;
// • the person, removed just outside the mask's edge (ears, the head and
//   neck where they're wider than the mask) above a line the mask sets —
//   below it, neck and shoulders are left alone.
// What's removed is painted over with the background around it, filled by
// push-pull interpolation at low resolution: average the known pixels (not
// the person, not under the mask) into a pyramid, then fill each unknown
// pixel from the coarsest level that has data. It reconstructs walls,
// doors and blurred rooms well — the smooth, low-detail backgrounds of
// selfies — and the mattes' soft edges blend it in.

const WASM_BASE_URL = '/mediapipe/wasm'
// Work resolution: the fill is smooth, so the long side of the head box is
// processed at this size and scaled back up.
const WORK = 240
const SEG = 512

const segmenters: { hair: ImageSegmenter | null; person: ImageSegmenter | null } = { hair: null, person: null }
let loading: Promise<void> | null = null

/** Loads the segmenters (once). Never rejects: without them, masks just
 *  render without hiding hair. */
export function preloadHair(): Promise<void> {
  loading ??= (async () => {
    try {
      const fileset = await FilesetResolver.forVisionTasks(WASM_BASE_URL)
      const make = (model: string) =>
        ImageSegmenter.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `/mediapipe/${model}`, delegate: 'GPU' },
          runningMode: 'IMAGE',
          outputConfidenceMasks: true,
          outputCategoryMask: false,
        })
      ;[segmenters.hair, segmenters.person] = await Promise.all([make('hair_segmenter.tflite'), make('selfie_segmenter.tflite')])
    } catch (err) {
      console.warn('Segmenters unavailable', err)
    }
  })()
  return loading
}

function canvas2d(w: number, h: number, c?: HTMLCanvasElement | null) {
  c ??= document.createElement('canvas')
  if (c.width !== w || c.height !== h) {
    c.width = w
    c.height = h
  }
  return c
}

const C: Record<'seg' | 'work' | 'keep' | 'zone' | 'out', HTMLCanvasElement | null> = { seg: null, work: null, keep: null, zone: null, out: null }

// Live preview: each matte is reused for a frame (the head doesn't move far
// in 1/30 s, and the mattes are dilated anyway); the two segmenters take
// turns.
const cache: Record<'hair' | 'person' | 'protect', { m: Float32Array; box: Box; w: number; h: number; age: number } | null> = { hair: null, person: null, protect: null }
let turn = 0

/** `kind`'s confidence over `box` of `frame`, at w × h. */
function matte(kind: 'hair' | 'person' | 'protect', frame: HTMLCanvasElement, box: Box, w: number, h: number, live: boolean, maxAge = 1, alone = false): Float32Array | null {
  const seg = segmenters[kind === 'hair' ? 'hair' : 'person']
  if (!seg) return null
  const c = cache[kind]
  const fresh = c && c.w === w && c.h === h && Math.abs(c.box.x - box.x) < box.w * 0.04 && Math.abs(c.box.y - box.y) < box.h * 0.04
  if (live && fresh && c.age < maxAge && (alone || kind === 'protect' || (turn & 1) === (kind === 'hair' ? 1 : 0))) {
    c.age++
    return c.m
  }
  const k = Math.min(1, SEG / Math.max(box.w, box.h))
  const sw = Math.max(1, Math.round(box.w * k))
  const sh = Math.max(1, Math.round(box.h * k))
  C.seg = canvas2d(sw, sh, C.seg)
  C.seg.getContext('2d')!.drawImage(frame, box.x, box.y, box.w, box.h, 0, 0, sw, sh)
  const result = seg.segment(C.seg)
  const masks = result.confidenceMasks
  // The last category is the foreground (hair; person).
  const conf = masks && masks[masks.length - 1]?.getAsFloat32Array()
  result.close()
  if (!conf) return null
  const m = new Float32Array(w * h)
  for (let y = 0; y < h; y++) {
    const fy = Math.min(sh - 1.001, Math.max(0, ((y + 0.5) / h) * sh - 0.5))
    const y0 = Math.floor(fy)
    const ty = fy - y0
    for (let x = 0; x < w; x++) {
      const fx = Math.min(sw - 1.001, Math.max(0, ((x + 0.5) / w) * sw - 0.5))
      const x0 = Math.floor(fx)
      const tx = fx - x0
      const a = conf[y0 * sw + x0] * (1 - tx) + conf[y0 * sw + x0 + 1] * tx
      const b = conf[(y0 + 1) * sw + x0] * (1 - tx) + conf[(y0 + 1) * sw + x0 + 1] * tx
      m[y * w + x] = a * (1 - ty) + b * ty
    }
  }
  cache[kind] = { m, box: { ...box }, w, h, age: 0 }
  return m
}

/** Running max over a window of radius r along a strided line of n
 *  values (van Herk / Gil–Werman: block-wise prefix and suffix maxima, so
 *  the cost doesn't grow with r). Edges are clamped. */
function maxLine(src: Float32Array, dst: Float32Array, start: number, stride: number, n: number, r: number, g: Float32Array, hh: Float32Array) {
  const k = 2 * r + 1
  const m = n + 2 * r
  const at = (i: number) => src[start + Math.min(n - 1, Math.max(0, i - r)) * stride]
  for (let i = 0; i < m; i++) g[i] = i % k === 0 ? at(i) : Math.max(g[i - 1], at(i))
  for (let i = m - 1; i >= 0; i--) hh[i] = i === m - 1 || (i + 1) % k === 0 ? at(i) : Math.max(hh[i + 1], at(i))
  for (let j = 0; j < n; j++) dst[start + j * stride] = Math.max(hh[j], g[j + k - 1])
}

/** Box blur of radius r (two passes, rows then columns). */
function boxBlur(src: Float32Array, w: number, h: number, r: number) {
  const tmp = new Float32Array(w * h)
  const out = new Float32Array(w * h)
  for (let y = 0; y < h; y++) {
    let s = 0
    for (let x = -r; x <= r; x++) s += src[y * w + Math.min(w - 1, Math.max(0, x))]
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = s / (2 * r + 1)
      s += src[y * w + Math.min(w - 1, x + r + 1)] - src[y * w + Math.max(0, x - r)]
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0
    for (let y = -r; y <= r; y++) s += tmp[Math.min(h - 1, Math.max(0, y)) * w + x]
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / (2 * r + 1)
      s += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x]
    }
  }
  return out
}

/** Max filter (dilation) with a square of radius r. */
export function dilate(src: Float32Array, w: number, h: number, r: number) {
  if (r <= 0) return src
  const tmp = new Float32Array(w * h)
  const out = new Float32Array(w * h)
  const g = new Float32Array(Math.max(w, h) + 2 * r)
  const hh = new Float32Array(Math.max(w, h) + 2 * r)
  for (let y = 0; y < h; y++) maxLine(src, tmp, y * w, 1, w, r, g, hh)
  for (let x = 0; x < w; x++) maxLine(tmp, out, x, w, h, r, g, hh)
  return out
}

/** Push-pull fill: colours where weight is 1 are kept, the rest filled
 *  from their known surroundings. rgb: 3 floats per pixel. */
export function pushPull(rgb: Float32Array, wt: Float32Array, w: number, h: number): Float32Array {
  const levels: { c: Float32Array; w: Float32Array; W: number; H: number }[] = [{ c: rgb, w: wt, W: w, H: h }]
  while (levels[levels.length - 1].W > 1 || levels[levels.length - 1].H > 1) {
    const p = levels[levels.length - 1]
    const W = Math.max(1, Math.ceil(p.W / 2))
    const H = Math.max(1, Math.ceil(p.H / 2))
    const c = new Float32Array(W * H * 3)
    const ww = new Float32Array(W * H)
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let s = 0
        let r = 0
        let g = 0
        let b = 0
        for (let dy = 0; dy < 2; dy++)
          for (let dx = 0; dx < 2; dx++) {
            const xx = x * 2 + dx
            const yy = y * 2 + dy
            if (xx >= p.W || yy >= p.H) continue
            const i = yy * p.W + xx
            const q = p.w[i]
            s += q
            r += p.c[i * 3] * q
            g += p.c[i * 3 + 1] * q
            b += p.c[i * 3 + 2] * q
          }
        const i = y * W + x
        if (s > 0) {
          c[i * 3] = r / s
          c[i * 3 + 1] = g / s
          c[i * 3 + 2] = b / s
        }
        ww[i] = Math.min(1, s)
      }
    levels.push({ c, w: ww, W, H })
  }
  // Pull: blend each level with the (bilinearly upsampled) coarser fill.
  let filled = levels[levels.length - 1].c
  for (let l = levels.length - 2; l >= 0; l--) {
    const L = levels[l]
    const up = levels[l + 1]
    const out = new Float32Array(L.W * L.H * 3)
    for (let y = 0; y < L.H; y++) {
      const fy = Math.min(up.H - 1, Math.max(0, (y + 0.5) / 2 - 0.5))
      const y0 = Math.floor(fy)
      const y1 = Math.min(up.H - 1, y0 + 1)
      const ty = fy - y0
      for (let x = 0; x < L.W; x++) {
        const fx = Math.min(up.W - 1, Math.max(0, (x + 0.5) / 2 - 0.5))
        const x0 = Math.floor(fx)
        const x1 = Math.min(up.W - 1, x0 + 1)
        const tx = fx - x0
        const i = y * L.W + x
        const q = L.w[i]
        for (let ch = 0; ch < 3; ch++) {
          const a = filled[(y0 * up.W + x0) * 3 + ch] * (1 - tx) + filled[(y0 * up.W + x1) * 3 + ch] * tx
          const b = filled[(y1 * up.W + x0) * 3 + ch] * (1 - tx) + filled[(y1 * up.W + x1) * 3 + ch] * tx
          out[i * 3 + ch] = L.c[i * 3 + ch] * q + (a * (1 - ty) + b * ty) * (1 - q)
        }
      }
    }
    filled = out
  }
  return filled
}

const ramp = (v: number, a: number, b: number) => Math.min(1, Math.max(0, (v - a) / (b - a)))

export interface HideHead {
  /** Draws (white, frame pixels) what must stay visible besides the mask
   *  itself: the face its openings show. */
  keep: (ctx: CanvasRenderingContext2D) => void
  /** Draws the region where the person may be removed (above the neck). */
  zone: (ctx: CanvasRenderingContext2D) => void
  /** How far past the mask's edge the person is removed (frame pixels). */
  margin: number
  /** Only hair, and only inside `zone` (a wig: the head under it stays, and
   *  so does hair hanging below it). One segmenter instead of two, its
   *  matte reused for a few live frames. */
  hairOnly?: boolean
}

/** Paints the background over hair and head showing outside a mask, in
 *  `box` of `frame`. `layer`: the 3D render over the box (see readLayer),
 *  whose coverage is the mask's silhouette. */
export function hideHead(frame: HTMLCanvasElement, box: Box, layer: HTMLCanvasElement, opts: HideHead, live: boolean) {
  if (box.w < 8 || box.h < 8) return
  turn++
  const k = Math.min(1, WORK / Math.max(box.w, box.h))
  const w = Math.max(4, Math.round(box.w * k))
  const h = Math.max(4, Math.round(box.h * k))
  const hairRaw = matte('hair', frame, box, w, h, live, opts.hairOnly ? 3 : 1, opts.hairOnly)
  const person = opts.hairOnly ? null : matte('person', frame, box, w, h, live)
  if (!hairRaw && !person) return
  const n = w * h

  // Coverage: the mask's render plus the face.
  C.keep = canvas2d(w, h, C.keep)
  const kc = C.keep.getContext('2d', { willReadFrequently: true })!
  kc.setTransform(1, 0, 0, 1, 0, 0)
  kc.clearRect(0, 0, w, h)
  kc.drawImage(layer, 0, 0, box.w, box.h, 0, 0, w, h)
  kc.setTransform(k, 0, 0, k, -box.x * k, -box.y * k)
  kc.fillStyle = '#fff'
  opts.keep(kc)
  const keepA = kc.getImageData(0, 0, w, h).data
  C.zone = canvas2d(w, h, C.zone)
  const zc = C.zone.getContext('2d', { willReadFrequently: true })!
  zc.setTransform(1, 0, 0, 1, 0, 0)
  zc.clearRect(0, 0, w, h)
  zc.setTransform(k, 0, 0, k, -box.x * k, -box.y * k)
  zc.fillStyle = '#fff'
  opts.zone(zc)
  const zoneA = zc.getImageData(0, 0, w, h).data
  const n0 = w * h
  // (soft at its edge for a wig: where hair is kept and where it goes
  // mustn't be a line)
  let zoneF = new Float32Array(n0)
  for (let i = 0; i < n0; i++) zoneF[i] = zoneA[i * 4 + 3] / 255
  if (opts.hairOnly) zoneF = boxBlur(zoneF, w, h, Math.max(2, Math.round(h * 0.04)))

  const kept = new Float32Array(n)
  for (let i = 0; i < n; i++) kept[i] = keepA[i * 4 + 3] / 255
  // Near the mask: within `margin` of its edge.
  const near = dilate(kept, w, h, Math.max(1, Math.round(opts.margin * k)))
  // Hair, grown a little to take the stray strands round its edge; and a
  // wider band of it kept out of the fill's samples.
  const hair = hairRaw ? dilate(hairRaw, w, h, 2) : new Float32Array(n)
  const hairWide = hairRaw ? dilate(hairRaw, w, h, 4) : new Float32Array(n)

  C.work = canvas2d(w, h, C.work)
  const wc = C.work.getContext('2d', { willReadFrequently: true })!
  wc.drawImage(frame, box.x, box.y, box.w, box.h, 0, 0, w, h)
  const px = wc.getImageData(0, 0, w, h)
  const rgb = new Float32Array(n * 3)
  const wt = new Float32Array(n)
  const remove = new Float32Array(n)
  let any = false
  for (let i = 0; i < n; i++) {
    const zone = zoneF[i]
    const pers = person ? person[i] : 0
    const head = ramp(pers, 0.35, 0.7) * zone * near[i]
    const x = i % w
    const y = (i / w) | 0
    // Fade out at the box's edges rather than end in a hard line.
    const edge = Math.min(1, Math.min(x, y, w - 1 - x, h - 1 - y) / Math.max(4, Math.min(w, h) * 0.08))
    // Under the mask's own soft edge too, or hair shows through its fringe.
    remove[i] = Math.max(ramp(hair[i], 0.1, 0.4) * (opts.hairOnly ? zone : 1), head) * (1 - ramp(kept[i], 0.5, 0.95)) * edge
    if (remove[i] > 0.02) any = true
    // Known background: not hair, not the person, not under the mask
    // (whose edge pixels are the face's and hair's colours, which mustn't
    // bleed outward).
    wt[i] = (1 - ramp(hairWide[i], 0.03, 0.2)) * (1 - ramp(pers, 0.2, 0.5)) * (kept[i] > 0.02 ? 0 : 1)
    rgb[i * 3] = px.data[i * 4]
    rgb[i * 3 + 1] = px.data[i * 4 + 1]
    rgb[i * 3 + 2] = px.data[i * 4 + 2]
  }
  if (!any) return
  const fill = pushPull(rgb, wt, w, h)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const debug = (globalThis as any).__arDebug === 'hair'
  for (let i = 0; i < n; i++) {
    px.data[i * 4] = debug ? 255 * remove[i] : fill[i * 3]
    px.data[i * 4 + 1] = debug ? 255 * kept[i] : fill[i * 3 + 1]
    px.data[i * 4 + 2] = debug ? 255 * zoneA[i * 4 + 3] / 510 : fill[i * 3 + 2]
    px.data[i * 4 + 3] = debug ? 220 : remove[i] * 255
  }
  C.out = canvas2d(w, h, C.out)
  C.out.getContext('2d')!.putImageData(px, 0, 0)
  const ctx = frame.getContext('2d')!
  ctx.save()
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(C.out, 0, 0, w, h, box.x, box.y, box.w, box.h)
  ctx.restore()
  // A still gets the photo's grain over the fill (live frames skip it: it
  // costs a full-size read-back, and the camera's own noise hides the lack).
  if (!live && !debug) addGrain(frame, box, remove, w, h)
}

/** Sensor-like noise over the filled areas, matched to the photo's own. */
function addGrain(frame: HTMLCanvasElement, box: Box, remove: Float32Array, w: number, h: number) {
  const ctx = frame.getContext('2d', { willReadFrequently: true })!
  const img = ctx.getImageData(box.x, box.y, box.w, box.h)
  const d = img.data
  const W = box.w
  // The photo's noise: deviation from the 4-neighbour mean, sampled where
  // nothing was filled.
  let noise = 0
  let nn = 0
  for (let y = 2; y < box.h - 2; y += 4)
    for (let x = 2; x < W - 2; x += 4) {
      if (remove[Math.min(h - 1, ((y / box.h) * h) | 0) * w + Math.min(w - 1, ((x / W) * w) | 0)] > 0.01) continue
      const k = (y * W + x) * 4 + 1
      const dev = Math.abs(d[k] - (d[k - 4] + d[k + 4] + d[k - W * 4] + d[k + W * 4]) / 4)
      if (dev < 18) {
        noise += dev
        nn++
      }
    }
  const amp = (nn ? noise / nn : 2) * 1.8
  let seed = 12345
  for (let y = 0; y < box.h; y++) {
    const ry = Math.min(h - 1, ((y / box.h) * h) | 0)
    for (let x = 0; x < W; x++) {
      const r = remove[ry * w + Math.min(w - 1, ((x / W) * w) | 0)]
      if (r < 0.02) continue
      seed = (seed * 1103515245 + 12345) & 0x7fffffff
      const n = ((seed / 0x7fffffff) - 0.5) * amp * r
      const k = (y * W + x) * 4
      d[k] += n
      d[k + 1] += n
      d[k + 2] += n
    }
  }
  ctx.putImageData(img, box.x, box.y)
}

export const personSegmenterReady = () => !!segmenters.person

/** The person's confidence over `box` of `frame`, at w × h (for background
 *  protection; live frames reuse it every other frame). */
// Live, its edge is smoothed over time — a fresh matte each frame jitters
// by a pixel or two, which made the protected edge shimmer.
let lastProtect: { m: Float32Array; w: number; h: number; box: Box } | null = null
export function personMatte(frame: HTMLCanvasElement, box: Box, w: number, h: number, live: boolean) {
  const m = matte('protect', frame, box, w, h, live)
  if (!m || !live) return m
  const p = lastProtect
  const out = new Float32Array(m.length)
  if (p && p.w === w && p.h === h && Math.abs(p.box.x - box.x) < box.w * 0.03 && Math.abs(p.box.y - box.y) < box.h * 0.03)
    for (let i = 0; i < m.length; i++) out[i] = m[i] * 0.55 + p.m[i] * 0.45
  else out.set(m)
  lastProtect = { m: out, w, h, box: { ...box } }
  return out
}
