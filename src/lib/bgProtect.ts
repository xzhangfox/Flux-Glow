import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { OVAL_LOOP, type ReshapeParams } from './deform'
import { renderMeshWarp } from './meshWarp'
import { dilate, personMatte, personSegmenterReady, preloadHair, pushPull } from './ar/hair'

// Background protection, the way beauty cameras (Meitu's 背景保护) do it.
// The face warp moves pixels inside a ring around the face, so slimming a
// cheek also bends whatever is beside it: a door frame, a shelf, the edge
// of a window. With protection on, only the person is warped:
//
// 1. the person is segmented around the face (MediaPipe's selfie
//    segmenter, the same one the head masks use);
// 2. that matte is warped with the very same mesh as the photo, so it
//    tracks the slimmed outline exactly;
// 3. a clean plate is made — the original frame with the person painted
//    out by push-pull fill from the background around them;
// 4. the warped person is laid over the clean plate. Background the warp
//    would have stretched is the original, untouched; what slimming
//    uncovers (where the cheek used to be) comes from the fill.

/** The work resolution for the matte and fill (long side of the box). */
const WORK = 256
/** The width of the full-frame matte that gets warped. */
const MASK_W = 360
/** How far past the warp's outer ring the protected box reaches. */
const SKIRT = 0.28
const MARGIN = 0.12

function canvas(w: number, h: number) {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  return c
}

const ramp = (v: number, a: number, b: number) => Math.min(1, Math.max(0, (v - a) / (b - a)))

/** Whether protection can run now (it starts loading the segmenter if
 *  not: a frame or two later it can). */
export function backgroundProtectionReady() {
  if (personSegmenterReady()) return true
  void preloadHair()
  return false
}

/** `warped`: `source` reshaped with `params` at `landmarks`. Returns it
 *  with the background restored. */
export function protectBackground(source: HTMLCanvasElement, warped: HTMLCanvasElement, landmarks: NormalizedLandmark[], params: ReshapeParams, live: boolean): HTMLCanvasElement {
  if (!backgroundProtectionReady()) return warped
  const W = source.width
  const H = source.height

  // The box the warp can touch: the face's outer anchor ring, padded.
  const cx = (landmarks[234].x + landmarks[454].x) / 2
  const cy = (landmarks[33].y + landmarks[263].y) / 2
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const i of OVAL_LOOP) {
    const p = landmarks[i]
    const sx = (p.x + (p.x - cx) * (SKIRT + MARGIN)) * W
    const sy = (p.y + (p.y - cy) * (SKIRT + MARGIN)) * H
    x0 = Math.min(x0, sx)
    y0 = Math.min(y0, sy)
    x1 = Math.max(x1, sx)
    y1 = Math.max(y1, sy)
  }
  x0 = Math.max(0, Math.floor(x0))
  y0 = Math.max(0, Math.floor(y0))
  x1 = Math.min(W, Math.ceil(x1))
  y1 = Math.min(H, Math.ceil(y1))
  const box = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
  if (box.w < 16 || box.h < 16) return warped

  const k = Math.min(1, WORK / Math.max(box.w, box.h))
  const w = Math.max(4, Math.round(box.w * k))
  const h = Math.max(4, Math.round(box.h * k))
  const person = personMatte(source, box, w, h, live)
  if (!person) return warped
  const n = w * h

  // The clean plate's fill: the background around the person, pushed in
  // over them (and over their soft edge, whose colours are half theirs).
  const grown = dilate(person, w, h, 2)
  const work = canvas(w, h)
  const wc = work.getContext('2d', { willReadFrequently: true })!
  wc.drawImage(source, box.x, box.y, box.w, box.h, 0, 0, w, h)
  const px = wc.getImageData(0, 0, w, h)
  const rgb = new Float32Array(n * 3)
  const wt = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    wt[i] = 1 - ramp(grown[i], 0.08, 0.35)
    rgb[i * 3] = px.data[i * 4]
    rgb[i * 3 + 1] = px.data[i * 4 + 1]
    rgb[i * 3 + 2] = px.data[i * 4 + 2]
  }
  const fill = pushPull(rgb, wt, w, h)
  // The warped person is cut out a little wider than the fill reaches, so
  // where nothing moved, the person's own edge (and a sliver of the
  // background right next to it) shows — not a ring of fill: only where
  // slimming pulled the cheek in further than that does the fill appear.
  const cutMatte = dilate(person, w, h, 4)
  const matte = new ImageData(w, h)
  for (let i = 0; i < n; i++) {
    px.data[i * 4] = fill[i * 3]
    px.data[i * 4 + 1] = fill[i * 3 + 1]
    px.data[i * 4 + 2] = fill[i * 3 + 2]
    px.data[i * 4 + 3] = (1 - wt[i]) * 255
    matte.data[i * 4] = matte.data[i * 4 + 1] = matte.data[i * 4 + 2] = 255
    matte.data[i * 4 + 3] = cutMatte[i] * 255
  }
  wc.putImageData(px, 0, 0)

  // The matte, over the whole frame (small), warped like the photo.
  const mk = MASK_W / W
  const mw = MASK_W
  const mh = Math.max(1, Math.round(H * mk))
  const small = canvas(w, h)
  small.getContext('2d')!.putImageData(matte, 0, 0)
  const mask = canvas(mw, mh)
  mask.getContext('2d')!.drawImage(small, 0, 0, w, h, box.x * mk, box.y * mk, box.w * mk, box.h * mk)
  const warpedMask = renderMeshWarp(mask, landmarks, { ...params, fillLight: 0 })

  // Warped person, cut out by the warped matte.
  const cut = canvas(box.w, box.h)
  const cc = cut.getContext('2d')!
  cc.drawImage(warped, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h)
  cc.globalCompositeOperation = 'destination-in'
  cc.imageSmoothingQuality = 'high'
  cc.drawImage(warpedMask, box.x * mk, box.y * mk, box.w * mk, box.h * mk, 0, 0, box.w, box.h)

  // Original background, the person filled out of it, the warped person
  // on top — over the box only, faded out toward its edges: the warp
  // moves nothing out there, so it meets the plain warped frame without a
  // seam (a hard-edged paste left a faint rectangle round the face).
  const prot = canvas(box.w, box.h)
  const pc = prot.getContext('2d')!
  pc.drawImage(source, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h)
  pc.imageSmoothingQuality = 'high'
  pc.drawImage(work, 0, 0, w, h, 0, 0, box.w, box.h)
  pc.drawImage(cut, 0, 0)
  pc.globalCompositeOperation = 'destination-in'
  pc.imageSmoothingEnabled = true
  pc.drawImage(featherMask(), 0, 0, box.w, box.h)
  const out = canvas(W, H)
  const ctx = out.getContext('2d')!
  ctx.drawImage(warped, 0, 0)
  ctx.drawImage(prot, box.x, box.y)
  return out
}

// Opaque in the middle, fading to nothing over the outer eighth of each
// side (a tiny canvas, stretched: its interpolation makes the ramps).
let feather: HTMLCanvasElement | null = null
function featherMask() {
  if (feather) return feather
  feather = canvas(16, 16)
  const g = feather.getContext('2d')!
  g.fillStyle = '#fff'
  g.fillRect(2, 2, 12, 12)
  return feather
}

// The switch is remembered: someone who wants it wants it every time.
const KEY = 'glow_bg_protect'
export function loadBgProtect() {
  try {
    return localStorage.getItem(KEY) === '1'
  } catch {
    return false
  }
}
export function saveBgProtect(on: boolean) {
  try {
    localStorage.setItem(KEY, on ? '1' : '0')
  } catch {
    // (private mode: just this session)
  }
}
