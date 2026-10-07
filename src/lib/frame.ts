import { eyeClosureOf, withEyeClosure } from './faceLandmarker'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

/** Output frame shape. 'full' matches the screen itself, so the live
 *  viewfinder fills it edge to edge and still shows exactly what gets saved. */
export type AspectMode = '3:4' | '1:1' | 'full'

export const ASPECT_MODES: AspectMode[] = ['3:4', '1:1', 'full']

/** width / height of the saved frame for a given mode — portrait, since
 *  this is a selfie camera locked to portrait orientation. */
export function aspectRatioFor(mode: AspectMode): number {
  if (mode === '1:1') return 1
  if (mode === '3:4') return 3 / 4
  return window.innerWidth / window.innerHeight
}

/** Normalized crop rect within the source frame: the largest centered
 *  region with the target aspect ratio, then shrunk further by `zoom`
 *  (digital zoom is just a tighter centered crop, scaled back up). */
export interface CropRect {
  x0: number
  y0: number
  fw: number
  fh: number
}

export function cropRectFor(srcW: number, srcH: number, targetAspect: number | null, zoom: number): CropRect {
  let fw = 1
  let fh = 1
  if (targetAspect) {
    const srcAspect = srcW / srcH
    if (srcAspect > targetAspect) fw = targetAspect / srcAspect
    else fh = srcAspect / targetAspect
  }
  // Zooming out (below 1) widens the crop until it reaches the frame's
  // edges — which changes the saved shape, but shows more of the scene.
  if (zoom !== 1) {
    fw = Math.min(1, fw / zoom)
    fh = Math.min(1, fh / zoom)
  }
  return { x0: (1 - fw) / 2, y0: (1 - fh) / 2, fw, fh }
}

/** Draws `source` cropped to `crop`, downscaled to at most `maxDim` on its
 *  long side, optionally mirrored. The front camera's raw stream is a
 *  mirror image (text backwards) unless corrected, and nothing upstream
 *  corrects it — so everything from it is mirrored here once. */
export function drawFrame(source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageBitmap, maxDim: number, crop: CropRect, mirror: boolean): HTMLCanvasElement {
  const w = source instanceof HTMLVideoElement ? source.videoWidth : source instanceof HTMLImageElement ? source.naturalWidth : source.width
  const h = source instanceof HTMLVideoElement ? source.videoHeight : source instanceof HTMLImageElement ? source.naturalHeight : source.height
  const cw = w * crop.fw
  const ch = h * crop.fh
  const scale = Math.min(1, maxDim / Math.max(cw, ch))
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(cw * scale))
  canvas.height = Math.max(1, Math.round(ch * scale))
  const ctx = canvas.getContext('2d')!
  // (a big downscale — a full-resolution still to the editing size —
  // aliases with the default smoothing)
  ctx.imageSmoothingQuality = 'high'
  if (mirror) {
    ctx.translate(canvas.width, 0)
    ctx.scale(-1, 1)
  }
  ctx.drawImage(source, crop.x0 * w, crop.y0 * h, cw, ch, 0, 0, canvas.width, canvas.height)
  return canvas
}

/** Live landmarks are detected on the raw (uncropped, unmirrored) video
 *  element, so they have to be mapped into the drawn frame's own
 *  coordinates before any reshape/mask pass turns them into pixels —
 *  otherwise every warp lands on the wrong feature. `z` is in MediaPipe's
 *  width-normalized units, so it rescales with the crop's width the same
 *  way `x` does; leaving it unscaled flattens the surface normals the 3D
 *  relighting is built from as the crop tightens. */
export function remapLandmarks(landmarks: NormalizedLandmark[] | null, crop: CropRect, mirror: boolean): NormalizedLandmark[] | null {
  if (!landmarks) return null
  const out = landmarks.map((p) => {
    let x = (p.x - crop.x0) / crop.fw
    if (mirror) x = 1 - x
    return { ...p, x, y: (p.y - crop.y0) / crop.fh, z: p.z / crop.fw }
  })
  // (eye closure is by anatomy, so mirroring leaves it as is)
  return withEyeClosure(out, eyeClosureOf(landmarks))
}
