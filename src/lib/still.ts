import type { CropRect } from './frame'

// The sharpest still the camera can give at the shutter, rather than a
// frame of the live stream (at most 1920 px — a quarter of what the phone's
// own camera app saves).
//
// Where the browser has ImageCapture (Chrome on Android), the camera takes
// a real photo at its full photo resolution. Elsewhere (Safari, Firefox) the
// stream is switched to the camera's largest size for a moment and a frame
// of that is taken. Either way it falls back to the live frame when the
// camera gives nothing better (or nothing at all, in time).

/** The long side a still is kept at, at most: about what phone cameras
 *  save by default (12 MP) — and within every phone's GPU texture limit,
 *  which the retouching passes need. */
export const STILL_MAX = 4096

export interface Still {
  image: HTMLVideoElement | ImageBitmap
  /** The part of `image` that matches the live view. */
  sx: number
  sy: number
  width: number
  height: number
  /** A later moment than the live frame on screen (the face may have
   *  moved, so look for it again). */
  fresh: boolean
  release(): void
}

const timeout = <T,>(p: Promise<T>, ms: number) => Promise.race([p, new Promise<never>((_, reject) => setTimeout(() => reject(new Error('timeout')), ms))])

const nextFrame = (video: HTMLVideoElement) =>
  new Promise<void>((resolve) => {
    const v = video as HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number }
    if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(() => resolve())
    else setTimeout(resolve, 40)
  })

function liveFrame(video: HTMLVideoElement, fresh: boolean): Still {
  return { image: video, sx: 0, sy: 0, width: video.videoWidth, height: video.videoHeight, fresh, release: () => {} }
}

interface PhotoCaps {
  imageWidth?: { max: number }
  imageHeight?: { max: number }
}
interface ImageCaptureLike {
  getPhotoCapabilities(): Promise<PhotoCaps>
  takePhoto(settings?: { imageWidth?: number; imageHeight?: number }): Promise<Blob>
}

/** A real photo (ImageCapture), cut to the live view's proportions; null
 *  when there's none to be had, or it doesn't match the live view. */
async function takePhoto(video: HTMLVideoElement, track: MediaStreamTrack): Promise<Still | null> {
  const IC = (window as unknown as { ImageCapture?: new (t: MediaStreamTrack) => ImageCaptureLike }).ImageCapture
  if (!IC) return null
  const ic = new IC(track)
  const caps = await timeout(ic.getPhotoCapabilities(), 1500)
  const mw = caps.imageWidth?.max ?? 0
  const mh = caps.imageHeight?.max ?? 0
  // Not past STILL_MAX: a 50 MP sensor's full photo is slow to take and
  // decode, and more than the editor keeps.
  const k = Math.min(1, STILL_MAX / Math.max(mw, mh, 1))
  const blob = await timeout(ic.takePhoto(mw && mh ? { imageWidth: Math.round(mw * k), imageHeight: Math.round(mh * k) } : undefined), 4000)
  const bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' } as ImageBitmapOptions)
  const vw = video.videoWidth
  const vh = video.videoHeight
  // Turned differently from the stream (an orientation the photo doesn't
  // record), or no sharper than it: the live frame is the better pick.
  if (bmp.width > bmp.height !== vw > vh || bmp.width * bmp.height < vw * vh * 1.2) {
    bmp.close()
    return null
  }
  // The photo can be a wider shape than the stream (a 4:3 photo behind a
  // 16:9 stream, which is the middle of it): keep the same middle.
  const va = vw / vh
  const pa = bmp.width / bmp.height
  const width = pa > va ? Math.round(bmp.height * va) : bmp.width
  const height = pa > va ? bmp.height : Math.round(bmp.width / va)
  return { image: bmp, sx: Math.round((bmp.width - width) / 2), sy: Math.round((bmp.height - height) / 2), width, height, fresh: true, release: () => bmp.close() }
}

/** The stream at the camera's largest size, for a frame or two — in the
 *  same proportions, since a camera's other shapes are cut from its
 *  sensor differently (a 16:9 mode drops the sides of a 4:3 one), and the
 *  photo must show what the viewfinder did. */
async function enlargeStream(video: HTMLVideoElement, track: MediaStreamTrack, zoom: number | null): Promise<Still | null> {
  const caps = track.getCapabilities?.() as (MediaTrackCapabilities & { width?: { max?: number } }) | undefined
  const maxW = caps?.width?.max ?? 0
  const st = track.getSettings()
  if (!st.width || !st.height || maxW < st.width * 1.2) return null
  const ar = st.width / st.height
  const shape = video.videoWidth / video.videoHeight
  const area = video.videoWidth * video.videoHeight
  const keep = zoom !== null ? { advanced: [{ zoom } as unknown as MediaTrackConstraintSet] } : {}
  await timeout(
    track.applyConstraints({
      width: { ideal: Math.min(maxW, STILL_MAX) },
      aspectRatio: { min: ar * 0.99, max: ar * 1.01 },
      frameRate: { ideal: 30 },
      ...keep,
    }),
    2000,
  )
  const t0 = performance.now()
  while (video.videoWidth * video.videoHeight < area * 1.2) {
    if (performance.now() - t0 > 1500) return null
    await nextFrame(video)
  }
  if (Math.abs(video.videoWidth / video.videoHeight - shape) > shape * 0.02) {
    // Another shape after all: back to the one on screen.
    await timeout(track.applyConstraints({ width: { ideal: st.width }, aspectRatio: { ideal: ar }, frameRate: { ideal: st.frameRate ?? 30 }, ...keep }), 2000)
    while (Math.abs(video.videoWidth / video.videoHeight - shape) > shape * 0.02) {
      if (performance.now() - t0 > 3000) throw new Error('camera stuck in another shape')
      await nextFrame(video)
    }
    return null
  }
  // (the first frames after the switch can still be settling)
  await nextFrame(video)
  await nextFrame(video)
  return liveFrame(video, true)
}

/** The sharpest still of the moment. `zoom`: the hardware zoom in use (to
 *  keep), null for none. The caller stops the camera afterwards — the
 *  stream may be left at a size the live view never asked for. */
export async function grabStill(video: HTMLVideoElement, track: MediaStreamTrack | null, zoom: number | null): Promise<Still> {
  if (!track || track.readyState !== 'live') return liveFrame(video, false)
  const t0 = performance.now()
  try {
    const photo = await takePhoto(video, track)
    if (photo) return photo
  } catch {
    // no photo: try the next way
  }
  try {
    const big = await enlargeStream(video, track, zoom)
    if (big) return big
  } catch {
    // the camera wouldn't switch
  }
  return liveFrame(video, performance.now() - t0 > 80)
}

/** `still` framed like the live view (`crop` of it, mirrored for the
 *  selfie camera), at up to `maxDim` on its long side. */
export function drawStill(still: Still, crop: CropRect, mirror: boolean, maxDim = STILL_MAX): HTMLCanvasElement {
  const cw = still.width * crop.fw
  const ch = still.height * crop.fh
  const k = Math.min(1, maxDim / Math.max(cw, ch))
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(cw * k))
  canvas.height = Math.max(1, Math.round(ch * k))
  const ctx = canvas.getContext('2d')!
  ctx.imageSmoothingQuality = 'high'
  if (mirror) {
    ctx.translate(canvas.width, 0)
    ctx.scale(-1, 1)
  }
  ctx.drawImage(still.image, still.sx + crop.x0 * still.width, still.sy + crop.y0 * still.height, cw, ch, 0, 0, canvas.width, canvas.height)
  return canvas
}
