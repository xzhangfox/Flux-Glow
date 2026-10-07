import { ALL_FORMATS, BlobSource, BufferTarget, Conversion, Input, Mp4OutputFormat, Output, QUALITY_HIGH, canEncodeAudio, getFirstEncodableVideoCodec } from 'mediabunny'
import { cropRectFor } from '../frame'
import type { EditParams } from '../pipeline'
import { analyzeVideo } from './analyze'
import { exportVideo } from './export'
import { withFrameDurations } from './source'

// A recording at full quality. While recording, the camera's own stream is
// recorded as it comes (the phone's hardware encoder: full resolution, up
// to 60 fps — nothing to do with how fast the live preview manages to
// retouch); afterwards, offline and frame by frame, it's framed the way the
// preview was (the aspect crop, digital zoom, the selfie mirror), every face
// in it found and tracked, and the look applied — the video editor's own
// pipeline — then encoded as an MP4.

/** How the preview framed the camera: the shape it showed (width /
 *  height, null for the camera's own), its digital zoom, and the mirror.
 *  The crop is worked out from the recording's own upright size. */
export interface RecordView {
  aspect: number | null
  zoom: number
  mirror: boolean
  /** The camera picture's own proportions as the preview showed it
   *  (width / height, upright). */
  source: number
}

/** The long side of the output, at most (1080p in portrait: 1080 × 1920). */
const OUT_MAX = 1920

export interface RenderStage {
  stage: 'framing' | 'tracking' | 'rendering'
  /** 0..1 over the whole job. */
  progress: number
}

/** The raw camera recording framed like the preview: the untouched copy
 *  the editor starts from, and what the look is rendered onto. */
export async function frameRecording(raw: Blob, view: RecordView, onProgress: (p: number) => void, signal?: AbortSignal): Promise<Blob> {
  const input = new Input({ source: new BlobSource(await withFrameDurations(raw)), formats: ALL_FORMATS })
  try {
    const vt = await input.getPrimaryVideoTrack()
    if (!vt) throw new Error('This recording has no video.')
    // Some phones' recorders store the upright picture in a frame of a
    // different shape than the camera's (portrait squeezed into a
    // landscape frame): each frame is stretched back to the proportions
    // the preview showed before anything else, or it comes out squeezed.
    let dw = vt.displayWidth
    let dh = vt.displayHeight
    if (view.source > 0 && Math.abs(dw / dh - view.source) / view.source > 0.02) {
      if (view.source < 1) dh = Math.round(dw / view.source)
      else dw = Math.round(dh * view.source)
      const s = Math.min(1, 2560 / Math.max(dw, dh))
      dw = Math.round(dw * s)
      dh = Math.round(dh * s)
    }
    const crop = cropRectFor(dw, dh, view.aspect, view.zoom)
    const cw = Math.round(dw * crop.fw)
    const ch = Math.round(dh * crop.fh)
    const k = Math.min(1, OUT_MAX / Math.max(cw, ch))
    // (even sizes: H.264 wants them)
    const W = Math.max(2, Math.round((cw * k) / 2) * 2)
    const H = Math.max(2, Math.round((ch * k) / 2) * 2)
    const codec = await getFirstEncodableVideoCodec(['avc', 'hevc', 'vp9', 'av1'], { width: W, height: H })
    if (!codec) throw new Error("This browser can't encode video.")
    const at = await input.getPrimaryAudioTrack()
    const aac = !!at && at.codec !== 'aac' && (await canEncodeAudio('aac'))
    const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() })
    // Framed by hand: each frame drawn upright first (phones often store a
    // recording as landscape frames plus a "rotate 90°" flag, which
    // drawing the sample applies), then cropped and mirrored with plain
    // canvas operations. Left to the converter, the crop was taken in the
    // stored orientation on such files, and the result came out squeezed.
    const upright = document.createElement('canvas')
    upright.width = dw
    upright.height = dh
    const uc = upright.getContext('2d')!
    const out = document.createElement('canvas')
    out.width = W
    out.height = H
    const oc = out.getContext('2d')!
    oc.imageSmoothingQuality = 'high'
    const sx = dw * crop.x0
    const sy = dh * crop.y0
    const conversion = await Conversion.init({
      input,
      output,
      video: {
        codec,
        quality: QUALITY_HIGH,
        allowTransformationMetadata: false,
        forceTranscode: true,
        processedWidth: W,
        processedHeight: H,
        process: (sample) => {
          uc.clearRect(0, 0, dw, dh)
          sample.draw(uc, 0, 0, dw, dh)
          oc.setTransform(1, 0, 0, 1, 0, 0)
          if (view.mirror) oc.setTransform(-1, 0, 0, 1, W, 0)
          oc.drawImage(upright, sx, sy, cw, ch, 0, 0, W, H)
          return out
        },
      },
      ...(aac ? { audio: { codec: 'aac' as const } } : {}),
    })
    if (!conversion.isValid) throw new Error("This recording can't be converted in this browser.")
    conversion.onProgress = onProgress
    const abort = () => void conversion.cancel()
    signal?.addEventListener('abort', abort)
    try {
      await conversion.execute()
    } finally {
      signal?.removeEventListener('abort', abort)
    }
    const buf = (output.target as BufferTarget).buffer
    if (!buf) throw new Error('Framing produced no data.')
    return new Blob([buf], { type: 'video/mp4' })
  } finally {
    input.dispose()
  }
}

/** Renders a raw camera recording with `params` (every face gets them).
 *  Returns the finished video and the framed, untouched copy. */
export async function renderRecording(
  raw: Blob,
  view: RecordView,
  params: EditParams,
  onProgress: (p: RenderStage) => void,
  signal?: AbortSignal,
): Promise<{ video: Blob; clean: Blob }> {
  // Rough shares of the time each step takes.
  const clean = await frameRecording(raw, view, (p) => onProgress({ stage: 'framing', progress: p * 0.15 }), signal)
  const analysis = await analyzeVideo(clean, (p) => onProgress({ stage: 'tracking', progress: 0.15 + 0.45 * (p.progress ?? 0) }), signal)
  const { blob } = await exportVideo(clean, analysis, () => params, params, { quality: 'fast' }, (p) => onProgress({ stage: 'rendering', progress: 0.6 + 0.4 * p }), signal)
  return { video: blob, clean }
}
