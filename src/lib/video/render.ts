import { ALL_FORMATS, BlobSource, BufferTarget, Conversion, Input, Mp4OutputFormat, Output, QUALITY_HIGH, canEncodeAudio, getFirstEncodableVideoCodec } from 'mediabunny'
import type { CropRect } from '../frame'
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

/** How the preview framed the camera: its crop (normalized) and mirror. */
export interface RecordView {
  crop: CropRect
  mirror: boolean
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
    const dw = vt.displayWidth
    const dh = vt.displayHeight
    const cw = Math.round(dw * view.crop.fw)
    const ch = Math.round(dh * view.crop.fh)
    const k = Math.min(1, OUT_MAX / Math.max(cw, ch))
    // (even sizes: H.264 wants them)
    const W = Math.max(2, Math.round((cw * k) / 2) * 2)
    const H = Math.max(2, Math.round((ch * k) / 2) * 2)
    const codec = await getFirstEncodableVideoCodec(['avc', 'hevc', 'vp9', 'av1'], { width: W, height: H })
    if (!codec) throw new Error("This browser can't encode video.")
    const at = await input.getPrimaryAudioTrack()
    const aac = !!at && at.codec !== 'aac' && (await canEncodeAudio('aac'))
    const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() })
    const conversion = await Conversion.init({
      input,
      output,
      video: {
        // The crop is centred, so it's the same before or after the mirror.
        flip: view.mirror,
        crop: { left: Math.round(dw * view.crop.x0), top: Math.round(dh * view.crop.y0), width: cw, height: ch },
        width: W,
        height: H,
        fit: 'fill',
        codec,
        quality: QUALITY_HIGH,
        allowTransformationMetadata: false,
        forceTranscode: true,
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
