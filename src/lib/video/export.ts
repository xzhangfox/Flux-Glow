import { ALL_FORMATS, BlobSource, BufferTarget, Conversion, Input, Mp4OutputFormat, Output, QUALITY_HIGH, canEncodeAudio, getFirstEncodableVideoCodec } from 'mediabunny'
import { processFaces, type EditParams, type FaceEdit } from '../pipeline'
import type { VideoAnalysis } from './analyze'

// Renders the edited video: every frame decoded, each person retouched with
// their own settings at their smoothed landmarks for that frame, the
// frame-wide filter on top, then encoded (WebCodecs) into an MP4 with the
// original sound copied across untouched. Frame-accurate and independent of
// playback speed — a slow phone just takes longer; it never drops frames.

/** The long side of the exported video, at most (1080p). */
const OUT_MAX = 1920

export interface ExportOptions {
  /** Full-quality retouching (edge-aware, as for photos) or the faster
   *  preview-grade pass. */
  quality: 'high' | 'fast'
}

export interface ExportResult {
  blob: Blob
  /** Audio couldn't be carried over (an unsupported codec): the video is
   *  silent. */
  droppedAudio: boolean
}

export const canExportVideo = () => typeof VideoEncoder !== 'undefined' && typeof VideoDecoder !== 'undefined'

/** `paramsFor(id)`: the edit for person `id` (null: leave them as they are). */
export async function exportVideo(
  file: Blob,
  analysis: VideoAnalysis,
  paramsFor: (id: number) => EditParams | null,
  frame: Pick<EditParams, 'filterId' | 'filterStrength'>,
  opts: ExportOptions,
  onProgress: (p: number) => void,
  signal?: AbortSignal,
): Promise<ExportResult> {
  if (!canExportVideo()) throw new Error('Exporting video needs a newer browser (WebCodecs).')
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
  try {
    const vt = await input.getPrimaryVideoTrack()
    if (!vt) throw new Error('This file has no video.')
    const k = Math.min(1, OUT_MAX / Math.max(vt.displayWidth, vt.displayHeight))
    // Even dimensions: H.264 wants them.
    const W = Math.max(2, Math.round((vt.displayWidth * k) / 2) * 2)
    const H = Math.max(2, Math.round((vt.displayHeight * k) / 2) * 2)
    const codec = await getFirstEncodableVideoCodec(['avc', 'hevc', 'vp9', 'av1'], { width: W, height: H })
    if (!codec) throw new Error("This browser can't encode video.")

    const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() })
    const canvas = document.createElement('canvas')
    canvas.width = W
    canvas.height = H
    const ctx = canvas.getContext('2d')!
    const hq = opts.quality === 'high'

    // AAC sound where this browser can encode it (a recording's Opus is
    // carried over otherwise): chat apps only play MP4s with AAC inline.
    const at = await input.getPrimaryAudioTrack()
    const aac = !!at && at.codec !== 'aac' && (await canEncodeAudio('aac'))
    const conversion = await Conversion.init({
      input,
      output,
      ...(aac ? { audio: { codec: 'aac' as const } } : {}),
      video: {
        width: W,
        height: H,
        fit: 'fill',
        // Phone videos are often stored sideways with a "rotate" flag; bake
        // the rotation into the frames so each one comes to `process`
        // upright, the way the analysis saw it.
        allowTransformationMetadata: false,
        codec,
        quality: QUALITY_HIGH,
        forceTranscode: true,
        processedWidth: W,
        processedHeight: H,
        process: (sample) => {
          sample.draw(ctx, 0, 0, W, H)
          const t = sample.timestamp
          const faces: FaceEdit[] = []
          for (const f of analysis.facesAt(t)) {
            const params = paramsFor(f.id)
            if (params) faces.push({ landmarks: f.landmarks, params, slot: `p${f.id}` })
          }
          return processFaces(canvas, faces, frame, hq, t)
        },
      },
    })
    if (!conversion.isValid) throw new Error("This video can't be converted in this browser.")
    const droppedAudio = conversion.discardedTracks.some((d) => d.track.type === 'audio')
    conversion.onProgress = (p) => onProgress(p)
    const abort = () => void conversion.cancel()
    signal?.addEventListener('abort', abort)
    try {
      await conversion.execute()
    } finally {
      signal?.removeEventListener('abort', abort)
    }
    const buf = (output.target as BufferTarget).buffer
    if (!buf) throw new Error('Export produced no data.')
    return { blob: new Blob([buf], { type: 'video/mp4' }), droppedAudio }
  } finally {
    input.dispose()
  }
}
