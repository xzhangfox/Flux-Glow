import { ALL_FORMATS, BlobSource, Conversion, Input, Mp4OutputFormat, Output, Quality, VideoSample, canEncodeAudio, getFirstEncodableVideoCodec } from 'mediabunny'
import { processFaces, type EditParams, type FaceEdit } from '../pipeline'
import type { VideoAnalysis } from './analyze'
import { outputSink } from './disk'
import { unpack } from './faces'
import { FaceScout } from './follow'
import { markShareable, withFrameDurations } from './source'

// Renders the edited video: every frame decoded, the people in it found
// again (FaceScout: exact for that frame, steadied with the frames either
// side), each retouched with their own settings, the frame-wide filter on
// top, then encoded (WebCodecs) into an MP4 with the original sound.
// Frame-accurate and independent of playback speed — a slow phone just
// takes longer; it never drops a frame. A long clip's output is streamed
// to storage as it's made (video/disk.ts), so even hours of video never
// sit in memory.
//
// The output can be smaller than the clip, and at a lower frame rate —
// never bigger or faster: there is nothing to add that isn't there.

export type BitrateTier = 'lower' | 'recommended' | 'higher'

export interface ExportOptions {
  /** The output's short side, px; null: the clip's own size. */
  size: number | null
  /** Frames a second; null: the clip's own — every frame, timed as it is. */
  fps: number | null
  bitrate: BitrateTier
  /** Full-quality retouching (edge-aware, as for photos) or the faster
   *  preview-grade pass. */
  quality: 'high' | 'fast'
}

export interface ExportResult {
  blob: Blob
  /** Audio couldn't be carried over (an unsupported codec): the video is
   *  silent. */
  droppedAudio: boolean
  /** Frames meant to go out, and what the finished file holds. */
  frames: { expected: number; written: number | null }
  width: number
  height: number
}

export interface SourceInfo {
  /** Upright size, px. */
  width: number
  height: number
  /** Frames a second (average). */
  fps: number
  duration: number
  /** Video bitrate, bits a second (estimated). */
  bitrate: number | null
}

export const canExportVideo = () => typeof VideoEncoder !== 'undefined' && typeof VideoDecoder !== 'undefined'

/** What the export sheet needs to know about a clip. */
export async function probeSource(file: Blob): Promise<SourceInfo> {
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
  try {
    const vt = await input.getPrimaryVideoTrack()
    if (!vt) throw new Error('This file has no video.')
    const duration = await input.computeDuration()
    const stats = await vt.computePacketStats(240).catch(() => null)
    const at = await input.getPrimaryAudioTrack()
    // (the file's size, less a typical soundtrack, over its length)
    const bitrate = duration > 0 ? Math.max(0, (file.size * 8) / duration - (at ? 160_000 : 0)) : null
    return { width: vt.displayWidth, height: vt.displayHeight, fps: stats?.averagePacketRate || 30, duration, bitrate: bitrate || null }
  } finally {
    input.dispose()
  }
}

// ---- Choices ----------------------------------------------------------------------

const SIZES = [2160, 1440, 1080, 720, 480]
const sizeName = (s: number) => (s >= 2100 ? '4K' : s >= 1400 ? '2K' : `${s}p`)

/** Output sizes: the clip's own, and the standard ones below it. */
export function sizeChoices(src: SourceInfo): { value: number | null; label: string; detail: string }[] {
  const s0 = Math.min(src.width, src.height)
  const out: { value: number | null; label: string; detail: string }[] = [{ value: null, label: sizeName(s0), detail: `${src.width}×${src.height}` }]
  for (const s of SIZES) {
    if (s >= s0 - 8) continue
    const { width, height } = outSize(src, s)
    out.push({ value: s, label: sizeName(s), detail: `${width}×${height}` })
  }
  return out
}

const fpsName = (f: number) => (Math.abs(f - Math.round(f)) < 0.02 ? String(Math.round(f)) : f.toFixed(2))

/** Frame rates: the clip's own, and the usual ones below it (halves first:
 *  every other frame, so motion stays even). */
export function fpsChoices(src: SourceInfo): { value: number | null; label: string }[] {
  const f0 = src.fps
  const vals: number[] = []
  for (let f = f0 / 2; f > 45 / 2 && vals.length < 2; f /= 2) if (f >= 23.5) vals.push(f)
  if (f0 > 31 && !vals.some((v) => Math.abs(v - 30) < 1)) vals.push(30)
  if (f0 > 26 && !vals.some((v) => Math.abs(v - 24) < 0.6)) vals.push(24)
  vals.sort((a, b) => b - a)
  // (an NTSC clip's half is 29.97, kept exactly: a frame every other frame)
  return [{ value: null, label: fpsName(f0) }, ...vals.slice(0, 3).map((v) => ({ value: v, label: fpsName(v) }))]
}

function outSize(src: SourceInfo, size: number | null) {
  const k = size ? Math.min(1, size / Math.min(src.width, src.height)) : 1
  // Even sizes: H.264 wants them.
  return { width: Math.max(2, Math.round((src.width * k) / 2) * 2), height: Math.max(2, Math.round((src.height * k) / 2) * 2) }
}

const TIER: Record<BitrateTier, number> = { lower: 1.1, recommended: 2, higher: 3.4 }
const AUDIO_BPS = 160_000

/** Bits a second of H.264 for this size, rate and tier. The top tier is
 *  never far above what the clip itself was made with (re-encoding can't
 *  add detail that wasn't there; it only makes the file bigger), and the
 *  others keep their proportion to it. */
function videoBitrate(src: SourceInfo, width: number, height: number, fps: number, tier: BitrateTier) {
  const px = Math.pow((width * height) / (1920 * 1080), 0.95)
  const rate = Math.pow(Math.max(12, fps) / 30, 0.6)
  const base = 3e6 * px * rate
  let top = base * TIER.higher
  if (src.bitrate) {
    // The clip's own bitrate at this size and rate. (Phones record H.265
    // as often as H.264; give the original the benefit of the doubt.)
    const own = src.bitrate * Math.pow((width * height) / (src.width * src.height), 0.95) * Math.pow(fps / src.fps, 0.6) * 1.4
    top = Math.min(top, Math.max(own * 1.2, base * TIER.lower))
  }
  return Math.round((top * TIER[tier]) / TIER.higher)
}

/** What these options make: size, rate, bitrate and about how big a file. */
export function planExport(src: SourceInfo, opts: Pick<ExportOptions, 'size' | 'fps' | 'bitrate'>) {
  const { width, height } = outSize(src, opts.size)
  const fps = opts.fps ?? src.fps
  const bitrate = videoBitrate(src, width, height, fps, opts.bitrate)
  return { width, height, fps, bitrate, bytes: Math.round(((bitrate + AUDIO_BPS) * src.duration) / 8) * 1.02 }
}

// ---- The export -------------------------------------------------------------------

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
  // (a recording's WebM has frames without durations, which re-encoding
  // would drop)
  const src = await withFrameDurations(file)
  const info = await probeSource(src)
  const plan = planExport(info, opts)
  const { width: W, height: H } = plan
  const F = opts.fps && opts.fps < info.fps - 0.01 ? opts.fps : null
  const codec = await getFirstEncodableVideoCodec(['avc', 'hevc', 'vp9', 'av1'], { width: W, height: H, bitrate: plan.bitrate })
  if (!codec) throw new Error(`This browser can't encode video at ${W}×${H}. Try a smaller size.`)

  const input = new Input({ source: new BlobSource(src), formats: ALL_FORMATS })
  const sink = await outputSink(plan.bytes)
  let scout: FaceScout | null = null
  try {
    const output = new Output({ format: new Mp4OutputFormat({ fastStart: sink.onDisk ? false : 'in-memory' }), target: sink.target })
    // Faces are found at up to 1280 px — sharper than the analysis's
    // working size, so meshes sit right on a 4K frame too.
    scout = await FaceScout.open(src, analysis, Math.min(1280, Math.max(W, H)))
    const canvas = document.createElement('canvas')
    canvas.width = W
    canvas.height = H
    const ctx = canvas.getContext('2d')!
    ctx.imageSmoothingQuality = 'high'
    const hq = opts.quality === 'high'
    // The converter starts its output at the first frame (or sound): source
    // times are this much later.
    const startTs = Math.max(0, await input.getFirstTimestamp())
    let expected = 0

    // AAC sound where this browser can encode it (a recording's Opus is
    // carried over otherwise): chat apps only play MP4s with AAC inline.
    const at = await input.getPrimaryAudioTrack()
    const aac = !!at && at.codec !== 'aac' && (await canEncodeAudio('aac'))
    const conversion = await Conversion.init({
      input,
      output,
      ...(aac ? { audio: { codec: 'aac' as const } } : {}),
      video: {
        // Phone videos are often stored sideways with a "rotate" flag; bake
        // the rotation into the frames so each one comes to `process`
        // upright, the way the analysis saw it.
        allowTransformationMetadata: false,
        codec,
        quality: new Quality({ bitrate: plan.bitrate }),
        forceTranscode: true,
        processedWidth: W,
        processedHeight: H,
        process: async (sample) => {
          const t = sample.timestamp
          const d = sample.duration > 0 ? sample.duration : 1 / info.fps
          // A lower frame rate: the frame on screen at each tick of the new
          // rate (none, here, if this frame falls between two ticks).
          let ticks: number[] | null = null
          if (F) {
            ticks = []
            for (let k = Math.ceil(t * F - 1e-6); k / F < t + d - 1e-6; k++) ticks.push(k)
            if (!ticks.length) return null
          }
          sample.draw(ctx, 0, 0, W, H)
          const faces: FaceEdit[] = []
          for (const f of await scout!.at(t + startTs)) {
            const params = paramsFor(f.id)
            if (params) faces.push({ landmarks: unpack(f.lm), params, slot: `p${f.id}` })
          }
          const out = processFaces(canvas, faces, frame, hq, t)
          if (!ticks) {
            expected++
            return out
          }
          expected += ticks.length
          return ticks.map((k) => new VideoSample(out, { timestamp: k / F!, duration: 1 / F! }))
        },
      },
    })
    if (!conversion.isValid) throw new Error("This video can't be converted in this browser.")
    const droppedAudio = conversion.discardedTracks.some((d) => d.track.type === 'audio')
    conversion.onProgress = (p) => onProgress(p)
    const abort = () => void conversion.cancel()
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError')
    signal?.addEventListener('abort', abort)
    try {
      await conversion.execute()
    } finally {
      signal?.removeEventListener('abort', abort)
    }
    const blob = await sink.finish('video/mp4')
    // H.264 + AAC already: no need to rewrite it before it's shared.
    if (codec === 'avc' && (!at || at.codec === 'aac' || aac)) markShareable(blob)
    return { blob, droppedAudio, frames: { expected, written: await countFrames(blob) }, width: W, height: H }
  } catch (e) {
    await sink.discard()
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError')
    throw e
  } finally {
    scout?.dispose()
    input.dispose()
  }
}

/** Video frames in a finished file (from its index), or null if it can't
 *  be read back. */
export async function countFrames(blob: Blob): Promise<number | null> {
  const input = new Input({ source: new BlobSource(blob), formats: ALL_FORMATS })
  try {
    const vt = await input.getPrimaryVideoTrack()
    return vt ? (await vt.computePacketStats()).packetCount : null
  } catch {
    return null
  } finally {
    input.dispose()
  }
}
