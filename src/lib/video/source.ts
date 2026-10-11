import { ALL_FORMATS, BlobSource, Conversion, Input, Mp4OutputFormat, Output, WebMOutputFormat, canEncodeAudio, canEncodeVideo } from 'mediabunny'
import { outputSink } from './disk'

/**
 * A copy of `file` that players can seek in. What MediaRecorder writes as
 * WebM has no index (and no duration), so a <video> can't jump around in
 * it; rewriting it — the same encoded frames and sound, just repackaged,
 * no re-encoding — adds both. Anything else is returned as is.
 */
export async function ensureSeekable(file: Blob): Promise<Blob> {
  if (!/webm|matroska/i.test(file.type)) return file
  return (await repackage(file, 'webm')) ?? file
}

/** `file`'s frames and sound, repackaged as WebM or MP4 (null if it can't
 *  be). A long one is written to storage, not memory. */
async function repackage(file: Blob, as: 'webm' | 'mp4'): Promise<Blob | null> {
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
  const sink = await outputSink(file.size * 1.05)
  try {
    const output = new Output({ format: as === 'webm' ? new WebMOutputFormat() : new Mp4OutputFormat(sink.onDisk ? { fastStart: false } : {}), target: sink.target })
    const conversion = await Conversion.init({ input, output })
    if (!conversion.isValid) {
      await sink.discard()
      return null
    }
    await conversion.execute()
    return await sink.finish(as === 'webm' ? 'video/webm' : 'video/mp4')
  } catch {
    await sink.discard()
    return null
  } finally {
    input.dispose()
  }
}

const shareable = new WeakMap<Blob, Promise<Blob>>()

/** Past this size a video is shared as it is: rewriting it would mean
 *  holding all of it in memory (and no chat app takes one this big). */
const SHAREABLE_MAX = 256 * 1024 * 1024

/** `blob` is already what toShareableMp4 would make of it (H.264 + AAC
 *  in an MP4): it's shared as it is. */
export function markShareable(blob: Blob) {
  shareable.set(blob, Promise.resolve(blob))
}

/**
 * `file` as the MP4 that chat apps (WeChat, iMessage, WhatsApp…) show as a
 * playable video rather than a file attachment: H.264 video, AAC sound,
 * and its index at the front ("fast start"), so a thumbnail and the
 * duration are there from the first bytes. What MediaRecorder writes is
 * WebM, or a fragmented MP4 without that index; both get rewritten —
 * streams that are already H.264 / AAC are copied as they are, anything
 * else is re-encoded. If this browser can't do it, `file` comes back
 * unchanged. Cached per file.
 */
export function toShareableMp4(file: Blob): Promise<Blob> {
  let p = shareable.get(file)
  if (p) return p
  if (file.size > SHAREABLE_MAX) return Promise.resolve(file)
  p = (async () => {
    const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
    try {
      const vt = await input.getPrimaryVideoTrack()
      if (!vt) return file
      if (vt.codec !== 'avc' && !(await canEncodeVideo('avc'))) return file
      const at = await input.getPrimaryAudioTrack()
      const aac = !at || at.codec === 'aac' || (await canEncodeAudio('aac'))
      const sink = await outputSink(0)
      const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: sink.target })
      const conversion = await Conversion.init({
        input,
        output,
        video: { codec: 'avc' },
        // (no AAC encoder here: keep the sound as it is rather than lose it)
        audio: aac ? { codec: 'aac' } : {},
      })
      if (!conversion.isValid) return file
      await conversion.execute()
      return await sink.finish('video/mp4')
    } catch {
      return file
    } finally {
      input.dispose()
    }
  })()
  shareable.set(file, p)
  return p
}

/**
 * `file` with a duration on every frame. MediaRecorder's WebM leaves the
 * frame at each chunk boundary without one, and a frame without a
 * duration is skipped when re-encoding — a third of a camera recording's
 * frames went missing that way. Repackaged into MP4 (the same encoded
 * frames and sound, no re-encode), every frame gets the time to the next
 * one. Anything that isn't WebM is returned as is.
 */
export function withFrameDurations(file: Blob): Promise<Blob> {
  if (!/webm|matroska/i.test(file.type)) return Promise.resolve(file)
  let p = timed.get(file)
  if (!p) timed.set(file, (p = repackage(file, 'mp4').then((b) => b ?? file)))
  return p
}
const timed = new WeakMap<Blob, Promise<Blob>>()
