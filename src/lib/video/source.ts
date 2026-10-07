import { ALL_FORMATS, BlobSource, BufferTarget, Conversion, Input, Mp4OutputFormat, Output, WebMOutputFormat, canEncodeAudio, canEncodeVideo } from 'mediabunny'

/**
 * A copy of `file` that players can seek in. What MediaRecorder writes as
 * WebM has no index (and no duration), so a <video> can't jump around in
 * it; rewriting it — the same encoded frames and sound, just repackaged,
 * no re-encoding — adds both. Anything else is returned as is.
 */
export async function ensureSeekable(file: Blob): Promise<Blob> {
  if (!/webm|matroska/i.test(file.type)) return file
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
  try {
    const output = new Output({ format: new WebMOutputFormat(), target: new BufferTarget() })
    const conversion = await Conversion.init({ input, output })
    if (!conversion.isValid) return file
    await conversion.execute()
    const buf = (output.target as BufferTarget).buffer
    return buf ? new Blob([buf], { type: 'video/webm' }) : file
  } catch {
    return file
  } finally {
    input.dispose()
  }
}

const shareable = new WeakMap<Blob, Promise<Blob>>()

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
  p = (async () => {
    const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
    try {
      const vt = await input.getPrimaryVideoTrack()
      if (!vt) return file
      if (vt.codec !== 'avc' && !(await canEncodeVideo('avc'))) return file
      const at = await input.getPrimaryAudioTrack()
      const aac = !at || at.codec === 'aac' || (await canEncodeAudio('aac'))
      const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() })
      const conversion = await Conversion.init({
        input,
        output,
        video: { codec: 'avc' },
        // (no AAC encoder here: keep the sound as it is rather than lose it)
        audio: aac ? { codec: 'aac' } : {},
      })
      if (!conversion.isValid) return file
      await conversion.execute()
      const buf = (output.target as BufferTarget).buffer
      return buf ? new Blob([buf], { type: 'video/mp4' }) : file
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
export async function withFrameDurations(file: Blob): Promise<Blob> {
  if (!/webm|matroska/i.test(file.type)) return file
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
  try {
    const output = new Output({ format: new Mp4OutputFormat(), target: new BufferTarget() })
    const conversion = await Conversion.init({ input, output })
    if (!conversion.isValid) return file
    await conversion.execute()
    const buf = (output.target as BufferTarget).buffer
    return buf ? new Blob([buf], { type: 'video/mp4' }) : file
  } catch {
    return file
  } finally {
    input.dispose()
  }
}
