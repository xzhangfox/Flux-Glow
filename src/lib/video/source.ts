import { ALL_FORMATS, BlobSource, BufferTarget, Conversion, Input, Output, WebMOutputFormat } from 'mediabunny'

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
