import { BufferTarget, StreamTarget, type StreamTargetChunk, type Target } from 'mediabunny'

// Where a written video goes. A short one is built in memory, as before; a
// long one (an hour-long export runs to gigabytes) is streamed to a file in
// the site's private storage as it's written, and what comes back is that
// file — read from disk when it's played, saved or shared, never loaded
// whole into memory.

const DIR = 'flux-glow-tmp'
/** Files from this page; anything else in DIR is left over from before. */
const SESSION = `s${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`
/** Outputs expected to be bigger than this go to disk. */
let inMemoryMax = 160 * 1024 * 1024
/** Dev tests only: every output to disk, however small. */
export const alwaysUseDisk = () => void (inMemoryMax = 0)

let seq = 0
const names = new WeakMap<Blob, string>()

export interface OutputSink {
  target: Target
  onDisk: boolean
  /** After the output is finalized: the file it wrote. */
  finish(type: string): Promise<Blob>
  /** After a failure or a cancel: drops whatever was written. */
  discard(): Promise<void>
}

export const canUseDisk = () => typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory && typeof Worker !== 'undefined'

async function tempDir() {
  const root = await navigator.storage.getDirectory()
  return root.getDirectoryHandle(DIR, { create: true })
}

/** Bytes this site can still store (null: the browser won't say). */
export async function freeSpace(): Promise<number | null> {
  try {
    const e = await navigator.storage?.estimate?.()
    return e?.quota !== undefined && e.usage !== undefined ? Math.max(0, e.quota - e.usage) : null
  } catch {
    return null
  }
}

/** Where to write an output of about `bytes`. */
export async function outputSink(bytes: number): Promise<OutputSink> {
  if (bytes > inMemoryMax && canUseDisk()) {
    try {
      return await diskSink()
    } catch (e) {
      console.warn('No private storage; writing in memory', e)
    }
  }
  const target = new BufferTarget()
  return {
    target,
    onDisk: false,
    finish: async (type) => {
      if (!target.buffer) throw new Error('Nothing was written.')
      return new Blob([target.buffer], { type })
    },
    discard: async () => {},
  }
}

async function diskSink(): Promise<OutputSink> {
  const name = `${SESSION}-${++seq}`
  const worker = new Worker(new URL('./diskWorker.ts', import.meta.url), { type: 'module' })
  let next = 0
  const waiting = new Map<number, { resolve: () => void; reject: (e: Error) => void }>()
  let failed: Error | null = null
  worker.onmessage = (e: MessageEvent<{ id: number; ok: boolean; error?: string; name?: string }>) => {
    const w = waiting.get(e.data.id)
    if (!w) return
    waiting.delete(e.data.id)
    if (e.data.ok) w.resolve()
    else {
      const full = e.data.name === 'QuotaExceededError' || /full|quota/i.test(e.data.error ?? '')
      w.reject(new Error(full ? 'Not enough free storage on this device for this video.' : `Couldn't write the video: ${e.data.error}`))
    }
  }
  worker.onerror = (e) => {
    failed = new Error(`Couldn't write the video: ${e.message}`)
    for (const w of waiting.values()) w.reject(failed)
    waiting.clear()
  }
  const call = (msg: Record<string, unknown>, transfer: Transferable[] = []) =>
    new Promise<void>((resolve, reject) => {
      if (failed) return reject(failed)
      const id = ++next
      waiting.set(id, { resolve, reject })
      worker.postMessage({ ...msg, id }, transfer)
    })
  await call({ type: 'open', name })
  const writable = new WritableStream<StreamTargetChunk>({
    // (a copy of its own: the muxer may reuse the chunk's memory)
    write: (chunk) => {
      const data = chunk.data.slice()
      return call({ type: 'write', data, position: chunk.position }, [data.buffer])
    },
    close: () => call({ type: 'close' }),
    abort: () => call({ type: 'abort' }),
  })
  const target = new StreamTarget(writable, { chunked: true, chunkSize: 8 * 1024 * 1024 })
  const remove = async () => {
    try {
      await (await tempDir()).removeEntry(name)
    } catch {
      // (already gone)
    }
  }
  return {
    target,
    onDisk: true,
    finish: async (type) => {
      worker.terminate()
      const file = await (await (await tempDir()).getFileHandle(name)).getFile()
      // (the same file on disk, typed as a video)
      const blob = file.slice(0, file.size, type)
      names.set(blob, name)
      return blob
    },
    discard: async () => {
      await call({ type: 'abort' }).catch(() => {})
      worker.terminate()
      await remove()
    },
  }
}

/** Deletes a file `outputSink` wrote (once nothing needs it any more). */
export async function removeTemp(blob: Blob | null | undefined) {
  const name = blob && names.get(blob)
  if (!name) return
  names.delete(blob)
  try {
    await (await tempDir()).removeEntry(name)
  } catch {
    // (already gone)
  }
}

/** Deletes files left over from earlier visits (a closed tab, a crash). */
export async function sweepTemp() {
  if (!canUseDisk()) return
  try {
    const dir = await tempDir()
    const stale: string[] = []
    for await (const name of (dir as unknown as { keys(): AsyncIterable<string> }).keys()) if (!name.startsWith(SESSION)) stale.push(name)
    for (const name of stale) await dir.removeEntry(name).catch(() => {})
  } catch {
    // (no private storage here)
  }
}
