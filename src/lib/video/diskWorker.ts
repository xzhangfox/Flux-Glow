/// <reference lib="webworker" />

// Writes one file in the origin's private storage, for lib/video/disk.ts:
// the sync access handle (the one way to write a file there that works in
// every browser that has it) only exists in a worker. (Safari before 17
// made its methods return promises: each call is awaited, which is a no-op
// on the synchronous ones.)

const DIR = 'flux-glow-tmp'

let handle: FileSystemSyncAccessHandle | null = null

type Msg = { id: number } & ({ type: 'open'; name: string } | { type: 'write'; data: Uint8Array; position: number } | { type: 'close' } | { type: 'abort' })

self.onmessage = async (e: MessageEvent<Msg>) => {
  const m = e.data
  try {
    if (m.type === 'open') {
      const root = await navigator.storage.getDirectory()
      const dir = await root.getDirectoryHandle(DIR, { create: true })
      const fh = await dir.getFileHandle(m.name, { create: true })
      handle = await fh.createSyncAccessHandle()
      await handle.truncate(0)
    } else if (m.type === 'write') {
      if (!handle) throw new Error('Not open')
      let done = 0
      while (done < m.data.byteLength) {
        const n = await handle.write(m.data.subarray(done), { at: m.position + done })
        if (n <= 0) throw new Error('Storage is full')
        done += n
      }
    } else {
      if (handle) {
        if (m.type === 'close') await handle.flush()
        await handle.close()
        handle = null
      }
    }
    self.postMessage({ id: m.id, ok: true })
  } catch (err) {
    self.postMessage({ id: m.id, ok: false, error: (err as Error).message || String(err), name: (err as Error).name })
  }
}
