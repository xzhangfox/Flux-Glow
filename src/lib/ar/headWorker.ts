// Sculpts and meshes an animal head off the page's main thread (see
// shiba.ts), so the camera keeps running while it's made.
import { computeHead, transferablesOf, type HeadId } from './shiba'

self.onmessage = (e: MessageEvent<HeadId>) => {
  const id = e.data
  try {
    const data = computeHead(id)
    // (copies: the worker keeps its own for a second request)
    const copy = structuredClone(data)
    ;(self as unknown as Worker).postMessage({ id, data: copy }, transferablesOf(copy))
  } catch (err) {
    ;(self as unknown as Worker).postMessage({ id, error: String(err) })
  }
}
