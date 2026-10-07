// Sculpts and meshes the wig off the page's main thread (see wig.ts).
import { computeWig, wigTransferables } from './wig'

self.onmessage = () => {
  try {
    const data = structuredClone(computeWig())
    ;(self as unknown as Worker).postMessage({ data }, wigTransferables(data))
  } catch (err) {
    ;(self as unknown as Worker).postMessage({ error: String(err) })
  }
}
