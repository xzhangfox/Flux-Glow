import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { processFrame, faceFocus, type EditParams } from './pipeline'

const THUMB_PX = 112
const WORK_MAX = 560

/** Renders each param set on this very photo's face (like the filter strip
 *  does), one at a time with a yield in between so opening a panel never
 *  freezes the UI. Done at a reduced working size with the fast pipeline —
 *  plenty for a 56px tile. `zoom` widens the crop (effects need room for
 *  ears and halos). Resolves early if `cancelled()` turns true. */
export async function renderFaceThumbs(
  base: HTMLCanvasElement,
  landmarks: NormalizedLandmark[] | null,
  items: { id: string; params: EditParams }[],
  zoom: number,
  onThumb: (id: string, url: string) => void,
  cancelled: () => boolean,
) {
  const scale = Math.min(1, WORK_MAX / Math.max(base.width, base.height))
  const small = document.createElement('canvas')
  small.width = Math.round(base.width * scale)
  small.height = Math.round(base.height * scale)
  small.getContext('2d')!.drawImage(base, 0, 0, small.width, small.height)
  const focus = faceFocus(landmarks, small.width, small.height)
  const side = Math.min(focus ? focus.size * zoom : Infinity, small.width, small.height)
  const cx = focus ? focus.x : small.width / 2
  const cy = focus ? focus.y - (zoom > 1.2 ? side * 0.08 : 0) : small.height / 2
  const sx = Math.max(0, Math.min(small.width - side, cx - side / 2))
  const sy = Math.max(0, Math.min(small.height - side, cy - side / 2))
  const tile = document.createElement('canvas')
  tile.width = tile.height = THUMB_PX
  const tctx = tile.getContext('2d')!
  for (const item of items) {
    if (cancelled()) return
    const out = processFrame(small, landmarks, item.params, false)
    tctx.drawImage(out, sx, sy, side, side, 0, 0, THUMB_PX, THUMB_PX)
    onThumb(item.id, tile.toDataURL('image/jpeg', 0.82))
    await new Promise((r) => setTimeout(r, 0))
  }
}
