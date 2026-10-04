import { useRef } from 'react'
import { hitSticker, type Sticker } from '../lib/stickers'
import { IconClose, IconRotate } from './icons'

const MIN_W = 0.04
const MAX_W = 1.6

type Gesture =
  | { kind: 'drag'; id: number; px: number; py: number; x0: number; y0: number }
  | { kind: 'pinch'; id: number; d0: number; a0: number; w0: number; r0: number; mx0: number; my0: number; x0: number; y0: number }
  | { kind: 'handle'; id: number; d0: number; a0: number; w0: number; r0: number }

// Sits exactly over the photo. Stickers themselves are painted into the
// canvas (so the preview is the export); this layer only handles touch and
// draws the selection frame. One finger drags, two fingers pinch to scale
// and twist to rotate; with a mouse, the corner handle does both and the
// wheel scales.
export default function StickerLayer({
  stickers,
  selectedId,
  width,
  height,
  onSelect,
  onUpdate,
  onDelete,
}: {
  stickers: Sticker[]
  selectedId: number | null
  width: number
  height: number
  onSelect: (id: number | null) => void
  onUpdate: (id: number, patch: Partial<Sticker>) => void
  onDelete: (id: number) => void
}) {
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const gesture = useRef<Gesture | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const byId = (id: number) => stickers.find((s) => s.id === id)

  const local = (e: { clientX: number; clientY: number }) => {
    const r = rootRef.current!.getBoundingClientRect()
    return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height }
  }
  const pinchState = (id: number) => {
    const [a, b] = [...pointers.current.values()]
    const s = byId(id)!
    return {
      kind: 'pinch' as const,
      id,
      d0: Math.hypot((b.x - a.x) * width, (b.y - a.y) * height),
      a0: Math.atan2((b.y - a.y) * height, (b.x - a.x) * width),
      w0: s.w,
      r0: s.rot,
      mx0: (a.x + b.x) / 2,
      my0: (a.y + b.y) / 2,
      x0: s.x,
      y0: s.y,
    }
  }

  const onPointerDown = (e: React.PointerEvent) => {
    const p = local(e)
    pointers.current.set(e.pointerId, p)
    if (pointers.current.size === 2 && gesture.current && gesture.current.kind !== 'handle') {
      gesture.current = pinchState(gesture.current.id)
      ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
      return
    }
    const hit = hitSticker(stickers, p.x, p.y, width, height)
    if (!hit) {
      pointers.current.delete(e.pointerId)
      onSelect(null)
      return
    }
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    onSelect(hit.id)
    gesture.current = { kind: 'drag', id: hit.id, px: p.x, py: p.y, x0: hit.x, y0: hit.y }
  }

  const onPointerMove = (e: React.PointerEvent) => {
    if (!pointers.current.has(e.pointerId)) return
    const p = local(e)
    pointers.current.set(e.pointerId, p)
    const g = gesture.current
    if (!g) return
    if (g.kind === 'drag') {
      onUpdate(g.id, { x: Math.min(1.1, Math.max(-0.1, g.x0 + p.x - g.px)), y: Math.min(1.1, Math.max(-0.1, g.y0 + p.y - g.py)) })
    } else if (g.kind === 'pinch' && pointers.current.size >= 2) {
      const [a, b] = [...pointers.current.values()]
      const d = Math.hypot((b.x - a.x) * width, (b.y - a.y) * height)
      const ang = Math.atan2((b.y - a.y) * height, (b.x - a.x) * width)
      onUpdate(g.id, {
        w: Math.min(MAX_W, Math.max(MIN_W, (g.w0 * d) / Math.max(1, g.d0))),
        rot: g.r0 + ang - g.a0,
        x: g.x0 + (a.x + b.x) / 2 - g.mx0,
        y: g.y0 + (a.y + b.y) / 2 - g.my0,
      })
    } else if (g.kind === 'handle') {
      const s = byId(g.id)
      if (!s) return
      const dx = (p.x - s.x) * width
      const dy = (p.y - s.y) * height
      onUpdate(g.id, { w: Math.min(MAX_W, Math.max(MIN_W, (g.w0 * Math.hypot(dx, dy)) / Math.max(1, g.d0))), rot: g.r0 + Math.atan2(dy, dx) - g.a0 })
    }
  }

  const onPointerUp = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId)
    const g = gesture.current
    if (pointers.current.size === 1 && g && g.kind === 'pinch') {
      // Lifting one finger of a pinch continues as a drag with the other.
      const [p] = [...pointers.current.values()]
      const s = byId(g.id)
      if (s) gesture.current = { kind: 'drag', id: g.id, px: p.x, py: p.y, x0: s.x, y0: s.y }
    } else if (pointers.current.size === 0) {
      gesture.current = null
    }
  }

  const sel = selectedId !== null ? byId(selectedId) : undefined
  const boxW = sel ? sel.w * width : 0
  const boxH = sel ? sel.w * width * sel.ratio : 0

  return (
    <div
      ref={rootRef}
      className="absolute inset-0"
      style={{ touchAction: 'none' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onWheel={(e) => {
        const p = local(e)
        const hit = hitSticker(stickers, p.x, p.y, width, height)
        if (!hit) return
        onSelect(hit.id)
        onUpdate(hit.id, { w: Math.min(MAX_W, Math.max(MIN_W, hit.w * Math.exp(-e.deltaY * 0.0015))) })
      }}
    >
      {sel && (
        <div
          className="absolute pointer-events-none"
          style={{ left: sel.x * width, top: sel.y * height, width: boxW, height: boxH, transform: `translate(-50%, -50%) rotate(${sel.rot}rad)` }}
        >
          <div className="absolute -inset-1.5 rounded-md border-[1.5px] border-dashed border-white/90 shadow-[0_0_0_1px_rgba(0,0,0,0.35)]" />
          <button
            aria-label="Delete sticker"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={() => onDelete(sel.id)}
            className="pointer-events-auto absolute -left-4 -top-4 w-7 h-7 rounded-full bg-white text-black flex items-center justify-center shadow-md"
          >
            <IconClose className="w-3.5 h-3.5" />
          </button>
          <button
            aria-label="Rotate and resize sticker"
            onPointerDown={(e) => {
              e.stopPropagation()
              const p = local(e)
              const dx = (p.x - sel.x) * width
              const dy = (p.y - sel.y) * height
              pointers.current.set(e.pointerId, p)
              ;(rootRef.current as HTMLElement).setPointerCapture(e.pointerId)
              gesture.current = { kind: 'handle', id: sel.id, d0: Math.hypot(dx, dy), a0: Math.atan2(dy, dx), w0: sel.w, r0: sel.rot }
            }}
            className="pointer-events-auto absolute -right-4 -bottom-4 w-7 h-7 rounded-full bg-primary text-black flex items-center justify-center shadow-md"
            style={{ touchAction: 'none' }}
          >
            <IconRotate className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  )
}
