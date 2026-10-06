import { useEffect, useRef, useState } from 'react'
import { IconClose, IconCheck } from './icons'

// Square cropper for the Custom effect: the photo sits under a fixed square
// frame; drag to move it, pinch / scroll / the slider to zoom. The image
// always covers the square, so the result never has empty corners.

const OUT = 512

export default function CropDialog({ file, onDone, onCancel }: { file: File; onDone: (square: HTMLCanvasElement) => void; onCancel: () => void }) {
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState(300)
  // View: zoom (1 = the image's short side just fills the square) and the
  // image centre's offset from the square's centre, in square pixels.
  const [zoom, setZoom] = useState(1)
  const [off, setOff] = useState({ x: 0, y: 0 })
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const pinch = useRef<{ dist: number; zoom: number } | null>(null)

  useEffect(() => {
    const url = URL.createObjectURL(file)
    const im = new Image()
    im.onload = () => setImg(im)
    im.src = url
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    const el = boxRef.current
    if (!el) return
    const update = () => setBox(el.clientWidth)
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Displayed size of the image at this zoom.
  const base = img ? box / Math.min(img.naturalWidth, img.naturalHeight) : 1
  const dw = img ? img.naturalWidth * base * zoom : box
  const dh = img ? img.naturalHeight * base * zoom : box
  // Keep the square covered.
  const clamp = (o: { x: number; y: number }, w = dw, h = dh) => ({
    x: Math.max(-(w - box) / 2, Math.min((w - box) / 2, o.x)),
    y: Math.max(-(h - box) / 2, Math.min((h - box) / 2, o.y)),
  })
  const setZoomClamped = (z: number) => {
    const nz = Math.max(1, Math.min(5, z))
    setZoom(nz)
    if (img) setOff((o) => clamp(o, img.naturalWidth * base * nz, img.naturalHeight * base * nz))
  }

  const onDown = (e: React.PointerEvent) => {
    ;(e.target as Element).setPointerCapture(e.pointerId)
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY })
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()]
      pinch.current = { dist: Math.hypot(a.x - b.x, a.y - b.y), zoom }
    }
  }
  const onMove = (e: React.PointerEvent) => {
    const prev = pointers.current.get(e.pointerId)
    if (!prev) return
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY })
    if (pointers.current.size === 2 && pinch.current) {
      const [a, b] = [...pointers.current.values()]
      setZoomClamped(pinch.current.zoom * (Math.hypot(a.x - b.x, a.y - b.y) / pinch.current.dist))
    } else if (pointers.current.size === 1) {
      setOff((o) => clamp({ x: o.x + e.clientX - prev.x, y: o.y + e.clientY - prev.y }))
    }
  }
  const onUp = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId)
    if (pointers.current.size < 2) pinch.current = null
  }

  const done = () => {
    if (!img) return
    const c = document.createElement('canvas')
    c.width = c.height = OUT
    const k = OUT / box
    // The square's top-left in display px relative to the image's top-left.
    const left = (dw - box) / 2 - off.x
    const top = (dh - box) / 2 - off.y
    const s = 1 / (base * zoom)
    c.getContext('2d')!.drawImage(img, left * s, top * s, box * s, box * s, 0, 0, box * k, box * k)
    onDone(c)
  }

  return (
    <div data-keep-panel className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex flex-col items-center justify-center px-6 select-none">
      <p className="text-white text-[15px] font-semibold mb-1">Custom face sticker</p>
      <p className="text-white/60 text-[12px] mb-4 text-center">Drag and pinch so the part you want fills the square — it will cover your face.</p>
      <div
        ref={boxRef}
        className="relative w-full max-w-[320px] aspect-square overflow-hidden rounded-2xl touch-none bg-white/5"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        onWheel={(e) => setZoomClamped(zoom * Math.exp(-e.deltaY * 0.0015))}
      >
        {img && (
          <img
            src={img.src}
            alt=""
            draggable={false}
            className="absolute max-w-none pointer-events-none"
            style={{ width: dw, height: dh, left: (box - dw) / 2 + off.x, top: (box - dh) / 2 + off.y }}
          />
        )}
        {/* Rule-of-thirds guide and a face-sized oval, for framing. */}
        <div className="absolute inset-0 pointer-events-none ring-2 ring-inset ring-primary rounded-2xl" />
        <div className="absolute inset-[14%] pointer-events-none rounded-[50%] border border-dashed border-white/50" />
      </div>
      <input type="range" min={1} max={5} step={0.01} value={zoom} onChange={(e) => setZoomClamped(Number(e.target.value))} aria-label="Zoom" className="w-full max-w-[320px] mt-5 accent-primary" />
      <div className="flex gap-3 mt-5">
        <button onClick={onCancel} className="h-11 px-5 rounded-full bg-white/10 text-white text-sm font-semibold flex items-center gap-1.5">
          <IconClose className="w-4 h-4" /> Cancel
        </button>
        <button onClick={done} disabled={!img} className="h-11 px-5 rounded-full bg-primary text-black text-sm font-semibold flex items-center gap-1.5 disabled:opacity-40">
          <IconCheck className="w-4 h-4" /> Use it
        </button>
      </div>
    </div>
  )
}
