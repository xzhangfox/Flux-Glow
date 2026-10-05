import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { buildZoomPresets, type ZoomRange } from '../lib/zoom'

// The scrubber: a fixed "droplet" lens stays centered while the tick
// ruler slides beneath it, tracking each drag's own delta from wherever it
// started — so the ruler never jumps when a drag begins.
//
// The scale is logarithmic, like a camera's: the same drag always zooms by
// the same ratio, so 0.6x→1x is as easy to hit as 2x→3.3x, and the whole
// range is a short swipe.
const RULER_PX_PER_DOUBLING = 80
const RULER_W = 340
const RULER_H = 64
const LONG_PRESS_MS = 250
// A sideways swipe this long, starting on the button, opens the ruler
// straight away.
const SWIPE_PX = 8

const rulerX = (v: number, min: number) => Math.log2(v / min) * RULER_PX_PER_DOUBLING
const tickStep = (v: number) => (v < 2 - 1e-6 ? 0.1 : v < 5 - 1e-6 ? 0.25 : 1)

function ZoomRuler({ min, max, value, onChange }: { min: number; max: number; value: number; onChange: (v: number) => void }) {
  const dragStartXRef = useRef<number | null>(null)
  const dragStartValueRef = useRef(value)
  const valueRef = useRef(value)
  useEffect(() => {
    valueRef.current = value
  }, [value])

  useEffect(() => {
    function handleMove(e: PointerEvent) {
      if (e.buttons === 0) return
      if (dragStartXRef.current === null) {
        dragStartXRef.current = e.clientX
        dragStartValueRef.current = valueRef.current
        return
      }
      const next = dragStartValueRef.current * 2 ** (-(e.clientX - dragStartXRef.current) / RULER_PX_PER_DOUBLING)
      onChange(Math.min(max, Math.max(min, Math.round(next * 100) / 100)))
    }
    function handleRelease() {
      dragStartXRef.current = null
    }
    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', handleRelease)
    window.addEventListener('pointercancel', handleRelease)
    return () => {
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', handleRelease)
      window.removeEventListener('pointercancel', handleRelease)
    }
  }, [min, max, onChange])

  const ticks: { v: number; major: boolean }[] = [{ v: min, major: true }]
  for (let v = Math.ceil((min + 0.01) * 10) / 10; v <= max + 1e-6; v = Math.round((v + tickStep(v)) * 100) / 100) {
    ticks.push({ v, major: Math.abs(v - Math.round(v)) < 0.001 || Math.abs(v - 0.5) < 0.001 })
  }

  return (
    <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-3 touch-none select-none" style={{ width: `min(${RULER_W}px, calc(100vw - 24px))`, height: RULER_H }}>
      <div className="absolute inset-0 overflow-hidden rounded-2xl bg-black/40 backdrop-blur-md border border-white/10">
        <div className="absolute left-1/2 top-1/2 h-9 w-0" style={{ transform: `translate(${-rulerX(value, min)}px, -50%)` }}>
          {ticks.map((t) => (
            <div key={t.v} className="absolute bottom-0" style={{ left: rulerX(t.v, min) }}>
              <div className={`rounded-full ${t.major ? 'bg-white/80' : 'bg-white/45'}`} style={{ width: t.major ? 2 : 1, height: t.major ? 18 : 10 }} />
              {t.major && <div className="absolute -translate-x-1/2 -top-4 text-[9px] font-semibold text-white/70 whitespace-nowrap">{+t.v.toFixed(1)}</div>}
            </div>
          ))}
        </div>
      </div>
      <div className="absolute left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2 pointer-events-none">
        <div className="relative w-14 h-14 rounded-full bg-white/10 backdrop-blur-md border border-white/40 shadow-lg overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-white/35 via-transparent to-transparent" />
          <div className="absolute left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2 w-[2px] h-7 bg-primary rounded-full" />
        </div>
      </div>
      <div className="absolute left-1/2 -translate-x-1/2 -top-7 text-[11px] font-semibold text-black bg-primary px-2 py-0.5 rounded-full whitespace-nowrap">{value < 1 ? value.toFixed(2).replace(/0$/, '') : value.toFixed(1)}×</div>
    </div>
  )
}

/** Tap to cycle preset stops; press and hold (or swipe sideways) to open
 *  the ruler for anything in between. The ruler stays open across separate drags and
 *  closes on the next tap of the pill — needing to keep holding to keep
 *  fine-tuning would defeat the point of fine adjustment. */
export default function ZoomControl({ range, value, onChange }: { range: ZoomRange; value: number; onChange: (v: number) => void }) {
  const [rulerOpen, setRulerOpen] = useState(false)
  const timerRef = useRef<number | null>(null)
  const movedRef = useRef(false)

  useEffect(
    () => () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current)
    },
    [],
  )

  const cancelTimer = () => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }

  const downXRef = useRef(0)
  // Whether the ruler opened during the current press: letting go then
  // leaves it open; a later tap closes it.
  const openedThisPressRef = useRef(false)
  const open = () => {
    openedThisPressRef.current = true
    setRulerOpen(true)
  }
  const onPointerDown = (e: ReactPointerEvent) => {
    openedThisPressRef.current = false
    if (rulerOpen) return
    downXRef.current = e.clientX
    movedRef.current = false
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null
      open()
    }, LONG_PRESS_MS)
  }
  const onPointerUp = () => {
    if (rulerOpen) {
      if (!openedThisPressRef.current) setRulerOpen(false)
      openedThisPressRef.current = false
      return
    }
    if (timerRef.current !== null) {
      cancelTimer()
      if (!movedRef.current) {
        const presets = buildZoomPresets(range)
        const i = presets.findIndex((p) => Math.abs(p - value) < 0.05)
        onChange(presets[(i + 1) % presets.length] ?? presets[0])
      }
    }
  }
  const onPointerMove = (e: ReactPointerEvent) => {
    if (rulerOpen || timerRef.current === null || Math.abs(e.clientX - downXRef.current) < SWIPE_PX) return
    cancelTimer()
    movedRef.current = true
    open()
  }
  // Leaving the pill's small bounds is exactly what happens when a finger
  // slides up onto the open ruler to drag it — never a close.
  const onPointerLeave = () => {
    if (!rulerOpen) cancelTimer()
  }

  return (
    <div className="relative">
      {rulerOpen && (
        <ZoomRuler
          min={range.min}
          max={range.max}
          value={value}
          onChange={(v) => {
            movedRef.current = true
            onChange(v)
          }}
        />
      )}
      <button
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerLeave}
        aria-label="Zoom — tap to cycle, hold or swipe for fine control"
        className={`w-10 h-10 rounded-full backdrop-blur-md border text-[11px] font-bold flex items-center justify-center select-none touch-none transition ${
          rulerOpen ? 'bg-primary text-black border-primary scale-110' : 'bg-black/45 text-white border-white/20'
        }`}
      >
        {value < 1 ? value.toFixed(2).replace(/0$/, '').replace(/^0/, '') : value < 10 ? value.toFixed(1).replace(/\.0$/, '') : Math.round(value)}×
      </button>
    </div>
  )
}
