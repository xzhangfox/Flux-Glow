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
const DRAG_SLOP_PX = 4
// The ruler fades out this long after the last adjustment — a little
// longer if it was opened and never moved, so there's time to reach it.
const HIDE_AFTER_MS = 1200
const HIDE_UNUSED_MS = 2000

const rulerX = (v: number, min: number) => Math.log2(v / min) * RULER_PX_PER_DOUBLING
const tickStep = (v: number) => (v < 2 - 1e-6 ? 0.1 : v < 5 - 1e-6 ? 0.25 : 1)

function ZoomRuler({ min, max, value, visible, interactive, onPointerDown }: { min: number; max: number; value: number; visible: boolean; interactive: boolean; onPointerDown: () => void }) {
  const ticks: { v: number; major: boolean }[] = [{ v: min, major: true }]
  for (let v = Math.ceil((min + 0.01) * 10) / 10; v <= max + 1e-6; v = Math.round((v + tickStep(v)) * 100) / 100) {
    ticks.push({ v, major: Math.abs(v - Math.round(v)) < 0.001 || Math.abs(v - 0.5) < 0.001 })
  }

  return (
    <div
      onPointerDown={onPointerDown}
      aria-hidden={!visible}
      className={`absolute left-1/2 -translate-x-1/2 bottom-full mb-3 touch-none select-none transition-opacity duration-300 ${visible ? 'opacity-100' : 'opacity-0'} ${interactive ? '' : 'pointer-events-none'}`}
      style={{ width: `min(${RULER_W}px, calc(100vw - 24px))`, height: RULER_H }}
    >
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
 *  the ruler and drag it for anything in between. The ruler fades away
 *  shortly after the finger lifts (touching it again before then keeps it
 *  and drags on from there); a tap on the pill closes it at once.
 *  `pinching`: the zoom is being pinched on the picture — the ruler shows
 *  (just to read) and follows along. */
export default function ZoomControl({ range, value, onChange, pinching = false }: { range: ZoomRange; value: number; onChange: (v: number) => void; pinching?: boolean }) {
  const [open, setOpen] = useState(false)
  const pressTimerRef = useRef<number | null>(null)
  const hideTimerRef = useRef<number | null>(null)
  const valueRef = useRef(value)
  useEffect(() => {
    valueRef.current = value
  }, [value])

  // A drag of the ruler: armed by a press on the ruler (or the pill press
  // that opened it), it follows that finger wherever it goes until it lifts.
  const armedRef = useRef(false)
  const dragRef = useRef<{ x: number; v: number; live: boolean } | null>(null)
  // Whether the current press moved the zoom (then it's no tap).
  const movedRef = useRef(false)

  const clearTimer = (r: { current: number | null }) => {
    if (r.current !== null) {
      clearTimeout(r.current)
      r.current = null
    }
  }
  const hideSoon = (ms: number) => {
    clearTimer(hideTimerRef)
    hideTimerRef.current = window.setTimeout(() => {
      hideTimerRef.current = null
      setOpen(false)
    }, ms)
  }
  const arm = () => {
    clearTimer(hideTimerRef)
    armedRef.current = true
    dragRef.current = null
    setOpen(true)
  }

  useEffect(() => {
    function handleMove(e: PointerEvent) {
      if (!armedRef.current || e.buttons === 0) return
      const d = dragRef.current
      if (d === null) {
        dragRef.current = { x: e.clientX, v: valueRef.current, live: false }
        return
      }
      // A finger's jitter isn't a drag: it starts once it has really
      // moved, from where it is then (so nothing jumps).
      if (!d.live) {
        if (Math.abs(e.clientX - d.x) < DRAG_SLOP_PX) return
        dragRef.current = { x: e.clientX, v: valueRef.current, live: true }
        return
      }
      const next = d.v * 2 ** (-(e.clientX - d.x) / RULER_PX_PER_DOUBLING)
      const v = Math.min(range.max, Math.max(range.min, Math.round(next * 100) / 100))
      if (v !== valueRef.current) {
        movedRef.current = true
        onChange(v)
      }
    }
    function handleRelease() {
      if (!armedRef.current) return
      armedRef.current = false
      dragRef.current = null
      hideSoon(movedRef.current ? HIDE_AFTER_MS : HIDE_UNUSED_MS)
    }
    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', handleRelease)
    window.addEventListener('pointercancel', handleRelease)
    return () => {
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', handleRelease)
      window.removeEventListener('pointercancel', handleRelease)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [range.min, range.max, onChange])

  useEffect(
    () => () => {
      clearTimer(pressTimerRef)
      clearTimer(hideTimerRef)
    },
    [],
  )

  const downXRef = useRef(0)
  // Whether the ruler already showed when this press began.
  const wasOpenRef = useRef(false)
  const onPointerDown = (e: ReactPointerEvent) => {
    movedRef.current = false
    downXRef.current = e.clientX
    wasOpenRef.current = open
    if (open) {
      // Already open: this press drags it on.
      arm()
      return
    }
    pressTimerRef.current = window.setTimeout(() => {
      pressTimerRef.current = null
      arm()
    }, LONG_PRESS_MS)
  }
  const onPointerUp = () => {
    if (wasOpenRef.current) {
      // A tap on the pill while the ruler shows: close it now. (Runs
      // before the window's release, which then finds nothing armed.)
      if (!movedRef.current) {
        armedRef.current = false
        clearTimer(hideTimerRef)
        setOpen(false)
      }
      return
    }
    if (pressTimerRef.current !== null) {
      clearTimer(pressTimerRef)
      if (!movedRef.current) {
        const presets = buildZoomPresets(range)
        const i = presets.findIndex((p) => Math.abs(p - value) < 0.05)
        onChange(presets[(i + 1) % presets.length] ?? presets[0])
      }
    }
  }
  const onPointerMove = (e: ReactPointerEvent) => {
    if (pressTimerRef.current === null || Math.abs(e.clientX - downXRef.current) < SWIPE_PX) return
    clearTimer(pressTimerRef)
    arm()
  }
  // Leaving the pill's small bounds is exactly what happens when a finger
  // slides up onto the open ruler to drag it — never a close.
  const onPointerLeave = () => {
    if (!armedRef.current) clearTimer(pressTimerRef)
  }

  const shown = open || pinching
  return (
    <div className="relative">
      <ZoomRuler min={range.min} max={range.max} value={value} visible={shown} interactive={open} onPointerDown={arm} />
      <button
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerLeave}
        aria-label="Zoom — tap to cycle, hold or swipe for fine control, or pinch the picture"
        className={`w-10 h-10 rounded-full backdrop-blur-md border text-[11px] font-bold flex items-center justify-center select-none touch-none transition ${
          shown ? 'bg-primary text-black border-primary scale-110' : 'bg-black/45 text-white border-white/20'
        }`}
      >
        {value < 1 ? value.toFixed(2).replace(/0$/, '').replace(/^0/, '') : value < 10 ? value.toFixed(1).replace(/\.0$/, '') : Math.round(value)}×
      </button>
    </div>
  )
}
