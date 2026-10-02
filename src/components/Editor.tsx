import { useEffect, useRef, useState, useCallback, type ComponentType, type ReactNode, type SVGProps } from 'react'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { detectFaceLandmarks, detectFaceLandmarksForVideo } from '../lib/faceLandmarker'
import { processFrame, type EditParams } from '../lib/pipeline'
import { FILTER_PRESETS } from '../lib/filters'
import Slider from './Slider'
import {
  IconSpinner,
  IconEye,
  IconCamera,
  IconInfo,
  IconError,
  IconClose,
  IconCheck,
  IconEdit,
  IconDownload,
  IconShare,
  IconRefresh,
  IconFaceOutline,
  IconPalette,
  IconFlipCamera,
  IconDroplet,
  IconImage,
  IconBack,
  IconSparkle,
  IconSun,
  IconWhiten,
  IconTarget,
  IconWave,
  RegionIcon,
} from './icons'

// Downscale before processing — phone photos run 3000px+ on a side, far
// more detail than this editor displays or than frequency separation
// needs; working at this size keeps every slider drag responsive.
const MAX_DIMENSION = 1600
// Live mode used to need a smaller working size than a captured photo
// because every pass — smoothing *and* reshape — was plain JS, not GPU
// shaders; reshape moved to a WebGL mesh warp (see meshWarp.ts), so the
// only remaining per-pixel CPU cost is smoothing (bounded to the face's
// own bounding box) and the small isolated nose warp, both cheap enough
// to afford matching the camera's own capture resolution instead of
// downscaling further. This is also the canvas's actual backing-store
// resolution (see `render()` — it's sized to match the processed frame,
// then stretched via CSS to fill the screen), so anything lower than
// what the camera actually delivers is a real, visible loss of sharpness
// on a high-DPI phone, not a hidden margin of safety.
const LIVE_MAX_DIMENSION = 1280
const LIVE_FRAME_INTERVAL_MS = 33 // ~30fps ceiling, not a promise — actual pace is still gated by processingRef below, so a slower device just falls short of it instead of backlogging

export type Source = { kind: 'image'; file: File } | { kind: 'live' }

type Status = 'loading' | 'ready' | 'no-face' | 'error'

type RegionKey = 'face' | 'temple' | 'cheekbone' | 'eyes' | 'eyebrow' | 'nose' | 'noseBridge' | 'mouth'
type MouthSubKey = 'mouthSize' | 'mouthUpperLip' | 'mouthLowerLip' | 'mouthCorners' | 'mouthCornerSmooth'
type BeautyKey = 'smooth' | 'fillLight' | 'whitening' | 'acneRemoval' | 'wrinkleRemoval'

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = URL.createObjectURL(file)
  })
}

// `zoom` crops a centered region of the source before scaling it up to
// fill the canvas — a digital stand-in for a focal-length switcher, since
// getUserMedia doesn't expose a phone's separate physical lenses. `mirror`
// flips horizontally: the front camera's raw stream reads as a mirror
// image (text backwards, etc.) unless something corrects it, and nothing
// upstream does — so the live preview, capture, and anything saved from it
// all need this to show a true (non-mirrored) orientation.
// MediaPipe detects landmarks on the raw `video` element directly — never
// on `base`, so its normalized coordinates are relative to the
// *unmirrored* frame regardless of what drawDownscaled did. Every reshape
// pass (and the skin mask, and the smoothing bounds) turns those
// coordinates straight into pixel positions on `base`, so whenever `base`
// was mirrored and/or digitally cropped (zoomed) and the landmarks
// weren't adjusted to match, every one of them pointed at the wrong
// position relative to the actual feature in `base` — for mirroring, the
// mirror-reflected x instead of the real one (an eye warp centered on
// whatever half-eyebrow/half-nose-bridge pixel happened to sit at that
// reflected spot, which is exactly the "horror movie" look a real,
// not-quite-symmetric face produces); for digital zoom, a position
// outside the actual cropped-and-rescaled frame entirely once zoomed in
// enough. This maps raw detection-space landmarks into `base`'s own
// coordinate space given the same (zoom, mirror) drawDownscaled used.
function remapLandmarksToBase(landmarks: NormalizedLandmark[] | null, zoom: number, mirror: boolean): NormalizedLandmark[] | null {
  if (!landmarks) return landmarks
  const cropFrac = zoom > 1 ? (1 - 1 / zoom) / 2 : 0
  return landmarks.map((p) => {
    let x = zoom > 1 ? (p.x - cropFrac) * zoom : p.x
    const y = zoom > 1 ? (p.y - cropFrac) * zoom : p.y
    if (mirror) x = 1 - x
    return { ...p, x, y }
  })
}

function drawDownscaled(source: HTMLImageElement | HTMLVideoElement, maxDim: number, zoom = 1, mirror = false): HTMLCanvasElement {
  const w = source instanceof HTMLVideoElement ? source.videoWidth : source.width
  const h = source instanceof HTMLVideoElement ? source.videoHeight : source.height
  const scale = Math.min(1, maxDim / Math.max(w, h))
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(w * scale)
  canvas.height = Math.round(h * scale)
  const ctx = canvas.getContext('2d')!
  if (mirror) {
    ctx.translate(canvas.width, 0)
    ctx.scale(-1, 1)
  }
  if (zoom > 1) {
    const cropW = w / zoom
    const cropH = h / zoom
    ctx.drawImage(source, (w - cropW) / 2, (h - cropH) / 2, cropW, cropH, 0, 0, canvas.width, canvas.height)
  } else {
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height)
  }
  return canvas
}

interface ZoomRange {
  min: number
  max: number
  mode: 'hardware' | 'digital'
}

// Digital zoom beyond ~4x is just cropping an already-modest live frame
// down to a quarter of its linear size — well past where it still looks
// like a lens and not a blown-up JPEG, so that's the fallback ceiling when
// the camera doesn't expose real optical/sensor zoom.
const DIGITAL_ZOOM_RANGE: ZoomRange = { min: 1, max: 4, mode: 'digital' }

// A handful of tap-to-cycle presets spanning the real available range
// (hardware min/max when the device exposes `MediaStreamTrack.zoom`
// capabilities, the digital fallback range otherwise) — rounded to values
// someone would actually reach for (0.5x ultra-wide, 1x, 2x, 3x…) rather
// than an arbitrary fixed [1, 2] regardless of what the hardware can do.
// Holding the button instead of tapping it opens a dial for anything
// continuous in between.
function buildZoomPresets({ min, max }: ZoomRange): number[] {
  const presets = new Set<number>()
  if (min < 1) presets.add(Math.round(min * 10) / 10)
  presets.add(1)
  for (const v of [2, 3, 5]) {
    if (v > min && v <= max) presets.add(v)
  }
  if (max > 1) presets.add(Math.round(max * 10) / 10)
  return Array.from(presets)
    .filter((v) => v >= min - 0.001 && v <= max + 0.001)
    .sort((a, b) => a - b)
}

// A fan/sector-shaped ruler, shown while the zoom pill is held, swept by
// dragging left-right — a protractor-like arc of tick marks rather than a
// straight track, closer to the dedicated zoom ring a real camera has.
// Reads clientX straight off `window` pointermove rather than using
// pointer capture: the finger is already down on the pill (not this fan)
// when the long-press timer opens it, and pointermove bubbles to window
// regardless of which element is currently under the finger, so this needs
// no capture handoff from the pill to work. Horizontal position across the
// whole control maps directly to a value in [min, max] (the same
// absolute-position-is-value approach the previous vertical strip used,
// just along the other axis) — simpler and more predictable than treating
// it as a literal rotating dial someone has to sweep an arc to turn.
// A 270° sweep (not just the ~110° wedge a semicircle-ish fan would give)
// — this reaches well past horizontal on both sides, which is why the
// pivot sits vertically centered in a taller box instead of at its bottom
// edge: at ±135° from straight up, the arc's own ends are already *below*
// the pivot's own height (sin/cos of 135° puts them out and down), so
// there has to be room for those two "wings" to hang below pivot level,
// not just space above it.
const FAN_MAX_ANGLE_DEG = 135
const FAN_PIVOT = { x: 140, y: 108 }
const FAN_OUTER_R = 92
const FAN_INNER_R = 66
const FAN_VIEW_W = 280
const FAN_VIEW_H = 200

function fanPoint(angleDeg: number, radius: number) {
  const rad = (angleDeg * Math.PI) / 180
  return { x: FAN_PIVOT.x + radius * Math.sin(rad), y: FAN_PIVOT.y - radius * Math.cos(rad) }
}

function ZoomFanDial({ min, max, value, onChange }: { min: number; max: number; value: number; onChange: (v: number) => void }) {
  const trackRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const handleMove = (e: PointerEvent) => {
      // Now that the fan stays open after a release (see handleZoomPressEnd),
      // a pointermove listener on `window` would otherwise react to every
      // idle mouse hover across the whole page, not just an actual drag —
      // harmless before, when the fan only existed for the instant a finger
      // was physically down on it, but very much not now. `buttons` is 0
      // whenever nothing is pressed, for both mouse and touch.
      if (e.buttons === 0) return
      const el = trackRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const fraction = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
      onChange(Math.round((min + fraction * (max - min)) * 10) / 10)
    }
    window.addEventListener('pointermove', handleMove)
    return () => window.removeEventListener('pointermove', handleMove)
  }, [min, max, onChange])

  const fraction = max - min > 0 ? (value - min) / (max - min) : 0
  const needleAngle = -FAN_MAX_ANGLE_DEG + fraction * (2 * FAN_MAX_ANGLE_DEG)
  const needleTip = fanPoint(needleAngle, FAN_OUTER_R)
  const arcStart = fanPoint(-FAN_MAX_ANGLE_DEG, FAN_OUTER_R)
  const arcEnd = fanPoint(FAN_MAX_ANGLE_DEG, FAN_OUTER_R)
  // 270° is more than a half-circle, so the large-arc-flag has to be 1 —
  // with it left at 0 (right for the ~110° sweep this used to be), SVG
  // would silently draw the *short* way around instead (90° the wrong way).
  const sectorPath = `M ${FAN_PIVOT.x} ${FAN_PIVOT.y} L ${arcStart.x} ${arcStart.y} A ${FAN_OUTER_R} ${FAN_OUTER_R} 0 1 1 ${arcEnd.x} ${arcEnd.y} Z`
  const tickCount = 25

  return (
    <div ref={trackRef} className="absolute left-1/2 -translate-x-1/2 bottom-56 w-[280px] h-[200px] touch-none">
      <svg viewBox={`0 0 ${FAN_VIEW_W} ${FAN_VIEW_H}`} className="w-full h-full overflow-visible">
        <path d={sectorPath} fill="rgba(15,15,15,0.55)" stroke="rgba(255,255,255,0.18)" strokeWidth={1} />
        {Array.from({ length: tickCount }, (_, i) => {
          const t = i / (tickCount - 1)
          const angle = -FAN_MAX_ANGLE_DEG + t * (2 * FAN_MAX_ANGLE_DEG)
          const major = i === 0 || i === tickCount - 1 || i === (tickCount - 1) / 2
          const inner = fanPoint(angle, major ? FAN_INNER_R - 6 : FAN_INNER_R)
          const outer = fanPoint(angle, FAN_OUTER_R - 3)
          return <line key={i} x1={inner.x} y1={inner.y} x2={outer.x} y2={outer.y} stroke="rgba(255,255,255,0.45)" strokeWidth={1.4} strokeLinecap="round" />
        })}
        <line x1={FAN_PIVOT.x} y1={FAN_PIVOT.y} x2={needleTip.x} y2={needleTip.y} className="stroke-primary" strokeWidth={2.5} strokeLinecap="round" />
        <circle cx={FAN_PIVOT.x} cy={FAN_PIVOT.y} r={4} className="fill-primary" />
      </svg>
      <div className="absolute left-1/2 -translate-x-1/2 top-0 text-[11px] font-semibold text-black bg-primary px-2 py-0.5 rounded-full whitespace-nowrap">
        {value.toFixed(1)}×
      </div>
    </div>
  )
}

// A round icon-over-label button for the bottom chrome's side clusters —
// Retouch, Retake, Save, Share.
function ChromeButton({
  icon: Icon,
  label,
  onClick,
  active,
  primary,
}: {
  icon: ComponentType<SVGProps<SVGSVGElement>>
  label: string
  onClick: () => void
  active?: boolean
  primary?: boolean
}) {
  return (
    <button onClick={onClick} aria-label={label} className="flex flex-col items-center gap-1 w-14 text-white/85">
      <span
        className={`w-11 h-11 rounded-full flex items-center justify-center backdrop-blur-sm border transition ${
          active || primary ? 'bg-primary text-black border-primary' : 'bg-black/40 border-white/15'
        }`}
      >
        <Icon className="w-5 h-5" />
      </span>
      <span className="text-[10px] font-medium leading-none">{label}</span>
    </button>
  )
}

export default function Editor({ source, onReset, onPickImage }: { source: Source; onReset: () => void; onPickImage?: (file: File) => void }) {
  const [status, setStatus] = useState<Status>('loading')
  const [smoothness, setSmoothness] = useState(0.6)
  const [face, setFace] = useState(0.25)
  const [eyes, setEyes] = useState(0)
  const [nose, setNose] = useState(0)
  const [mouth, setMouth] = useState(0)
  const [eyebrowHeight, setEyebrowHeight] = useState(0)
  const [noseBridge, setNoseBridge] = useState(0)
  const [temple, setTemple] = useState(0)
  const [cheekbone, setCheekbone] = useState(0)
  const [mouthUpperLip, setMouthUpperLip] = useState(0)
  const [mouthLowerLip, setMouthLowerLip] = useState(0)
  const [mouthCorners, setMouthCorners] = useState(0)
  const [fillLight, setFillLight] = useState(0)
  const [whitening, setWhitening] = useState(0)
  const [acneRemoval, setAcneRemoval] = useState(0)
  const [wrinkleRemoval, setWrinkleRemoval] = useState(0)
  const [mouthCornerSmooth, setMouthCornerSmooth] = useState(0)
  const [filterId, setFilterId] = useState('none')
  const [showBefore, setShowBefore] = useState(false)
  const [liveActive, setLiveActive] = useState(source.kind === 'live')
  const [confirmed, setConfirmed] = useState(false)
  const [openPanel, setOpenPanel] = useState<'retouch' | 'beauty' | 'filter' | null>(null)
  const [selectedRegion, setSelectedRegion] = useState<RegionKey | null>(null)
  const [selectedMouthSub, setSelectedMouthSub] = useState<MouthSubKey | null>(null)
  const [selectedBeauty, setSelectedBeauty] = useState<BeautyKey | null>(null)
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user')
  const [zoom, setZoom] = useState(1)
  const [zoomRange, setZoomRange] = useState<ZoomRange>(DIGITAL_ZOOM_RANGE)
  const [zoomDialOpen, setZoomDialOpen] = useState(false)

  const displayRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const baseRef = useRef<HTMLCanvasElement | null>(null)
  const landmarksRef = useRef<NormalizedLandmark[] | null>(null)
  const resultRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef(0)
  const lastProcessRef = useRef(0)
  const processingRef = useRef(false)
  const liveActiveRef = useRef(liveActive)
  const paramsRef = useRef<EditParams>({ smoothness, face, eyes, nose, mouth, eyebrowHeight, noseBridge, temple, cheekbone, mouthUpperLip, mouthLowerLip, mouthCorners, fillLight, whitening, acneRemoval, wrinkleRemoval, mouthCornerSmooth, filterId })
  const facingModeRef = useRef(facingMode)
  const zoomRef = useRef(zoom)
  const zoomRangeRef = useRef(zoomRange)
  const zoomTrackRef = useRef<MediaStreamTrack | null>(null)
  const zoomPressTimerRef = useRef<number | null>(null)
  const zoomPressMovedRef = useRef(false)
  const panelRef = useRef<HTMLDivElement>(null)
  const retouchButtonRef = useRef<HTMLButtonElement>(null)
  const beautyButtonRef = useRef<HTMLButtonElement>(null)
  const filterButtonRef = useRef<HTMLButtonElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    paramsRef.current = { smoothness, face, eyes, nose, mouth, eyebrowHeight, noseBridge, temple, cheekbone, mouthUpperLip, mouthLowerLip, mouthCorners, fillLight, whitening, acneRemoval, wrinkleRemoval, mouthCornerSmooth, filterId }
  }, [smoothness, face, eyes, nose, mouth, eyebrowHeight, noseBridge, temple, cheekbone, mouthUpperLip, mouthLowerLip, mouthCorners, fillLight, whitening, acneRemoval, wrinkleRemoval, mouthCornerSmooth, filterId])
  useEffect(() => {
    liveActiveRef.current = liveActive
  }, [liveActive])
  useEffect(() => {
    zoomRef.current = zoom
  }, [zoom])
  useEffect(() => {
    zoomRangeRef.current = zoomRange
  }, [zoomRange])


  // Leaving a panel always resets it back to its own top-level grid, so
  // reopening it never silently drops the visitor into whichever
  // slider/sub-grid they happened to be adjusting last time.
  useEffect(() => {
    if (openPanel !== 'retouch') {
      setSelectedRegion(null)
      setSelectedMouthSub(null)
    }
  }, [openPanel])
  useEffect(() => {
    if (selectedRegion !== 'mouth') setSelectedMouthSub(null)
  }, [selectedRegion])
  useEffect(() => {
    if (openPanel !== 'beauty') setSelectedBeauty(null)
  }, [openPanel])

  // Tapping anywhere outside the open panel (or the Retouch/Filter buttons
  // that toggle it, which handle themselves) collapses it — same pattern
  // as a tap-away menu.
  useEffect(() => {
    function onClick(e: MouseEvent) {
      const target = e.target as Node
      if (panelRef.current?.contains(target)) return
      if (retouchButtonRef.current?.contains(target)) return
      if (beautyButtonRef.current?.contains(target)) return
      if (filterButtonRef.current?.contains(target)) return
      setOpenPanel(null)
    }
    // Capture phase, not bubble: a click inside the panel (e.g. picking a
    // region) can make React synchronously swap that exact element out of
    // the DOM (region grid -> slider) as part of handling the very same
    // click. By the time a bubble-phase listener on `document` ran, the
    // clicked node was already detached, and a detached node's
    // `.contains()` check always reads as "outside" no matter where it
    // used to be — closing the whole panel the moment anyone picked a
    // region. Capture fires before the target's own handlers (and any
    // resulting DOM mutation), while the node is still exactly where this
    // check needs it to be.
    document.addEventListener('click', onClick, true)
    return () => document.removeEventListener('click', onClick, true)
  }, [])

  const render = useCallback(() => {
    const canvas = displayRef.current
    const base = baseRef.current
    const result = resultRef.current
    if (!canvas || !base) return
    canvas.width = base.width
    canvas.height = base.height
    canvas.getContext('2d')!.drawImage(showBefore || !result ? base : result, 0, 0)
  }, [showBefore])

  const recomputeStatic = useCallback(() => {
    const base = baseRef.current
    if (!base) return
    resultRef.current = processFrame(base, landmarksRef.current, paramsRef.current)
    render()
  }, [render])

  const stopLive = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    zoomTrackRef.current = null
    // In case a detect+process cycle was still in flight (e.g. stopped
    // mid-capture) — don't leave a future startLive() permanently stuck
    // behind a guard that'll never clear on its own.
    processingRef.current = false
    if (zoomPressTimerRef.current !== null) {
      clearTimeout(zoomPressTimerRef.current)
      zoomPressTimerRef.current = null
    }
    setZoomDialOpen(false)
  }, [])

  const startLive = useCallback(async () => {
    setStatus('loading')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facingModeRef.current, width: { ideal: 1280 }, height: { ideal: 1280 } },
      })
      streamRef.current = stream
      const video = videoRef.current!
      video.srcObject = stream
      await video.play()
      setStatus('ready')
      setLiveActive(true)

      // Real optical/sensor zoom when the browser exposes it (mainly the
      // rear camera on Android Chrome today) beats a digital crop — no
      // resolution lost to cropping a frame that was never much bigger
      // than what we display. Each camera (front vs back) can have a
      // different range, so this re-detects on every startLive(), which
      // handleFlipCamera already calls via stop+start.
      const track = stream.getVideoTracks()[0]
      zoomTrackRef.current = track
      const caps = track.getCapabilities?.() as (MediaTrackCapabilities & { zoom?: { min: number; max: number; step: number } }) | undefined
      if (caps?.zoom && caps.zoom.max > caps.zoom.min) {
        const range: ZoomRange = { min: caps.zoom.min, max: caps.zoom.max, mode: 'hardware' }
        zoomRangeRef.current = range
        setZoomRange(range)
        const initial = Math.min(range.max, Math.max(range.min, 1))
        zoomRef.current = initial
        setZoom(initial)
        track.applyConstraints({ advanced: [{ zoom: initial } as unknown as MediaTrackConstraintSet] }).catch(() => {})
      } else {
        zoomRangeRef.current = DIGITAL_ZOOM_RANGE
        setZoomRange(DIGITAL_ZOOM_RANGE)
        zoomRef.current = 1
        setZoom(1)
      }

      const loop = () => {
        rafRef.current = requestAnimationFrame(loop)
        if (!liveActiveRef.current) return
        // Without this guard, a frame that takes longer than the interval
        // to process (landmark detection + skin mask + frequency-separation
        // smoothing + reshape warps is real work, easily >60ms) doesn't
        // skip a beat — the next rAF tick only checks elapsed time, so it
        // fires another overlapping detect+process cycle anyway. Those pile
        // up faster than they resolve, and the preview falls further and
        // further behind "live" the longer it runs. Only ever start a new
        // cycle once the previous one has actually finished.
        if (processingRef.current) return
        const now = performance.now()
        if (now - lastProcessRef.current < LIVE_FRAME_INTERVAL_MS) return
        lastProcessRef.current = now
        if (video.readyState < 2) return
        processingRef.current = true
        // Hardware zoom already zoomed the sensor output itself — cropping
        // again on top of that would double-zoom.
        const cropZoom = zoomRangeRef.current.mode === 'hardware' ? 1 : zoomRef.current
        const base = drawDownscaled(video, LIVE_MAX_DIMENSION, cropZoom, facingModeRef.current === 'user')
        baseRef.current = base
        detectFaceLandmarksForVideo(video, now)
          .then((rawLandmarks) => {
            const landmarks = remapLandmarksToBase(rawLandmarks, cropZoom, facingModeRef.current === 'user')
            landmarksRef.current = landmarks
            setStatus(landmarks ? 'ready' : 'no-face')
            resultRef.current = processFrame(base, landmarks, paramsRef.current, false)
            render()
          })
          .finally(() => {
            processingRef.current = false
          })
      }
      loop()
    } catch {
      setStatus('error')
    }
  }, [render])

  // Load an uploaded photo once.
  useEffect(() => {
    if (source.kind !== 'image') return
    let cancelled = false
    setStatus('loading')
    // Switching source from live to an uploaded photo (the new toolbar
    // upload icon) is a mid-session transition the live/capture flow
    // never used to need — picking a photo always used to mean Editor
    // itself was just being mounted fresh with an image source, so
    // nothing previously had to reset `liveActive` on an existing
    // instance. Without this, the LIVE badge and flip-camera button kept
    // showing over a now-static photo.
    setLiveActive(false)
    setConfirmed(false)
    ;(async () => {
      try {
        const img = await loadImage(source.file)
        const base = drawDownscaled(img, MAX_DIMENSION)
        if (cancelled) return
        baseRef.current = base
        const landmarks = await detectFaceLandmarks(base)
        if (cancelled) return
        landmarksRef.current = landmarks
        setStatus(landmarks ? 'ready' : 'no-face')
        recomputeStatic()
      } catch {
        if (!cancelled) setStatus('error')
      }
    })()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  // Start the camera once for a live source.
  useEffect(() => {
    if (source.kind !== 'live') return
    startLive()
    return () => stopLive()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  // Re-render when the before/after toggle flips.
  useEffect(() => {
    render()
  }, [render])

  // Recompute on control changes — only while not actively streaming live
  // (the live loop already re-applies the latest params every frame on
  // its own, using paramsRef).
  useEffect(() => {
    if (liveActive) return
    if (!baseRef.current) return
    const t = setTimeout(recomputeStatic, 60)
    return () => clearTimeout(t)
  }, [smoothness, face, eyes, nose, mouth, eyebrowHeight, noseBridge, temple, cheekbone, mouthUpperLip, mouthLowerLip, mouthCorners, fillLight, whitening, acneRemoval, wrinkleRemoval, mouthCornerSmooth, filterId, liveActive, recomputeStatic])

  // Any further adjustment after confirming means the exported image would
  // no longer match what's on screen — fall back to Confirm again rather
  // than silently leaving a stale Save/Share up.
  useEffect(() => {
    setConfirmed(false)
  }, [smoothness, face, eyes, nose, mouth, eyebrowHeight, noseBridge, temple, cheekbone, mouthUpperLip, mouthLowerLip, mouthCorners, fillLight, whitening, acneRemoval, wrinkleRemoval, mouthCornerSmooth, filterId])

  const handleCapture = () => {
    stopLive()
    setLiveActive(false)
    setConfirmed(false)
  }

  const handleRetake = () => {
    setConfirmed(false)
    setOpenPanel(null)
    startLive()
  }

  const handleConfirm = () => {
    setConfirmed(true)
    setOpenPanel(null)
  }

  const handleFlipCamera = () => {
    const next = facingMode === 'user' ? 'environment' : 'user'
    facingModeRef.current = next
    setFacingMode(next)
    stopLive()
    startLive()
  }

  // Shared by the tap-to-cycle presets and the long-press dial — keeps the
  // actual hardware track (when we have real optical/sensor zoom) in sync
  // with whatever value the UI just settled on, instead of only updating
  // the digital crop factor and silently ignoring the lens.
  const applyZoom = (value: number) => {
    const { min, max, mode } = zoomRangeRef.current
    const clamped = Math.min(max, Math.max(min, value))
    zoomRef.current = clamped
    setZoom(clamped)
    if (mode === 'hardware' && zoomTrackRef.current) {
      zoomTrackRef.current.applyConstraints({ advanced: [{ zoom: clamped } as unknown as MediaTrackConstraintSet] }).catch(() => {})
    }
  }

  const handleToggleZoom = () => {
    const presets = buildZoomPresets(zoomRange)
    const i = presets.findIndex((p) => Math.abs(p - zoom) < 0.05)
    const next = presets[(i + 1 + presets.length) % presets.length] ?? presets[0]
    applyZoom(next)
  }

  // Tap the zoom pill to cycle presets; press and hold it to open a dial
  // for anything continuous in between — the same two-tier interaction
  // most phone camera apps use, since a tap-only cycle can't reach, say,
  // 2.4x, and a drag-only dial is overkill for the common "just go to 2x"
  // case. The fan stays open across multiple separate drags once a
  // long-press opens it — releasing mid-adjustment doesn't close it, only
  // a direct tap on the pill while it's already open does — since needing
  // to hold the whole time to keep fine-tuning open defeats the point of
  // "fine" adjustment.
  const ZOOM_LONG_PRESS_MS = 350
  const handleZoomPressStart = () => {
    if (zoomDialOpen) return // this press's matching release is the close-tap, not a new long-press
    zoomPressMovedRef.current = false
    zoomPressTimerRef.current = window.setTimeout(() => {
      zoomPressTimerRef.current = null
      setZoomDialOpen(true)
    }, ZOOM_LONG_PRESS_MS)
  }
  // A genuine release on the pill itself: closes the fan if it's already
  // open (the only way to close it now that it persists across drags),
  // otherwise this was a quick tap — cycle presets.
  const handleZoomPointerUp = () => {
    if (zoomDialOpen) {
      setZoomDialOpen(false)
      return
    }
    if (zoomPressTimerRef.current !== null) {
      clearTimeout(zoomPressTimerRef.current)
      zoomPressTimerRef.current = null
      if (!zoomPressMovedRef.current) handleToggleZoom()
    }
  }
  // Only ever cancels a *pending* long-press timer — never closes an
  // already-open fan. The finger leaving the pill's small bounds is
  // exactly what happens the instant someone slides up onto the fan
  // itself to start dragging it; that's not a release.
  const handleZoomPointerLeave = () => {
    if (zoomDialOpen) return
    if (zoomPressTimerRef.current !== null) {
      clearTimeout(zoomPressTimerRef.current)
      zoomPressTimerRef.current = null
    }
  }

  const disabled = status === 'no-face'

  const resultBlob = (): Promise<Blob | null> => {
    const canvas = resultRef.current ?? baseRef.current
    if (!canvas) return Promise.resolve(null)
    return new Promise((resolve) => canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.95))
  }

  const handleDownload = async () => {
    const blob = await resultBlob()
    if (!blob) return
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'flux-glow.jpg'
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleShare = async () => {
    const blob = await resultBlob()
    if (!blob) return
    const file = new File([blob], 'flux-glow.jpg', { type: 'image/jpeg' })
    if (navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({ files: [file], title: 'Flux Glow' })
        return
      } catch {
        // user cancelled or share failed — fall through to download
      }
    }
    handleDownload()
  }

  // The retouch drill-down's region list — tapping one in the grid morphs
  // the same row into this region's own slider (see the panel JSX below).
  // `active` drives the highlighted-icon state in the grid, so someone can
  // see at a glance which regions they've already touched without having
  // to open each one.
  const regions: { key: RegionKey; label: string; icon: ReactNode; value: number; onChange: (v: number) => void; bidirectional: boolean; active: boolean }[] = [
    { key: 'face', label: 'Jaw', icon: <RegionIcon dot={[12, 16.8]} className="w-5 h-5" />, value: face, onChange: setFace, bidirectional: true, active: face !== 0.25 },
    { key: 'temple', label: 'Temple', icon: <RegionIcon dot={[7.2, 7.6]} pair className="w-5 h-5" />, value: temple, onChange: setTemple, bidirectional: true, active: temple !== 0 },
    { key: 'cheekbone', label: 'Cheekbone', icon: <RegionIcon dot={[6.8, 11.5]} pair className="w-5 h-5" />, value: cheekbone, onChange: setCheekbone, bidirectional: true, active: cheekbone !== 0 },
    { key: 'eyes', label: 'Eyes', icon: <RegionIcon dot={[9, 10.2]} pair className="w-5 h-5" />, value: eyes, onChange: setEyes, bidirectional: true, active: eyes !== 0 },
    { key: 'eyebrow', label: 'Eyebrow', icon: <RegionIcon dot={[9, 8]} pair className="w-5 h-5" />, value: eyebrowHeight, onChange: setEyebrowHeight, bidirectional: true, active: eyebrowHeight !== 0 },
    { key: 'nose', label: 'Nose', icon: <RegionIcon dot={[12, 12.5]} className="w-5 h-5" />, value: nose, onChange: setNose, bidirectional: true, active: nose !== 0 },
    { key: 'noseBridge', label: 'Bridge', icon: <RegionIcon dot={[12, 9.3]} className="w-5 h-5" />, value: noseBridge, onChange: setNoseBridge, bidirectional: true, active: noseBridge !== 0 },
    {
      key: 'mouth',
      label: 'Mouth',
      icon: <RegionIcon dot={[12, 14.3]} className="w-5 h-5" />,
      value: mouth,
      onChange: setMouth,
      bidirectional: true,
      active: mouth !== 0 || mouthUpperLip !== 0 || mouthLowerLip !== 0 || mouthCorners !== 0 || mouthCornerSmooth !== 0,
    },
  ]
  // Mouth drills one level deeper than every other region: tapping it opens
  // its own sub-grid (size/upper lip/lower lip/corners/fold) instead of
  // going straight to a slider, since "mouth" bundles several independently
  // adjustable things rather than being one knob the way jaw/nose are.
  const mouthSubRegions: { key: MouthSubKey; label: string; icon: ReactNode; value: number; onChange: (v: number) => void; bidirectional: boolean; active: boolean }[] = [
    { key: 'mouthSize', label: 'Size', icon: <RegionIcon dot={[12, 14.3]} className="w-5 h-5" />, value: mouth, onChange: setMouth, bidirectional: true, active: mouth !== 0 },
    { key: 'mouthUpperLip', label: 'Upper Lip', icon: <RegionIcon dot={[12, 13.4]} className="w-5 h-5" />, value: mouthUpperLip, onChange: setMouthUpperLip, bidirectional: true, active: mouthUpperLip !== 0 },
    { key: 'mouthLowerLip', label: 'Lower Lip', icon: <RegionIcon dot={[12, 15.3]} className="w-5 h-5" />, value: mouthLowerLip, onChange: setMouthLowerLip, bidirectional: true, active: mouthLowerLip !== 0 },
    { key: 'mouthCorners', label: 'Corners', icon: <RegionIcon dot={[9.3, 14.3]} pair className="w-5 h-5" />, value: mouthCorners, onChange: setMouthCorners, bidirectional: true, active: mouthCorners !== 0 },
    { key: 'mouthCornerSmooth', label: 'Fold', icon: <RegionIcon dot={[9, 15.6]} pair className="w-5 h-5" />, value: mouthCornerSmooth, onChange: setMouthCornerSmooth, bidirectional: false, active: mouthCornerSmooth !== 0 },
  ]
  const beautyRegions: { key: BeautyKey; label: string; icon: ReactNode; value: number; onChange: (v: number) => void; bidirectional: boolean; active: boolean }[] = [
    { key: 'smooth', label: 'Smooth', icon: <IconDroplet className="w-5 h-5" />, value: smoothness, onChange: setSmoothness, bidirectional: false, active: smoothness !== 0.6 },
    { key: 'fillLight', label: 'Fill Light', icon: <IconSun className="w-5 h-5" />, value: fillLight, onChange: setFillLight, bidirectional: false, active: fillLight !== 0 },
    { key: 'whitening', label: 'Whiten', icon: <IconWhiten className="w-5 h-5" />, value: whitening, onChange: setWhitening, bidirectional: false, active: whitening !== 0 },
    { key: 'acneRemoval', label: 'Acne', icon: <IconTarget className="w-5 h-5" />, value: acneRemoval, onChange: setAcneRemoval, bidirectional: false, active: acneRemoval !== 0 },
    { key: 'wrinkleRemoval', label: 'Wrinkle', icon: <IconWave className="w-5 h-5" />, value: wrinkleRemoval, onChange: setWrinkleRemoval, bidirectional: false, active: wrinkleRemoval !== 0 },
  ]
  const activeRegion = regions.find((r) => r.key === selectedRegion) ?? null
  const activeMouthSub = mouthSubRegions.find((r) => r.key === selectedMouthSub) ?? null
  const activeBeauty = beautyRegions.find((r) => r.key === selectedBeauty) ?? null

  let centerButton: React.ReactNode
  if (liveActive) {
    centerButton = (
      <button
        onClick={handleCapture}
        disabled={status !== 'ready'}
        aria-label="Capture"
        className="w-16 h-16 rounded-full bg-primary border-4 border-white/80 shadow-glow-strong flex items-center justify-center hover:brightness-110 transition disabled:opacity-50"
      >
        <IconCamera className="w-7 h-7 text-black" />
      </button>
    )
  } else if (!confirmed) {
    centerButton = (
      <button
        onClick={handleConfirm}
        disabled={status === 'loading'}
        aria-label="Confirm"
        className="w-16 h-16 rounded-full bg-primary border-4 border-white/20 shadow-glow-strong flex items-center justify-center hover:brightness-110 transition disabled:opacity-40"
      >
        <IconCheck className="w-7 h-7 text-black" />
      </button>
    )
  } else {
    centerButton = (
      <button
        onClick={() => setConfirmed(false)}
        aria-label="Edit"
        className="w-16 h-16 rounded-full bg-black/50 border-4 border-white/20 backdrop-blur-sm flex items-center justify-center text-white hover:bg-black/60 transition"
      >
        <IconEdit className="w-6 h-6" />
      </button>
    )
  }

  return (
    <div className="fixed inset-0 bg-black overflow-hidden select-none">
      <canvas ref={displayRef} className="absolute inset-0 w-full h-full object-cover" />
      <video ref={videoRef} autoPlay playsInline muted className="hidden" />

      <div className="absolute top-0 inset-x-0 h-28 bg-gradient-to-b from-black/70 to-transparent pointer-events-none" />

      {status === 'loading' && (
        <div className="absolute inset-0 flex items-center justify-center">
          <IconSpinner className="w-9 h-9 text-primary animate-spin" />
        </div>
      )}

      <div className="absolute inset-x-0 flex items-center justify-between px-4" style={{ top: 'max(1rem, env(safe-area-inset-top))' }}>
        <button
          onClick={onReset}
          aria-label="Close"
          className="w-10 h-10 rounded-full bg-black/40 backdrop-blur-sm flex items-center justify-center text-white"
        >
          <IconClose className="w-5 h-5" />
        </button>
        <div className="flex items-center gap-2.5">
          {liveActive && (
            <button
              onClick={handleFlipCamera}
              disabled={status === 'loading'}
              aria-label="Flip camera"
              className="w-10 h-10 rounded-full bg-black/40 backdrop-blur-sm flex items-center justify-center text-white disabled:opacity-40"
            >
              <IconFlipCamera className="w-5 h-5" />
            </button>
          )}
          <button
            onMouseDown={() => setShowBefore(true)}
            onMouseUp={() => setShowBefore(false)}
            onMouseLeave={() => setShowBefore(false)}
            onTouchStart={() => setShowBefore(true)}
            onTouchEnd={() => setShowBefore(false)}
            disabled={status === 'loading'}
            aria-label="Hold to compare with the original"
            className={`w-10 h-10 rounded-full backdrop-blur-sm flex items-center justify-center transition ${
              showBefore ? 'bg-primary text-black' : 'bg-black/40 text-white'
            }`}
          >
            <IconEye className="w-5 h-5" />
          </button>
        </div>
      </div>

      {liveActive && status !== 'loading' && (
        <>
          {zoomDialOpen && (
            <ZoomFanDial
              min={zoomRange.min}
              max={zoomRange.max}
              value={zoom}
              onChange={(v) => {
                zoomPressMovedRef.current = true
                applyZoom(v)
              }}
            />
          )}
          <button
            onPointerDown={handleZoomPressStart}
            onPointerUp={handleZoomPointerUp}
            onPointerLeave={handleZoomPointerLeave}
            aria-label="Zoom level — tap to cycle, hold for fine control"
            className={`absolute left-1/2 -translate-x-1/2 bottom-44 w-9 h-9 rounded-full backdrop-blur-sm border text-white text-[11px] font-semibold flex items-center justify-center select-none touch-none transition ${
              zoomDialOpen ? 'bg-primary text-black border-primary scale-110' : 'bg-black/50 border-white/20'
            }`}
          >
            {zoom < 10 ? zoom.toFixed(1).replace(/\.0$/, '') : Math.round(zoom)}×
          </button>
        </>
      )}

      {liveActive && status !== 'loading' && (
        <div
          className="absolute left-4 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur-sm border border-white/10"
          style={{ top: 'calc(max(1rem, env(safe-area-inset-top)) + 3.25rem)' }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-danger animate-pulse" />
          <span className="text-[10px] font-semibold tracking-wide text-white/90">LIVE</span>
        </div>
      )}

      <div className="absolute inset-x-0 bottom-0 flex flex-col">
        {(status === 'no-face' || status === 'error') && (
          <p
            className={`self-center mb-3 text-xs text-center flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-sm ${
              status === 'error' ? 'text-danger' : 'text-white/80'
            }`}
          >
            {status === 'error' ? <IconError className="w-3.5 h-3.5 flex-shrink-0" /> : <IconInfo className="w-3.5 h-3.5 flex-shrink-0" />}
            {status === 'error'
              ? source.kind === 'live'
                ? "Couldn't access the camera — check your browser permissions."
                : "Couldn't load that photo."
              : 'No face detected — smooth and contour need one, filters still work.'}
          </p>
        )}

        <div
          ref={panelRef}
          className={`grid transition-all duration-250 ease-out px-4 ${openPanel ? 'grid-rows-[1fr] opacity-100 mb-3' : 'grid-rows-[0fr] opacity-0'}`}
          style={{ transitionProperty: 'grid-template-rows, opacity, margin' }}
        >
          <div className="overflow-hidden">
            <div className="bg-black/55 backdrop-blur-2xl rounded-2xl p-4 border border-white/10">
              {openPanel === 'retouch' ? (
                selectedRegion === 'mouth' ? (
                  activeMouthSub ? (
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => setSelectedMouthSub(null)}
                        aria-label="Back to mouth"
                        className="w-8 h-8 rounded-full bg-white/10 border border-white/15 flex items-center justify-center text-white flex-shrink-0"
                      >
                        <IconBack className="w-4 h-4" />
                      </button>
                      <div className="flex-1">
                        <Slider label={activeMouthSub.label} value={activeMouthSub.value} onChange={activeMouthSub.onChange} bidirectional={activeMouthSub.bidirectional} disabled={disabled} />
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => setSelectedRegion(null)}
                        aria-label="Back to regions"
                        className="w-8 h-8 rounded-full bg-white/10 border border-white/15 flex items-center justify-center text-white flex-shrink-0"
                      >
                        <IconBack className="w-4 h-4" />
                      </button>
                      <div className="flex items-center gap-4 overflow-x-auto pb-0.5 -mx-1 px-1">
                        {mouthSubRegions.map((r) => (
                          <button key={r.key} onClick={() => setSelectedMouthSub(r.key)} aria-label={`Adjust ${r.label}`} className="flex flex-col items-center gap-1 flex-shrink-0 w-14">
                            <span
                              className={`w-11 h-11 rounded-full flex items-center justify-center backdrop-blur-sm border transition ${
                                r.active ? 'bg-primary/20 border-primary text-primary' : 'bg-white/5 border-white/15 text-white/85'
                              }`}
                            >
                              {r.icon}
                            </span>
                            <span className="text-[10px] font-medium leading-none text-white/85 whitespace-nowrap">{r.label}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )
                ) : activeRegion ? (
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setSelectedRegion(null)}
                      aria-label="Back to regions"
                      className="w-8 h-8 rounded-full bg-white/10 border border-white/15 flex items-center justify-center text-white flex-shrink-0"
                    >
                      <IconBack className="w-4 h-4" />
                    </button>
                    <div className="flex-1">
                      <Slider label={activeRegion.label} value={activeRegion.value} onChange={activeRegion.onChange} bidirectional={activeRegion.bidirectional} disabled={disabled} />
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center gap-4 overflow-x-auto pb-0.5 -mx-1 px-1">
                    {regions.map((r) => (
                      <button key={r.key} onClick={() => setSelectedRegion(r.key)} aria-label={`Adjust ${r.label}`} className="flex flex-col items-center gap-1 flex-shrink-0 w-14">
                        <span
                          className={`w-11 h-11 rounded-full flex items-center justify-center backdrop-blur-sm border transition ${
                            r.active ? 'bg-primary/20 border-primary text-primary' : 'bg-white/5 border-white/15 text-white/85'
                          }`}
                        >
                          {r.icon}
                        </span>
                        <span className="text-[10px] font-medium leading-none text-white/85 whitespace-nowrap">{r.label}</span>
                      </button>
                    ))}
                  </div>
                )
              ) : openPanel === 'beauty' ? (
                activeBeauty ? (
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setSelectedBeauty(null)}
                      aria-label="Back to beauty"
                      className="w-8 h-8 rounded-full bg-white/10 border border-white/15 flex items-center justify-center text-white flex-shrink-0"
                    >
                      <IconBack className="w-4 h-4" />
                    </button>
                    <div className="flex-1">
                      <Slider label={activeBeauty.label} value={activeBeauty.value} onChange={activeBeauty.onChange} bidirectional={activeBeauty.bidirectional} disabled={disabled} />
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center gap-4 overflow-x-auto pb-0.5 -mx-1 px-1">
                    {beautyRegions.map((r) => (
                      <button key={r.key} onClick={() => setSelectedBeauty(r.key)} aria-label={`Adjust ${r.label}`} className="flex flex-col items-center gap-1 flex-shrink-0 w-14">
                        <span
                          className={`w-11 h-11 rounded-full flex items-center justify-center backdrop-blur-sm border transition ${
                            r.active ? 'bg-primary/20 border-primary text-primary' : 'bg-white/5 border-white/15 text-white/85'
                          }`}
                        >
                          {r.icon}
                        </span>
                        <span className="text-[10px] font-medium leading-none text-white/85 whitespace-nowrap">{r.label}</span>
                      </button>
                    ))}
                  </div>
                )
              ) : (
                <div className="grid grid-cols-3 gap-2">
                  {FILTER_PRESETS.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => setFilterId(p.id)}
                      className={`py-2 rounded-lg text-xs font-medium transition border ${
                        filterId === p.id ? 'bg-primary text-black border-primary' : 'bg-secondary text-white/70 border-transparent hover:bg-white/10'
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* The frosted toolbar tray — a native-camera-style shutter button
            straddles its top edge (half inside the tray, half protruding
            into the preview above it), rather than sitting in the row with
            everything else. */}
        <div
          className="relative bg-black/55 backdrop-blur-2xl border-t border-white/10 rounded-t-[28px] px-4 pt-5"
          style={{ paddingBottom: 'max(1.1rem, env(safe-area-inset-bottom))' }}
        >
          {/* Centered independently of the upload icon beside it — sharing
              one centered flex group (as this used to) centers the *pair*,
              which pulls the shutter itself off-center to the left. The
              upload icon is instead positioned at a fixed offset to the
              shutter's right (half the shutter's own width, plus a gap). */}
          <div className="absolute left-1/2 -translate-x-1/2 -top-8">{centerButton}</div>
          {liveActive && !confirmed && (
            <button
              onClick={() => fileInputRef.current?.click()}
              aria-label="Upload a photo instead"
              className="absolute -top-2 w-9 h-9 rounded-full bg-white/15 backdrop-blur-sm border border-white/25 flex items-center justify-center text-white"
              style={{ left: 'calc(50% + 44px)' }}
            >
              <IconImage className="w-4 h-4" />
            </button>
          )}

          <div className="grid grid-cols-2 items-center gap-2">
            <div className="flex items-center gap-3 justify-self-start">
              <button
                ref={retouchButtonRef}
                onClick={() => setOpenPanel((v) => (v === 'retouch' ? null : 'retouch'))}
                aria-label="Retouch"
                className="flex flex-col items-center gap-1 w-14 text-white/85"
              >
                <span
                  className={`w-11 h-11 rounded-full flex items-center justify-center backdrop-blur-sm border transition ${
                    openPanel === 'retouch' ? 'bg-primary text-black border-primary' : 'bg-white/5 border-white/15'
                  }`}
                >
                  <IconFaceOutline className="w-5 h-5" />
                </span>
                <span className="text-[10px] font-medium leading-none">Retouch</span>
              </button>
              <button
                ref={beautyButtonRef}
                onClick={() => setOpenPanel((v) => (v === 'beauty' ? null : 'beauty'))}
                aria-label="Beauty"
                className="flex flex-col items-center gap-1 w-14 text-white/85"
              >
                <span
                  className={`w-11 h-11 rounded-full flex items-center justify-center backdrop-blur-sm border transition ${
                    openPanel === 'beauty' ? 'bg-primary text-black border-primary' : 'bg-white/5 border-white/15'
                  }`}
                >
                  <IconSparkle className="w-5 h-5" />
                </span>
                <span className="text-[10px] font-medium leading-none">Beauty</span>
              </button>
              <button
                ref={filterButtonRef}
                onClick={() => setOpenPanel((v) => (v === 'filter' ? null : 'filter'))}
                aria-label="Filter"
                className="flex flex-col items-center gap-1 w-14 text-white/85"
              >
                <span
                  className={`w-11 h-11 rounded-full flex items-center justify-center backdrop-blur-sm border transition ${
                    openPanel === 'filter' ? 'bg-primary text-black border-primary' : 'bg-white/5 border-white/15'
                  }`}
                >
                  <IconPalette className="w-5 h-5" />
                </span>
                <span className="text-[10px] font-medium leading-none">Filter</span>
              </button>
              {source.kind === 'live' && !liveActive && <ChromeButton icon={IconRefresh} label="Retake" onClick={handleRetake} />}
            </div>

            <div className="flex items-center gap-3 justify-self-end">
              {confirmed && (
                <>
                  <ChromeButton icon={IconDownload} label="Save" primary onClick={handleDownload} />
                  <ChromeButton icon={IconShare} label="Share" onClick={handleShare} />
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onPickImage?.(file)
          e.target.value = ''
        }}
      />
    </div>
  )
}
