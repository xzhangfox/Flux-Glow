import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ButtonHTMLAttributes, type ComponentType, type ReactNode, type RefObject, type SVGProps } from 'react'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { detectFaceLandmarks, LiveFaceTracker } from '../lib/faceLandmarker'
import { processFrame, DEFAULT_PARAMS, faceFocus, type EditParams, type NumericParam } from '../lib/pipeline'
import { renderFilterThumbnails } from '../lib/filters'
import { ASPECT_MODES, aspectRatioFor, cropRectFor, drawFrame, remapLandmarks, type AspectMode } from '../lib/frame'
import AdjustPanel, { type AdjustItem } from './AdjustPanel'
import FilterPanel from './FilterPanel'
import ThumbStrip from './ThumbStrip'
import Slider from './Slider'
import StickerPanel, { type StickerRequest } from './StickerPanel'
import StickerLayer from './StickerLayer'
import { SHAPE_PARAMS } from '../lib/deform'
import { LOOKS, applyLook, findLook } from '../lib/looks'
import { EFFECTS, findEffect, preloadAR, setCustomImage } from '../lib/effects'
import CropDialog from './CropDialog'
import { renderFaceThumbs } from '../lib/thumbs'
import { artImage, drawStickers, emojiCanvas, photoSticker, placeSticker, textCanvas, type Sticker } from '../lib/stickers'
import ZoomControl from './ZoomControl'
import { DIGITAL_ZOOM_RANGE, cropZoom, wideZoom, type ZoomRange } from '../lib/zoom'
import {
  IconSpinner,
  IconCompare,
  IconInfo,
  IconError,
  IconClose,
  IconCheck,
  IconDownload,
  IconShare,
  IconFaceOutline,
  IconPalette,
  IconFlipCamera,
  IconDroplet,
  IconImage,
  IconSparkle,
  IconSun,
  IconWhiten,
  IconTarget,
  IconWave,
  IconTimer,
  IconGrid,
  RegionIcon,
  IconWand,
  IconEars,
  IconSticker,
  IconRefresh,
  IconPlus,
} from './icons'

// Saved photos and uploads are worked on at up to this size — phone photos
// run 3000px+ on a side, far beyond what this editor displays or what the
// retouching passes need, and a smaller working size keeps every slider
// drag responsive.
const MAX_DIMENSION = 1600
// The live viewfinder's working size. It is also the display canvas's real
// backing-store resolution, so going lower would visibly soften the preview
// on a high-DPI phone rather than buy hidden headroom.
const LIVE_MAX_DIMENSION = 1280
// ~30fps ceiling. Actual pace is still gated by the in-flight guard in the
// loop, so a slower device just falls short of it instead of backlogging.
const LIVE_FRAME_INTERVAL_MS = 33
const TIMER_STEPS = [0, 3, 10] as const

const BEAUTY_KEYS: NumericParam[] = ['smoothness', 'whitening', 'acneRemoval', 'wrinkleRemoval', 'mouthCornerSmooth', 'fillLight']
const SHAPE_KEYS: NumericParam[] = [...SHAPE_PARAMS]

export type Source = { kind: 'image'; file: File } | { kind: 'live' }

type Status = 'loading' | 'ready' | 'no-face' | 'error'
type Panel = 'looks' | 'beauty' | 'shape' | 'filter' | 'effects' | 'stickers'

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      URL.revokeObjectURL(url)
      resolve(img)
    }
    img.onerror = (e) => {
      URL.revokeObjectURL(url)
      reject(e)
    }
    img.src = url
  })
}

const FULL_FRAME = { x0: 0, y0: 0, fw: 1, fh: 1 }

const nextPaint = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)))

const isIOS = typeof navigator !== 'undefined' && (/iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1))

function timestampedName(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `FluxGlow_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}.jpg`
}

function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function useElementSize(ref: RefObject<HTMLElement | null>) {
  const [size, setSize] = useState({ width: 0, height: 0 })
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => setSize((s) => (s.width === el.clientWidth && s.height === el.clientHeight ? s : { width: el.clientWidth, height: el.clientHeight }))
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return size
}

// Fits a frame of the given aspect ratio inside a box — the canvas is
// sized to exactly what will be saved, never cropped by `object-cover`,
// so the preview is what you get.
function fitContain(box: { width: number; height: number }, aspect: number) {
  if (!box.width || !box.height || !aspect) return { width: 0, height: 0 }
  let width = box.width
  let height = width / aspect
  if (height > box.height) {
    height = box.height
    width = height * aspect
  }
  return { width: Math.round(width), height: Math.round(height) }
}

function TopButton({ label, active, children, ...rest }: { label: string; active?: boolean; children: ReactNode } & Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children' | 'className'>) {
  return (
    <button
      aria-label={label}
      className={`h-10 min-w-10 px-2.5 rounded-full backdrop-blur-md flex items-center justify-center gap-1 text-[11px] font-semibold transition disabled:opacity-40 ${
        active ? 'bg-primary text-black' : 'bg-black/40 text-white'
      }`}
      {...rest}
    >
      {children}
    </button>
  )
}

function TrayButton({ icon: Icon, label, onClick, active, dot, toggle, compact }: { icon: ComponentType<SVGProps<SVGSVGElement>>; label: string; onClick: () => void; active?: boolean; dot?: boolean; toggle?: boolean; compact?: boolean }) {
  // Compact: icon only, smaller — the slim bar under an open panel.
  if (compact)
    return (
      <button onClick={onClick} aria-label={label} aria-pressed={toggle ? !!active : undefined} data-panel-toggle={toggle ? '' : undefined} className="text-white/85">
        <span className={`relative w-9 h-9 rounded-full flex items-center justify-center transition ${active ? 'bg-primary text-black' : 'bg-white/[0.06]'}`}>
          <Icon className="w-[18px] h-[18px]" />
          {dot && !active && <span className="absolute top-0 right-0 w-2 h-2 rounded-full bg-primary ring-2 ring-black/60" />}
        </span>
      </button>
    )
  return (
    <button onClick={onClick} aria-label={label} aria-pressed={toggle ? !!active : undefined} data-panel-toggle={toggle ? '' : undefined} className="flex flex-col items-center gap-1.5 w-[3.15rem] text-white/85">
      <span className={`relative w-11 h-11 rounded-full flex items-center justify-center border transition ${active ? 'bg-primary text-black border-primary' : 'bg-white/[0.06] border-white/15'}`}>
        <Icon className="w-5 h-5" />
        {dot && !active && <span className="absolute top-0.5 right-0.5 w-2 h-2 rounded-full bg-primary ring-2 ring-black/60" />}
      </span>
      <span className="text-[10.5px] font-medium leading-none">{label}</span>
    </button>
  )
}

function GridOverlay() {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {[1, 2].map((i) => (
        <div key={`v${i}`} className="absolute top-0 bottom-0 w-px bg-white/35" style={{ left: `${(i * 100) / 3}%` }} />
      ))}
      {[1, 2].map((i) => (
        <div key={`h${i}`} className="absolute left-0 right-0 h-px bg-white/35" style={{ top: `${(i * 100) / 3}%` }} />
      ))}
    </div>
  )
}

export default function Editor({ source, onReset, onPickImage, onTrySample }: { source: Source; onReset: () => void; onPickImage?: (file: File) => void; onTrySample?: () => void }) {
  const [params, setParams] = useState<EditParams>(DEFAULT_PARAMS)
  const [status, setStatus] = useState<Status>('loading')
  const [live, setLive] = useState(source.kind === 'live')
  // A photo just taken with the camera (not an upload): shown and edited
  // exactly as the viewfinder showed it — same framing, same face fit,
  // same rendering — nothing re-detected or re-processed differently.
  const [captured, setCaptured] = useState(false)
  const capturedRef = useRef(false)
  // Video: long-press the shutter to record.
  const [recordingSince, setRecordingSince] = useState<number | null>(null)
  const [recordTick, setRecordTick] = useState(0)
  const recRef = useRef<{ rec: MediaRecorder; chunks: Blob[]; audio: MediaStream | null } | null>(null)
  const pressTimerRef = useRef<number | null>(null)
  const pressHandledRef = useRef(false)
  const [processing, setProcessing] = useState(false)
  const [panel, setPanel] = useState<Panel | null>(null)
  const [showBefore, setShowBefore] = useState(false)
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user')
  const [zoom, setZoom] = useState(1)
  const [zoomRange, setZoomRange] = useState<ZoomRange>(DIGITAL_ZOOM_RANGE)
  const [aspect, setAspect] = useState<AspectMode>('full')
  const [timer, setTimer] = useState<(typeof TIMER_STEPS)[number]>(0)
  const [grid, setGrid] = useState(false)
  const [countdown, setCountdown] = useState<number | null>(null)
  const [flashKey, setFlashKey] = useState(0)
  const [toast, setToast] = useState<{ key: number; text: string } | null>(null)
  const [frameAspect, setFrameAspect] = useState(3 / 4)
  const [frameVersion, setFrameVersion] = useState(0)
  const [thumbs, setThumbs] = useState<Map<string, string> | null>(null)
  const [lookId, setLookId] = useState<string | null>(null)
  const [lookStrength, setLookStrength] = useState(1)
  const [lookThumbs, setLookThumbs] = useState<Record<string, string>>({})
  const [effectThumbs, setEffectThumbs] = useState<Record<string, string>>({})
  const [stickers, setStickers] = useState<Sticker[]>([])
  const [selectedSticker, setSelectedSticker] = useState<number | null>(null)

  const displayRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const previewRef = useRef<HTMLDivElement>(null)
  const trayRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  // Custom effect: the photo being cropped, and the cropped square's preview.
  const customInputRef = useRef<HTMLInputElement>(null)
  const [cropFile, setCropFile] = useState<File | null>(null)
  const [customUrl, setCustomUrl] = useState<string | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const zoomTrackRef = useRef<MediaStreamTrack | null>(null)
  const baseRef = useRef<HTMLCanvasElement | null>(null)
  const landmarksRef = useRef<NormalizedLandmark[] | null>(null)
  const resultRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef(0)
  const lastFrameRef = useRef(0)
  const frameInFlightRef = useRef(false)
  // Bumped on every stop/start of the camera. A live frame whose detection
  // is still in flight when the shutter fires (or the camera flips) must
  // not land afterwards and overwrite the captured photo with a stale,
  // preview-quality frame — each async step checks it still owns the loop.
  const liveGenRef = useRef(0)
  const staticBusyRef = useRef(false)
  const paramsRef = useRef(params)
  const showBeforeRef = useRef(showBefore)
  const facingModeRef = useRef(facingMode)
  const zoomRef = useRef(zoom)
  const zoomRangeRef = useRef(zoomRange)
  const aspectRef = useRef(aspect)
  const frameAspectRef = useRef(frameAspect)
  const countdownTimerRef = useRef<number | null>(null)

  const stickersRef = useRef<Sticker[]>(stickers)
  paramsRef.current = params
  showBeforeRef.current = showBefore
  stickersRef.current = stickers

  const previewBox = useElementSize(previewRef)
  const trayBox = useElementSize(trayRef)
  const fit = fitContain(previewBox, frameAspect)

  const set = (key: NumericParam) => (v: number) => setParams((p) => ({ ...p, [key]: v }))
  const showToast = (text: string) => setToast({ key: Date.now(), text })

  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 1800)
    return () => clearTimeout(t)
  }, [toast])

  // Tapping anywhere outside the open panel (or the tray buttons that
  // toggle it, which handle themselves) collapses it. Capture phase: a
  // click inside the panel can synchronously swap the clicked node out of
  // the DOM (grid -> slider) before a bubble-phase listener runs, and a
  // detached node always reads as "outside".
  useEffect(() => {
    function onClick(e: MouseEvent) {
      const target = e.target as Element
      if (panelRef.current?.contains(target)) return
      if (target.closest?.('[data-panel-toggle], [data-keep-panel]')) return
      setPanel(null)
    }
    document.addEventListener('click', onClick, true)
    return () => document.removeEventListener('click', onClick, true)
  }, [])

  const render = useCallback(() => {
    const canvas = displayRef.current
    const base = baseRef.current
    if (!canvas || !base) return
    const src = showBeforeRef.current || !resultRef.current ? base : resultRef.current
    if (canvas.width !== src.width || canvas.height !== src.height) {
      canvas.width = src.width
      canvas.height = src.height
    }
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(src, 0, 0)
    if (!showBeforeRef.current && stickersRef.current.length) drawStickers(ctx, stickersRef.current, canvas.width, canvas.height)
    const a = base.width / base.height
    if (Math.abs(a - frameAspectRef.current) > 0.002) {
      frameAspectRef.current = a
      setFrameAspect(a)
    }
  }, [])

  useEffect(() => {
    render()
  }, [showBefore, stickers, render])

  const recomputeStatic = useCallback(() => {
    const base = baseRef.current
    if (!base) return
    resultRef.current = processFrame(base, landmarksRef.current, paramsRef.current, !capturedRef.current)
    render()
  }, [render])

  const cancelCountdown = useCallback(() => {
    if (countdownTimerRef.current !== null) clearTimeout(countdownTimerRef.current)
    countdownTimerRef.current = null
    setCountdown(null)
  }, [])

  const stopLive = useCallback(() => {
    liveGenRef.current++
    cancelAnimationFrame(rafRef.current)
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    zoomTrackRef.current = null
    frameInFlightRef.current = false
  }, [])

  const startLive = useCallback(async () => {
    stopLive()
    const gen = liveGenRef.current
    setLive(true)
    setStatus('loading')
    try {
      // No `height` constraint: pinning both dimensions makes the browser
      // center-crop the lens's real field of view before our own framing
      // ever runs. A 4:3 shape instead: phone sensors are natively 4:3, and
      // the 16:9 modes browsers otherwise pick cut a quarter of the width
      // off a portrait selfie.
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facingModeRef.current, width: { ideal: 1440 }, aspectRatio: { ideal: 4 / 3 } } })
      if (gen !== liveGenRef.current) {
        stream.getTracks().forEach((t) => t.stop())
        return
      }
      streamRef.current = stream
      const video = videoRef.current!
      video.srcObject = stream
      await video.play()
      if (gen !== liveGenRef.current) return

      // Real optical/sensor zoom when the browser exposes it beats a
      // digital crop. Front and back cameras differ, so this re-detects on
      // every start (flipping goes through here too).
      const track = stream.getVideoTracks()[0]
      zoomTrackRef.current = track
      const caps = track.getCapabilities?.() as (MediaTrackCapabilities & { zoom?: { min: number; max: number } }) | undefined
      const range: ZoomRange = caps?.zoom && caps.zoom.max > caps.zoom.min ? { min: caps.zoom.min, max: caps.zoom.max, mode: 'hardware', floor: caps.zoom.min } : DIGITAL_ZOOM_RANGE
      const initial = Math.min(range.max, Math.max(range.min, 1))
      if (range.mode === 'hardware') track.applyConstraints({ advanced: [{ zoom: initial } as unknown as MediaTrackConstraintSet] }).catch(() => {})
      updateZoomRange(range, initial)
      const tracker = new LiveFaceTracker()

      const loop = () => {
        if (gen !== liveGenRef.current) return
        rafRef.current = requestAnimationFrame(loop)
        // Only start a new detect+process cycle once the previous one has
        // finished — otherwise slow frames overlap, pile up, and the
        // preview drifts further and further behind real time.
        if (frameInFlightRef.current || video.readyState < 2) return
        const now = performance.now()
        if (now - lastFrameRef.current < LIVE_FRAME_INTERVAL_MS) return
        lastFrameRef.current = now
        frameInFlightRef.current = true
        const mirror = facingModeRef.current === 'user'
        // Hardware zoom already zoomed the sensor output — cropping again
        // on top of it would double-zoom.
        const crop = cropRectFor(video.videoWidth, video.videoHeight, aspectRatioFor(aspectRef.current), cropZoom(zoomRef.current, zoomRangeRef.current))
        const base = drawFrame(video, LIVE_MAX_DIMENSION, crop, mirror)
        tracker
          .detect(video, now, { x: crop.x0, y: crop.y0, w: crop.fw, h: crop.fh })
          .then((raw) => {
            if (gen !== liveGenRef.current) return
            const landmarks = remapLandmarks(raw, crop, mirror)
            baseRef.current = base
            landmarksRef.current = landmarks
            resultRef.current = processFrame(base, landmarks, paramsRef.current, false)
            setStatus(landmarks ? 'ready' : 'no-face')
            render()
          })
          .catch(() => {})
          .finally(() => {
            if (gen === liveGenRef.current) frameInFlightRef.current = false
          })
      }
      loop()
    } catch {
      if (gen === liveGenRef.current) setStatus('error')
    }
  }, [render, stopLive])

  // Turns a still frame into the editable photo: full-quality landmark
  // detection on that exact frame (IMAGE mode, not the live tracker's
  // estimate), then the high-quality retouching pass.
  const loadStill = useCallback(
    async (base: HTMLCanvasElement, liveLandmarks: NormalizedLandmark[] | null = null) => {
      staticBusyRef.current = true
      setProcessing(true)
      baseRef.current = base
      render()
      await nextPaint()
      try {
        // A captured frame keeps the live tracker's landmarks — the face fit
        // the viewfinder showed (they're relative to the same crop) — and
        // only detects afresh if there were none.
        landmarksRef.current = capturedRef.current && liveLandmarks ? liveLandmarks : ((await detectFaceLandmarks(base, liveLandmarks)) ?? liveLandmarks)
      } catch {
        landmarksRef.current = liveLandmarks
      }
      setStatus(landmarksRef.current ? 'ready' : 'no-face')
      await nextPaint()
      recomputeStatic()
      staticBusyRef.current = false
      setProcessing(false)
      setFrameVersion((v) => v + 1)
    },
    [render, recomputeStatic],
  )

  const capture = useCallback(() => {
    const video = videoRef.current
    if (!video || video.readyState < 2) return
    setFlashKey((k) => k + 1)
    navigator.vibrate?.(15)
    const mirror = facingModeRef.current === 'user'
    const crop = cropRectFor(video.videoWidth, video.videoHeight, aspectRatioFor(aspectRef.current), cropZoom(zoomRef.current, zoomRangeRef.current))
    const base = drawFrame(video, MAX_DIMENSION, crop, mirror)
    const liveLandmarks = landmarksRef.current
    staticBusyRef.current = true
    capturedRef.current = true
    setCaptured(true)
    stopLive()
    setLive(false)
    // The last live frame (already retouched at preview quality) stays on
    // screen while the full-quality pass runs, so there's no flash of the
    // untouched photo in between.
    loadStill(base, liveLandmarks)
  }, [stopLive, loadStill])

  // Uploaded photo.
  useEffect(() => {
    if (source.kind !== 'image') return
    let cancelled = false
    staticBusyRef.current = true
    capturedRef.current = false
    setCaptured(false)
    setLive(false)
    setStatus('loading')
    setStickers([])
    setSelectedSticker(null)
    resultRef.current = null
    ;(async () => {
      try {
        const img = await loadImage(source.file)
        if (cancelled) return
        await loadStill(drawFrame(img, MAX_DIMENSION, FULL_FRAME, false))
      } catch {
        if (!cancelled) {
          staticBusyRef.current = false
          setStatus('error')
        }
      }
    })()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  // Live camera.
  useEffect(() => {
    if (source.kind !== 'live') return
    startLive()
    return () => {
      stopLive()
      cancelCountdown()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  // Re-run the full-quality pass on any control change while editing a
  // still (the live loop picks up paramsRef on its own every frame).
  useEffect(() => {
    if (live || staticBusyRef.current || !baseRef.current) return
    const t = setTimeout(recomputeStatic, 60)
    return () => clearTimeout(t)
  }, [params, live, recomputeStatic])

  // Filter thumbnails are rendered from the current frame whenever the
  // strip opens (and again for each new still), not every live frame.
  const frameReady = status !== 'loading'
  useEffect(() => {
    if (panel !== 'filter') return
    const base = baseRef.current
    if (!base) return
    const t = setTimeout(() => setThumbs(renderFilterThumbnails(base, faceFocus(landmarksRef.current, base.width, base.height))), 30)
    return () => clearTimeout(t)
  }, [panel, frameVersion, frameReady])

  // The 3D effects engine loads on first use; once it's in, redo the
  // still so the chosen effect appears (the live loop picks it up itself).
  useEffect(() => {
    if (params.effectId === 'none') return
    let cancelled = false
    preloadAR().then(() => {
      if (!cancelled && !live && !staticBusyRef.current) recomputeStatic()
    })
    return () => {
      cancelled = true
    }
  }, [params.effectId, live, recomputeStatic])

  // Looks and Effects preview on this very face, rendered when their panel
  // opens (and again for each new still), progressively, not per live frame.
  useEffect(() => {
    if (panel !== 'looks' && panel !== 'effects') return
    const base = baseRef.current
    if (!base || status === 'loading') return
    let cancelled = false
    const t = setTimeout(() => {
      if (panel === 'looks') {
        const items = LOOKS.map((l) => ({ id: l.id, params: applyLook(l, 1, { ...DEFAULT_PARAMS, effectId: 'none' }) }))
        renderFaceThumbs(base, landmarksRef.current, items, 1, (id, url) => setLookThumbs((m) => ({ ...m, [id]: url })), () => cancelled)
      } else {
        const items = EFFECTS.filter((e) => e.id !== 'none' && e.id !== 'custom').map((e) => ({ id: e.id, params: { ...paramsRef.current, effectId: e.id } }))
        preloadAR().then(() => {
          if (!cancelled) renderFaceThumbs(base, landmarksRef.current, items, 1.75, (id, url) => setEffectThumbs((m) => ({ ...m, [id]: url })), () => cancelled)
        })
      }
    }, 40)
    return () => {
      cancelled = true
      clearTimeout(t)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panel, frameVersion, frameReady])

  const selectLook = (id: string | null, strength = lookStrength) => {
    setLookId(id)
    const look = findLook(id)
    setParams((p) => (look ? applyLook(look, strength, p) : { ...DEFAULT_PARAMS, effectId: p.effectId }))
  }

  const addSticker = async (req: StickerRequest) => {
    const base = baseRef.current
    if (!base) return
    let img: CanvasImageSource
    let ratio: number
    let w = 0.3
    try {
      if (req.kind === 'art') {
        const el = await artImage(req.svg)
        img = el
        ratio = el.naturalHeight / el.naturalWidth || 1
      } else if (req.kind === 'emoji') {
        img = emojiCanvas(req.char)
        ratio = 1
        w = 0.24
      } else if (req.kind === 'text') {
        const c = textCanvas(req.text, req.style)
        img = c
        ratio = c.height / c.width
        w = Math.min(0.75, 0.16 * Math.max(2.5, req.text.length * 0.55))
      } else {
        const c = await photoSticker(req.file)
        img = c
        ratio = c.height / c.width
        w = 0.42
      }
    } catch {
      showToast("Couldn't add that sticker")
      return
    }
    const st = placeSticker(img, ratio, w, base.width / base.height)
    // Stagger new stickers so they don't land exactly on top of each other.
    const k = stickers.length % 5
    st.x += (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 0.07
    st.y += Math.ceil(k / 2) * 0.05
    setStickers((list) => [...list, st])
    setSelectedSticker(st.id)
  }
  const updateSticker = (id: number, patch: Partial<Sticker>) => setStickers((list) => list.map((s) => (s.id === id ? { ...s, ...patch } : s)))
  const selectSticker = (id: number | null) => {
    setSelectedSticker(id)
    // The touched sticker comes to the front.
    if (id !== null) setStickers((list) => (list[list.length - 1]?.id === id ? list : [...list.filter((s) => s.id !== id), list.find((s) => s.id === id)!]))
  }

  // ---- Video ----
  const saveVideo = async (blob: Blob, ext: string) => {
    const name = timestampedName().replace(/\.jpg$/, `.${ext}`)
    const file = new File([blob], name, { type: blob.type })
    if (navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({ files: [file] })
        return
      } catch {
        // dismissed: fall through to a download
      }
    }
    downloadBlob(blob, name)
    showToast('Video saved')
  }

  const stopRecording = useCallback(() => {
    const r = recRef.current
    if (r && r.rec.state !== 'inactive') r.rec.stop()
  }, [])

  const startRecording = async () => {
    const canvas = displayRef.current as (HTMLCanvasElement & { captureStream?: (fps?: number) => MediaStream }) | null
    if (!canvas?.captureStream || typeof MediaRecorder === 'undefined') {
      showToast("Video recording isn't supported in this browser")
      return
    }
    setPanel(null)
    const stream = canvas.captureStream(30)
    // Sound too, if the microphone is allowed; a silent video otherwise.
    let audio: MediaStream | null = null
    try {
      audio = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })
      audio.getAudioTracks().forEach((t) => stream.addTrack(t))
    } catch {
      audio = null
    }
    if (!pressHandledRef.current) {
      // Let go while the microphone was being set up: nothing to record.
      audio?.getTracks().forEach((t) => t.stop())
      return
    }
    const type = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'].find((t) => MediaRecorder.isTypeSupported(t))
    let rec: MediaRecorder
    try {
      rec = new MediaRecorder(stream, type ? { mimeType: type, videoBitsPerSecond: 6_000_000 } : undefined)
    } catch {
      audio?.getTracks().forEach((t) => t.stop())
      showToast("Video recording isn't supported in this browser")
      return
    }
    const chunks: Blob[] = []
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
    rec.onstop = () => {
      recRef.current?.audio?.getTracks().forEach((t) => t.stop())
      recRef.current = null
      setRecordingSince(null)
      const mime = rec.mimeType || type || 'video/webm'
      if (chunks.length) saveVideo(new Blob(chunks, { type: mime }), mime.includes('mp4') ? 'mp4' : 'webm')
    }
    recRef.current = { rec, chunks, audio }
    rec.start(250)
    navigator.vibrate?.(25)
    setRecordingSince(Date.now())
  }

  // Recording timer (and a one-minute cap).
  useEffect(() => {
    if (recordingSince === null) return
    const id = setInterval(() => {
      setRecordTick((n) => n + 1)
      if (Date.now() - recordingSince > 60_000) stopRecording()
    }, 250)
    return () => clearInterval(id)
  }, [recordingSince, stopRecording])

  // Leaving the camera ends a recording in progress.
  useEffect(() => {
    if (!live) stopRecording()
  }, [live, stopRecording])

  // Shutter: tap = photo, press and hold = video while held.
  const onShutterDown = () => {
    if (countdown !== null || status === 'loading' || status === 'error') return
    pressHandledRef.current = false
    pressTimerRef.current = window.setTimeout(() => {
      pressTimerRef.current = null
      pressHandledRef.current = true
      startRecording()
    }, 400)
  }
  const onShutterUp = () => {
    if (pressTimerRef.current !== null) {
      clearTimeout(pressTimerRef.current)
      pressTimerRef.current = null
      pressHandledRef.current = true
      handleShutter()
      return
    }
    if (pressHandledRef.current) {
      pressHandledRef.current = false
      stopRecording()
    }
  }

  const handleShutter = () => {
    if (countdown !== null) {
      cancelCountdown()
      return
    }
    setPanel(null)
    if (timer === 0) {
      capture()
      return
    }
    let remaining: number = timer
    setCountdown(remaining)
    const tick = () => {
      remaining -= 1
      if (remaining <= 0) {
        countdownTimerRef.current = null
        setCountdown(null)
        capture()
      } else {
        setCountdown(remaining)
        countdownTimerRef.current = window.setTimeout(tick, 1000)
      }
    }
    countdownTimerRef.current = window.setTimeout(tick, 1000)
  }

  // ✕ in review: discard and go back to the camera.
  const handleClose = () => {
    capturedRef.current = false
    setCaptured(false)
    setPanel(null)
    setStickers([])
    setSelectedSticker(null)
    if (source.kind === 'live') startLive()
    else onReset()
  }

  const handleFlip = () => {
    cancelCountdown()
    const next = facingMode === 'user' ? 'environment' : 'user'
    facingModeRef.current = next
    setFacingMode(next)
    startLive()
  }

  const applyZoom = useCallback((value: number) => {
    const { min, max, mode, floor } = zoomRangeRef.current
    const clamped = Math.min(max, Math.max(min, value))
    const hw = Math.max(floor, clamped)
    const prevHw = Math.max(floor, zoomRef.current)
    zoomRef.current = clamped
    setZoom(clamped)
    if (mode === 'hardware' && zoomTrackRef.current && hw !== prevHw) {
      zoomTrackRef.current.applyConstraints({ advanced: [{ zoom: hw } as unknown as MediaTrackConstraintSet] }).catch(() => {})
    }
  }, [])

  // The zoom range for the current camera and frame shape: zooming out
  // below 1x reaches as far as the camera's whole field of view.
  function updateZoomRange(base: ZoomRange, value: number) {
    const video = videoRef.current
    const wide = video?.videoWidth ? wideZoom(cropRectFor(video.videoWidth, video.videoHeight, aspectRatioFor(aspectRef.current), 1)) : 1
    const range = { ...base, min: Math.min(base.floor, Math.ceil(base.floor * wide * 100) / 100) }
    const clamped = Math.min(range.max, Math.max(range.min, value))
    zoomRangeRef.current = range
    zoomRef.current = clamped
    setZoomRange(range)
    setZoom(clamped)
  }

  const cycleAspect = () => {
    const next = ASPECT_MODES[(ASPECT_MODES.indexOf(aspect) + 1) % ASPECT_MODES.length]
    aspectRef.current = next
    setAspect(next)
    updateZoomRange(zoomRangeRef.current, zoomRef.current)
  }

  const exportBlob = (): Promise<Blob | null> => {
    const src = resultRef.current ?? baseRef.current
    if (!src) return Promise.resolve(null)
    let canvas = src
    if (stickers.length) {
      canvas = document.createElement('canvas')
      canvas.width = src.width
      canvas.height = src.height
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(src, 0, 0)
      drawStickers(ctx, stickers, canvas.width, canvas.height)
    }
    return new Promise((resolve) => canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.95))
  }

  // iOS has no "download a file to Photos" — the share sheet's Save Image
  // is the only way a web app's photo reaches the camera roll, so Save goes
  // through it there and through a plain download everywhere else.
  const handleSave = async () => {
    const blob = await exportBlob()
    if (!blob) return
    const name = timestampedName()
    const file = new File([blob], name, { type: 'image/jpeg' })
    if (isIOS && navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({ files: [file] })
      } catch {
        // dismissed
      }
      return
    }
    downloadBlob(blob, name)
    showToast('Saved')
  }

  const handleShare = async () => {
    const blob = await exportBlob()
    if (!blob) return
    const name = timestampedName()
    const file = new File([blob], name, { type: 'image/jpeg' })
    if (navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({ files: [file], title: 'Flux Glow' })
      } catch {
        // dismissed
      }
      return
    }
    downloadBlob(blob, name)
    showToast('Sharing unavailable — saved instead')
  }

  const item = (key: NumericParam, label: string, icon: ReactNode, bidirectional = false): AdjustItem => ({
    key,
    label,
    icon,
    value: params[key],
    defaultValue: DEFAULT_PARAMS[key],
    onChange: set(key),
    bidirectional,
  })

  const beautyItems: AdjustItem[] = [
    item('smoothness', 'Smooth', <IconDroplet className="w-5 h-5" />),
    item('whitening', 'Whiten', <IconWhiten className="w-5 h-5" />),
    item('acneRemoval', 'Blemish', <IconTarget className="w-5 h-5" />),
    item('wrinkleRemoval', 'Wrinkle', <IconWave className="w-5 h-5" />),
    item('mouthCornerSmooth', 'Folds', <RegionIcon dot={[9, 15.6]} pair className="w-5 h-5" />),
    item('fillLight', 'Light', <IconSun className="w-5 h-5" />),
  ]
  const R = (dot: [number, number], pair = false) => <RegionIcon dot={dot} pair={pair} className="w-5 h-5" />
  const group = (key: string, label: string, children: AdjustItem[]): AdjustItem => ({ key, label, icon: null, children })
  const shapeItems: AdjustItem[] = [
    group('faceGroup', 'Face', [
      item('face', 'Slim', R([7.2, 13.8], true), true),
      item('vJaw', 'V Jaw', R([8.2, 15.6], true), true),
      item('chin', 'Chin', R([12, 17.2]), true),
      item('forehead', 'Forehead', R([12, 5.2]), true),
      item('temple', 'Temple', R([7.2, 7.6], true), true),
      item('cheekbone', 'Cheekbone', R([6.8, 11.5], true), true),
    ]),
    group('eyeGroup', 'Eyes', [
      item('eyes', 'Size', R([9, 10.2], true), true),
      item('eyeWidth', 'Width', R([8.4, 10.2], true), true),
      item('eyeHeight', 'Height', R([9, 9.8], true), true),
      item('eyeTilt', 'Tilt', R([7.8, 9.7], true), true),
      item('eyeDistance', 'Spacing', R([7.6, 10.2], true), true),
      item('eyePosition', 'Position', R([9, 9.4], true), true),
    ]),
    group('browGroup', 'Brows', [
      item('eyebrowHeight', 'Height', R([9, 8], true), true),
      item('browTilt', 'Tilt', R([7.4, 7.6], true), true),
      item('browDistance', 'Spacing', R([10.4, 8], true), true),
    ]),
    group('noseGroup', 'Nose', [
      item('nose', 'Size', R([12, 12.5]), true),
      item('noseWings', 'Wings', R([10.8, 12.8], true), true),
      item('noseTip', 'Tip', R([12, 13]), true),
      item('noseLength', 'Length', R([12, 11.6]), true),
      item('noseBridge', 'Bridge', R([12, 9.3]), true),
    ]),
    group('mouthGroup', 'Mouth', [
      item('mouth', 'Size', R([12, 14.3]), true),
      item('mouthWidth', 'Width', R([10, 14.3], true), true),
      item('mouthUpperLip', 'Upper Lip', R([12, 13.6]), true),
      item('mouthLowerLip', 'Lower Lip', R([12, 15.1]), true),
      item('mouthCorners', 'Smile', R([9.6, 14], true), true),
      item('mouthPosition', 'Position', R([12, 13.2]), true),
    ]),
  ]
  const changedFrom = (keys: NumericParam[]) => keys.some((k) => Math.abs(params[k] - DEFAULT_PARAMS[k]) > 0.005)

  const togglePanel = (p: Panel) => setPanel((v) => (v === p ? null : p))
  const noFace = status === 'no-face'
  const fullBleed = (live || captured) && aspect === 'full'
  const topInset = 'max(0.75rem, env(safe-area-inset-top))'
  // The viewfinder sits below the top controls and clears the shutter that
  // straddles the tray's top edge — except in Full, which fills the screen.
  // (Measured with no panel open: opening one overlays the picture rather
  // than resizing it.)
  const trayH = useRef(0)
  if (!panel && trayBox.height) trayH.current = trayBox.height
  const previewStyle = fullBleed ? { top: 0, bottom: 0 } : { top: `calc(${topInset} + 3.25rem)`, bottom: (trayH.current || trayBox.height) + 44 }

  return (
    <div className="fixed inset-0 bg-black overflow-hidden select-none">
      <video ref={videoRef} autoPlay playsInline muted className="hidden" />

      <div ref={previewRef} className={`absolute inset-x-0 flex justify-center ${(live || captured) && !fullBleed ? 'items-start' : 'items-center'}`} style={previewStyle}>
        <div className="relative" style={{ width: fit.width, height: fit.height }}>
          <canvas ref={displayRef} className="block w-full h-full" />
          {grid && live && <GridOverlay />}
          {!live && !showBefore && status !== 'loading' && (stickers.length > 0 || panel === 'stickers') && (
            <div data-keep-panel className="absolute inset-0">
              <StickerLayer
                stickers={stickers}
                selectedId={selectedSticker}
                width={fit.width}
                height={fit.height}
                onSelect={selectSticker}
                onUpdate={updateSticker}
                onDelete={(id) => {
                  setStickers((list) => list.filter((x) => x.id !== id))
                  setSelectedSticker(null)
                }}
              />
            </div>
          )}
          {countdown !== null && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <span key={countdown} className="fg-count w-36 h-36 rounded-full bg-black/35 backdrop-blur-sm flex items-center justify-center text-8xl font-light text-white tabular-nums">
                {countdown}
              </span>
            </div>
          )}
          {status === 'loading' && (
            <div className="absolute inset-0 flex items-center justify-center">
              <IconSpinner className="w-9 h-9 text-primary animate-spin" />
            </div>
          )}
        </div>
      </div>

      {flashKey > 0 && <div key={flashKey} className="fg-flash absolute inset-0 bg-white pointer-events-none z-40 opacity-0" />}

      <div className="absolute top-0 inset-x-0 h-24 bg-gradient-to-b from-black/60 to-transparent pointer-events-none" />

      <div className="absolute inset-x-0 flex items-center justify-between px-3 z-10" style={{ top: topInset }}>
        <div className="flex items-center gap-2">
          {live ? (
            <>
              <TopButton label={`Self-timer: ${timer ? `${timer} seconds` : 'off'}`} active={timer > 0} onClick={() => setTimer((t) => TIMER_STEPS[(TIMER_STEPS.indexOf(t) + 1) % TIMER_STEPS.length])}>
                <IconTimer className="w-[18px] h-[18px]" />
                {timer > 0 && <span>{timer}s</span>}
              </TopButton>
              <TopButton label={`Aspect ratio ${aspect}`} onClick={cycleAspect}>
                <span className="px-0.5">{aspect === 'full' ? 'Full' : aspect}</span>
              </TopButton>
              <TopButton label="Grid" active={grid} onClick={() => setGrid((g) => !g)}>
                <IconGrid className="w-[18px] h-[18px]" />
              </TopButton>
            </>
          ) : (
            <TopButton label="Discard and return to camera" onClick={handleClose}>
              <IconClose className="w-5 h-5" />
            </TopButton>
          )}
        </div>
        <div className="flex items-center gap-2">
          <TopButton
            label="Hold to compare with the original"
            active={showBefore}
            disabled={status === 'loading'}
            onPointerDown={() => setShowBefore(true)}
            onPointerUp={() => setShowBefore(false)}
            onPointerLeave={() => setShowBefore(false)}
            onPointerCancel={() => setShowBefore(false)}
            onContextMenu={(e) => e.preventDefault()}
            style={{ touchAction: 'none' }}
          >
            <IconCompare className="w-5 h-5" />
          </TopButton>
          {live ? (
            <TopButton label="Flip camera" onClick={handleFlip} disabled={status === 'loading'}>
              <IconFlipCamera className="w-5 h-5" />
            </TopButton>
          ) : (
            <TopButton label="Share" onClick={handleShare} disabled={status === 'loading' || status === 'error' || processing}>
              <IconShare className="w-5 h-5" />
            </TopButton>
          )}
        </div>
      </div>

      {recordingSince !== null && (
        <div data-tick={recordTick} className="absolute left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 px-3 py-1 rounded-full bg-black/60 backdrop-blur-md text-[12px] font-semibold text-white tabular-nums" style={{ top: `calc(${topInset} + 3.25rem)` }}>
          <span className="w-2 h-2 rounded-full bg-danger animate-pulse" />
          {(() => {
            const sec = Math.floor((Date.now() - recordingSince) / 1000)
            return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`
          })()}
        </div>
      )}
      {(showBefore || processing) && (
        <div className="absolute left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 px-3 py-1 rounded-full bg-black/60 backdrop-blur-md text-[11px] font-semibold text-white" style={{ top: `calc(${topInset} + 3.25rem)` }}>
          {processing && !showBefore && <IconSpinner className="w-3 h-3 animate-spin text-primary" />}
          {showBefore ? 'Original' : 'Retouching…'}
        </div>
      )}

      {toast && (
        <div key={toast.key} className="fg-toast absolute left-1/2 z-30 flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-black/75 backdrop-blur-md text-xs font-medium text-white" style={{ top: '42%' }}>
          <IconCheck className="w-3.5 h-3.5 text-primary" />
          {toast.text}
        </div>
      )}

      <div className="absolute inset-x-0 bottom-0 flex flex-col items-stretch">
        {(noFace || status === 'error') && !panel && (
          <p className={`self-center mb-3 text-xs flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md ${status === 'error' ? 'text-danger' : 'text-white/85'}`}>
            {status === 'error' ? <IconError className="w-3.5 h-3.5 flex-shrink-0" /> : <IconInfo className="w-3.5 h-3.5 flex-shrink-0" />}
            {status === 'error'
              ? source.kind === 'live'
                ? 'Camera unavailable — allow camera access in your browser settings.'
                : "Couldn't open that photo."
              : 'No face found — Beauty and Shape need one. Filters still apply.'}
          </p>
        )}
        {/* No camera (denied, absent, or a desktop without one): offer a
            ready portrait so every tool can still be tried right away. */}
        {status === 'error' && source.kind === 'live' && !panel && onTrySample && (
          <button
            onClick={onTrySample}
            className="self-center mb-14 px-5 py-2.5 rounded-full bg-primary text-black text-sm font-semibold shadow-glow transition active:scale-95"
          >
            Try a sample photo
          </button>
        )}

        {live && !panel && status !== 'loading' && status !== 'error' && countdown === null && (
          <div className="self-center mb-12">
            <ZoomControl range={zoomRange} value={zoom} onChange={applyZoom} />
          </div>
        )}

        {panel && (
          <div ref={panelRef} key={panel} className="fg-panel rounded-t-[22px] bg-black/45 backdrop-blur-xl border-t border-white/10 px-3.5 pt-3 pb-1">
            {panel === 'looks' && (
              <div>
                <div className="flex items-center justify-between h-7 mb-3">
                  <span className="text-[13px] font-semibold text-white">Looks</span>
                  <button onClick={() => selectLook(null)} disabled={!lookId} className="flex items-center gap-1 text-[11px] font-medium text-white/70 hover:text-white disabled:opacity-30 transition">
                    <IconRefresh className="w-3.5 h-3.5" />
                    Reset
                  </button>
                </div>
                {noFace ? (
                  <p className="text-xs text-white/60 py-4 text-center">Looks need a face in the frame.</p>
                ) : (
                  <ThumbStrip items={LOOKS} thumbs={lookThumbs} selected={lookId} onSelect={(id) => selectLook(id)} />
                )}
                {findLook(lookId) && !noFace && (
                  <div className="mt-3">
                    <p className="text-[11px] text-white/55 mb-2">{findLook(lookId)!.hint} · fine-tune any part in Beauty and Shape.</p>
                    <Slider
                      label="Intensity"
                      value={lookStrength}
                      defaultValue={1}
                      onChange={(v) => {
                        setLookStrength(v)
                        selectLook(lookId, v)
                      }}
                    />
                  </div>
                )}
              </div>
            )}
            {panel === 'beauty' && <AdjustPanel title="Beauty" items={beautyItems} disabled={noFace} />}
            {panel === 'shape' && <AdjustPanel title="Shape" items={shapeItems} disabled={noFace} tabs />}
            {panel === 'effects' && (
              <div>
                <div className="flex items-center justify-between h-7 mb-3">
                  <span className="text-[13px] font-semibold text-white">Effects</span>
                  <span className="text-[10.5px] text-white/45">{live ? 'Tracks your face live' : 'Placed on your face'}</span>
                </div>
                {noFace ? (
                  <p className="text-xs text-white/60 py-4 text-center">Effects need a face in the frame.</p>
                ) : (
                  <ThumbStrip
                    items={EFFECTS.map((e) => ({ id: e.id, label: e.label, badge: e.boost ? '♥' : undefined }))}
                    thumbs={effectThumbs}
                    selected={params.effectId}
                    onSelect={(id) => {
                      // Custom: pick a photo first (or a new one, tapping it again).
                      if (id === 'custom' && (!customUrl || params.effectId === 'custom')) customInputRef.current?.click()
                      else setParams((p) => ({ ...p, effectId: id }))
                    }}
                    fallback={(id) =>
                      id === 'none' ? (
                        <IconClose className="w-6 h-6 text-white/60" />
                      ) : id === 'custom' ? (
                        customUrl ? <img src={customUrl} alt="" className="w-full h-full object-cover" draggable={false} /> : <IconPlus className="w-6 h-6 text-white/60" />
                      ) : null
                    }
                  />
                )}
                {params.effectId === 'custom' && !noFace && <p className="text-[11px] text-white/55 mt-2.5">Tap Custom again to use a different picture.</p>}
                {findEffect(params.effectId).boost && !noFace && <p className="text-[11px] text-white/55 mt-2.5">Comes with a baby-face touch-up on top of your own Beauty and Shape settings.</p>}
              </div>
            )}
            {panel === 'stickers' && <StickerPanel count={stickers.length} onAdd={addSticker} onClearAll={() => { setStickers([]); setSelectedSticker(null) }} />}
            {panel === 'filter' && (
              <FilterPanel
                thumbs={thumbs}
                filterId={params.filterId}
                strength={params.filterStrength}
                defaultStrength={DEFAULT_PARAMS.filterStrength}
                onSelect={(id) => setParams((p) => ({ ...p, filterId: id }))}
                onStrength={set('filterStrength')}
              />
            )}
          </div>
        )}

        {/* Frosted tray with a native-camera-style shutter straddling its
            top edge, centered on its own rather than as part of the icon
            row (sharing a centered group would pull it off-center). With a
            panel open it shrinks to a slim bar right under the panel —
            icons only and a smaller shutter — so the picture stays in view. */}
        {(() => {
          const shutter = (small: boolean) =>
            live ? (
              <button
                onPointerDown={onShutterDown}
                onPointerUp={onShutterUp}
                onPointerCancel={onShutterUp}
                onPointerLeave={() => recordingSince === null && pressTimerRef.current !== null && onShutterUp()}
                onContextMenu={(e) => e.preventDefault()}
                // Keyboard: Enter / Space take a photo.
                onClick={(e) => e.detail === 0 && handleShutter()}
                disabled={status === 'loading' || status === 'error'}
                aria-label={recordingSince !== null ? 'Recording — release to stop' : countdown !== null ? 'Cancel timer' : timer ? `Take photo in ${timer} seconds` : 'Take photo (hold to record video)'}
                style={{ touchAction: 'none' }}
                className={`${small ? 'w-[50px] h-[50px] border-[3px]' : 'w-[76px] h-[76px] border-[4px]'} rounded-full ${recordingSince !== null ? 'border-danger scale-110' : 'border-white/95'} bg-black/20 flex items-center justify-center shadow-glow-strong transition active:scale-95 disabled:opacity-50`}
              >
                <span className={`rounded-full transition-all ${countdown !== null || recordingSince !== null ? 'w-6 h-6 rounded-md bg-danger' : small ? 'w-[38px] h-[38px] bg-primary' : 'w-[60px] h-[60px] bg-primary'}`} />
              </button>
            ) : (
              <button
                onClick={handleSave}
                disabled={status === 'loading' || status === 'error' || processing}
                aria-label="Save photo"
                className={`${small ? 'w-[50px] h-[50px] border-[3px]' : 'w-[76px] h-[76px] border-[4px]'} rounded-full bg-primary border-white/20 shadow-glow-strong flex items-center justify-center text-black transition active:scale-95 disabled:opacity-50`}
              >
                <IconDownload className={small ? 'w-5 h-5' : 'w-7 h-7'} />
              </button>
            )
          const c = !!panel
          const left = (
            <>
              <TrayButton compact={c} icon={IconWand} label="Looks" toggle active={panel === 'looks'} dot={lookId !== null} onClick={() => togglePanel('looks')} />
              <TrayButton compact={c} icon={IconSparkle} label="Beauty" toggle active={panel === 'beauty'} dot={changedFrom(BEAUTY_KEYS)} onClick={() => togglePanel('beauty')} />
              <TrayButton compact={c} icon={IconFaceOutline} label="Shape" toggle active={panel === 'shape'} dot={changedFrom(SHAPE_KEYS)} onClick={() => togglePanel('shape')} />
            </>
          )
          const right = (
            <>
              <TrayButton compact={c} icon={IconPalette} label="Filter" toggle active={panel === 'filter'} dot={params.filterId !== 'none'} onClick={() => togglePanel('filter')} />
              <TrayButton compact={c} icon={IconEars} label="Effects" toggle active={panel === 'effects'} dot={params.effectId !== 'none'} onClick={() => togglePanel('effects')} />
              {live ? (
                <TrayButton compact={c} icon={IconImage} label="Album" onClick={() => fileInputRef.current?.click()} />
              ) : (
                <TrayButton compact={c} icon={IconSticker} label="Stickers" toggle active={panel === 'stickers'} dot={stickers.length > 0} onClick={() => togglePanel('stickers')} />
              )}
            </>
          )
          return c ? (
            <div ref={trayRef} className="bg-black/45 backdrop-blur-xl px-3 pt-1.5" style={{ paddingBottom: 'max(0.6rem, env(safe-area-inset-bottom))' }}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">{left}</div>
                {shutter(true)}
                <div className="flex items-center gap-1.5">{right}</div>
              </div>
            </div>
          ) : (
            <div ref={trayRef} className="relative bg-black/55 backdrop-blur-2xl border-t border-white/10 rounded-t-[28px] px-3 pt-12" style={{ paddingBottom: 'max(1.1rem, env(safe-area-inset-bottom))' }}>
              <div className="absolute left-1/2 -translate-x-1/2 -top-9">{shutter(false)}</div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1">{left}</div>
                <div className="flex items-center gap-1">{right}</div>
              </div>
            </div>
          )
        })()}
      </div>
      <input
        ref={customInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) setCropFile(file)
          e.target.value = ''
        }}
      />
      {cropFile && (
        <CropDialog
          file={cropFile}
          onCancel={() => setCropFile(null)}
          onDone={(square) => {
            setCustomImage(square)
            setCustomUrl(square.toDataURL('image/jpeg', 0.85))
            setCropFile(null)
            setParams((p) => ({ ...p, effectId: 'custom' }))
          }}
        />
      )}
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
