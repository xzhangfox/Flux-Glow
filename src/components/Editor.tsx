import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ButtonHTMLAttributes, type ComponentType, type ReactNode, type RefObject, type SVGProps } from 'react'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { detectFaceLandmarks, LiveFaceTracker } from '../lib/faceLandmarker'
import { processFrame, processFrameInSteps, DEFAULT_PARAMS, faceFocus, type EditParams, type NumericParam } from '../lib/pipeline'
import { renderFilterThumbnails } from '../lib/filters'
import { ASPECT_MODES, aspectRatioFor, cropRectFor, drawFrame, remapLandmarks, type AspectMode } from '../lib/frame'
import { STILL_MAX, cameraConstraints, previewCopy, settlePhotoMode, switchCamera } from '../lib/camera'
import { nextTask } from '../lib/schedule'
import BgProtectToggle from './BgProtectToggle'
import { loadBgProtect, saveBgProtect } from '../lib/bgProtect'
import SettingsSheet from './SettingsSheet'
import AdjustPanel from './AdjustPanel'
import { BEAUTY_KEYS, SHAPE_KEYS, beautyItems, changedFrom, shapeItems } from './adjustItems'
import FilterPanel from './FilterPanel'
import ThumbStrip from './ThumbStrip'
import Slider from './Slider'
import StickerPanel, { type StickerRequest } from './StickerPanel'
import StickerLayer from './StickerLayer'
import { LOOKS, applyLook, findLook } from '../lib/looks'
import { EFFECTS, EFFECT_THUMBS, findEffect, isEffectReady, onEffectReady, prepareEffect, setCustomImage } from '../lib/effects'
import CropDialog from './CropDialog'
import VideoReview from './VideoReview'
import type { RecordView } from '../lib/video/render'
import { downloadBlob, isIOS, isVideoFile, timestampedName } from '../lib/save'
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
  IconImage,
  IconSparkle,
  IconTimer,
  IconGrid,
  IconWand,
  IconEars,
  IconSticker,
  IconRefresh,
  IconPlus,
  IconSettings,
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
// How long the zoom ruler stays after a pinch ends.
const PINCH_RULER_LINGER_MS = 1000


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
// Rendering a recording at full quality needs WebCodecs (decode + encode).
const canRenderVideo = () => typeof VideoEncoder !== 'undefined' && typeof VideoDecoder !== 'undefined'

const nextPaint = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)))

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

export default function Editor({
  source,
  onReset,
  onPickImage,
  onEditVideo,
  onTrySample,
}: {
  source: Source
  onReset: () => void
  onPickImage?: (file: File) => void
  /** Opens a video in the video editor, starting from `params`. */
  onEditVideo?: (file: Blob, params?: EditParams) => void
  onTrySample?: () => void
}) {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [params, setParams] = useState<EditParams>(() => ({ ...DEFAULT_PARAMS, protectBackground: loadBgProtect() }))
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
  // Two recordings at once: what the viewfinder shows (effects and all) to
  // keep as is, and the same frames before any retouch, so the video editor
  // can redo the retouch person by person.
  const recRef = useRef<{ rec: MediaRecorder; audio: MediaStream | null; raw: MediaRecorder | null } | null>(null)
  // `raw`: the camera's own stream, recorded as it came, with how the
  // preview framed it — rendered to full quality afterwards.
  const [review, setReview] = useState<{ video: Blob; raw: Blob | null; view: RecordView; params: EditParams } | null>(null)
  const pressTimerRef = useRef<number | null>(null)
  const pressHandledRef = useRef(false)
  const [processing, setProcessing] = useState(false)
  const [panel, setPanel] = useState<Panel | null>(null)
  const [showBefore, setShowBefore] = useState(false)
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user')
  const [zoom, setZoom] = useState(1)
  const [zoomRange, setZoomRange] = useState<ZoomRange>(DIGITAL_ZOOM_RANGE)
  // A two-finger pinch on the picture is zooming (the zoom ruler shows
  // while it does, and for a moment after).
  const [pinching, setPinching] = useState(false)
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
  // The chosen effect is still loading (in the background).
  const [effectLoading, setEffectLoading] = useState(false)
  const [stickers, setStickers] = useState<Sticker[]>([])
  const [selectedSticker, setSelectedSticker] = useState<number | null>(null)

  const displayRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  // A photo-sized camera at full size, for the shutter, while the preview
  // (videoRef) plays a smaller copy of it (see previewCopy); unused (no
  // source) when the preview plays the camera itself.
  const stillVideoRef = useRef<HTMLVideoElement>(null)
  const previewTrackRef = useRef<MediaStreamTrack | null>(null)
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
  // A photo taken with the camera, at the camera's full resolution (the
  // editor works on a copy at MAX_DIMENSION; Save and Share render this
  // with the same edit), and that render for the edit it was made with.
  const fullBaseRef = useRef<HTMLCanvasElement | null>(null)
  const fullResultRef = useRef<{ params: EditParams; canvas: HTMLCanvasElement } | null>(null)
  const rafRef = useRef(0)
  const lastFrameRef = useRef(0)
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
      // (decided now, done once the tap has been handled: closing it at
      // once swapped the slim bar under the panel for the full tray before
      // the tapped button — the shutter, Save, Album — got its click, so
      // the tap did nothing but close the panel)
      setTimeout(() => setPanel(null), 0)
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

  // `quick`: the fast retouch (the live one) even for a still that gets the
  // full-quality pass — while a control is moving.
  const recomputeStatic = useCallback(
    (quick = false) => {
      const base = baseRef.current
      if (!base) return
      resultRef.current = processFrame(base, landmarksRef.current, paramsRef.current, !capturedRef.current && !quick)
      render()
    },
    [render],
  )

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
    previewTrackRef.current?.stop()
    previewTrackRef.current = null
    if (stillVideoRef.current) stillVideoRef.current.srcObject = null
    zoomTrackRef.current = null
  }, [])

  const startLive = useCallback(async () => {
    stopLive()
    fullBaseRef.current = null
    fullResultRef.current = null
    const gen = liveGenRef.current
    setLive(true)
    setStatus('loading')
    try {
      // Photo mode: the camera's largest 4:3 size, so the shutter keeps the
      // frame on screen at full resolution (see camera.ts); recording
      // switches it to 1080p60.
      const stream = await navigator.mediaDevices.getUserMedia({ video: cameraConstraints('photo', facingModeRef.current) })
      if (gen !== liveGenRef.current) {
        stream.getTracks().forEach((t) => t.stop())
        return
      }
      streamRef.current = stream
      const still = stillVideoRef.current!
      still.srcObject = stream
      await still.play()
      if (gen !== liveGenRef.current) return
      await settlePhotoMode(stream.getVideoTracks()[0], still, null)
      if (gen !== liveGenRef.current) return
      // The preview plays a 1440p copy of a photo-sized camera — a fraction
      // of the pixels to handle every frame — or the camera itself.
      const copy = await previewCopy(stream.getVideoTracks()[0], still, null)
      if (gen !== liveGenRef.current) {
        copy?.stop()
        return
      }
      previewTrackRef.current = copy
      const video = videoRef.current!
      video.srcObject = copy ? new MediaStream([copy]) : stream
      if (!copy) still.srcObject = null
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
      await tracker.ready()
      if (gen !== liveGenRef.current) return

      // Two stages, side by side: the tracker finds the face in the newest
      // frame (in a worker of its own, where it has one) while the page
      // retouches the frame before it — so the preview keeps up with the
      // slower of the two, not both added together. A tracked frame waits
      // for the retouch only while the one before is still in it, and no
      // further frame is tracked meanwhile (nothing tracked to be dropped).
      // Both run in tasks of their own, outside the frame callback, broken
      // up between stages, so taps and slider moves are handled — and
      // painted — in between instead of waiting for a whole frame.
      let tracking = false
      let retouching = false
      let tracked: { base: HTMLCanvasElement; landmarks: NormalizedLandmark[] | null } | null = null
      const retouchNext = async () => {
        if (retouching || !tracked || gen !== liveGenRef.current) return
        const job = tracked
        tracked = null
        retouching = true
        try {
          const result = await processFrameInSteps(job.base, job.landmarks, paramsRef.current, nextTask)
          if (gen !== liveGenRef.current) return
          baseRef.current = job.base
          landmarksRef.current = job.landmarks
          resultRef.current = result
          setStatus(job.landmarks ? 'ready' : 'no-face')
          render()
        } catch {
          // (a frame lost; the next one tries again)
        } finally {
          retouching = false
          void retouchNext()
        }
      }

      const loop = () => {
        if (gen !== liveGenRef.current) return
        rafRef.current = requestAnimationFrame(loop)
        if (tracking || tracked || video.readyState < 2) return
        const now = performance.now()
        if (now - lastFrameRef.current < LIVE_FRAME_INTERVAL_MS) return
        lastFrameRef.current = now
        tracking = true
        void (async () => {
          try {
            await nextTask()
            if (gen !== liveGenRef.current) return
            const mirror = facingModeRef.current === 'user'
            // Hardware zoom already zoomed the sensor output — cropping again
            // on top of it would double-zoom.
            const crop = cropRectFor(video.videoWidth, video.videoHeight, aspectRatioFor(aspectRef.current), cropZoom(zoomRef.current, zoomRangeRef.current))
            // (the frame's picture and the tracker's copy, taken together)
            const base = drawFrame(video, LIVE_MAX_DIMENSION, crop, mirror)
            const raw = await tracker.detect(video, now, { x: crop.x0, y: crop.y0, w: crop.fw, h: crop.fh })
            if (gen !== liveGenRef.current) return
            tracked = { base, landmarks: remapLandmarks(raw, crop, mirror) }
            void retouchNext()
          } catch {
            // (a frame lost; the next one tries again)
          } finally {
            tracking = false
          }
        })()
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
    // (the camera at full size, where the preview plays a smaller copy)
    const video = stillVideoRef.current?.srcObject ? stillVideoRef.current : videoRef.current
    if (!video || video.readyState < 2) return
    setFlashKey((k) => k + 1)
    navigator.vibrate?.(15)
    const mirror = facingModeRef.current === 'user'
    // This very frame, at the camera's full resolution: the moment the
    // viewfinder freezes on is the one saved.
    const crop = cropRectFor(video.videoWidth, video.videoHeight, aspectRatioFor(aspectRef.current), cropZoom(zoomRef.current, zoomRangeRef.current))
    const full = drawFrame(video, STILL_MAX, crop, mirror, 'high')
    const base = Math.max(full.width, full.height) > MAX_DIMENSION ? drawFrame(full, MAX_DIMENSION, FULL_FRAME, false, 'high') : full
    fullBaseRef.current = base === full ? null : full
    fullResultRef.current = null
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
    fullBaseRef.current = null
    fullResultRef.current = null
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
        await loadStill(drawFrame(img, MAX_DIMENSION, FULL_FRAME, false, 'high'))
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

  // A still follows its controls as they move (the live loop picks up
  // paramsRef on its own every frame): the retouch reruns on the latest
  // values once a frame at most, in a task after the frame's paint, never
  // queued up behind itself. A photo that gets the slower full-quality
  // pass (an upload) gets the quick one while a control moves, and the full
  // one once it's been let go. (It used to wait for the control to stop
  // moving, so the picture lagged behind the finger.)
  useEffect(() => {
    if (live || staticBusyRef.current || !baseRef.current) return
    let cancelled = false
    const twoPass = !capturedRef.current
    const f = requestAnimationFrame(() => void nextTask().then(() => !cancelled && recomputeStatic(twoPass)))
    const t = twoPass ? setTimeout(() => !cancelled && recomputeStatic(), 280) : undefined
    return () => {
      cancelled = true
      cancelAnimationFrame(f)
      clearTimeout(t)
    }
  }, [params, live, recomputeStatic])

  // Filter thumbnails are rendered from the current frame whenever the
  // strip opens (and again for each new still), not every live frame.
  const frameReady = status !== 'loading'
  useEffect(() => {
    if (panel !== 'filter') return
    const base = baseRef.current
    if (!base) return
    let cancelled = false
    const t = setTimeout(() => {
      // (all at once, not one re-render each: the strip shows them together)
      const made = new Map<string, string>()
      void renderFilterThumbnails(base, faceFocus(landmarksRef.current, base.width, base.height), (id, url) => made.set(id, url), () => cancelled).then(() => !cancelled && setThumbs(made))
    }, 30)
    return () => {
      cancelled = true
      clearTimeout(t)
    }
  }, [panel, frameVersion, frameReady])

  // Effects load in the background — the camera keeps running, the effect
  // just appears once it's in; a still is redrawn then.
  useEffect(() => {
    const id = params.effectId
    if (id === 'none' || isEffectReady(id)) {
      setEffectLoading(false)
      return
    }
    let cancelled = false
    setEffectLoading(true)
    prepareEffect(id)
      .catch(() => {})
      .finally(() => {
        if (cancelled) return
        setEffectLoading(false)
        if (!live && !staticBusyRef.current) recomputeStatic()
      })
    return () => {
      cancelled = true
    }
  }, [params.effectId, live, recomputeStatic])
  useEffect(() => onEffectReady(() => !live && !staticBusyRef.current && recomputeStatic()), [live, recomputeStatic])

  // Looks preview on this very face, rendered when the panel opens (and
  // again for each new still), progressively, not per live frame.
  useEffect(() => {
    if (panel !== 'looks') return
    const base = baseRef.current
    if (!base || status === 'loading') return
    let cancelled = false
    const t = setTimeout(() => {
      const items = LOOKS.map((l) => ({ id: l.id, params: applyLook(l, 1, { ...DEFAULT_PARAMS, effectId: 'none' }) }))
      renderFaceThumbs(base, landmarksRef.current, items, 1, (id, url) => setLookThumbs((m) => ({ ...m, [id]: url })), () => cancelled)
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
    setParams((p) => (look ? applyLook(look, strength, p) : { ...DEFAULT_PARAMS, effectId: p.effectId, protectBackground: p.protectBackground }))
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
  const stopRecording = useCallback(() => {
    const r = recRef.current
    if (!r) return
    if (r.rec.state !== 'inactive') r.rec.stop()
    if (r.raw && r.raw.state !== 'inactive') r.raw.stop()
  }, [])

  const startRecording = async () => {
    type CaptureCanvas = HTMLCanvasElement & { captureStream?: (fps?: number) => MediaStream }
    const canvas = displayRef.current as CaptureCanvas | null
    if (!canvas?.captureStream || typeof MediaRecorder === 'undefined') {
      showToast("Video recording isn't supported in this browser")
      return
    }
    setPanel(null)
    const stream = canvas.captureStream(30)
    // The camera to video mode (1080p60, from photo mode's full size) while
    // the microphone is set up; back again if nothing gets recorded.
    const cam = streamRef.current?.getVideoTracks()[0]
    const camVideo = stillVideoRef.current?.srcObject ? stillVideoRef.current : videoRef.current
    const zr = zoomRangeRef.current
    const hwZoom = zr.mode === 'hardware' ? Math.max(zr.floor, zoomRef.current) : null
    const toVideoMode = cam && camVideo ? switchCamera(cam, camVideo, 'video', hwZoom) : Promise.resolve()
    const backToPhotoMode = () => {
      if (cam && camVideo && cam.readyState === 'live') void switchCamera(cam, camVideo, 'photo', hwZoom)
    }
    // Sound too, if the microphone is allowed; a silent video otherwise.
    let audio: MediaStream | null = null
    try {
      audio = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })
      audio.getAudioTracks().forEach((t) => stream.addTrack(t))
    } catch {
      audio = null
    }
    await toVideoMode
    if (!pressHandledRef.current) {
      // Let go while the microphone was being set up: nothing to record.
      audio?.getTracks().forEach((t) => t.stop())
      backToPhotoMode()
      return
    }
    // MP4 with H.264 where it's offered (Safari, current Chrome): it plays
    // everywhere. Otherwise WebM — never a bare "video/mp4", for which some
    // browsers write VP9 into MP4 with a codec tag decoders then reject
    // (the video editor couldn't open the recording).
    const type = ['video/mp4;codecs=avc1,mp4a.40.2', 'video/mp4;codecs=avc1', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'].find((t) => MediaRecorder.isTypeSupported(t))
    const opts = (bps: number) => (type ? { mimeType: type, videoBitsPerSecond: bps } : undefined)
    let rec: MediaRecorder
    try {
      rec = new MediaRecorder(stream, opts(6_000_000))
    } catch {
      audio?.getTracks().forEach((t) => t.stop())
      backToPhotoMode()
      showToast("Video recording isn't supported in this browser")
      return
    }
    // The camera's own stream, recorded as it comes: 1080p at 60 fps
    // where it has them (video mode), the phone's hardware encoder. The look is rendered onto it afterwards, frame by frame —
    // the recording above is only what the preview managed live. Best
    // effort: without it the review keeps that one.
    const camTrack = streamRef.current?.getVideoTracks()[0]
    const v0 = videoRef.current
    const view: RecordView = {
      source: v0?.videoWidth ? v0.videoWidth / v0.videoHeight : 0,
      aspect: aspectRatioFor(aspectRef.current),
      zoom: cropZoom(zoomRef.current, zoomRangeRef.current),
      mirror: facingModeRef.current === 'user',
    }
    let raw: MediaRecorder | null = null
    try {
      if (camTrack && canRenderVideo()) {
        const rs = new MediaStream([camTrack])
        audio?.getAudioTracks().forEach((t) => rs.addTrack(t.clone()))
        raw = new MediaRecorder(rs, opts(16_000_000))
      }
    } catch {
      raw = null
    }
    const chunks: Blob[] = []
    const rawChunks: Blob[] = []
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
    if (raw) raw.ondataavailable = (e) => e.data.size && rawChunks.push(e.data)
    const mime = () => rec.mimeType || type || 'video/webm'
    const stopped = (r: MediaRecorder | null) => new Promise<void>((res) => (!r || r.state === 'inactive' ? res() : r.addEventListener('stop', () => res(), { once: true })))
    const params = { ...paramsRef.current }
    recRef.current = { rec, audio, raw }
    rec.start(250)
    raw?.start(250)
    // (only once both have started: a recorder not yet started reads as
    // already stopped)
    Promise.all([stopped(rec), stopped(raw)]).then(() => {
      const r = recRef.current
      r?.audio?.getTracks().forEach((t) => t.stop())
      recRef.current = null
      setRecordingSince(null)
      if (!chunks.length) return
      const video = new Blob(chunks, { type: mime() })
      const rawBlob = rawChunks.length ? new Blob(rawChunks, { type: raw?.mimeType || mime() }) : null
      // Watch it before keeping it: the camera pauses under the review.
      stopLive()
      setReview({ video, raw: rawBlob, view, params })
    })
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
    fullBaseRef.current = null
    fullResultRef.current = null
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

  // Pinch to zoom, like a phone camera: the zoom follows the fingers'
  // spread (the ratio of their distance now to when the second one landed).
  useEffect(() => {
    const el = previewRef.current
    if (!el || !live) return
    const pts = new Map<number, { x: number; y: number }>()
    let start: { d: number; z: number } | null = null
    let hide: number | undefined
    const spread = () => {
      const [a, b] = [...pts.values()]
      return Math.max(1, Math.hypot(a.x - b.x, a.y - b.y))
    }
    const down = (e: PointerEvent) => {
      if (e.pointerType === 'mouse') return
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY })
      if (pts.size === 2) {
        start = { d: spread(), z: zoomRef.current }
        clearTimeout(hide)
        setPinching(true)
      }
    }
    const move = (e: PointerEvent) => {
      if (!pts.has(e.pointerId)) return
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY })
      if (start && pts.size === 2) applyZoom(Math.round(start.z * (spread() / start.d) * 100) / 100)
    }
    const up = (e: PointerEvent) => {
      if (!pts.delete(e.pointerId)) return
      if (start && pts.size < 2) {
        start = null
        hide = window.setTimeout(() => setPinching(false), PINCH_RULER_LINGER_MS)
      }
    }
    // (Safari's own page zoom ignores user-scalable=no.)
    const noPageZoom = (e: Event) => e.preventDefault()
    el.addEventListener('pointerdown', down)
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    window.addEventListener('pointercancel', up)
    el.addEventListener('gesturestart', noPageZoom)
    return () => {
      clearTimeout(hide)
      setPinching(false)
      el.removeEventListener('pointerdown', down)
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      window.removeEventListener('pointercancel', up)
      el.removeEventListener('gesturestart', noPageZoom)
    }
  }, [live, applyZoom])

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

  /** The edited photo at full resolution: a camera photo is edited at
   *  MAX_DIMENSION, then rendered again from the camera's full-size still
   *  (same edit — the landmarks are relative, the passes scale) to save. */
  const fullResult = async (): Promise<HTMLCanvasElement | null> => {
    const full = fullBaseRef.current
    if (!full) return null
    const params = paramsRef.current
    if (fullResultRef.current?.params === params) return fullResultRef.current.canvas
    setProcessing(true)
    await nextPaint()
    try {
      const canvas = processFrame(full, landmarksRef.current, params, !capturedRef.current)
      fullResultRef.current = { params, canvas }
      return canvas
    } catch {
      // (out of memory, say: the editing-size result is saved instead)
      return null
    } finally {
      setProcessing(false)
    }
  }

  const exportBlob = async (): Promise<Blob | null> => {
    const src = (await fullResult()) ?? resultRef.current ?? baseRef.current
    if (!src) return null
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
    const name = timestampedName('jpg')
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
    const name = timestampedName('jpg')
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

  const setParam = (key: NumericParam, v: number) => setParams((p) => ({ ...p, [key]: v }))
  const beauty = beautyItems(params, setParam)
  const shape = shapeItems(params, setParam)
  const bgToggle = (
    <BgProtectToggle
      on={params.protectBackground}
      disabled={status === 'no-face'}
      onChange={(on) => {
        saveBgProtect(on)
        setParams((p) => ({ ...p, protectBackground: on }))
      }}
    />
  )

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
    // (No backdrop of its own: around the picture, the theme's ambient
    // background shows through.)
    <div className="fixed inset-0 overflow-hidden select-none">
      <video ref={videoRef} autoPlay playsInline muted className="hidden" />
      <video ref={stillVideoRef} autoPlay playsInline muted className="hidden" />

      <div
        ref={previewRef}
        className={`absolute inset-x-0 flex justify-center ${(live || captured) && !fullBleed ? 'items-start' : 'items-center'}`}
        style={live ? { ...previewStyle, touchAction: 'none' } : previewStyle}
      >
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
          <TopButton label="Settings" onClick={() => setSettingsOpen(true)}>
            <IconSettings className="w-5 h-5" />
          </TopButton>
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
      {effectLoading && !showBefore && !processing && recordingSince === null && (
        <div className="absolute left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 px-3 py-1 rounded-full bg-black/60 backdrop-blur-md text-[11px] font-semibold text-white" style={{ top: `calc(${topInset} + 3.25rem)` }}>
          <IconSpinner className="w-3 h-3 animate-spin text-primary" />
          Loading {findEffect(params.effectId).label}…
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
            <ZoomControl range={zoomRange} value={zoom} onChange={applyZoom} pinching={pinching} />
          </div>
        )}

        {panel && (
          <div ref={panelRef} key={panel} className="fg-panel rounded-t-[22px] bg-black/45 backdrop-blur-xl border-t border-white/10 px-3.5 pt-3 pb-1">
            {panel === 'looks' && (
              <div>
                <div className="flex items-center justify-between h-7 mb-3">
                  <span className="text-[13px] font-semibold text-white">Looks</span>
                  <div className="flex items-center gap-3">
                  {bgToggle}
                  <button onClick={() => selectLook(null)} disabled={!lookId} className="flex items-center gap-1 text-[11px] font-medium text-white/70 hover:text-white disabled:opacity-30 transition">
                    <IconRefresh className="w-3.5 h-3.5" />
                    Reset
                  </button>
                  </div>
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
            {panel === 'beauty' && <AdjustPanel title="Beauty" items={beauty} disabled={noFace} />}
            {panel === 'shape' && <AdjustPanel title="Shape" items={shape} disabled={noFace} tabs extra={bgToggle} />}
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
                    thumbs={EFFECT_THUMBS}
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
              <TrayButton compact={c} icon={IconSparkle} label="Beauty" toggle active={panel === 'beauty'} dot={changedFrom(params, BEAUTY_KEYS)} onClick={() => togglePanel('beauty')} />
              <TrayButton compact={c} icon={IconFaceOutline} label="Shape" toggle active={panel === 'shape'} dot={changedFrom(params, SHAPE_KEYS)} onClick={() => togglePanel('shape')} />
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
        accept={onEditVideo ? 'image/*,video/*' : 'image/*'}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file && isVideoFile(file) && onEditVideo) onEditVideo(file)
          else if (file) onPickImage?.(file)
          e.target.value = ''
        }}
      />
      {review && (
        <VideoReview
          blob={review.video}
          hd={review.raw ? { raw: review.raw, view: review.view, params: review.params } : undefined}
          onClose={() => {
            setReview(null)
            startLive()
          }}
          onEdit={
            onEditVideo
              ? (clean?: Blob) => {
                  const r = review
                  setReview(null)
                  // The untouched copy starts from the settings it was
                  // filmed with; the retouched one (no copy) from scratch.
                  onEditVideo(clean ?? r.video, clean ? r.params : { ...DEFAULT_PARAMS, smoothness: 0, face: 0 })
                }
              : undefined
          }
          editLabel="Edit people"
        />
      )}
      {settingsOpen && <SettingsSheet onClose={() => setSettingsOpen(false)} />}
    </div>
  )
}
