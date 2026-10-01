import { useEffect, useRef, useState, useCallback, type ComponentType, type SVGProps } from 'react'
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
} from './icons'

// Downscale before processing — phone photos run 3000px+ on a side, far
// more detail than this editor displays or than frequency separation
// needs; working at this size keeps every slider drag responsive.
const MAX_DIMENSION = 1600
// Live mode works at a much smaller size: the per-pixel smoothing/contour
// passes are plain JS, not GPU shaders, so this is what keeps the preview
// loop actually feeling live rather than stuttering.
const LIVE_MAX_DIMENSION = 480
const LIVE_FRAME_INTERVAL_MS = 90 // ~11fps cap on the heavy processing

export type Source = { kind: 'image'; file: File } | { kind: 'live' }

type Status = 'loading' | 'ready' | 'no-face' | 'error'

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
// getUserMedia doesn't expose a phone's separate physical lenses.
function drawDownscaled(source: HTMLImageElement | HTMLVideoElement, maxDim: number, zoom = 1): HTMLCanvasElement {
  const w = source instanceof HTMLVideoElement ? source.videoWidth : source.width
  const h = source instanceof HTMLVideoElement ? source.videoHeight : source.height
  const scale = Math.min(1, maxDim / Math.max(w, h))
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(w * scale)
  canvas.height = Math.round(h * scale)
  const ctx = canvas.getContext('2d')!
  if (zoom > 1) {
    const cropW = w / zoom
    const cropH = h / zoom
    ctx.drawImage(source, (w - cropW) / 2, (h - cropH) / 2, cropW, cropH, 0, 0, canvas.width, canvas.height)
  } else {
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height)
  }
  return canvas
}

const ZOOM_LEVELS = [1, 2]

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

export default function Editor({ source, onReset }: { source: Source; onReset: () => void }) {
  const [status, setStatus] = useState<Status>('loading')
  const [smoothness, setSmoothness] = useState(0.6)
  const [face, setFace] = useState(0.25)
  const [eyes, setEyes] = useState(0)
  const [nose, setNose] = useState(0)
  const [mouth, setMouth] = useState(0)
  const [filterId, setFilterId] = useState('none')
  const [showBefore, setShowBefore] = useState(false)
  const [liveActive, setLiveActive] = useState(source.kind === 'live')
  const [confirmed, setConfirmed] = useState(false)
  const [openPanel, setOpenPanel] = useState<'retouch' | 'filter' | null>(null)
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user')
  const [zoom, setZoom] = useState(1)

  const displayRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const baseRef = useRef<HTMLCanvasElement | null>(null)
  const landmarksRef = useRef<NormalizedLandmark[] | null>(null)
  const resultRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef(0)
  const lastProcessRef = useRef(0)
  const liveActiveRef = useRef(liveActive)
  const paramsRef = useRef<EditParams>({ smoothness, face, eyes, nose, mouth, filterId })
  const facingModeRef = useRef(facingMode)
  const zoomRef = useRef(zoom)
  const panelRef = useRef<HTMLDivElement>(null)
  const retouchButtonRef = useRef<HTMLButtonElement>(null)
  const filterButtonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    paramsRef.current = { smoothness, face, eyes, nose, mouth, filterId }
  }, [smoothness, face, eyes, nose, mouth, filterId])
  useEffect(() => {
    liveActiveRef.current = liveActive
  }, [liveActive])
  useEffect(() => {
    zoomRef.current = zoom
  }, [zoom])

  // Tapping anywhere outside the open panel (or the Retouch/Filter buttons
  // that toggle it, which handle themselves) collapses it — same pattern
  // as a tap-away menu.
  useEffect(() => {
    function onClick(e: MouseEvent) {
      const target = e.target as Node
      if (panelRef.current?.contains(target)) return
      if (retouchButtonRef.current?.contains(target)) return
      if (filterButtonRef.current?.contains(target)) return
      setOpenPanel(null)
    }
    document.addEventListener('click', onClick)
    return () => document.removeEventListener('click', onClick)
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
  }, [])

  const startLive = useCallback(async () => {
    setStatus('loading')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facingModeRef.current, width: { ideal: 480 } },
      })
      streamRef.current = stream
      const video = videoRef.current!
      video.srcObject = stream
      await video.play()
      setStatus('ready')
      setLiveActive(true)

      const loop = () => {
        rafRef.current = requestAnimationFrame(loop)
        if (!liveActiveRef.current) return
        const now = performance.now()
        if (now - lastProcessRef.current < LIVE_FRAME_INTERVAL_MS) return
        lastProcessRef.current = now
        if (video.readyState < 2) return
        const base = drawDownscaled(video, LIVE_MAX_DIMENSION, zoomRef.current)
        baseRef.current = base
        detectFaceLandmarksForVideo(video, now).then((landmarks) => {
          landmarksRef.current = landmarks
          setStatus(landmarks ? 'ready' : 'no-face')
          resultRef.current = processFrame(base, landmarks, paramsRef.current)
          render()
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
  }, [smoothness, face, eyes, nose, mouth, filterId, liveActive, recomputeStatic])

  // Any further adjustment after confirming means the exported image would
  // no longer match what's on screen — fall back to Confirm again rather
  // than silently leaving a stale Save/Share up.
  useEffect(() => {
    setConfirmed(false)
  }, [smoothness, face, eyes, nose, mouth, filterId])

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

  const handleToggleZoom = () => {
    const i = ZOOM_LEVELS.indexOf(zoom)
    const next = ZOOM_LEVELS[(i + 1) % ZOOM_LEVELS.length]
    zoomRef.current = next
    setZoom(next)
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
        <button
          onClick={handleToggleZoom}
          aria-label="Zoom level"
          className="absolute left-1/2 -translate-x-1/2 bottom-44 w-9 h-9 rounded-full bg-black/50 backdrop-blur-sm border border-white/20 text-white text-[11px] font-semibold flex items-center justify-center"
        >
          {zoom}×
        </button>
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

      <div
        className="absolute inset-x-0 bottom-0 flex flex-col gap-3 px-4 pt-8"
        style={{ paddingBottom: 'max(1.25rem, env(safe-area-inset-bottom))', background: 'linear-gradient(to top, rgba(0,0,0,0.85), rgba(0,0,0,0.4) 60%, transparent)' }}
      >
        {(status === 'no-face' || status === 'error') && (
          <p
            className={`self-center text-xs text-center flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-sm ${
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
          className={`grid transition-all duration-250 ease-out ${openPanel ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}
          style={{ transitionProperty: 'grid-template-rows, opacity' }}
        >
          <div className="overflow-hidden">
            <div className="bg-black/70 backdrop-blur-md rounded-2xl p-4 border border-white/10">
              {openPanel === 'retouch' ? (
                <div className="space-y-4 max-h-[42vh] overflow-y-auto pr-1">
                  <Slider label="Smooth & Clear" value={smoothness} onChange={setSmoothness} disabled={disabled} />
                  <Slider label="Face" value={face} onChange={setFace} disabled={disabled} />
                  <Slider label="Eyes" value={eyes} onChange={setEyes} disabled={disabled} />
                  <Slider label="Nose" value={nose} onChange={setNose} disabled={disabled} />
                  <Slider label="Mouth" value={mouth} onChange={setMouth} disabled={disabled} />
                </div>
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

        <div className="grid grid-cols-[1fr_auto_1fr] items-end gap-2 pt-1">
          <div className="flex items-center gap-3 justify-self-start">
            <button
              ref={retouchButtonRef}
              onClick={() => setOpenPanel((v) => (v === 'retouch' ? null : 'retouch'))}
              aria-label="Retouch"
              className="flex flex-col items-center gap-1 w-14 text-white/85"
            >
              <span
                className={`w-11 h-11 rounded-full flex items-center justify-center backdrop-blur-sm border transition ${
                  openPanel === 'retouch' ? 'bg-primary text-black border-primary' : 'bg-black/40 border-white/15'
                }`}
              >
                <IconFaceOutline className="w-5 h-5" />
              </span>
              <span className="text-[10px] font-medium leading-none">Retouch</span>
            </button>
            <button
              ref={filterButtonRef}
              onClick={() => setOpenPanel((v) => (v === 'filter' ? null : 'filter'))}
              aria-label="Filter"
              className="flex flex-col items-center gap-1 w-14 text-white/85"
            >
              <span
                className={`w-11 h-11 rounded-full flex items-center justify-center backdrop-blur-sm border transition ${
                  openPanel === 'filter' ? 'bg-primary text-black border-primary' : 'bg-black/40 border-white/15'
                }`}
              >
                <IconPalette className="w-5 h-5" />
              </span>
              <span className="text-[10px] font-medium leading-none">Filter</span>
            </button>
            {source.kind === 'live' && !liveActive && <ChromeButton icon={IconRefresh} label="Retake" onClick={handleRetake} />}
          </div>

          <div className="justify-self-center">{centerButton}</div>

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
  )
}
