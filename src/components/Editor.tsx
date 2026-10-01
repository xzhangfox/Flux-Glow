import { useEffect, useRef, useState, useCallback } from 'react'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { detectFaceLandmarks, detectFaceLandmarksForVideo } from '../lib/faceLandmarker'
import { processFrame, type EditParams } from '../lib/pipeline'
import { FILTER_PRESETS } from '../lib/filters'
import Slider from './Slider'
import ParamToolbar, { type ParamDef } from './ParamToolbar'

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

function drawDownscaled(source: HTMLImageElement | HTMLVideoElement, maxDim: number): HTMLCanvasElement {
  const w = source instanceof HTMLVideoElement ? source.videoWidth : source.width
  const h = source instanceof HTMLVideoElement ? source.videoHeight : source.height
  const scale = Math.min(1, maxDim / Math.max(w, h))
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(w * scale)
  canvas.height = Math.round(h * scale)
  canvas.getContext('2d')!.drawImage(source, 0, 0, canvas.width, canvas.height)
  return canvas
}

export default function Editor({ source, onReset }: { source: Source; onReset: () => void }) {
  const [status, setStatus] = useState<Status>('loading')
  const [smoothness, setSmoothness] = useState(0.6)
  const [contour, setContour] = useState(0.25)
  const [filterId, setFilterId] = useState('none')
  const [showBefore, setShowBefore] = useState(false)
  const [liveActive, setLiveActive] = useState(source.kind === 'live')

  const displayRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const baseRef = useRef<HTMLCanvasElement | null>(null)
  const landmarksRef = useRef<NormalizedLandmark[] | null>(null)
  const resultRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef(0)
  const lastProcessRef = useRef(0)
  const liveActiveRef = useRef(liveActive)
  const paramsRef = useRef<EditParams>({ smoothness, contour, filterId })

  useEffect(() => {
    paramsRef.current = { smoothness, contour, filterId }
  }, [smoothness, contour, filterId])
  useEffect(() => {
    liveActiveRef.current = liveActive
  }, [liveActive])

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
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 480 } } })
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
        const base = drawDownscaled(video, LIVE_MAX_DIMENSION)
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
  }, [smoothness, contour, filterId, liveActive, recomputeStatic])

  const handleCapture = () => {
    stopLive()
    setLiveActive(false)
  }

  const handleRetake = () => {
    startLive()
  }

  const disabled = status === 'no-face'
  const params: ParamDef[] = [
    {
      id: 'skin',
      icon: 'blur_on',
      label: 'Smooth',
      isActive: smoothness > 0.01,
      render: () => <Slider label="Smooth & Clear" value={smoothness} onChange={setSmoothness} disabled={disabled} />,
    },
    {
      id: 'contour',
      icon: 'face',
      label: 'Contour',
      isActive: contour > 0.01,
      render: () => <Slider label="Slim" value={contour} onChange={setContour} disabled={disabled} />,
    },
    {
      id: 'filter',
      icon: 'palette',
      label: 'Filter',
      isActive: filterId !== 'none',
      render: () => (
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
      ),
    },
  ]

  const handleDownload = () => {
    const canvas = resultRef.current ?? baseRef.current
    if (!canvas) return
    canvas.toBlob(
      (blob) => {
        if (!blob) return
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'flux-glow.jpg'
        a.click()
        URL.revokeObjectURL(url)
      },
      'image/jpeg',
      0.95
    )
  }

  return (
    <div className="flex flex-col items-center gap-4 w-full max-w-md">
      <div className="relative w-full rounded-3xl overflow-hidden bg-surface border border-white/10 shadow-glow">
        <canvas ref={displayRef} className="w-full h-auto block" />
        <video ref={videoRef} autoPlay playsInline muted className="hidden" />

        {status === 'loading' && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60">
            <span className="material-symbols-outlined animate-spin text-primary text-3xl">progress_activity</span>
          </div>
        )}

        {liveActive && status !== 'loading' && (
          <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur-sm border border-white/10">
            <span className="w-1.5 h-1.5 rounded-full bg-danger animate-pulse" />
            <span className="text-[10px] font-semibold tracking-wide text-white/90">LIVE</span>
          </div>
        )}

        <button
          onMouseDown={() => setShowBefore(true)}
          onMouseUp={() => setShowBefore(false)}
          onMouseLeave={() => setShowBefore(false)}
          onTouchStart={() => setShowBefore(true)}
          onTouchEnd={() => setShowBefore(false)}
          disabled={status === 'loading'}
          className="absolute bottom-3 right-3 px-3 py-1.5 rounded-lg bg-black/60 text-white text-xs font-medium backdrop-blur-sm border border-white/10 select-none flex items-center gap-1"
        >
          <span className="material-symbols-outlined text-sm">compare</span>
          Hold for Before
        </button>

        {liveActive && status === 'ready' && (
          <button
            onClick={handleCapture}
            className="absolute bottom-3 left-1/2 -translate-x-1/2 w-14 h-14 rounded-full bg-primary border-4 border-white/80 shadow-glow-strong flex items-center justify-center hover:brightness-110 transition"
            aria-label="Capture"
          >
            <span className="material-symbols-outlined text-black text-2xl">photo_camera</span>
          </button>
        )}
      </div>
      {status === 'no-face' && (
        <p className="text-text-secondary text-xs text-center max-w-md flex items-center gap-1.5">
          <span className="material-symbols-outlined text-sm">info</span>
          No face detected — skin and contour need a visible face, filters still work.
        </p>
      )}
      {status === 'error' && (
        <p className="text-danger text-xs flex items-center gap-1.5">
          <span className="material-symbols-outlined text-sm">error</span>
          {source.kind === 'live' ? "Couldn't access the camera — check your browser permissions." : "Couldn't load that photo."}
        </p>
      )}

      <ParamToolbar params={params} />

      <div className="flex gap-3 w-full">
        {liveActive ? (
          <button
            onClick={onReset}
            className="flex-1 py-3 bg-secondary text-white rounded-xl hover:bg-white/10 transition flex items-center justify-center gap-1.5"
          >
            <span className="material-symbols-outlined text-lg">close</span>
            Exit Live
          </button>
        ) : (
          <>
            <button
              onClick={handleDownload}
              disabled={status === 'loading'}
              className="flex-1 py-3 bg-primary text-black font-semibold rounded-xl hover:brightness-110 transition shadow-glow disabled:opacity-40 flex items-center justify-center gap-1.5"
            >
              <span className="material-symbols-outlined text-lg">download</span>
              Download
            </button>
            {source.kind === 'live' ? (
              <button onClick={handleRetake} className="px-4 py-3 bg-secondary text-white rounded-xl hover:bg-white/10 transition" aria-label="Retake">
                <span className="material-symbols-outlined text-lg">replay</span>
              </button>
            ) : (
              <button onClick={onReset} className="px-4 py-3 bg-secondary text-white rounded-xl hover:bg-white/10 transition" aria-label="New photo">
                <span className="material-symbols-outlined text-lg">refresh</span>
              </button>
            )}
          </>
        )}
      </div>
    </div>
  )
}
