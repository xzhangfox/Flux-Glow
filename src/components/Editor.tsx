import { useEffect, useRef, useState } from 'react'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { detectFaceLandmarks } from '../lib/faceLandmarker'
import { buildSkinMask } from '../lib/skinMask'
import { smoothSkin } from '../lib/smoothing'
import { applyFilter, FILTER_PRESETS } from '../lib/filters'
import Slider from './Slider'

// Downscale before processing — phone photos run 3000px+ on a side, far
// more detail than this editor displays or than frequency separation
// needs; working at this size keeps every slider drag responsive.
const MAX_DIMENSION = 1600

type Status = 'loading' | 'ready' | 'no-face' | 'error'

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = URL.createObjectURL(file)
  })
}

function drawDownscaled(img: HTMLImageElement): HTMLCanvasElement {
  const scale = Math.min(1, MAX_DIMENSION / Math.max(img.width, img.height))
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(img.width * scale)
  canvas.height = Math.round(img.height * scale)
  canvas.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height)
  return canvas
}

export default function Editor({ file, onReset }: { file: File; onReset: () => void }) {
  const [status, setStatus] = useState<Status>('loading')
  const [smoothness, setSmoothness] = useState(0.6)
  const [filterId, setFilterId] = useState('none')
  const [showBefore, setShowBefore] = useState(false)
  const displayRef = useRef<HTMLCanvasElement>(null)

  const baseRef = useRef<HTMLCanvasElement | null>(null)
  const maskRef = useRef<HTMLCanvasElement | null>(null)
  const landmarksRef = useRef<NormalizedLandmark[] | null>(null)
  const resultRef = useRef<HTMLCanvasElement | null>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Load + detect once per incoming file.
  useEffect(() => {
    let cancelled = false
    setStatus('loading')
    ;(async () => {
      try {
        const img = await loadImage(file)
        const base = drawDownscaled(img)
        if (cancelled) return
        baseRef.current = base

        const landmarks = await detectFaceLandmarks(base)
        if (cancelled) return
        landmarksRef.current = landmarks
        maskRef.current = landmarks ? buildSkinMask(landmarks, base.width, base.height) : null
        setStatus(landmarks ? 'ready' : 'no-face')
      } catch {
        if (!cancelled) setStatus('error')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [file])

  // Recompute the processed result whenever the controls or the loaded
  // image change — debounced so dragging a slider stays smooth instead of
  // re-running frequency separation on every intermediate value.
  useEffect(() => {
    if (status !== 'ready' && status !== 'no-face') return
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      const base = baseRef.current
      if (!base) return
      const preset = FILTER_PRESETS.find((p) => p.id === filterId) ?? FILTER_PRESETS[0]
      const mask = maskRef.current
      const smoothed = mask ? smoothSkin(base, mask, smoothness) : base
      resultRef.current = applyFilter(smoothed, preset)
      render()
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, 60)
  }, [status, smoothness, filterId, showBefore])

  function render() {
    const canvas = displayRef.current
    const base = baseRef.current
    const result = resultRef.current
    if (!canvas || !base) return
    canvas.width = base.width
    canvas.height = base.height
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(showBefore || !result ? base : result, 0, 0)
  }

  useEffect(() => {
    render()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showBefore])

  const handleDownload = () => {
    const canvas = resultRef.current ?? baseRef.current
    if (!canvas) return
    canvas.toBlob((blob) => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'flux-glow.jpg'
      a.click()
      URL.revokeObjectURL(url)
    }, 'image/jpeg', 0.95)
  }

  return (
    <div className="flex flex-col lg:flex-row gap-6 w-full max-w-4xl">
      <div className="flex-1 flex flex-col items-center gap-3">
        <div className="relative w-full max-w-md rounded-2xl overflow-hidden bg-surface border border-white/10">
          <canvas ref={displayRef} className="w-full h-auto block" />
          {status === 'loading' && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/60">
              <span className="material-symbols-outlined animate-spin text-primary text-3xl">progress_activity</span>
            </div>
          )}
          <button
            onMouseDown={() => setShowBefore(true)}
            onMouseUp={() => setShowBefore(false)}
            onMouseLeave={() => setShowBefore(false)}
            onTouchStart={() => setShowBefore(true)}
            onTouchEnd={() => setShowBefore(false)}
            disabled={status === 'loading'}
            className="absolute bottom-3 right-3 px-3 py-1.5 rounded-lg bg-black/60 text-white text-xs font-medium backdrop-blur-sm border border-white/10 select-none"
          >
            Hold for Before
          </button>
        </div>
        {status === 'no-face' && (
          <p className="text-text-secondary text-xs text-center max-w-md">
            No face detected — skin smoothing needs a visible face, but filters still work below.
          </p>
        )}
        {status === 'error' && <p className="text-danger text-xs">Couldn't load that photo. Try another one.</p>}
      </div>

      <div className="w-full lg:w-72 flex flex-col gap-6">
        <div className="bg-surface rounded-2xl p-5 border border-white/10">
          <h3 className="text-sm font-semibold text-white/90 mb-4">Skin</h3>
          <Slider label="Smooth & Clear" value={smoothness} onChange={setSmoothness} disabled={status === 'no-face'} />
        </div>

        <div className="bg-surface rounded-2xl p-5 border border-white/10">
          <h3 className="text-sm font-semibold text-white/90 mb-4">Filter</h3>
          <div className="grid grid-cols-3 gap-2">
            {FILTER_PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => setFilterId(p.id)}
                className={`py-2 rounded-lg text-xs font-medium transition border ${
                  filterId === p.id
                    ? 'bg-primary text-black border-primary'
                    : 'bg-secondary text-white/70 border-transparent hover:bg-white/10'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex gap-3 mt-auto">
          <button
            onClick={handleDownload}
            disabled={status === 'loading'}
            className="flex-1 py-3 bg-primary text-black font-semibold rounded-xl hover:brightness-110 transition shadow-glow disabled:opacity-40 flex items-center justify-center gap-1.5"
          >
            <span className="material-symbols-outlined text-lg">download</span>
            Download
          </button>
          <button
            onClick={onReset}
            className="px-4 py-3 bg-secondary text-white rounded-xl hover:bg-white/10 transition"
            aria-label="New photo"
          >
            <span className="material-symbols-outlined text-lg">refresh</span>
          </button>
        </div>
      </div>
    </div>
  )
}
