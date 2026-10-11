import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ComponentType, type ReactNode, type RefObject, type SVGProps } from 'react'
import { DEFAULT_PARAMS, faceFocus, processFaces, type EditParams, type FaceEdit, type NumericParam } from '../lib/pipeline'
import { analyzeVideo, preloadVideoModels, type PersonInfo, type VideoAnalysis } from '../lib/video/analyze'
import { canExportVideo, exportVideo, fpsChoices, planExport, probeSource, sizeChoices, type BitrateTier, type ExportResult, type SourceInfo } from '../lib/video/export'
import { FaceFinder, LiveSmoother, type Found } from '../lib/video/follow'
import { boxOf, unpack } from '../lib/video/faces'
import { freeSpace, removeTemp, sweepTemp } from '../lib/video/disk'
import { ensureSeekable } from '../lib/video/source'
import { holdScreenOn } from '../lib/wakeLock'
import { EFFECTS, EFFECT_THUMBS, effectsChanged, findEffect, onEffectReady, prepareEffect } from '../lib/effects'
import { effectsBlink, setEffectsBlink } from '../lib/effectPrefs'
import { LOOKS, applyLook, findLook } from '../lib/looks'
import { renderFaceThumbs } from '../lib/thumbs'
import { renderFilterThumbnails } from '../lib/filters'
import AdjustPanel from './AdjustPanel'
import BgProtectToggle from './BgProtectToggle'
import { loadBgProtect, saveBgProtect } from '../lib/bgProtect'
import FilterPanel from './FilterPanel'
import ThumbStrip from './ThumbStrip'
import Slider from './Slider'
import VideoReview from './VideoReview'
import { BEAUTY_KEYS, SHAPE_KEYS, beautyItems, changedFrom, shapeItems } from './adjustItems'
import {
  IconCheck,
  IconClose,
  IconCompare,
  IconEars,
  IconError,
  IconEyeOff,
  IconFaceOutline,
  IconInfo,
  IconMerge,
  IconPalette,
  IconPause,
  IconPeople,
  IconPlay,
  IconRefresh,
  IconSparkle,
  IconSpinner,
  IconWand,
} from './icons'

// The video editor: every person found in the clip gets their own Beauty,
// Shape, Look and Effect — or one setting for everyone — over a frame-wide
// Filter. The clip is analysed once up front (who is where, and who is who;
// see video/analyze.ts); after that, each frame shown or exported has its
// faces found again and put to the right person (video/follow.ts), so
// effects stay locked on each person through the whole clip, however long.
//
// Preview plays the real video (with its sound) and retouches each shown
// frame at a working size; the export (video/export.ts) redoes every frame
// at the chosen size, in order, at full quality.

/** The preview's working size (long side). */
const PREVIEW_MAX = 720
/** Tag and lane colours, one per person (cycled). */
const PERSON_COLORS = ['#f5c542', '#4fd1c5', '#f687b3', '#90cdf4', '#b794f4', '#f6ad55', '#68d391', '#fc8181']
const colorOf = (id: number) => PERSON_COLORS[(id - 1) % PERSON_COLORS.length]

type Phase = 'preparing' | 'analyzing' | 'ready' | 'error'
type Panel = 'looks' | 'beauty' | 'shape' | 'effects' | 'filter'
type Target = 'all' | number

function fmt(s: number) {
  if (!Number.isFinite(s)) return '0:00'
  const t = Math.max(0, Math.floor(s))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}` : `${m}:${String(t % 60).padStart(2, '0')}`
}

/** "About 3 min", "about 1 h 20 min". */
function fmtLeft(s: number) {
  if (s < 60) return 'under a minute'
  const m = Math.round(s / 60)
  return m < 60 ? `about ${m} min` : `about ${Math.floor(m / 60)} h ${m % 60} min`
}

const FIXED_KEY = 'glow_fixed_camera'
function loadFixedCamera() {
  try {
    return localStorage.getItem(FIXED_KEY) === '1'
  } catch {
    return false
  }
}
function saveFixedCamera(on: boolean) {
  try {
    localStorage.setItem(FIXED_KEY, on ? '1' : '0')
  } catch {
    // (private mode: just this session)
  }
}

function fmtBytes(b: number) {
  return b >= 1e9 ? `${(b / 1e9).toFixed(b >= 1e10 ? 0 : 1)} GB` : `${Math.max(1, Math.round(b / 1e6))} MB`
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

function Choice<T>({ label, options, value, onChange }: { label: string; options: { value: T; label: string; detail?: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="mb-3">
      <p className="text-white/55 text-[11px] font-medium mb-1.5">{label}</p>
      <div className="flex gap-1.5">
        {options.map((o) => {
          const on = o.value === value
          return (
            <button key={String(o.value)} onClick={() => onChange(o.value)} aria-pressed={on} className={`flex-1 min-w-0 rounded-xl px-2 py-2 border text-center ${on ? 'border-primary bg-primary/10' : 'border-white/10 bg-white/[0.04]'}`}>
              <span className={`block text-[12.5px] font-semibold leading-tight ${on ? 'text-primary' : 'text-white'}`}>{o.label}</span>
              {o.detail && <span className="block text-white/45 text-[10px] mt-0.5 truncate">{o.detail}</span>}
            </button>
          )
        })}
      </div>
    </div>
  )
}

const sameParams = (a: EditParams, b: EditParams) =>
  (Object.keys(a) as (keyof EditParams)[]).every((k) => k === 'filterId' || k === 'filterStrength' || (typeof a[k] === 'number' ? Math.abs((a[k] as number) - (b[k] as number)) < 0.005 : a[k] === b[k]))

function TrayButton({ icon: Icon, label, onClick, active, dot }: { icon: ComponentType<SVGProps<SVGSVGElement>>; label: string; onClick: () => void; active?: boolean; dot?: boolean }) {
  return (
    <button onClick={onClick} aria-label={label} aria-pressed={active} data-panel-toggle="" className="flex flex-col items-center gap-1 w-[3.4rem] text-white/85">
      <span className={`relative w-10 h-10 rounded-full flex items-center justify-center border transition ${active ? 'bg-primary text-black border-primary' : 'bg-white/[0.06] border-white/15'}`}>
        <Icon className="w-[18px] h-[18px]" />
        {dot && !active && <span className="absolute top-0 right-0 w-2 h-2 rounded-full bg-primary ring-2 ring-black/60" />}
      </span>
      <span className="text-[10px] font-medium leading-none">{label}</span>
    </button>
  )
}

function Sheet({ children, onDismiss }: { children: ReactNode; onDismiss: () => void }) {
  return (
    <div className="absolute inset-0 z-40 bg-black/60 flex items-end justify-center" onClick={onDismiss}>
      <div data-keep-panel className="fg-panel w-full max-w-md rounded-t-[22px] bg-surface border-t border-white/10 px-5 pt-5" style={{ paddingBottom: 'max(1.25rem, env(safe-area-inset-bottom))' }} onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  )
}

export default function VideoEditor({ file, startParams, onClose }: { file: Blob; startParams?: EditParams; onClose: () => void }) {
  const initial = useMemo(() => ({ ...(startParams ?? { ...DEFAULT_PARAMS, protectBackground: loadBgProtect() }) }), [startParams])
  const [phase, setPhase] = useState<Phase>('preparing')
  const [error, setError] = useState('')
  const [progress, setProgress] = useState(0)
  const [found, setFound] = useState(0)
  const [source, setSource] = useState<Blob | null>(null)
  const [srcInfo, setSrcInfo] = useState<SourceInfo | null>(null)
  // A fixed camera (interviews, talk shows): analysed knowing people stay
  // in their seats. Remembered: whoever edits those edits them every time.
  const [fixedCam, setFixedCam] = useState(loadFixedCamera)
  const [askReanalyze, setAskReanalyze] = useState(false)
  const [url, setUrl] = useState<string | null>(null)
  const [analyzeStart, setAnalyzeStart] = useState(0)
  const analysisRef = useRef<VideoAnalysis | null>(null)
  // The faces in the frame on show (found afresh for each one), and the
  // smoothing that steadies them while it plays.
  const finderRef = useRef<FaceFinder | null>(null)
  const smoother = useRef(new LiveSmoother())
  const shownRef = useRef<{ time: number; w: number; faces: Found[] } | null>(null)
  const [people, setPeople] = useState<PersonInfo[]>([])

  // Edits: `all` is the Everyone setting and also carries the frame-wide
  // filter; `per` each person's own; `off` people left untouched.
  const [all, setAll] = useState<EditParams>(initial)
  const [per, setPer] = useState<Record<number, EditParams>>({})
  const [off, setOff] = useState<Record<number, boolean>>({})
  const [target, setTarget] = useState<Target>('all')
  const [blink, setBlink] = useState(effectsBlink)
  const [lookSel, setLookSel] = useState<Record<string, { id: string | null; strength: number }>>({})
  const [panel, setPanel] = useState<Panel | null>(null)
  const [mergeFrom, setMergeFrom] = useState<number | null>(null)

  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [aspect, setAspect] = useState(9 / 16)
  const [showBefore, setShowBefore] = useState(false)
  const [tags, setTags] = useState<{ id: number; x0: number; y0: number; x1: number; y1: number }[]>([])
  const [lookThumbs, setLookThumbs] = useState<Record<string, string>>({})
  const [filterThumbs, setFilterThumbs] = useState<Map<string, string> | null>(null)
  const [toast, setToast] = useState<{ key: number; text: string } | null>(null)

  const [exportSheet, setExportSheet] = useState(false)
  const [quality, setQuality] = useState<'high' | 'fast'>('high')
  const [outSize, setOutSize] = useState<number | null>(null)
  const [outFps, setOutFps] = useState<number | null>(null)
  const [outRate, setOutRate] = useState<BitrateTier>('recommended')
  const [space, setSpace] = useState<number | null>(null)
  const [exporting, setExporting] = useState<{ progress: number; started: number } | null>(null)
  const exportAbort = useRef<AbortController | null>(null)
  const [exported, setExported] = useState<ExportResult | null>(null)
  const [askLeave, setAskLeave] = useState(false)
  const dirty = useRef(false)

  const videoRef = useRef<HTMLVideoElement>(null)
  const displayRef = useRef<HTMLCanvasElement>(null)
  const workRef = useRef<HTMLCanvasElement | null>(null)
  const previewRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const state = useRef({ all, per, off, showBefore })
  state.current = { all, per, off, showBefore }

  const previewBox = useElementSize(previewRef)
  const bottomBox = useElementSize(bottomRef)
  const fit = fitContain(previewBox, aspect)
  const showToast = (text: string) => setToast({ key: Date.now(), text })

  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 1800)
    return () => clearTimeout(t)
  }, [toast])

  // Tapping outside the open panel closes it (as in the camera).
  useEffect(() => {
    function onClick(e: MouseEvent) {
      const t = e.target as Element
      if (panelRef.current?.contains(t)) return
      if (t.closest?.('[data-panel-toggle], [data-keep-panel]')) return
      setPanel(null)
    }
    document.addEventListener('click', onClick, true)
    return () => document.removeEventListener('click', onClick, true)
  }, [])

  // ---- Load ----
  useEffect(() => {
    const ac = new AbortController()
    let objectUrl: string | null = null
    ;(async () => {
      try {
        preloadVideoModels().catch(() => {})
        void sweepTemp()
        const src = await ensureSeekable(file)
        if (ac.signal.aborted) return
        objectUrl = URL.createObjectURL(src)
        setSource(src)
        setUrl(objectUrl)
        probeSource(src).then((i) => !ac.signal.aborted && setSrcInfo(i), () => {})
      } catch (e) {
        if ((e as Error).name === 'AbortError') return
        setError((e as Error).message || "Couldn't open this video.")
        setPhase('error')
      }
    })()
    return () => {
      ac.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [file])

  // ---- Analyse ---- (again if the camera kind changes)
  useEffect(() => {
    if (!source) return
    const ac = new AbortController()
    // (an hour-long clip takes a while: the screen stays on meanwhile)
    const release = holdScreenOn()
    ;(async () => {
      try {
        analysisRef.current = null
        finderRef.current = null
        shownRef.current = null
        setPeople([])
        setTags([])
        setTarget('all')
        setMergeFrom(null)
        setPanel(null)
        setProgress(0)
        setFound(0)
        setPhase('analyzing')
        setAnalyzeStart(performance.now())
        // (a few updates a second: re-rendering the editor for every frame
        // analysed would slow the analysis down)
        let shown = 0
        const an = await analyzeVideo(
          source,
          (p) => {
            const now = performance.now()
            if (now - shown < 200) return
            shown = now
            setProgress(p.progress)
            setFound(p.people)
          },
          ac.signal,
          { fixed: fixedCam },
        )
        if (ac.signal.aborted) return
        finderRef.current = await FaceFinder.create(an)
        if (ac.signal.aborted) return
        analysisRef.current = an
        smoother.current = new LiveSmoother()
        smoother.current.calm = an.fixed
        setPeople(an.people)
        setPer(Object.fromEntries(an.people.map((p) => [p.id, { ...initial }])))
        setOff({})
        setProgress(1)
        setPhase('ready')
      } catch (e) {
        if ((e as Error).name === 'AbortError') return
        setError((e as Error).message || "Couldn't open this video.")
        setPhase('error')
      } finally {
        release()
      }
    })()
    return () => {
      ac.abort()
      release()
    }
  }, [source, fixedCam, initial])
  const chooseFixedCam = (on: boolean) => {
    saveFixedCamera(on)
    setFixedCam(on)
  }

  // 3D effects load in the background, one model per person wearing one;
  // the (paused) preview redraws as each comes in.
  const anyEffect = all.effectId !== 'none' || Object.values(per).some((p) => p.effectId !== 'none')

  // ---- Rendering ----
  const paramsFor = (id: number, s = state.current): EditParams | null => (s.off[id] ? null : (s.per[id] ?? s.all))

  // Tags on the paused frame: each face found in it, and anyone the
  // analysis has there whose face isn't showing (turned away, say).
  const updateTags = useCallback((t: number) => {
    const an = analysisRef.current
    if (!an) return setTags([])
    const shown = shownRef.current?.time === t ? shownRef.current.faces : []
    const tags = shown.map((f) => ({ id: f.id, ...boxOf(f.lm, 1, 1) }))
    for (const e of an.boxesAt(t)) if (!tags.some((x) => x.id === e.id)) tags.push({ id: e.id, ...e.box })
    setTags(tags)
  }, [])

  const renderNow = useCallback(
    (t?: number) => {
      const v = videoRef.current
      const out = displayRef.current
      if (!v || !out || v.readyState < 2 || !v.videoWidth) return
      const time = t ?? v.currentTime
      const k = Math.min(1, PREVIEW_MAX / Math.max(v.videoWidth, v.videoHeight))
      const w = Math.round(v.videoWidth * k)
      const h = Math.round(v.videoHeight * k)
      const work = (workRef.current ??= document.createElement('canvas'))
      if (work.width !== w || work.height !== h) {
        work.width = w
        work.height = h
      }
      work.getContext('2d')!.drawImage(v, 0, 0, w, h)
      const s = state.current
      // The faces in this frame (once per frame: a slider moved on a paused
      // frame reuses them).
      const finder = finderRef.current
      let shown = shownRef.current
      if (finder && (!shown || shown.time !== time || shown.w !== w)) {
        const playing = !v.paused
        const found = finder.find(work, time, playing)
        if (!playing) smoother.current.reset()
        shown = shownRef.current = { time, w, faces: playing ? smoother.current.apply(found, time) : found }
      }
      let result: HTMLCanvasElement = work
      if (!s.showBefore) {
        const faces: FaceEdit[] = []
        for (const f of shown?.faces ?? []) {
          const params = paramsFor(f.id, s)
          if (params) faces.push({ landmarks: unpack(f.lm), params, slot: `p${f.id}` })
        }
        result = processFaces(work, faces, s.all, false, time)
      }
      if (out.width !== w || out.height !== h) {
        out.width = w
        out.height = h
      }
      out.getContext('2d')!.drawImage(result, 0, 0)
      if (v.paused) updateTags(time)
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [updateTags],
  )

  // (paused: redraw as each effect comes in)
  useEffect(() => onEffectReady(() => !videoRef.current?.paused || renderNow()), [renderNow])

  // Playback: retouch each frame as the video presents it.
  useEffect(() => {
    const v = videoRef.current
    if (!v || !playing) return
    let stop = false
    let handle = 0
    let lastUi = 0
    const rvfc = typeof v.requestVideoFrameCallback === 'function'
    const tick = (_now: number, meta?: VideoFrameCallbackMetadata) => {
      if (stop) return
      renderNow(meta?.mediaTime ?? v.currentTime)
      const n = performance.now()
      if (n - lastUi > 120) {
        lastUi = n
        setTime(v.currentTime)
      }
      handle = rvfc ? v.requestVideoFrameCallback(tick) : requestAnimationFrame((x) => tick(x))
    }
    handle = rvfc ? v.requestVideoFrameCallback(tick) : requestAnimationFrame((x) => tick(x))
    return () => {
      stop = true
      if (rvfc) v.cancelVideoFrameCallback(handle)
      else cancelAnimationFrame(handle)
    }
  }, [playing, renderNow])

  // Paused: redraw on any change.
  useEffect(() => {
    if (playing) return
    const t = setTimeout(() => renderNow(), 16)
    return () => clearTimeout(t)
  }, [all, per, off, showBefore, playing, phase, renderNow])

  const togglePlay = () => {
    const v = videoRef.current
    if (!v || phase !== 'ready') return
    if (v.paused) {
      setTags([])
      if (v.ended || v.currentTime >= v.duration - 0.05) v.currentTime = 0
      v.play().catch(() => {})
    } else v.pause()
  }

  // Space plays and pauses on a keyboard.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || (e.target as Element).closest?.('input, textarea, button')) return
      e.preventDefault()
      togglePlay()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // ---- Editing the target ----
  const current: EditParams = target === 'all' ? all : (per[target] ?? all)
  const editTarget = (fn: (p: EditParams) => EditParams) => {
    dirty.current = true
    if (target === 'all') {
      setAll((p) => fn(p))
      setPer((m) => Object.fromEntries(Object.entries(m).map(([id, p]) => [id, { ...fn(p), filterId: p.filterId, filterStrength: p.filterStrength }])))
    } else setPer((m) => ({ ...m, [target]: fn(m[target] ?? all) }))
  }
  const setKey = (key: NumericParam, v: number) => editTarget((p) => ({ ...p, [key]: v }))
  const lookKey = String(target)
  const look = lookSel[lookKey] ?? { id: null, strength: 1 }
  const selectLook = (id: string | null, strength = look.strength) => {
    setLookSel((m) => ({ ...m, [lookKey]: { id, strength } }))
    const l = findLook(id)
    editTarget((p) => (l ? applyLook(l, strength, p) : { ...DEFAULT_PARAMS, effectId: p.effectId, filterId: p.filterId, filterStrength: p.filterStrength, protectBackground: p.protectBackground }))
  }

  const focusPerson = target === 'all' ? people[0] : people.find((p) => p.id === target)
  const anyFaces = people.length > 0

  // Previews of looks on the person being edited (effects have ready-made
  // tiles).
  useEffect(() => {
    if (panel !== 'looks' || !focusPerson) return
    let cancelled = false
    const { frame, landmarks } = focusPerson.best
    const t = setTimeout(() => {
      setLookThumbs({})
      const items = LOOKS.map((l) => ({ id: l.id, params: applyLook(l, 1, { ...DEFAULT_PARAMS, effectId: 'none' }) }))
      renderFaceThumbs(frame, landmarks, items, 1, (id, u) => setLookThumbs((m) => ({ ...m, [id]: u })), () => cancelled)
    }, 40)
    return () => {
      cancelled = true
      clearTimeout(t)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panel, focusPerson?.id])

  useEffect(() => {
    if (panel !== 'filter') return
    const work = workRef.current
    if (!work) return
    const snap = document.createElement('canvas')
    snap.width = work.width
    snap.height = work.height
    snap.getContext('2d')!.drawImage(work, 0, 0)
    const f0 = shownRef.current?.faces[0]
    const lm = f0 ? unpack(f0.lm) : null
    let cancelled = false
    const t = setTimeout(() => {
      const made = new Map<string, string>()
      void renderFilterThumbnails(snap, faceFocus(lm, snap.width, snap.height), (id, url) => made.set(id, url), () => cancelled).then(() => !cancelled && setFilterThumbs(made))
    }, 30)
    return () => {
      cancelled = true
      clearTimeout(t)
    }
  }, [panel])

  // ---- People ----
  const pickPerson = (id: Target) => {
    if (mergeFrom !== null && id !== 'all' && id !== mergeFrom) {
      const an = analysisRef.current!
      an.merge(mergeFrom, id)
      setPeople(an.people)
      setPer((m) => {
        const n = { ...m }
        delete n[mergeFrom]
        return n
      })
      setOff((m) => {
        const n = { ...m }
        delete n[mergeFrom]
        return n
      })
      setMergeFrom(null)
      setTarget(id)
      showToast(`Merged into Person ${id}`)
      shownRef.current = null
      renderNow()
      return
    }
    setMergeFrom(null)
    setTarget(id)
  }

  const customised = (id: number) => !!per[id] && !sameParams(per[id], all)

  // ---- Export ----
  // The last export stays on disk (a long one is gigabytes) until the next
  // one starts, or the editor closes.
  const lastExport = useRef<Blob | null>(null)
  useEffect(() => () => void removeTemp(lastExport.current), [])
  const openExportSheet = () => {
    setExportSheet(true)
    freeSpace().then(setSpace, () => setSpace(null))
  }
  const plan = srcInfo ? planExport(srcInfo, { size: outSize, fps: outFps, bitrate: outRate }) : null

  const startExport = async () => {
    const an = analysisRef.current
    if (!an || !source) return
    setExportSheet(false)
    videoRef.current?.pause()
    if (!canExportVideo()) {
      showToast('Exporting needs a newer browser')
      return
    }
    // Every effect anyone wears, ready before the first frame goes out.
    if (anyEffect) await Promise.all(an.people.map((p) => paramsFor(p.id)).flatMap((q, i) => (q && q.effectId !== 'none' ? [prepareEffect(q.effectId, `p${an.people[i].id}`)] : []))).catch(() => {})
    const ac = new AbortController()
    exportAbort.current = ac
    setExporting({ progress: 0, started: performance.now() })
    const snapshot = { ...state.current }
    const release = holdScreenOn()
    await removeTemp(lastExport.current)
    lastExport.current = null
    try {
      let shown = 0
      const onProgress = (p: number) => {
        const now = performance.now()
        if (now - shown < 250) return
        shown = now
        setExporting((e) => (e ? { ...e, progress: p } : e))
      }
      const res = await exportVideo(source, an, (id) => paramsFor(id, snapshot), snapshot.all, { size: outSize, fps: outFps, bitrate: outRate, quality }, onProgress, ac.signal)
      lastExport.current = res.blob
      if (res.frames.written !== null && res.frames.written !== res.frames.expected) console.warn(`Export: ${res.frames.expected} frames sent, ${res.frames.written} in the file`)
      setExported(res)
      dirty.current = false
    } catch (e) {
      if (!ac.signal.aborted) showToast((e as Error).message || 'Export failed')
    } finally {
      release()
      setExporting(null)
      exportAbort.current = null
      renderNow()
    }
  }

  const leave = () => (dirty.current ? setAskLeave(true) : onClose())

  // ---- Layout ----
  const topInset = 'max(0.75rem, env(safe-area-inset-top))'
  const bottomH = useRef(0)
  if (!panel && bottomBox.height) bottomH.current = bottomBox.height
  const ready = phase === 'ready'
  const eta = exporting && exporting.progress > 0.01 ? ((performance.now() - exporting.started) / exporting.progress) * (1 - exporting.progress) : null
  const analyzeEta = phase === 'analyzing' && progress > 0.02 && analyzeStart ? ((performance.now() - analyzeStart) / progress) * (1 - progress) : null

  const beauty = beautyItems(current, setKey)
  const shape = shapeItems(current, setKey)
  const bgToggle = (
    <BgProtectToggle
      on={current.protectBackground}
      disabled={!anyFaces}
      onChange={(on) => {
        saveBgProtect(on)
        editTarget((p) => ({ ...p, protectBackground: on }))
      }}
    />
  )
  const targetName = target === 'all' ? 'Everyone' : `Person ${target}`

  return (
    <div className="fixed inset-0 z-30 overflow-hidden select-none">
      {url && (
        <video
          ref={videoRef}
          src={url}
          playsInline
          preload="auto"
          className="hidden"
          onLoadedMetadata={(e) => {
            const v = e.currentTarget
            setDuration(v.duration)
            if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight)
          }}
          onLoadedData={() => renderNow()}
          onDurationChange={(e) => setDuration(e.currentTarget.duration)}
          onSeeked={(e) => {
            setTime(e.currentTarget.currentTime)
            renderNow()
          }}
          onPlay={() => setPlaying(true)}
          onPause={(e) => {
            setPlaying(false)
            setTime(e.currentTarget.currentTime)
            renderNow()
          }}
          onEnded={() => setPlaying(false)}
        />
      )}

      {/* Preview */}
      <div ref={previewRef} className="absolute inset-x-0 flex items-center justify-center" style={{ top: `calc(${topInset} + 3.25rem)`, bottom: bottomH.current || bottomBox.height }}>
        <div className="relative" style={{ width: fit.width, height: fit.height }} onClick={() => ready && !panel && togglePlay()}>
          <canvas ref={displayRef} className="block w-full h-full rounded-md" />
          {ready && !playing && !showBefore && (
            <div className="absolute inset-0">
              {tags.map((t) => {
                const sel = target === t.id
                const c = colorOf(t.id)
                const pad = 0.12
                const w = t.x1 - t.x0
                const h = t.y1 - t.y0
                return (
                  <button
                    key={t.id}
                    data-keep-panel
                    aria-label={`Edit Person ${t.id}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      pickPerson(t.id)
                    }}
                    className="absolute rounded-[28%] transition"
                    style={{
                      left: `${(t.x0 - w * pad) * 100}%`,
                      top: `${(t.y0 - h * pad) * 100}%`,
                      width: `${w * (1 + 2 * pad) * 100}%`,
                      height: `${h * (1 + 2 * pad) * 100}%`,
                      border: `${sel ? 2.5 : 1.5}px ${off[t.id] ? 'dashed' : 'solid'} ${c}`,
                      boxShadow: sel ? `0 0 0 3px ${c}33, 0 0 18px ${c}55` : undefined,
                      opacity: target === 'all' || sel ? 1 : 0.55,
                    }}
                  >
                    <span className="absolute left-1/2 -translate-x-1/2 -top-[22px] px-1.5 h-[18px] rounded-full text-[10.5px] font-bold text-black flex items-center whitespace-nowrap" style={{ background: c }}>
                      {t.id}
                    </span>
                  </button>
                )
              })}
            </div>
          )}
          {ready && !playing && tags.length === 0 && !panel && (
            <span className="absolute inset-0 m-auto w-14 h-14 rounded-full bg-black/45 backdrop-blur-md flex items-center justify-center text-white pointer-events-none">
              <IconPlay className="w-7 h-7 ml-0.5" />
            </span>
          )}
        </div>
      </div>

      <div className="absolute top-0 inset-x-0 h-24 bg-gradient-to-b from-black/60 to-transparent pointer-events-none" />

      {/* Top bar */}
      <div className="absolute inset-x-0 flex items-center justify-between px-3 z-10" style={{ top: topInset }}>
        <button aria-label="Close the video editor" onClick={leave} className="h-10 w-10 rounded-full bg-black/40 backdrop-blur-md text-white flex items-center justify-center">
          <IconClose className="w-5 h-5" />
        </button>
        <span className="text-white text-[13px] font-semibold">Edit video</span>
        <div className="flex items-center gap-2">
          <button
            aria-label="Hold to compare with the original"
            disabled={!ready}
            onPointerDown={() => setShowBefore(true)}
            onPointerUp={() => setShowBefore(false)}
            onPointerLeave={() => setShowBefore(false)}
            onPointerCancel={() => setShowBefore(false)}
            onContextMenu={(e) => e.preventDefault()}
            style={{ touchAction: 'none' }}
            className={`h-10 w-10 rounded-full backdrop-blur-md flex items-center justify-center disabled:opacity-40 ${showBefore ? 'bg-primary text-black' : 'bg-black/40 text-white'}`}
          >
            <IconCompare className="w-5 h-5" />
          </button>
          <button onClick={openExportSheet} disabled={!ready || !!exporting} className="h-10 px-4 rounded-full bg-primary text-black text-[13px] font-semibold disabled:opacity-40">
            Export
          </button>
        </div>
      </div>
      {showBefore && (
        <div className="absolute left-1/2 -translate-x-1/2 z-10 px-3 py-1 rounded-full bg-black/60 backdrop-blur-md text-[11px] font-semibold text-white" style={{ top: `calc(${topInset} + 3.25rem)` }}>
          Original
        </div>
      )}

      {toast && (
        <div key={toast.key} className="fg-toast absolute left-1/2 z-30 flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-black/75 backdrop-blur-md text-xs font-medium text-white" style={{ top: '42%' }}>
          <IconCheck className="w-3.5 h-3.5 text-primary" />
          {toast.text}
        </div>
      )}

      {/* Analysis */}
      {(phase === 'preparing' || phase === 'analyzing') && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/55 backdrop-blur-[2px]">
          <div className="w-[min(320px,86vw)] rounded-2xl bg-black/70 border border-white/10 px-5 py-5 text-center">
            <IconPeople className="w-7 h-7 text-primary mx-auto mb-3" />
            <p className="text-white text-[14px] font-semibold">{phase === 'preparing' ? 'Opening video…' : 'Finding the people in your video…'}</p>
            <p className="text-white/55 text-[11.5px] mt-1 mb-4">Each person gets their own retouch and effects, tracked through every frame. This runs on your device.</p>
            <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
              <div className="h-full bg-primary transition-[width] duration-200" style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
            <div className="flex justify-between text-[11px] text-white/60 mt-2 tabular-nums">
              <span>
                {Math.round(progress * 100)}%{analyzeEta !== null && analyzeEta > 20_000 ? ` · ${fmtLeft(analyzeEta / 1000)} left` : ''}
              </span>
              <span>
                {found} {found === 1 ? 'person' : 'people'} found
              </span>
            </div>
            <div className="mt-4 pt-3.5 border-t border-white/10 text-left">
              <button role="switch" aria-checked={fixedCam} aria-label="Fixed camera" onClick={() => chooseFixedCam(!fixedCam)} className="w-full flex items-center justify-between gap-3 text-left">
                <span>
                  <span className="block text-white text-[12.5px] font-semibold">Fixed camera</span>
                  <span className="block text-white/50 text-[11px] leading-snug mt-0.5">Interviews, talk shows: people stay in their seats. Fewer mix-ups between people; effects stay on through a turned head.</span>
                </span>
                <span className={`relative w-9 h-[22px] flex-shrink-0 rounded-full transition ${fixedCam ? 'bg-primary' : 'bg-white/20'}`}>
                  <span className={`absolute top-[3px] w-4 h-4 rounded-full bg-white shadow transition-all ${fixedCam ? 'left-[17px]' : 'left-[3px]'}`} />
                </span>
              </button>
            </div>
            <button onClick={onClose} className="mt-4 h-9 px-4 rounded-full bg-white/10 text-white text-[12px] font-semibold">
              Cancel
            </button>
          </div>
        </div>
      )}
      {phase === 'error' && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/70">
          <div className="w-[min(320px,86vw)] rounded-2xl bg-black/70 border border-white/10 px-5 py-5 text-center">
            <IconError className="w-7 h-7 text-danger mx-auto mb-3" />
            <p className="text-white text-[14px] font-semibold mb-1">Can't edit this video</p>
            <p className="text-white/60 text-[12px] mb-4">{error}</p>
            <button onClick={onClose} className="h-10 px-5 rounded-full bg-white/10 text-white text-sm font-semibold">
              Close
            </button>
          </div>
        </div>
      )}

      {/* Bottom: panel, timeline, people, tools */}
      <div className="absolute inset-x-0 bottom-0 flex flex-col items-stretch">
        {panel && ready && (
          <div ref={panelRef} key={panel + lookKey} className="fg-panel rounded-t-[22px] bg-black/55 backdrop-blur-xl border-t border-white/10 px-3.5 pt-3 pb-1">
            {panel !== 'filter' && (
              <div className="flex items-center gap-1.5 mb-1.5 text-[11px] text-white/60">
                {target === 'all' ? <IconPeople className="w-3.5 h-3.5" /> : <span className="w-2.5 h-2.5 rounded-full" style={{ background: colorOf(target) }} />}
                Editing <span className="text-white font-semibold">{targetName}</span>
                {target !== 'all' && off[target] && <span className="text-white/45">· left as is</span>}
              </div>
            )}
            {panel === 'looks' && (
              <div>
                <div className="flex items-center justify-between h-7 mb-2">
                  <span className="text-[13px] font-semibold text-white">Looks</span>
                  <div className="flex items-center gap-3">
                    {bgToggle}
                    <button onClick={() => selectLook(null)} disabled={!look.id} className="flex items-center gap-1 text-[11px] font-medium text-white/70 disabled:opacity-30">
                      <IconRefresh className="w-3.5 h-3.5" />
                      Reset
                    </button>
                  </div>
                </div>
                {!anyFaces ? (
                  <p className="text-xs text-white/60 py-4 text-center">Looks need a face in the video.</p>
                ) : (
                  <ThumbStrip items={LOOKS} thumbs={lookThumbs} selected={look.id} onSelect={(id) => selectLook(id)} />
                )}
                {findLook(look.id) && (
                  <div className="mt-3">
                    <Slider label="Intensity" value={look.strength} defaultValue={1} onChange={(v) => selectLook(look.id, v)} />
                  </div>
                )}
              </div>
            )}
            {panel === 'beauty' && <AdjustPanel title="Beauty" items={beauty} disabled={!anyFaces} />}
            {panel === 'shape' && <AdjustPanel title="Shape" items={shape} disabled={!anyFaces} tabs extra={bgToggle} />}
            {panel === 'effects' && (
              <div>
                <div className="flex items-center justify-between h-7 mb-2">
                  <span className="text-[13px] font-semibold text-white">Effects</span>
                  <div className="flex items-center gap-2.5">
                    <span className="text-[10.5px] text-white/45">On {target === 'all' ? 'every face' : `Person ${target}`}</span>
                    {/* (the same switch as in Settings) */}
                    <button
                      role="switch"
                      aria-checked={blink}
                      aria-label="Animal heads blink"
                      title="Animal heads close their eyes when the wearer does"
                      onClick={() => {
                        setBlink(!blink)
                        setEffectsBlink(!blink)
                        effectsChanged()
                      }}
                      className="flex items-center gap-1.5 h-7 pl-2.5 pr-1 rounded-full bg-white/[0.07] text-[11px] font-medium text-white/80 transition active:scale-95"
                    >
                      Blink
                      <span className={`relative w-7 h-[18px] rounded-full transition ${blink ? 'bg-primary' : 'bg-white/20'}`}>
                        <span className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white shadow transition-all ${blink ? 'left-[12px]' : 'left-[2px]'}`} />
                      </span>
                    </button>
                  </div>
                </div>
                {!anyFaces ? (
                  <p className="text-xs text-white/60 py-4 text-center">Effects need a face in the video.</p>
                ) : (
                  <ThumbStrip
                    items={EFFECTS.filter((e) => e.id !== 'custom').map((e) => ({ id: e.id, label: e.label, badge: e.boost ? '♥' : undefined }))}
                    thumbs={EFFECT_THUMBS}
                    selected={current.effectId}
                    onSelect={(id) => editTarget((p) => ({ ...p, effectId: id }))}
                    fallback={(id) => (id === 'none' ? <IconClose className="w-6 h-6 text-white/60" /> : null)}
                  />
                )}
                {findEffect(current.effectId).boost && <p className="text-[11px] text-white/55 mt-2.5">Comes with a baby-face touch-up on top of Beauty and Shape.</p>}
              </div>
            )}
            {panel === 'filter' && (
              <div>
                <p className="text-[11px] text-white/55 mb-1">The filter applies to the whole video.</p>
                <FilterPanel
                  thumbs={filterThumbs}
                  filterId={all.filterId}
                  strength={all.filterStrength}
                  defaultStrength={DEFAULT_PARAMS.filterStrength}
                  onSelect={(id) => {
                    dirty.current = true
                    setAll((p) => ({ ...p, filterId: id }))
                  }}
                  onStrength={(v) => {
                    dirty.current = true
                    setAll((p) => ({ ...p, filterStrength: v }))
                  }}
                />
              </div>
            )}
          </div>
        )}

        <div ref={bottomRef} className="bg-black/60 backdrop-blur-2xl border-t border-white/10 px-3 pt-2.5" style={{ paddingBottom: 'max(0.7rem, env(safe-area-inset-bottom))' }}>
          {/* Timeline */}
          <div className="flex items-center gap-2.5">
            <button aria-label={playing ? 'Pause' : 'Play'} disabled={!ready} onClick={togglePlay} className="w-9 h-9 rounded-full bg-white/10 text-white flex items-center justify-center disabled:opacity-40">
              {playing ? <IconPause className="w-[18px] h-[18px]" /> : <IconPlay className="w-[18px] h-[18px] ml-0.5" />}
            </button>
            <div className="relative flex-1 h-9">
              {/* Where each person is on screen. */}
              <div className="absolute inset-x-[7px] top-[24px] h-[9px] flex flex-col gap-[2px] pointer-events-none">
                {people.slice(0, 4).map((p) => (
                  <div key={p.id} className="relative h-[1.5px] flex-1">
                    {p.spans.map(([a, b], i) => (
                      <span
                        key={i}
                        className="absolute h-full rounded-full"
                        style={{
                          left: `${(a / Math.max(0.001, duration)) * 100}%`,
                          width: `${(Math.max(0.05, b - a) / Math.max(0.001, duration)) * 100}%`,
                          background: colorOf(p.id),
                          opacity: target === 'all' || target === p.id ? 0.95 : 0.3,
                        }}
                      />
                    ))}
                  </div>
                ))}
              </div>
              <input
                type="range"
                min={0}
                max={Number.isFinite(duration) && duration > 0 ? duration : 1}
                step={0.001}
                value={time}
                disabled={!ready}
                aria-label="Position in the video"
                onChange={(e) => {
                  const v = videoRef.current
                  const t = Number(e.target.value)
                  setTime(t)
                  if (v) {
                    if (typeof v.fastSeek === 'function' && !v.paused) v.fastSeek(t)
                    else v.currentTime = t
                  }
                }}
                className="absolute inset-x-0 top-0 h-6 w-full accent-primary"
              />
            </div>
            <span className="text-[10.5px] text-white/65 tabular-nums w-[70px] text-right">
              {fmt(time)} / {fmt(duration)}
            </span>
          </div>

          {/* People */}
          <div className="mt-2">
            {mergeFrom !== null ? (
              <div className="flex items-center justify-between h-6 mb-1 text-[11px]">
                <span className="text-white/80">
                  Tap who <span className="font-semibold" style={{ color: colorOf(mergeFrom) }}>Person {mergeFrom}</span> really is
                </span>
                <button data-keep-panel onClick={() => setMergeFrom(null)} className="text-white/60 font-semibold">
                  Cancel
                </button>
              </div>
            ) : (
              <div className="flex items-center justify-between gap-2 h-6 mb-1 whitespace-nowrap">
                {target === 'all' ? (
                  <>
                    <span className="text-[11px] text-white/60 truncate">
                      {ready ? (anyFaces ? `${people.length} ${people.length === 1 ? 'person' : 'people'} · tap a face to edit just them` : 'No faces found — the filter still applies') : ' '}
                    </span>
                    {ready && (
                      <button data-keep-panel onClick={() => setAskReanalyze(true)} className={`h-6 px-2 rounded-full text-[10.5px] font-semibold flex-shrink-0 ${fixedCam ? 'bg-primary/15 text-primary' : 'bg-white/10 text-white/70'}`}>
                        {fixedCam ? 'Fixed camera' : 'Moving camera'}
                      </button>
                    )}
                  </>
                ) : (
                  <span className="text-[11px] text-white/60 flex items-center gap-1.5 truncate">
                    <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: colorOf(target) }} />
                    <span className="text-white font-semibold">Person {target}</span>
                  </span>
                )}
                {target !== 'all' && (
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    <button
                      data-keep-panel
                      onClick={() => {
                        dirty.current = true
                        setOff((m) => ({ ...m, [target]: !m[target] }))
                      }}
                      className={`h-6 px-2 rounded-full text-[10.5px] font-semibold flex items-center gap-1 ${off[target] ? 'bg-primary text-black' : 'bg-white/10 text-white/80'}`}
                    >
                      <IconEyeOff className="w-3 h-3" />
                      {off[target] ? 'Untouched' : 'Leave as is'}
                    </button>
                    {people.length > 1 && (
                      <button data-keep-panel onClick={() => setMergeFrom(target)} className="h-6 px-2 rounded-full bg-white/10 text-white/80 text-[10.5px] font-semibold flex items-center gap-1">
                        <IconMerge className="w-3 h-3" />
                        Same as…
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}
            <div className="flex gap-2.5 overflow-x-auto no-scrollbar py-1 -mx-1 px-1">
              <button data-keep-panel onClick={() => pickPerson('all')} aria-pressed={target === 'all'} disabled={!ready} className="flex flex-col items-center gap-1 flex-shrink-0 disabled:opacity-40">
                <span className={`w-11 h-11 rounded-full flex items-center justify-center bg-white/[0.07] ${target === 'all' ? 'ring-2 ring-primary text-primary' : 'ring-1 ring-white/15 text-white/80'}`}>
                  <IconPeople className="w-5 h-5" />
                </span>
                <span className={`text-[10px] font-medium leading-none ${target === 'all' ? 'text-primary' : 'text-white/70'}`}>Everyone</span>
              </button>
              {people.map((p) => {
                const sel = target === p.id
                const c = colorOf(p.id)
                const merging = mergeFrom !== null && mergeFrom !== p.id
                return (
                  <button key={p.id} data-keep-panel onClick={() => pickPerson(p.id)} aria-pressed={sel} aria-label={`Person ${p.id}`} className="flex flex-col items-center gap-1 flex-shrink-0">
                    <span className={`relative w-11 h-11 rounded-full overflow-hidden ${merging ? 'animate-pulse' : ''}`} style={{ boxShadow: `0 0 0 ${sel ? 2.5 : 1.5}px ${c}` }}>
                      <img src={p.thumb} alt="" className={`w-full h-full object-cover ${off[p.id] ? 'opacity-40 grayscale' : ''}`} draggable={false} />
                      {customised(p.id) && <span className="absolute top-0.5 right-0.5 w-2 h-2 rounded-full bg-primary ring-2 ring-black/60" />}
                    </span>
                    <span className="text-[10px] font-medium leading-none" style={{ color: sel ? c : 'rgba(255,255,255,0.7)' }}>
                      Person {p.id}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Tools */}
          <div className="flex items-center justify-between mt-2">
            <TrayButton icon={IconWand} label="Looks" active={panel === 'looks'} dot={!!look.id} onClick={() => setPanel((v) => (v === 'looks' ? null : 'looks'))} />
            <TrayButton icon={IconSparkle} label="Beauty" active={panel === 'beauty'} dot={changedFrom(current, BEAUTY_KEYS)} onClick={() => setPanel((v) => (v === 'beauty' ? null : 'beauty'))} />
            <TrayButton icon={IconFaceOutline} label="Shape" active={panel === 'shape'} dot={changedFrom(current, SHAPE_KEYS)} onClick={() => setPanel((v) => (v === 'shape' ? null : 'shape'))} />
            <TrayButton icon={IconEars} label="Effects" active={panel === 'effects'} dot={current.effectId !== 'none'} onClick={() => setPanel((v) => (v === 'effects' ? null : 'effects'))} />
            <TrayButton icon={IconPalette} label="Filter" active={panel === 'filter'} dot={all.filterId !== 'none'} onClick={() => setPanel((v) => (v === 'filter' ? null : 'filter'))} />
          </div>
        </div>
      </div>

      {exportSheet && (
        <Sheet onDismiss={() => setExportSheet(false)}>
          <p className="text-white text-[15px] font-semibold mb-1">Export video</p>
          <p className="text-white/55 text-[12px] mb-4">MP4 with the original sound, every frame retouched on your device — up to the original's own size and frame rate.</p>
          {srcInfo && (
            <>
              <Choice label="Resolution" options={sizeChoices(srcInfo)} value={outSize} onChange={setOutSize} />
              <Choice label="Frame rate" options={fpsChoices(srcInfo).map((o) => ({ ...o, detail: o.value === null ? 'original' : 'fps' }))} value={outFps} onChange={setOutFps} />
              <Choice
                label="Bitrate"
                options={(['lower', 'recommended', 'higher'] as const).map((b) => ({ value: b, label: b === 'lower' ? 'Lower' : b === 'recommended' ? 'Recommended' : 'Higher', detail: `${(planExport(srcInfo, { size: outSize, fps: outFps, bitrate: b }).bitrate / 1e6).toFixed(1)} Mbps` }))}
                value={outRate}
                onChange={setOutRate}
              />
            </>
          )}
          <Choice
            label="Retouching"
            options={[
              { value: 'high' as const, label: 'Best quality', detail: 'photo-grade, slower' },
              { value: 'fast' as const, label: 'Fast', detail: 'preview-grade, quicker' },
            ]}
            value={quality}
            onChange={setQuality}
          />
          {plan && srcInfo && (
            <div className="text-[11.5px] text-white/60 mb-1 space-y-1">
              <p>
                {plan.width}×{plan.height} · {outFps ? fpsChoices(srcInfo).find((o) => o.value === outFps)?.label : fpsChoices(srcInfo)[0].label} fps · about <span className="text-white font-semibold">{fmtBytes(plan.bytes)}</span> for {fmt(srcInfo.duration)}
              </p>
              {space !== null && space < plan.bytes * 1.1 && <p className="text-danger">This device has only {fmtBytes(space)} free for it — choose a smaller size or bitrate, or free up space.</p>}
              {srcInfo.duration > 15 * 60 && <p>A long video takes a long time — keep this page open and the screen on until it's done.</p>}
            </div>
          )}
          <button onClick={startExport} className="w-full h-11 mt-2 rounded-full bg-primary text-black text-sm font-semibold">
            Export
          </button>
        </Sheet>
      )}

      {exporting && (
        <div className="absolute inset-0 z-40 bg-black/75 backdrop-blur-sm flex items-center justify-center">
          <div className="w-[min(320px,86vw)] rounded-2xl bg-black/70 border border-white/10 px-5 py-5 text-center">
            <IconSpinner className="w-7 h-7 text-primary animate-spin mx-auto mb-3" />
            <p className="text-white text-[14px] font-semibold">Exporting your video…</p>
            <p className="text-white/55 text-[11.5px] mt-1 mb-4">Keep this page open until it finishes.</p>
            <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
              <div className="h-full bg-primary transition-[width] duration-200" style={{ width: `${Math.round(exporting.progress * 100)}%` }} />
            </div>
            <div className="flex justify-between text-[11px] text-white/60 mt-2 tabular-nums">
              <span>{Math.round(exporting.progress * 100)}%</span>
              <span>{eta !== null && exporting.progress > 0.002 ? `${fmtLeft(eta / 1000)} left` : 'starting…'}</span>
            </div>
            <button onClick={() => exportAbort.current?.abort()} className="mt-4 h-9 px-4 rounded-full bg-white/10 text-white text-[12px] font-semibold">
              Cancel
            </button>
          </div>
        </div>
      )}

      {exported && (
        <VideoReview
          blob={exported.blob}
          onClose={() => setExported(null)}
          closeLabel="Back to editing"
          confirmClose={false}
          notice={exported.droppedAudio ? "This browser couldn't carry the sound across, so the export is silent." : undefined}
        />
      )}

      {askReanalyze && (
        <Sheet onDismiss={() => setAskReanalyze(false)}>
          <p className="text-white text-[15px] font-semibold mb-1">{fixedCam ? 'Analyse for a moving camera?' : 'Analyse for a fixed camera?'}</p>
          <p className="text-white/60 text-[12px] mb-1.5">
            {fixedCam
              ? 'For a camera that moves or cuts between shots, or people who move around.'
              : 'For interviews and talk shows, where the camera stays put and people stay in their seats: fewer mix-ups between people, and effects stay on through a turned head.'}
          </p>
          <p className="text-white/45 text-[11.5px] mb-4">The video is analysed again; settings for individual people start over.</p>
          <div className="flex gap-3">
            <button onClick={() => setAskReanalyze(false)} className="flex-1 h-11 rounded-full bg-white/10 text-white text-sm font-semibold">
              Cancel
            </button>
            <button
              onClick={() => {
                setAskReanalyze(false)
                videoRef.current?.pause()
                chooseFixedCam(!fixedCam)
              }}
              className="flex-1 h-11 rounded-full bg-primary text-black text-sm font-semibold"
            >
              Analyse again
            </button>
          </div>
        </Sheet>
      )}

      {askLeave && (
        <Sheet onDismiss={() => setAskLeave(false)}>
          <p className="text-white text-[15px] font-semibold mb-1">Leave the video editor?</p>
          <p className="text-white/60 text-[12px] mb-4">Your edits haven't been exported and will be lost.</p>
          <div className="flex gap-3">
            <button onClick={() => setAskLeave(false)} className="flex-1 h-11 rounded-full bg-white/10 text-white text-sm font-semibold">
              Keep editing
            </button>
            <button onClick={onClose} className="flex-1 h-11 rounded-full bg-danger text-white text-sm font-semibold">
              Leave
            </button>
          </div>
        </Sheet>
      )}

      {ready && !anyFaces && !panel && (
        <p className="absolute left-1/2 -translate-x-1/2 z-10 text-xs flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md text-white/85" style={{ bottom: (bottomH.current || bottomBox.height) + 12 }}>
          <IconInfo className="w-3.5 h-3.5" />
          No faces found in this video.
        </p>
      )}
    </div>
  )
}
