import { useEffect, useRef, useState } from 'react'
import { extFor, saveBlob, shareBlob, timestampedName } from '../lib/save'
import { IconCheck, IconClose, IconDownload, IconPause, IconPlay, IconShare, IconSpinner, IconWand } from './icons'

// A finished video, played back before anything is saved: watch it first,
// then save it, share it, take it into the editor, or throw it away. Shown
// right after recording and again after exporting from the editor.

function fmt(s: number) {
  if (!Number.isFinite(s)) return '0:00'
  const t = Math.max(0, Math.floor(s))
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`
}

export default function VideoReview({
  blob,
  onClose,
  closeLabel = 'Discard video',
  confirmClose = true,
  onEdit,
  editLabel = 'Edit',
  notice,
}: {
  blob: Blob
  onClose: () => void
  closeLabel?: string
  /** Ask before throwing away a video that was never saved. */
  confirmClose?: boolean
  onEdit?: () => void
  editLabel?: string
  notice?: string
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  // What MediaRecorder writes as WebM has no duration or index: players
  // show a duration of Infinity and can't seek it. A repackaged copy (the
  // same frames and sound, no re-encode) fixes both; it's ready in a blink.
  const [playable, setPlayable] = useState<Blob | null>(/webm/i.test(blob.type) ? null : blob)
  useEffect(() => {
    if (playable) return
    let live = true
    import('../lib/video/source').then(({ ensureSeekable }) => ensureSeekable(blob)).then((b) => live && setPlayable(b), () => live && setPlayable(blob))
    return () => {
      live = false
    }
  }, [blob, playable])
  // What's saved and shared: a standard MP4 (H.264 + AAC, index up front),
  // which WeChat and other chat apps show as a playable video, not a file.
  // Made in the background as soon as the review opens — iOS only opens
  // the share sheet straight from a tap, so it has to be ready by then.
  const [shareFile, setShareFile] = useState<Blob | null>(null)
  useEffect(() => {
    let live = true
    import('../lib/video/source').then(({ toShareableMp4 }) => toShareableMp4(blob)).then((b) => live && setShareFile(b), () => live && setShareFile(blob))
    return () => {
      live = false
    }
  }, [blob])
  // (Made and revoked in one effect: a URL made during render would be
  // revoked by React's dev-mode effect double run and never remade.)
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!playable) return
    const u = URL.createObjectURL(playable)
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setUrl(u)
    return () => URL.revokeObjectURL(u)
  }, [playable])
  const [playing, setPlaying] = useState(false)
  const [muted, setMuted] = useState(true)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [saved, setSaved] = useState(false)
  const [askDiscard, setAskDiscard] = useState(false)
  const [toast, setToast] = useState<{ key: number; text: string } | null>(null)
  const stamp = useRef(timestampedName('x').replace(/\.x$/, ''))
  const nameFor = (b: Blob) => `${stamp.current}.${extFor(b)}`



  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 1800)
    return () => clearTimeout(t)
  }, [toast])

  const toggle = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) v.play().catch(() => {})
    else v.pause()
  }

  const save = async () => {
    if (!shareFile) return
    const r = await saveBlob(shareFile, nameFor(shareFile))
    if (r !== 'dismissed') {
      setSaved(true)
      if (r === 'saved') setToast({ key: Date.now(), text: 'Video saved' })
    }
  }
  const share = async () => {
    if (!shareFile) return
    const r = await shareBlob(shareFile, nameFor(shareFile))
    if (r !== 'dismissed') setSaved(true)
    if (r === 'saved') setToast({ key: Date.now(), text: 'Sharing unavailable — saved instead' })
  }
  const close = () => (confirmClose && !saved ? setAskDiscard(true) : onClose())

  return (
    <div className="fixed inset-0 z-50 bg-black flex flex-col select-none">
      <div className="relative flex-1 min-h-0 flex items-center justify-center" onClick={toggle}>
        {url && (
          <video
            ref={videoRef}
            src={url}
            autoPlay
            loop
            playsInline
            muted={muted}
            className="max-w-full max-h-full"
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
            onDurationChange={(e) => setDuration(e.currentTarget.duration)}
          />
        )}
        {!playing && (
          <span className="absolute w-16 h-16 rounded-full bg-black/45 backdrop-blur-md flex items-center justify-center text-white pointer-events-none">
            <IconPlay className="w-8 h-8 ml-1" />
          </span>
        )}
      </div>

      <div className="absolute inset-x-0 flex items-center justify-between px-3" style={{ top: 'max(0.75rem, env(safe-area-inset-top))' }}>
        <button aria-label={closeLabel} onClick={close} className="h-10 w-10 rounded-full bg-black/45 backdrop-blur-md text-white flex items-center justify-center">
          <IconClose className="w-5 h-5" />
        </button>
        <button
          aria-label={muted ? 'Sound on' : 'Sound off'}
          onClick={() => setMuted((m) => !m)}
          className={`h-10 px-3.5 rounded-full backdrop-blur-md text-[11px] font-semibold ${muted ? 'bg-black/45 text-white' : 'bg-primary text-black'}`}
        >
          {muted ? 'Tap for sound' : 'Sound on'}
        </button>
      </div>

      <div className="bg-black/70 backdrop-blur-xl border-t border-white/10 px-4 pt-3" style={{ paddingBottom: 'max(1rem, env(safe-area-inset-bottom))' }}>
        {notice && <p className="text-[11px] text-white/60 mb-2.5 text-center">{notice}</p>}
        <div className="flex items-center gap-3 mb-4">
          <button aria-label={playing ? 'Pause' : 'Play'} onClick={toggle} className="text-white">
            {playing ? <IconPause className="w-5 h-5" /> : <IconPlay className="w-5 h-5" />}
          </button>
          <input
            type="range"
            min={0}
            max={Number.isFinite(duration) && duration > 0 ? duration : 1}
            step={0.01}
            value={time}
            aria-label="Position"
            onChange={(e) => {
              const v = videoRef.current
              if (v) v.currentTime = Number(e.target.value)
            }}
            className="flex-1 accent-primary"
          />
          <span className="text-[11px] text-white/70 tabular-nums w-[74px] text-right">
            {fmt(time)} / {fmt(duration)}
          </span>
        </div>
        <div className="flex items-center justify-center gap-3">
          {onEdit && (
            <button onClick={onEdit} className="h-12 px-5 rounded-full bg-white/10 text-white text-sm font-semibold flex items-center gap-2">
              <IconWand className="w-[18px] h-[18px]" />
              {editLabel}
            </button>
          )}
          <button onClick={save} disabled={!shareFile} className="h-12 px-6 rounded-full bg-primary text-black text-sm font-semibold flex items-center gap-2 shadow-glow disabled:opacity-60">
            {!shareFile ? <IconSpinner className="w-[18px] h-[18px] animate-spin" /> : saved ? <IconCheck className="w-[18px] h-[18px]" /> : <IconDownload className="w-[18px] h-[18px]" />}
            {!shareFile ? 'Preparing…' : saved ? 'Saved' : 'Save'}
          </button>
          <button aria-label="Share" onClick={share} disabled={!shareFile} className="h-12 w-12 rounded-full bg-white/10 text-white flex items-center justify-center disabled:opacity-40">
            <IconShare className="w-5 h-5" />
          </button>
        </div>
      </div>

      {toast && (
        <div key={toast.key} className="fg-toast absolute left-1/2 z-30 flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-black/75 backdrop-blur-md text-xs font-medium text-white" style={{ top: '42%' }}>
          <IconCheck className="w-3.5 h-3.5 text-primary" />
          {toast.text}
        </div>
      )}

      {askDiscard && (
        <div className="absolute inset-0 z-40 bg-black/60 flex items-end justify-center" onClick={() => setAskDiscard(false)}>
          <div className="fg-panel w-full max-w-md rounded-t-[22px] bg-surface border-t border-white/10 px-5 pt-5" style={{ paddingBottom: 'max(1.25rem, env(safe-area-inset-bottom))' }} onClick={(e) => e.stopPropagation()}>
            <p className="text-white text-[15px] font-semibold mb-1">Discard this video?</p>
            <p className="text-white/60 text-[12px] mb-4">It hasn't been saved. This can't be undone.</p>
            <div className="flex gap-3">
              <button onClick={() => setAskDiscard(false)} className="flex-1 h-11 rounded-full bg-white/10 text-white text-sm font-semibold">
                Keep
              </button>
              <button onClick={onClose} className="flex-1 h-11 rounded-full bg-danger text-white text-sm font-semibold">
                Discard
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
