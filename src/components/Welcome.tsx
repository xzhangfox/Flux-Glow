import { useEffect, useLayoutEffect, useRef, useState, type ComponentType, type CSSProperties, type PointerEvent, type SVGProps } from 'react'
import InfiniteMenu, { type InfiniteMenuItem } from './InfiniteMenu'
import SettingsSheet from './SettingsSheet'
import type { EditorStart } from './Editor'
import { IconCamera, IconImage, IconSettings } from './icons'

// The opening screen: a slowly turning sphere of travel photographs over a
// glow that takes on the colours of the one in front, a line about Glow,
// and the two ways in, Camera and Edit. The photos are only there to look
// at (drag to spin them; tapping one does nothing). They're openly licensed:
// see public/welcome/travel/CREDITS.md.

type Photo = InfiniteMenuItem & { place: string; by?: string; glow: [string, string] }

const photo = (file: string, place: string, glow: [string, string], by?: string): Photo => ({ image: `/welcome/travel/${file}.webp`, place, by, glow })

// (Warm and cool alternate, so the glow keeps changing as it turns.)
const PHOTOS: Photo[] = [
  photo('antelope-canyon', 'Antelope Canyon, Arizona', ['#e8834a', '#8a3f5c'], "Ashim D'Silva"),
  photo('ocean-above', 'The sea from above', ['#1f9fd1', '#0b5f8a'], 'Nattu Adnan'),
  photo('monument-valley', 'Monument Valley', ['#d9764a', '#2f6f9e'], 'Jasper van der Meij'),
  photo('peak-dusk', 'A granite peak at dusk', ['#3c6a8f', '#8a7fb0']),
  photo('pier-sunset', 'A pier at sunset', ['#e98e6f', '#3d5a7a']),
  photo('lake-range', 'Mountains over the lake', ['#4fa3c9', '#8da046'], 'Peter Thomas'),
  photo('grand-canyon', 'Grand Canyon', ['#c76f7e', '#5b3f86'], 'Jad Limcaco'),
  photo('dune-stars', 'Dunes under the stars', ['#2a5aa0', '#d39a6e'], 'Jared Evans'),
  photo('yosemite-valley', 'Yosemite Valley, California', ['#c26a3a', '#5b7f95'], 'Aniket Deole'),
  photo('star-trails', 'Star trails', ['#2b4a7d', '#7a5aa8'], 'Ahmadreza Sajadi'),
  photo('dune-footprints', 'Footprints in the dunes', ['#d8b48c', '#7fa3b8'], 'David Emrich'),
  photo('mirror-lake', 'A mirror of a lake', ['#c9855a', '#6aa3c0'], 'Sean Afnan'),
  photo('dolomites', 'Canazei, the Dolomites', ['#a8805f', '#5f7f9a'], 'Benjamin Voros'),
  photo('colorado-sign', 'Colorado', ['#4a9ad0', '#c48a5a'], 'Kait Herzog'),
]

function hasWebGL2() {
  try {
    const gl = document.createElement('canvas').getContext('webgl2')
    gl?.getExtension('WEBGL_lose_context')?.loseContext()
    return !!gl
  } catch {
    return false
  }
}

/** The sphere's zoom for its box (the front picture's half-width is
 *  0.23 × zoom × the box's shorter side): far enough out that the globe of
 *  photos shows round it, and clear of the header on a short phone. */
function zoomFor(width: number, height: number) {
  if (!width || !height) return 1.2
  const fit = width <= height ? (height / 2 - 72) / 1.15 / (0.23 * width) : (width / 2 - 200) / (0.23 * height)
  return Math.min(width <= height ? 1.3 : 1.05, Math.max(0.85, fit))
}

/** One of the two ways in: a capsule that rises in, follows the finger with
 *  a soft light, and springs back when pressed. */
function ModeButton({ icon: Icon, title, hint, hintLong, primary, delay, onClick }: { icon: ComponentType<SVGProps<SVGSVGElement>>; title: string; hint: string; hintLong: string; primary?: boolean; delay: number; onClick: () => void }) {
  const follow = (e: PointerEvent<HTMLButtonElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    e.currentTarget.style.setProperty('--mx', `${((e.clientX - r.left) / r.width) * 100}%`)
    e.currentTarget.style.setProperty('--my', `${((e.clientY - r.top) / r.height) * 100}%`)
  }
  return (
    <button onClick={onClick} onPointerMove={follow} onPointerDown={follow} className={`fg-mode ${primary ? 'fg-mode-primary' : 'fg-mode-soft'}`} style={{ animationDelay: `${delay}ms` } as CSSProperties}>
      <span className="fg-mode-icon">
        <Icon className="h-[22px] w-[22px]" />
      </span>
      <span className="min-w-0 text-left">
        <span className="block text-[15px] font-bold leading-tight lg:text-[17px]">{title}</span>
        <span className="fg-mode-hint block truncate text-[11px] leading-snug lg:text-[13px]">
          <span className="lg:hidden">{hint}</span>
          <span className="hidden lg:inline">{hintLong}</span>
        </span>
      </span>
    </button>
  )
}

/** Without WebGL 2 (old browsers): the same photos, one at a time. */
function Slides({ onShow }: { onShow: (i: number) => void }) {
  const [i, setI] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setI((n) => (n + 1) % PHOTOS.length), 3600)
    return () => clearInterval(t)
  }, [])
  useEffect(() => onShow(i), [i, onShow])
  return (
    <div className="absolute inset-0 flex items-center justify-center px-8 pt-14">
      <div className="aspect-square w-full max-w-[min(68vw,46vh)] overflow-hidden rounded-[32px] shadow-[0_30px_80px_-20px_rgba(0,0,0,0.8)]">
        <img key={i} src={PHOTOS[i].image} alt="" className="fg-photo-in h-full w-full object-cover" />
      </div>
    </div>
  )
}

export default function Welcome({ onCamera, onEdit, onSample }: { onCamera: (start?: EditorStart) => void; onEdit: (file: File, start?: EditorStart) => void; onSample: () => void }) {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [webgl] = useState(hasWebGL2)
  const [front, setFront] = useState(0)
  const boxRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ width: 0, height: 0 })
  const fileRef = useRef<HTMLInputElement>(null)

  useLayoutEffect(() => {
    const el = boxRef.current
    if (!el) return
    const update = () => setBox((b) => (b.width === el.clientWidth && b.height === el.clientHeight ? b : { width: el.clientWidth, height: el.clientHeight }))
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // (The picker has to open within the tap itself, or browsers block it.)
  const pick = () => {
    const input = fileRef.current
    if (!input) return
    input.accept = 'image/*,video/*'
    input.click()
  }

  const shown = PHOTOS[front]
  return (
    <div className="fg-welcome fixed inset-0 flex flex-col overflow-hidden select-none lg:flex-row" style={{ '--glow-a': shown.glow[0], '--glow-b': shown.glow[1] } as CSSProperties}>
      {/* The glow: the front photo's colours and the theme's, drifting. */}
      <div className="fg-aurora" aria-hidden>
        <i className="fg-aurora-a" />
        <i className="fg-aurora-b" />
        <i className="fg-aurora-c" />
        <i className="fg-sparkle" />
        <i className="fg-sparkle fg-sparkle-late" />
      </div>

      <div className="relative min-h-0 min-w-0 flex-1">
        <div ref={boxRef} className="absolute inset-0 [mask-image:linear-gradient(to_bottom,#000_86%,transparent)] lg:[mask-image:linear-gradient(to_right,#000_84%,transparent)]">
          {/* (a halo behind the photo in front, in its colours) */}
          <i className="fg-halo" style={{ width: Math.min(box.width, box.height) * 0.95, height: Math.min(box.width, box.height) * 0.95 }} aria-hidden />
          {webgl ? (
            <InfiniteMenu
              items={PHOTOS}
              count={160}
              tileSize={0.63}
              roundness={0.22}
              zoom={zoomFor(box.width, box.height)}
              pullBack={1}
              stretch={0.6}
              inertia={0.6}
              autoplay={3}
              grayscale={false}
              dim={0.3}
              intro
              theme="dark"
              onActiveChange={(_, i) => setFront(i)}
            />
          ) : (
            <Slides onShow={setFront} />
          )}
        </div>
        <header className="pointer-events-none absolute inset-x-0 top-0 flex items-center gap-2.5 bg-gradient-to-b from-black/50 to-transparent px-4 pb-8 lg:px-8" style={{ paddingTop: 'max(0.85rem, env(safe-area-inset-top))' }}>
          <img src="/favicon.svg" alt="" className="fg-rise h-9 w-9" />
          <div className="fg-rise" style={{ animationDelay: '80ms' }}>
            <p className="text-[15px] font-bold leading-tight text-white">Flux Glow</p>
            <p className="text-[11px] leading-tight text-white/60">Beauty camera · private by design</p>
          </div>
        </header>
      </div>

      <aside className="relative shrink-0 px-5 pt-1 lg:flex lg:w-[480px] lg:flex-col lg:justify-center lg:pl-6 lg:pr-12" style={{ paddingBottom: 'max(1rem, env(safe-area-inset-bottom))' }}>
        <p key={front} className="fg-caption flex items-center gap-1.5 text-[11px] text-white/70 lg:text-xs">
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 shrink-0 text-primary" fill="currentColor" aria-hidden>
            <path d="M12 2a7 7 0 0 0-7 7c0 5.2 7 13 7 13s7-7.8 7-13a7 7 0 0 0-7-7Zm0 9.5A2.5 2.5 0 1 1 12 6.5a2.5 2.5 0 0 1 0 5Z" />
          </svg>
          <span className="truncate">
            {shown.place}
            {shown.by && <span className="text-white/45"> · photo by {shown.by}</span>}
          </span>
        </p>
        <h1 className="fg-rise mt-2 text-[30px] font-extrabold leading-[1.02] tracking-[-0.02em] text-white min-[400px]:text-[34px] lg:mt-3 lg:text-[48px]" style={{ animationDelay: '120ms' }}>
          Glow,
          <span className="fg-serif fg-shimmer block pb-1 text-[36px] font-semibold tracking-[-0.01em] min-[400px]:text-[40px] lg:text-[58px]">wherever you go.</span>
        </h1>
        <p className="fg-rise mt-1 hidden text-[15px] leading-relaxed text-white/65 lg:block" style={{ animationDelay: '200ms' }}>
          A beauty camera and photo studio that does everything on your device. Shoot it glowing, or fix it after.
        </p>
        <div className="mt-4 grid grid-cols-2 gap-2.5 lg:mt-8 lg:grid-cols-1 lg:gap-3.5">
          <ModeButton primary icon={IconCamera} title="Camera" hint="Live retouch" hintLong="Shoot with live retouch" delay={260} onClick={() => onCamera()} />
          <ModeButton icon={IconImage} title="Edit" hint="Photo or video" hintLong="Retouch a photo or video" delay={360} onClick={pick} />
        </div>
        <p className="fg-rise mt-3.5 text-center text-[11px] text-white/55 lg:mt-6 lg:text-left lg:text-xs" style={{ animationDelay: '460ms' }}>
          Nothing leaves this device.{' '}
          <button onClick={onSample} className="font-semibold text-primary underline-offset-2 hover:underline">
            Try a sample photo
          </button>
        </p>
      </aside>

      <button
        onClick={() => setSettingsOpen(true)}
        aria-label="Settings"
        className="absolute right-3 z-20 flex h-10 w-10 items-center justify-center rounded-full bg-black/40 text-white transition active:scale-95"
        style={{ top: 'max(0.75rem, env(safe-area-inset-top))' }}
      >
        <IconSettings className="h-5 w-5" />
      </button>

      <input
        ref={fileRef}
        type="file"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          e.target.value = ''
          if (file) onEdit(file)
        }}
      />
      {settingsOpen && <SettingsSheet onClose={() => setSettingsOpen(false)} />}
    </div>
  )
}
