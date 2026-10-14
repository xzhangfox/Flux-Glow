import { useEffect, useLayoutEffect, useRef, useState, type ComponentType, type SVGProps } from 'react'
import InfiniteMenu, { type InfiniteMenuItem } from './InfiniteMenu'
import SettingsSheet from './SettingsSheet'
import { useThemeColors } from '../context/ThemeContext'
import type { EditorStart } from './Editor'
import { IconBack, IconCamera, IconImage, IconSettings } from './icons'

// The opening screen: a sphere of what Glow does — screens from the app and
// photos it made, the one in front in color — beside the two ways in,
// Camera and Edit. Each picture is also a door: tapping the one in front
// (or its button) opens the camera with that very look, filter or effect
// already on, or the photo picker for the things done to a photo or video.

type Go = { mode: 'camera'; start?: EditorStart } | { mode: 'edit'; accept: string; start?: EditorStart }
type Feature = InfiniteMenuItem & { title: string; description: string; go: Go }

const camera = (start?: EditorStart): Go => ({ mode: 'camera', start })
const effect = (effectId: string): Go => camera({ panel: 'effects', params: { effectId } })
const filter = (filterId: string): Go => camera({ panel: 'filter', params: { filterId } })

// (Descriptions stay under ~40 characters: on a phone they get one line.)
const FEATURES: Feature[] = [
  { image: '/welcome/screen-camera.webp', title: 'Beauty camera', description: 'Retouched live as you shoot. Hold to film.', go: camera() },
  { image: '/welcome/screen-camera-fox.webp', title: 'Live effects', description: 'Heads, masks and shades that follow you.', go: effect('foxhead') },
  { image: '/welcome/screen-looks.webp', title: 'Looks', description: 'Baby Face, Chiseled, Glam: one tap each.', go: camera({ panel: 'looks' }) },
  { image: '/welcome/screen-beauty.webp', title: 'Beauty', description: 'Smooth, brighten, clear blemishes, relight.', go: camera({ panel: 'beauty' }) },
  { image: '/welcome/screen-shape.webp', title: 'Shape', description: 'Jaw, eyes, brows, nose, lips: 26 sliders.', go: camera({ panel: 'shape' }) },
  { image: '/welcome/screen-filter.webp', title: 'Filters', description: 'Ten graded looks, previewed on your face.', go: camera({ panel: 'filter' }) },
  { image: '/welcome/screen-stickers.webp', title: 'Stickers', description: 'Art, emoji and styled text on a photo.', go: { mode: 'edit', accept: 'image/*', start: { panel: 'stickers' } } },
  { image: '/welcome/screen-video.webp', title: 'Video', description: 'Every face in a clip, tracked and retouched.', go: { mode: 'edit', accept: 'video/*' } },
  { image: '/welcome/compare.webp', title: 'Before / after', description: 'Hold compare to see the original.', go: camera({ panel: 'beauty' }) },
  { image: '/welcome/huskyhead.webp', title: 'Husky Head', description: 'A 3D head that blinks when you do.', go: effect('huskyhead') },
  { image: '/welcome/shibahead.webp', title: 'Shiba Head', description: 'Turn, tilt, smile: it follows along.', go: effect('shibahead') },
  { image: '/welcome/spider.webp', title: 'Spider', description: 'A full-head mask, hair tucked away.', go: effect('spider') },
  { image: '/welcome/bat.webp', title: 'Bat', description: 'The cowl, fitted to your face.', go: effect('bat') },
  { image: '/welcome/lavender.webp', title: 'ArtPapi', description: 'Lilac bob with matching eyes and lips.', go: effect('lavender') },
  { image: '/welcome/glam.webp', title: 'ArtPop', description: 'Drawn-on lashes and glossy red lips.', go: effect('glam') },
  { image: '/welcome/sport.webp', title: 'Sport', description: 'Wraparound shades that sit on your nose.', go: effect('sport') },
  { image: '/welcome/faun.webp', title: 'Faun', description: 'Curled horns and forest markings.', go: effect('faun') },
  { image: '/welcome/kitty.webp', title: 'Kitty', description: 'Ears and whiskers, tracked live.', go: effect('kitty') },
  { image: '/welcome/bunny.webp', title: 'Bunny', description: 'Bunny ears that move with you.', go: effect('bunny') },
  { image: '/welcome/fox.webp', title: 'Fox', description: 'Fox ears and a button nose.', go: effect('fox') },
  { image: '/welcome/peach.webp', title: 'Peach', description: 'A warm, rosy filter made for skin.', go: filter('peach') },
  { image: '/welcome/film.webp', title: 'Film', description: 'Faded blacks and a soft film tone.', go: filter('film') },
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
 *  0.23 × zoom × the box's shorter side): large on a phone, but clear of the
 *  header on a short one; on a wide screen, small enough that the title and
 *  description fit beside it. */
function zoomFor(width: number, height: number) {
  if (!width || !height) return 1.65
  const fit = width <= height ? (height / 2 - 96) / 1.15 / (0.23 * width) : (width / 2 - 270) / (0.23 * height)
  return Math.min(1.65, Math.max(0.9, fit))
}

function ModeButton({ icon: Icon, title, hint, inside, primary, onClick }: { icon: ComponentType<SVGProps<SVGSVGElement>>; title: string; hint: string; inside: string; primary?: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`group flex min-w-0 items-center gap-2.5 rounded-[22px] border p-3 text-left min-[400px]:gap-3 min-[400px]:p-3.5 transition active:scale-[0.98] lg:gap-4 lg:p-5 ${
        primary ? 'border-primary bg-primary text-black shadow-glow-strong' : 'glass-strong text-white hover:border-primary/50'
      }`}
    >
      <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full min-[400px]:h-11 min-[400px]:w-11 lg:h-12 lg:w-12 ${primary ? 'bg-black/15' : 'bg-white/10 text-primary'}`}>
        <Icon className="h-[22px] w-[22px]" />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-[15px] font-bold leading-tight lg:text-lg">{title}</span>
        <span className={`mt-0.5 block text-[11px] leading-snug lg:text-[13px] ${primary ? 'text-black/70' : 'text-muted'}`}>{hint}</span>
        <span className={`mt-2 hidden text-[10.5px] font-bold uppercase tracking-wider lg:block ${primary ? 'text-black/55' : 'text-primary/85'}`}>{inside}</span>
      </span>
      <IconBack className="hidden h-5 w-5 shrink-0 rotate-180 opacity-60 transition group-hover:translate-x-0.5 lg:block" />
    </button>
  )
}

/** Without WebGL 2 (old browsers): the same pictures, one at a time. */
function Slides({ onGo }: { onGo: (f: Feature) => void }) {
  const [i, setI] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setI((n) => (n + 1) % FEATURES.length), 3200)
    return () => clearInterval(t)
  }, [])
  const f = FEATURES[i]
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 px-6 pt-16">
      <button onClick={() => onGo(f)} className="aspect-square w-full max-w-[min(70vw,48vh)] overflow-hidden rounded-[28px] shadow-glow-strong" aria-label={`Try ${f.title}`}>
        <img key={f.image} src={f.image} alt="" className="fg-panel h-full w-full object-cover" />
      </button>
      <div className="text-center">
        <p className="text-xl font-semibold">{f.title}</p>
        <p className="text-sm text-muted">{f.description}</p>
      </div>
    </div>
  )
}

export default function Welcome({ onCamera, onEdit, onSample }: { onCamera: (start?: EditorStart) => void; onEdit: (file: File, start?: EditorStart) => void; onSample: () => void }) {
  const colors = useThemeColors()
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [webgl] = useState(hasWebGL2)
  const boxRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ width: 0, height: 0 })
  const fileRef = useRef<HTMLInputElement>(null)
  const pickStart = useRef<EditorStart | undefined>(undefined)

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
  const pick = (accept: string, start?: EditorStart) => {
    const input = fileRef.current
    if (!input) return
    pickStart.current = start
    input.accept = accept
    input.click()
  }
  const go = (f: Feature) => (f.go.mode === 'camera' ? onCamera(f.go.start) : pick(f.go.accept, f.go.start))

  return (
    <div className="fixed inset-0 flex flex-col overflow-hidden select-none lg:flex-row">
      <div className="relative min-h-0 min-w-0 flex-1">
        <div ref={boxRef} className="absolute inset-0">
          {webgl ? (
            <InfiniteMenu
              items={FEATURES}
              count={160}
              tileSize={0.63}
              roundness={0.2}
              zoom={zoomFor(box.width, box.height)}
              pullBack={1}
              stretch={0.6}
              inertia={0.6}
              autoplay={3}
              grayscale
              dim={0.5}
              intro
              showInfo
              theme="dark"
              accentColor={colors.primary}
              onItemClick={(_, index) => go(FEATURES[index])}
            />
          ) : (
            <Slides onGo={go} />
          )}
        </div>
        <header className="pointer-events-none absolute inset-x-0 top-0 flex items-center gap-2.5 bg-gradient-to-b from-black/55 to-transparent px-4 pb-6 lg:px-8" style={{ paddingTop: 'max(0.85rem, env(safe-area-inset-top))' }}>
          <img src="/favicon.svg" alt="" className="h-9 w-9" />
          <div>
            <p className="text-[15px] font-bold leading-tight text-white">Flux Glow</p>
            <p className="text-[11px] leading-tight text-muted">
              <span className="lg:hidden">Spin to explore · tap a picture to try it</span>
              <span className="hidden lg:inline">The beauty camera that never uploads a thing</span>
            </p>
          </div>
        </header>
      </div>

      <aside className="relative shrink-0 px-4 pt-2 lg:flex lg:w-[400px] lg:flex-col lg:justify-center lg:border-l lg:border-white/10 lg:bg-black/35 lg:px-8 lg:backdrop-blur-xl" style={{ paddingBottom: 'max(0.9rem, env(safe-area-inset-bottom))' }}>
        <div className="mb-8 hidden lg:block">
          <p className="text-[11px] font-bold uppercase tracking-[0.25em] text-primary">Choose a mode</p>
          <h1 className="mt-2 text-[28px] font-bold leading-tight text-white">Shoot it glowing, or fix it after.</h1>
          <p className="mt-2 text-sm leading-relaxed text-muted">Spin the sphere to see what's inside. Tap the picture in front to try that one straight away.</p>
        </div>
        <div className="grid grid-cols-2 gap-2.5 lg:grid-cols-1 lg:gap-3">
          <ModeButton primary icon={IconCamera} title="Camera" hint="Shoot with live retouch" inside="Looks · Beauty · Shape · Filters · Effects" onClick={() => onCamera()} />
          <ModeButton icon={IconImage} title="Edit" hint="Retouch a photo or video" inside="Photos · Videos · Stickers" onClick={() => pick('image/*,video/*')} />
        </div>
        <p className="mt-3 text-center text-[11px] text-muted lg:mt-6 lg:text-left">
          Everything stays on this device.{' '}
          <button onClick={onSample} className="font-semibold text-primary underline-offset-2 hover:underline">
            Try a sample photo
          </button>
        </p>
      </aside>

      <button
        onClick={() => setSettingsOpen(true)}
        aria-label="Settings"
        className="absolute right-3 z-20 flex h-10 w-10 items-center justify-center rounded-full bg-black/40 text-white backdrop-blur-md transition active:scale-95"
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
          if (file) onEdit(file, pickStart.current)
        }}
      />
      {settingsOpen && <SettingsSheet onClose={() => setSettingsOpen(false)} />}
    </div>
  )
}
