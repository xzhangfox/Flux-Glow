import { useRef, type ReactNode } from 'react'
import { FADE_RIGHT_STYLE, useScrollFade } from './useScrollFade'
import { IconSpinner } from './icons'

// A row of previews rendered on the user's own face (Looks, Effects), the
// same pattern as the filter strip.
export default function ThumbStrip({
  items,
  thumbs,
  selected,
  onSelect,
  fallback,
}: {
  items: { id: string; label: string; badge?: string }[]
  thumbs: Record<string, string>
  selected: string | null
  onSelect: (id: string) => void
  /** Rendered in place of a thumbnail (e.g. an icon for "None"). */
  fallback?: (id: string) => ReactNode | null
}) {
  const rowRef = useRef<HTMLDivElement>(null)
  const fadeRight = useScrollFade(rowRef)
  return (
    <div ref={rowRef} className="flex gap-2.5 overflow-x-auto no-scrollbar -mx-1 px-1 pb-1 pt-1" style={fadeRight ? FADE_RIGHT_STYLE : undefined}>
      {items.map((it) => {
        const on = it.id === selected
        const custom = fallback?.(it.id)
        const src = thumbs[it.id]
        return (
          <button key={it.id} onClick={() => onSelect(it.id)} aria-label={it.label} aria-pressed={on} className="flex flex-col items-center gap-1.5 flex-shrink-0">
            <span className={`relative w-14 h-14 rounded-xl overflow-hidden flex items-center justify-center bg-white/[0.06] transition ${on ? 'ring-2 ring-primary ring-offset-2 ring-offset-black/60' : 'ring-1 ring-white/10'}`}>
              {custom ?? (src ? <img src={src} alt="" className="w-full h-full object-cover" draggable={false} /> : <IconSpinner className="w-4 h-4 text-white/40 animate-spin" />)}
              {it.badge && <span className="absolute bottom-0.5 right-0.5 w-4 h-4 rounded-full bg-black/60 text-[9px] text-[#ff8fb1] flex items-center justify-center">{it.badge}</span>}
            </span>
            <span className={`text-[10.5px] font-medium leading-none whitespace-nowrap ${on ? 'text-primary' : 'text-white/75'}`}>{it.label}</span>
          </button>
        )
      })}
    </div>
  )
}
