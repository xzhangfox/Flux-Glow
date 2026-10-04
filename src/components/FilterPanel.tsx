import { useRef } from 'react'
import { FILTER_PRESETS } from '../lib/filters'
import { FADE_RIGHT_STYLE, useScrollFade } from './useScrollFade'
import Slider from './Slider'
import { IconSpinner } from './icons'

// A thumbnail strip — each preset rendered on this very photo's face —
// instead of text-only chips, plus an intensity slider for the selected
// look. Both are table stakes in commercial camera apps: a name like
// "Film" says little about what it will do to a particular face, and a
// preset tuned to read clearly at full strength is usually too much.
export default function FilterPanel({
  thumbs,
  filterId,
  strength,
  defaultStrength,
  onSelect,
  onStrength,
}: {
  thumbs: Map<string, string> | null
  filterId: string
  strength: number
  defaultStrength: number
  onSelect: (id: string) => void
  onStrength: (v: number) => void
}) {
  const rowRef = useRef<HTMLDivElement>(null)
  const fadeRight = useScrollFade(rowRef)
  return (
    <div>
      <div className="flex items-center h-7 mb-3">
        <span className="text-[13px] font-semibold text-white">Filter</span>
      </div>
      <div ref={rowRef} className="flex gap-2.5 overflow-x-auto no-scrollbar -mx-1 px-1 pb-1 pt-1" style={fadeRight ? FADE_RIGHT_STYLE : undefined}>
        {FILTER_PRESETS.map((p) => {
          const selected = p.id === filterId
          const src = thumbs?.get(p.id)
          return (
            <button key={p.id} onClick={() => onSelect(p.id)} aria-label={`${p.label} filter`} aria-pressed={selected} className="flex flex-col items-center gap-1.5 flex-shrink-0">
              <span
                className={`w-14 h-14 rounded-xl overflow-hidden flex items-center justify-center bg-white/[0.06] transition ${
                  selected ? 'ring-2 ring-primary ring-offset-2 ring-offset-black/60' : 'ring-1 ring-white/10'
                }`}
              >
                {src ? <img src={src} alt="" className="w-full h-full object-cover" draggable={false} /> : <IconSpinner className="w-4 h-4 text-white/40 animate-spin" />}
              </span>
              <span className={`text-[10.5px] font-medium leading-none whitespace-nowrap ${selected ? 'text-primary' : 'text-white/75'}`}>{p.label}</span>
            </button>
          )
        })}
      </div>
      {filterId !== 'none' && (
        <div className="mt-3">
          <Slider label="Intensity" value={strength} defaultValue={defaultStrength} onChange={onStrength} />
        </div>
      )}
    </div>
  )
}
