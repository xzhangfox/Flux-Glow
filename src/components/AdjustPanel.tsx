import { useRef, useState, type ReactNode } from 'react'
import Slider from './Slider'
import { FADE_RIGHT_STYLE, useScrollFade } from './useScrollFade'
import { IconBack, IconRefresh } from './icons'

export interface AdjustItem {
  key: string
  label: string
  icon: ReactNode
  value?: number
  defaultValue?: number
  onChange?: (v: number) => void
  bidirectional?: boolean
  /** A group (e.g. Mouth) drills into its own sub-grid instead of a slider. */
  children?: AdjustItem[]
}

function isChanged(item: AdjustItem): boolean {
  if (item.children) return item.children.some(isChanged)
  return item.value !== undefined && Math.abs(item.value - (item.defaultValue ?? 0)) > 0.005
}

function resetItem(item: AdjustItem) {
  if (item.children) item.children.forEach(resetItem)
  else item.onChange?.(item.defaultValue ?? 0)
}

// One panel shape for Beauty and Shape alike: a horizontal row of round
// icon buttons, where tapping one swaps the row for that control's slider
// (or, for a group like Mouth, for its own sub-row). The header carries
// the way back up a level and a Reset that restores every control in the
// panel at once — so nobody has to hunt down which of a dozen sliders
// they nudged. Drill-down state is local: the parent unmounts the panel
// on close, so reopening always starts at the top level.
// `tabs`: the top-level items are categories (Face, Eyes, Nose…) shown as a
// chip row, with the chosen category's controls in the icon row below —
// one tap to any control instead of drilling through a group first.
// `extra`: a control in the header beside Reset (Shape's background
// protection), there at every level, sliders included.
export default function AdjustPanel({ title, items, disabled, tabs, extra }: { title: string; items: AdjustItem[]; disabled?: boolean; tabs?: boolean; extra?: ReactNode }) {
  const [path, setPath] = useState<string[]>([])
  const [tab, setTab] = useState(items[0]?.key)
  const rowRef = useRef<HTMLDivElement>(null)
  const fadeRight = useScrollFade(rowRef, [path.join('/'), tab])

  const tabItem = tabs ? items.find((i) => i.key === tab) ?? items[0] : null
  let level = tabItem?.children ?? items
  const trail: AdjustItem[] = tabItem ? [tabItem] : []
  for (const key of path) {
    const next = level.find((i) => i.key === key)
    if (!next) break
    trail.push(next)
    if (!next.children) break
    level = next.children
  }
  const current = trail[trail.length - 1]
  const showingSlider = current && !current.children
  const depth = trail.length - (tabItem ? 1 : 0)
  const anyChanged = items.some(isChanged)

  return (
    <div>
      <div className="flex items-center justify-between h-7 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          {depth > 0 && (
            <button
              onClick={() => setPath((p) => p.slice(0, -1))}
              aria-label="Back"
              className="w-7 h-7 -ml-1 rounded-full flex items-center justify-center text-white/80 hover:bg-white/10 transition"
            >
              <IconBack className="w-4 h-4" />
            </button>
          )}
          <span className="text-[13px] font-semibold text-white truncate">{[title, ...trail.slice(tabItem && !showingSlider ? 1 : 0).map((t) => t.label)].join(' · ')}</span>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          {extra}
          <button
            onClick={() => items.forEach(resetItem)}
            disabled={!anyChanged || disabled}
            className="flex items-center gap-1 text-[11px] font-medium text-white/70 hover:text-white disabled:opacity-30 transition"
          >
            <IconRefresh className="w-3.5 h-3.5" />
            Reset
          </button>
        </div>
      </div>

      {showingSlider ? (
        <Slider
          label={current.label}
          value={current.value ?? 0}
          defaultValue={current.defaultValue ?? 0}
          onChange={(v) => current.onChange?.(v)}
          bidirectional={current.bidirectional}
          disabled={disabled}
        />
      ) : (
        <>
        {tabs && (
          <div className="flex gap-1.5 mb-3 overflow-x-auto no-scrollbar -mx-1 px-1" role="tablist">
            {items.map((t) => {
              const on = t.key === tabItem?.key
              return (
                <button
                  key={t.key}
                  role="tab"
                  aria-selected={on}
                  onClick={() => {
                    setTab(t.key)
                    setPath([])
                  }}
                  className={`relative flex-shrink-0 h-7 px-3 rounded-full text-[11.5px] font-semibold transition ${on ? 'bg-primary text-black' : 'bg-white/[0.07] text-white/75'}`}
                >
                  {t.label}
                  {!on && isChanged(t) && <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-primary ring-2 ring-black/60" />}
                </button>
              )
            })}
          </div>
        )}
        <div ref={rowRef} className="flex gap-3 overflow-x-auto no-scrollbar -mx-1 px-1 pb-0.5" style={fadeRight ? FADE_RIGHT_STYLE : undefined}>
          {level.map((item) => {
            const changed = isChanged(item)
            return (
              <button
                key={item.key}
                onClick={() => setPath((p) => [...p, item.key])}
                aria-label={item.children ? `${item.label} options` : `Adjust ${item.label}`}
                className="flex flex-col items-center gap-1.5 flex-shrink-0 w-[3.6rem]"
              >
                <span
                  className={`relative w-12 h-12 rounded-full flex items-center justify-center border transition ${
                    changed ? 'bg-primary/15 border-primary/70 text-primary' : 'bg-white/[0.06] border-white/15 text-white/85'
                  }`}
                >
                  {item.icon}
                  {changed && <span className="absolute top-0.5 right-0.5 w-2 h-2 rounded-full bg-primary ring-2 ring-black/60" />}
                </span>
                <span className={`text-[10.5px] font-medium leading-none whitespace-nowrap ${changed ? 'text-primary' : 'text-white/80'}`}>{item.label}</span>
              </button>
            )
          })}
        </div>
        </>
      )}
    </div>
  )
}
