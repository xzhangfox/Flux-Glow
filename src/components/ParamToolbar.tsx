import { useEffect, useRef, useState, type ReactNode, type ComponentType, type SVGProps } from 'react'

export interface ParamDef {
  id: string
  icon: ComponentType<SVGProps<SVGSVGElement>>
  label: string
  /** Shows a small dot on the icon when the control isn't at its default
   *  (a quick "this one's been touched" cue, same idea Meitu's own
   *  bottom toolbar uses). */
  isActive: boolean
  render: () => ReactNode
}

export interface ActionDef {
  id: string
  icon: ComponentType<SVGProps<SVGSVGElement>>
  label: string
  onClick: () => void
  /** 'primary' gets the gold fill (the one emphasized action — Confirm,
   *  or Download once confirmed); everything else is the plain outline. */
  variant?: 'primary' | 'secondary'
  disabled?: boolean
}

const paramButtonClass =
  'relative flex-shrink-0 flex flex-col items-center gap-1 px-4 py-2.5 rounded-xl transition-colors border'

/**
 * A bottom icon toolbar, horizontally scrollable once there are more
 * icons than fit: tapping one opens a small panel above the bar (a
 * slider, a filter grid, whatever `render` returns); tapping the same
 * icon again, or anywhere outside the bar/panel, closes it. `actions`
 * renders a pinned, non-scrolling group of plain-click icon buttons
 * (Confirm, Download, Share, …) at the right end, in the same visual
 * language as the param icons.
 */
export default function ParamToolbar({ params, actions }: { params: ParamDef[]; actions?: ActionDef[] }) {
  const [openId, setOpenId] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // 'click' (fires after mouseup, once the click target is already fixed)
    // rather than 'pointerdown' — closing the panel on pointerdown shifts
    // the layout *before* the click resolves, which can make a tap on a
    // button just below the panel (e.g. Confirm) miss its target.
    function onClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpenId(null)
      }
    }
    document.addEventListener('click', onClick)
    return () => document.removeEventListener('click', onClick)
  }, [])

  const open = params.find((p) => p.id === openId)

  return (
    <div ref={containerRef} className="w-full">
      <div
        className={`grid transition-all duration-250 ease-out ${open ? 'grid-rows-[1fr] opacity-100 mb-3' : 'grid-rows-[0fr] opacity-0'}`}
        style={{ transitionProperty: 'grid-template-rows, opacity, margin' }}
      >
        <div className="overflow-hidden">
          <div className="bg-surface rounded-2xl p-4 border border-white/10">{open?.render()}</div>
        </div>
      </div>

      <div className="flex items-stretch gap-2">
        <div className="flex gap-2 overflow-x-auto no-scrollbar px-1 py-1 -mx-1 flex-1 min-w-0">
          {params.map((p) => {
            const isOpen = openId === p.id
            const Icon = p.icon
            return (
              <button
                key={p.id}
                onClick={() => setOpenId(isOpen ? null : p.id)}
                className={`${paramButtonClass} ${
                  isOpen ? 'bg-primary text-black border-primary' : 'bg-surface text-white/70 border-white/10 hover:bg-white/10'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="text-[10px] font-medium leading-none whitespace-nowrap">{p.label}</span>
                {p.isActive && !isOpen && <span className="absolute top-1.5 right-2.5 w-1.5 h-1.5 rounded-full bg-primary" />}
              </button>
            )
          })}
        </div>

        {actions && actions.length > 0 && (
          <div className="flex gap-2 flex-shrink-0 py-1 pl-2 border-l border-white/10">
            {actions.map((a) => {
              const Icon = a.icon
              return (
                <button
                  key={a.id}
                  onClick={a.onClick}
                  disabled={a.disabled}
                  className={`${paramButtonClass} disabled:opacity-40 ${
                    a.variant === 'primary'
                      ? 'bg-primary text-black border-primary hover:brightness-110'
                      : 'bg-surface text-white/70 border-white/10 hover:bg-white/10'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="text-[10px] font-medium leading-none whitespace-nowrap">{a.label}</span>
                </button>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
