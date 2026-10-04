import { IconRefresh } from './icons'

// `bidirectional`: the stored value is -1..1 with 0 (untouched) at the
// track's midpoint, and dragging either way moves the feature in opposite
// directions (narrower/wider). The filled part of the track runs from the
// neutral point to the thumb either way, so it's obvious at a glance how
// far — and which way — a control has been pushed. A tick marks the
// control's own default, and a revert button appears once it's moved off
// it: commercial beauty cameras all make "undo just this one" a single
// tap rather than asking someone to drag back to an exact number.
export default function Slider({
  label,
  value,
  defaultValue = 0,
  onChange,
  disabled,
  bidirectional,
}: {
  label: string
  value: number
  defaultValue?: number
  onChange: (v: number) => void
  disabled?: boolean
  bidirectional?: boolean
}) {
  const min = bidirectional ? -100 : 0
  const display = Math.round(value * 100)
  const pct = (v: number) => ((v * 100 - min) / (100 - min)) * 100
  const anchor = bidirectional ? 50 : 0
  const at = pct(value)
  const lo = Math.min(anchor, at)
  const hi = Math.max(anchor, at)
  const changed = Math.abs(value - defaultValue) > 0.005

  return (
    <div className={disabled ? 'opacity-40' : ''} onDoubleClick={() => !disabled && onChange(defaultValue)}>
      <div className="flex items-center justify-between mb-2 h-5">
        <span className="text-[13px] text-white/85">{label}</span>
        <div className="flex items-center gap-2">
          {changed && !disabled && (
            <button onClick={() => onChange(defaultValue)} aria-label={`Reset ${label}`} className="text-white/50 hover:text-white transition">
              <IconRefresh className="w-3.5 h-3.5" />
            </button>
          )}
          <span className="text-xs font-semibold tabular-nums text-primary w-9 text-right">
            {bidirectional && display > 0 ? '+' : ''}
            {display}
          </span>
        </div>
      </div>
      <div className="relative h-7 flex items-center">
        <div className="absolute inset-x-0 h-1 rounded-full bg-white/15" />
        <div className="absolute h-1 rounded-full bg-primary" style={{ left: `${lo}%`, width: `${hi - lo}%` }} />
        {(bidirectional || defaultValue !== 0) && (
          <div className="absolute w-0.5 h-2.5 rounded-full bg-white/40 -translate-x-1/2" style={{ left: `${pct(defaultValue)}%` }} />
        )}
        <input
          type="range"
          min={min}
          max={100}
          value={display}
          disabled={disabled}
          aria-label={label}
          onChange={(e) => onChange(Number(e.target.value) / 100)}
          className="fg-range relative w-full"
        />
      </div>
    </div>
  )
}
