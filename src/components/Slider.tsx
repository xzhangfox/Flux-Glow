// `bidirectional` makes this a from-the-middle drag bar: the stored value
// is -1..1, 0 (the original, untouched state) sits at the track's own
// midpoint, and dragging left or right moves the feature the two opposite
// ways (e.g. narrower/wider) instead of a one-directional 0..1 intensity.
// A plain `<input type=range min=-100 max=100>` already centers a value of
// 0 at the midpoint on its own — no custom thumb math needed — so this
// just adds the center tick mark that makes that midpoint visible at rest.
export default function Slider({
  label,
  value,
  onChange,
  disabled,
  bidirectional,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  disabled?: boolean
  bidirectional?: boolean
}) {
  const min = bidirectional ? -100 : 0
  const displayValue = Math.round(value * 100)
  return (
    <div className={disabled ? 'opacity-40' : ''}>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-sm text-white/80">{label}</span>
        <span className="text-xs font-mono text-primary">{bidirectional && displayValue > 0 ? '+' : ''}{displayValue}</span>
      </div>
      <div className="relative">
        {bidirectional && <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-0.5 h-3 rounded-full bg-white/30 pointer-events-none" />}
        <input
          type="range"
          min={min}
          max={100}
          value={displayValue}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value) / 100)}
          className="relative w-full h-1.5 rounded-full appearance-none bg-secondary accent-primary cursor-pointer disabled:cursor-not-allowed"
        />
      </div>
    </div>
  )
}
