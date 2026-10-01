export default function Slider({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  disabled?: boolean
}) {
  return (
    <div className={disabled ? 'opacity-40' : ''}>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-sm text-white/80">{label}</span>
        <span className="text-xs font-mono text-primary">{Math.round(value * 100)}</span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={Math.round(value * 100)}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        className="w-full h-1.5 rounded-full appearance-none bg-secondary accent-primary cursor-pointer disabled:cursor-not-allowed"
      />
    </div>
  )
}
