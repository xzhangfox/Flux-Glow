// A settings row: what it does, and an on/off switch.
export default function SwitchRow({ label, hint, on, onChange }: { label: string; hint?: string; on: boolean; onChange: (on: boolean) => void }) {
  return (
    <button role="switch" aria-checked={on} aria-label={label} onClick={() => onChange(!on)} className="flex w-full items-center justify-between gap-4 text-left">
      <span>
        <span className="block text-sm font-semibold text-white">{label}</span>
        {hint && <span className="mt-0.5 block text-xs leading-relaxed text-muted">{hint}</span>}
      </span>
      <span className={`relative h-6 w-10 flex-shrink-0 rounded-full transition ${on ? 'bg-primary' : 'bg-white/20'}`}>
        <span className={`absolute top-[3px] h-[18px] w-[18px] rounded-full bg-white shadow transition-all ${on ? 'left-[19px]' : 'left-[3px]'}`} />
      </span>
    </button>
  )
}
