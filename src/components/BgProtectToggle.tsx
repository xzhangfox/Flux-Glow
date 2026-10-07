// The background-protection switch shown while reshaping (Shape, Looks):
// on, only the person is warped and the background beside them stays put
// (see lib/bgProtect.ts).
export default function BgProtectToggle({ on, onChange, disabled }: { on: boolean; onChange: (on: boolean) => void; disabled?: boolean }) {
  return (
    <button
      role="switch"
      aria-checked={on}
      aria-label="Background protection"
      title="Keep the background beside you from bending when reshaping"
      onClick={() => onChange(!on)}
      disabled={disabled}
      className="flex items-center gap-1.5 h-7 pl-2.5 pr-1 rounded-full bg-white/[0.07] text-[11px] font-medium text-white/80 disabled:opacity-30 transition active:scale-95"
    >
      BG protect
      <span className={`relative w-7 h-[18px] rounded-full transition ${on ? 'bg-primary' : 'bg-white/20'}`}>
        <span className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white shadow transition-all ${on ? 'left-[12px]' : 'left-[2px]'}`} />
      </span>
    </button>
  )
}
