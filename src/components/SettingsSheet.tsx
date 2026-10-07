import { useEffect } from 'react'
import { ThemePicker } from './ThemePicker'
import OtherApps from './OtherApps'
import ShareAppButton from './ShareAppButton'
import { IconClose } from './icons'

// Settings: the family's sheet — app theme, share this app, the other Flux
// apps — plus Glow's privacy note.
export default function SettingsSheet({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div data-keep-panel className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 backdrop-blur-sm sm:items-center sm:p-6" onClick={onClose} role="dialog" aria-modal aria-label="Settings">
      <div className="glass-strong fg-panel w-full max-w-lg max-h-[92dvh] overflow-y-auto no-scrollbar space-y-7 rounded-t-[28px] p-6 pb-[max(1.5rem,env(safe-area-inset-bottom))] sm:rounded-[28px]" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white">Settings</h2>
          <button onClick={onClose} aria-label="Close" className="rounded-full p-2 text-muted transition hover:bg-white/10 hover:text-white active:scale-95">
            <IconClose className="h-5 w-5" />
          </button>
        </div>
        <ThemePicker />
        <ShareAppButton />
        <OtherApps />
        <p className="text-xs leading-relaxed text-muted">Everything runs on this device: photos and videos never leave your browser.</p>
      </div>
    </div>
  )
}
