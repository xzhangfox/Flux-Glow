import { useState } from 'react'
import { IconCheck, IconShare } from './icons'
import { APP_URL, shareApp } from '../lib/shareApp'

// "Share this app" (see lib/shareApp.ts), for the Settings sheet.

export default function ShareAppButton() {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={async () => {
        if ((await shareApp()) === 'copied') {
          setCopied(true)
          setTimeout(() => setCopied(false), 1800)
        }
      }}
      className="glass flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-left transition hover:border-primary/50 active:scale-95"
    >
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary">
        {copied ? <IconCheck className="h-[18px] w-[18px]" /> : <IconShare className="h-[18px] w-[18px]" />}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-semibold text-white">{copied ? 'Link copied' : 'Share Flux Glow'}</span>
        <span className="block truncate text-[11px] text-muted">{APP_URL.replace('https://', '').replace(/\/$/, '')}</span>
      </span>
    </button>
  )
}
