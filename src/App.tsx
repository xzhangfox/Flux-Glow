import { useState } from 'react'
import UploadArea from './components/UploadArea'
import Editor, { type Source } from './components/Editor'
import { IconFaceGlow } from './components/icons'

export default function App() {
  const [source, setSource] = useState<Source | null>(null)

  return (
    <div className="min-h-screen bg-background flex flex-col items-center relative overflow-hidden">
      <div
        className="pointer-events-none absolute top-0 left-1/2 -translate-x-1/2 w-[900px] h-[500px] opacity-20 blur-3xl"
        style={{ background: 'radial-gradient(ellipse at top, #D4AF37, transparent 70%)' }}
      />

      <header className="relative w-full max-w-4xl px-6 py-6 flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg border border-primary/40 bg-primary/5 flex items-center justify-center">
          <IconFaceGlow className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-white tracking-tight">Flux Glow</h1>
          <p className="text-xs text-text-secondary">Skin, contour &amp; filters — live or from a photo, all in your browser</p>
        </div>
      </header>

      <main className="relative flex-1 w-full flex items-center justify-center px-6 pb-16">
        {source ? (
          <Editor source={source} onReset={() => setSource(null)} />
        ) : (
          <UploadArea onImage={(file) => setSource({ kind: 'image', file })} onLive={() => setSource({ kind: 'live' })} />
        )}
      </main>

      <footer className="relative pb-8 text-center">
        <p className="text-text-secondary text-xs">Photos never leave your device — no uploads, no account.</p>
      </footer>
    </div>
  )
}
