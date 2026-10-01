import { useState } from 'react'
import UploadArea from './components/UploadArea'
import Editor from './components/Editor'

export default function App() {
  const [file, setFile] = useState<File | null>(null)

  return (
    <div className="min-h-screen bg-background flex flex-col items-center">
      <header className="w-full max-w-4xl px-6 py-6 flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg border border-primary/40 bg-primary/5 flex items-center justify-center">
          <span className="material-symbols-outlined text-primary text-xl">auto_awesome</span>
        </div>
        <div>
          <h1 className="text-lg font-bold text-white tracking-tight">Flux Glow</h1>
          <p className="text-xs text-text-secondary">Skin smoothing &amp; filters — all in your browser</p>
        </div>
      </header>

      <main className="flex-1 w-full flex items-center justify-center px-6 pb-16">
        {file ? <Editor file={file} onReset={() => setFile(null)} /> : <UploadArea onImage={setFile} />}
      </main>

      <footer className="pb-8 text-center">
        <p className="text-text-secondary text-xs">Photos never leave your device — no uploads, no account.</p>
      </footer>
    </div>
  )
}
