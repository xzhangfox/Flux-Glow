import { useRef, useState } from 'react'
import { IconSparkle, IconImage, IconCamera } from './icons'

export default function UploadArea({ onImage, onLive }: { onImage: (file: File) => void; onLive: () => void }) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragOver(false)
        const file = e.dataTransfer.files[0]
        if (file) onImage(file)
      }}
      className={`flex flex-col items-center justify-center gap-6 w-full max-w-md aspect-[3/4] rounded-3xl border-2 border-dashed transition-colors ${
        dragOver ? 'border-primary bg-primary/5' : 'border-white/15 bg-surface'
      }`}
    >
      <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/30 flex items-center justify-center shadow-glow">
        <IconSparkle className="w-8 h-8 text-primary" />
      </div>
      <div className="text-center px-6">
        <p className="text-white/90 font-medium text-lg">Drop a photo here</p>
        <p className="text-text-secondary text-sm mt-1">or pick an option below</p>
      </div>
      <div className="flex gap-3">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="px-5 py-3 bg-primary text-black font-semibold rounded-xl hover:brightness-110 transition shadow-glow text-sm flex items-center gap-2"
        >
          <IconImage className="w-4 h-4" />
          Upload Photo
        </button>
        <button
          onClick={onLive}
          className="px-5 py-3 bg-secondary text-white rounded-xl hover:bg-white/10 transition text-sm flex items-center gap-2 border border-white/10"
        >
          <IconCamera className="w-4 h-4" />
          Live Camera
        </button>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onImage(file)
        }}
      />
    </div>
  )
}
