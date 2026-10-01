import { useRef, useState } from 'react'

export default function UploadArea({ onImage }: { onImage: (file: File) => void }) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [cameraOpen, setCameraOpen] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const openCamera = async () => {
    setCameraError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
      streamRef.current = stream
      setCameraOpen(true)
      // The <video> element only mounts once cameraOpen flips true, so wait
      // a tick before attaching the stream to it.
      requestAnimationFrame(() => {
        if (videoRef.current) videoRef.current.srcObject = stream
      })
    } catch {
      setCameraError('Could not access the camera — check your browser permissions.')
    }
  }

  const closeCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setCameraOpen(false)
  }

  const capturePhoto = () => {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d')!.drawImage(video, 0, 0)
    canvas.toBlob((blob) => {
      if (blob) onImage(new File([blob], 'capture.jpg', { type: 'image/jpeg' }))
      closeCamera()
    }, 'image/jpeg', 0.95)
  }

  if (cameraOpen) {
    return (
      <div className="flex flex-col items-center gap-4">
        <div className="relative w-full max-w-md aspect-[3/4] rounded-2xl overflow-hidden bg-black border border-white/10">
          <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover -scale-x-100" />
        </div>
        {cameraError && <p className="text-danger text-sm">{cameraError}</p>}
        <div className="flex gap-3">
          <button
            onClick={capturePhoto}
            className="px-6 py-3 bg-primary text-black font-semibold rounded-xl hover:brightness-110 transition shadow-glow"
          >
            Capture
          </button>
          <button
            onClick={closeCamera}
            className="px-6 py-3 bg-secondary text-white rounded-xl hover:bg-white/10 transition"
          >
            Cancel
          </button>
        </div>
      </div>
    )
  }

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
      className={`flex flex-col items-center justify-center gap-5 w-full max-w-md aspect-[3/4] rounded-2xl border-2 border-dashed transition-colors ${
        dragOver ? 'border-primary bg-primary/5' : 'border-white/15 bg-surface'
      }`}
    >
      <span className="material-symbols-outlined text-5xl text-primary">add_a_photo</span>
      <div className="text-center px-6">
        <p className="text-white/90 font-medium">Drop a photo here</p>
        <p className="text-text-secondary text-sm mt-1">or choose one below</p>
      </div>
      <div className="flex gap-3">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="px-5 py-2.5 bg-primary text-black font-semibold rounded-xl hover:brightness-110 transition shadow-glow text-sm"
        >
          Upload Photo
        </button>
        <button
          onClick={openCamera}
          className="px-5 py-2.5 bg-secondary text-white rounded-xl hover:bg-white/10 transition text-sm flex items-center gap-1.5"
        >
          <span className="material-symbols-outlined text-base">photo_camera</span>
          Camera
        </button>
      </div>
      {cameraError && <p className="text-danger text-xs px-6 text-center">{cameraError}</p>}
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
