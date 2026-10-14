import { lazy, Suspense, useEffect, useState } from 'react'
import Editor, { type EditorStart, type Source } from './components/Editor'
import Welcome from './components/Welcome'
import type { EditParams } from './lib/pipeline'
import { isVideoFile } from './lib/save'

// The video editor (and the video toolkit under it) loads only when a
// video is opened.
const VideoEditor = lazy(() => import('./components/VideoEditor'))

// `backHome`: ✕ on this photo goes back to the welcome screen (it was opened
// from there), not to the camera. `key`: a fresh editor each time one is
// opened from the welcome screen, so its `start` applies.
type Editing = { source: Source; start?: EditorStart; backHome: boolean; key: number }

// Opens on the welcome screen — what Glow does, and the two ways in: Camera
// (live retouch) and Edit (a photo or video you already have). Leaving it
// is a step in the browser's history, so the phone's back gesture returns
// to it. (/?mode=camera, the home-screen shortcut, skips it.)
export default function App() {
  const [editing, setEditing] = useState<Editing | null>(() =>
    new URLSearchParams(location.search).get('mode') === 'camera' ? { source: { kind: 'live' }, backHome: false, key: 0 } : null,
  )
  const [video, setVideo] = useState<{ file: Blob; params?: EditParams; backHome: boolean } | null>(null)

  useEffect(() => {
    const onPop = () => {
      setVideo(null)
      setEditing(null)
    }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const leaveHome = () => {
    if (!history.state?.glow) history.pushState({ glow: true }, '')
  }
  const home = () => {
    if (history.state?.glow) history.back()
    else {
      setVideo(null)
      setEditing(null)
    }
  }

  const openCamera = (start?: EditorStart) => {
    leaveHome()
    setEditing({ source: { kind: 'live' }, start, backHome: true, key: Date.now() })
  }
  const openFile = (file: File, start?: EditorStart) => {
    leaveHome()
    if (isVideoFile(file)) setVideo({ file, backHome: true })
    else setEditing({ source: { kind: 'image', file }, start, backHome: true, key: Date.now() })
  }

  // A ready portrait for anyone without a usable camera (or a photo to hand).
  const sampleFile = async () => {
    const blob = await (await fetch('/samples/portrait.jpg')).blob()
    return new File([blob], 'sample-portrait.jpg', { type: blob.type || 'image/jpeg' })
  }

  if (video)
    return (
      <Suspense fallback={<div className="fixed inset-0 bg-black" />}>
        <VideoEditor file={video.file} startParams={video.params} onClose={() => (video.backHome ? home() : setVideo(null))} />
      </Suspense>
    )
  if (!editing) return <Welcome onCamera={openCamera} onEdit={openFile} onSample={() => void sampleFile().then((f) => openFile(f))} />
  return (
    <Editor
      key={editing.key}
      source={editing.source}
      start={editing.start}
      onHome={home}
      onReset={() => (editing.backHome && editing.source.kind === 'image' ? home() : setEditing((e) => e && { ...e, source: { kind: 'live' } }))}
      onPickImage={(file) => setEditing((e) => e && { ...e, source: { kind: 'image', file }, backHome: false })}
      onEditVideo={(file, params) => {
        // (back from the video, the camera opens as it was left, not as the
        // welcome screen first set it)
        setEditing((e) => e && { ...e, start: undefined })
        setVideo({ file, params, backHome: false })
      }}
      // No camera: ✕ on the sample goes home, where Edit is, not back to a
      // camera that doesn't work.
      onTrySample={() => void sampleFile().then((file) => setEditing((e) => e && { ...e, source: { kind: 'image', file }, backHome: true }))}
    />
  )
}
