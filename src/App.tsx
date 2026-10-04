import { useState } from 'react'
import Editor, { type Source } from './components/Editor'

// Opens straight into the live camera — no upload-or-live landing screen —
// since that's the primary path (a selfie retouch app someone opens to
// take a photo, not browse to one); Editor's own toolbar carries a small
// photo icon next to the shutter for the secondary "edit an existing
// photo" path instead of presenting both as an equal up-front choice.
export default function App() {
  const [source, setSource] = useState<Source>({ kind: 'live' })

  // A ready portrait for anyone without a usable camera.
  const trySample = async () => {
    const blob = await (await fetch('/samples/portrait.jpg')).blob()
    setSource({ kind: 'image', file: new File([blob], 'sample-portrait.jpg', { type: blob.type || 'image/jpeg' }) })
  }

  return (
    <Editor
      source={source}
      onReset={() => setSource({ kind: 'live' })}
      onPickImage={(file) => setSource({ kind: 'image', file })}
      onTrySample={trySample}
    />
  )
}
