// Getting a finished photo or video off the page: iOS has no "download a
// file to Photos" — the share sheet's Save Image / Save Video is the only
// way a web app's file reaches the camera roll — so saving goes through it
// there, and through a plain download everywhere else.

export const isIOS = typeof navigator !== 'undefined' && (/iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1))

export function timestampedName(ext: string): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `FluxGlow_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}.${ext}`
}

export function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export const extFor = (blob: Blob) => (blob.type.includes('mp4') ? 'mp4' : blob.type.includes('webm') ? 'webm' : blob.type.includes('quicktime') ? 'mov' : blob.type.includes('png') ? 'png' : 'jpg')

/** Saves `blob`. Resolves to what happened: 'saved' (downloaded),
 *  'shared' (handed to the share sheet) or 'dismissed'. */
export async function saveBlob(blob: Blob, name: string): Promise<'saved' | 'shared' | 'dismissed'> {
  const file = new File([blob], name, { type: blob.type })
  if (isIOS && navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file] })
      return 'shared'
    } catch {
      return 'dismissed'
    }
  }
  downloadBlob(blob, name)
  return 'saved'
}

/** Shares `blob` (the share sheet), or saves it where there is none. */
export async function shareBlob(blob: Blob, name: string): Promise<'saved' | 'shared' | 'dismissed'> {
  const file = new File([blob], name, { type: blob.type })
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: 'Flux Glow' })
      return 'shared'
    } catch {
      return 'dismissed'
    }
  }
  downloadBlob(blob, name)
  return 'saved'
}

/** Is this file a video (by type, or by name when the type is missing)? */
export const isVideoFile = (f: File) => f.type.startsWith('video/') || /\.(mp4|mov|m4v|webm|mkv)$/i.test(f.name)
