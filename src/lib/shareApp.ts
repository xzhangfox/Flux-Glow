// "Share this app" — every Flux app carries one: the system share sheet
// with the app's own link where there is one, else the link copied.
export const APP_URL = 'https://flux-glow.vercel.app/'

export async function shareApp(): Promise<'shared' | 'copied' | 'dismissed' | 'failed'> {
  const data = { title: 'Flux Glow', text: 'Flux Glow — a beauty camera that runs entirely in your browser.', url: APP_URL }
  if (navigator.share) {
    try {
      await navigator.share(data)
      return 'shared'
    } catch (err) {
      if ((err as DOMException)?.name === 'AbortError') return 'dismissed'
    }
  }
  try {
    await navigator.clipboard.writeText(APP_URL)
    return 'copied'
  } catch {
    return 'failed'
  }
}

