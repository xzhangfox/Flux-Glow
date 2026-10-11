// Keeps the screen on through a long job (analysing or exporting a long
// video): a phone that dims and locks halfway through suspends the page
// and the job with it. The lock lapses whenever the page is hidden, so it's
// taken again each time the page comes back.

/** Holds the screen on until the returned function is called. */
export function holdScreenOn(): () => void {
  const wl = (navigator as Navigator & { wakeLock?: { request(type: 'screen'): Promise<{ released: boolean; release(): Promise<void> }> } }).wakeLock
  if (!wl) return () => {}
  let lock: { released: boolean; release(): Promise<void> } | null = null
  let done = false
  const take = async () => {
    if (done || document.visibilityState !== 'visible' || (lock && !lock.released)) return
    try {
      const l = await wl.request('screen')
      if (done) void l.release().catch(() => {})
      else lock = l
    } catch {
      // (not allowed here: low battery, an iframe…)
    }
  }
  const onVisible = () => void take()
  document.addEventListener('visibilitychange', onVisible)
  void take()
  return () => {
    done = true
    document.removeEventListener('visibilitychange', onVisible)
    void lock?.release().catch(() => {})
  }
}
