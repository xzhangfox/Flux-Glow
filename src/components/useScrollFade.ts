import { useEffect, useState, type RefObject } from 'react'

/** Whether a horizontal scroll row has more content past its right edge —
 *  drives a fade on that edge so a control that's off-screen doesn't look
 *  like it doesn't exist. */
export function useScrollFade(ref: RefObject<HTMLElement | null>, deps: unknown[] = []): boolean {
  const [fade, setFade] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => setFade(el.scrollLeft + el.clientWidth < el.scrollWidth - 4)
    update()
    el.addEventListener('scroll', update, { passive: true })
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => {
      el.removeEventListener('scroll', update)
      ro.disconnect()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ref, ...deps])
  return fade
}

export const FADE_RIGHT_STYLE = { maskImage: 'linear-gradient(to right, #000 82%, transparent)', WebkitMaskImage: 'linear-gradient(to right, #000 82%, transparent)' }
