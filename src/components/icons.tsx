// Small self-contained line icons — no web-font dependency (Material
// Symbols needed Google Fonts, which failed to load under any network
// restriction/ad-blocker and silently fell back to literal text like
// "blur_on"). Plain inline SVG always renders, and it fits a page whose
// whole pitch is "nothing leaves your device" better than a Google CDN.
import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement>

const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
}

export function IconSparkle(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z" />
    </svg>
  )
}

export function IconImage(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="3" y="4" width="18" height="16" rx="2.5" />
      <circle cx="9" cy="10" r="1.6" />
      <path d="M3 17l5.5-5.5a2 2 0 0 1 2.8 0L15 15l1.2-1.2a2 2 0 0 1 2.8 0L21 16" />
    </svg>
  )
}

export function IconCamera(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4 8.5A1.5 1.5 0 0 1 5.5 7h2l1-2h7l1 2h2A1.5 1.5 0 0 1 20 8.5v9A1.5 1.5 0 0 1 18.5 19h-13A1.5 1.5 0 0 1 4 17.5v-9z" />
      <circle cx="12" cy="12.5" r="3.3" />
    </svg>
  )
}

export function IconDroplet(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3.5c3 3.6 6 7 6 10.3a6 6 0 1 1-12 0c0-3.3 3-6.7 6-10.3z" />
    </svg>
  )
}

export function IconFaceOutline(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3.5c-3.6 0-5.8 2.6-5.8 6.3 0 2 .5 4 1.5 5.7.9 1.5 2.4 2.5 4.3 2.5s3.4-1 4.3-2.5c1-1.7 1.5-3.7 1.5-5.7 0-3.7-2.2-6.3-5.8-6.3z" />
      <path d="M9 10.2h.01M15 10.2h.01" strokeWidth={2.4} />
      <path d="M9.5 14c.7.6 1.6.9 2.5.9s1.8-.3 2.5-.9" />
    </svg>
  )
}

export function IconPalette(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3.5a8.3 8.3 0 1 0 0 16.6c1 0 1.8-.8 1.8-1.8 0-.5-.2-.9-.5-1.3-.3-.3-.5-.7-.5-1.2 0-1 .8-1.8 1.8-1.8H16c2.2 0 4-1.8 4-4 0-3.7-3.6-6.5-8-6.5z" />
      <circle cx="8" cy="10" r="1" fill="currentColor" stroke="none" />
      <circle cx="11.5" cy="7.3" r="1" fill="currentColor" stroke="none" />
      <circle cx="15.3" cy="8.5" r="1" fill="currentColor" stroke="none" />
    </svg>
  )
}

export function IconCompare(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3v18" />
      <path d="M7 7H4a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h3" />
      <path d="M17 7h3a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1h-3" />
    </svg>
  )
}

export function IconCheck(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4.5 12.5l5 5L19.5 7" />
    </svg>
  )
}

export function IconDownload(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3.5v11.5" />
      <path d="M7.5 11l4.5 4.5L16.5 11" />
      <path d="M4.5 18.5h15" />
    </svg>
  )
}

export function IconShare(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="6" cy="12" r="2.3" />
      <circle cx="17.5" cy="6" r="2.3" />
      <circle cx="17.5" cy="18" r="2.3" />
      <path d="M8.1 10.8l7.3-3.6M8.1 13.2l7.3 3.6" />
    </svg>
  )
}

export function IconRefresh(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M19 12a7 7 0 1 1-2.3-5.2" />
      <path d="M19 4v4.5h-4.5" />
    </svg>
  )
}

export function IconClose(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M5.5 5.5l13 13M18.5 5.5l-13 13" />
    </svg>
  )
}

export function IconInfo(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="12" cy="12" r="8.3" />
      <path d="M12 11v5.2" />
      <circle cx="12" cy="8" r="0.15" fill="currentColor" stroke="currentColor" strokeWidth={2.4} />
    </svg>
  )
}

export function IconError(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="12" cy="12" r="8.3" />
      <path d="M12 7.5v5.3" />
      <circle cx="12" cy="16" r="0.15" fill="currentColor" stroke="currentColor" strokeWidth={2.4} />
    </svg>
  )
}

export function IconSpinner(props: IconProps) {
  return (
    <svg {...base} viewBox="0 0 24 24" fill="none" {...props}>
      <path d="M12 3.5a8.5 8.5 0 1 0 8.5 8.5" stroke="currentColor" strokeWidth={2} strokeLinecap="round" />
    </svg>
  )
}
