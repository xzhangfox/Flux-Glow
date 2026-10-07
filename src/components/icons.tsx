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

export function IconFlipCamera(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M17 8H9.5a4.5 4.5 0 0 0-4.3 3.2" />
      <path d="M14.5 5.5L17 8l-2.5 2.5" />
      <path d="M7 16h7.5a4.5 4.5 0 0 0 4.3-3.2" />
      <path d="M9.5 18.5L7 16l2.5-2.5" />
    </svg>
  )
}

// One shared face silhouette with a dot at a different spot for each of the
// retouch drill-down's regions — faster and more legible at a glance than
// nine unrelated abstract glyphs would be, since it shows *where on the
// face* each control reaches rather than asking the label alone to carry
// that meaning (eyebrow/temple/cheekbone are easy to mix up as text).
export function RegionIcon({ dot, pair, ...props }: IconProps & { dot: [number, number]; pair?: boolean }) {
  const [x, y] = dot
  return (
    <svg {...base} {...props}>
      <path d="M12 3.5c-3.6 0-5.8 2.6-5.8 6.3 0 2 .5 4 1.5 5.7.9 1.5 2.4 2.5 4.3 2.5s3.4-1 4.3-2.5c1-1.7 1.5-3.7 1.5-5.7 0-3.7-2.2-6.3-5.8-6.3z" opacity={0.45} />
      <circle cx={x} cy={y} r={1.4} fill="currentColor" stroke="none" />
      {pair && <circle cx={24 - x} cy={y} r={1.4} fill="currentColor" stroke="none" />}
    </svg>
  )
}

// The Beauty panel's four tone/light effects — distinct glyphs since none
// of them map to a face position the way RegionIcon's dot does.
export function IconSun(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 3v2.2M12 18.8V21M4.5 12H3M21 12h-1.5M6.3 6.3L5 5M18 5l-1.3 1.3M6.3 17.7L5 19M18 19l-1.3-1.3" />
    </svg>
  )
}

export function IconWhiten(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3.5l1.6 4.6 4.9.2-3.9 3 1.4 4.8L12 13.4l-4 2.7 1.4-4.8-3.9-3 4.9-.2z" strokeLinejoin="round" />
    </svg>
  )
}

export function IconTarget(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="4" />
      <circle cx="12" cy="12" r="0.4" fill="currentColor" stroke="none" />
    </svg>
  )
}

export function IconWave(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M3 9c2 0 2-2.5 4-2.5S9 9 11 9s2-2.5 4-2.5S17 9 19 9" />
      <path d="M3 14.5c2 0 2-2.5 4-2.5s2 2.5 4 2.5 2-2.5 4-2.5 2 2.5 4 2.5" />
    </svg>
  )
}

export function IconBack(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M15 5l-7 7 7 7" />
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

export function IconTimer(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="12" cy="13.5" r="7" />
      <path d="M12 9.8v3.9l2.4 1.5" />
      <path d="M9.5 3.5h5" />
    </svg>
  )
}

export function IconGrid(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="3.5" y="3.5" width="17" height="17" rx="2.5" />
      <path d="M9.2 3.5v17M14.8 3.5v17M3.5 9.2h17M3.5 14.8h17" />
    </svg>
  )
}

// Looks: a wand with a spark — "apply a finished style".
export function IconWand(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4 20L15 9" />
      <path d="M13.5 7.5l3 3" />
      <path d="M18 3v3M16.5 4.5h3M20.5 9.5v2M19.5 10.5h2M8.5 3.5v2M7.5 4.5h2" />
    </svg>
  )
}

// Effects: a face with cat ears.
export function IconEars(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M5.5 10.5L5 4l4.5 3.2M18.5 10.5L19 4l-4.5 3.2" strokeLinejoin="round" />
      <path d="M12 6.6c-4.2 0-6.8 2.7-6.8 6.6 0 4 3 6.8 6.8 6.8s6.8-2.8 6.8-6.8c0-3.9-2.6-6.6-6.8-6.6z" />
      <circle cx="9.4" cy="12.6" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="14.6" cy="12.6" r="0.9" fill="currentColor" stroke="none" />
      <path d="M11 15.4l1 .8 1-.8" />
    </svg>
  )
}

// Stickers: a square with a peeled corner.
export function IconSticker(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M20 13V6.5A2.5 2.5 0 0 0 17.5 4h-11A2.5 2.5 0 0 0 4 6.5v11A2.5 2.5 0 0 0 6.5 20H13z" strokeLinejoin="round" />
      <path d="M13 20c0-3.9 3.1-7 7-7" />
      <circle cx="9.3" cy="10" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="14.7" cy="10" r="0.9" fill="currentColor" stroke="none" />
      <path d="M9.5 14.2c.8.8 1.6 1.1 2.5 1.1" />
    </svg>
  )
}

export function IconText(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M5 6.5V5h14v1.5M12 5v14M9.5 19h5" />
    </svg>
  )
}

export function IconTrash(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4.5 7h15M9.5 7V4.8h5V7M6.5 7l.8 12.2h9.4L17.5 7" strokeLinejoin="round" />
    </svg>
  )
}

export function IconRotate(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M19 12a7 7 0 1 1-2.1-5" />
      <path d="M17.5 3.5v3.8h-3.8" />
    </svg>
  )
}

export function IconPlus(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 5v14M5 12h14" />
    </svg>
  )
}

export function IconPlay(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M8 5.5v13l10.5-6.5L8 5.5z" fill="currentColor" />
    </svg>
  )
}

export function IconPause(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="6.5" y="5.5" width="3.6" height="13" rx="1" fill="currentColor" />
      <rect x="13.9" y="5.5" width="3.6" height="13" rx="1" fill="currentColor" />
    </svg>
  )
}

export function IconVideo(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="3" y="6" width="13" height="12" rx="2.5" />
      <path d="M16 10.5l5-3v9l-5-3" />
    </svg>
  )
}

export function IconPeople(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="9" cy="8.5" r="3.2" />
      <path d="M3.5 19c.6-3.2 2.8-5 5.5-5s4.9 1.8 5.5 5" />
      <circle cx="16.8" cy="9.2" r="2.6" />
      <path d="M16.3 14c2.3.1 3.8 1.6 4.3 4.3" />
    </svg>
  )
}

export function IconMerge(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M6 4v4.5a4 4 0 0 0 4 4h4a4 4 0 0 1 4 4V20" />
      <path d="M18 4v4.5a4 4 0 0 1-4 4" />
      <path d="M15.5 17.5L18 20l2.5-2.5" />
    </svg>
  )
}

export function IconEyeOff(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M3 12s3.2-6 9-6c1.6 0 3 .4 4.2 1M21 12s-3.2 6-9 6c-1.6 0-3-.4-4.2-1" />
      <path d="M9.8 14.2a3 3 0 0 1 4.4-4.4" />
      <path d="M4 20L20 4" />
    </svg>
  )
}
