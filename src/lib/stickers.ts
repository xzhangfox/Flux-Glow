// Stickers for a captured/edited photo. Every sticker — preset art, emoji,
// styled text or the user's own photo — is turned into one bitmap when it's
// added, then placed with a center, a width (as a fraction of the photo's
// width, so it survives the preview→export size change) and a rotation.

export interface Sticker {
  id: number
  img: CanvasImageSource
  /** height / width of the bitmap */
  ratio: number
  /** center, normalized to the photo */
  x: number
  y: number
  /** width as a fraction of the photo width */
  w: number
  rot: number
}

let nextId = 1

export function placeSticker(img: CanvasImageSource, ratio: number, w: number, photoAspect: number): Sticker {
  // Keep tall stickers from overflowing a landscape frame.
  const maxW = Math.min(w, (0.6 * photoAspect) / ratio)
  return { id: nextId++, img, ratio, x: 0.5, y: 0.45, w: maxW, rot: 0 }
}

/** Draws stickers in order (last on top) onto a canvas of the photo's size. */
export function drawStickers(ctx: CanvasRenderingContext2D, stickers: Sticker[], width: number, height: number) {
  for (const s of stickers) {
    const w = s.w * width
    const h = w * s.ratio
    ctx.save()
    ctx.translate(s.x * width, s.y * height)
    ctx.rotate(s.rot)
    ctx.drawImage(s.img, -w / 2, -h / 2, w, h)
    ctx.restore()
  }
}

/** Topmost sticker under a normalized point, honoring rotation. */
export function hitSticker(stickers: Sticker[], px: number, py: number, width: number, height: number): Sticker | null {
  for (let i = stickers.length - 1; i >= 0; i--) {
    const s = stickers[i]
    const dx = (px - s.x) * width
    const dy = (py - s.y) * height
    const lx = dx * Math.cos(-s.rot) - dy * Math.sin(-s.rot)
    const ly = dx * Math.sin(-s.rot) + dy * Math.cos(-s.rot)
    const hw = (s.w * width) / 2
    const hh = (s.w * width * s.ratio) / 2
    // A little slack so small stickers are easy to grab with a finger.
    const pad = Math.max(0, 22 - Math.min(hw, hh))
    if (Math.abs(lx) <= hw + pad && Math.abs(ly) <= hh + pad) return s
  }
  return null
}

// ---- Preset art --------------------------------------------------------------
// Hand-drawn SVG, so they stay sharp at any size the sticker is scaled to.

const G = {
  pink: '<linearGradient id="p" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ff9ac1"/><stop offset="1" stop-color="#ff3d85"/></linearGradient>',
  gold: '<linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FCF6BA"/><stop offset=".5" stop-color="#E6B93A"/><stop offset="1" stop-color="#AA771C"/></linearGradient>',
  sky: '<linearGradient id="s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#dbe9ff"/></linearGradient>',
}
const svg = (defs: string, body: string, vb = '0 0 120 120') => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${vb}"><defs>${defs}</defs>${body}</svg>`
const bubble = (text: string, fill: string, color: string) =>
  svg('', `<path d="M14 18h92a10 10 0 0 1 10 10v46a10 10 0 0 1-10 10H52l-20 18 4-18H14A10 10 0 0 1 4 74V28a10 10 0 0 1 10-10z" fill="${fill}" stroke="#111" stroke-width="4" stroke-linejoin="round"/><text x="60" y="61" text-anchor="middle" font-family="Arial Black, Arial, sans-serif" font-weight="900" font-size="${text.length > 4 ? 22 : 28}" fill="${color}">${text}</text>`)

export const ART: { id: string; label: string; svg: string }[] = [
  { id: 'heart', label: 'Heart', svg: svg(G.pink, '<path d="M60 104C24 80 8 60 8 38a24 24 0 0 1 52-14 24 24 0 0 1 52 14c0 22-16 42-52 66z" fill="url(#p)" stroke="#fff" stroke-width="5"/><path d="M28 34c4-8 12-11 18-9" stroke="#fff" stroke-width="6" stroke-linecap="round" fill="none" opacity=".8"/>') },
  { id: 'sparkle', label: 'Sparkle', svg: svg(G.gold, '<path d="M60 4c5 30 12 44 52 56-40 12-47 26-52 56-5-30-12-44-52-56 40-12 47-26 52-56z" fill="url(#g)"/><path d="M98 6c2 10 4 14 14 16-10 2-12 6-14 16-2-10-4-14-14-16 10-2 12-6 14-16z" fill="#fff"/>') },
  { id: 'star', label: 'Star', svg: svg(G.gold, '<path d="M60 6l15 33 36 4-27 24 8 36-32-19-32 19 8-36L9 43l36-4z" fill="url(#g)" stroke="#fff" stroke-width="5" stroke-linejoin="round"/>') },
  { id: 'crown', label: 'Crown', svg: svg(G.gold, '<path d="M10 92l-4-60 30 26 24-40 24 40 30-26-4 60z" fill="url(#g)" stroke="#8a5f12" stroke-width="4" stroke-linejoin="round"/><circle cx="60" cy="72" r="8" fill="#ff4d6d"/><circle cx="32" cy="76" r="6" fill="#4dc3ff"/><circle cx="88" cy="76" r="6" fill="#4dc3ff"/>') },
  { id: 'bow', label: 'Bow', svg: svg(G.pink, '<path d="M60 58C40 30 6 26 8 54c2 26 34 26 52 4zM60 58c20-28 54-32 52-4-2 26-34 26-52 4z" fill="url(#p)" stroke="#fff" stroke-width="4"/><path d="M50 62l-10 40 14-8 6 12 6-12 14 8-10-40z" fill="url(#p)" stroke="#fff" stroke-width="4" stroke-linejoin="round"/><ellipse cx="60" cy="58" rx="12" ry="14" fill="#ff6aa4" stroke="#fff" stroke-width="4"/>') },
  { id: 'flower', label: 'Flower', svg: svg('', '<g fill="#ffd1e3" stroke="#ff7fb0" stroke-width="3">' + [0, 72, 144, 216, 288].map((a) => `<ellipse cx="60" cy="32" rx="18" ry="26" transform="rotate(${a} 60 60)"/>`).join('') + '</g><circle cx="60" cy="60" r="15" fill="#ffd36b" stroke="#f5a623" stroke-width="3"/>') },
  { id: 'cloud', label: 'Cloud', svg: svg(G.sky, '<path d="M30 92a22 22 0 0 1-2-44 28 28 0 0 1 52-10 22 22 0 0 1 20 34 18 18 0 0 1-8 20z" fill="url(#s)" stroke="#9cc3ff" stroke-width="4"/><circle cx="46" cy="70" r="4" fill="#333"/><circle cx="72" cy="70" r="4" fill="#333"/><path d="M53 78q6 6 12 0" stroke="#333" stroke-width="3" fill="none" stroke-linecap="round"/><circle cx="38" cy="78" r="5" fill="#ffb3c8"/><circle cx="80" cy="78" r="5" fill="#ffb3c8"/>') },
  { id: 'bolt', label: 'Bolt', svg: svg(G.gold, '<path d="M70 4L20 68h32l-8 48 56-68H66z" fill="url(#g)" stroke="#fff" stroke-width="5" stroke-linejoin="round"/>') },
  { id: 'rainbow', label: 'Rainbow', svg: svg('', ['#ff5a5f', '#ffb347', '#ffe066', '#7bd389', '#5ab0ff', '#a07bff'].map((c, i) => `<path d="M${10 + i * 7} 92a${50 - i * 7} ${50 - i * 7} 0 0 1 ${100 - i * 14} 0" stroke="${c}" stroke-width="7" fill="none"/>`).join('') + '<ellipse cx="18" cy="94" rx="16" ry="10" fill="#fff"/><ellipse cx="102" cy="94" rx="16" ry="10" fill="#fff"/>') },
  { id: 'cherry', label: 'Cherry', svg: svg('', '<path d="M40 74C44 44 58 24 82 12M80 80C76 52 78 30 82 12" stroke="#4a8a2a" stroke-width="5" fill="none" stroke-linecap="round"/><path d="M82 12c10 2 18 10 18 20-12 0-18-8-18-20z" fill="#6cc04a"/><circle cx="40" cy="84" r="22" fill="#e8263c"/><circle cx="82" cy="88" r="22" fill="#d81f34"/><circle cx="33" cy="76" r="6" fill="#fff" opacity=".7"/><circle cx="75" cy="80" r="6" fill="#fff" opacity=".7"/>') },
  { id: 'butterfly', label: 'Butterfly', svg: svg('<linearGradient id="b" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#9ad7ff"/><stop offset="1" stop-color="#b48bff"/></linearGradient>', '<path d="M58 60C40 20 8 14 10 40c2 18 22 22 48 20zM62 60c18-40 50-46 48-20-2 18-22 22-48 20zM58 64c-22 0-40 10-34 30 8 18 28 0 34-30zM62 64c22 0 40 10 34 30-8 18-28 0-34-30z" fill="url(#b)" stroke="#fff" stroke-width="4"/><rect x="56" y="40" width="8" height="48" rx="4" fill="#3b2a55"/>') },
  { id: 'paw', label: 'Paw', svg: svg(G.pink, '<ellipse cx="60" cy="80" rx="30" ry="24" fill="url(#p)"/><ellipse cx="28" cy="50" rx="11" ry="14" fill="url(#p)"/><ellipse cx="48" cy="34" rx="11" ry="14" fill="url(#p)"/><ellipse cx="72" cy="34" rx="11" ry="14" fill="url(#p)"/><ellipse cx="92" cy="50" rx="11" ry="14" fill="url(#p)"/>') },
  { id: 'fire', label: 'Fire', svg: svg('<linearGradient id="f" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#ff3d00"/><stop offset=".6" stop-color="#ff9100"/><stop offset="1" stop-color="#ffd54f"/></linearGradient>', '<path d="M60 4c8 26 34 36 34 66a34 34 0 0 1-68 0c0-16 8-26 16-32 0 12 6 18 12 20-6-20 0-38 6-54z" fill="url(#f)"/><path d="M60 62c4 12 16 16 16 30a16 16 0 0 1-32 0c0-12 10-16 16-30z" fill="#ffe082"/>') },
  { id: 'note', label: 'Music', svg: svg('', '<path d="M44 86V26l52-12v58" stroke="#1d1d1f" stroke-width="8" fill="none" stroke-linejoin="round"/><ellipse cx="32" cy="88" rx="16" ry="12" fill="#1d1d1f"/><ellipse cx="84" cy="74" rx="16" ry="12" fill="#1d1d1f"/><path d="M44 40l52-12" stroke="#1d1d1f" stroke-width="8"/>') },
  { id: 'shades', label: 'Shades', svg: svg('', '<path d="M4 40h112v8h-6l-4 22c-2 12-14 18-26 18s-20-8-22-20l-2-8h-6l-2 8c-2 12-10 20-22 20S6 82 4 70L2 48z" fill="#111"/><path d="M14 50l10 0-8 18zM72 50l10 0-8 18z" fill="#fff" opacity=".45"/>', '0 20 120 80') },
  { id: 'lips', label: 'Kiss', svg: svg('', '<path d="M60 44c10-14 24-18 34-12 8 4 14 12 22 16-10 6-12 30-56 36C16 78 14 54 4 48c8-4 14-12 22-16 10-6 24-2 34 12z" fill="#e8263c"/><path d="M8 48c16 4 34 6 52 6s36-2 52-6" stroke="#9c0f22" stroke-width="3" fill="none"/>', '0 20 120 80') },
  { id: 'wow', label: 'WOW', svg: bubble('WOW!', '#ffe14d', '#111') },
  { id: 'love', label: 'LOVE', svg: bubble('LOVE', '#ff7eb6', '#fff') },
  { id: 'omg', label: 'OMG', svg: bubble('OMG', '#7fe3ff', '#111') },
  { id: 'cute', label: 'CUTE', svg: bubble('so cute', '#ffffff', '#ff3d85') },
]

export const EMOJI = ['😍', '🥰', '😎', '🤩', '🥳', '😜', '😘', '🙈', '✨', '💖', '💕', '🔥', '🌈', '⭐', '🌸', '🍓', '🍒', '🎀', '👑', '🦋', '🐱', '🐶', '🐰', '🐻', '🍀', '☁️', '⚡', '💫', '🎉', '💯', '🫶', '👀']

export function artImage(svgText: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgText)
  })
}

export function emojiCanvas(char: string): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = c.height = 256
  const ctx = c.getContext('2d')!
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.font = '200px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif'
  ctx.fillText(char, 128, 140)
  return c
}

export interface TextStyle {
  id: string
  label: string
}
export const TEXT_STYLES: TextStyle[] = [
  { id: 'pop', label: 'Pop' },
  { id: 'gold', label: 'Gold' },
  { id: 'bubble', label: 'Bubble' },
  { id: 'neon', label: 'Neon' },
  { id: 'tag', label: 'Tag' },
  { id: 'script', label: 'Script' },
]

/** Renders text in one of the styles above to a tightly-cropped canvas. */
export function textCanvas(text: string, style: string): HTMLCanvasElement {
  const size = 120
  const font =
    style === 'script'
      ? `italic 700 ${size}px "Snell Roundhand", "Brush Script MT", "Segoe Script", cursive`
      : `900 ${size}px Manrope, "Arial Black", Arial, sans-serif`
  const measure = document.createElement('canvas').getContext('2d')!
  measure.font = font
  const lines = text.split('\n').slice(0, 3)
  const tw = Math.max(...lines.map((l) => measure.measureText(l).width), size)
  const padX = style === 'bubble' || style === 'tag' ? 60 : 30
  const lineH = size * 1.15
  const c = document.createElement('canvas')
  c.width = Math.ceil(tw + padX * 2)
  c.height = Math.ceil(lineH * lines.length + (style === 'bubble' || style === 'tag' ? 70 : 40))
  const ctx = c.getContext('2d')!
  ctx.font = font
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  const cx = c.width / 2
  const y0 = c.height / 2 - (lineH * (lines.length - 1)) / 2
  if (style === 'bubble') {
    ctx.fillStyle = '#ff6aa4'
    ctx.beginPath()
    ctx.roundRect(6, 6, c.width - 12, c.height - 12, c.height / 2)
    ctx.fill()
  }
  if (style === 'tag') {
    ctx.fillStyle = '#ffe14d'
    ctx.save()
    ctx.translate(cx, c.height / 2)
    ctx.rotate(-0.03)
    ctx.fillRect(-c.width / 2 + 10, -c.height / 2 + 10, c.width - 20, c.height - 20)
    ctx.restore()
  }
  lines.forEach((line, i) => {
    const y = y0 + i * lineH
    switch (style) {
      case 'pop':
        ctx.lineWidth = 18
        ctx.lineJoin = 'round'
        ctx.strokeStyle = '#111'
        ctx.strokeText(line, cx, y)
        ctx.fillStyle = '#fff'
        ctx.fillText(line, cx, y)
        break
      case 'gold': {
        const g = ctx.createLinearGradient(0, y - size / 2, 0, y + size / 2)
        g.addColorStop(0, '#FCF6BA')
        g.addColorStop(0.5, '#E6B93A')
        g.addColorStop(1, '#AA771C')
        ctx.lineWidth = 10
        ctx.lineJoin = 'round'
        ctx.strokeStyle = '#5a3d0a'
        ctx.strokeText(line, cx, y)
        ctx.fillStyle = g
        ctx.fillText(line, cx, y)
        break
      }
      case 'neon':
        ctx.shadowColor = '#3df2ff'
        ctx.shadowBlur = 30
        ctx.lineWidth = 6
        ctx.strokeStyle = '#3df2ff'
        ctx.strokeText(line, cx, y)
        ctx.shadowBlur = 12
        ctx.fillStyle = '#e8feff'
        ctx.fillText(line, cx, y)
        break
      case 'script':
        ctx.shadowColor = 'rgba(0,0,0,0.45)'
        ctx.shadowBlur = 12
        ctx.fillStyle = '#fff'
        ctx.fillText(line, cx, y)
        break
      default:
        ctx.fillStyle = style === 'bubble' ? '#fff' : '#111'
        ctx.fillText(line, cx, y)
    }
  })
  return c
}

/** A photo from the album as a sticker: downscaled, and — unless it's a
 *  PNG that may already have its own transparent cut-out — given a white
 *  instant-photo border and soft shadow. */
export async function photoSticker(file: File): Promise<HTMLCanvasElement> {
  const bmp = await createImageBitmap(file)
  const scale = Math.min(1, 900 / Math.max(bmp.width, bmp.height))
  const w = Math.round(bmp.width * scale)
  const h = Math.round(bmp.height * scale)
  const framed = file.type !== 'image/png' && file.type !== 'image/webp' && file.type !== 'image/gif'
  const border = framed ? Math.round(Math.max(w, h) * 0.05) : 0
  const shadow = framed ? border : 0
  const c = document.createElement('canvas')
  c.width = w + border * 2 + shadow * 2
  c.height = h + border * 2 + shadow * 2
  const ctx = c.getContext('2d')!
  if (framed) {
    ctx.shadowColor = 'rgba(0,0,0,0.35)'
    ctx.shadowBlur = shadow
    ctx.shadowOffsetY = shadow * 0.3
    ctx.fillStyle = '#fff'
    ctx.beginPath()
    ctx.roundRect(shadow, shadow, w + border * 2, h + border * 2, border * 0.5)
    ctx.fill()
    ctx.shadowColor = 'transparent'
  }
  ctx.drawImage(bmp, shadow + border, shadow + border, w, h)
  bmp.close?.()
  return c
}
