// Painted makeup, drawn on the face from its landmarks (pixels) — the
// Lavender look after the reference: vivid pink-to-magenta shimmer over
// the lids, a black winged liner, a cool blue sparkle along the lower lash
// line, and pale lavender gloss lips.

interface Pt {
  x: number
  y: number
}

// Landmark runs (MediaPipe face mesh). Upper lids from the outer corner
// to the inner one; lower lids likewise; the brows' lower edge.
const LID_UP_R = [33, 246, 161, 160, 159, 158, 157, 173, 133]
const LID_LO_R = [33, 7, 163, 144, 145, 153, 154, 155, 133]
const BROW_R = [46, 53, 52, 65, 55]
const LID_UP_L = [263, 466, 388, 387, 386, 385, 384, 398, 362]
const LID_LO_L = [263, 249, 390, 373, 374, 380, 381, 382, 362]
const BROW_L = [276, 283, 282, 295, 285]
const LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
const LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

const lerp = (a: Pt, b: Pt, t: number) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t })
const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y)

/** The point along a polyline at fraction t of its length. */
function along(pts: Pt[], t: number) {
  const seg = pts.slice(1).map((p, i) => dist(pts[i], p))
  let d = t * seg.reduce((a, b) => a + b, 0)
  for (let i = 0; i < seg.length; i++) {
    if (d <= seg[i]) return lerp(pts[i], pts[i + 1], seg[i] ? d / seg[i] : 0)
    d -= seg[i]
  }
  return pts[pts.length - 1]
}

function path(ctx: CanvasRenderingContext2D, pts: Pt[], close = true) {
  ctx.beginPath()
  pts.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)))
  if (close) ctx.closePath()
}

/** A smooth curve through `pts` (midpoint quadratics). */
function smooth(ctx: CanvasRenderingContext2D, pts: Pt[]) {
  ctx.moveTo(pts[0].x, pts[0].y)
  for (let i = 1; i < pts.length - 1; i++) {
    const m = lerp(pts[i], pts[i + 1], 0.5)
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, m.x, m.y)
  }
  const last = pts[pts.length - 1]
  ctx.lineTo(last.x, last.y)
}

function eye(ctx: CanvasRenderingContext2D, P: Pt[], up: number[], lo: number[], brow: number[], E: number) {
  const lid = up.map((i) => P[i])
  const lower = lo.map((i) => P[i])
  const outer = lid[0]
  const inner = lid[lid.length - 1]
  const w = dist(outer, inner)
  // "Up" for this eye: perpendicular to its corner line, toward the brow.
  let ux = -(inner.y - outer.y)
  let uy = inner.x - outer.x
  const l = Math.hypot(ux, uy) || 1
  ux /= l
  uy /= l
  const browPts = brow.map((i) => P[i])
  const browMid = browPts[Math.floor(browPts.length / 2)]
  if ((browMid.x - outer.x) * ux + (browMid.y - outer.y) * uy < 0) {
    ux = -ux
    uy = -uy
  }
  // The outward direction (from the inner corner to the outer one).
  const ox = (outer.x - inner.x) / w
  const oy = (outer.y - inner.y) / w

  // Lid shadow: from the lash line up to just under the brow, deepest
  // magenta at the outer corner, swept a little out past it.
  const top = lid.map((p, k) => {
    const t = k / (lid.length - 1)
    // (the brow point above this one, along the eye's own "up")
    const b = along(browPts.slice().sort((a, c) => (a.x - c.x) * ox + (a.y - c.y) * oy).reverse(), t)
    return lerp(p, b, 0.62 - 0.12 * t)
  })
  const flick = { x: outer.x + ox * w * 0.3 + ux * w * 0.22, y: outer.y + oy * w * 0.3 + uy * w * 0.22 }
  ctx.save()
  ctx.filter = `blur(${Math.max(1, E * 0.035)}px)`
  const grad = ctx.createLinearGradient(inner.x, inner.y, flick.x, flick.y)
  grad.addColorStop(0, 'rgba(255,150,205,0.55)')
  grad.addColorStop(0.55, 'rgba(244,84,176,0.62)')
  grad.addColorStop(1, 'rgba(196,72,214,0.6)')
  ctx.fillStyle = grad
  path(ctx, [...lid, flick, ...top.slice().reverse()])
  ctx.fill()
  // A shimmer lift on the centre of the lid.
  const c = lerp(lid[4], top[4], 0.4)
  const r = w * 0.32
  const sh = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, r)
  sh.addColorStop(0, 'rgba(255,215,240,0.45)')
  sh.addColorStop(1, 'rgba(255,215,240,0)')
  ctx.globalCompositeOperation = 'screen'
  ctx.fillStyle = sh
  ctx.fillRect(c.x - r, c.y - r, r * 2, r * 2)
  ctx.restore()

  // Cool blue sparkle along the lower lash line.
  ctx.save()
  ctx.filter = `blur(${Math.max(0.6, E * 0.012)}px)`
  ctx.strokeStyle = 'rgba(120,190,255,0.55)'
  ctx.lineWidth = Math.max(1, E * 0.035)
  ctx.lineCap = 'round'
  ctx.beginPath()
  smooth(ctx, lower.slice(0, 7).map((p) => ({ x: p.x - ux * E * 0.012, y: p.y - uy * E * 0.012 })))
  ctx.stroke()
  ctx.restore()

  // Winged liner: thin at the inner corner, thickening outward along the
  // lash line, then a sharp wing flicked up and out.
  const wingTip = { x: outer.x + ox * w * 0.42 + ux * w * 0.24, y: outer.y + oy * w * 0.42 + uy * w * 0.24 }
  const thick = (t: number) => E * (0.012 + 0.05 * t * t)
  const upper = lid
    .slice()
    .reverse()
    .map((p, k, a) => {
      const t = k / (a.length - 1)
      return { x: p.x + ux * thick(t), y: p.y + uy * thick(t) }
    })
  ctx.save()
  ctx.fillStyle = 'rgba(14,10,16,0.92)'
  ctx.filter = `blur(${Math.max(0.4, E * 0.004)}px)`
  ctx.beginPath()
  const lashLine = lid.slice().reverse()
  ctx.moveTo(lashLine[0].x, lashLine[0].y)
  for (const p of lashLine) ctx.lineTo(p.x, p.y)
  ctx.lineTo(wingTip.x, wingTip.y)
  for (const p of upper.slice().reverse()) ctx.lineTo(p.x, p.y)
  ctx.closePath()
  ctx.fill()
  ctx.restore()
}

let maskCanvas: HTMLCanvasElement | null = null

/** Pale lavender gloss: the lips recoloured, keeping their own light and
 *  texture, with the gloss's highlights lifted. */
function lips(ctx: CanvasRenderingContext2D, P: Pt[], E: number) {
  const outer = LIPS_OUTER.map((i) => P[i])
  const inner = LIPS_INNER.map((i) => P[i])
  const pad = Math.ceil(E * 0.08)
  const x0 = Math.max(0, Math.floor(Math.min(...outer.map((p) => p.x)) - pad))
  const y0 = Math.max(0, Math.floor(Math.min(...outer.map((p) => p.y)) - pad))
  const x1 = Math.min(ctx.canvas.width, Math.ceil(Math.max(...outer.map((p) => p.x)) + pad))
  const y1 = Math.min(ctx.canvas.height, Math.ceil(Math.max(...outer.map((p) => p.y)) + pad))
  const w = x1 - x0
  const h = y1 - y0
  if (w < 4 || h < 4) return
  // The lips' mask: the outer outline minus the mouth's opening, feathered.
  maskCanvas ??= document.createElement('canvas')
  maskCanvas.width = w
  maskCanvas.height = h
  const m = maskCanvas.getContext('2d', { willReadFrequently: true })!
  m.filter = `blur(${Math.max(0.6, E * 0.012)}px)`
  m.translate(-x0, -y0)
  m.fillStyle = '#fff'
  path(m, outer)
  m.fill()
  m.globalCompositeOperation = 'destination-out'
  path(m, inner)
  m.fill()
  const mask = m.getImageData(0, 0, w, h).data
  const img = ctx.getImageData(x0, y0, w, h)
  const d = img.data
  // The lips' own mean brightness, to keep their light and shade.
  let sum = 0
  let n = 0
  for (let i = 0; i < d.length; i += 4)
    if (mask[i + 3] > 200) {
      sum += 0.3 * d[i] + 0.59 * d[i + 1] + 0.11 * d[i + 2]
      n++
    }
  const mean = n ? sum / n : 128
  const lav = [206, 190, 238]
  for (let i = 0; i < d.length; i += 4) {
    const a = (mask[i + 3] / 255) * 0.9
    if (a <= 0.01) continue
    const L = 0.3 * d[i] + 0.59 * d[i + 1] + 0.11 * d[i + 2]
    // Shade follows the lips' own (an opaque cream lipstick flattens it
    // somewhat), and the brightest points become gloss.
    const shade = Math.min(1.25, Math.max(0.55, 0.45 + 0.55 * (L / mean)))
    const gloss = Math.max(0, (L - mean * 1.35) / (255 - mean * 1.35)) * 0.8
    for (let k = 0; k < 3; k++) {
      const c = lav[k] * shade + (255 - lav[k] * shade) * gloss
      d[i + k] = d[i + k] + (c - d[i + k]) * a
    }
  }
  ctx.putImageData(img, x0, y0)
}

/** The Lavender look's makeup on the face at landmarks `P`. */
export function drawLavenderMakeup(ctx: CanvasRenderingContext2D, P: Pt[]) {
  const E = dist(lerp(P[33], P[133], 0.5), lerp(P[263], P[362], 0.5))
  if (E < 8) return
  eye(ctx, P, LID_UP_R, LID_LO_R, BROW_R, E)
  eye(ctx, P, LID_UP_L, LID_LO_L, BROW_L, E)
  lips(ctx, P, E)
}
