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

// Soft edges without canvas filters (not every browser has them, and a
// blur filter is slow): draw at reduced resolution and scale it back up —
// the upscale's own interpolation feathers every edge by about a
// low-res pixel.
let softCanvas: HTMLCanvasElement | null = null
function soft(ctx: CanvasRenderingContext2D, x0: number, y0: number, w: number, h: number, blur: number, draw: (g: CanvasRenderingContext2D) => void) {
  const k = Math.min(1, Math.max(0.12, 1 / Math.max(1, blur)))
  const sw = Math.max(2, Math.ceil(w * k))
  const sh = Math.max(2, Math.ceil(h * k))
  softCanvas ??= document.createElement('canvas')
  if (softCanvas.width < sw + 2 || softCanvas.height < sh + 2) {
    softCanvas.width = Math.max(softCanvas.width, sw + 2)
    softCanvas.height = Math.max(softCanvas.height, sh + 2)
  }
  const g = softCanvas.getContext('2d')!
  g.setTransform(1, 0, 0, 1, 0, 0)
  g.globalCompositeOperation = 'source-over'
  g.globalAlpha = 1
  g.clearRect(0, 0, sw + 2, sh + 2)
  g.setTransform(k, 0, 0, k, -x0 * k, -y0 * k)
  draw(g)
  ctx.save()
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(softCanvas, 0, 0, sw, sh, x0, y0, sw / k, sh / k)
  ctx.restore()
}

const boundsOf = (pts: Pt[], pad: number) => {
  const x0 = Math.floor(Math.min(...pts.map((p) => p.x)) - pad)
  const y0 = Math.floor(Math.min(...pts.map((p) => p.y)) - pad)
  return { x0, y0, w: Math.ceil(Math.max(...pts.map((p) => p.x)) + pad) - x0, h: Math.ceil(Math.max(...pts.map((p) => p.y)) + pad) - y0 }
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
  const bb = boundsOf([...lid, flick, ...top], E * 0.12)
  soft(ctx, bb.x0, bb.y0, bb.w, bb.h, E * 0.035, (g) => {
    const grad = g.createLinearGradient(inner.x, inner.y, flick.x, flick.y)
    grad.addColorStop(0, 'rgba(255,150,205,0.55)')
    grad.addColorStop(0.55, 'rgba(244,84,176,0.62)')
    grad.addColorStop(1, 'rgba(196,72,214,0.6)')
    g.fillStyle = grad
    path(g, [...lid, flick, ...top.slice().reverse()])
    g.fill()
  })
  // A shimmer lift on the centre of the lid.
  const c = lerp(lid[4], top[4], 0.4)
  const r = w * 0.32
  ctx.save()
  const sh = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, r)
  sh.addColorStop(0, 'rgba(255,215,240,0.45)')
  sh.addColorStop(1, 'rgba(255,215,240,0)')
  ctx.globalCompositeOperation = 'screen'
  ctx.fillStyle = sh
  ctx.beginPath()
  ctx.arc(c.x, c.y, r, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()

  // Cool blue sparkle along the lower lash line.
  const lashLo = lower.slice(0, 7).map((p) => ({ x: p.x - ux * E * 0.012, y: p.y - uy * E * 0.012 }))
  const lb = boundsOf(lashLo, E * 0.08)
  soft(ctx, lb.x0, lb.y0, lb.w, lb.h, E * 0.012, (g) => {
    g.strokeStyle = 'rgba(120,190,255,0.55)'
    g.lineWidth = Math.max(1, E * 0.035)
    g.lineCap = 'round'
    g.beginPath()
    smooth(g, lashLo)
    g.stroke()
  })

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

let lipLayer: HTMLCanvasElement | null = null

/** Pale lavender gloss: the lips recoloured with blend modes (no pixel
 *  read-back, which stalls the GPU every frame) — 'color' takes the
 *  lavender's hue and saturation while keeping the lips' own light and
 *  texture, then 'screen' lifts them to a pale cream shade. */
function lips(ctx: CanvasRenderingContext2D, P: Pt[], E: number) {
  const outer = LIPS_OUTER.map((i) => P[i])
  const inner = LIPS_INNER.map((i) => P[i])
  const bb = boundsOf(outer, E * 0.06)
  if (bb.w < 4 || bb.h < 4) return
  lipLayer ??= document.createElement('canvas')
  if (lipLayer.width < bb.w || lipLayer.height < bb.h) {
    lipLayer.width = Math.max(lipLayer.width, bb.w)
    lipLayer.height = Math.max(lipLayer.height, bb.h)
  }
  const L = lipLayer.getContext('2d')!
  L.setTransform(1, 0, 0, 1, 0, 0)
  L.globalCompositeOperation = 'source-over'
  L.clearRect(0, 0, lipLayer.width, lipLayer.height)
  // The lips' shape, feathered (the outline minus the mouth's opening).
  soft(L, 0, 0, bb.w, bb.h, E * 0.012, (g) => {
    g.translate(-bb.x0, -bb.y0)
    g.fillStyle = '#fff'
    path(g, outer)
    g.fill()
    g.globalCompositeOperation = 'destination-out'
    path(g, inner)
    g.fill()
  })
  L.globalCompositeOperation = 'source-in'
  L.fillStyle = 'rgb(200,182,240)'
  L.fillRect(0, 0, bb.w, bb.h)
  ctx.save()
  ctx.globalCompositeOperation = 'color'
  ctx.globalAlpha = 0.9
  ctx.drawImage(lipLayer, 0, 0, bb.w, bb.h, bb.x0, bb.y0, bb.w, bb.h)
  ctx.globalCompositeOperation = 'screen'
  ctx.globalAlpha = 0.55
  ctx.drawImage(lipLayer, 0, 0, bb.w, bb.h, bb.x0, bb.y0, bb.w, bb.h)
  ctx.restore()
}

/** The Lavender look's makeup on the face at landmarks `P`. */
export function drawLavenderMakeup(ctx: CanvasRenderingContext2D, P: Pt[]) {
  const E = dist(lerp(P[33], P[133], 0.5), lerp(P[263], P[362], 0.5))
  if (E < 8) return
  eye(ctx, P, LID_UP_R, LID_LO_R, BROW_R, E)
  eye(ctx, P, LID_UP_L, LID_LO_L, BROW_L, E)
  lips(ctx, P, E)
}
