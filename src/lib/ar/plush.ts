import * as THREE from 'three'

// Plush ears as painted fur cards. Real-time shell fur never quite reads as
// a photographed plush accessory — it looks like moulded felt. So each ear
// is painted once, procedurally, as ~20k individual hair strokes: dark
// under-coat first, then longer, lighter hairs on top, all following a flow
// field that runs toward the tip and fans out at the edges (so the outline
// itself is fuzzy, hairs overlapping it), a pink plush inner ear and long
// pale tufts fanning out of its base. The painting goes on a gently curved
// card that keeps everything 3D gives: head pose, lighting from the photo,
// shadows on the head, occlusion by the head, and the photo-matched
// compositing.

type XY = [number, number]

export interface PlushSpec {
  /** Canvas size (px) and the card's size in rig units. */
  w: number
  h: number
  cardW: number
  /** Closed outline of the ear in canvas px; base along the bottom (or top
   *  when `hang`), tip where the fur flows to. */
  outline: XY[]
  inner?: XY[]
  tip: XY
  root: string
  mid: string
  light: string
  /** Optional dark tip (fox), as [colour, canvas-y below which it applies]. */
  earTip?: [string, number]
  skin?: string
  plush?: string
  tuft?: string
  strand: [number, number]
  hang?: boolean
}

function inside(p: XY, poly: XY[]) {
  let c = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) c = !c
  }
  return c
}

function edgeDist(p: XY, poly: XY[]) {
  let best = Infinity
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [ax, ay] = poly[j]
    const [bx, by] = poly[i]
    const dx = bx - ax
    const dy = by - ay
    const t = Math.max(0, Math.min(1, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy || 1)))
    best = Math.min(best, Math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy)))
  }
  return best
}

/** Smooth closed outline through control points (Catmull-Rom), as a polygon. */
export function smooth(ctrl: XY[], per = 10): XY[] {
  const out: XY[] = []
  const n = ctrl.length
  for (let i = 0; i < n; i++) {
    const p0 = ctrl[(i - 1 + n) % n]
    const p1 = ctrl[i]
    const p2 = ctrl[(i + 1) % n]
    const p3 = ctrl[(i + 2) % n]
    for (let k = 0; k < per; k++) {
      const t = k / per
      const t2 = t * t
      const t3 = t2 * t
      const f = (a: number, b: number, c: number, d: number) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3)
      out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])])
    }
  }
  return out
}

const hex = (c: string) => new THREE.Color(c)
const mixHex = (a: string, b: string, t: number) => '#' + hex(a).lerp(hex(b), t).getHexString()

export function paintEar(spec: PlushSpec): HTMLCanvasElement {
  const { w, h, outline, inner, tip } = spec
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')!
  ctx.lineCap = 'round'
  let seed = 7
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647
  const cx = outline.reduce((s, p) => s + p[0], 0) / outline.length
  const cy = outline.reduce((s, p) => s + p[1], 0) / outline.length
  const bbox = outline.reduce((b, p) => [Math.min(b[0], p[0]), Math.min(b[1], p[1]), Math.max(b[2], p[0]), Math.max(b[3], p[1])], [Infinity, Infinity, -Infinity, -Infinity])

  const flow = (p: XY, fanOut: number): XY => {
    let dx = tip[0] - p[0]
    let dy = tip[1] - p[1]
    let l = Math.hypot(dx, dy) || 1
    dx /= l
    dy /= l
    // Near the outline, hairs fan outward over the edge.
    const e = Math.max(0, 1 - edgeDist(p, outline) / fanOut)
    let ox = p[0] - cx
    let oy = p[1] - cy
    l = Math.hypot(ox, oy) || 1
    ox /= l
    oy /= l
    const fx = dx * (1 - e * 0.75) + ox * e * 0.9
    const fy = dy * (1 - e * 0.75) + oy * e * 0.9
    l = Math.hypot(fx, fy) || 1
    return [fx / l, fy / l]
  }
  const hair = (p: XY, dir: XY, len: number, width: number, color: string, alpha: number) => {
    const bend = (rnd() - 0.5) * len * 0.35
    const ex = p[0] + dir[0] * len
    const ey = p[1] + dir[1] * len
    ctx.strokeStyle = color
    ctx.globalAlpha = alpha
    ctx.lineWidth = width
    ctx.beginPath()
    ctx.moveTo(p[0], p[1])
    ctx.quadraticCurveTo((p[0] + ex) / 2 - dir[1] * bend, (p[1] + ey) / 2 + dir[0] * bend, ex, ey)
    ctx.stroke()
  }
  const sample = (poly: XY[]): XY => {
    for (let k = 0; k < 50; k++) {
      const p: XY = [bbox[0] + rnd() * (bbox[2] - bbox[0]), bbox[1] + rnd() * (bbox[3] - bbox[1])]
      if (inside(p, poly)) return p
    }
    return [cx, cy]
  }
  const colorAt = (p: XY, k: number) => {
    // Painted key light from the upper left, plus a darker base: the ear
    // already has form before the 3D lighting is applied.
    const lightness = THREE.MathUtils.clamp(0.5 + ((cx - p[0]) / w) * 0.6 + ((cy - p[1]) / h) * 0.5, 0, 1)
    let col = mixHex(spec.root, spec.mid, k)
    col = mixHex(col, spec.light, Math.pow(lightness, 2) * k * 0.7)
    if (spec.earTip && (spec.hang ? p[1] > spec.earTip[1] : p[1] < spec.earTip[1])) col = mixHex(col, spec.earTip[0], 0.85)
    return col
  }

  // 1. Base silhouette: soft dark under-coat.
  ctx.save()
  ctx.beginPath()
  outline.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])))
  ctx.closePath()
  ctx.fillStyle = spec.root
  ctx.fill()
  ctx.restore()

  // 2. Hair, back to front: short dark first, longer and lighter on top.
  const [l0, l1] = spec.strand
  const N = 16000
  for (let i = 0; i < N; i++) {
    const k = i / N
    const p = sample(outline)
    const len = l0 + (l1 - l0) * (0.4 + 0.6 * rnd()) * (0.6 + 0.6 * k)
    hair(p, flow(p, 46), len, 1 + rnd() * 1.6, colorAt(p, k * (0.6 + 0.4 * rnd())), 0.5 + 0.45 * rnd())
  }
  // 3. Edge fluff: hairs rooted just inside the outline, reaching past it.
  for (let i = 0; i < 3200; i++) {
    const a = outline[Math.floor(rnd() * outline.length)]
    const p: XY = [a[0] + (cx - a[0]) * 0.05 * rnd(), a[1] + (cy - a[1]) * 0.05 * rnd()]
    const len = l1 * (0.6 + 0.9 * rnd())
    hair(p, flow(p, 80), len, 0.8 + rnd() * 1.2, colorAt(p, 0.5 + 0.5 * rnd()), 0.35 + 0.5 * rnd())
  }

  // 4. Inner ear: pink plush with long pale tufts fanning out of its base.
  if (inner && spec.skin && spec.plush && spec.tuft) {
    ctx.save()
    ctx.globalAlpha = 1
    ctx.beginPath()
    inner.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])))
    ctx.closePath()
    const ib = inner.reduce((b, p) => [Math.min(b[0], p[0]), Math.min(b[1], p[1]), Math.max(b[2], p[0]), Math.max(b[3], p[1])], [Infinity, Infinity, -Infinity, -Infinity])
    const g = ctx.createLinearGradient(0, ib[1], 0, ib[3])
    g.addColorStop(0, mixHex(spec.skin, '#000000', 0.15))
    g.addColorStop(1, mixHex(spec.skin, '#000000', 0.32))
    ctx.fillStyle = g
    ctx.fill()
    ctx.clip()
    for (let i = 0; i < 6000; i++) {
      const p = sample(inner)
      const t = (p[1] - ib[1]) / (ib[3] - ib[1])
      hair(p, flow(p, 30), l0 * (0.5 + rnd() * 0.6), 0.9 + rnd() * 1.1, mixHex(spec.plush, '#ffffff', rnd() * 0.15 + (1 - t) * 0.08), 0.45 + 0.4 * rnd())
    }
    ctx.restore()
    const icx = (ib[0] + ib[2]) / 2
    const baseY = spec.hang ? ib[1] : ib[3]
    for (let i = 0; i < 420; i++) {
      const p: XY = [icx + (rnd() - 0.5) * (ib[2] - ib[0]) * 0.65, baseY + (spec.hang ? 1 : -1) * rnd() * (ib[3] - ib[1]) * 0.3]
      if (!inside(p, inner)) continue
      const spread = (p[0] - icx) / ((ib[2] - ib[0]) / 2)
      const ang = (spec.hang ? Math.PI / 2 : -Math.PI / 2) + spread * 0.7 + (rnd() - 0.5) * 0.4
      hair(p, [Math.cos(ang), Math.sin(ang)], l1 * (1.1 + rnd() * 1.3), 0.7 + rnd() * 1.1, mixHex(spec.tuft, spec.plush, rnd() * 0.3), 0.3 + 0.45 * rnd())
    }
  }
  ctx.globalAlpha = 1
  return c
}

/** A card for the painted ear, gently curved (convex back, slightly cupped
 *  front), base at y = 0, rising (or hanging) along y. */
export function plushCard(spec: PlushSpec) {
  const aspect = spec.h / spec.w
  const cw = spec.cardW
  const chh = cw * aspect
  const g = new THREE.PlaneGeometry(cw, chh, 24, 24)
  const p = g.getAttribute('position') as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i) / (cw / 2)
    const y = p.getY(i) / (chh / 2)
    p.setZ(i, -0.14 * cw * x * x - 0.04 * cw * y * y)
  }
  g.computeVertexNormals()
  // The ear's base (canvas bottom, or top when hanging) sits at y = 0.
  g.translate(0, spec.hang ? -chh / 2 : chh / 2, 0)
  const tex = new THREE.CanvasTexture(paintEar(spec))
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 8
  const mat = new THREE.MeshStandardMaterial({ map: tex, transparent: true, alphaTest: 0.02, roughness: 0.95, metalness: 0, side: THREE.DoubleSide, depthWrite: false })
  const mesh = new THREE.Mesh(g, mat)
  mesh.castShadow = true
  // Shadows follow the painted silhouette, not the card's rectangle.
  mesh.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: tex, alphaTest: 0.5 })
  mesh.renderOrder = 2
  return mesh
}
