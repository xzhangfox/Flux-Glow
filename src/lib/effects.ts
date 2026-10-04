import { FaceLandmarker } from '@mediapipe/tasks-vision'
import { connectorsToLoop } from './landmarks'
import { LIPS_LOOP, OVAL_LOOP } from './deform'
import type { EditParams } from './pipeline'

// Face-tracked AR effects — ears, masks, props — drawn as vectors straight
// onto the frame each time it's processed (every live frame, or once for a
// still), from the face mesh's landmark positions AFTER Shape has moved
// them, so ears sit on the reshaped head and a mask hugs the slimmed jaw.
// All art is procedural Canvas 2D: it's crisp at any resolution, needs no
// downloaded assets, and keeps the app's "nothing leaves your device".
//
// Anchors come from real landmarks (temples, forehead, nose tip, eye and
// lip loops), and everything is drawn in the face's own rotated frame
// where 1 unit = the distance between the eyes — so effects scale with the
// face, roll with a tilted head, and narrow naturally on a turned one.

export interface Pt {
  x: number
  y: number
}

export interface EffectDef {
  id: string
  label: string
  /** Shape/Beauty nudges layered on top of the user's own sliders while the
   *  effect is on (e.g. the cute animal ears come with a baby-face boost). */
  boost?: Partial<Record<keyof EditParams, number>>
}

const BABY = { eyes: 0.3, eyeHeight: 0.2, noseTip: 0.25, chin: -0.3, face: 0.15, forehead: 0.2, smoothness: 0.15, fillLight: 0.15 }

export const EFFECTS: EffectDef[] = [
  { id: 'none', label: 'None' },
  { id: 'kitty', label: 'Kitty', boost: BABY },
  { id: 'puppy', label: 'Puppy', boost: BABY },
  { id: 'bunny', label: 'Bunny', boost: BABY },
  { id: 'bear', label: 'Bear', boost: BABY },
  { id: 'doll', label: 'Doll', boost: { ...BABY, eyes: 0.45, eyeHeight: 0.3, chin: -0.4, whitening: 0.25 } },
  { id: 'spider', label: 'Spider' },
  { id: 'bat', label: 'Bat' },
  { id: 'crown', label: 'Crown' },
  { id: 'hearts', label: 'Hearts' },
  { id: 'devil', label: 'Devil' },
  { id: 'angel', label: 'Angel' },
  { id: 'stars', label: 'Stars' },
]

export const findEffect = (id: string) => EFFECTS.find((e) => e.id === id) ?? EFFECTS[0]

/** The user's params with the effect's boost added (clamped to each
 *  slider's own range). */
export function withEffectBoost<T extends EditParams>(params: T): T {
  const boost = findEffect(params.effectId).boost
  if (!boost) return params
  const out = { ...params }
  for (const [k, b] of Object.entries(boost) as [keyof EditParams, number][]) {
    const lo = BEAUTY_KEYS.has(k) ? 0 : -1
    ;(out[k] as number) = Math.min(1, Math.max(lo, (params[k] as number) + b))
  }
  return out
}
const BEAUTY_KEYS = new Set<keyof EditParams>(['smoothness', 'whitening', 'acneRemoval', 'wrinkleRemoval', 'mouthCornerSmooth', 'fillLight'])

const LEFT_EYE_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
const RIGHT_EYE_LOOP = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)

const add = (a: Pt, b: Pt, k = 1): Pt => ({ x: a.x + b.x * k, y: a.y + b.y * k })
const lerp = (a: Pt, b: Pt, t: number): Pt => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t })
const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y)
const centroid = (loop: number[], P: Pt[]): Pt => {
  let x = 0
  let y = 0
  for (const i of loop) {
    x += P[i].x
    y += P[i].y
  }
  return { x: x / loop.length, y: y / loop.length }
}

interface Face {
  P: Pt[]
  E: number
  roll: number
  right: Pt
  up: Pt
  eyeL: Pt
  eyeR: Pt
  eyeLoopL: number[]
  eyeLoopR: number[]
  templeL: Pt
  templeR: Pt
  cheekL: Pt
  cheekR: Pt
  top: Pt
  nose: Pt
  /** How far each side of the face is turned toward camera (≈1 frontal). */
  sideL: number
  sideR: number
  /** Lip gap relative to mouth width: ~0 closed, ~0.4 wide open. */
  mouthOpen: number
  center: Pt
}

function faceGeometry(P: Pt[]): Face {
  const a = centroid(LEFT_EYE_LOOP, P)
  const b = centroid(RIGHT_EYE_LOOP, P)
  // Landmark indices are anatomical, so which one shows on the image's left
  // depends on mirroring — sort by position instead of trusting the index.
  const [eyeL, eyeR, eyeLoopL, eyeLoopR] = a.x <= b.x ? [a, b, LEFT_EYE_LOOP, RIGHT_EYE_LOOP] : [b, a, RIGHT_EYE_LOOP, LEFT_EYE_LOOP]
  const E = Math.max(1, dist(eyeL, eyeR))
  const roll = Math.atan2(eyeR.y - eyeL.y, eyeR.x - eyeL.x)
  const right = { x: Math.cos(roll), y: Math.sin(roll) }
  const up = { x: Math.sin(roll), y: -Math.cos(roll) }
  const across = (p: Pt) => p.x * right.x + p.y * right.y
  const [templeL, templeR] = across(P[21]) <= across(P[251]) ? [P[21], P[251]] : [P[251], P[21]]
  const [cheekL, cheekR] = across(P[234]) <= across(P[454]) ? [P[234], P[454]] : [P[454], P[234]]
  const mid = across(P[168])
  const hwL = Math.max(1, mid - across(cheekL))
  const hwR = Math.max(1, across(cheekR) - mid)
  const avg = (hwL + hwR) / 2
  const mouthW = Math.max(1, dist(P[61], P[291]))
  return {
    P, E, roll, right, up, eyeL, eyeR, eyeLoopL, eyeLoopR, templeL, templeR, cheekL, cheekR,
    top: P[10],
    nose: P[4],
    // Square-rooted: real perspective shrinks the far side less than the
    // half-width ratio suggests (ears stand off the head, not on it).
    sideL: Math.min(1.15, Math.max(0.75, Math.sqrt(hwL / avg))),
    sideR: Math.min(1.15, Math.max(0.75, Math.sqrt(hwR / avg))),
    mouthOpen: dist(P[13], P[14]) / mouthW,
    center: lerp(P[168], P[4], 0.3),
  }
}

/** Run `fn` in a local frame at `o`, rotated by `rot`, where 1 unit = `s` px. */
function frame(ctx: CanvasRenderingContext2D, o: Pt, rot: number, s: number, fn: () => void, mirror = false) {
  ctx.save()
  ctx.translate(o.x, o.y)
  ctx.rotate(rot)
  ctx.scale(mirror ? -s : s, s)
  fn()
  ctx.restore()
}

function radial(ctx: CanvasRenderingContext2D, c: Pt, r: number, inner: string, outer: string) {
  const g = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, r)
  g.addColorStop(0, inner)
  g.addColorStop(1, outer)
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.arc(c.x, c.y, r, 0, Math.PI * 2)
  ctx.fill()
}

function blush(ctx: CanvasRenderingContext2D, f: Face, color = 'rgba(255,110,140,0.32)') {
  for (const [cheek, side] of [[f.cheekL, f.sideL], [f.cheekR, f.sideR]] as const) {
    const c = add(lerp(cheek, f.nose, 0.42), f.up, -0.12 * f.E)
    frame(ctx, c, f.roll, f.E * side, () => {
      ctx.scale(1, 0.62)
      radial(ctx, { x: 0, y: 0 }, 0.36, color, 'rgba(255,110,140,0)')
    })
  }
}

// Where an animal ear sits: on the head's outline above the temple,
// leaning outward. `t` slides it toward (0) or away from (1) the midline.
function earAnchor(f: Face, side: -1 | 1, t: number, lift: number, out = 0): Pt {
  const temple = side < 0 ? f.templeL : f.templeR
  return add(add(lerp(f.top, temple, t), f.up, lift * f.E), f.right, side * out * f.E)
}

function pointyEar(ctx: CanvasRenderingContext2D, outer: [string, string], inner: string) {
  // Designed as the right-hand ear (outward = +x, up = −y).
  ctx.beginPath()
  ctx.moveTo(-0.36, 0.08)
  ctx.quadraticCurveTo(-0.3, -0.55, 0.06, -0.95)
  ctx.quadraticCurveTo(0.14, -1.0, 0.18, -0.9)
  ctx.quadraticCurveTo(0.4, -0.45, 0.4, 0.08)
  ctx.quadraticCurveTo(0, 0.18, -0.36, 0.08)
  const g = ctx.createLinearGradient(0, -1, 0, 0.1)
  g.addColorStop(0, outer[0])
  g.addColorStop(1, outer[1])
  ctx.fillStyle = g
  ctx.fill()
  ctx.lineWidth = 0.025
  ctx.strokeStyle = 'rgba(60,40,30,0.35)'
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(-0.2, 0.02)
  ctx.quadraticCurveTo(-0.15, -0.45, 0.07, -0.74)
  ctx.quadraticCurveTo(0.24, -0.4, 0.25, 0.02)
  ctx.closePath()
  const gi = ctx.createLinearGradient(0, -0.75, 0, 0.05)
  gi.addColorStop(0, inner)
  gi.addColorStop(1, 'rgba(255,170,190,0.55)')
  ctx.fillStyle = gi
  ctx.fill()
}

function kitty(ctx: CanvasRenderingContext2D, f: Face) {
  for (const side of [-1, 1] as const) {
    const s = side < 0 ? f.sideL : f.sideR
    frame(ctx, earAnchor(f, side, 0.62, 0.42), f.roll + side * 0.32, f.E * 0.95 * s, () => pointyEar(ctx, ['#fbf3ea', '#e9d3bf'], '#ff9db5'), side < 0)
  }
  blush(ctx, f)
  // Whiskers fan out from beside the nose wings.
  for (const side of [-1, 1] as const) {
    const wing = f.P[side < 0 ? (f.P[98].x < f.P[327].x ? 98 : 327) : f.P[98].x < f.P[327].x ? 327 : 98]
    const base = add(add(wing, f.right, side * 0.12 * f.E), f.up, -0.05 * f.E)
    frame(ctx, base, f.roll, f.E * (side < 0 ? f.sideL : f.sideR), () => {
      ctx.strokeStyle = 'rgba(40,30,30,0.75)'
      ctx.lineWidth = 0.018
      ctx.lineCap = 'round'
      for (const a of [-0.16, 0.02, 0.2]) {
        ctx.beginPath()
        ctx.moveTo(0, a * 0.25)
        ctx.quadraticCurveTo(side * 0.4, a * 0.9 - 0.04, side * 0.78, a * 1.7)
        ctx.stroke()
      }
    })
  }
  frame(ctx, add(f.nose, f.up, 0.02 * f.E), f.roll, f.E, () => {
    ctx.beginPath()
    ctx.moveTo(-0.13, -0.06)
    ctx.quadraticCurveTo(0, -0.12, 0.13, -0.06)
    ctx.quadraticCurveTo(0.1, 0.04, 0, 0.09)
    ctx.quadraticCurveTo(-0.1, 0.04, -0.13, -0.06)
    const g = ctx.createLinearGradient(0, -0.1, 0, 0.1)
    g.addColorStop(0, '#ffb3c6')
    g.addColorStop(1, '#f06a8f')
    ctx.fillStyle = g
    ctx.fill()
    radial(ctx, { x: -0.03, y: -0.05 }, 0.035, 'rgba(255,255,255,0.9)', 'rgba(255,255,255,0)')
  })
}

function puppy(ctx: CanvasRenderingContext2D, f: Face) {
  for (const side of [-1, 1] as const) {
    const s = side < 0 ? f.sideL : f.sideR
    frame(ctx, earAnchor(f, side, 1, 0.22, 0.22), f.roll + side * 0.3, f.E * 0.9 * s, () => {
      ctx.beginPath()
      ctx.moveTo(-0.12, -0.05)
      ctx.bezierCurveTo(0.35, -0.2, 0.62, 0.55, 0.42, 1.2)
      ctx.bezierCurveTo(0.32, 1.42, 0.02, 1.4, -0.04, 1.12)
      ctx.bezierCurveTo(-0.14, 0.7, -0.22, 0.25, -0.12, -0.05)
      const g = ctx.createLinearGradient(0, 0, 0.3, 1.3)
      g.addColorStop(0, '#a8714a')
      g.addColorStop(1, '#6b4228')
      ctx.fillStyle = g
      ctx.fill()
      ctx.beginPath()
      ctx.moveTo(0.02, 0.2)
      ctx.bezierCurveTo(0.3, 0.3, 0.42, 0.75, 0.3, 1.12)
      ctx.bezierCurveTo(0.22, 1.25, 0.08, 1.22, 0.05, 1.05)
      ctx.closePath()
      ctx.fillStyle = 'rgba(70,40,22,0.45)'
      ctx.fill()
    }, side < 0)
  }
  // Tongue when the mouth opens.
  const open = Math.min(1, Math.max(0, (f.mouthOpen - 0.12) / 0.3))
  if (open > 0) {
    frame(ctx, f.P[14], f.roll, f.E, () => {
      const len = 0.15 + 0.55 * open
      ctx.beginPath()
      ctx.moveTo(-0.17, 0)
      ctx.bezierCurveTo(-0.2, len * 0.7, -0.12, len, 0, len)
      ctx.bezierCurveTo(0.12, len, 0.2, len * 0.7, 0.17, 0)
      ctx.closePath()
      const g = ctx.createLinearGradient(0, 0, 0, len)
      g.addColorStop(0, '#e2506e')
      g.addColorStop(1, '#ff8aa3')
      ctx.fillStyle = g
      ctx.fill()
      ctx.strokeStyle = 'rgba(150,30,60,0.6)'
      ctx.lineWidth = 0.018
      ctx.beginPath()
      ctx.moveTo(0, 0.03)
      ctx.lineTo(0, len * 0.6)
      ctx.stroke()
    })
  }
  frame(ctx, add(f.nose, f.up, 0.03 * f.E), f.roll, f.E, () => {
    ctx.beginPath()
    ctx.ellipse(0, 0, 0.22, 0.15, 0, 0, Math.PI * 2)
    const g = ctx.createRadialGradient(-0.05, -0.06, 0.01, 0, 0, 0.24)
    g.addColorStop(0, '#4a4a4f')
    g.addColorStop(1, '#0d0d10')
    ctx.fillStyle = g
    ctx.fill()
    radial(ctx, { x: -0.07, y: -0.06 }, 0.06, 'rgba(255,255,255,0.75)', 'rgba(255,255,255,0)')
  })
}

function bunny(ctx: CanvasRenderingContext2D, f: Face) {
  for (const side of [-1, 1] as const) {
    const s = side < 0 ? f.sideL : f.sideR
    frame(ctx, earAnchor(f, side, 0.42, 0.42), f.roll + side * 0.16, f.E * s, () => {
      ctx.beginPath()
      ctx.moveTo(-0.2, 0.08)
      ctx.bezierCurveTo(-0.32, -0.6, -0.28, -1.55, 0.02, -1.62)
      ctx.bezierCurveTo(0.3, -1.55, 0.3, -0.6, 0.2, 0.08)
      ctx.closePath()
      const g = ctx.createLinearGradient(-0.3, 0, 0.3, 0)
      g.addColorStop(0, '#e9e6ec')
      g.addColorStop(0.5, '#ffffff')
      g.addColorStop(1, '#dcd8e0')
      ctx.fillStyle = g
      ctx.fill()
      ctx.strokeStyle = 'rgba(120,110,130,0.35)'
      ctx.lineWidth = 0.02
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(-0.09, 0)
      ctx.bezierCurveTo(-0.16, -0.6, -0.14, -1.3, 0.02, -1.36)
      ctx.bezierCurveTo(0.17, -1.3, 0.17, -0.6, 0.09, 0)
      ctx.closePath()
      const gi = ctx.createLinearGradient(0, -1.3, 0, 0)
      gi.addColorStop(0, '#ffc2d3')
      gi.addColorStop(1, '#ff9fb8')
      ctx.fillStyle = gi
      ctx.fill()
    }, side < 0)
  }
  blush(ctx, f, 'rgba(255,130,160,0.3)')
  frame(ctx, add(f.nose, f.up, 0.02 * f.E), f.roll, f.E, () => {
    ctx.beginPath()
    ctx.ellipse(0, 0, 0.09, 0.065, 0, 0, Math.PI * 2)
    ctx.fillStyle = '#ff8fab'
    ctx.fill()
  })
  // Two little front teeth just under the upper lip.
  frame(ctx, f.P[13], f.roll, f.E, () => {
    ctx.fillStyle = '#ffffff'
    ctx.strokeStyle = 'rgba(120,110,120,0.6)'
    ctx.lineWidth = 0.012
    for (const x of [-0.075, 0.005]) {
      ctx.beginPath()
      ctx.roundRect(x, -0.01, 0.07, 0.11, [0, 0, 0.02, 0.02])
      ctx.fill()
      ctx.stroke()
    }
  })
}

function bear(ctx: CanvasRenderingContext2D, f: Face) {
  for (const side of [-1, 1] as const) {
    const s = side < 0 ? f.sideL : f.sideR
    frame(ctx, earAnchor(f, side, 0.78, 0.3), f.roll, f.E * s, () => {
      radial(ctx, { x: 0, y: 0 }, 0.34, '#9a6a45', '#6e4a2f')
      radial(ctx, { x: 0, y: 0.02 }, 0.19, '#e8bf9a', '#c99a74')
    })
  }
  blush(ctx, f)
  frame(ctx, add(f.nose, f.up, 0.02 * f.E), f.roll, f.E, () => {
    ctx.beginPath()
    ctx.ellipse(0, 0, 0.15, 0.1, 0, 0, Math.PI * 2)
    ctx.fillStyle = '#3a2618'
    ctx.fill()
    radial(ctx, { x: -0.04, y: -0.04 }, 0.04, 'rgba(255,255,255,0.7)', 'rgba(255,255,255,0)')
  })
}

function doll(ctx: CanvasRenderingContext2D, f: Face) {
  blush(ctx, f, 'rgba(255,105,140,0.38)')
  // Lip tint.
  ctx.save()
  ctx.globalCompositeOperation = 'multiply'
  ctx.beginPath()
  LIPS_LOOP.forEach((i, k) => (k ? ctx.lineTo(f.P[i].x, f.P[i].y) : ctx.moveTo(f.P[i].x, f.P[i].y)))
  ctx.closePath()
  ctx.fillStyle = 'rgba(255,90,130,0.35)'
  ctx.fill()
  ctx.restore()
  // Freckles across the nose and cheeks (fixed pattern, so they don't
  // shimmer frame to frame).
  frame(ctx, lerp(f.P[168], f.nose, 0.55), f.roll, f.E, () => {
    let seed = 7
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
    ctx.fillStyle = 'rgba(140,70,50,0.5)'
    for (let i = 0; i < 30; i++) {
      const side = i % 2 ? 1 : -1
      const x = side * (0.12 + rnd() * 0.5)
      const y = -0.05 + rnd() * 0.3 + Math.abs(x) * 0.15
      ctx.beginPath()
      ctx.arc(x, y, 0.016 + rnd() * 0.014, 0, Math.PI * 2)
      ctx.fill()
    }
  })
  // Under-eye glow (aegyo-sal) and a sparkle highlight on each eye.
  for (const [eye, loop] of [[f.eyeL, f.eyeLoopL], [f.eyeR, f.eyeLoopR]] as const) {
    const w = Math.max(...loop.map((i) => dist(f.P[i], eye)))
    frame(ctx, add(eye, f.up, -0.22 * f.E), f.roll, w, () => {
      ctx.scale(1, 0.32)
      radial(ctx, { x: 0, y: 0 }, 1, 'rgba(255,235,230,0.35)', 'rgba(255,235,230,0)')
    })
    sparkle(ctx, add(add(eye, f.up, 0.05 * f.E), f.right, 0.06 * f.E), f.E * 0.07, 'rgba(255,255,255,0.95)')
  }
}

function sparkle(ctx: CanvasRenderingContext2D, c: Pt, r: number, color: string) {
  ctx.save()
  ctx.translate(c.x, c.y)
  ctx.fillStyle = color
  ctx.shadowColor = color
  ctx.shadowBlur = r * 1.5
  ctx.beginPath()
  for (let i = 0; i < 8; i++) {
    const a = (i * Math.PI) / 4
    const rr = i % 2 ? r * 0.28 : r
    ctx.lineTo(Math.cos(a) * rr, Math.sin(a) * rr)
  }
  ctx.closePath()
  ctx.fill()
  ctx.restore()
}

// ---- Masks ----------------------------------------------------------------

/** The face oval pushed outward (more over the forehead/hair), as a path. */
function headOutline(f: Face, grow: number, crown: number): Pt[] {
  const c = f.center
  return OVAL_LOOP.map((i) => {
    const p = f.P[i]
    const d = { x: p.x - c.x, y: p.y - c.y }
    const upness = Math.max(0, (d.x * f.up.x + d.y * f.up.y) / Math.hypot(d.x, d.y))
    const k = grow + crown * upness * upness
    return { x: p.x + d.x * k, y: p.y + d.y * k }
  })
}

function pathOf(ctx: CanvasRenderingContext2D, pts: Pt[]) {
  ctx.beginPath()
  pts.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)))
  ctx.closePath()
}

let shadeCanvas: HTMLCanvasElement | null = null
/** Re-light a flat mask with the face's own shading: a soft grayscale of
 *  the underlying photo, multiplied over the mask inside `clip`. Computed
 *  by hand on a small canvas (not ctx.filter, which Safari lacks). */
function shadeMask(ctx: CanvasRenderingContext2D, photo: HTMLCanvasElement, clip: () => void, gain: number) {
  const w = Math.max(1, Math.round(photo.width / 4))
  const h = Math.max(1, Math.round(photo.height / 4))
  shadeCanvas ??= document.createElement('canvas')
  shadeCanvas.width = w
  shadeCanvas.height = h
  const sctx = shadeCanvas.getContext('2d', { willReadFrequently: true })!
  sctx.drawImage(photo, 0, 0, w, h)
  const img = sctx.getImageData(0, 0, w, h)
  const d = img.data
  for (let i = 0; i < d.length; i += 4) {
    const l = Math.min(255, (0.3 * d[i] + 0.59 * d[i + 1] + 0.11 * d[i + 2]) * gain)
    d[i] = d[i + 1] = d[i + 2] = l
  }
  sctx.putImageData(img, 0, 0)
  ctx.save()
  clip()
  ctx.clip()
  ctx.globalCompositeOperation = 'multiply'
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(shadeCanvas, 0, 0, photo.width, photo.height)
  ctx.restore()
}

function spider(ctx: CanvasRenderingContext2D, f: Face, photo: HTMLCanvasElement) {
  const outline = headOutline(f, 0.06, 0.32)
  pathOf(ctx, outline)
  const g = ctx.createRadialGradient(f.center.x, f.center.y, 0, f.center.x, f.center.y, f.E * 2.2)
  g.addColorStop(0, '#e2222e')
  g.addColorStop(1, '#a5101a')
  ctx.fillStyle = g
  ctx.fill()
  shadeMask(ctx, photo, () => pathOf(ctx, outline), 1.55)

  // Web: radial threads from between the eyes, sagging rings between them.
  ctx.save()
  pathOf(ctx, outline)
  ctx.clip()
  const W = f.P[168]
  ctx.strokeStyle = 'rgba(10,10,12,0.85)'
  ctx.lineWidth = f.E * 0.018
  const spokes = outline.filter((_, i) => i % 2 === 0)
  for (const q of spokes) {
    ctx.beginPath()
    ctx.moveTo(W.x, W.y)
    ctx.lineTo(W.x + (q.x - W.x) * 1.1, W.y + (q.y - W.y) * 1.1)
    ctx.stroke()
  }
  for (const k of [0.16, 0.3, 0.45, 0.6, 0.75, 0.9]) {
    ctx.beginPath()
    spokes.forEach((q, i) => {
      const a = lerp(W, q, k)
      const b = lerp(W, spokes[(i + 1) % spokes.length], k)
      const m = lerp(W, lerp(a, b, 0.5), 0.93)
      if (i === 0) ctx.moveTo(a.x, a.y)
      ctx.quadraticCurveTo(m.x, m.y, b.x, b.y)
    })
    ctx.stroke()
  }
  ctx.restore()

  // The big white lenses, outer corners swept up.
  for (const [eye, side] of [[f.eyeL, -1], [f.eyeR, 1]] as const) {
    const c = add(add(eye, f.up, 0.04 * f.E), f.right, side * 0.04 * f.E)
    frame(ctx, c, f.roll, f.E, () => {
      ctx.beginPath()
      ctx.moveTo(-0.24, 0.1)
      ctx.bezierCurveTo(-0.3, -0.12, -0.02, -0.24, 0.3, -0.2)
      ctx.bezierCurveTo(0.28, 0.02, 0.1, 0.17, -0.24, 0.1)
      ctx.closePath()
      ctx.lineWidth = 0.075
      ctx.strokeStyle = '#0b0b0d'
      ctx.stroke()
      const lg = ctx.createLinearGradient(0, -0.2, 0, 0.15)
      lg.addColorStop(0, '#ffffff')
      lg.addColorStop(1, '#cfd6de')
      ctx.fillStyle = lg
      ctx.fill()
    }, side < 0)
  }
}

function bat(ctx: CanvasRenderingContext2D, f: Face, photo: HTMLCanvasElement) {
  const outline = headOutline(f, 0.08, 0.36)
  // Keep only the upper part of the head outline (cheek to cheek over the
  // top), then close it along a line under the nose so the mouth and chin
  // stay bare, as with the real cowl.
  const c = f.center
  const isUpper = (p: Pt) => (p.x - c.x) * f.up.x + (p.y - c.y) * f.up.y > -0.25 * f.E
  const upper: Pt[] = []
  const n = outline.length
  // Start from the outline point nearest the left cheek and walk over the top.
  let start = 0
  let best = Infinity
  outline.forEach((p, i) => {
    const d = dist(p, f.cheekL)
    if (d < best) {
      best = d
      start = i
    }
  })
  const dir = isUpper(outline[(start + 1) % n]) ? 1 : -1
  for (let k = 0; k < n; k++) {
    const p = outline[(start + dir * k + n) % n]
    if (k > 2 && !isUpper(p)) break
    upper.push(p)
  }
  const under = add(f.P[2], f.up, -0.03 * f.E)
  const shape = [...upper, add(lerp(f.cheekR, under, 0.55), f.up, -0.05 * f.E), add(under, f.right, 0.12 * f.E), add(under, f.right, -0.12 * f.E), add(lerp(f.cheekL, under, 0.55), f.up, -0.05 * f.E)]

  const off = document.createElement('canvas')
  off.width = photo.width
  off.height = photo.height
  const o = off.getContext('2d')!
  // Ears, then the cowl.
  for (const side of [-1, 1] as const) {
    frame(o, earAnchor(f, side, 0.6, 0.55), f.roll + side * 0.1, f.E, () => {
      o.beginPath()
      o.moveTo(-0.25, 0.25)
      o.lineTo(0.06, -0.85)
      o.lineTo(0.3, 0.25)
      o.closePath()
      o.fillStyle = '#1f2126'
      o.fill()
    }, side < 0)
  }
  pathOf(o, shape)
  const g = o.createLinearGradient(f.top.x, f.top.y - f.E, f.P[2].x, f.P[2].y)
  g.addColorStop(0, '#3a3d45')
  g.addColorStop(1, '#16171b')
  o.fillStyle = g
  o.fill()
  shadeMask(o, photo, () => pathOf(o, shape), 1.9)
  // Eye openings: cut through so the real eyes show, with a dark rim.
  for (const loop of [f.eyeLoopL, f.eyeLoopR]) {
    const eye = centroid(loop, f.P)
    const pts = loop.map((i) => add(eye, { x: f.P[i].x - eye.x, y: f.P[i].y - eye.y }, 1.45))
    o.save()
    o.globalCompositeOperation = 'destination-out'
    pathOf(o, pts)
    o.fill()
    o.restore()
  }
  ctx.drawImage(off, 0, 0)
  for (const loop of [f.eyeLoopL, f.eyeLoopR]) {
    const eye = centroid(loop, f.P)
    const pts = loop.map((i) => add(eye, { x: f.P[i].x - eye.x, y: f.P[i].y - eye.y }, 1.5))
    ctx.save()
    pathOf(ctx, pts)
    ctx.lineWidth = f.E * 0.05
    ctx.strokeStyle = 'rgba(8,8,10,0.85)'
    ctx.stroke()
    ctx.restore()
  }
}

// ---- Props -----------------------------------------------------------------

function crown(ctx: CanvasRenderingContext2D, f: Face) {
  frame(ctx, add(f.top, f.up, 0.62 * f.E), f.roll, f.E, () => {
    ctx.shadowColor = 'rgba(0,0,0,0.35)'
    ctx.shadowBlur = 0.08 * f.E
    ctx.beginPath()
    ctx.moveTo(-0.7, 0.25)
    ctx.lineTo(-0.78, -0.3)
    ctx.lineTo(-0.4, 0)
    ctx.lineTo(-0.2, -0.45)
    ctx.lineTo(0, -0.08)
    ctx.lineTo(0.2, -0.45)
    ctx.lineTo(0.4, 0)
    ctx.lineTo(0.78, -0.3)
    ctx.lineTo(0.7, 0.25)
    ctx.closePath()
    const g = ctx.createLinearGradient(0, -0.45, 0, 0.25)
    g.addColorStop(0, '#fff3b0')
    g.addColorStop(0.45, '#e6b93a')
    g.addColorStop(1, '#a8761c')
    ctx.fillStyle = g
    ctx.fill()
    ctx.shadowBlur = 0
    ctx.strokeStyle = '#8a5f12'
    ctx.lineWidth = 0.025
    ctx.stroke()
    ctx.fillStyle = 'rgba(120,80,10,0.6)'
    ctx.fillRect(-0.7, 0.1, 1.4, 0.05)
    for (const [x, y, col] of [[-0.78, -0.3, '#ff4d6d'], [-0.2, -0.45, '#4dc3ff'], [0.2, -0.45, '#4dc3ff'], [0.78, -0.3, '#ff4d6d'], [0, 0.04, '#b04dff']] as const) {
      radial(ctx, { x, y }, 0.07, '#ffffff', col)
    }
  })
}

function hearts(ctx: CanvasRenderingContext2D, f: Face) {
  const heart = () => {
    ctx.beginPath()
    ctx.moveTo(0, 0.32)
    ctx.bezierCurveTo(-0.5, 0, -0.42, -0.42, 0, -0.2)
    ctx.bezierCurveTo(0.42, -0.42, 0.5, 0, 0, 0.32)
    ctx.closePath()
  }
  // Arms to the sides of the head.
  ctx.save()
  ctx.strokeStyle = '#ff4d8d'
  ctx.lineWidth = f.E * 0.05
  ctx.lineCap = 'round'
  for (const [eye, cheek] of [[f.eyeL, f.cheekL], [f.eyeR, f.cheekR]] as const) {
    const a = lerp(eye, cheek, 0.55)
    ctx.beginPath()
    ctx.moveTo(a.x, a.y)
    ctx.lineTo(cheek.x, cheek.y - 0.05 * f.E)
    ctx.stroke()
  }
  ctx.restore()
  for (const eye of [f.eyeL, f.eyeR]) {
    frame(ctx, add(eye, f.up, 0.02 * f.E), f.roll, f.E * 0.95, () => {
      heart()
      const g = ctx.createLinearGradient(0, -0.4, 0, 0.3)
      g.addColorStop(0, 'rgba(255,120,170,0.55)')
      g.addColorStop(1, 'rgba(255,60,130,0.35)')
      ctx.fillStyle = g
      ctx.fill()
      ctx.lineWidth = 0.06
      ctx.strokeStyle = '#ff3d85'
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(-0.22, -0.18)
      ctx.quadraticCurveTo(-0.28, -0.02, -0.12, 0.08)
      ctx.lineWidth = 0.035
      ctx.strokeStyle = 'rgba(255,255,255,0.75)'
      ctx.stroke()
    })
  }
  frame(ctx, lerp(f.eyeL, f.eyeR, 0.5), f.roll, f.E, () => {
    ctx.beginPath()
    ctx.moveTo(-0.15, -0.06)
    ctx.quadraticCurveTo(0, -0.15, 0.15, -0.06)
    ctx.lineWidth = 0.05
    ctx.strokeStyle = '#ff3d85'
    ctx.stroke()
  })
}

function devil(ctx: CanvasRenderingContext2D, f: Face) {
  for (const side of [-1, 1] as const) {
    frame(ctx, earAnchor(f, side, 0.5, 0.35), f.roll + side * 0.25, f.E * (side < 0 ? f.sideL : f.sideR), () => {
      ctx.beginPath()
      ctx.moveTo(-0.16, 0.1)
      ctx.bezierCurveTo(-0.14, -0.35, 0.1, -0.55, 0.32, -0.72)
      ctx.bezierCurveTo(0.2, -0.45, 0.2, -0.15, 0.18, 0.1)
      ctx.closePath()
      const g = ctx.createLinearGradient(0, -0.7, 0, 0.1)
      g.addColorStop(0, '#ff6a3d')
      g.addColorStop(1, '#a3121e')
      ctx.fillStyle = g
      ctx.shadowColor = 'rgba(255,40,40,0.6)'
      ctx.shadowBlur = 0.12 * f.E
      ctx.fill()
    }, side < 0)
  }
}

function angel(ctx: CanvasRenderingContext2D, f: Face, t: number) {
  const bob = Math.sin(t * 2.2) * 0.04
  frame(ctx, add(f.top, f.up, (0.78 + bob) * f.E), f.roll, f.E, () => {
    ctx.save()
    ctx.scale(1, 0.26)
    ctx.beginPath()
    ctx.arc(0, 0, 0.62, 0, Math.PI * 2)
    ctx.lineWidth = 0.28
    ctx.strokeStyle = 'rgba(255,214,102,0.35)'
    ctx.shadowColor = 'rgba(255,220,120,0.9)'
    ctx.shadowBlur = 0.4 * f.E
    ctx.stroke()
    ctx.lineWidth = 0.12
    ctx.strokeStyle = '#ffe28a'
    ctx.stroke()
    ctx.restore()
  })
  for (let i = 0; i < 4; i++) {
    const a = t * 0.8 + i * 1.6
    const p = add(add(f.top, f.up, (0.9 + 0.25 * Math.sin(a)) * f.E), f.right, Math.cos(a) * 0.9 * f.E)
    sparkle(ctx, p, f.E * (0.05 + 0.03 * Math.sin(a * 2)), 'rgba(255,240,200,0.95)')
  }
}

function stars(ctx: CanvasRenderingContext2D, f: Face, t: number) {
  const c = add(f.top, f.up, 0.35 * f.E)
  for (let i = 0; i < 12; i++) {
    const a = t * 0.9 + (i / 12) * Math.PI * 2
    const p = add(add(c, f.right, Math.cos(a) * 1.3 * f.E), f.up, Math.sin(a) * 0.36 * f.E)
    // Behind-the-head half of the orbit is fainter and smaller.
    const front = Math.sin(a) < 0 ? 1 : 0.55
    const tw = 0.75 + 0.25 * Math.sin(t * 5 + i * 2.1)
    sparkle(ctx, p, f.E * 0.16 * front * tw, i % 3 === 0 ? 'rgba(255,215,90,0.95)' : 'rgba(255,255,255,0.95)')
  }
}

function snapshot(canvas: HTMLCanvasElement) {
  const c = document.createElement('canvas')
  c.width = canvas.width
  c.height = canvas.height
  c.getContext('2d')!.drawImage(canvas, 0, 0)
  return c
}

/** Draw `effectId` onto `canvas` (in place), from pixel-space landmarks. */
export function drawEffect(canvas: HTMLCanvasElement, P: Pt[], effectId: string, t = 0) {
  if (effectId === 'none') return
  const ctx = canvas.getContext('2d')!
  const f = faceGeometry(P)
  ctx.save()
  switch (effectId) {
    case 'kitty': kitty(ctx, f); break
    case 'puppy': puppy(ctx, f); break
    case 'bunny': bunny(ctx, f); break
    case 'bear': bear(ctx, f); break
    case 'doll': doll(ctx, f); break
    case 'spider': spider(ctx, f, snapshot(canvas)); break
    case 'bat': bat(ctx, f, canvas); break
    case 'crown': crown(ctx, f); break
    case 'hearts': hearts(ctx, f); break
    case 'devil': devil(ctx, f); break
    case 'angel': angel(ctx, f, t); break
    case 'stars': stars(ctx, f, t); break
  }
  ctx.restore()
}
