import { FaceLandmarker } from '@mediapipe/tasks-vision'
import { connectorsToLoop } from './landmarks'
import { LIPS_LOOP } from './deform'
import type { EditParams } from './pipeline'
import type { P3 } from './ar/scene'

type AR = typeof import('./ar/index')
let ar: AR | null = null
let arLoading: Promise<AR> | null = null
/** Loads the 3D engine (once). Resolves when 3D effects can render. */
export function preloadAR(): Promise<AR> {
  arLoading ??= import('./ar/index').then((m) => (ar = m))
  return arLoading
}

// Face-tracked AR effects, applied each time a frame is processed (every
// live frame, or once for a still) from the face mesh's landmarks AFTER
// Shape has moved them, so ears sit on the reshaped head and a mask hugs
// the slimmed jaw.
//
// Anything with volume — fur ears, noses, masks, glasses, crown, horns,
// halo — is a real 3D model rendered by ar/scene.ts, lit by light
// estimated from the photo and occluded by the head. What's genuinely
// paint on skin (blush, freckles, lip tint, eye black under the cowl,
// sparkles) stays a 2D layer here, drawn under or over the 3D pass.

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
  { id: 'kitty', label: 'Kitty' },
  { id: 'fox', label: 'Fox' },
  { id: 'puppy', label: 'Puppy' },
  { id: 'bunny', label: 'Bunny' },
  { id: 'bear', label: 'Bear' },
  { id: 'doll', label: 'Doll', boost: { ...BABY, eyes: 0.45, eyeHeight: 0.3, chin: -0.4, whitening: 0.25 } },
  { id: 'foxhead', label: 'Fox Head' },
  { id: 'huskyhead', label: 'Husky Head' },
  { id: 'spider', label: 'Spider' },
  { id: 'bat', label: 'Bat' },
  { id: 'square', label: 'Square' },
  { id: 'oval', label: 'Oval' },
  { id: 'shield', label: 'Shield' },
  { id: 'cateye', label: 'Cat-eye' },
  { id: 'tinted', label: 'Tinted' },
  { id: 'aviator', label: 'Aviator' },
  { id: 'hearts', label: 'Hearts' },
  { id: 'crown', label: 'Crown' },
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


const THREE_D = new Set(['foxhead', 'huskyhead', 'kitty', 'fox', 'puppy', 'bunny', 'bear', 'spider', 'bat', 'aviator', 'square', 'oval', 'shield', 'cateye', 'tinted', 'hearts', 'crown', 'devil', 'angel', 'stars'])

/** Draw `effectId` onto `canvas` (in place). `P` are landmark pixel
 *  positions; z is MediaPipe's relative depth in pixels (negative = nearer). */
export function drawEffect(canvas: HTMLCanvasElement, P: P3[], effectId: string, t = 0, live = false) {
  if (effectId === 'none') return
  const ctx = canvas.getContext('2d')!
  const f = faceGeometry(P)
  ctx.save()
  // Paint under the 3D layer.
  if (effectId === 'kitty' || effectId === 'bear' || effectId === 'fox') blush(ctx, f)
  if (effectId === 'bunny') blush(ctx, f, 'rgba(255,130,160,0.3)')
  if (effectId === 'doll') doll(ctx, f)
  ctx.restore()

  if (THREE_D.has(effectId)) {
    // First use: start loading the engine; the caller re-renders when it's in.
    if (!ar) {
      preloadAR()
      return
    }
    const { renderAR, spiderMask, batCowl, foxHead, huskyHead, buildModel } = ar
    const special: Record<string, () => import('./ar/scene').Model> = { spider: spiderMask, bat: batCowl, foxhead: foxHead, huskyhead: huskyHead }
    const build = special[effectId] ?? (() => buildModel(effectId))
    try {
      renderAR(canvas, P, effectId, build, t, live)
    } catch (err) {
      // No WebGL (or it was lost): the photo is still fine without the prop.
      console.warn('AR effect unavailable', err)
    }
  }

  // Sparkle over the halo.
  if (effectId === 'angel') {
    for (let i = 0; i < 4; i++) {
      const a = t * 0.8 + i * 1.6
      const p = add(add(f.top, f.up, (1.3 + 0.25 * Math.sin(a)) * f.E), f.right, Math.cos(a) * 0.9 * f.E)
      sparkle(ctx, p, f.E * (0.05 + 0.03 * Math.sin(a * 2)), 'rgba(255,240,200,0.95)')
    }
  }
}
