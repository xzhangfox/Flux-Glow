import { FaceLandmarker } from '@mediapipe/tasks-vision'
import { connectorsToLoop } from './landmarks'
import { drawGlam, glamReady, loadGlam } from './glam'
import { drawLavenderMakeup } from './makeup'
import type { EditParams } from './pipeline'
import type { P3 } from './ar/scene'

type AR = typeof import('./ar/index')
let ar: AR | null = null
let arLoading: Promise<AR> | null = null
/** Loads the 3D engine (once). Resolves when 3D effects can render. */
export function preloadAR(): Promise<AR> {
  // The hair segmenter (for hoods and cowls) loads with it.
  arLoading ??= import('./ar/index').then(async (m) => {
    await m.preloadHair()
    return (ar = m)
  })
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

export const EFFECTS: EffectDef[] = [
  { id: 'none', label: 'None' },
  { id: 'kitty', label: 'Kitty' },
  { id: 'fox', label: 'Fox' },
  { id: 'bunny', label: 'Bunny' },
  { id: 'bear', label: 'Bear' },
  { id: 'glam', label: 'ArtPop' },
  { id: 'lavender', label: 'ArtPapi' },
  { id: 'foxhead', label: 'Fox Head' },
  { id: 'huskyhead', label: 'Husky Head' },
  { id: 'shibahead', label: 'Shiba Head' },
  { id: 'spider', label: 'Spider' },
  { id: 'bat', label: 'Bat' },
  { id: 'sport', label: 'Sport' },
  { id: 'wayfarer', label: 'Wayfarer' },
  { id: 'crown', label: 'Crown' },
  { id: 'faun', label: 'Faun' },
  { id: 'angel', label: 'Angel' },
  { id: 'stars', label: 'Stars' },
  { id: 'custom', label: 'Custom' },
]

export const findEffect = (id: string) => EFFECTS.find((e) => e.id === id) ?? EFFECTS[0]

/** Each effect's preview tile: pre-rendered (public/effects), so opening
 *  the strip never builds every model on the spot — doing that froze the
 *  camera for seconds on a phone. */
export const EFFECT_THUMBS: Record<string, string> = Object.fromEntries(EFFECTS.filter((e) => e.id !== 'none' && e.id !== 'custom').map((e) => [e.id, `/effects/${e.id}.jpg`]))

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


/** How to build `effectId`'s model. The animal heads are sculpted in a
 *  worker; the rest are quick enough to make on the page, between frames. */
function builderFor(a: AR, effectId: string, _slot: string): () => import('./ar/scene').Model | Promise<import('./ar/scene').Model> {
  const heads: Record<string, 'shiba' | 'fox' | 'husky'> = { shibahead: 'shiba', foxhead: 'fox', huskyhead: 'husky' }
  if (heads[effectId]) return () => a.animalHeadAsync(heads[effectId])
  if (effectId === 'lavender') return a.lavenderWig
  const special: Record<string, () => import('./ar/scene').Model> = { spider: a.spiderMask, bat: a.batCowl, custom: () => a.faceSticker(() => custom) }
  return special[effectId] ?? (() => a.buildModel(effectId))
}

// Effects that came in after being asked for: listeners redraw stills,
// thumbnails, paused video.
const readyListeners = new Set<() => void>()
const watching = new Set<string>()
/** Calls `fn` whenever an effect finishes getting ready. Returns the
 *  unsubscribe. */
export function onEffectReady(fn: () => void): () => void {
  readyListeners.add(fn)
  return () => readyListeners.delete(fn)
}
function watchReady(effectId: string, slot: string) {
  const key = `${slot}:${effectId}`
  if (watching.has(key) || !ar) return
  watching.add(key)
  ar.prepareModel(effectId, slot, builderFor(ar, effectId, slot))
    .catch((err) => console.warn('Effect failed to load', effectId, err))
    .finally(() => {
      watching.delete(key)
      readyListeners.forEach((fn) => fn())
    })
}

/** Gets `effectId` ready for face `slot` (loading the 3D engine first if
 *  need be), in the background. Resolves when it can be drawn. */
export async function prepareEffect(effectId: string, slot = ''): Promise<void> {
  if (effectId === 'glam') return loadGlam()
  if (!THREE_D.has(effectId)) return
  const a = await preloadAR()
  await a.prepareModel(effectId, slot, builderFor(a, effectId, slot))
}

/** Is `effectId` ready to draw for face `slot` (no wait on first use)? */
export const isEffectReady = (effectId: string, slot = '') => (effectId === 'glam' ? glamReady() : !THREE_D.has(effectId) || (!!ar && ar.isModelReady(effectId, slot)))

let glamWatched = false
function watchGlam() {
  if (glamWatched) return
  glamWatched = true
  loadGlam()
    .catch((err) => console.warn('Effect failed to load', 'glam', err))
    .finally(() => readyListeners.forEach((fn) => fn()))
}

const THREE_D = new Set(['lavender', 'foxhead', 'huskyhead', 'shibahead', 'kitty', 'fox', 'bunny', 'bear', 'spider', 'bat', 'sport', 'wayfarer', 'crown', 'faun', 'angel', 'stars', 'custom'])

// The Custom effect's picture: a square the user cropped from their own
// photo, worn in front of the face. Bumped version = a new picture.
const custom = { image: null as HTMLCanvasElement | null, version: 0 }
export function setCustomImage(image: HTMLCanvasElement) {
  custom.image = image
  custom.version++
}
export const getCustomImage = () => custom.image

/** Draw `effectId` onto `canvas` (in place). `P` are landmark pixel
 *  positions; z is MediaPipe's relative depth in pixels (negative = nearer).
 *  `slot`: whose face this is, when a frame has several (each keeps its
 *  own model). */
export function drawEffect(canvas: HTMLCanvasElement, P: P3[], effectId: string, t = 0, live = false, slot = '') {
  if (effectId === 'none') return
  const ctx = canvas.getContext('2d')!
  const f = faceGeometry(P)
  // Makeup, painted on the face under the 3D layer (the wig's bangs fall
  // over it).
  if (effectId === 'lavender') drawLavenderMakeup(ctx, P)
  // Painted eyes and lips: 2D, worn over the face.
  if (effectId === 'glam') {
    if (!glamReady()) watchGlam()
    drawGlam(ctx, P, slot, live)
    return
  }

  if (THREE_D.has(effectId)) {
    // First use: start loading the engine; the caller re-renders when it's in.
    if (!ar) {
      preloadAR()
      return
    }
    try {
      if (!ar.renderAR(canvas, P, effectId, builderFor(ar, effectId, slot), t, live, slot)) {
        // Still being got ready (in the background): the frame goes out
        // without it, and whoever is waiting hears when it's in.
        watchReady(effectId, slot)
        return
      }
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
