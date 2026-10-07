import { eyeClosureOf } from './faceLandmarker'

// "Glam": painted eyes and lips — cut out of a pop-art drawing — worn over
// the user's own, tracked every frame. Each eye is pinned by its two
// corners and squashes shut when the user blinks; the lips are split along
// their seam, so the top lip rides the user's upper lip and the bottom
// lip the lower one, and the mouth shows between them when it opens.

interface Pt {
  x: number
  y: number
}

interface Sprite {
  src: string
  img?: HTMLImageElement
  /** The two anchor points (left, right in the picture), in sprite px. */
  a: Pt
  b: Pt
}

// Anchors read off the sprites (public/effects/glam/*.png): the corners of
// each eye's opening, the corners of the mouth. The sprites carry their
// own soft contact shadow, baked in round a PAD-pixel margin (a canvas
// shadow is slow, and some browsers draw an image's shadow as its whole
// rectangle — a faint box round each eye).
const PAD = 24
const at = (x: number, y: number) => ({ x: x + PAD, y: y + PAD })
const SPRITES = {
  eyeL: { src: '/effects/glam/eyeL.png', a: at(95, 170), b: at(328, 172) } as Sprite,
  eyeR: { src: '/effects/glam/eyeR.png', a: at(60, 190), b: at(300, 178) } as Sprite,
  lipsTop: { src: '/effects/glam/lipsTop.png', a: at(25, 100), b: at(420, 100) } as Sprite,
  lipsBot: { src: '/effects/glam/lipsBot.png', a: at(25, 100), b: at(420, 100) } as Sprite,
}
/** The middle of the lips' seam (where they're cut apart). */
const LIP_SEAM = at(222, 117)

let loading: Promise<void> | null = null
/** Loads the sprites (once). */
export function loadGlam(): Promise<void> {
  loading ??= Promise.all(
    Object.values(SPRITES).map(
      (s) =>
        new Promise<void>((resolve, reject) => {
          const img = new Image()
          img.onload = () => {
            s.img = img
            resolve()
          }
          img.onerror = () => reject(new Error(`Couldn't load ${s.src}`))
          img.src = s.src
        }),
    ),
  ).then(() => undefined)
  return loading
}
export const glamReady = () => Object.values(SPRITES).every((s) => s.img)

const sub = (a: Pt, b: Pt) => ({ x: a.x - b.x, y: a.y - b.y })
const len = (v: Pt) => Math.hypot(v.x, v.y)

/** Draws `s` with its anchors' midpoint at `at`, its anchor line along
 *  `angle`, `kx` across and `ky` up and down; `pivot` (sprite px) instead of
 *  the anchors' midpoint, if given. */
function place(ctx: CanvasRenderingContext2D, s: Sprite, at: Pt, angle: number, kx: number, ky: number, pivot?: Pt) {
  const img = s.img!
  const d = sub(s.b, s.a)
  const sa = Math.atan2(d.y, d.x)
  const p = pivot ?? { x: (s.a.x + s.b.x) / 2, y: (s.a.y + s.b.y) / 2 }
  ctx.save()
  ctx.translate(at.x, at.y)
  ctx.rotate(angle)
  ctx.scale(kx, ky)
  ctx.rotate(-sa)
  ctx.drawImage(img, -p.x, -p.y)
  ctx.restore()
}

// Blinks, smoothed per face: a resting baseline (so looking down isn't
// read as half shut) and both eyes as one unless clearly apart (a wink).
const blinkState = new Map<string, { base: { r: number; l: number }; shut: { r: number; l: number }; wink: number }>()
const ss = (x: number, a: number, b: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

function blinks(P: Pt[], slot: string, live: boolean) {
  const ratio = (up: number, lo: number, a: number, b: number) => len(sub(P[up], P[lo])) / Math.max(1e-6, len(sub(P[a], P[b])))
  const score = (q: number) => Math.min(1, Math.max(0, 1 - q / 0.32))
  const raw = eyeClosureOf(P) ?? { r: score(ratio(159, 145, 33, 133)), l: score(ratio(386, 374, 362, 263)) }
  let st = blinkState.get(slot)
  if (!st) blinkState.set(slot, (st = { base: { r: 0.15, l: 0.15 }, shut: { r: 0, l: 0 }, wink: 0 }))
  const out = { r: 0, l: 0 }
  for (const k of ['r', 'l'] as const) {
    const v = raw[k]
    st.base[k] = v < st.base[k] ? v : live ? st.base[k] + (v - st.base[k]) * 0.004 : st.base[k]
    out[k] = ss(v - Math.min(st.base[k], 0.45), 0.12, 0.42)
  }
  const apart = Math.abs(out.r - out.l) > 0.45
  st.wink = live ? (apart ? Math.min(1, st.wink + 0.34) : Math.max(0, st.wink - 0.5)) : apart ? 1 : 0
  const both = Math.max(out.r, out.l) * 0.7 + Math.min(out.r, out.l) * 0.3
  for (const k of ['r', 'l'] as const) {
    const target = both + (out[k] - both) * st.wink
    st.shut[k] = live ? st.shut[k] + (target - st.shut[k]) * (target > st.shut[k] ? 0.85 : 0.6) : target
  }
  return st.shut
}

/** Draws the Glam eyes and lips at landmarks `P` (pixels). */
export function drawGlam(ctx: CanvasRenderingContext2D, P: Pt[], slot: string, live: boolean) {
  if (!glamReady()) {
    void loadGlam()
    return
  }
  // The face's own scale (eye centre to eye centre), steadier than any one
  // feature's width when the head turns.
  const cR = { x: (P[33].x + P[133].x) / 2, y: (P[33].y + P[133].y) / 2 }
  const cL = { x: (P[362].x + P[263].x) / 2, y: (P[362].y + P[263].y) / 2 }
  const E = Math.max(1, len(sub(cR, cL)))
  const shut = blinks(P, slot, live)

  ctx.save()
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'

  // Eyes: the subject's right eye (33/133) shows on the picture's left
  // unless the frame is mirrored, so go by where each actually is.
  const eyes = [
    { c: [P[33], P[133]], up: P[159], lo: P[145], shut: shut.r },
    { c: [P[362], P[263]], up: P[386], lo: P[374], shut: shut.l },
  ].sort((a, b) => a.c[0].x + a.c[1].x - b.c[0].x - b.c[1].x)
  eyes.forEach((e, i) => {
    const s = i === 0 ? SPRITES.eyeL : SPRITES.eyeR
    const [A, B] = e.c[0].x <= e.c[1].x ? [e.c[0], e.c[1]] : [e.c[1], e.c[0]]
    const v = sub(B, A)
    const spriteW = len(sub(s.b, s.a))
    // A touch larger than the real eye, so none of it shows round the edge.
    const kx = (len(v) * 1.18) / spriteW
    const ky = ((E * 0.47 * 1.18) / spriteW) * Math.max(0.1, 1 - e.shut * 0.92)
    place(ctx, s, { x: (A.x + B.x) / 2, y: (A.y + B.y) / 2 }, Math.atan2(v.y, v.x), kx, ky)
  })

  // Lips: pinned by the mouth corners; each half at its own inner lip.
  const [mA, mB] = P[61].x <= P[291].x ? [P[61], P[291]] : [P[291], P[61]]
  const mv = sub(mB, mA)
  const angle = Math.atan2(mv.y, mv.x)
  const spriteW = len(sub(SPRITES.lipsTop.b, SPRITES.lipsTop.a))
  const kx = (len(mv) * 1.12) / spriteW
  const ky = ((len(mv) + E * 0.9) * 0.5 * 1.12) / spriteW
  place(ctx, SPRITES.lipsBot, P[14], angle, kx, ky, LIP_SEAM)
  place(ctx, SPRITES.lipsTop, P[13], angle, kx, ky, LIP_SEAM)
  ctx.restore()
}
