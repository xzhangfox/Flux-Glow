import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { buildSkinMask } from './skinMask'
import { smoothSkin } from './smoothing'
import { applyWhitening, applyAcneRemoval, applyWrinkleRemoval, applyMouthCornerSmoothing } from './beauty'
import { applyLiveTonePass } from './livePass'
import { applyReshape } from './reshape'
import { SHAPE_PARAMS, deformTargets, type ReshapeParams } from './deform'
import { drawEffect, withEffectBoost } from './effects'
import { LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER } from './meshWarp'
import { applyFilter, findPreset } from './filters'
import { connectorsToLoop, loopBoundsPx } from './landmarks'

export interface EditParams extends ReshapeParams {
  smoothness: number
  fillLight: number
  whitening: number
  acneRemoval: number
  wrinkleRemoval: number
  mouthCornerSmooth: number
  filterId: string
  filterStrength: number
  /** Face-tracked AR effect (see effects.ts), 'none' for off. */
  effectId: string
}

export type NumericParam = Exclude<keyof EditParams, 'filterId' | 'effectId'>

/** The untouched starting point — also what each panel's Reset restores
 *  and what "has this control been changed" is measured against. Light
 *  smoothing and a slight jaw slim are on by default, the same "natural"
 *  baseline commercial beauty cameras open with. */
export const DEFAULT_PARAMS: EditParams = {
  smoothness: 0.6,
  fillLight: 0,
  whitening: 0,
  acneRemoval: 0,
  wrinkleRemoval: 0,
  mouthCornerSmooth: 0,
  ...(Object.fromEntries(SHAPE_PARAMS.map((k) => [k, 0])) as Record<(typeof SHAPE_PARAMS)[number], number>),
  face: 0.25,
  filterId: 'none',
  filterStrength: 0.8,
  effectId: 'none',
}

/** Beauty sliders whose raw effect is too strong at the top of the track:
 *  full fill light blew the face out and full whitening went grey, so the
 *  slider's 100 maps to the strongest setting that still looks like skin. */
function calibrate(p: EditParams): EditParams {
  return { ...p, fillLight: p.fillLight * 0.4, whitening: p.whitening * 0.75 }
}

/** Square crop around the face oval (padded) in pixel space, for the
 *  filter strip's thumbnails. */
export function faceFocus(landmarks: NormalizedLandmark[] | null, w: number, h: number): { x: number; y: number; size: number } | null {
  if (!landmarks) return null
  const b = loopBoundsPx(connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL), landmarks, w, h, 0)
  return { x: (b.minX + b.maxX) / 2, y: (b.minY + b.maxY) / 2, size: Math.max(b.maxX - b.minX, b.maxY - b.minY) * 1.35 }
}

/** The full edit pipeline, shared by the static photo editor and the live
 *  camera preview so the two never drift into visibly different results:
 *  skin mask -> beauty (whitening, acne, wrinkles) -> frequency-separation
 *  smoothing -> per-region reshape + 3D relighting (face/eyes/nose/
 *  mouth/fill light/...) -> filter. Reshape and fill light both run as
 *  part of the same GPU triangulated-mesh pass (see meshWarp.ts) — fill
 *  light specifically needs the mesh's 3D vertex geometry for its per-
 *  vertex surface normals, which only exists in that pass, so it isn't a
 *  CPU beauty.ts effect the way whitening/acne/wrinkles are. Unlike the
 *  CPU per-pixel reshape approach this replaced, the GPU pass doesn't
 *  need a precision/cost tradeoff between static photos and live video,
 *  so there's no grid-step parameter to thread through here the way the
 *  old MLS version needed, and no highQuality branch for it below either.
 */
/**
 * `highQuality` switches the CPU tone-based effects (whitening, acne,
 * wrinkle removal, skin smoothing) between a guided filter — edge-aware,
 * but a CPU per-pixel operation costing 100ms+ per call at face-region
 * size — and a plain Gaussian blur. The static photo editor always wants
 * the accurate pass; the live preview, which must redo all of this every
 * frame at 30fps, uses the cheaper one so the viewfinder stays responsive
 * and gets upgraded to full quality the moment a photo is actually
 * captured or confirmed. Fill light's GPU relighting is unaffected by
 * this flag — it's equally cheap (a shader, not a CPU blur) either way.
 */
export function processFrame(base: HTMLCanvasElement, landmarks: NormalizedLandmark[] | null, rawParams: EditParams, highQuality = true): HTMLCanvasElement {
  const params = calibrate(withEffectBoost(rawParams))
  const preset = findPreset(params.filterId)
  if (!landmarks) return applyFilter(base, preset, params.filterStrength)

  const mask = buildSkinMask(landmarks, base.width, base.height)
  const ovalLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
  const bounds = loopBoundsPx(ovalLoop, landmarks, base.width, base.height, Math.max(20, base.width * 0.05))

  // One shared canvas that every tone/beauty effect mutates in place over
  // just its own bounds rect, instead of each allocating a full-frame
  // copy and handing it to the next — chaining five full-frame
  // allocate+drawImage round trips was the dominant cost in the live
  // preview loop. A slider at its default 0 still skips its effect's own
  // work entirely, same as before.
  const working = document.createElement('canvas')
  working.width = base.width
  working.height = base.height
  working.getContext('2d')!.drawImage(base, 0, 0)

  if (highQuality) {
    if (params.whitening > 0.001) applyWhitening(working, mask, params.whitening, bounds)
    if (params.acneRemoval > 0.001) applyAcneRemoval(working, mask, params.acneRemoval, bounds, true)
    if (params.wrinkleRemoval > 0.001) applyWrinkleRemoval(working, mask, params.wrinkleRemoval, bounds, true)
    smoothSkin(working, mask, params.smoothness, bounds, true)
  } else {
    // Even with each effect already cropping its own blurs to the face's
    // bounding box instead of the full frame, running the five separate
    // functions still cost 150-200ms/frame at live resolution — the
    // remaining dominant cost was each one doing its own
    // getImageData/putImageData round trip on the same ~500x600 region.
    // This fuses all five into one shared read, one shared per-pixel
    // loop, and one shared write (see livePass.ts for the accepted
    // fidelity tradeoffs that come with fusing them).
    applyLiveTonePass(working, mask, params, bounds)
  }
  if (params.mouthCornerSmooth > 0.001) {
    const corners = [LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER].map((idx) => ({ x: landmarks[idx].x * base.width, y: landmarks[idx].y * base.height }))
    const radius = (bounds.maxX - bounds.minX) * 0.12
    applyMouthCornerSmoothing(working, corners, radius, params.mouthCornerSmooth)
  }
  const reshaped = applyReshape(working, landmarks, params)
  if (params.effectId !== 'none') {
    // Effects track the face as reshaped, so ears sit on the slimmed head.
    const t = deformTargets(landmarks, params, base.width / base.height)
    const pts = Array.from({ length: t.length / 2 }, (_, i) => ({ x: t[i * 2] * base.width, y: t[i * 2 + 1] * base.height, z: landmarks[i].z * base.width }))
    drawEffect(reshaped, pts, params.effectId, highQuality ? 0.6 : performance.now() / 1000, !highQuality)
  }
  return applyFilter(reshaped, preset, params.filterStrength)
}
