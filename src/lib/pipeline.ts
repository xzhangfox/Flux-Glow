import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { buildSkinMask } from './skinMask'
import { smoothSkin } from './smoothing'
import { applyFillLight, applyWhitening, applyAcneRemoval, applyWrinkleRemoval, applyMouthCornerSmoothing } from './beauty'
import { applyLiveTonePass } from './livePass'
import { applyReshape, type ReshapeParams } from './reshape'
import { LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER } from './meshWarp'
import { applyFilter, FILTER_PRESETS, type FilterPreset } from './filters'
import { connectorsToLoop, loopBoundsPx } from './landmarks'

export interface EditParams extends ReshapeParams {
  smoothness: number
  fillLight: number
  whitening: number
  acneRemoval: number
  wrinkleRemoval: number
  mouthCornerSmooth: number
  filterId: string
}

/** The full edit pipeline, shared by the static photo editor and the live
 *  camera preview so the two never drift into visibly different results:
 *  skin mask -> beauty (fill light, whitening, acne, wrinkles) ->
 *  frequency-separation smoothing -> per-region reshape (face/eyes/
 *  nose/mouth/...) -> filter. Reshape is a GPU triangulated-mesh warp
 *  (see meshWarp.ts) — unlike the CPU per-pixel approach it replaced, it
 *  doesn't need a precision/cost tradeoff between static photos and live
 *  video, so there's no grid-step parameter to thread through here the
 *  way the old MLS version needed. */
/**
 * `highQuality` switches the tone-based effects (fill light, acne,
 * wrinkle removal, skin smoothing) between a guided filter — edge-aware,
 * but a CPU per-pixel operation costing 100ms+ per call at face-region
 * size — and a plain Gaussian blur. The static photo editor always wants
 * the accurate pass; the live preview, which must redo all of this every
 * frame at 30fps, uses the cheaper one so the viewfinder stays responsive
 * and gets upgraded to full quality the moment a photo is actually
 * captured or confirmed.
 */
export function processFrame(base: HTMLCanvasElement, landmarks: NormalizedLandmark[] | null, params: EditParams, highQuality = true): HTMLCanvasElement {
  const preset: FilterPreset = FILTER_PRESETS.find((p) => p.id === params.filterId) ?? FILTER_PRESETS[0]
  if (!landmarks) return applyFilter(base, preset)

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
    if (params.fillLight > 0.001) applyFillLight(working, mask, params.fillLight, bounds, true)
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
  return applyFilter(reshaped, preset)
}
