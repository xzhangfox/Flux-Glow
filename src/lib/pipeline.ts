import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { buildSkinMask } from './skinMask'
import { smoothSkin } from './smoothing'
import { applyFillLight, applyWhitening, applyAcneRemoval, applyWrinkleRemoval, applyMouthCornerSmoothing } from './beauty'
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
export function processFrame(base: HTMLCanvasElement, landmarks: NormalizedLandmark[] | null, params: EditParams): HTMLCanvasElement {
  const preset: FilterPreset = FILTER_PRESETS.find((p) => p.id === params.filterId) ?? FILTER_PRESETS[0]
  if (!landmarks) return applyFilter(base, preset)

  const mask = buildSkinMask(landmarks, base.width, base.height)
  const ovalLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
  const bounds = loopBoundsPx(ovalLoop, landmarks, base.width, base.height, Math.max(20, base.width * 0.05))

  // Skipped outright (not just a cheap no-op inside each function) when a
  // slider is at its default 0 — each still involves at least one extra
  // canvas allocation and drawImage even when it would end up doing
  // nothing, worth avoiding since most of these will be untouched most
  // of the time, especially in the live loop.
  let working = base
  if (params.fillLight > 0.001) working = applyFillLight(working, mask, params.fillLight, bounds)
  if (params.whitening > 0.001) working = applyWhitening(working, mask, params.whitening, bounds)
  if (params.acneRemoval > 0.001) working = applyAcneRemoval(working, mask, params.acneRemoval, bounds)
  if (params.wrinkleRemoval > 0.001) working = applyWrinkleRemoval(working, mask, params.wrinkleRemoval, bounds)
  if (params.mouthCornerSmooth > 0.001) {
    const corners = [LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER].map((idx) => ({ x: landmarks[idx].x * base.width, y: landmarks[idx].y * base.height }))
    const radius = (bounds.maxX - bounds.minX) * 0.12
    working = applyMouthCornerSmoothing(working, corners, radius, params.mouthCornerSmooth)
  }
  const smoothed = smoothSkin(working, mask, params.smoothness, bounds)
  const reshaped = applyReshape(smoothed, landmarks, params)
  return applyFilter(reshaped, preset)
}
