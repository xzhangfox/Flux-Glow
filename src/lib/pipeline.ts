import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { buildSkinMask } from './skinMask'
import { smoothSkin } from './smoothing'
import { applyReshape, type ReshapeParams } from './reshape'
import { applyFilter, FILTER_PRESETS, type FilterPreset } from './filters'
import { connectorsToLoop, loopBoundsPx } from './landmarks'

export interface EditParams extends ReshapeParams {
  smoothness: number
  filterId: string
}

/** The full edit pipeline, shared by the static photo editor and the live
 *  camera preview so the two never drift into visibly different results:
 *  skin mask -> frequency-separation smoothing -> per-region reshape
 *  (face/eyes/nose/mouth) -> filter. Reshape is a GPU triangulated-mesh
 *  warp (see meshWarp.ts) — unlike the CPU per-pixel approach it replaced,
 *  it doesn't need a precision/cost tradeoff between static photos and
 *  live video, so there's no grid-step parameter to thread through here
 *  the way the old MLS version needed. */
export function processFrame(base: HTMLCanvasElement, landmarks: NormalizedLandmark[] | null, params: EditParams): HTMLCanvasElement {
  const preset: FilterPreset = FILTER_PRESETS.find((p) => p.id === params.filterId) ?? FILTER_PRESETS[0]
  if (!landmarks) return applyFilter(base, preset)

  const mask = buildSkinMask(landmarks, base.width, base.height)
  const ovalLoop = connectorsToLoop(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
  const smoothBounds = loopBoundsPx(ovalLoop, landmarks, base.width, base.height, Math.max(20, base.width * 0.05))
  const smoothed = smoothSkin(base, mask, params.smoothness, smoothBounds)
  const reshaped = applyReshape(smoothed, landmarks, params)
  return applyFilter(reshaped, preset)
}
