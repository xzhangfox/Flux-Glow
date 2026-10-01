import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { buildSkinMask } from './skinMask'
import { smoothSkin } from './smoothing'
import { applyReshape, type ReshapeParams } from './reshape'
import { applyFilter, FILTER_PRESETS, type FilterPreset } from './filters'

export interface EditParams extends ReshapeParams {
  smoothness: number
  filterId: string
}

/** The full edit pipeline, shared by the static photo editor and the live
 *  camera preview so the two never drift into visibly different results:
 *  skin mask -> frequency-separation smoothing -> per-region reshape
 *  (face/eyes/nose/mouth) -> filter. `reshapeGrid` controls the MLS warp's
 *  grid step (see mls.ts) — 1 for full per-pixel precision (static
 *  photos), higher for live video where that precision isn't worth the
 *  per-frame cost. */
export function processFrame(base: HTMLCanvasElement, landmarks: NormalizedLandmark[] | null, params: EditParams, reshapeGrid = 1): HTMLCanvasElement {
  const preset: FilterPreset = FILTER_PRESETS.find((p) => p.id === params.filterId) ?? FILTER_PRESETS[0]
  if (!landmarks) return applyFilter(base, preset)

  const mask = buildSkinMask(landmarks, base.width, base.height)
  const smoothed = smoothSkin(base, mask, params.smoothness)
  const reshaped = applyReshape(smoothed, landmarks, params, reshapeGrid)
  return applyFilter(reshaped, preset)
}
