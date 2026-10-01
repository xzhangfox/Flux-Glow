import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { buildSkinMask } from './skinMask'
import { smoothSkin } from './smoothing'
import { applyContour } from './contour'
import { applyFilter, FILTER_PRESETS, type FilterPreset } from './filters'

export interface EditParams {
  smoothness: number
  contour: number
  filterId: string
}

/** The full edit pipeline, shared by the static photo editor and the live
 *  camera preview so the two never drift into visibly different results:
 *  skin mask -> frequency-separation smoothing -> cheek/jaw pinch -> filter. */
export function processFrame(base: HTMLCanvasElement, landmarks: NormalizedLandmark[] | null, params: EditParams): HTMLCanvasElement {
  const preset: FilterPreset = FILTER_PRESETS.find((p) => p.id === params.filterId) ?? FILTER_PRESETS[0]
  if (!landmarks) return applyFilter(base, preset)

  const mask = buildSkinMask(landmarks, base.width, base.height)
  const smoothed = smoothSkin(base, mask, params.smoothness)
  const contoured = applyContour(smoothed, landmarks, params.contour)
  return applyFilter(contoured, preset)
}
