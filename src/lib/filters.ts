export interface FilterPreset {
  id: string
  label: string
  cssFilter: string
}

export const FILTER_PRESETS: FilterPreset[] = [
  { id: 'none', label: 'Natural', cssFilter: 'none' },
  { id: 'warm', label: 'Warm', cssFilter: 'saturate(1.15) sepia(0.12) brightness(1.03)' },
  { id: 'cool', label: 'Cool', cssFilter: 'saturate(1.05) hue-rotate(-6deg) brightness(1.02) contrast(1.03)' },
  { id: 'soft', label: 'Soft Glow', cssFilter: 'brightness(1.06) contrast(0.95) saturate(1.08)' },
  { id: 'vivid', label: 'Vivid', cssFilter: 'saturate(1.4) contrast(1.1)' },
  { id: 'bw', label: 'Mono', cssFilter: 'grayscale(1) contrast(1.1)' },
]

export function applyFilter(source: HTMLCanvasElement, preset: FilterPreset): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = source.width
  canvas.height = source.height
  const ctx = canvas.getContext('2d')!
  ctx.filter = preset.cssFilter
  ctx.drawImage(source, 0, 0)
  return canvas
}
