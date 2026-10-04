import { DEFAULT_PARAMS, type EditParams, type NumericParam } from './pipeline'

// One-tap style templates: each is a full recipe across Beauty, Shape and
// Filter, tuned on real faces so it reads as a recognizable look at 100%
// rather than a random stack of sliders. Applying one replaces the current
// Beauty/Shape/Filter settings (Effects and stickers are left alone), and
// the Looks panel's Intensity slider blends between the untouched default
// and the full recipe — every value stays inside the calibrated slider
// ranges, so no look (or intensity) can push a face into distortion.
export interface Look {
  id: string
  label: string
  /** Short hint shown under the selected look. */
  hint: string
  params: Partial<Omit<EditParams, 'effectId'>>
}

export const LOOKS: Look[] = [
  {
    id: 'natural',
    label: 'Natural',
    hint: 'Light polish — still unmistakably you',
    params: { smoothness: 0.45, acneRemoval: 0.4, face: 0.2, eyes: 0.12, fillLight: 0.15 },
  },
  {
    // 幼态: the "baby schema" proportions — larger, rounder, slightly lower
    // eyes, a taller forehead, a short small nose and chin, soft rounded
    // cheeks rather than a sharp V, plus dewy, rosy skin.
    id: 'baby',
    label: 'Baby Face',
    hint: 'Big round eyes, short chin, soft cheeks',
    params: {
      smoothness: 0.7, whitening: 0.35, acneRemoval: 0.5, fillLight: 0.35,
      face: 0.25, vJaw: 0.35, chin: -0.55, forehead: 0.45, cheekbone: 0.2, temple: -0.2,
      eyes: 0.55, eyeHeight: 0.4, eyePosition: -0.3, eyeTilt: -0.15,
      nose: 0.3, noseTip: 0.35, noseLength: -0.45,
      mouth: -0.2, mouthUpperLip: 0.2, mouthCorners: 0.25, mouthPosition: 0.2,
      eyebrowHeight: 0.15, browTilt: -0.2,
      filterId: 'peach', filterStrength: 0.6,
    },
  },
  {
    // 硬朗: a strong, defined jaw and chin, straighter lower brows, a
    // higher bridge, a touch narrower eyes, and texture left on the skin.
    id: 'chiseled',
    label: 'Chiseled',
    hint: 'Defined jaw, strong brows, high bridge',
    params: {
      smoothness: 0.25, acneRemoval: 0.4, fillLight: 0.3,
      face: -0.15, vJaw: -0.45, chin: 0.4, cheekbone: -0.25, temple: 0.1,
      eyes: -0.1, eyeHeight: -0.25, eyeWidth: 0.15,
      eyebrowHeight: -0.35, browTilt: -0.3, browDistance: -0.2,
      noseBridge: 0.6, noseWings: 0.25, noseLength: 0.15,
      mouthWidth: 0.1, mouthUpperLip: -0.15,
      filterId: 'film', filterStrength: 0.5,
    },
  },
  {
    // 精致: a slim V line and small refined features.
    id: 'delicate',
    label: 'Delicate',
    hint: 'Slim V line, refined small features',
    params: {
      smoothness: 0.6, whitening: 0.25, acneRemoval: 0.5, fillLight: 0.25,
      face: 0.45, vJaw: 0.5, cheekbone: 0.35, temple: -0.25, chin: 0.15,
      eyes: 0.35, eyeHeight: 0.15,
      nose: 0.45, noseWings: 0.45, noseBridge: 0.4, noseTip: 0.3,
      mouth: -0.15, mouthCorners: 0.3,
      filterId: 'cream', filterStrength: 0.5,
    },
  },
  {
    // 甜美: lifted smile, rounder eyes, fuller upper lip, rosy filter.
    id: 'sweet',
    label: 'Sweet',
    hint: 'Smiling corners, round eyes, rosy glow',
    params: {
      smoothness: 0.65, whitening: 0.3, acneRemoval: 0.5, fillLight: 0.3,
      face: 0.3, vJaw: 0.2, chin: -0.2,
      eyes: 0.45, eyeHeight: 0.35, eyeTilt: -0.1,
      noseTip: 0.25,
      mouthCorners: 0.6, mouthUpperLip: 0.3, mouthLowerLip: 0.15,
      eyebrowHeight: 0.15, browTilt: -0.15,
      filterId: 'peach', filterStrength: 0.75,
    },
  },
  {
    // 清冷: a longer, cooler face — lifted outer eye corners, arched brows,
    // a slender nose and a calm, smaller mouth.
    id: 'elegant',
    label: 'Elegant',
    hint: 'Lifted eyes, arched brows, cool tone',
    params: {
      smoothness: 0.55, whitening: 0.3, acneRemoval: 0.4, fillLight: 0.2,
      face: 0.35, vJaw: 0.3, chin: 0.3, cheekbone: 0.2,
      eyes: 0.15, eyeWidth: 0.35, eyeTilt: 0.5, eyeHeight: -0.1,
      browTilt: 0.45, eyebrowHeight: 0.1,
      nose: 0.3, noseBridge: 0.5, noseWings: 0.3,
      mouth: -0.15, mouthUpperLip: -0.1,
      filterId: 'fresh', filterStrength: 0.6,
    },
  },
  {
    // 少年感: clean and youthful without softening the structure away.
    id: 'boyish',
    label: 'Boyish',
    hint: 'Clean skin, youthful, structure kept',
    params: {
      smoothness: 0.4, acneRemoval: 0.6, wrinkleRemoval: 0.3, fillLight: 0.25,
      face: 0.2, vJaw: 0.15, chin: -0.1,
      eyes: 0.2, eyeHeight: 0.15,
      eyebrowHeight: -0.15, browTilt: -0.1,
      noseBridge: 0.3,
      filterId: 'fresh', filterStrength: 0.4,
    },
  },
  {
    // Glam: the "full beauty camera" look, still inside safe ranges.
    id: 'glam',
    label: 'Glam',
    hint: 'Full glow — bright skin, big eyes, sharp V',
    params: {
      smoothness: 0.8, whitening: 0.5, acneRemoval: 0.7, wrinkleRemoval: 0.5, mouthCornerSmooth: 0.4, fillLight: 0.5,
      face: 0.6, vJaw: 0.55, cheekbone: 0.4, chin: 0.2,
      eyes: 0.6, eyeHeight: 0.3, eyeTilt: 0.2,
      nose: 0.5, noseWings: 0.4, noseBridge: 0.6,
      mouthUpperLip: 0.3, mouthCorners: 0.35,
      filterId: 'soft', filterStrength: 0.6,
    },
  },
]

export const findLook = (id: string | null) => LOOKS.find((l) => l.id === id) ?? null

const NUMERIC_KEYS = Object.keys(DEFAULT_PARAMS).filter((k) => typeof DEFAULT_PARAMS[k as keyof EditParams] === 'number') as NumericParam[]

/** The look's recipe blended toward the defaults by `strength` (0..1),
 *  keeping the current effect. */
export function applyLook(look: Look, strength: number, current: EditParams): EditParams {
  const next: EditParams = { ...DEFAULT_PARAMS, effectId: current.effectId }
  for (const k of NUMERIC_KEYS) {
    const target = look.params[k]
    if (target === undefined) continue
    next[k] = DEFAULT_PARAMS[k] + (target - DEFAULT_PARAMS[k]) * strength
  }
  if (look.params.filterId) {
    next.filterId = look.params.filterId
    next.filterStrength = (look.params.filterStrength ?? DEFAULT_PARAMS.filterStrength) * Math.max(0.35, strength)
  }
  return next
}
