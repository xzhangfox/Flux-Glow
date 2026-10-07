import type { ReactNode } from 'react'
import type { AdjustItem } from './AdjustPanel'
import { DEFAULT_PARAMS, type EditParams, type NumericParam } from '../lib/pipeline'
import { SHAPE_PARAMS } from '../lib/deform'
import { IconDroplet, IconWhiten, IconTarget, IconWave, IconSun, RegionIcon } from './icons'

// The Beauty and Shape controls, shared by the photo/camera editor and the
// video editor so both offer exactly the same retouch.

export const BEAUTY_KEYS: NumericParam[] = ['smoothness', 'whitening', 'acneRemoval', 'wrinkleRemoval', 'mouthCornerSmooth', 'fillLight']
export const SHAPE_KEYS: NumericParam[] = [...SHAPE_PARAMS]

export const changedFrom = (params: EditParams, keys: NumericParam[]) => keys.some((k) => Math.abs(params[k] - DEFAULT_PARAMS[k]) > 0.005)

type Set = (key: NumericParam, v: number) => void

function itemFor(params: EditParams, set: Set) {
  return (key: NumericParam, label: string, icon: ReactNode, bidirectional = false): AdjustItem => ({
    key,
    label,
    icon,
    value: params[key],
    defaultValue: DEFAULT_PARAMS[key],
    onChange: (v) => set(key, v),
    bidirectional,
  })
}

export function beautyItems(params: EditParams, set: Set): AdjustItem[] {
  const item = itemFor(params, set)
  return [
    item('smoothness', 'Smooth', <IconDroplet className="w-5 h-5" />),
    item('whitening', 'Whiten', <IconWhiten className="w-5 h-5" />),
    item('acneRemoval', 'Blemish', <IconTarget className="w-5 h-5" />),
    item('wrinkleRemoval', 'Wrinkle', <IconWave className="w-5 h-5" />),
    item('mouthCornerSmooth', 'Folds', <RegionIcon dot={[9, 15.6]} pair className="w-5 h-5" />),
    item('fillLight', 'Light', <IconSun className="w-5 h-5" />),
  ]
}

export function shapeItems(params: EditParams, set: Set): AdjustItem[] {
  const item = itemFor(params, set)
  const R = (dot: [number, number], pair = false) => <RegionIcon dot={dot} pair={pair} className="w-5 h-5" />
  const group = (key: string, label: string, children: AdjustItem[]): AdjustItem => ({ key, label, icon: null, children })
  return [
    group('faceGroup', 'Face', [
      item('face', 'Slim', R([7.2, 13.8], true), true),
      item('vJaw', 'V Jaw', R([8.2, 15.6], true), true),
      item('chin', 'Chin', R([12, 17.2]), true),
      item('forehead', 'Forehead', R([12, 5.2]), true),
      item('temple', 'Temple', R([7.2, 7.6], true), true),
      item('cheekbone', 'Cheekbone', R([6.8, 11.5], true), true),
    ]),
    group('eyeGroup', 'Eyes', [
      item('eyes', 'Size', R([9, 10.2], true), true),
      item('eyeWidth', 'Width', R([8.4, 10.2], true), true),
      item('eyeHeight', 'Height', R([9, 9.8], true), true),
      item('eyeTilt', 'Tilt', R([7.8, 9.7], true), true),
      item('eyeDistance', 'Spacing', R([7.6, 10.2], true), true),
      item('eyePosition', 'Position', R([9, 9.4], true), true),
    ]),
    group('browGroup', 'Brows', [
      item('eyebrowHeight', 'Height', R([9, 8], true), true),
      item('browTilt', 'Tilt', R([7.4, 7.6], true), true),
      item('browDistance', 'Spacing', R([10.4, 8], true), true),
    ]),
    group('noseGroup', 'Nose', [
      item('nose', 'Size', R([12, 12.5]), true),
      item('noseWings', 'Wings', R([10.8, 12.8], true), true),
      item('noseTip', 'Tip', R([12, 13]), true),
      item('noseLength', 'Length', R([12, 11.6]), true),
      item('noseBridge', 'Bridge', R([12, 9.3]), true),
    ]),
    group('mouthGroup', 'Mouth', [
      item('mouth', 'Size', R([12, 14.3]), true),
      item('mouthWidth', 'Width', R([10, 14.3], true), true),
      item('mouthUpperLip', 'Upper Lip', R([12, 13.6]), true),
      item('mouthLowerLip', 'Lower Lip', R([12, 15.1]), true),
      item('mouthCorners', 'Smile', R([9.6, 14], true), true),
      item('mouthPosition', 'Position', R([12, 13.2]), true),
    ]),
  ]
}
