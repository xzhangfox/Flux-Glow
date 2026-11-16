import { isEffectReady, prepareEffect } from '../effects'

// An export waits for each face's effect to be ready before it renders the
// frame — so no frame goes out without it while it's still loading. That
// happens for a camera recording's people (the live camera drew the effect
// for a face of its own, not for each person the recording is analysed
// into), for someone first seen part way through a clip, and for the 3D
// engine built again after the browser took its GPU context back. An
// effect that won't load (tried a few times, each given a while) is left
// off, rather than holding the export up for good.

const TRIES = 3
const WAIT_MS = 45_000

const within = <T,>(p: Promise<T>, ms: number) =>
  new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`not ready after ${ms / 1000} s`)), ms)
    p.then(
      (v) => (clearTimeout(timer), resolve(v)),
      (e) => (clearTimeout(timer), reject(e)),
    )
  })

export interface EffectFace {
  params: { effectId?: string }
  slot?: string
}

/** For one export: `await wait(faces)` before a frame is rendered. */
export function effectWaiter(opts: { tries?: number; waitMs?: number } = {}, deps = { isReady: isEffectReady, prepare: prepareEffect }) {
  const tries = opts.tries ?? TRIES
  const waitMs = opts.waitMs ?? WAIT_MS
  // (attempts that didn't make it ready, by face and effect)
  const failed = new Map<string, number>()
  return async function wait(faces: readonly EffectFace[]) {
    for (const f of faces) {
      const id = f.params.effectId
      if (!id || id === 'none') continue
      const slot = f.slot || ''
      const key = `${slot}:${id}`
      while (!deps.isReady(id, slot) && (failed.get(key) || 0) < tries) {
        try {
          await within(deps.prepare(id, slot), waitMs)
        } catch (err) {
          console.warn('Export: effect not ready', id, err)
        }
        if (!deps.isReady(id, slot)) failed.set(key, (failed.get(key) || 0) + 1)
      }
    }
  }
}
