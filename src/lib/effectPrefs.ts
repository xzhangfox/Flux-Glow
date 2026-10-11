// How effects behave, remembered across visits (Settings).

const BLINK_KEY = 'glow_effects_blink'

let blink: boolean | null = null

/** Do the animal heads blink with the wearer (on by default)? Off, their
 *  eyes stay open whatever the wearer's do. */
export function effectsBlink(): boolean {
  if (blink === null) {
    try {
      blink = localStorage.getItem(BLINK_KEY) !== '0'
    } catch {
      blink = true
    }
  }
  return blink
}

export function setEffectsBlink(on: boolean) {
  blink = on
  try {
    localStorage.setItem(BLINK_KEY, on ? '1' : '0')
  } catch {
    // (private mode: just this session)
  }
}
