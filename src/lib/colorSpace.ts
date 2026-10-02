// Standard sRGB <-> CIELAB conversion (D65 white point). This is the color
// space real skin-tone/whitening algorithms in both the cosmetics-science
// and beauty-camera literature actually work in, rather than adjusting R/G/B
// channels independently: L* is lightness alone, decoupled from a* (red-
// green) and b* (yellow-blue) chroma, so a lightness lift or a redness
// correction can't accidentally shift hue or desaturate the way independent
// per-channel RGB math does. That decoupling is what "natural" cashes out
// to here — the same correction expressed in RGB can't help nudging color,
// not just brightness, since R/G/B each mix luminance and chrominance
// together.

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}
function linearToSrgb(c: number): number {
  const v = c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055
  return Math.min(1, Math.max(0, v))
}

const WHITE_X = 0.95047
const WHITE_Y = 1.0
const WHITE_Z = 1.08883

function labF(t: number): number {
  return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116
}
function labFInv(f: number): number {
  const f3 = f * f * f
  return f3 > 0.008856 ? f3 : (f - 16 / 116) / 7.787
}

export function rgbToLab(r: number, g: number, b: number): [number, number, number] {
  const rl = srgbToLinear(r / 255)
  const gl = srgbToLinear(g / 255)
  const bl = srgbToLinear(b / 255)

  const x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / WHITE_X
  const y = (0.2126729 * rl + 0.7151522 * gl + 0.072175 * bl) / WHITE_Y
  const z = (0.0193339 * rl + 0.119192 * gl + 0.9503041 * bl) / WHITE_Z

  const fx = labF(x)
  const fy = labF(y)
  const fz = labF(z)

  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]
}

export function labToRgb(L: number, a: number, b: number): [number, number, number] {
  const fy = (L + 16) / 116
  const fx = fy + a / 500
  const fz = fy - b / 200

  const x = labFInv(fx) * WHITE_X
  const y = labFInv(fy) * WHITE_Y
  const z = labFInv(fz) * WHITE_Z

  const rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
  const gl = -0.969266 * x + 1.8760108 * y + 0.041556 * z
  const bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

  return [linearToSrgb(rl) * 255, linearToSrgb(gl) * 255, linearToSrgb(bl) * 255]
}
