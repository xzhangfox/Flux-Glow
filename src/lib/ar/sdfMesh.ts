// Signed distance field meshing, shared by the sculpted models (animal
// heads, the wig).

interface V3 {
  x: number
  y: number
  z: number
}

/** Meshes the zero surface of `f` over the box min..max with surface nets
 *  (one vertex per surface cell, quads across sign changes) on a grid of
 *  `cell`: coarse samples first, exact ones only near the surface. */
export function surfaceNets(f: (x: number, y: number, z: number) => number, min: V3, max: V3, cell: number) {
  const nx = Math.ceil((max.x - min.x) / cell) + 1
  const ny = Math.ceil((max.y - min.y) / cell) + 1
  const nz = Math.ceil((max.z - min.z) / cell) + 1
  // Coarse pass first; exact values only near the surface.
  const C = 4
  const cnx = Math.ceil(nx / C) + 1
  const cny = Math.ceil(ny / C) + 1
  const cnz = Math.ceil(nz / C) + 1
  const coarse = new Float32Array(cnx * cny * cnz)
  for (let k = 0; k < cnz; k++)
    for (let j = 0; j < cny; j++)
      for (let i = 0; i < cnx; i++) coarse[(k * cny + j) * cnx + i] = f(min.x + i * C * cell, min.y + j * C * cell, min.z + k * C * cell)
  const band = cell * C * 2.5
  const vals = new Float32Array(nx * ny * nz)
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const ci = Math.min(cnx - 2, Math.floor(i / C))
        const cj = Math.min(cny - 2, Math.floor(j / C))
        const ck = Math.min(cnz - 2, Math.floor(k / C))
        const tx = i / C - ci
        const ty = j / C - cj
        const tz = k / C - ck
        const g = (a: number, b: number, c: number) => coarse[((ck + c) * cny + (cj + b)) * cnx + (ci + a)]
        const lerp = (a: number, b: number, t: number) => a + (b - a) * t
        const est = lerp(lerp(lerp(g(0, 0, 0), g(1, 0, 0), tx), lerp(g(0, 1, 0), g(1, 1, 0), tx), ty), lerp(lerp(g(0, 0, 1), g(1, 0, 1), tx), lerp(g(0, 1, 1), g(1, 1, 1), tx), ty), tz)
        vals[(k * ny + j) * nx + i] = Math.abs(est) > band ? est : f(min.x + i * cell, min.y + j * cell, min.z + k * cell)
      }
  const idx = (i: number, j: number, k: number) => (k * ny + j) * nx + i
  const cellVert = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1)
  const cidx = (i: number, j: number, k: number) => (k * (ny - 1) + j) * (nx - 1) + i
  const pos: number[] = []
  const corners = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
  ]
  const edges = [
    [0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7],
  ]
  const cv = new Float32Array(8)
  for (let k = 0; k < nz - 1; k++)
    for (let j = 0; j < ny - 1; j++)
      for (let i = 0; i < nx - 1; i++) {
        let neg = 0
        for (let c = 0; c < 8; c++) {
          cv[c] = vals[idx(i + corners[c][0], j + corners[c][1], k + corners[c][2])]
          if (cv[c] < 0) neg++
        }
        if (neg === 0 || neg === 8) continue
        let sx = 0
        let sy = 0
        let sz = 0
        let n = 0
        for (const [a, b] of edges) {
          if (cv[a] < 0 === cv[b] < 0) continue
          const t = cv[a] / (cv[a] - cv[b])
          sx += corners[a][0] + (corners[b][0] - corners[a][0]) * t
          sy += corners[a][1] + (corners[b][1] - corners[a][1]) * t
          sz += corners[a][2] + (corners[b][2] - corners[a][2]) * t
          n++
        }
        cellVert[cidx(i, j, k)] = pos.length / 3
        pos.push(min.x + (i + sx / n) * cell, min.y + (j + sy / n) * cell, min.z + (k + sz / n) * cell)
      }
  const index: number[] = []
  const quad = (a: number, b: number, c: number, d: number, flip: boolean) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return
    if (flip) index.push(a, c, b, a, d, c)
    else index.push(a, b, c, a, c, d)
  }
  for (let k = 1; k < nz - 1; k++)
    for (let j = 1; j < ny - 1; j++)
      for (let i = 1; i < nx - 1; i++) {
        const v0 = vals[idx(i, j, k)] < 0
        // Edge along x to (i+1, j, k): the four cells round it.
        if (i < nx - 1 && v0 !== vals[idx(i + 1, j, k)] < 0)
          quad(cellVert[cidx(i, j - 1, k - 1)], cellVert[cidx(i, j, k - 1)], cellVert[cidx(i, j, k)], cellVert[cidx(i, j - 1, k)], v0)
        if (j < ny - 1 && v0 !== vals[idx(i, j + 1, k)] < 0)
          quad(cellVert[cidx(i - 1, j, k - 1)], cellVert[cidx(i - 1, j, k)], cellVert[cidx(i, j, k)], cellVert[cidx(i, j, k - 1)], v0)
        if (k < nz - 1 && v0 !== vals[idx(i, j, k + 1)] < 0)
          quad(cellVert[cidx(i - 1, j - 1, k)], cellVert[cidx(i, j - 1, k)], cellVert[cidx(i, j, k)], cellVert[cidx(i - 1, j, k)], v0)
      }
  return { pos, index }
}
