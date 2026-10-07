import * as THREE from 'three'
import type { Model } from './scene'
import { surfaceNets } from './sdfMesh'

// "Lavender": a lilac bob wig after the reference photo — a full, glossy
// synthetic bob: a domed crown, heavy blunt bangs down to the brows, and
// straight sides that frame the face and fall to the jaw, curving in a
// little at the ends.
//
// Sculpted as one signed distance field in the head rig's units (origin
// between the eyes, 1 = the eye spacing, y up, z out of the face): an outer
// helmet of hair (solid — the face occluding it is enough), with the face
// opening cut out;
// meshed with surface nets (off the main thread — see wigWorker.ts). Shaded as synthetic fibre:
// strand texture, a stretched shine band, a soft sheen.

const smin = (a: number, b: number, k: number) => {
  const h = Math.max(k - Math.abs(a - b), 0) / k
  return Math.min(a, b) - h * h * k * 0.25
}
const smax = (a: number, b: number, k: number) => -smin(-a, -b, k)
const ell = (x: number, y: number, z: number, rx: number, ry: number, rz: number) => (Math.hypot(x / rx, y / ry, z / rz) - 1) * Math.min(rx, ry, rz)
const ell2 = (x: number, z: number, rx: number, rz: number) => (Math.hypot(x / rx, z / rz) - 1) * Math.min(rx, rz)
const clamp01 = (v: number) => Math.min(1, Math.max(0, v))

/** The head's centre line (front to back), rig z. */
const HZ = -1.55
/** The bangs' blunt edge and the bob's hem (rig y), each a little ragged
 *  the way a cut edge of fibre is. */
const bangsAt = (x: number) => 0.31 - 0.04 * (x / 0.9) ** 2 + 0.006 * Math.sin(x * 57) + 0.004 * Math.sin(x * 133 + 1.3)
const hemAt = (x: number, z: number) => -1.5 + 0.008 * Math.sin(Math.atan2(x, z - HZ) * 41) + 0.005 * Math.sin(x * 97)

export function wigSdf(x: number, y: number, z: number) {
  return wigParts(x, y, z).d
}

/** The field, and how far into the face opening's cut (> 0: on the cut
 *  walls — the inner face of the hair framing the face). */
function wigParts(x: number, y: number, z: number) {
  const ax = Math.abs(x)
  const zc = z - HZ
  // The outside: a rounded crown flowing into a bell — fullest by the
  // cheeks, the ends turning in under the jaw.
  const dome = ell(ax, y - 0.45, zc, 1.33, 1.38, 1.58)
  const rx = 1.24 + 0.07 * Math.exp(-(((y + 0.55) / 0.75) ** 2)) - 0.07 * clamp01((-1.05 - y) / 0.45)
  const side = ell2(ax, zc, rx, rx * 1.2)
  const body = smax(smax(side, y - 0.55, 0.3), hemAt(x, z) - y, 0.16)
  let outer = smin(dome, body, 0.4)
  // Solid: no hollow for the head — the face, which occludes, hides
  // whatever is behind it, and a hollow only doubled the triangles.
  const shell = outer
  // The face opening: everything in front, between the side curtains and
  // under the bangs; the curtains come in a little toward the jaw.
  const half = 0.92 - 0.2 * clamp01((-0.5 - y) / 1.0)
  // (its top corners rounded: the bangs blend into the sides)
  const opening = Math.max(smax(ax - half, y - bangsAt(x), 0.14), -(z + 0.9))
  return { d: smax(shell, -opening, 0.025), cut: -opening - shell }
}

export interface WigData {
  pos: Float32Array
  nrm: Float32Array
  uv: Float32Array
  col: Float32Array
  index: Uint32Array
}

let cache: WigData | null = null
/** The wig, meshed (the heavy part: plain numbers, so it can run in a
 *  worker). */
export function computeWig(): WigData {
  if (cache) return cache
  // (a coarse grid is plenty: the strands are in the texture, not the
  // mesh, and fewer triangles keep live frames quick)
  const { pos, index } = surfaceNets(wigSdf, { x: -1.75, y: -1.85, z: -3.45 }, { x: 1.75, y: 2.45, z: 0.45 }, 0.04)
  const n = pos.length / 3
  const nrm = new Float32Array(n * 3)
  const uv = new Float32Array(n * 2)
  const col = new Float32Array(n * 3)
  const e = 0.004
  // Lilac, a little deeper at the roots and paler at the tips (sRGB hex,
  // to linear for the vertex colours).
  const lin = (hex: number) => [16, 8, 0].map((b) => ((((hex >> b) & 255) / 255 + 0.055) / 1.055) ** 2.4)
  const root = lin(0x8a6dd2)
  const mid = lin(0xa58be8)
  const tip = lin(0xbca7f2)
  for (let i = 0; i < n; i++) {
    const x = pos[i * 3]
    const y = pos[i * 3 + 1]
    const z = pos[i * 3 + 2]
    let gx = wigSdf(x + e, y, z) - wigSdf(x - e, y, z)
    let gy = wigSdf(x, y + e, z) - wigSdf(x, y - e, z)
    let gz = wigSdf(x, y, z + e) - wigSdf(x, y, z - e)
    const l = Math.hypot(gx, gy, gz) || 1
    gx /= l
    gy /= l
    gz /= l
    nrm.set([gx, gy, gz], i * 3)
    // Strands run down from the crown: u round the head, v down it.
    uv[i * 2] = Math.atan2(x, z - HZ) / (Math.PI * 2) + 0.5
    uv[i * 2 + 1] = y * 0.25
    const t = clamp01((1.9 - y) / 3.4)
    const c = t < 0.35 ? root.map((v, k) => v + (mid[k] - v) * (t / 0.35)) : mid.map((v, k) => v + (tip[k] - v) * ((t - 0.35) / 0.65))
    // Undersides and the inner face of the hair round the face (the cut
    // walls) deeper: the light that reaches them has come through hair.
    const under = 0.62 + 0.38 * clamp01(gy * 0.9 + 0.7)
    const wall = 1 - 0.5 * clamp01(wigParts(x, y, z).cut / 0.04 + 0.5)
    // …and the curtains a shade deeper toward the face, where they turn in.
    const half = 0.92 - 0.2 * clamp01((-0.5 - y) / 1.0)
    const rim = y < 0.3 && z > -1.3 ? 0.72 + 0.28 * clamp01((Math.abs(x) - half) / 0.25) : 1
    col.set(c.map((v) => v * under * wall * rim), i * 3)
  }
  // Wind every triangle to face along its normal (the mesher's winding
  // depends on the field's sign convention; double-sided rendering flips
  // the normals of anything wound the other way).
  const idx = Uint32Array.from(index)
  for (let t = 0; t < idx.length; t += 3) {
    const [a, b, c] = [idx[t], idx[t + 1], idx[t + 2]]
    const e1 = [0, 1, 2].map((k) => pos[b * 3 + k] - pos[a * 3 + k])
    const e2 = [0, 1, 2].map((k) => pos[c * 3 + k] - pos[a * 3 + k])
    const cx = e1[1] * e2[2] - e1[2] * e2[1]
    const cy = e1[2] * e2[0] - e1[0] * e2[2]
    const cz = e1[0] * e2[1] - e1[1] * e2[0]
    if (cx * nrm[a * 3] + cy * nrm[a * 3 + 1] + cz * nrm[a * 3 + 2] < 0) {
      idx[t + 1] = c
      idx[t + 2] = b
    }
  }
  // The strands' u wraps round the back of the head: a triangle across the
  // wrap gets its low side's vertices copied at u + 1, or it would squeeze
  // the whole texture into one thin line.
  const extra = new Map<number, number>()
  const P2 = Array.from(pos)
  const N2 = Array.from(nrm)
  const U2 = Array.from(uv)
  const C2 = Array.from(col)
  for (let t = 0; t < idx.length; t += 3) {
    const us = [0, 1, 2].map((k) => U2[idx[t + k] * 2])
    if (Math.max(...us) - Math.min(...us) < 0.5) continue
    for (let k = 0; k < 3; k++) {
      const v = idx[t + k]
      if (U2[v * 2] >= 0.5) continue
      let w = extra.get(v)
      if (w === undefined) {
        w = P2.length / 3
        extra.set(v, w)
        P2.push(P2[v * 3], P2[v * 3 + 1], P2[v * 3 + 2])
        N2.push(N2[v * 3], N2[v * 3 + 1], N2[v * 3 + 2])
        U2.push(U2[v * 2] + 1, U2[v * 2 + 1])
        C2.push(C2[v * 3], C2[v * 3 + 1], C2[v * 3 + 2])
      }
      idx[t + k] = w
    }
  }
  cache = { pos: Float32Array.from(P2), nrm: Float32Array.from(N2), uv: Float32Array.from(U2), col: Float32Array.from(C2), index: idx }
  return cache
}

export const wigTransferables = (d: WigData) => [d.pos, d.nrm, d.uv, d.col, d.index].map((a) => a.buffer as ArrayBuffer)

let worker: Worker | null = null
let pending: Promise<WigData> | null = null
function computeWigOffThread(): Promise<WigData> {
  if (cache) return Promise.resolve(cache)
  pending ??= new Promise<WigData>((resolve) => {
    try {
      worker ??= new Worker(new URL('./wigWorker.ts', import.meta.url), { type: 'module' })
    } catch {
      resolve(computeWig())
      return
    }
    worker.onmessage = (e: MessageEvent<{ data?: WigData }>) => {
      if (e.data.data) resolve((cache = e.data.data))
      else resolve(computeWig())
    }
    worker.postMessage('wig')
  })
  return pending
}

/** Fine strands: a colour map of thin streaks (each strand a little
 *  lighter or darker) and the normal map they make, so light catches
 *  them strand by strand. */
function strandMaps() {
  const W = 1024
  const H = 256
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const g = c.getContext('2d', { willReadFrequently: true })!
  g.fillStyle = 'rgb(228,228,228)'
  g.fillRect(0, 0, W, H)
  let seed = 11
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647
  for (let i = 0; i < 2600; i++) {
    const x = rnd() * W
    const v = 185 + rnd() * 70
    g.strokeStyle = `rgba(${v},${v},${v},${0.35 + rnd() * 0.5})`
    g.lineWidth = 0.6 + rnd() * 1.6
    g.beginPath()
    g.moveTo(x, -10)
    // (a slight wave down each strand)
    g.bezierCurveTo(x + (rnd() - 0.5) * 6, H * 0.33, x + (rnd() - 0.5) * 6, H * 0.66, x + (rnd() - 0.5) * 4, H + 10)
    g.stroke()
  }
  const map = new THREE.CanvasTexture(c)
  map.colorSpace = THREE.SRGBColorSpace
  // Normals from the streaks' brightness across them.
  const src = g.getImageData(0, 0, W, H).data
  const n = document.createElement('canvas')
  n.width = W
  n.height = H
  const ng = n.getContext('2d')!
  const out = ng.createImageData(W, H)
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const at = (xx: number) => src[(y * W + ((xx + W) % W)) * 4] / 255
      const dx = (at(x + 1) - at(x - 1)) * 2.2
      const k = (y * W + x) * 4
      const l = Math.hypot(dx, 1)
      out.data[k] = (-dx / l) * 127 + 128
      out.data[k + 1] = 128
      out.data[k + 2] = (1 / l) * 127 + 128
      out.data[k + 3] = 255
    }
  ng.putImageData(out, 0, 0)
  const normal = new THREE.CanvasTexture(n)
  for (const t of [map, normal]) {
    t.wrapS = t.wrapT = THREE.RepeatWrapping
    t.repeat.set(3, 2)
    t.anisotropy = 4
  }
  return { map, normal }
}

// The face opening's outline (rig x, y), for the hair remover to leave
// alone; scaled with the wig each frame.
const OPENING0: [number, number][] = [
  [-0.92, 0.3], [0.92, 0.3], [0.92, -0.5], [0.72, -1.5], [-0.72, -1.5], [-0.92, -0.5],
]

/** The wig, ready to wear. */
export async function lavenderWig(): Promise<Model> {
  const d = await computeWigOffThread()
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(d.pos, 3))
  g.setAttribute('normal', new THREE.BufferAttribute(d.nrm, 3))
  g.setAttribute('uv', new THREE.BufferAttribute(d.uv, 2))
  g.setAttribute('color', new THREE.BufferAttribute(d.col, 3))
  g.setIndex(new THREE.BufferAttribute(d.index, 1))
  g.computeBoundingSphere()
  const { map, normal } = strandMaps()
  // Synthetic fibre: a plain standard material (a physical one with
  // anisotropy, sheen and clear coat cost twice the frame time live) plus
  // a strand highlight of its own — Kajiya-Kay: light glints off each
  // fibre in a band across the strands, the ring of shine round a head of
  // smooth hair — a bright white lobe and a softer lilac one below it.
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    map,
    normalMap: normal,
    normalScale: new THREE.Vector2(0.6, 0.6),
    roughness: 0.5,
    metalness: 0,
    envMapIntensity: 0.7,
  })
  material.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vStrand;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vStrand = normalize(normalMatrix * vec3(0.0, 1.0, 0.0));')
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vStrand;')
      .replace(
        '#include <opaque_fragment>',
        `#if NUM_DIR_LIGHTS > 0
  {
    vec3 Lh = directionalLights[0].direction;
    vec3 Vh = normalize(vViewPosition);
    vec3 Hh = normalize(Lh + Vh);
    // The strand's direction, in the surface: the head's "up" with the
    // normal taken out, tilted a touch to shift each lobe along it.
    vec3 T = normalize(vStrand - normal * dot(normal, vStrand));
    float t1 = dot(normalize(T + normal * 0.12), Hh);
    float t2 = dot(normalize(T - normal * 0.18), Hh);
    float s1 = pow(max(0.0, sqrt(max(0.0, 1.0 - t1 * t1))), 160.0);
    float s2 = pow(max(0.0, sqrt(max(0.0, 1.0 - t2 * t2))), 45.0);
    float lit = clamp(dot(normal, Lh) * 0.5 + 0.5, 0.0, 1.0);
    outgoingLight += directionalLights[0].color * lit * (s1 * 0.13 * vec3(0.95, 0.9, 1.0) + s2 * 0.12 * diffuseColor.rgb);
  }
#endif
#include <opaque_fragment>`,
      )
  }
  material.customProgramCacheKey = () => 'wig-strands'
  const mesh = new THREE.Mesh(g, material)
  // (no shadow pass for it: the bangs' shade on the brow is too subtle to
  // be worth drawing the whole wig twice a frame)
  mesh.castShadow = false
  mesh.frustumCulled = false
  const wig = new THREE.Group()
  wig.add(mesh)
  const root = new THREE.Group()
  root.add(wig)
  const OPENING = OPENING0.map((p) => [...p] as [number, number])
  return {
    root,
    // The face shows through the opening, and only the face hides the
    // inside of the wig; whatever of the wearer's own hair shows outside
    // it is painted out, and the head just past its edge down to the jaw.
    occludeFace: true,
    // Hair is painted out above the hem only (just inside it, so the change
    // hides under the wig's edge); hair falling below the wig stays.
    hidesHead: -1.4,
    hairOnly: true,
    // (the face opening, at its natural size — scaled with the wig below)
    keepOpening: OPENING,
    vivid: true,
    update(rig) {
      // One size, scaled to this face's width at the cheeks (it was made
      // round a face 2.16 eye spacings wide).
      const w = Math.abs(rig.local(454).x - rig.local(234).x)
      const k = THREE.MathUtils.clamp(w / 2.16, 0.88, 1.2)
      wig.scale.setScalar(k)
      OPENING.forEach((p, i) => ((p[0] = OPENING0[i][0] * k), (p[1] = OPENING0[i][1] * k)))
    },
  }
}
