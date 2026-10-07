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
/** The neck: an elliptic column under the head (rig units). */
const NECK_Z = -1.3
const NECK_RX = 0.6
const NECK_RZ = 0.66
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
  // whatever is behind it, and a hollow only doubled the triangles — but a
  // channel for the neck under it, so the back hair falls round the neck
  // instead of filling the space where it is.
  const neck = smax(ell2(ax, z - NECK_Z, NECK_RX + 0.06, NECK_RZ + 0.06), y + 0.45, 0.25)
  const shell = smax(outer, -neck, 0.08)
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
  const root = lin(0x7e60cf)
  const mid = lin(0x9c80e6)
  const tip = lin(0xb39cf0)
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
  g.fillStyle = 'rgb(214,214,214)'
  g.fillRect(0, 0, W, H)
  let seed = 11
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647
  // Lighter strands and violet-shadowed lowlights between them, as in a
  // synthetic wig's glossy, separated fibre.
  for (let i = 0; i < 3000; i++) {
    const x = rnd() * W
    const v = rnd() < 0.3 ? 120 + rnd() * 50 : 190 + rnd() * 65
    g.strokeStyle = `rgba(${v},${v},${v},${0.35 + rnd() * 0.55})`
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

// ---- Wisps -----------------------------------------------------------------------
//
// What makes a cut edge of hair look like hair rather than a moulding: the
// ends aren't one blunt surface but strands, a little uneven in length, the
// light coming through between them. So along the bangs and round the hem,
// thin cards of strand texture hang from the solid edge, mostly opaque at
// the root and breaking up into see-through strands at the tips.

/** The strand-ends texture: white (tinted by the material), alpha strands
 *  of uneven length tapering to nothing. */
function wispTexture() {
  const W = 128
  const H = 256
  const c = document.createElement('canvas')
  c.width = W
  c.height = H
  const g = c.getContext('2d')!
  let seed = 5
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647
  // A solid band at the root, so a card starts as the edge of the hair.
  const root = g.createLinearGradient(0, 0, 0, H * 0.3)
  root.addColorStop(0, 'rgba(225,225,225,1)')
  root.addColorStop(1, 'rgba(225,225,225,0)')
  g.fillStyle = root
  g.fillRect(0, 0, W, H * 0.3)
  for (let i = 0; i < 70; i++) {
    const x = rnd() * W
    const len = H * (0.45 + rnd() * 0.55)
    const grad = g.createLinearGradient(0, 0, 0, len)
    const v = 165 + rnd() * 70
    grad.addColorStop(0, `rgba(${v},${v},${v},0.95)`)
    grad.addColorStop(0.75, `rgba(${v},${v},${v},0.55)`)
    grad.addColorStop(1, `rgba(${v},${v},${v},0)`)
    g.strokeStyle = grad
    g.lineWidth = 1 + rnd() * 2.4
    g.lineCap = 'round'
    g.beginPath()
    g.moveTo(x, 0)
    g.quadraticCurveTo(x + (rnd() - 0.5) * 8, len * 0.6, x + (rnd() - 0.5) * 14, len)
    g.stroke()
  }
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  t.anisotropy = 4
  return t
}

/** Where a ray from `from` along `dir` first meets the wig (null: misses). */
function hit(from: THREE.Vector3, dir: THREE.Vector3) {
  const p = from.clone()
  for (let i = 0; i < 80; i++) {
    const d = wigSdf(p.x, p.y, p.z)
    if (d < 0.002) return p
    p.addScaledVector(dir, Math.max(d * 0.9, 0.004))
    if (p.lengthSq() > 40) return null
  }
  return null
}

/** The wisp cards: quads hung from the bangs' edge (facing forward) and
 *  round the hem (facing out), each with a sway weight (0 at its root, 1
 *  at its tip) and a phase of its own for the breeze. */
function wispGeometry() {
  const pos: number[] = []
  const uv: number[] = []
  const sway: number[] = []
  const phase: number[] = []
  const index: number[] = []
  let seed = 9
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647
  const card = (top: THREE.Vector3, down: THREE.Vector3, across: THREE.Vector3, out: THREE.Vector3, w: number, len: number) => {
    const base = pos.length / 3
    const ph = rnd() * Math.PI * 2
    const u0 = Math.floor(rnd() * 4) / 4
    for (const [sx, sy] of [[-0.5, 0], [0.5, 0], [-0.5, 1], [0.5, 1]]) {
      const p = top.clone().addScaledVector(across, sx * w).addScaledVector(down, sy * len).addScaledVector(out, 0.004 + sy * 0.012)
      pos.push(p.x, p.y, p.z)
      uv.push(u0 + (sx + 0.5) * 0.25, 1 - sy)
      sway.push(sy)
      phase.push(ph)
    }
    index.push(base, base + 2, base + 1, base + 1, base + 2, base + 3)
  }
  // Bangs: across the front, just above the blunt edge, a little in front
  // of the hair's surface.
  for (let x = -0.97; x <= 0.97; x += 0.034) {
    const xx = x + (rnd() - 0.5) * 0.01
    const y = bangsAt(xx) + 0.07
    const p = hit(new THREE.Vector3(xx, y, 1.2), new THREE.Vector3(0, 0, -1))
    if (!p) continue
    const len = 0.1 + rnd() * 0.07
    card(p, new THREE.Vector3(0, -1, 0.08).normalize(), new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 1), 0.07, len)
  }
  // The hem: round the sides and back, hung from just above the edge,
  // facing outward; none across the face opening.
  for (let a = -Math.PI; a < Math.PI; a += 0.042) {
    const aa = a + (rnd() - 0.5) * 0.02
    const dir = new THREE.Vector3(Math.sin(aa), 0, Math.cos(aa))
    const y = -1.4 + (rnd() - 0.5) * 0.03
    const from = new THREE.Vector3(0, y, HZ).addScaledVector(dir, 2.4)
    const p = hit(from, dir.clone().negate())
    if (!p) continue
    if (p.z > -0.95 && Math.abs(p.x) < 1.0) continue
    const len = 0.13 + rnd() * 0.08
    // (curving in a touch under the bob's turned-in ends)
    card(p, new THREE.Vector3(0, -1, 0).addScaledVector(dir, -0.15).normalize(), new THREE.Vector3(dir.z, 0, -dir.x), dir, 0.09, len)
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2))
  g.setAttribute('aSway', new THREE.Float32BufferAttribute(sway, 1))
  g.setAttribute('aPhase', new THREE.Float32BufferAttribute(phase, 1))
  g.setIndex(index)
  g.computeVertexNormals()
  return g
}

// ---- Movement ---------------------------------------------------------------------
//
// Hair lags behind the head and swings back: the bob's lower half (and the
// wisps' tips most of all) follows a point under the head on a damped
// spring, so a turn or a nod sets it swaying, and a faint breeze keeps it
// from ever being quite still. The crown and bangs' roots stay put.

const SWAY_GLSL = `
uniform vec3 uSway;
uniform float uTime;
vec3 swayed(vec3 p, float w, float phase) {
  vec3 breeze = vec3(sin(uTime * 1.7 + phase) * 0.012 + sin(uTime * 2.9 + phase * 1.7) * 0.006, 0.0, sin(uTime * 1.3 + phase) * 0.006);
  return p + (uSway + breeze) * w;
}
`

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
  const motion = { uSway: { value: new THREE.Vector3() }, uTime: { value: 0 } }
  // Synthetic fibre: a plain standard material (a physical one with
  // anisotropy, sheen and clear coat cost twice the frame time live) plus
  // a strand highlight of its own — Kajiya-Kay: light glints off each
  // fibre in a band across the strands, the ring of shine round a head of
  // smooth hair — a bright lobe and a softer lilac one below it, both
  // broken up strand by strand like the gloss on a synthetic wig.
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    map,
    normalMap: normal,
    normalScale: new THREE.Vector2(0.7, 0.7),
    roughness: 0.42,
    metalness: 0,
    envMapIntensity: 0.8,
  })
  material.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, motion)
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vStrand;\n' + SWAY_GLSL)
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
  vStrand = normalize(normalMatrix * vec3(0.0, 1.0, 0.0));
  // (the lower half sways, more toward the hem; never the crown)
  float sw = smoothstep(-0.2, -1.5, position.y);
  transformed = swayed(transformed, sw * sw, position.x * 3.0 + position.z * 2.0);`,
      )
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
    float s1 = pow(max(0.0, sqrt(max(0.0, 1.0 - t1 * t1))), 140.0);
    float s2 = pow(max(0.0, sqrt(max(0.0, 1.0 - t2 * t2))), 40.0);
    float lit = clamp(dot(normal, Lh) * 0.5 + 0.5, 0.0, 1.0);
    // How light this strand is (the strand texture, over the base colour):
    // the gloss streaks along the lighter fibres.
    float strand = clamp(dot(diffuseColor.rgb, vec3(0.333)) / max(dot(vColor.rgb, vec3(0.333)), 1e-3), 0.0, 1.4);
    float streak = smoothstep(0.55, 1.0, strand);
    outgoingLight += directionalLights[0].color * lit * (s1 * 0.22 * streak * vec3(0.96, 0.92, 1.0) + s2 * 0.14 * diffuseColor.rgb);
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

  // The wisps, drawn after the wig (see-through: blended, not writing
  // depth), lilac like the ends of the hair.
  const lin = (hex: number) => new THREE.Color(hex)
  const wispMat = new THREE.MeshStandardMaterial({ color: lin(0x8c72da), map: wispTexture(), transparent: true, depthWrite: false, side: THREE.DoubleSide, roughness: 0.45, alphaTest: 0.02 })
  wispMat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, motion)
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float aSway;\nattribute float aPhase;\n' + SWAY_GLSL)
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
  // (each wisp hangs from where it meets the hair, which sways too)
  float base = smoothstep(-0.2, -1.5, position.y);
  transformed = swayed(transformed, base * base + aSway * 0.6, aPhase);
  transformed.x += sin(uTime * 3.1 + aPhase) * 0.008 * aSway;`,
      )
  }
  wispMat.customProgramCacheKey = () => 'wig-wisps'
  const wisps = new THREE.Mesh(wispGeometry(), wispMat)
  wisps.renderOrder = 2
  wisps.frustumCulled = false

  // The neck, depth only: it hides the hair behind it, as the face does
  // above it (only the face is in the scene's own occluder).
  const neckGeo = new THREE.CylinderGeometry(1, 1, 3, 32, 1, true)
  neckGeo.translate(0, -2.05, 0)
  const neck = new THREE.Mesh(neckGeo, new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.DoubleSide }))
  neck.scale.set(NECK_RX - 0.04, 1, NECK_RZ - 0.04)
  neck.position.z = NECK_Z
  neck.renderOrder = -5
  neck.frustumCulled = false

  const wig = new THREE.Group()
  wig.add(neck, mesh, wisps)
  const root = new THREE.Group()
  root.add(wig)
  const OPENING = OPENING0.map((p) => [...p] as [number, number])

  // The spring: where the hair's lower half "is" (world), chasing a point
  // under the head.
  const anchorLocal = new THREE.Vector3(0, -1.4, -1.5)
  const spring = { p: null as THREE.Vector3 | null, v: new THREE.Vector3(), t: 0 }
  const tmp = new THREE.Vector3()
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
    update(rig, t) {
      // One size, scaled to this face's width at the cheeks (it was made
      // round a face 2.16 eye spacings wide).
      const w = Math.abs(rig.local(454).x - rig.local(234).x)
      const k = THREE.MathUtils.clamp(w / 2.16, 0.88, 1.2)
      wig.scale.setScalar(k)
      OPENING.forEach((p, i) => ((p[0] = OPENING0[i][0] * k), (p[1] = OPENING0[i][1] * k)))
      motion.uTime.value = t
      // Stills: hanging straight.
      if (!rig.live) {
        spring.p = null
        motion.uSway.value.set(0, 0, 0)
        return
      }
      const anchor = anchorLocal.clone().applyMatrix4(rig.matrix)
      const dt = Math.min(0.05, Math.max(0, t - spring.t))
      spring.t = t
      if (!spring.p || dt === 0) {
        spring.p = spring.p ?? anchor.clone()
        spring.v.set(0, 0, 0)
      } else {
        // A soft, under-damped spring (in pixels; stiffness per second²).
        const K = 70
        const C = 7
        tmp.copy(anchor).sub(spring.p).multiplyScalar(K).addScaledVector(spring.v, -C)
        spring.v.addScaledVector(tmp, dt)
        spring.p.addScaledVector(spring.v, dt)
      }
      // The lag, in the head's own units and axes, kept within reason.
      const lag = spring.p.clone().sub(anchor)
      lag.applyMatrix3(new THREE.Matrix3().setFromMatrix4(rig.inverse)).divideScalar(k)
      lag.z *= 0.5
      if (lag.length() > 0.22) lag.setLength(0.22)
      motion.uSway.value.copy(lag)
    },
  }
}
