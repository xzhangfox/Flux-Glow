import * as THREE from 'three'

// Shell-rendered fur — the standard real-time technique (used in games for
// grass and animal coats): the ear is drawn N times, each copy pushed a bit
// further out along its normals, and each layer keeps only the pixels that
// belong to a strand still that long. Stacked, those dots become individual
// tapered hairs with real depth and a soft, uneven silhouette.
//
// Shading is done by hand for fur: wrapped diffuse (light bleeds around
// the soft surface), ambient occlusion that darkens toward the roots, a
// rim/sheen at grazing angles, and a root→tip colour ramp — which is what
// reads as "fluffy" rather than "furry-textured plastic".

export interface FurLook {
  root: string
  tip: string
  /** Bare skin colour on the inner ear. */
  skin: string
  /** Strand length, in ear-local units. */
  length: number
  /** Strands across one UV unit. */
  density: number
  /** Comb direction in the ear's own space (fur leans this way). */
  comb: [number, number, number]
  /** Fraction of the inner ear left bare (0 = fully furred). */
  innerBare: number
  /** Colour of the long tufts fanning out of the inner ear. */
  tuft: string
  /** Inner tuft length relative to `length`. */
  tuftLength: number
  /** Optional colour for the top of the ear (fox-style dark tips). */
  earTip?: string
}

const VERT = /* glsl */ `
  attribute float aInner;
  uniform float uShell;
  uniform float uLen;
  uniform float uTuftLen;
  uniform vec3 uComb;
  varying vec2 vUv;
  varying vec3 vN;
  varying vec3 vT;
  varying float vInner;
  void main() {
    float len = uLen * mix(1.0, uTuftLen, aInner);
    // Shells rise off the skin and lean along the comb, so the outer layers
    // of each hair drift toward the ear tip — visible as strands even when
    // the ear faces the camera. Inner-ear tufts fan up and out of the cup,
    // like the long pale tufts on a plush cat ear.
    vec3 fan = normalize(vec3(position.x * 1.8, 0.9, 0.45));
    vec3 comb = mix(uComb * 1.6, fan * 1.9, aInner);
    vec3 p = position + normal * len * uShell + comb * len * uShell;
    vUv = uv;
    vInner = aInner;
    vN = normalize(mat3(modelMatrix) * normal);
    vT = normalize(mat3(modelMatrix) * mix(vec3(0.0, 1.0, 0.0), fan, aInner));
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(p, 1.0);
  }
`

const FRAG = /* glsl */ `
  uniform float uShell;
  uniform sampler2D uFur;
  uniform vec2 uRepeat;
  uniform float uInnerBare;
  uniform vec3 uRoot;
  uniform vec3 uTip;
  uniform vec3 uSkin;
  uniform vec3 uTuft;
  uniform vec3 uEarTip;
  uniform float uHasEarTip;
  uniform vec3 uLightDir;
  uniform vec3 uLightColor;
  uniform vec3 uSky;
  uniform vec3 uGround;
  varying vec2 vUv;
  varying vec3 vN;
  varying vec3 vT;
  varying float vInner;

  void main() {
    float t = uShell;
    // r: strand brightness/height (0 gaps … 1 long hair tips), g: per-hair tone.
    vec4 fur = texture2D(uFur, vUv * uRepeat);
    float h = fur.r;
    float cut = 0.12 + 0.86 * t;
    // Soft coverage instead of a hard cut: with alpha-to-coverage + MSAA the
    // hair edges resolve smooth and wispy rather than stair-stepped.
    float alpha = 1.0;
    if (t > 0.0) {
      // Inner tufts: sparse, and thinning out toward the top of the ear, so
      // pink skin shows between them and they read as tufts, not a sheet.
      float bare = uInnerBare + (1.0 - uInnerBare) * smoothstep(0.2, 0.75, vUv.y);
      float c = vInner > 0.5 ? max(cut, bare) : cut;
      alpha = smoothstep(c - 0.06, c + 0.06, h) * (1.0 - 0.35 * t);
      // Tufts are soft and wispy: lower coverage, so they layer into fluff.
      alpha *= mix(1.0, 0.55, vInner);
      if (alpha < 0.02) discard;
    }
    vec3 hair = mix(uRoot, uTip, clamp(h * 0.75 + t * 0.45, 0.0, 1.0)) * mix(0.88, 1.08, fur.g);
    hair = mix(hair, uEarTip, uHasEarTip * smoothstep(0.7, 0.86, vUv.y));
    // Inner ear: pink skin at the base, long pale tufts above it.
    vec3 tuft = mix(uSkin, uTuft, smoothstep(0.0, 0.35, t)) * mix(0.9, 1.05, fur.g);
    vec3 albedo = mix(hair, t <= 0.0 ? uSkin : tuft, vInner);

    vec3 N = normalize(vN) * (gl_FrontFacing ? 1.0 : -1.0);
    float ndl = dot(N, uLightDir);
    float wrap = clamp((ndl + 0.65) / 1.65, 0.0, 1.0);
    vec3 amb = mix(uGround, uSky, N.y * 0.5 + 0.5);
    // Depth inside the coat: gaps between hairs (dark h) and roots are in shade.
    float ao = mix(0.68, 1.0, clamp(0.3 * h + 0.8 * t, 0.0, 1.0));
    vec3 col = albedo * (amb * 0.7 + uLightColor * wrap * 0.9) * ao;
    // Soft sheen on hair tips catching the light, and a grazing-angle rim.
    float rim = pow(1.0 - abs(N.z), 2.0);
    col += uLightColor * albedo * (rim * 0.4 + 0.12 * h * t);
    // Strand highlight (Kajiya-Kay): hair reflects in a band across its
    // length — what gives dark fur its glossy, combed sheen.
    vec3 H = normalize(uLightDir + vec3(0.0, 0.0, 1.0));
    float th = dot(normalize(vT), H);
    col += uLightColor * pow(sqrt(max(0.0, 1.0 - th * th)), 60.0) * 0.35 * t * (0.6 + 0.4 * fur.g);
    // Thin ear skin is translucent: light passing through tints it warm pink.
    col += uSkin * uLightColor * vInner * 0.22 * (1.0 - t);
    gl_FragColor = vec4(col, alpha);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`

let furTex: THREE.CanvasTexture | null = null
/** Thousands of fine, slightly curved hairs combed toward +v (the ear tip),
 *  drawn once: brightness = how far each hair reaches (the shells' cut-off),
 *  green = per-hair tone variation. Tiles seamlessly. */
function furTexture() {
  if (furTex) return furTex
  const N = 512
  const c = document.createElement('canvas')
  c.width = c.height = N
  const ctx = c.getContext('2d')!
  ctx.fillStyle = 'rgb(20,128,0)'
  ctx.fillRect(0, 0, N, N)
  ctx.lineCap = 'round'
  let seed = 11
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647
  for (let i = 0; i < 14000; i++) {
    const x = rnd() * N
    const y = rnd() * N
    const len = 34 + rnd() * 52
    const ang = -Math.PI / 2 + (rnd() - 0.5) * 0.5
    const bend = (rnd() - 0.5) * 18
    const reach = 0.35 + 0.65 * (i / 14000) * (0.5 + 0.5 * rnd())
    const ex = x + Math.cos(ang) * len
    const ey = y + Math.sin(ang) * len
    const g = ctx.createLinearGradient(x, y, ex, ey)
    // Brighter (taller) toward the tip of each hair.
    g.addColorStop(0, `rgb(${Math.round(255 * reach * 0.55)},${Math.round(rnd() * 255)},0)`)
    g.addColorStop(1, `rgb(${Math.round(255 * reach)},${Math.round(rnd() * 255)},0)`)
    ctx.strokeStyle = g
    ctx.lineWidth = 1.1 + rnd() * 1.6
    for (const ox of [-N, 0, N])
      for (const oy of [-N, 0, N]) {
        ctx.beginPath()
        ctx.moveTo(x + ox, y + oy)
        ctx.quadraticCurveTo((x + ex) / 2 + bend + ox, (y + ey) / 2 + oy, ex + ox, ey + oy)
        ctx.stroke()
      }
  }
  furTex = new THREE.CanvasTexture(c)
  furTex.wrapS = furTex.wrapT = THREE.RepeatWrapping
  furTex.colorSpace = THREE.NoColorSpace
  furTex.anisotropy = 8
  return furTex
}

/** A cupped ear: a cone whose cross-section is a crescent — rounded at the
 *  back, scooped in at the front where the inner ear is. Base at y=0, tip
 *  at y=height, front facing +z. `aInner` marks the scooped front. */
export function earGeometry(height: number, width: (v: number) => number, depth: (v: number) => number, bend: (v: number) => [number, number], cup = 0.45, segU = 40, segV = 28) {
  const pos: number[] = []
  const uv: number[] = []
  const inner: number[] = []
  for (let j = 0; j <= segV; j++) {
    const v = j / segV
    const w = Math.max(0.0005, width(v))
    const d = Math.max(0.0005, depth(v))
    const [bx, bz] = bend(v)
    for (let i = 0; i <= segU; i++) {
      const phi = (i / segU) * Math.PI * 2
      const s = Math.sin(phi)
      const c = Math.cos(phi)
      const z = c >= 0 ? -cup * d * c : d * c
      pos.push(w * s + bx, v * height, z + bz)
      uv.push(i / segU, v)
      inner.push(c > 0 ? THREE.MathUtils.smoothstep(c, 0.62, 0.86) * (1 - THREE.MathUtils.smoothstep(v, 0.62, 0.82)) : 0)
    }
  }
  const idx: number[] = []
  for (let j = 0; j < segV; j++)
    for (let i = 0; i < segU; i++) {
      const a = j * (segU + 1) + i
      const b = a + segU + 1
      idx.push(a, b, a + 1, a + 1, b, b + 1)
    }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2))
  g.setAttribute('aInner', new THREE.Float32BufferAttribute(inner, 1))
  g.setIndex(idx)
  g.computeVertexNormals()
  // Normals must point out of the ear (shells grow along them). Check a
  // vertex on the rounded back (φ = π, mid-height), whose outward normal
  // faces −z, and flip everything if the winding came out inside-out.
  const n = g.getAttribute('normal') as THREE.BufferAttribute
  const probe = Math.round(segV / 2) * (segU + 1) + Math.round(segU / 2)
  if (n.getZ(probe) > 0) {
    for (let i = 0; i < n.count; i++) n.setXYZ(i, -n.getX(i), -n.getY(i), -n.getZ(i))
  }
  return g
}

// Shared lighting uniforms, refreshed from the scene's estimated lights
// before each render (see furLighting).
const lightUniforms = {
  uLightDir: { value: new THREE.Vector3(0, 0.3, 1).normalize() },
  uLightColor: { value: new THREE.Color(1, 1, 1) },
  uSky: { value: new THREE.Color(0.6, 0.6, 0.6) },
  uGround: { value: new THREE.Color(0.3, 0.3, 0.3) },
}

export function furLighting(key: THREE.DirectionalLight, hemi: THREE.HemisphereLight) {
  lightUniforms.uLightDir.value.copy(key.position).sub(key.target.position).normalize()
  lightUniforms.uLightColor.value.copy(key.color).multiplyScalar(key.intensity * 0.42)
  lightUniforms.uSky.value.copy(hemi.color).multiplyScalar(hemi.intensity)
  lightUniforms.uGround.value.copy(hemi.groundColor).multiplyScalar(hemi.intensity)
}

/** The ear drawn as `shells` fur layers (layer 0 is the opaque skin/under-
 *  coat, which is also what casts the shadow). */
export function furMesh(geo: THREE.BufferGeometry, look: FurLook, shells: number): THREE.Group {
  const g = new THREE.Group()
  for (let k = 0; k < shells; k++) {
    const mat = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      side: THREE.DoubleSide,
      alphaToCoverage: true,
      uniforms: {
        ...lightUniforms,
        uShell: { value: k / (shells - 1) },
        uLen: { value: look.length },
        uComb: { value: new THREE.Vector3(...look.comb) },
        uFur: { value: furTexture() },
        // The texture is uploaded flipped (canvas top = v 1), so hairs drawn
        // "up" the canvas point toward the ear tip. `density` scales the tiling.
        uRepeat: { value: new THREE.Vector2(look.density / 40, look.density / 60) },
        uInnerBare: { value: look.innerBare },
        uRoot: { value: new THREE.Color(look.root) },
        uTip: { value: new THREE.Color(look.tip) },
        uSkin: { value: new THREE.Color(look.skin) },
        uTuft: { value: new THREE.Color(look.tuft) },
        uTuftLen: { value: look.tuftLength },
        uEarTip: { value: new THREE.Color(look.earTip ?? '#000000') },
        uHasEarTip: { value: look.earTip ? 1 : 0 },
      },
    })
    const m = new THREE.Mesh(geo, mat)
    m.renderOrder = k
    m.frustumCulled = false
    if (k === 0) {
      // Only the base layer casts the ear's shadow.
      m.castShadow = true
    }
    g.add(m)
  }
  return g
}
