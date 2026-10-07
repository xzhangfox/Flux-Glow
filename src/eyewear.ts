// Dev-only: a model on its own (sunglasses, ?id=lavender for the wig),
// front / 3-4 / side, on a studio grey.
import * as THREE from 'three'
import { buildModel } from './lib/ar/models'
import { lavenderWig } from './lib/ar/wig'
const q = new URLSearchParams(location.search)
const model = q.get('id') === 'lavender' ? await lavenderWig() : buildModel(q.get('id') ?? 'sport')
const W = 420, H = 300
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true })
renderer.setSize(W * 3, H)
renderer.outputColorSpace = THREE.SRGBColorSpace
renderer.toneMapping = THREE.ACESFilmicToneMapping
document.body.appendChild(renderer.domElement)
renderer.setScissorTest(true)
const scene = new THREE.Scene()
scene.background = new THREE.Color(q.get('bg') ?? '#d8d8dc')
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.7))
const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(-2, 4, 6); scene.add(key)
const glasses = model.root.children[0]
glasses.position.set(0, 0, 0)
const pivot = new THREE.Group(); pivot.add(glasses); scene.add(pivot)
const big = q.get('id') === 'lavender'
const cam = big ? new THREE.OrthographicCamera(-2.4, 2.4, 2.3, -2.4, -50, 50) : new THREE.OrthographicCamera(-1.6, 1.6, 1.15, -1.15, -50, 50); cam.position.set(0, 0, 10)
;[[0, 0], [0.75, 0.15], [big ? 2.6 : 1.45, 0.05]].forEach(([yaw, pitch], i) => {
  pivot.rotation.set(pitch, yaw, 0)
  renderer.setViewport(i * W, 0, W, H); renderer.setScissor(i * W, 0, W, H)
  renderer.render(scene, cam)
})
;(window as any).__done = true
