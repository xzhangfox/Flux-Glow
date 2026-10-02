# Flux Glow

Skin smoothing, blemish reduction, and filters — entirely in the browser. No uploads, no account, no VIP paywall.

Part of the [Flux](https://xzhangfox.github.io) family of apps.

## How it works

- **Face detection**: [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (478-point face mesh), self-hosted (WASM runtime + model are committed under `public/mediapipe/`, not fetched from a third-party CDN at runtime).
- **Skin mask**: the face oval minus eyes, eyebrows, and lips, built from the detected landmarks, so smoothing only ever touches actual skin.
- **Smoothing**: real frequency separation (not a flat blur) — a lightly-blurred copy of the image gives the fine texture layer (pores, fine lines), a second, heavier blur of that same layer gives the tone/blemish layer, and the two recombine with only the tone layer smoothed. That's what keeps retouched skin looking like skin instead of plastic, and it reduces blemishes as a side effect of smoothing blotchy tone, without a separate spot-removal tool.
- **Reshape**: nine independent, bidirectional regions — jaw, temple, cheekbone, eyes, eyebrow height, nose width, nose bridge height, and mouth (smoothing is the only one-directional control) — all a GPU triangulated-mesh warp, the same technique real-time beauty-camera filters (Snapchat, Meitu-style effects) use, not a hand-rolled pixel-pushing algorithm. MediaPipe's 468 landmarks are vertices of a fixed, published triangle mesh (`faceTriangulation.ts`, extracted from MediaPipe's own canonical face model); adjusting a region is just moving its vertices toward a target position (negative/positive move the two opposite ways, 0 is the untouched original), and a WebGL vertex/fragment shader pair (`meshWarp.ts`) texture-maps the original photo onto the moved mesh in real time. The un-warped background is a completely separate full-image layer drawn first and never touched by the warp math, and a small ring of anchored "skirt" vertices just outside the face oval absorbs the jaw/temple/cheekbone pushes into the background smoothly. Temple and cheekbone are specific points on that same oval (not separate landmark loops — verified by rendering all 468 indices on a real photo), given their own cosine-squared falloff in oval arc position so they can be adjusted independently of jaw without a seam where their influence ends. A triangle mesh can't fold the way a numerically-solved deformation field can — moving vertices on a fixed, fine mesh is what actually fixes the two failure modes [an earlier Moving Least Squares-based version](https://www.cs.jhu.edu/~misha/Fall15/Papers/Schaefer06.pdf) kept hitting on real photos: wavy backgrounds from under-anchored control points, and eyes folding into solid-color smears when pinned too tightly. Jaw/temple/cheekbone's horizontal push also fades out as a turned head moves away from frontal (measured via eye-corner vs. cheek-to-cheek midpoint offset) — pushing both sides of the oval toward one centerX only makes sense face-on, and forcing it at an angle was a second, independent source of visible seams. Nose width stays a separate, isolated monotonic radial zoom — it has no official MediaPipe landmark loop to pick mesh vertices from without guessing raw indices.
- **Filters**: a handful of CSS canvas filter presets (warm, cool, vivid, mono, soft glow).

The app opens straight into the live camera; a small photo icon next to the shutter switches to editing an uploaded photo instead. Retouch is a two-level drill-down — tap a region's icon (each shows *where* on a face it adjusts) and that row becomes that region's own slider — rather than nine sliders all shown flat at once. Everything runs client-side on an uploaded or captured photo; nothing is sent anywhere.

## Development

```bash
npm install
npm run dev
npm run build
```
