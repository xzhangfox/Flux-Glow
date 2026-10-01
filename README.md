# Flux Glow

Skin smoothing, blemish reduction, and filters — entirely in the browser. No uploads, no account, no VIP paywall.

Part of the [Flux](https://xzhangfox.github.io) family of apps.

## How it works

- **Face detection**: [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (478-point face mesh), self-hosted (WASM runtime + model are committed under `public/mediapipe/`, not fetched from a third-party CDN at runtime).
- **Skin mask**: the face oval minus eyes, eyebrows, and lips, built from the detected landmarks, so smoothing only ever touches actual skin.
- **Smoothing**: real frequency separation (not a flat blur) — a lightly-blurred copy of the image gives the fine texture layer (pores, fine lines), a second, heavier blur of that same layer gives the tone/blemish layer, and the two recombine with only the tone layer smoothed. That's what keeps retouched skin looking like skin instead of plastic, and it reduces blemishes as a side effect of smoothing blotchy tone, without a separate spot-removal tool.
- **Reshape**: face/jaw (slim), eyes (enlarge), and mouth (plump) are a GPU triangulated-mesh warp — the same technique real-time beauty-camera filters (Snapchat, Meitu-style effects) use, not a hand-rolled pixel-pushing algorithm. MediaPipe's 468 landmarks are vertices of a fixed, published triangle mesh (`faceTriangulation.ts`, extracted from MediaPipe's own canonical face model); warping a feature is just moving its vertices toward a target position, and a WebGL vertex/fragment shader pair (`meshWarp.ts`) texture-maps the original photo onto the moved mesh in real time. The un-warped background is a completely separate full-image layer drawn first and never touched by the warp math, and a small ring of anchored "skirt" vertices just outside the face oval absorbs the jaw's inward push into the background smoothly. A triangle mesh can't fold the way a numerically-solved deformation field can — moving vertices on a fixed, fine mesh is what actually fixes the two failure modes [an earlier Moving Least Squares-based version](https://www.cs.jhu.edu/~misha/Fall15/Papers/Schaefer06.pdf) kept hitting on real photos: wavy backgrounds from under-anchored control points, and eyes folding into solid-color smears when pinned too tightly. Nose (narrow) stays a separate, isolated monotonic radial zoom — it has no official MediaPipe landmark loop to pick mesh vertices from without guessing raw indices.
- **Filters**: a handful of CSS canvas filter presets (warm, cool, vivid, mono, soft glow).

Everything runs client-side on an uploaded or captured photo; nothing is sent anywhere.

## Development

```bash
npm install
npm run dev
npm run build
```
