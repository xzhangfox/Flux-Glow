# Flux Glow

Skin smoothing, blemish reduction, and filters — entirely in the browser. No uploads, no account, no VIP paywall.

Part of the [Flux](https://xzhangfox.github.io) family of apps.

## How it works

- **Face detection**: [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (478-point face mesh), self-hosted (WASM runtime + model are committed under `public/mediapipe/`, not fetched from a third-party CDN at runtime).
- **Skin mask**: the face oval minus eyes, eyebrows, and lips, built from the detected landmarks, so smoothing only ever touches actual skin.
- **Smoothing**: real frequency separation (not a flat blur) — a lightly-blurred copy of the image gives the fine texture layer (pores, fine lines), a second, heavier blur of that same layer gives the tone/blemish layer, and the two recombine with only the tone layer smoothed. That's what keeps retouched skin looking like skin instead of plastic, and it reduces blemishes as a side effect of smoothing blotchy tone, without a separate spot-removal tool.
- **Reshape**: per-region landmark-based warping — face/jaw (slim), eyes (enlarge), nose (narrow), mouth (plump) — each a localized radial pinch or bulge around that region's own landmarks (exact centroids for eyes/mouth from MediaPipe's own connector loops; the nose, which has no official loop, is interpolated 42% of the way from the eye-line to the mouth, verified against real detected faces rather than guessed).
- **Filters**: a handful of CSS canvas filter presets (warm, cool, vivid, mono, soft glow).

Everything runs client-side on an uploaded or captured photo; nothing is sent anywhere.

## Development

```bash
npm install
npm run dev
npm run build
```
