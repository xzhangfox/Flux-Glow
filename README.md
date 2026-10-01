# Flux Glow

Skin smoothing, blemish reduction, and filters — entirely in the browser. No uploads, no account, no VIP paywall.

Part of the [Flux](https://xzhangfox.github.io) family of apps.

## How it works

- **Face detection**: [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (478-point face mesh), self-hosted (WASM runtime + model are committed under `public/mediapipe/`, not fetched from a third-party CDN at runtime).
- **Skin mask**: the face oval minus eyes, eyebrows, and lips, built from the detected landmarks, so smoothing only ever touches actual skin.
- **Smoothing**: real frequency separation (not a flat blur) — a lightly-blurred copy of the image gives the fine texture layer (pores, fine lines), a second, heavier blur of that same layer gives the tone/blemish layer, and the two recombine with only the tone layer smoothed. That's what keeps retouched skin looking like skin instead of plastic, and it reduces blemishes as a side effect of smoothing blotchy tone, without a separate spot-removal tool.
- **Reshape**: face/jaw (slim), eyes (enlarge), and mouth (plump) are driven by a single [Moving Least Squares](https://www.cs.jhu.edu/~misha/Fall15/Papers/Schaefer06.pdf) similarity deformation (Schaefer, McPhail & Warren, SIGGRAPH 2006) — every relevant landmark is a control point in one combined field, moving when its slider is active, pinned at zero displacement when it isn't, so adjacent features blend into each other instead of warping independently. Nose (narrow) is a simpler isolated radial pinch, since it has no official MediaPipe landmark loop to build verified control points from — its center is interpolated 42% of the way from the eye-line to the mouth.
- **Filters**: a handful of CSS canvas filter presets (warm, cool, vivid, mono, soft glow).

Everything runs client-side on an uploaded or captured photo; nothing is sent anywhere.

## Development

```bash
npm install
npm run dev
npm run build
```
