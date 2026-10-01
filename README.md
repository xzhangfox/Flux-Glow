# Flux Glow

Skin smoothing, blemish reduction, and filters — entirely in the browser. No uploads, no account, no VIP paywall.

Part of the [Flux](https://xzhangfox.github.io) family of apps.

## How it works

- **Face detection**: [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (478-point face mesh), self-hosted (WASM runtime + model are committed under `public/mediapipe/`, not fetched from a third-party CDN at runtime).
- **Skin mask**: the face oval minus eyes, eyebrows, and lips, built from the detected landmarks, so smoothing only ever touches actual skin.
- **Smoothing**: real frequency separation (not a flat blur) — a lightly-blurred copy of the image gives the fine texture layer (pores, fine lines), a second, heavier blur of that same layer gives the tone/blemish layer, and the two recombine with only the tone layer smoothed. That's what keeps retouched skin looking like skin instead of plastic, and it reduces blemishes as a side effect of smoothing blotchy tone, without a separate spot-removal tool.
- **Reshape**: face/jaw (slim) uses [Moving Least Squares](https://www.cs.jhu.edu/~misha/Fall15/Papers/Schaefer06.pdf) similarity deformation (Schaefer, McPhail & Warren, SIGGRAPH 2006) — the jaw-line landmarks below the eye-line are moving control points pushed toward the face's centerline, pinned by a dense ring of zero-displacement anchors tightly around just that region (loose or sparse anchoring lets the field sag into the background between anchors, which is visible as wavy straight lines behind the head — the ring has to be dense *relative to its own box*, not just dense in absolute terms). Eyes (enlarge), mouth (plump), and nose (narrow) all use a simpler elliptical radial zoom instead of MLS — sized to each feature's own aspect ratio so it doesn't reach into a neighboring feature (an earlier version scaled eyes/mouth via MLS loop points pinned by a nearby anchor ring, but with no intermediate control points to guide the gap between "move" and "pin", MLS's closed-form solution folded that gap into a solid-color smear at just moderate strength — a radial zoom is monotonic by construction and can't fold).
- **Filters**: a handful of CSS canvas filter presets (warm, cool, vivid, mono, soft glow).

Everything runs client-side on an uploaded or captured photo; nothing is sent anywhere.

## Development

```bash
npm install
npm run dev
npm run build
```
