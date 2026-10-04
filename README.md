# Flux Glow

A beauty camera that runs entirely in the browser: retouching, face shaping, and color grading. No uploads, no account, no VIP paywall.

Part of the [Flux](https://xzhangfox.github.io) family of apps.

## Features

- **Camera**: front/back switch, real optical zoom where the browser exposes it (tap the zoom pill to cycle stops, hold it for a fine-control ruler), 3:4 / 1:1 / full-screen framing, a rule-of-thirds grid, and a 3s/10s self-timer. The viewfinder is exactly what gets saved — no hidden crop.
- **Beauty**: skin smoothing, whitening, blemish removal, wrinkle softening, smile-fold softening, and 3D fill light.
- **Shape**: jaw, temple, cheekbone, eyes, brow height, nose width, nose bridge, and mouth (size, upper lip, lower lip, corners), each bidirectional.
- **Filter**: ten graded looks (Natural, Peach, Cream, Fresh, Warm, Cool, Soft Glow, Film, Vivid, Mono), previewed as thumbnails on your own face, with an intensity slider.
- **Review**: hold the compare button to see the original, then Save (via the share sheet on iOS, where that's the only route into Photos) or Share. Every panel has a Reset, and any single slider reverts with one tap.

Everything runs client-side; nothing is uploaded.

## How it works

- **Face detection**: [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (478-point face mesh), self-hosted under `public/mediapipe/` rather than fetched from a CDN. Live preview uses the video tracker; a captured or uploaded still is re-detected in image mode on that exact frame.
- **Skin mask**: the face oval minus eyes, eyebrows, and lips, so skin effects only ever touch skin.
- **Smoothing**: frequency separation — fine texture (pores) is kept and only the blotchy tone layer is smoothed, with a guided filter so real edges (brows, lids, lips) stay sharp.
- **Whitening**: CIELAB tone mapping baked into a 3D LUT, applied to the low-frequency layer only so texture survives.
- **Fill light**: per-vertex surface normals from the 3D face mesh, lit in a WebGL shader (Lambert + Blinn-Phong, with a warm terminator tint approximating subsurface scattering).
- **Reshape**: a GPU triangulated-mesh warp over MediaPipe's canonical face topology (`faceTriangulation.ts`, `meshWarp.ts`); nose width is a separate isolated radial warp.
- **Filters**: a grading model (exposure curve, white balance, split-toning, fade, contrast, saturation) blended by intensity — not CSS `filter()` presets.
- **Live performance**: the live preview fuses the CPU tone effects into a single pass and swaps the guided filter for a cheaper blur; the full-quality pass runs once a photo is captured.

## Development

```bash
npm install
npm run dev
npm run build
```
