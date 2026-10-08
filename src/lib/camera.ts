// The camera's two modes. For photos the stream runs at the camera's
// largest 4:3 size (up to STILL_MAX), so a photo is simply the frame on
// screen when the shutter fires — that very moment, at the camera's own
// resolution. (Taking a separate still instead — a real photo, or a frame
// after switching up — lands a moment later than the shutter, and the
// picture saved wasn't the one the viewfinder froze on.) Recording switches
// to 1080p at 60 fps, which is what the phone's video encoder wants.

/** A photo's long side, at most: about what phone cameras save by default
 *  (12 MP) — and within every phone's GPU texture limit, which the
 *  retouching passes need. */
export const STILL_MAX = 4096

export type CameraMode = 'photo' | 'video'

const MODES: Record<CameraMode, MediaTrackConstraints> = {
  // 4:3, both ways: phone sensors are natively 4:3, and their 16:9 modes
  // cut a quarter of the width off a portrait selfie (with a width alone,
  // a wide 16:9 mode can win). Ideals only — a camera without the size
  // gives its nearest.
  photo: { width: { ideal: STILL_MAX }, height: { ideal: (STILL_MAX * 3) / 4 }, aspectRatio: { ideal: 4 / 3 }, frameRate: { ideal: 30 } },
  video: { width: { ideal: 1920 }, height: { ideal: 1440 }, aspectRatio: { ideal: 4 / 3 }, frameRate: { ideal: 60 } },
}
/** The live preview's copy of a photo-sized camera (see previewCopy). */
const PREVIEW: MediaTrackConstraints = { width: { ideal: 1920 }, height: { ideal: 1440 }, aspectRatio: { ideal: 4 / 3 }, frameRate: { ideal: 30 } }

/** getUserMedia's video constraints for `mode`. */
export const cameraConstraints = (mode: CameraMode, facingMode: string): MediaTrackConstraints => ({ facingMode, ...MODES[mode] })

const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))

const isFourThree = (video: HTMLVideoElement) => {
  const r = Math.max(video.videoWidth, video.videoHeight) / Math.max(1, Math.min(video.videoWidth, video.videoHeight))
  return Math.abs(r - 4 / 3) < 0.04
}

async function apply(track: MediaStreamTrack, video: HTMLVideoElement, c: MediaTrackConstraints, zoom: number | null) {
  const before = track.getSettings()
  const w0 = video.videoWidth
  const h0 = video.videoHeight
  try {
    await track.applyConstraints({ ...c, ...(zoom !== null ? { advanced: [{ zoom } as unknown as MediaTrackConstraintSet] } : {}) })
  } catch {
    return
  }
  const after = track.getSettings()
  if (after.width === before.width && after.height === before.height) return
  const t0 = performance.now()
  while (video.videoWidth === w0 && video.videoHeight === h0 && performance.now() - t0 < 1500) await nextFrame()
}

/** After opening the camera in photo mode: a camera whose largest size
 *  isn't 4:3 (only a 16:9 one that big) would show less of the scene than
 *  its 4:3 modes — those win, at the size they come in. */
export async function settlePhotoMode(track: MediaStreamTrack, video: HTMLVideoElement, zoom: number | null): Promise<void> {
  if (!video.videoWidth || isFourThree(video)) return
  await apply(track, video, { ...MODES.video, frameRate: { ideal: 30 } }, zoom)
}

/** Switches a running camera to `mode` (keeping its hardware `zoom`, if
 *  any) and resolves once frames of the new size reach `video` — or at
 *  once, if the camera has only the one size. Never throws: a camera that
 *  won't switch just carries on as it was. */
export async function switchCamera(track: MediaStreamTrack, video: HTMLVideoElement, mode: CameraMode, zoom: number | null): Promise<void> {
  await apply(track, video, MODES[mode], zoom)
  if (mode === 'photo') await settlePhotoMode(track, video, zoom)
}

/** The live preview's own copy of a photo-sized camera, at 1440p: the
 *  browser scales each frame down itself, off the page's thread, so every
 *  live frame the page handles is a fraction of the size (handling the
 *  full 12 MP frames cost the preview most of its frame rate) — while the
 *  camera, and the track the shutter draws from, stay at full size. Null
 *  when the camera isn't photo-sized, or when this browser can't size a
 *  copy apart from its camera (the camera itself would drop with it: then
 *  it's put back). `fullVideo` plays the full-size track. */
export async function previewCopy(full: MediaStreamTrack, fullVideo: HTMLVideoElement, zoom: number | null): Promise<MediaStreamTrack | null> {
  const w0 = fullVideo.videoWidth
  const h0 = fullVideo.videoHeight
  if (Math.max(w0, h0) <= 2400) return null
  const before = full.getSettings()
  const copy = full.clone()
  try {
    await copy.applyConstraints(PREVIEW)
  } catch {
    copy.stop()
    return null
  }
  // (a few frames for a camera that would follow the copy down)
  for (let i = 0; i < 6; i++) await nextFrame()
  const s = copy.getSettings()
  const now = full.getSettings()
  const kept = fullVideo.videoWidth === w0 && fullVideo.videoHeight === h0 && now.width === before.width && now.height === before.height
  if (kept && !!s.width && !!s.height && s.width * s.height < w0 * h0 * 0.6) return copy
  copy.stop()
  if (!kept) await apply(full, fullVideo, MODES.photo, zoom)
  return null
}
