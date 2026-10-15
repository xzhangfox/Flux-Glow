import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { BufferTarget, EncodedPacket, EncodedVideoPacketSource, Mp4OutputFormat, Output } from 'mediabunny'

// Live Photos: a photo that also keeps the moments around it — about 1.5 s
// before the shutter and 1.5 s after — with the photo's own edit on every
// frame of them: its effect, retouch, filter and stickers, from the first
// frame to the last, whatever the viewfinder happened to show at the time
// (an effect picked a moment before the shutter, or still loading) and
// however the photo is edited afterwards.
//
// So while the camera runs, every frame it gives is encoded as it came (the
// browser's own video encoder) into a short rolling buffer, with the face
// found in it; the shutter keeps what's buffered and records on for a
// moment more (a take). The clip is rendered from the take for the edit
// the photo has — again whenever that edit changes.
//
// Saved, it's one file where the phone can show it — a Motion Photo: the
// JPEG with the clip after it and the XMP that tells Google Photos and
// Samsung's Gallery where it is (elsewhere it's just the photo) — or, on
// iOS, where a web app can't make a Live Photo, the photo and the clip.

export const LIVE_BEFORE_MS = 1500
export const LIVE_AFTER_MS = 1500
/** A keyframe this often, so the clip can start close to 1.5 s back. */
const KEY_EVERY_MS = 500
/** The clip's long side. */
const CLIP_LONG = 1080
/** A frame with no face found takes the nearest one's this close. */
const FACE_GAP_US = 300_000
/** H.264 first (what phones' galleries and iOS Photos play), VP9 where a
 *  browser has no H.264 encoder (the clip still plays here and saves). */
const CODECS: { codec: string; container: 'avc' | 'vp9' }[] = [
  { codec: 'avc1.640028', container: 'avc' },
  { codec: 'avc1.4d0028', container: 'avc' },
  { codec: 'avc1.42e028', container: 'avc' },
  { codec: 'avc1.42e01f', container: 'avc' },
  { codec: 'vp09.00.40.08', container: 'vp9' },
]
const encoderConfig = (codec: string, w: number, h: number): VideoEncoderConfig => ({
  codec,
  width: w,
  height: h,
  bitrate: 6_000_000,
  framerate: 30,
  latencyMode: 'realtime',
  ...(codec.startsWith('avc') ? { avc: { format: 'avc' as const } } : {}),
})

const KEY = 'flux-glow-live-photo'

/** Whether this browser can record Live Photos: it has WebCodecs' video
 *  encoder and decoder, and one of the codecs above. */
export async function canRecordLive(): Promise<boolean> {
  if (typeof VideoEncoder === 'undefined' || typeof VideoDecoder === 'undefined' || typeof VideoFrame === 'undefined') return false
  for (const { codec } of CODECS) {
    try {
      if ((await VideoEncoder.isConfigSupported(encoderConfig(codec, 810, 1080))).supported) return true
    } catch {
      // (next)
    }
  }
  return false
}

/** The Live Photo switch, as last set on this device (on by default). */
export function loadLivePhoto(): boolean {
  try {
    return localStorage.getItem(KEY) !== 'off'
  } catch {
    return true
  }
}
export function saveLivePhoto(on: boolean) {
  try {
    localStorage.setItem(KEY, on ? 'on' : 'off')
  } catch {
    // (private mode: for this visit only)
  }
}

/** A finished clip: the MP4, and where in it the photo was taken (µs). */
export interface LiveClip {
  blob: Blob
  stillUs: number
}

type Chunk = { data: Uint8Array; key: boolean; t: number }
type Face = NormalizedLandmark[] | null

/** What a clip is made from: the camera's frames around the shutter as it
 *  gave them (encoded, from a keyframe on), the face found in each (keyed
 *  by the frame's timestamp, µs), and the shutter's moment (µs, the same
 *  clock). */
export interface LiveTake {
  codec: string
  container: 'avc' | 'vp9'
  decoderConfig: VideoDecoderConfig
  width: number
  height: number
  chunks: Chunk[]
  faces: Map<number, Face>
  shutterUs: number
}

const even = (n: number) => Math.max(2, Math.round(n / 2) * 2)
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

/** Encodes the camera's frames into a rolling buffer; `shutter()` makes the
 *  take. One per camera session. */
export class LiveRecorder {
  private encoder: VideoEncoder | null = null
  private codec = ''
  private container: 'avc' | 'vp9' = 'avc'
  private decoderConfig: VideoDecoderConfig | undefined
  private chunks: Chunk[] = []
  private faces = new Map<number, Face>()
  private canvas = document.createElement('canvas')
  private w = 0
  private h = 0
  private lastKey = -Infinity
  private shutterAt: number | null = null
  private done: ((take: LiveTake | null) => void) | null = null
  private closed = false

  /** Adds the camera's frame at `at` (performance.now()), as drawn for the
   *  viewfinder (framed, mirrored) but not retouched, and the face in it. */
  push(frame: HTMLCanvasElement, at: number, face: Face): void {
    if (this.closed) return
    // (a new shape — the aspect changed — starts the buffer over; a new
    // size, as the preview steps its working size, is scaled to the first;
    // and once the shutter's gone, everything just scales to fit)
    if (this.shutterAt === null && (!this.w || Math.abs(frame.width / frame.height - this.w / this.h) > 0.01)) {
      const k = Math.min(1, CLIP_LONG / Math.max(frame.width, frame.height))
      this.restart(even(frame.width * k), even(frame.height * k))
    }
    const enc = this.encoder
    if (enc?.state === 'configured' && enc.encodeQueueSize <= 2) {
      this.canvas.getContext('2d')!.drawImage(frame, 0, 0, this.w, this.h)
      const t = Math.round(at * 1000)
      const vf = new VideoFrame(this.canvas, { timestamp: t })
      const key = at - this.lastKey >= KEY_EVERY_MS
      if (key) this.lastKey = at
      try {
        enc.encode(vf, { keyFrame: key })
        this.faces.set(t, face)
      } catch {
        // (an encoder that's gone: no clip this time)
      }
      vf.close()
    }
    if (this.shutterAt === null) this.trim(at)
    else if (at >= this.shutterAt + LIVE_AFTER_MS) void this.finish()
  }

  /** The shutter, at `at`: resolves with the take once the moments after
   *  it are in (or, if the camera stops first, with what there is) — null
   *  if there's nothing to make a clip from. */
  shutter(at: number): Promise<LiveTake | null> {
    if (this.closed || !this.encoder) return Promise.resolve(null)
    this.shutterAt = at
    // (and if the camera's frames stop coming, with what there is)
    setTimeout(() => void this.finish(), LIVE_AFTER_MS + 1000)
    return new Promise((resolve) => (this.done = resolve))
  }

  /** The camera's stopped: a take waiting for its after-moments ends here. */
  stop(): void {
    if (this.shutterAt !== null) void this.finish()
    else this.close()
  }

  private restart(w: number, h: number) {
    this.close()
    this.closed = false
    this.w = w
    this.h = h
    this.canvas.width = w
    this.canvas.height = h
    this.chunks = []
    this.faces.clear()
    this.decoderConfig = undefined
    this.lastKey = -Infinity
    void this.configure(w, h)
  }

  private async configure(w: number, h: number) {
    for (const { codec, container } of CODECS) {
      const config = encoderConfig(codec, w, h)
      try {
        if (!(await VideoEncoder.isConfigSupported(config)).supported) continue
      } catch {
        continue
      }
      if (this.closed || w !== this.w || h !== this.h) return
      const encoder = new VideoEncoder({
        output: (chunk, meta) => {
          if (meta?.decoderConfig) this.decoderConfig = meta.decoderConfig
          const data = new Uint8Array(chunk.byteLength)
          chunk.copyTo(data)
          this.chunks.push({ data, key: chunk.type === 'key', t: chunk.timestamp })
        },
        error: () => this.close(),
      })
      encoder.configure(config)
      this.encoder = encoder
      this.codec = codec
      this.container = container
      return
    }
    this.close()
  }

  /** Drops what's older than the clip could start with: everything before
   *  the last keyframe from at least LIVE_BEFORE_MS ago. */
  private trim(at: number) {
    const from = (at - LIVE_BEFORE_MS) * 1000
    let i = -1
    for (let j = 0; j < this.chunks.length && this.chunks[j].t <= from; j++) if (this.chunks[j].key) i = j
    if (i > 0) {
      this.chunks.splice(0, i)
      const t0 = this.chunks[0].t
      for (const t of this.faces.keys()) if (t < t0) this.faces.delete(t)
    }
  }

  private async finish() {
    const done = this.done
    const shutterAt = this.shutterAt
    if (!done || shutterAt === null) return
    this.done = null
    let take: LiveTake | null = null
    try {
      await this.encoder?.flush()
      take = this.take(shutterAt)
    } catch {
      take = null
    }
    this.close()
    done(take)
  }

  /** The frames from the last keyframe at least LIVE_BEFORE_MS before the
   *  shutter to LIVE_AFTER_MS after it. */
  private take(shutterAt: number): LiveTake | null {
    const from = (shutterAt - LIVE_BEFORE_MS) * 1000
    const to = (shutterAt + LIVE_AFTER_MS) * 1000
    let start = this.chunks.findIndex((c) => c.key)
    for (let j = start; j >= 0 && j < this.chunks.length && this.chunks[j].t <= from; j++) if (this.chunks[j].key) start = j
    if (start < 0 || !this.decoderConfig) return null
    const chunks = this.chunks.slice(start).filter((c) => c.t <= to)
    if (chunks.length < 2) return null
    const faces = new Map<number, Face>()
    for (const c of chunks) faces.set(c.t, this.faces.get(c.t) ?? null)
    return { codec: this.codec, container: this.container, decoderConfig: this.decoderConfig, width: this.w, height: this.h, chunks, faces, shutterUs: Math.round(shutterAt * 1000) }
  }

  private close() {
    this.closed = true
    try {
      if (this.encoder && this.encoder.state !== 'closed') this.encoder.close()
    } catch {
      // (already gone)
    }
    this.encoder = null
  }
}

/** The face for the frame at `t`: its own, or — a frame the tracker lost
 *  the face in for a moment — the nearest frame's within FACE_GAP_US. */
function faceAt(faces: Map<number, Face>, times: number[], t: number): Face {
  const own = faces.get(t)
  if (own) return own
  let best: Face = null
  let gap = FACE_GAP_US
  for (const u of times) {
    const f = faces.get(u)
    if (f && Math.abs(u - t) <= gap) {
      gap = Math.abs(u - t)
      best = f
    }
  }
  return best
}

/** Renders `take` into a clip: each frame decoded, painted by `paint` (the
 *  photo's edit: given the frame, its face and its moment in seconds, it
 *  returns the finished frame) and encoded again. Null if it can't be made
 *  or `cancelled()` turns true on the way. */
export async function renderLive(take: LiveTake, paint: (frame: HTMLCanvasElement, face: Face, t: number) => Promise<CanvasImageSource>, cancelled: () => boolean): Promise<LiveClip | null> {
  const { width: w, height: h } = take
  const times = [...take.faces.keys()]
  const base = document.createElement('canvas')
  base.width = w
  base.height = h
  const decoded: VideoFrame[] = []
  const out: Chunk[] = []
  let outConfig: VideoDecoderConfig | undefined
  let failed = false
  let wake: (() => void) | null = null
  const decoder = new VideoDecoder({
    output: (f) => {
      decoded.push(f)
      wake?.()
    },
    error: () => {
      failed = true
      wake?.()
    },
  })
  const encoder = new VideoEncoder({
    output: (chunk, meta) => {
      if (meta?.decoderConfig) outConfig = meta.decoderConfig
      const data = new Uint8Array(chunk.byteLength)
      chunk.copyTo(data)
      out.push({ data, key: chunk.type === 'key', t: chunk.timestamp })
    },
    error: () => (failed = true),
  })
  try {
    decoder.configure(take.decoderConfig)
    encoder.configure(encoderConfig(take.codec, w, h))
    let next = 0
    let flushed = false
    let drained = false
    let lastKey = -Infinity
    for (;;) {
      if (failed || cancelled()) return null
      // (a few frames ahead at most: a decoder's frames are its own until
      // they're closed, and it stalls when they're all out)
      while (next < take.chunks.length && decoded.length + decoder.decodeQueueSize < 3) {
        const c = take.chunks[next++]
        decoder.decode(new EncodedVideoChunk({ type: c.key ? 'key' : 'delta', timestamp: c.t, data: c.data }))
      }
      if (next === take.chunks.length && !flushed) {
        flushed = true
        void decoder.flush().then(
          () => {
            drained = true
            wake?.()
          },
          () => {
            failed = true
            wake?.()
          },
        )
      }
      const frame = decoded.shift()
      if (!frame) {
        if (drained) break
        // (until the decoder has the next one — or a moment, for a decoder
        // that holds a frame back until it's given the next chunk)
        await Promise.race([new Promise<void>((r) => (wake = r)), sleep(30)])
        wake = null
        continue
      }
      const t = frame.timestamp
      base.getContext('2d')!.drawImage(frame, 0, 0, w, h)
      frame.close()
      const painted = await paint(base, faceAt(take.faces, times, t), t / 1e6)
      if (failed || cancelled()) return null
      while (encoder.encodeQueueSize > 3) await sleep(4)
      const vf = new VideoFrame(painted, { timestamp: t })
      const key = t - lastKey >= 1_000_000
      if (key) lastKey = t
      encoder.encode(vf, { keyFrame: key })
      vf.close()
    }
    await encoder.flush()
    if (failed || cancelled() || !outConfig) return null
    return await mux(out, outConfig, take.container, take.shutterUs)
  } catch {
    return null
  } finally {
    decoded.forEach((f) => f.close())
    for (const c of [decoder, encoder])
      try {
        if (c.state !== 'closed') c.close()
      } catch {
        // (already gone)
      }
  }
}

async function mux(chunks: Chunk[], decoderConfig: VideoDecoderConfig, container: 'avc' | 'vp9', shutterUs: number): Promise<LiveClip | null> {
  if (chunks.length < 2) return null
  const target = new BufferTarget()
  const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target })
  const source = new EncodedVideoPacketSource(container)
  output.addVideoTrack(source)
  await output.start()
  const t0 = chunks[0].t
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i]
    const next = i + 1 < chunks.length ? chunks[i + 1].t : c.t + 33_333
    const packet = new EncodedPacket(c.data, c.key ? 'key' : 'delta', (c.t - t0) / 1e6, Math.max(1, next - c.t) / 1e6)
    await source.add(packet, i === 0 ? { decoderConfig } : undefined)
  }
  await output.finalize()
  if (!target.buffer) return null
  return { blob: new Blob([target.buffer], { type: 'video/mp4' }), stillUs: Math.max(0, shutterUs - t0) }
}

/** A Motion Photo: `jpeg` with `clip` after it, and the XMP that says so
 *  (Google's Motion Photo format, with its older MicroVideo tags for
 *  phones that only know those). */
export async function motionPhoto(jpeg: Blob, clip: LiveClip): Promise<Blob> {
  const j = new Uint8Array(await jpeg.arrayBuffer())
  const len = clip.blob.size
  const xmp =
    `<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Flux Glow"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">` +
    `<rdf:Description rdf:about="" xmlns:GCamera="http://ns.google.com/photos/1.0/camera/" xmlns:Container="http://ns.google.com/photos/1.0/container/" xmlns:Item="http://ns.google.com/photos/1.0/container/item/"` +
    ` GCamera:MotionPhoto="1" GCamera:MotionPhotoVersion="1" GCamera:MotionPhotoPresentationTimestampUs="${clip.stillUs}"` +
    ` GCamera:MicroVideo="1" GCamera:MicroVideoVersion="1" GCamera:MicroVideoOffset="${len}" GCamera:MicroVideoPresentationTimestampUs="${clip.stillUs}">` +
    `<Container:Directory><rdf:Seq>` +
    `<rdf:li rdf:parseType="Resource"><Container:Item Item:Mime="image/jpeg" Item:Semantic="Primary" Item:Length="0" Item:Padding="0"/></rdf:li>` +
    `<rdf:li rdf:parseType="Resource"><Container:Item Item:Mime="video/mp4" Item:Semantic="MotionPhoto" Item:Length="${len}" Item:Padding="0"/></rdf:li>` +
    `</rdf:Seq></Container:Directory></rdf:Description></rdf:RDF></x:xmpmeta>`
  const payload = new TextEncoder().encode(`http://ns.adobe.com/xap/1.0/\0${xmp}`)
  const segment = new Uint8Array(4 + payload.length)
  segment.set([0xff, 0xe1, ((payload.length + 2) >> 8) & 0xff, (payload.length + 2) & 0xff])
  segment.set(payload, 4)
  // (after the start-of-image marker and the JFIF header, if there is one)
  let at = 2
  if (j[2] === 0xff && j[3] === 0xe0) at = 4 + ((j[4] << 8) | j[5])
  return new Blob([j.subarray(0, at), segment, j.subarray(at), clip.blob], { type: 'image/jpeg' })
}
