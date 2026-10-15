import { BufferTarget, EncodedPacket, EncodedVideoPacketSource, Mp4OutputFormat, Output } from 'mediabunny'

// Live Photos: a photo that also keeps the moments around it — about 1.5 s
// before the shutter and 1.5 s after, as the viewfinder showed them
// (retouched, with the effect on). While the camera runs, every frame shown
// is encoded (the browser's own video encoder) into a short rolling buffer;
// the shutter keeps what's buffered and records on for a moment more.
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
 *  encoder, and one of the codecs above. */
export async function canRecordLive(): Promise<boolean> {
  if (typeof VideoEncoder === 'undefined' || typeof VideoFrame === 'undefined') return false
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
const even = (n: number) => Math.max(2, Math.round(n / 2) * 2)

/** Encodes the frames shown into a rolling buffer; `shutter()` makes the
 *  clip. One per camera session. */
export class LiveRecorder {
  private encoder: VideoEncoder | null = null
  private container: 'avc' | 'vp9' = 'avc'
  private decoderConfig: VideoDecoderConfig | undefined
  private chunks: Chunk[] = []
  private canvas = document.createElement('canvas')
  private w = 0
  private h = 0
  private lastKey = -Infinity
  private shutterAt: number | null = null
  private done: ((clip: LiveClip | null) => void) | null = null
  private closed = false

  /** Adds the frame shown at `at` (performance.now()). */
  push(frame: HTMLCanvasElement, at: number): void {
    if (this.closed) return
    const k = Math.min(1, CLIP_LONG / Math.max(frame.width, frame.height))
    const w = even(frame.width * k)
    const h = even(frame.height * k)
    // (a new size or shape — the aspect changed — starts the buffer over;
    // not once the shutter's gone, when the frames just scale to fit)
    if ((w !== this.w || h !== this.h) && this.shutterAt === null) this.restart(w, h)
    const enc = this.encoder
    if (enc?.state === 'configured' && enc.encodeQueueSize <= 2) {
      this.canvas.getContext('2d')!.drawImage(frame, 0, 0, this.w, this.h)
      const vf = new VideoFrame(this.canvas, { timestamp: Math.round(at * 1000) })
      const key = at - this.lastKey >= KEY_EVERY_MS
      if (key) this.lastKey = at
      try {
        enc.encode(vf, { keyFrame: key })
      } catch {
        // (an encoder that's gone: no clip this time)
      }
      vf.close()
    }
    if (this.shutterAt === null) this.trim(at)
    else if (at >= this.shutterAt + LIVE_AFTER_MS) void this.finish()
  }

  /** The shutter, at `at`: resolves with the clip once the moments after
   *  it are in (or, if the camera stops first, with what there is) — null
   *  if there's nothing to make one from. */
  shutter(at: number): Promise<LiveClip | null> {
    if (this.closed || !this.encoder) return Promise.resolve(null)
    this.shutterAt = at
    return new Promise((resolve) => (this.done = resolve))
  }

  /** The camera's stopped: a clip waiting for its after-moments ends here. */
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
    if (i > 0) this.chunks.splice(0, i)
  }

  private async finish() {
    const done = this.done
    const shutterAt = this.shutterAt
    if (!done || shutterAt === null) return
    this.done = null
    let clip: LiveClip | null = null
    try {
      await this.encoder?.flush()
      clip = await this.mux(shutterAt)
    } catch {
      clip = null
    }
    this.close()
    done(clip)
  }

  private async mux(shutterAt: number): Promise<LiveClip | null> {
    const from = (shutterAt - LIVE_BEFORE_MS) * 1000
    const to = (shutterAt + LIVE_AFTER_MS) * 1000
    let start = this.chunks.findIndex((c) => c.key)
    for (let j = start; j >= 0 && j < this.chunks.length && this.chunks[j].t <= from; j++) if (this.chunks[j].key) start = j
    if (start < 0 || !this.decoderConfig) return null
    const sel = this.chunks.slice(start).filter((c) => c.t <= to)
    if (sel.length < 2) return null
    const target = new BufferTarget()
    const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target })
    const source = new EncodedVideoPacketSource(this.container)
    output.addVideoTrack(source)
    await output.start()
    const t0 = sel[0].t
    for (let i = 0; i < sel.length; i++) {
      const c = sel[i]
      const next = i + 1 < sel.length ? sel[i + 1].t : c.t + 33_333
      const packet = new EncodedPacket(c.data, c.key ? 'key' : 'delta', (c.t - t0) / 1e6, Math.max(1, next - c.t) / 1e6)
      await source.add(packet, i === 0 ? { decoderConfig: this.decoderConfig } : undefined)
    }
    await output.finalize()
    if (!target.buffer) return null
    return { blob: new Blob([target.buffer], { type: 'video/mp4' }), stillUs: Math.max(0, Math.round(shutterAt * 1000 - t0)) }
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
