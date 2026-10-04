import { useEffect, useRef, useState } from 'react'
import { ART, EMOJI, TEXT_STYLES, textCanvas } from '../lib/stickers'
import { IconImage, IconPlus } from './icons'

type Tab = 'art' | 'emoji' | 'text' | 'photo'

export type StickerRequest = { kind: 'art'; svg: string } | { kind: 'emoji'; char: string } | { kind: 'text'; text: string; style: string } | { kind: 'photo'; file: File }

const TABS: { id: Tab; label: string }[] = [
  { id: 'art', label: 'Stickers' },
  { id: 'emoji', label: 'Emoji' },
  { id: 'text', label: 'Text' },
  { id: 'photo', label: 'Photo' },
]

const artUrl = (svg: string) => 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg)

// Two-row horizontally scrolling grids keep the panel short enough to leave
// the photo visible while picking. Tapping adds the sticker to the middle
// of the photo, selected, ready to drag/pinch into place.
export default function StickerPanel({ count, onAdd, onClearAll }: { count: number; onAdd: (req: StickerRequest) => void; onClearAll: () => void }) {
  const [tab, setTab] = useState<Tab>('art')
  const [text, setText] = useState('')
  const [style, setStyle] = useState(TEXT_STYLES[0].id)
  const fileRef = useRef<HTMLInputElement>(null)

  return (
    <div>
      <div className="flex items-center justify-between h-7 mb-3">
        <span className="text-[13px] font-semibold text-white">Stickers</span>
        <button onClick={onClearAll} disabled={!count} className="text-[11px] font-medium text-white/70 hover:text-white disabled:opacity-30 transition">
          Clear all
        </button>
      </div>
      <div className="flex gap-1.5 mb-3" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`h-7 px-3 rounded-full text-[11.5px] font-semibold transition ${tab === t.id ? 'bg-primary text-black' : 'bg-white/[0.07] text-white/75'}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'art' && (
        <div className="grid grid-rows-2 grid-flow-col auto-cols-[3.25rem] gap-2 overflow-x-auto no-scrollbar -mx-1 px-1 pb-1">
          {ART.map((a) => (
            <button key={a.id} onClick={() => onAdd({ kind: 'art', svg: a.svg })} aria-label={`Add ${a.label} sticker`} className="w-[3.25rem] h-[3.25rem] rounded-xl bg-white/[0.06] border border-white/10 flex items-center justify-center active:scale-95 transition">
              <img src={artUrl(a.svg)} alt="" className="w-9 h-9 object-contain" draggable={false} />
            </button>
          ))}
        </div>
      )}

      {tab === 'emoji' && (
        <div className="grid grid-rows-2 grid-flow-col auto-cols-[3rem] gap-2 overflow-x-auto no-scrollbar -mx-1 px-1 pb-1">
          {EMOJI.map((e) => (
            <button key={e} onClick={() => onAdd({ kind: 'emoji', char: e })} aria-label={`Add ${e}`} className="w-12 h-12 rounded-xl bg-white/[0.06] border border-white/10 text-[26px] leading-none flex items-center justify-center active:scale-95 transition">
              {e}
            </button>
          ))}
        </div>
      )}

      {tab === 'text' && (
        <div>
          <div className="flex gap-2 mb-3">
            <input
              value={text}
              onChange={(e) => setText(e.target.value.slice(0, 40))}
              placeholder="Type something…"
              aria-label="Sticker text"
              className="flex-1 min-w-0 h-10 px-3 rounded-xl bg-white/[0.08] border border-white/15 text-sm text-white placeholder:text-white/40 outline-none focus:border-primary/70"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && text.trim()) onAdd({ kind: 'text', text: text.trim(), style })
              }}
            />
            <button
              onClick={() => text.trim() && onAdd({ kind: 'text', text: text.trim(), style })}
              disabled={!text.trim()}
              aria-label="Add text"
              className="h-10 px-3.5 rounded-xl bg-primary text-black text-sm font-semibold flex items-center gap-1 disabled:opacity-40 transition active:scale-95"
            >
              <IconPlus className="w-4 h-4" />
              Add
            </button>
          </div>
          <div className="flex gap-2 overflow-x-auto no-scrollbar -mx-1 px-1 pb-1">
            {TEXT_STYLES.map((t) => (
              <StylePreview key={t.id} id={t.id} label={t.label} sample={text.trim() || 'Aa'} selected={style === t.id} onClick={() => setStyle(t.id)} />
            ))}
          </div>
        </div>
      )}

      {tab === 'photo' && (
        <button onClick={() => fileRef.current?.click()} className="w-full h-[6.75rem] rounded-xl border border-dashed border-white/25 bg-white/[0.04] flex flex-col items-center justify-center gap-2 text-white/80 active:scale-[0.99] transition">
          <IconImage className="w-7 h-7 text-primary" />
          <span className="text-[12.5px] font-medium">Add a photo from your album</span>
          <span className="text-[10.5px] text-white/45">PNGs with transparency are placed as cut-outs</span>
        </button>
      )}
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onAdd({ kind: 'photo', file })
          e.target.value = ''
        }}
      />
    </div>
  )
}

function StylePreview({ id, label, sample, selected, onClick }: { id: string; label: string; sample: string; selected: boolean; onClick: () => void }) {
  const [url, setUrl] = useState('')
  useEffect(() => {
    const t = setTimeout(() => setUrl(textCanvas(sample.slice(0, 8), id).toDataURL()), 120)
    return () => clearTimeout(t)
  }, [id, sample])
  return (
    <button onClick={onClick} aria-label={`${label} text style`} aria-pressed={selected} className="flex flex-col items-center gap-1.5 flex-shrink-0">
      <span className={`w-16 h-11 rounded-xl bg-white/[0.06] flex items-center justify-center overflow-hidden transition ${selected ? 'ring-2 ring-primary ring-offset-2 ring-offset-black/60' : 'ring-1 ring-white/10'}`}>
        {url && <img src={url} alt="" className="max-w-[90%] max-h-[80%] object-contain" draggable={false} />}
      </span>
      <span className={`text-[10.5px] font-medium leading-none ${selected ? 'text-primary' : 'text-white/75'}`}>{label}</span>
    </button>
  )
}
