// "Try Our Other Apps" — the same family row every Flux app carries, in
// the canonical order (Glow itself left out), each with its gold mark.
const APPS = [
  { name: 'Portfolio Website', url: 'https://xzhangfox.github.io', logo: '/logos/portfolio-website.svg' },
  { name: 'Flux Nutrition', url: 'https://flux-fox-1121.vercel.app/', logo: '/logos/flux-nutrition.svg' },
  { name: 'Flux Path', url: 'https://parallax-nine-taupe.vercel.app/', logo: '/logos/flux-path.svg' },
  { name: 'Flux Career', url: 'https://flux-career-cyan.vercel.app/', logo: '/logos/flux-career.svg' },
  { name: 'Flux Finance', url: 'https://flux-finance-ivory.vercel.app/', logo: '/logos/flux-finance.svg' },
  { name: 'Flux AI Bubble Monitor', url: 'https://ai-bubble-monitor-delta.vercel.app/', logo: '/logos/ai-bubble-monitor.svg' },
  { name: 'Flux Scribe', url: 'https://flux-scribe-otzr.vercel.app/', logo: '/logos/flux-scribe.svg' },
]

export default function OtherApps() {
  return (
    <div>
      <h3 className="mb-3 px-1 text-[11px] font-bold uppercase tracking-[0.2em] text-muted">Try Our Other Apps</h3>
      <div className="mask-gradient-sides -mx-1 flex gap-3 overflow-x-auto px-1 pb-1 no-scrollbar">
        {APPS.map((a) => (
          <a
            key={a.name}
            href={a.url}
            target="_blank"
            rel="noopener noreferrer"
            title={a.name}
            className="glass group flex w-[86px] shrink-0 flex-col items-center gap-2 rounded-2xl px-2 py-3 transition hover:border-primary/50 active:scale-95"
          >
            <img src={a.logo} alt="" className="h-9 w-9 transition group-hover:-translate-y-0.5" />
            <span className="text-center text-[9px] font-bold uppercase leading-tight tracking-wider text-muted group-hover:text-primary">{a.name.replace('Flux ', '')}</span>
          </a>
        ))}
      </div>
    </div>
  )
}
