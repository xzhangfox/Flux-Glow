import React from 'react';
import { THEMES, useTheme, type ThemeName } from '../context/ThemeContext';

// Inline SVG rather than Material Symbols ligatures: if the icon web font
// is slow or blocked, ligature icons render as their raw names
// ("dark_mode", "forest"…) instead of failing quietly.
const ICONS: Record<ThemeName, React.ReactNode> = {
  dark: <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />,
  nature: (
    <>
      <path d="M12 3l-5.5 8h3L6 16.5h12L14.5 11h3z" />
      <path d="M12 16.5V21" />
    </>
  ),
  attention: (
    <>
      <path d="M12 3l1.6 4.6L18 9l-4.4 1.4L12 15l-1.6-4.6L6 9l4.4-1.4z" />
      <path d="M18.5 15l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z" />
    </>
  ),
  ocean: (
    <>
      <path d="M3 9c2 0 2-1.8 4.5-1.8S9.5 9 12 9s2-1.8 4.5-1.8S19 9 21 9" />
      <path d="M3 13.5c2 0 2-1.8 4.5-1.8s2 1.8 4.5 1.8 2-1.8 4.5-1.8 2.5 1.8 4.5 1.8" />
      <path d="M3 18c2 0 2-1.8 4.5-1.8S9.5 18 12 18s2-1.8 4.5-1.8S19 18 21 18" />
    </>
  ),
};

// "App Theme" — the same four-way picker as Flux's Profile screen.
export const ThemePicker: React.FC = () => {
  const { theme, setTheme } = useTheme();
  return (
    <div>
      <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-secondary)] mb-4 px-1">App Theme</h3>
      <div className="grid grid-cols-4 gap-2 sm:gap-3">
        {THEMES.map((t) => {
          const active = theme === t.id;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => setTheme(t.id)}
              aria-pressed={active}
              aria-label={t.label}
              className={`relative px-1 py-3 sm:p-4 rounded-2xl border backdrop-blur-xl transition-all flex flex-col items-center gap-2 ${
                active
                  ? 'bg-[rgba(var(--primary-rgb),0.1)] border-[var(--primary)] text-[var(--primary)]'
                  : 'bg-[var(--surface-dark)] border-transparent text-[var(--text-secondary)] hover:border-white/10'
              }`}
            >
              <svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
                {ICONS[t.id]}
              </svg>
              <span className="text-[9px] sm:text-[10px] font-bold uppercase tracking-wider text-center leading-tight">{t.label}</span>
              <span className="absolute top-2 right-2 w-1.5 h-1.5 rounded-full" style={{ background: t.swatch, boxShadow: `0 0 6px ${t.swatch}` }} />
            </button>
          );
        })}
      </div>
    </div>
  );
};
