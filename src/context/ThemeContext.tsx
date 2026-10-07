import React, { createContext, useContext, useMemo, useState } from 'react';

// The four Flux family themes — same ids and names across every app, with
// colors defined once per theme in flux-theme.css (driven by
// <html data-theme>).
export type ThemeName = 'dark' | 'nature' | 'attention' | 'ocean';

export const THEMES: { id: ThemeName; label: string; swatch: string }[] = [
  { id: 'dark', label: 'Dark Luxury', swatch: '#cba35c' },
  { id: 'nature', label: 'Zen Nature', swatch: '#59c0ab' },
  { id: 'attention', label: 'Attention', swatch: '#ef8fc9' },
  { id: 'ocean', label: 'Calm Ocean', swatch: '#59a1d5' },
];

const STORAGE_KEY = 'flux_theme';

function isTheme(v: unknown): v is ThemeName {
  return THEMES.some((t) => t.id === v);
}

function applyTheme(theme: ThemeName) {
  if (theme === 'dark') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', theme);
}

function readStoredTheme(): ThemeName {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return isTheme(stored) ? stored : 'dark';
  } catch {
    return 'dark';
  }
}

interface ThemeContextType {
  theme: ThemeName;
  setTheme: (theme: ThemeName) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setThemeState] = useState<ThemeName>(() => {
    const initial = readStoredTheme();
    applyTheme(initial);
    return initial;
  });

  const setTheme = (next: ThemeName) => {
    applyTheme(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // storage unavailable — theme still applies for this session
    }
    setThemeState(next);
  };

  return <ThemeContext.Provider value={{ theme, setTheme }}>{children}</ThemeContext.Provider>;
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) throw new Error('useTheme must be used within a ThemeProvider');
  return context;
};

/** Concrete color strings for the active theme, for places CSS variables
 *  can't reach reliably — SVG attributes set by D3/Recharts, canvas
 *  drawing. Recomputed whenever the theme changes. */
export function useThemeColors() {
  const { theme } = useTheme();
  return useMemo(() => {
    const cs = getComputedStyle(document.documentElement);
    const v = (name: string) => cs.getPropertyValue(name).trim();
    const primaryRgb = v('--primary-rgb');
    return {
      primary: v('--primary'),
      primaryRgb,
      alpha: (a: number) => `rgba(${primaryRgb}, ${a})`,
      grad: [1, 2, 3, 4, 5].map((i) => v(`--grad-${i}`)),
      textMain: v('--text-main'),
      textSecondary: v('--text-secondary'),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);
}
