import { useEffect, useState } from 'react';
import { Monitor, Moon, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';

// Light → Dark → System, one click each. Rendered only after mount so the
// icon matches the theme the page actually loaded with.
const ORDER = ['light', 'dark', 'system'] as const;
const LABEL: Record<(typeof ORDER)[number], string> = {
  light: 'Light theme',
  dark: 'Dark theme',
  system: 'Follow system theme',
};

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const current = (ORDER as readonly string[]).includes(theme ?? '') ? (theme as (typeof ORDER)[number]) : 'system';
  const next = ORDER[(ORDER.indexOf(current) + 1) % ORDER.length];
  const Icon = current === 'light' ? Sun : current === 'dark' ? Moon : Monitor;

  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      className="flex items-center gap-2 rounded-lg border border-gray-300 dark:border-border bg-white dark:bg-card px-3 py-2 text-sm text-gray-700 dark:text-foreground/90 transition-colors hover:bg-gray-50 dark:hover:bg-muted/60"
      title={mounted ? `${LABEL[current]} (click for ${next})` : 'Theme'}
      aria-label={mounted ? LABEL[current] : 'Theme'}
    >
      <Icon className="h-4 w-4" />
    </button>
  );
}
