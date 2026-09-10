import tailwindcssAnimate from 'tailwindcss-animate';
import type { Config } from 'tailwindcss';
import pearlDesign from '@pearl/ui/tailwind';

// all in fixtures is set to tailwind v3 as interims solutions

const config: Config = {
  darkMode: ['class'],
  content: ['./src/renderer/src/**/*.{js,ts,jsx,tsx,mdx}', './src/renderer/index.html'],
  presets: [pearlDesign],
  theme: {
    extend: {
      // Additional pearl-desktop-wallet specific extensions can go here
      keyframes: {
        'accordion-down': {
          from: {
            height: '0',
          },
          to: {
            height: 'var(--radix-accordion-content-height)',
          },
        },
        'accordion-up': {
          from: {
            height: 'var(--radix-accordion-content-height)',
          },
          to: {
            height: '0',
          },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
      },
    },
  },
  plugins: [tailwindcssAnimate],
};
export default config;
