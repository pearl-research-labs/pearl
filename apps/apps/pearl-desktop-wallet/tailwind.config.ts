import type { Config } from 'tailwindcss';
import pearlDesign from '@pearl/ui/tailwind';

// all in fixtures is set to tailwind v3 as interims solutions

// Light palette colors that pages use directly (bg-white, text-gray-900, ...) resolve
// through --themed-* variables so dark mode can swap them. The light values in
// index.css match Tailwind's defaults, so light mode renders exactly as before.
const themed = (name: string) => `rgb(var(--themed-${name}) / <alpha-value>)`;

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
      backgroundColor: {
        white: themed('bg-white'),
        gray: {
          50: themed('bg-gray-50'),
          100: themed('bg-gray-100'),
          200: themed('bg-gray-200'),
          300: themed('bg-gray-300'),
        },
        red: {50: themed('bg-red-50')},
        green: {50: themed('bg-green-50'), 100: themed('bg-green-100')},
        amber: {50: themed('bg-amber-50'), 100: themed('bg-amber-100')},
        blue: {50: themed('bg-blue-50')},
      },
      gradientColorStops: {
        gray: {
          50: themed('gradient-gray-50'),
          100: themed('gradient-gray-100'),
        },
      },
      textColor: {
        gray: {
          400: themed('text-gray-400'),
          500: themed('text-gray-500'),
          600: themed('text-gray-600'),
          700: themed('text-gray-700'),
          900: themed('text-gray-900'),
        },
        red: {
          600: themed('text-red-600'),
          700: themed('text-red-700'),
          800: themed('text-red-800'),
          900: themed('text-red-900'),
        },
        green: {
          600: themed('text-green-600'),
          700: themed('text-green-700'),
          800: themed('text-green-800'),
          900: themed('text-green-900'),
        },
        amber: {
          600: themed('text-amber-600'),
          700: themed('text-amber-700'),
          800: themed('text-amber-800'),
          900: themed('text-amber-900'),
        },
        blue: {700: themed('text-blue-700')},
        orange: {600: themed('text-orange-600')},
        yellow: {600: themed('text-yellow-600')},
      },
      placeholderColor: {
        gray: {400: themed('text-gray-400')},
      },
      borderColor: {
        gray: {
          100: themed('border-gray-100'),
          200: themed('border-gray-200'),
          300: themed('border-gray-300'),
          400: themed('border-gray-400'),
          900: themed('border-gray-900'),
        },
        red: {200: themed('border-red-200'), 300: themed('border-red-300')},
        green: {200: themed('border-green-200'), 300: themed('border-green-300')},
        amber: {300: themed('border-amber-300')},
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
};
export default config;
