/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Base: deep slate-blue, not near-black -- an instrument-panel ground,
        // not the near-black+neon default.
        panel: {
          950: '#0A0E16',
          900: '#0E1420',
          800: '#141B2B',
          700: '#1D2638',
          600: '#2A3548',
          500: '#465066',
          400: '#6B7690',
          300: '#98A2B8',
          200: '#C5CCDA',
          100: '#E8EBF1',
        },
        // Semantic states, mapped 1:1 to trading meaning -- never used decoratively.
        nominal: {
          DEFAULT: '#4EA89E',
          dim: '#2E5F59',
        },
        caution: {
          DEFAULT: '#D9A441',
          dim: '#6B5326',
        },
        danger: {
          DEFAULT: '#E0524F',
          dim: '#6B2E2D',
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
