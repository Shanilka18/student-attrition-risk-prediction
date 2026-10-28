/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    // Override default colors entirely to prevent generic template look or forbidden colors
    colors: {
      transparent: 'transparent',
      current: 'currentColor',
      paper: '#F5F2EC',
      surface: '#FFFFFF',
      ink: '#1A1D21',
      muted: '#5B6168',
      line: '#DDD8CE',
      grid: '#E7E2D8',
      
      // Primary brand / action
      forest: {
        DEFAULT: '#1F4E46',
        hover: '#173C36',
        active: '#112B26',
        subtle: '#EBF2EE',
      },

      // Contextual / Status
      info: {
        DEFAULT: '#2B5C8A',
        hover: '#22496E',
        subtle: '#EDF3F8',
      },
      warning: {
        DEFAULT: '#C98A00',
        hover: '#A67200',
        subtle: '#FCF6E8',
      },

      // Risk colors (ONLY for risk semantics)
      risk: {
        low: '#2E7D4F',
        'low-subtle': '#EBF5EE',
        moderate: '#C98A00',
        'moderate-subtle': '#FCF6E8',
        high: '#B3261E',
        'high-subtle': '#FBECEB',
      },

      // Extra categorical tokens for multi-series charts & patterns
      clay: {
        DEFAULT: '#8C4A3B',
        subtle: '#F6EEEB',
      },
      olive: {
        DEFAULT: '#6E7F3F',
        subtle: '#F2F5EB',
      },
      stone: {
        DEFAULT: '#7A6F63',
        subtle: '#F3F1EF',
      },
    },
    extend: {
      fontFamily: {
        serif: ['"Source Serif 4"', 'Georgia', 'serif'],
        sans: ['"Public Sans"', 'system-ui', '-apple-system', 'sans-serif'],
      },
      fontSize: {
        'page-title': ['28px', { lineHeight: '34px', fontWeight: '600' }],
        'section-title': ['18px', { lineHeight: '26px', fontWeight: '600' }],
        body: ['15px', { lineHeight: '24px' }],
        table: ['14px', { lineHeight: '20px' }],
        caption: ['12.5px', { lineHeight: '18px' }],
      },
      borderRadius: {
        DEFAULT: '6px',
        sm: '4px',
        md: '6px',
        lg: '6px',
        xl: '6px',
      },
      boxShadow: {
        none: 'none',
        menu: '0 4px 16px -2px rgba(26, 29, 33, 0.12), 0 2px 6px -1px rgba(26, 29, 33, 0.08)',
        popover: '0 4px 16px -2px rgba(26, 29, 33, 0.12), 0 2px 6px -1px rgba(26, 29, 33, 0.08)',
        toast: '0 4px 16px -2px rgba(26, 29, 33, 0.12)',
      },
      maxWidth: {
        content: '1280px',
      },
      width: {
        sidebar: '232px',
      },
    },
  },
  plugins: [],
};
