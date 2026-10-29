/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        sliit: {
          navy: '#0B192C',
          deep: '#1E3E62',
          orange: '#FF6500',
          dark: '#000000',
        },
      },
    },
  },
  plugins: [],
}
