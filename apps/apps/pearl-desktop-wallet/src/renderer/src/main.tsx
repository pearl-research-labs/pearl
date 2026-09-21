import React from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider } from 'next-themes';
import './assets/index.css';
import App from './App';

// Sets the `dark` class on <html>, which tailwind.config.ts already keys on.
// Follows the OS setting by default and remembers an explicit choice.
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      storageKey="pearl-wallet-theme"
      disableTransitionOnChange
    >
      <App />
    </ThemeProvider>
  </React.StrictMode>
);
