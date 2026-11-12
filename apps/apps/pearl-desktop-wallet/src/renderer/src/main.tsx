import React, {useEffect} from 'react';
import ReactDOM from 'react-dom/client';
import {ThemeProvider, useTheme} from 'next-themes';
import './assets/index.css';
import App from './App';

// Keeps the native macOS title bar in step with the in-app theme.
function NativeThemeSync() {
  const {resolvedTheme} = useTheme();

  useEffect(() => {
    if (!resolvedTheme) return;
    window.appBridge.window
      .setNativeTheme(resolvedTheme === 'dark' ? 'dark' : 'light')
      .catch(error => console.error('Failed to set native theme:', error));
  }, [resolvedTheme]);

  return null;
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false} storageKey="pearl-theme">
      <NativeThemeSync />
      <App />
    </ThemeProvider>
  </React.StrictMode>
);
