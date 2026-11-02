import {
  app,
  BrowserWindow,
  dialog,
  MessageBoxOptions,
  nativeTheme,
  OpenDialogOptions,
  SaveDialogOptions,
  shell,
} from 'electron';
import {WindowApi} from '../../../types/app-bridge.ts';

class WindowService implements WindowApi {
  constructor(private mainWindow: BrowserWindow) {}

  setMainWindow(mainWindow: BrowserWindow) {
    this.mainWindow = mainWindow;
  }

  getVersion() {
    return app.getVersion();
  }

  minimizeWindow() {
    return this.mainWindow.minimize();
  }

  maximizeWindow() {
    if (this.mainWindow.isMaximized()) {
      this.mainWindow.unmaximize();
    } else {
      this.mainWindow.maximize();
    }
  }

  closeWindow() {
    this.mainWindow.close();
  }

  isMaximized() {
    return this.mainWindow.isMaximized();
  }

  showMessageBox(options: MessageBoxOptions) {
    return dialog.showMessageBox(this.mainWindow, options);
  }

  showOpenDialog(options: OpenDialogOptions) {
    return dialog.showOpenDialog(this.mainWindow, options);
  }

  showSaveDialog(options: SaveDialogOptions) {
    return dialog.showSaveDialog(this.mainWindow, options);
  }

  openExternal(url: string) {
    shell.openExternal(url);
  }

  // Dark mode forces a dark title bar; light keeps Electron's default of following the OS.
  setNativeTheme(theme: 'light' | 'dark') {
    nativeTheme.themeSource = theme === 'dark' ? 'dark' : 'system';
  }
  //   platform: process.platform;
  //   isElectron: true;
}

export {WindowService};
