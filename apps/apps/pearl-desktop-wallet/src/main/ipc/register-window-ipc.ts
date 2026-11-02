import {ipcMain} from 'electron';
import {WindowService} from '../services/window-service/window-service';

function registerWindowIpc(was: WindowService) {
  ipcMain.handle('app-version', () => was.getVersion());
  ipcMain.handle('show-message-box', (_, options) => was.showMessageBox(options));
  ipcMain.handle('show-open-dialog', (_, options) => was.showOpenDialog(options));
  ipcMain.handle('show-save-dialog', (_, options) => was.showSaveDialog(options));
  ipcMain.handle('window-minimize', () => was.minimizeWindow());
  ipcMain.handle('window-maximize', () => was.maximizeWindow());
  ipcMain.handle('window-close', () => was.closeWindow());
  ipcMain.handle('window-is-maximized', () => was.isMaximized());
  ipcMain.handle('open-external', (_, url: string) => was.openExternal(url));
  ipcMain.handle('window-set-native-theme', (_, theme) => was.setNativeTheme(theme));
}

export {registerWindowIpc};
