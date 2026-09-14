import { app, BrowserWindow } from 'electron';
import { electronApp, optimizer } from '@electron-toolkit/utils';
import log from 'electron-log';
import { registerWalletIpc } from './ipc/register-wallet-ipc';
import { WindowService } from './services/window-service/window-service';
import { createMainWindow } from './services/window-service/create-window';
import { registerWindowIpc } from './ipc/register-window-ipc';
import { registerManagerIpc } from './ipc/register-manager-ipc';
import { ManagerService } from './services/manager-service';
import { registerSyncIpc } from './ipc/register-sync-ipc';
import { SyncService } from './services/sync-service';
import { UpdateService } from './services/update-service/update-service';
import { registerUpdateIpc } from './ipc/register-update-ipc';
import {
  UPDATE_RELEASE_TAG_PREFIX,
  UPDATE_REPO_NAME,
  UPDATE_REPO_OWNER,
} from './config/consts';

// Setup logging for both dev and production
log.transports.file.level = 'info';
log.transports.console.level = 'info';

const MAX_LOG_SIZE = 5 * 1024 * 1024; // 5MB

function enforceLogSizeLimit() {
  try {
    const logFile = log.transports.file.getFile().path;
    const stat = require('fs').statSync(logFile);
    if (stat.size > MAX_LOG_SIZE) {
      const content = require('fs').readFileSync(logFile, 'utf8');
      // Keep only the last 100KB of logs (most recent entries)
      const trimmed = content.slice(-MAX_LOG_SIZE);
      // Find the first complete line to avoid cutting mid-line
      const firstNewline = trimmed.indexOf('\n');
      require('fs').writeFileSync(logFile, firstNewline > -1 ? trimmed.slice(firstNewline + 1) : trimmed);
    }
  } catch {
    // ignore errors
  }
}

setInterval(enforceLogSizeLimit, 300_000); // Every 5 minutes enforce log file size is not exceeding 5MB

// Override console methods to use electron-log
console.log = log.log.bind(log);
console.error = log.error.bind(log);
console.warn = log.warn.bind(log);
console.info = log.info.bind(log);
console.debug = log.debug.bind(log);

log.info('====================================');
log.info('Pearl Desktop Wallet starting...');
log.info(`App version: ${app.getVersion()}`);
log.info(`Electron version: ${process.versions.electron}`);
log.info(`Node version: ${process.versions.node}`);
log.info(`Platform: ${process.platform}`);
log.info(`Log file location: ${log.transports.file.getFile().path}`);
log.info('====================================');

// Enforce a single running instance. Each instance generates its own RPC
// credentials, so a duplicate launch spawns a wallet process that cannot bind
// the RPC port and every call to the surviving listener fails with 401.
if (!app.requestSingleInstanceLock()) {
  log.info('Pearl Wallet is already running; exiting this duplicate instance.');
  app.exit(0);
} else {
  app.on('second-instance', () => {
    const existingWindow = BrowserWindow.getAllWindows()[0];
    if (existingWindow) {
      if (existingWindow.isMinimized()) {
        existingWindow.restore();
      }
      existingWindow.show();
      existingWindow.focus();
    }
  });
}

let managerService: ManagerService | null = null;
let updateService: UpdateService | null = null;

app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.pearl.wallet');

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window);
  });

  const windowAS = new WindowService(createMainWindow());
  registerWindowIpc(windowAS);

  managerService = new ManagerService();
  registerManagerIpc(managerService);

  registerWalletIpc(managerService);

  const syncService = new SyncService(managerService);
  registerSyncIpc(syncService);

  updateService = new UpdateService({
    owner: UPDATE_REPO_OWNER,
    repo: UPDATE_REPO_NAME,
    tagPrefix: UPDATE_RELEASE_TAG_PREFIX,
  });
  registerUpdateIpc(updateService);
  updateService.start();

  // Run an update check when the user brings the wallet back to the
  // foreground. The service throttles internally to one attempt per
  // `minIntervalMs`, so fast focus/blur cycles are free.
  app.on('browser-window-focus', () => {
    updateService?.notifyWindowFocus();
  });

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) {
      windowAS.setMainWindow(createMainWindow());
    }
  });
});

app.on('window-all-closed', async () => {
  try {
    if (process.platform !== 'darwin') {
      app.quit();
    }
  } catch (error) {
    // Do nothing
  }
});

// Perform async cleanup exactly once before quitting
let isQuitting = false;

app.on('before-quit', async event => {
  if (isQuitting) {
    return; // allow the quit to proceed
  }
  event.preventDefault(); // pause the quit
  isQuitting = true;
  try {
    updateService?.stop();
    await managerService?.stopWalletProcess();
  } catch {
    // swallow errors
  }
  app.quit(); // resume quit; handler runs again but won't prevent
});
