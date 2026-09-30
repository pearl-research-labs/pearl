import fs from 'fs';
import path from 'path';

export const SETUP_FILE_NAME = 'wallet-setup.json';

const DATA_DIR_MODE = 0o700;
const SETUP_FILE_MODE = 0o600;

const activeSetupFiles = new Set<string>();

export function setupFilePath(dataDir: string): string {
  return path.join(dataDir, SETUP_FILE_NAME);
}

export function ensureDataDir(dir: string): void {
  fs.mkdirSync(dir, { recursive: true, mode: DATA_DIR_MODE });
  if (process.platform === 'win32') {
    return;
  }
  try {
    fs.chmodSync(dir, DATA_DIR_MODE);
  } catch {
    // ignore
  }
}

export function writeSetupFile(dataDir: string, contents: string): string {
  const file = setupFilePath(dataDir);
  removeSetupFile(file);
  fs.writeFileSync(file, contents, { mode: SETUP_FILE_MODE, flag: 'wx' });
  activeSetupFiles.add(file);
  return file;
}

export function removeSetupFile(file: string): void {
  activeSetupFiles.delete(file);

  let size: number;
  try {
    size = fs.statSync(file).size;
  } catch {
    return;
  }

  if (size > 0) {
    try {
      const fd = fs.openSync(file, 'r+');
      try {
        fs.writeSync(fd, Buffer.alloc(size), 0, size, 0);
        fs.fsyncSync(fd);
      } finally {
        fs.closeSync(fd);
      }
    } catch {
      // ignore
    }
  }

  try {
    fs.unlinkSync(file);
  } catch {
    // ignore
  }
}

export function removeActiveSetupFiles(): void {
  for (const file of Array.from(activeSetupFiles)) {
    removeSetupFile(file);
  }
}

export function removeStaleSetupFiles(baseWalletDir: string): void {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(baseWalletDir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.isDirectory()) {
      removeSetupFile(setupFilePath(path.join(baseWalletDir, entry.name)));
    }
  }
}
