import fs from 'fs';
import path from 'path';

export const SETUP_FILE_NAME = 'wallet-setup.json';

const DATA_DIR_MODE = 0o700;
const SETUP_FILE_MODE = 0o600;

const activeSetupFiles = new Set<string>();
// Bumped on every rewrite of a path. A setup child's close handler captures
// the generation it wrote and must not wipe a later rewrite of the same path.
const setupFileGenerations = new Map<string, number>();
// Paths a live setup child may still have open. Unlink can fail for these
// (Windows); zeroing them publishes zeros to that reader.
const openSetupFiles = new Set<string>();

export function setupFileGeneration(file: string): number {
  return setupFileGenerations.get(file) ?? 0;
}

function advanceSetupFileGeneration(file: string): void {
  setupFileGenerations.set(file, setupFileGeneration(file) + 1);
}

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

function isErrno(error: unknown, code: string): error is NodeJS.ErrnoException {
  return error instanceof Error && (error as NodeJS.ErrnoException).code === code;
}

function openNoFollow(file: string, flags: number): number {
  // O_NOFOLLOW is 0 on Windows; callers lstat first so a symlink is never opened there.
  return fs.openSync(file, flags | (fs.constants.O_NOFOLLOW || 0));
}

function isMissing(file: string): boolean {
  try {
    fs.lstatSync(file);
    return false;
  } catch (error) {
    return isErrno(error, 'ENOENT');
  }
}

export function writeSetupFile(dataDir: string, contents: string): string {
  const file = setupFilePath(dataDir);
  // Publish the new generation before replacing bytes. stopSetupChild can
  // return while the previous child's close handler is still armed; once this
  // generation is current that handler must not zero the file we write here.
  advanceSetupFileGeneration(file);
  activeSetupFiles.add(file);
  const stillOpen = openSetupFiles.delete(file);
  if (stillOpen) {
    // Child is still alive. Do not zero. Unlink detaches the name on POSIX
    // without altering the inode the child is reading. If unlink fails, the
    // exclusive-create fallback below overwrites a regular file in place.
    try {
      fs.unlinkSync(file);
    } catch {
      // ignore
    }
  } else {
    removeSetupFile(file);
  }
  try {
    try {
      fs.writeFileSync(file, contents, { mode: SETUP_FILE_MODE, flag: 'wx' });
    } catch (error) {
      if (!isErrno(error, 'EEXIST')) {
        throw error;
      }
      // Unlink can fail while another handle still has the file open (Windows).
      // Replace a regular file in place. Never write through a symlink: wx
      // already refused, and O_NOFOLLOW keeps the fallback from following one.
      const stat = fs.lstatSync(file);
      if (stat.isSymbolicLink() || !stat.isFile()) {
        throw error;
      }
      const fd = openNoFollow(file, fs.constants.O_WRONLY | fs.constants.O_TRUNC);
      try {
        fs.writeFileSync(fd, contents);
        try {
          fs.fchmodSync(fd, SETUP_FILE_MODE);
        } catch {
          // ignore
        }
        fs.fsyncSync(fd);
      } finally {
        fs.closeSync(fd);
      }
    }
  } catch (error) {
    if (isMissing(file)) {
      activeSetupFiles.delete(file);
    } else {
      activeSetupFiles.add(file);
      if (stillOpen) {
        openSetupFiles.add(file);
      }
    }
    throw error;
  }
  activeSetupFiles.add(file);
  return file;
}

export function untrackSetupFile(file: string): void {
  activeSetupFiles.delete(file);
  // Give-up left the setup child alive. The next rewrite must not zero a
  // file that child may still be reading.
  openSetupFiles.add(file);
}

export function removeSetupFile(file: string, generation?: number): void {
  if (generation !== undefined && setupFileGenerations.get(file) !== generation) {
    return;
  }
  openSetupFiles.delete(file);

  let stat: fs.Stats;
  try {
    stat = fs.lstatSync(file);
  } catch (error) {
    // Quit retries tracked paths. ENOENT has nothing left to retry.
    if (isErrno(error, 'ENOENT')) {
      activeSetupFiles.delete(file);
    }
    return;
  }

  // stat/open follow symlinks. Unlink the link itself and leave its target alone.
  if (!stat.isSymbolicLink() && stat.isFile() && stat.size > 0) {
    try {
      const fd = openNoFollow(file, fs.constants.O_RDWR);
      try {
        fs.writeSync(fd, Buffer.alloc(stat.size), 0, stat.size, 0);
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
    activeSetupFiles.delete(file);
  } catch {
    // Leave the path tracked so quit cleanup can retry a locked leftover.
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
