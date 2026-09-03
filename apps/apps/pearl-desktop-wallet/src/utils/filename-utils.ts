type Platform = 'windows' | 'mac' | 'linux';

/** Detect current platform without needing Node types */
function detectPlatform(): Platform {
  const p = (globalThis as Record<string, unknown>)?.process?.platform as string | undefined;
  if (p === 'darwin') return 'mac';
  if (p === 'win32') return 'windows';
  return 'linux'; // default bucket for others
}

// Count Unicode code points (not UTF-16 code units)
function codePointLen(s: string): number {
  return Array.from(s).length;
}

// Windows reserved device names (case-insensitive), with or without extension
const WINDOWS_RESERVED = new Set([
  'CON',
  'PRN',
  'AUX',
  'NUL',
  'COM1',
  'COM2',
  'COM3',
  'COM4',
  'COM5',
  'COM6',
  'COM7',
  'COM8',
  'COM9',
  'LPT1',
  'LPT2',
  'LPT3',
  'LPT4',
  'LPT5',
  'LPT6',
  'LPT7',
  'LPT8',
  'LPT9',
]);

function hasAsciiControl(name: string): boolean {
  for (let i = 0; i < name.length; i++) {
    const c = name.charCodeAt(i);
    if (c >= 0 && c <= 31) return true;
  }
  return false;
}

function nativeSeparators(platform: Platform): string[] {
  return platform === 'windows' ? ['\\', '/'] : ['/'];
}

/** Internal validation per host OS */
function validateForPlatform(name: string, platform: Platform): boolean {
  if (typeof name !== 'string') return false;
  if (name.length === 0) return false;
  if (name === '.' || name === '..') return false;
  if (codePointLen(name) > 255) return false;
  if (hasAsciiControl(name)) return false;
  if (name.includes('\u0000')) return false;

  switch (platform) {
    case 'windows': {
      if (/[<>:"/\\|?*]/.test(name)) return false;
      if (/[ .]$/.test(name)) return false;
      const base = name.split('.')[0].toUpperCase();
      if (WINDOWS_RESERVED.has(base)) return false;
      return true;
    }
    case 'mac': {
      // Only '/' is outright forbidden in a component.
      if (name.includes('/')) return false;
      // ':' is allowed (maps to '/' internally by the FS).
      return true;
    }
    case 'linux': {
      // Only '/' and NUL are forbidden; NUL already checked.
      if (name.includes('/')) return false;
      return true;
    }
  }
}

/** Guard: ensure we're dealing with a single path component, not a full path */
function assertSingleComponent(name: string, platform: Platform): void {
  const seps = nativeSeparators(platform);
  if (seps.some(sep => name.includes(sep))) {
    throw new Error('Expected a single filename component, not a full path.');
  }
}

/**
 * Convert a *display* filename to the underlying *filesystem* form.
 * - On macOS: replaces ':' → '/' (APFS/HFS+ on-disk encoding).
 * - On Windows/Linux: returns the input unchanged.
 * Expects a single filename component (not a full path).
 */
export function displayToFs(filename: string): string {
  const platform = detectPlatform();
  assertSingleComponent(filename, platform);
  if (platform === 'mac') {
    // ':' typed by user → '/' stored on disk within the component
    return filename.replace(/:/g, '/');
  }
  return filename.trim();
}

/**
 * Convert a *filesystem-stored* filename to the typical *display* form.
 * - On macOS: replaces '/' → ':' within a single component.
 * - On Windows/Linux: returns the input unchanged.
 * Expects a single filename component (not a full path).
 */
export function fsToDisplay(filename: string): string {
  const platform = detectPlatform();
  assertSingleComponent(filename, platform);
  if (platform === 'mac') {
    // Stored '/' within the component → shown as ':'
    return filename.replace(/\//g, ':');
  }
  return filename;
}

/**
 * Validate a single filename component for the **current host OS**.
 * (Windows/macOS/Linux rules auto-detected.)
 */
export function isValidFilename(filename: string): boolean {
  const platform = detectPlatform();
  return validateForPlatform(filename, platform);
}
