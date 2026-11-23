/**
 * Appearance settings.
 *
 * The renderer keeps its theme in localStorage, which the main process cannot
 * read before the window opens. The native theme (macOS title bar) is mirrored
 * here so it can be applied before the window is created.
 */
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

type NativeTheme = 'light' | 'dark';

const SETTINGS_DIR = path.join(os.homedir(), '.pearl-wallet', 'settings');
const SETTINGS_FILE = path.join(SETTINGS_DIR, 'appearance-settings.json');

// Load the saved native theme, defaulting to light when none is saved
function loadNativeTheme(): NativeTheme {
  try {
    const parsed = JSON.parse(fs.readFileSync(SETTINGS_FILE, 'utf-8'));
    return parsed.theme === 'dark' ? 'dark' : 'light';
  } catch {
    return 'light';
  }
}

function saveNativeTheme(theme: NativeTheme) {
  try {
    fs.mkdirSync(SETTINGS_DIR, {recursive: true});
    fs.writeFileSync(SETTINGS_FILE, JSON.stringify({theme}, null, 2));
  } catch (error) {
    console.error('Failed to save appearance settings:', error);
  }
}

export {loadNativeTheme, saveNativeTheme};
export type {NativeTheme};
