import assert from 'node:assert/strict';
import {mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {test} from 'node:test';
import {fileURLToPath} from 'node:url';
import ts from 'typescript';

const appDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sourceFile = resolve(appDir, 'src/main/config/appearance-settings.ts');

// Loads appearance-settings.ts with os.homedir() pointed at a throwaway directory
function loadAppearanceSettings(home) {
  const {outputText} = ts.transpileModule(readFileSync(sourceFile, 'utf8'), {
    compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true},
  });
  const realRequire = createRequire(sourceFile);
  const require = request => (request === 'os' ? {...realRequire('os'), homedir: () => home} : realRequire(request));
  const module = {exports: {}};
  new Function('require', 'module', 'exports', outputText)(require, module, module.exports);
  return module.exports;
}

function withHome(fn) {
  const home = mkdtempSync(join(tmpdir(), 'pearl-appearance-test-'));
  try {
    fn(home, loadAppearanceSettings(home));
  } finally {
    rmSync(home, {recursive: true, force: true});
  }
}

test('defaults to light when nothing is saved', () => {
  withHome((_, {loadNativeTheme}) => {
    assert.equal(loadNativeTheme(), 'light');
  });
});

test('a saved theme is loaded on the next launch', () => {
  withHome((home, {loadNativeTheme, saveNativeTheme}) => {
    saveNativeTheme('dark');
    assert.equal(loadNativeTheme(), 'dark');
    const saved = JSON.parse(readFileSync(join(home, '.pearl-wallet/settings/appearance-settings.json'), 'utf8'));
    assert.deepEqual(saved, {theme: 'dark'});

    saveNativeTheme('light');
    assert.equal(loadNativeTheme(), 'light');
  });
});

test('unreadable or unexpected settings fall back to light', () => {
  withHome((home, {loadNativeTheme}) => {
    const dir = join(home, '.pearl-wallet/settings');
    mkdirSync(dir, {recursive: true});

    writeFileSync(join(dir, 'appearance-settings.json'), '{not json');
    assert.equal(loadNativeTheme(), 'light');

    writeFileSync(join(dir, 'appearance-settings.json'), JSON.stringify({theme: 'purple'}));
    assert.equal(loadNativeTheme(), 'light');
  });
});
