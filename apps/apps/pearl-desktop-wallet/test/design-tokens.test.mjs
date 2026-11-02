import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {test} from 'node:test';
import {fileURLToPath} from 'node:url';

// Every CSS variable the @pearl/ui Tailwind preset reads must be defined for light mode.
// An undefined variable makes the whole color invalid, so text silently inherits, borders
// fall back to currentColor and backgrounds go transparent (e.g. white text on bg-brand-green).

const appDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const uiDir = resolve(appDir, '../../packages/ui');

function rootVariables(cssFile) {
  const css = readFileSync(cssFile, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
  const defined = new Set();
  for (const [, selector, body] of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    if (!/(^|[\s,]):root\b/.test(selector)) continue;
    for (const [, name] of body.matchAll(/--([\w-]+)\s*:/g)) defined.add(name);
  }
  return defined;
}

test('every variable used by the Tailwind preset is defined in :root', () => {
  const preset = readFileSync(resolve(uiDir, 'tailwind.config.js'), 'utf8');
  const used = new Set([...preset.matchAll(/var\(--([\w-]+)\)/g)].map(([, name]) => name));
  const defined = new Set([
    ...rootVariables(resolve(uiDir, 'styles/design-tokens.css')),
    ...rootVariables(resolve(appDir, 'src/renderer/src/assets/index.css')),
  ]);

  const missing = [...used].filter(name => !defined.has(name)).sort();
  assert.deepEqual(missing, [], `undefined CSS variables: ${missing.join(', ')}`);
});
