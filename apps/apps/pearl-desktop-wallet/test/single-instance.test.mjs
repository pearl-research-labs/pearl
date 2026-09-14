// Regression test for the single-instance guard in src/main/index.ts.
//
// Without the guard, a second launch of the app spawns a wallet process that
// cannot bind the RPC port (held by the first instance's wallet process), and
// its RPC calls hit the surviving listener with mismatched per-session
// credentials, failing with HTTP 401. The guard makes that state unreachable:
// a duplicate launch must exit immediately and leave the primary untouched.
//
// Uses only node:test and the electron devDependency — no additional test
// framework. Run via `pnpm test` (builds first; the test exercises the real
// production bundle in out/). Needs a display server: on headless machines
// run it under xvfb-run (Electron's ozone headless mode cannot render this
// app's window and crashes, so the test skips itself when no display exists).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const electronBinary = require('electron');
const appDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const mainBundle = join(appDir, 'out', 'main', 'index.js');

const STARTUP_BANNER = 'Pearl Desktop Wallet starting...';
const DUPLICATE_MESSAGE = 'already running; exiting this duplicate instance';

const STARTUP_TIMEOUT_MS = 20_000;
const DUPLICATE_EXIT_TIMEOUT_MS = 15_000;

function launchApp(configHome) {
  // XDG_CONFIG_HOME redirects userData, so the test uses an isolated
  // single-instance lock and log file and never touches a real wallet profile.
  const child = spawn(
    electronBinary,
    [appDir, '--no-sandbox', '--disable-gpu'],
    {
      env: { ...process.env, XDG_CONFIG_HOME: configHome },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  child.output = '';
  for (const stream of [child.stdout, child.stderr]) {
    stream.setEncoding('utf8');
    stream.on('data', (chunk) => {
      child.output += chunk;
    });
  }
  return child;
}

function waitForOutput(child, needle, timeoutMs, label) {
  return new Promise((resolve, reject) => {
    const check = () => {
      if (child.output.includes(needle)) {
        cleanup();
        resolve();
      }
    };
    const onExit = (code) => {
      cleanup();
      reject(new Error(`${label} exited (code ${code}) before printing ${JSON.stringify(needle)}. Output:\n${child.output}`));
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error(`${label} did not print ${JSON.stringify(needle)} within ${timeoutMs}ms. Output:\n${child.output}`));
    }, timeoutMs);
    const poll = setInterval(check, 50);
    const cleanup = () => {
      clearTimeout(timer);
      clearInterval(poll);
      child.off('exit', onExit);
    };
    child.on('exit', onExit);
    check();
  });
}

function waitForExit(child, timeoutMs) {
  return new Promise((resolve, reject) => {
    if (child.exitCode !== null) {
      resolve(child.exitCode);
      return;
    }
    const timer = setTimeout(() => {
      reject(new Error(`process did not exit within ${timeoutMs}ms. Output:\n${child.output}`));
    }, timeoutMs);
    child.on('exit', (code) => {
      clearTimeout(timer);
      resolve(code);
    });
  });
}

async function terminate(child) {
  if (child.exitCode !== null) return;
  child.kill('SIGTERM');
  try {
    await waitForExit(child, 10_000);
  } catch {
    child.kill('SIGKILL');
  }
}

test('duplicate launch exits immediately and leaves the primary instance running', async (t) => {
  if (!process.env.DISPLAY && !process.env.WAYLAND_DISPLAY) {
    t.skip('requires a display server — run under xvfb-run on headless machines');
    return;
  }
  assert.ok(
    existsSync(mainBundle),
    `missing ${mainBundle} — run \`pnpm build\` first (or run the test via \`pnpm test\`)`,
  );

  const configHome = mkdtempSync(join(tmpdir(), 'pearl-wallet-test-'));
  const mainLog = join(configHome, '@pearl', 'pearl-desktop-wallet', 'logs', 'main.log');
  const primary = launchApp(configHome);
  let duplicate;
  try {
    await waitForOutput(primary, STARTUP_BANNER, STARTUP_TIMEOUT_MS, 'primary instance');
    // The banner is logged immediately before the lock is acquired; give the
    // primary a beat to finish acquiring it before racing a duplicate.
    await new Promise((r) => setTimeout(r, 500));

    duplicate = launchApp(configHome);
    const exitCode = await waitForExit(duplicate, DUPLICATE_EXIT_TIMEOUT_MS);

    assert.equal(exitCode, 0, `duplicate should exit cleanly. Output:\n${duplicate.output}`);
    // Assert on the shared log file rather than the duplicate's stdout: the
    // file transport writes synchronously, while pipe output can be lost when
    // the process exits immediately.
    assert.ok(
      readFileSync(mainLog, 'utf8').includes(DUPLICATE_MESSAGE),
      'duplicate should log the single-instance message before exiting',
    );
    assert.equal(primary.exitCode, null, 'primary instance must still be running after the duplicate exits');
  } finally {
    await terminate(primary);
    if (duplicate) await terminate(duplicate);
    rmSync(configHome, { recursive: true, force: true });
  }
});
