import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import {
  removeActiveSetupFiles,
  removeSetupFile,
  writeSetupFile,
} from '../src/main/services/setup-file.ts';

function tempWalletDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'setup-file-'));
}

function withDirMode(dir: string, mode: number, fn: () => void): void {
  fs.chmodSync(dir, mode);
  try {
    fn();
  } finally {
    fs.chmodSync(dir, 0o700);
  }
}

test('unlink failure stays tracked so quit cleanup can retry', () => {
  const dir = tempWalletDir();
  try {
    const file = writeSetupFile(dir, 'seed-material');
    withDirMode(dir, 0o555, () => {
      removeSetupFile(file);
    });
    assert.equal(fs.existsSync(file), true);
    removeActiveSetupFiles();
    assert.equal(fs.existsSync(file), false);
  } finally {
    fs.chmodSync(dir, 0o700);
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('wipe and unlink failure stays tracked so quit cleanup can retry', () => {
  const dir = tempWalletDir();
  try {
    const file = writeSetupFile(dir, 'seed-material');
    fs.chmodSync(file, 0o444);
    withDirMode(dir, 0o555, () => {
      removeSetupFile(file);
    });
    fs.chmodSync(file, 0o600);
    assert.equal(fs.readFileSync(file, 'utf8'), 'seed-material');
    removeActiveSetupFiles();
    assert.equal(fs.existsSync(file), false);
  } finally {
    fs.chmodSync(dir, 0o700);
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('stat errors other than ENOENT stay tracked so quit cleanup can retry', () => {
  const dir = tempWalletDir();
  try {
    const file = writeSetupFile(dir, 'seed-material');
    withDirMode(dir, 0o000, () => {
      removeSetupFile(file);
    });
    assert.equal(fs.readFileSync(file, 'utf8'), 'seed-material');
    removeActiveSetupFiles();
    assert.equal(fs.existsSync(file), false);
  } finally {
    fs.chmodSync(dir, 0o700);
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('ENOENT drops the path so quit cleanup does not remove a later file', () => {
  const dir = tempWalletDir();
  try {
    const file = writeSetupFile(dir, 'seed-material');
    fs.unlinkSync(file);
    removeSetupFile(file);
    fs.writeFileSync(file, 'recreated', { mode: 0o600 });
    removeActiveSetupFiles();
    assert.equal(fs.readFileSync(file, 'utf8'), 'recreated');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('successful unlink drops the path so quit cleanup does not remove a later file', () => {
  const dir = tempWalletDir();
  try {
    const file = writeSetupFile(dir, 'seed-material');
    removeSetupFile(file);
    assert.equal(fs.existsSync(file), false);
    fs.writeFileSync(file, 'recreated', { mode: 0o600 });
    removeActiveSetupFiles();
    assert.equal(fs.readFileSync(file, 'utf8'), 'recreated');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
