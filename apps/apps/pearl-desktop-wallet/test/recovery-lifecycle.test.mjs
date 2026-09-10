import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const appDir = join(dirname(fileURLToPath(import.meta.url)), '..');

function source(path) {
  return readFileSync(join(appDir, path), 'utf8');
}

test('daemon readiness does not wait for a balance DB read during seed recovery', () => {
  const walletProcess = source('src/main/services/wallet-process.ts');
  const readinessBlock = walletProcess.slice(
    walletProcess.indexOf('const startPolling'),
    walletProcess.indexOf("this.process.on('error'"),
  );

  assert.match(readinessBlock, /getSyncProgress\(\)/);
  assert.doesNotMatch(readinessBlock, /getBalance\(/);
});

test('desktop lock stops the daemon without blocking on walletlock RPC', () => {
  const manager = source('src/main/services/manager-service.ts');
  const lockMethod = manager.slice(
    manager.indexOf('async lockWallet()'),
    manager.indexOf('async forceLockWallet()'),
  );

  assert.match(lockMethod, /stopWalletProcess\(\)/);
  assert.doesNotMatch(lockMethod, /walletService\.lockWallet/);
});

test('wallet selection propagates startup failure instead of entering a broken session', () => {
  const manager = source('src/main/services/manager-service.ts');
  const selectMethod = manager.slice(
    manager.indexOf('async selectWallet('),
    manager.indexOf('async create('),
  );

  assert.match(selectMethod, /await this\.startWalletProcess\(\)/);
  assert.doesNotMatch(selectMethod, /Failed to start wallet process/);
});
