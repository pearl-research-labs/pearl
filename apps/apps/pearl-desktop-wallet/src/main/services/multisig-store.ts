import { app } from 'electron';
import fs from 'fs';
import path from 'path';

interface MultisigPersistedState {
  vaults: unknown[];
  pendingTxs: unknown[];
  sentTxs: unknown[];
}

function defaultState(): MultisigPersistedState {
  return {
    vaults: [],
    pendingTxs: [],
    sentTxs: [],
  };
}

function ensureStateDir(): void {
  fs.mkdirSync(path.dirname(getMultisigStatePath()), { recursive: true });
}

export function getMultisigStatePath(): string {
  return path.join(app.getPath('appData'), '@pearl', 'pearl-desktop-wallet', 'multisig-state.json');
}

export function loadMultisigState(): MultisigPersistedState {
  const filePath = getMultisigStatePath();
  if (!fs.existsSync(filePath)) {
    return defaultState();
  }

  try {
    const raw = fs.readFileSync(filePath, 'utf8');
    const parsed = JSON.parse(raw) as Partial<MultisigPersistedState>;
    return {
      vaults: Array.isArray(parsed.vaults) ? parsed.vaults : [],
      pendingTxs: Array.isArray(parsed.pendingTxs) ? parsed.pendingTxs : [],
      sentTxs: Array.isArray(parsed.sentTxs) ? parsed.sentTxs : [],
    };
  } catch (error) {
    console.error('Failed to load multisig state:', error);
    return defaultState();
  }
}

export function saveMultisigState(state: MultisigPersistedState): void {
  ensureStateDir();
  fs.writeFileSync(getMultisigStatePath(), JSON.stringify(state, null, 2), {
    encoding: 'utf8',
    mode: 0o600,
  });
}
