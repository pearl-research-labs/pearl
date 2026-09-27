const ACTIVE_WALLET_KEY = "pearl-desktop-wallet.active-wallet-name";
const WALLET_SEED_KEY_PREFIX = "pearl-desktop-wallet.wallet-seed.";
const LATEST_WALLET_SEED_KEY = "pearl-desktop-wallet.latest-wallet-seed";

let sessionWalletName: string | null = null;
let sessionWalletSeed: string | null = null;

function normalize(value: string): string {
  return value.trim();
}

function seedStorageKey(walletName: string): string {
  return `${WALLET_SEED_KEY_PREFIX}${normalize(walletName).toLowerCase()}`;
}

function readStoredSeed(walletName: string): string | null {
  if (typeof localStorage === "undefined") return null;
  const key = seedStorageKey(walletName);
  const value = localStorage.getItem(key);
  return value && value.trim() ? value.trim() : null;
}

function storeSeed(walletName: string, seed: string): void {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(seedStorageKey(walletName), seed);
  localStorage.setItem(LATEST_WALLET_SEED_KEY, seed);
}

function readLatestStoredSeed(): string | null {
  if (typeof localStorage === "undefined") return null;
  const value = localStorage.getItem(LATEST_WALLET_SEED_KEY);
  return value && value.trim() ? value.trim() : null;
}

function readAnyStoredSeed(): string | null {
  if (typeof localStorage === "undefined") return null;
  const seeds: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (!key || !key.startsWith(WALLET_SEED_KEY_PREFIX)) continue;
    const value = localStorage.getItem(key);
    if (value && value.trim()) {
      seeds.push(value.trim());
    }
  }
  if (seeds.length === 1) {
    return seeds[0] ?? null;
  }
  return null;
}

export function setActiveWalletName(walletName: string): void {
  const name = normalize(walletName);
  if (!name) return;
  sessionWalletName = name;
  if (typeof localStorage !== "undefined") {
    localStorage.setItem(ACTIVE_WALLET_KEY, name);
  }
  if (!sessionWalletSeed) {
    sessionWalletSeed = readStoredSeed(name) ?? readLatestStoredSeed() ?? readAnyStoredSeed();
  }
}

export function getActiveWalletName(): string | null {
  if (sessionWalletName && sessionWalletName.trim()) {
    return sessionWalletName.trim();
  }
  if (typeof localStorage === "undefined") return null;
  const value = localStorage.getItem(ACTIVE_WALLET_KEY);
  return value && value.trim() ? value.trim() : null;
}

export function setSessionWalletSeed(walletName: string, seed: string): void {
  const name = normalize(walletName);
  const value = normalize(seed);
  if (!name || !value) return;
  sessionWalletName = name;
  sessionWalletSeed = value;
  storeSeed(name, value);
  if (typeof localStorage !== "undefined") {
    localStorage.setItem(ACTIVE_WALLET_KEY, name);
  }
}

export function clearSessionWalletSeed(): void {
  sessionWalletName = null;
  sessionWalletSeed = null;
}

export function getActiveWalletSeed(): string | null {
  if (sessionWalletSeed && sessionWalletSeed.trim()) {
    return sessionWalletSeed.trim();
  }
  const activeWallet = getActiveWalletName();
  if (!activeWallet) return null;
  const stored = readStoredSeed(activeWallet) ?? readLatestStoredSeed() ?? readAnyStoredSeed();
  if (stored) {
    sessionWalletName = activeWallet;
    sessionWalletSeed = stored;
    return stored;
  }
  return sessionWalletSeed;
}

export function rememberWalletSeed(walletName: string, seed: string): void {
  setSessionWalletSeed(walletName, seed);
}

export function clearWalletSeed(): void {
  clearSessionWalletSeed();
}
