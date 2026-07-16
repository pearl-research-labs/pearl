import {
  MessageBoxOptions,
  MessageBoxReturnValue,
  OpenDialogOptions,
  OpenDialogReturnValue,
  SaveDialogOptions,
  SaveDialogReturnValue,
} from 'electron';
import type { Transaction } from './transaction';

type PromisifyInterface<T> = {
  [K in keyof T]: T[K] extends (...args: infer A) => infer R
  ? (...args: A) => Promise<Awaited<R>>
  : never;
};

type Ipc<T> = PromisifyInterface<T>;

interface WindowApi {
  getVersion: () => string;
  minimizeWindow: () => void;
  maximizeWindow: () => void;
  closeWindow: () => void;
  isMaximized: () => boolean;
  showMessageBox: (options: MessageBoxOptions) => Promise<MessageBoxReturnValue>;
  showOpenDialog: (options: OpenDialogOptions) => Promise<OpenDialogReturnValue>;
  showSaveDialog: (options: SaveDialogOptions) => Promise<SaveDialogReturnValue>;
  openExternal: (url: string) => void;
}

interface WalletApi {
  getNewAddress: () => Promise<string>;

  unlockWallet: (passphrase: string, timeout?: number) => Promise<void>;

  lockWallet: () => Promise<void>;

  forceLockWallet: () => Promise<void>;

  changeWalletPassphrase: (currentPassword: string, newPassword: string) => Promise<void>;

  sendFromDefaultAccount: (toAddress: string, amount: number, feeRate: number) => Promise<string>;

  listAllTransactions: () => Promise<Transaction[]>;

  listTransactions: (count?: number, from?: number) => Promise<Transaction[]>;

  getBalance: (account?: string, minconf?: number) => Promise<number>;

  validateAddress: (address: string) => Promise<{ isValid: boolean }>;

  estimateFee: (numBlocks: number) => Promise<number>;
}

interface ManagerApi {
  getWalletsStats: () => { name?: string };

  selectWallet: (walletName: string) => Promise<void>;

  create: (options: { name: string; password: string }) => Promise<{ seed: string }>;

  import: (options: {
    name: string;
    seed: string;
    password?: string;
  }) => Promise<{ name: string; seed: string }>;

  getExistingWallets: () => Promise<{
    walletNames: string[];
    defaultWallet: string | undefined;
  }>;

  // Network management
  getNetworkInfo: () => Promise<{
    currentNetwork: string;
    availableNetworks: string[];
    networkConfig: {
      name: string;
      displayName: string;
      addressPrefix: string;
    };
  }>;

  setNetwork: (network: string) => Promise<{ success: boolean; network: string }>;

  getPeerSettings: () => Promise<{
    network: string;
    customPeerAddress: string;
    customPeerPort?: number;
    isCustom: boolean;
  }>;

  validatePeerAddress: (address: string, port: number) => Promise<{ valid: boolean; error?: string }>;

  setCustomPeerAddress: (address: string, port: number) => Promise<{ success: boolean }>;

  resetPeerToDefault: () => Promise<{ success: boolean }>;
}

type UpdateSeverity = 'none' | 'patch' | 'minor' | 'major';

interface UpdateStatus {
  severity: UpdateSeverity;
  localVersion: string;
  latestVersion: string | null;
  releaseUrl: string | null;
  releaseName: string | null;
  publishedAt: string | null;
  checkedAt: number | null;
  error: string | null;
}

interface UpdateApi {
  getStatus: () => Promise<UpdateStatus>;
  checkForUpdates: () => Promise<UpdateStatus>;
  openReleasePage: () => Promise<void>;
  onStatusChanged: (listener: (status: UpdateStatus) => void) => () => void;
}

type SyncPhase = 'idle' | 'headers' | 'filters' | 'blocks' | 'synced';

interface SyncProgress {
  headerHeight: number;
  filterHeaderHeight: number;
  blockHeight: number;
  bestPeerHeight: number;
  synced: boolean;
}

interface SyncApi {
  getSyncProgress: () => Promise<SyncProgress>;

  isSyncing: () => Promise<{
    syncing: boolean;
    walletHeight?: number;
    networkHeight?: number;
    error?: string;
  }>;

  waitForSync: () => Promise<
    | {
      success: true;
      synced: boolean;
      error?: undefined;
    }
    | {
      success: true;
      synced: boolean;
      error: string;
    }
  >;
}

interface AppBridge {
  window: Ipc<WindowApi>;
  wallet: Ipc<WalletApi>;
  manager: Ipc<ManagerApi>;
  sync: Ipc<SyncApi>;
  update: UpdateApi;
}

export type { AppBridge, WindowApi, WalletApi, ManagerApi, SyncApi, SyncProgress, SyncPhase, UpdateApi, UpdateStatus, UpdateSeverity, Ipc };
