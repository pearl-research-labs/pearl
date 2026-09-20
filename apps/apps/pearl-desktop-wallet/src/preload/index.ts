import { contextBridge, ipcRenderer } from 'electron';
import {
  AppBridge,
  Ipc,
  WindowApi,
  WalletApi,
  ManagerApi,
  SyncApi,
  UpdateApi,
  UpdateStatus,
} from '../types/app-bridge';

const windowIpc: Ipc<WindowApi> = {
  getVersion: () => ipcRenderer.invoke('app-version'),
  minimizeWindow: () => ipcRenderer.invoke('window-minimize'),
  maximizeWindow: () => ipcRenderer.invoke('window-maximize'),
  closeWindow: () => ipcRenderer.invoke('window-close'),
  isMaximized: () => ipcRenderer.invoke('window-is-maximized'),
  showMessageBox: options => ipcRenderer.invoke('show-message-box', options),
  showOpenDialog: options => ipcRenderer.invoke('show-open-dialog', options),
  showSaveDialog: options => ipcRenderer.invoke('show-save-dialog', options),
  openExternal: url => ipcRenderer.invoke('open-external', url),
};

const walletIpc: Ipc<WalletApi> = {
  getNewAddress: () => ipcRenderer.invoke('wallet-get-new-address'),
  unlockWallet: (passphrase, timeout) => ipcRenderer.invoke('wallet-unlock', passphrase, timeout),
  deriveMultisigKey: (vaultAccount, keyIndex) =>
    ipcRenderer.invoke('wallet-derive-multisig-key', vaultAccount, keyIndex),
  lockWallet: () => ipcRenderer.invoke('wallet-lock'),
  forceLockWallet: () => ipcRenderer.invoke('wallet-force-lock'),
  changeWalletPassphrase: (currentPassword, newPassword) =>
    ipcRenderer.invoke('wallet-change-password', currentPassword, newPassword),
  sendFromDefaultAccount: (toAddress: string, amount: number, feeRate: number) =>
    ipcRenderer.invoke('wallet-send-from-default-account', toAddress, amount, feeRate),
  listAllTransactions: () => ipcRenderer.invoke('wallet-list-all-transactions'),
  listTransactions: (count, from) => ipcRenderer.invoke('wallet-list-transactions', count, from),
  getBalance: (account, minconf) => ipcRenderer.invoke('wallet-get-balance', account, minconf),
  getVaultBalance: address => ipcRenderer.invoke('wallet-get-vault-balance', address),
  getVaultUtxos: address => ipcRenderer.invoke('wallet-get-vault-utxos', address),
  broadcastPearlTx: rawHex => ipcRenderer.invoke('wallet-broadcast-pearl-tx', rawHex),
  getTransactionInfo: txid => ipcRenderer.invoke('wallet-get-transaction-info', txid),
  validateAddress: address => ipcRenderer.invoke('wallet-validate-address', address),
  estimateFee: numBlocks => ipcRenderer.invoke('wallet-estimate-fee', numBlocks),
  getMultisigState: () => ipcRenderer.invoke('wallet-get-multisig-state'),
  saveMultisigState: state => ipcRenderer.invoke('wallet-save-multisig-state', state),
};

const managerIpc: Ipc<ManagerApi> = {
  getWalletsStats: () => ipcRenderer.invoke('get-wallets-stats'),
  selectWallet: walletName => ipcRenderer.invoke('select-wallet', walletName),
  create: options => ipcRenderer.invoke('wallet-create', options),
  import: options => ipcRenderer.invoke('wallet-import', options),
  getExistingWallets: () => ipcRenderer.invoke('get-existing-wallets'),
  getNetworkInfo: () => ipcRenderer.invoke('get-network-info'),
  setNetwork: network => ipcRenderer.invoke('set-network', network),
  getPeerSettings: () => ipcRenderer.invoke('get-peer-settings'),
  validatePeerAddress: (address, port) => ipcRenderer.invoke('validate-peer', address, port),
  setCustomPeerAddress: (address, port) => ipcRenderer.invoke('set-custom-peer', address, port),
  resetPeerToDefault: () => ipcRenderer.invoke('reset-peer-to-default'),
};

const syncIpc: Ipc<SyncApi> = {
  getSyncProgress: () => ipcRenderer.invoke('sync-get-progress'),
  isSyncing: () => ipcRenderer.invoke('sync-is-syncing'),
  waitForSync: () => ipcRenderer.invoke('sync-wait-for-sync'),
};

const updateIpc: UpdateApi = {
  getStatus: () => ipcRenderer.invoke('update-get-status'),
  checkForUpdates: () => ipcRenderer.invoke('update-check'),
  openReleasePage: () => ipcRenderer.invoke('update-open-release-page'),
  onStatusChanged: listener => {
    const handler = (_event: unknown, status: UpdateStatus) => listener(status);
    ipcRenderer.on('update-status-changed', handler);
    return () => {
      ipcRenderer.removeListener('update-status-changed', handler);
    };
  },
};

const appBridge: AppBridge = {
  window: windowIpc,
  wallet: walletIpc,
  manager: managerIpc,
  sync: syncIpc,
  update: updateIpc,
};

contextBridge.exposeInMainWorld('appBridge', appBridge);
