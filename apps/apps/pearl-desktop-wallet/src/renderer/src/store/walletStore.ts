import { Transaction } from '../../../types/transaction';
import type { SyncPhase, SyncProgress } from '../../../types/app-bridge';
import { create } from 'zustand';

interface WalletState {
  walletName: string;
  balance?: number;
  availableBalance?: number;
  unconfirmedBalance?: number;
  activitiesPreview: Transaction[];
  isSyncing: boolean;
  headerHeight: number;
  filterHeaderHeight: number;
  blockHeight: number;
  bestPeerHeight: number;
  connections?: number;
  syncPhase: SyncPhase;
  isBlockchainSynced: boolean;
  syncWalletData: () => Promise<void>;
  updateSyncProgress: () => Promise<void>;
  clearWalletData: () => void;
  validateAddress: (address: string) => Promise<{ isValid: boolean }>;
}

// Derive the current phase from a raw progress response. The ordering mirrors how the
// node's SPV sync actually advances: headers first, then filter headers, then full blocks
// during birthday recovery. We trust the node's authoritative `synced` flag for the
// terminal state rather than re-deriving it from heights (headers can briefly equal the
// peer tip while block recovery is still outstanding).
export function derivePhase(p: SyncProgress): SyncPhase {
  if (p.synced) return 'synced';
  if (p.bestPeerHeight <= 0) return 'idle';
  if (p.headerHeight < p.bestPeerHeight) return 'headers';
  if (p.filterHeaderHeight < p.headerHeight) return 'filters';
  return 'blocks';
}

export const useWalletStore = create<WalletState>()((set, get) => ({
  isSyncing: false,
  walletName: 'Pearl Wallet',
  balance: undefined,
  availableBalance: undefined,
  unconfirmedBalance: undefined,
  activitiesPreview: [],
  headerHeight: 0,
  filterHeaderHeight: 0,
  blockHeight: 0,
  bestPeerHeight: 0,
  connections: undefined,
  syncPhase: 'idle',
  isBlockchainSynced: false,

  async syncWalletData() {
    const { isSyncing } = get();
    if (isSyncing) {
      return;
    }

    try {
      set({ isSyncing: true });
      const [walletsStats, totalBalance, availableBalance, transactions] = await Promise.all([
        window.appBridge.manager.getWalletsStats(),
        window.appBridge.wallet.getBalance('default', 0),
        window.appBridge.wallet.getBalance('default', 0),
        window.appBridge.wallet.listTransactions(3, 0),
      ]);
      set({ activitiesPreview: transactions });

      if (walletsStats.name) {
        set({ walletName: walletsStats.name });
      }

      const unconfirmed = totalBalance - availableBalance;
      set({ balance: totalBalance, availableBalance, unconfirmedBalance: Math.max(0, unconfirmed) });
    } catch (err) {
      console.error('❌ [useWalletStore] Failed to fetch wallet data:', err);
    } finally {
      set({ isSyncing: false });
    }
  },

  async updateSyncProgress() {
    try {
      const p = await window.appBridge.sync.getSyncProgress();
      set({
        headerHeight: p.headerHeight,
        filterHeaderHeight: p.filterHeaderHeight,
        blockHeight: p.blockHeight,
        bestPeerHeight: p.bestPeerHeight,
        connections: p.connections,
        syncPhase: derivePhase(p),
        isBlockchainSynced: p.synced,
      });
    } catch {
      // ignore errors during sync polling
    }
  },

  clearWalletData: () => {
    set(useWalletStore.getInitialState());
  },

  validateAddress: async (address: string) => {
    const validationResult = await window.appBridge.wallet.validateAddress(address);
    return validationResult;
  },
}));
