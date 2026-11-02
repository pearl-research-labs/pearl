import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { useWalletStore } from './store/walletStore';

const UNLOCKED_ROUTES = ['/wallet', '/send', '/receive', '/activity', '/change-password', '/sign-message'];

function SyncWallet() {
  const { syncWalletData, updateSyncProgress, isBlockchainSynced } = useWalletStore();
  const { pathname } = useLocation();
  const dataIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const syncIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isUnlocked = UNLOCKED_ROUTES.includes(pathname);

  useEffect(() => {
    if (!isUnlocked) {
      if (dataIntervalRef.current) {
        clearInterval(dataIntervalRef.current);
        dataIntervalRef.current = null;
      }
      if (syncIntervalRef.current) {
        clearInterval(syncIntervalRef.current);
        syncIntervalRef.current = null;
      }
      return;
    }

    syncWalletData();
    updateSyncProgress();

    dataIntervalRef.current = setInterval(() => {
      syncWalletData();
    }, 10000);

    syncIntervalRef.current = setInterval(() => {
      updateSyncProgress();
    }, 5000);

    return () => {
      if (dataIntervalRef.current) {
        clearInterval(dataIntervalRef.current);
        dataIntervalRef.current = null;
      }
      if (syncIntervalRef.current) {
        clearInterval(syncIntervalRef.current);
        syncIntervalRef.current = null;
      }
    };
  }, [isUnlocked]);

  useEffect(() => {
    if (!isUnlocked) return;

    if (isBlockchainSynced) {
      if (syncIntervalRef.current) {
        clearInterval(syncIntervalRef.current);
        syncIntervalRef.current = null;
      }
      syncWalletData();
    } else if (!syncIntervalRef.current) {
      updateSyncProgress();
      syncIntervalRef.current = setInterval(() => {
        updateSyncProgress();
      }, 5000);
    }
  }, [isBlockchainSynced, isUnlocked]);

  return null;
}

export { SyncWallet };
