import { useCallback, useEffect, useState } from 'react';
import { Transaction } from '../../../types/transaction';

interface UsePaginationOptions {
  pageSize?: number;
}

interface UsePaginationResult {
  activities: Transaction[];
  loading: boolean;
  hasMore: boolean;
  loadMore: () => Promise<void>;
  reload: () => Promise<void>;
}

export function usePagination(options: UsePaginationOptions = {}): UsePaginationResult {
  const { pageSize = 20 } = options;

  const [activities, setActivities] = useState<Transaction[]>([]);
  const [count] = useState<number>(pageSize);
  const [offset, setOffset] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [hasMore, setHasMore] = useState<boolean>(true);

  const loadMore = useCallback(async () => {
    if (loading || !hasMore) return;
    setLoading(true);

    try {
      const txs = await window.appBridge.wallet.listTransactions(count, offset);
      setActivities(prev => {
        const existingTxids = new Set(prev.map(tx => tx.txid));
        const newTxs = txs.filter(tx => !existingTxids.has(tx.txid));
        return [...prev, ...newTxs];
      });
      setOffset(offset + count);
      if (txs.length < count) {
        setHasMore(false);
      }
    } catch (err) {
      console.error('Failed to load activities:', err);
      setHasMore(false);
    } finally {
      setLoading(false);
    }
  }, [loading, hasMore, count, offset]);

  // Refetches the pages loaded so far in one request, so a change made by
  // the user (rebroadcast, remove) shows up without losing their scroll
  // position.
  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const loaded = Math.max(offset, count);
      const txs = await window.appBridge.wallet.listTransactions(loaded, 0);
      setActivities(txs);
      setOffset(loaded);
      setHasMore(txs.length >= loaded);
    } catch (err) {
      console.error('Failed to reload activities:', err);
    } finally {
      setLoading(false);
    }
  }, [count, offset]);

  useEffect(() => {
    loadMore();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { activities, loading, hasMore, loadMore, reload };
}
