import { useCallback, useEffect, useState } from 'react';
import { Transaction } from '../../../types/transaction';

// A transaction can list several rows (a payment to several addresses), so match rows already shown by row, not txid.
const rowKey = (tx: Transaction) => `${tx.type}_${tx.txid}_${tx.address}_${tx.amount}`;

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
        const existingRows = new Set(prev.map(rowKey));
        const newRows = txs.filter(tx => !existingRows.has(rowKey(tx)));
        return [...prev, ...newRows];
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

  // Refetch everything loaded so far in one request so the list keeps its length and the user's scroll position.
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
