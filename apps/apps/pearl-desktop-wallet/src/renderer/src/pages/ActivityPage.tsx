import { ArrowLeft, ArrowUpRight, ArrowDownLeft, Copy, Check, Loader2 } from 'lucide-react';
import { Transaction } from '../../../types/transaction';
import { usePagination } from '../hooks/usePagination';
import { useWalletStore } from '../store/walletStore';
import { Button } from '@/components/ui/button';
import { formatTimeAgo, getErrorMessage } from '@/lib/utils';
import {
  isNotRelayedError,
  pendingStatusLabel,
  REBROADCAST_NOT_RELAYED_MESSAGE,
  REBROADCAST_REJECTED_DETAIL,
  REMOVE_WARNING,
} from '@/lib/pending-tx';
import { useState } from 'react';

interface ActivityPageProps {
  onBack: () => void;
}

const formatFullDate = (timestamp: number): string => {
  const date = new Date(timestamp);
  return date.toLocaleString();
};

const truncateAddress = (address: string): string => {
  if (address.length <= 12) return address;
  return `${address.slice(0, 6)}...${address.slice(-6)}`;
};

const truncateTxId = (txid: string): string => {
  if (txid.length <= 16) return txid;
  return `${txid.slice(0, 8)}...${txid.slice(-8)}`;
};

type PendingAction = 'rebroadcast' | 'remove';

interface PendingNotice {
  txid: string;
  tone: 'success' | 'warning';
  message: string;
}

export default function ActivityPage({ onBack }: ActivityPageProps) {
  const { activities, loading, hasMore, loadMore, reload } = usePagination({
    pageSize: 10,
  });
  const { syncWalletData } = useWalletStore();
  const [copiedTxId, setCopiedTxId] = useState<string | null>(null);
  const [copiedAddress, setCopiedAddress] = useState<string | null>(null);
  const [busy, setBusy] = useState<{ txid: string; action: PendingAction } | null>(null);
  const [notice, setNotice] = useState<PendingNotice | null>(null);

  // run resolves to the success notice, or null for none. The listing is
  // refetched whatever happens, and a rejected rebroadcast deletes its
  // record, so errors go to a dialog: an inline notice could have no row.
  const runPendingAction = async (
    txid: string,
    action: PendingAction,
    run: () => Promise<string | null>
  ) => {
    setBusy({ txid, action });
    setNotice(null);
    try {
      const message = await run();
      if (message) setNotice({ txid, tone: 'success', message });
    } catch (err) {
      const message = getErrorMessage(err);
      if (action === 'rebroadcast' && isNotRelayedError(message)) {
        setNotice({ txid, tone: 'warning', message: REBROADCAST_NOT_RELAYED_MESSAGE });
      } else {
        void window.appBridge.window.showMessageBox({
          type: 'error',
          title: action === 'rebroadcast' ? 'Rebroadcast failed' : 'Remove failed',
          message,
          detail: action === 'rebroadcast' ? REBROADCAST_REJECTED_DETAIL : undefined,
          buttons: ['OK'],
        });
      }
    } finally {
      setBusy(null);
      await Promise.all([reload(), syncWalletData()]);
    }
  };

  const handleRebroadcast = (txid: string) =>
    runPendingAction(txid, 'rebroadcast', async () => {
      const announced = await window.appBridge.wallet.rebroadcastTransaction(txid);
      const ancestors = announced.length - 1;
      return ancestors > 0
        ? `A peer requested the transaction and ${ancestors} pending ancestor${ancestors === 1 ? '' : 's'}.`
        : 'A peer requested the transaction.';
    });

  const handleRemove = async (txid: string) => {
    const { response } = await window.appBridge.window.showMessageBox({
      type: 'warning',
      title: 'Remove pending transaction',
      message: 'Remove this pending transaction from the wallet?',
      detail: REMOVE_WARNING,
      buttons: ['Cancel', 'Remove'],
      defaultId: 0,
      cancelId: 0,
    });
    if (response !== 1) return;

    await runPendingAction(txid, 'remove', async () => {
      await window.appBridge.wallet.removeTransaction(txid);
      return null;
    });
  };

  const handleCopyTxId = async (txid: string) => {
    try {
      await navigator.clipboard.writeText(txid);
      setCopiedTxId(txid);
      setTimeout(() => setCopiedTxId(null), 2000);
    } catch (err) {
      console.error('Failed to copy transaction ID:', err);
    }
  };

  const handleCopyAddress = async (address: string) => {
    try {
      await navigator.clipboard.writeText(address);
      setCopiedAddress(address);
      setTimeout(() => setCopiedAddress(null), 2000);
    } catch (err) {
      console.error('Failed to copy address:', err);
    }
  };

  return (
    <div className="flex h-screen w-full flex-col bg-transparent">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center gap-4 border-b border-gray-200 bg-white/80 p-6 shadow-sm backdrop-blur-sm">
        <button onClick={onBack} className="rounded-lg p-2 transition-colors hover:bg-gray-100">
          <ArrowLeft className="h-5 w-5 text-gray-700" />
        </button>
        <h1 className="text-2xl font-semibold text-gray-900">Activity</h1>
      </div>

      {/* Content - Scrollable */}
      <div className="flex-1 overflow-y-auto p-6">
        {loading && activities.length === 0 ? (
          <div className="py-12 text-center text-gray-500">
            <p>Loading activities...</p>
          </div>
        ) : activities.length === 0 ? (
          <div className="py-12 text-center text-gray-500">
            <p>No activity found</p>
          </div>
        ) : (
          /* Activity List - Scrollable */
          <div className="space-y-4">
            {activities.map((activity: Transaction, index) => (
              <div
                key={`${activity.type}_${activity.txid}_${index}`}
                className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm transition-all hover:shadow-md"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gray-100">
                      {activity.type === 'received' ? (
                        <ArrowDownLeft className="text-brand-green h-6 w-6" />
                      ) : (
                        <ArrowUpRight className="h-6 w-6 text-red-500" />
                      )}
                    </div>
                    <div>
                      <div className="text-lg font-medium text-gray-900">
                        {activity.type === 'received' ? 'Received' : 'Sent'}
                      </div>
                      <div className="text-sm text-gray-600">
                        {formatTimeAgo(activity.time)} • {formatFullDate(activity.time)}
                      </div>
                      <div className="mt-1 flex items-center gap-2">
                        <span className="text-xs text-gray-500">
                          Tx ID: {truncateTxId(activity.txid)}
                        </span>
                        <button
                          onClick={() => handleCopyTxId(activity.txid)}
                          className="rounded p-1 transition-colors hover:bg-gray-100"
                          title="Copy transaction ID"
                        >
                          {copiedTxId === activity.txid ? (
                            <Check className="text-brand-green h-3 w-3" />
                          ) : (
                            <Copy className="h-3 w-3 text-gray-400" />
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div
                      className={`text-lg font-bold ${activity.type === 'received' ? 'text-green-700' : 'text-red-500'
                        }`}
                    >
                      {activity.type === 'received' ? '+' : '-'}
                      {activity.amount} PRL
                    </div>
                    <div className="text-sm text-gray-600">
                      {activity.confirmations === 0
                        ? pendingStatusLabel(activity)
                        : `${activity.confirmations} confirmations`}
                    </div>
                    {activity.fee > 0 && (
                      <div className="text-xs text-gray-500">
                        Fee: {activity.fee.toFixed(8)} PRL
                      </div>
                    )}
                  </div>
                </div>
                {activity.confirmations === 0 && activity.type === 'sent' && (
                  <div className="mt-3 flex items-center justify-end gap-2 border-t border-gray-100 pt-3">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={busy !== null}
                      onClick={() => handleRebroadcast(activity.txid)}
                    >
                      {busy?.txid === activity.txid && busy.action === 'rebroadcast' ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        'Rebroadcast'
                      )}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={busy !== null}
                      className="text-red-600 hover:text-red-700"
                      onClick={() => handleRemove(activity.txid)}
                    >
                      {busy?.txid === activity.txid && busy.action === 'remove' ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        'Remove'
                      )}
                    </Button>
                  </div>
                )}
                {notice?.txid === activity.txid && (
                  <div
                    className={`mt-2 rounded-md px-3 py-2 text-xs ${
                      notice.tone === 'success'
                        ? 'bg-green-50 text-green-800'
                        : 'bg-amber-50 text-amber-800'
                    }`}
                  >
                    {notice.message}
                  </div>
                )}
              </div>
            ))}

            {loading && activities.length > 0 && (
              <div className="py-2 text-center text-sm text-gray-500">Loading more...</div>
            )}

            {!hasMore && activities.length > 0 && (
              <div className="py-2 text-center text-xs text-gray-400">No more activity</div>
            )}

            {hasMore ? (
              <div className="flex justify-center pt-2">
                <Button
                  onClick={loadMore}
                  disabled={loading}
                  className="bg-brand-green hover:bg-brand-green/90 px-6 text-white"
                >
                  {loading ? 'Loading…' : 'Load More'}
                </Button>
              </div>
            ) : null}

            {/* Total count indicator */}
            <div className="mt-6 border-t border-gray-200 py-4 text-center text-sm text-gray-500">
              {activities.length} transactions loaded
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
