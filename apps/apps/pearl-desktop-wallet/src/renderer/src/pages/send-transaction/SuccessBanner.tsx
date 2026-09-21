import { CheckCircle2, Copy } from 'lucide-react';
import { useState } from 'react';

type SuccessBannerProps = {
  message: string;
  txid: string;
  formatTxid: (txid: string) => string;
};

export default function SuccessBanner({ message, txid, formatTxid }: SuccessBannerProps) {
  const [copiedTxid, setCopiedTxid] = useState(false);

  async function copyTxidToClipboard(txid: string) {
    try {
      await navigator.clipboard.writeText(txid);
      setCopiedTxid(true);
      setTimeout(() => setCopiedTxid(false), 2000);
    } catch (err) {
      console.error('Failed to copy transaction ID:', err);
    }
  }

  return (
    <div className="rounded-lg border-2 border-green-500 bg-green-50 dark:bg-green-950/40 p-4 shadow-md">
      <div className="mb-3 flex items-center gap-3">
        <CheckCircle2 className="h-6 w-6 flex-shrink-0 text-green-600 dark:text-green-400" />
        <span className="text-lg font-semibold text-green-900 dark:text-green-100">{message}</span>
      </div>
      <div className="flex items-center justify-between rounded-lg border border-gray-200 dark:border-border bg-white dark:bg-card p-3 shadow-sm">
        <div className="flex min-w-0 flex-col">
          <span className="mb-1 text-xs text-gray-600 dark:text-muted-foreground">Transaction ID</span>
          <span className="font-mono text-sm text-gray-900 dark:text-foreground">{formatTxid(txid)}</span>
        </div>
        <button
          onClick={(e) => {
            e.preventDefault();
            copyTxidToClipboard(txid);
          }}
          className="ml-3 flex-shrink-0 rounded-lg p-2 transition-colors hover:bg-gray-100 dark:hover:bg-muted"
        >
          {copiedTxid ? (
            <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400" />
          ) : (
            <Copy className="h-5 w-5 text-gray-600 dark:text-muted-foreground" />
          )}
        </button>
      </div>
    </div>
  );
}
