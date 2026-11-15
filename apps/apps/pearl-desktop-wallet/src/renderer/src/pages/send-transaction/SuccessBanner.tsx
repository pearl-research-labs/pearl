import {CheckCircle2, Copy} from 'lucide-react';
import {useState} from 'react';

type SuccessBannerProps = {
  message: string;
  txid: string;
  formatTxid: (txid: string) => string;
};

export default function SuccessBanner({message, txid, formatTxid}: SuccessBannerProps) {
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
    <div className="rounded-lg border-2 border-green-500 bg-green-50 p-4 shadow-md">
      <div className="mb-3 flex items-center gap-3">
        <CheckCircle2 className="h-6 w-6 flex-shrink-0 text-green-600" />
        <span className="text-lg font-semibold text-green-900">{message}</span>
      </div>
      <p className="mb-3 text-sm text-green-900">
        If needed, open Activity and choose Rebroadcast on this payment. Creating another payment could pay the
        recipient twice.
      </p>
      <div className="flex items-center justify-between rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
        <div className="flex min-w-0 flex-col">
          <span className="mb-1 text-xs text-gray-600">Transaction ID</span>
          <span className="font-mono text-sm text-gray-900">{formatTxid(txid)}</span>
        </div>
        <button
          onClick={e => {
            e.preventDefault();
            copyTxidToClipboard(txid);
          }}
          className="ml-3 flex-shrink-0 rounded-lg p-2 transition-colors hover:bg-gray-100"
        >
          {copiedTxid ? (
            <CheckCircle2 className="h-5 w-5 text-green-600" />
          ) : (
            <Copy className="h-5 w-5 text-gray-600" />
          )}
        </button>
      </div>
    </div>
  );
}
