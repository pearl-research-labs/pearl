import { useState, useEffect } from 'react';
import { ArrowLeft, Copy, CheckCircle2, AlertTriangle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useWalletStore } from '../store/walletStore';
import { QrcodeCanvas } from 'react-qrcode-pretty';

export default function ReceiveTransaction() {
  const navigate = useNavigate();
  const { walletName } = useWalletStore();
  const [receiveAddress, setReceiveAddress] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copiedAddress, setCopiedAddress] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchReceiveAddress = async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      console.log('🔍 Fetching new receive address...');
      // Get a new receiving address using getnewaddress RPC
      const newAddress = await window.appBridge.wallet.getNewAddress();

      console.log('✅ Got receive address:', newAddress);
      setReceiveAddress(newAddress);
    } catch (err) {
      console.error('❌ Exception while fetching receive address:', err);
      const message = err instanceof Error ? err.message : 'Unable to generate address';
      setReceiveAddress('');
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedAddress(text);
      setTimeout(() => setCopiedAddress(null), 2000);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  };

  useEffect(() => {
    fetchReceiveAddress();
  }, []);

  return (
    <div className="flex h-full w-full flex-col bg-transparent">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center gap-4 border-b border-gray-200 dark:border-border bg-white/80 dark:bg-card/80 p-6 shadow-sm backdrop-blur-sm">
        <button
          onClick={() => navigate('/wallet')}
          className="rounded-lg p-2 transition-colors hover:bg-gray-100 dark:hover:bg-muted"
        >
          <ArrowLeft className="h-5 w-5 text-gray-700 dark:text-foreground/90" />
        </button>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-foreground">Receive Pearl</h1>
      </div>

      {/* Content - Scrollable */}
      <div className="flex-1 overflow-y-auto px-8 py-12">
        <div className="mx-auto flex max-w-md flex-col items-center">
          {/* Wallet Info */}
          <div className="mb-8 text-center">
            <h2 className="mb-2 text-2xl font-bold text-gray-900 dark:text-foreground">Receive to {walletName}</h2>
            <p className="text-gray-600 dark:text-muted-foreground">Share this address or QR code to receive Pearl tokens</p>
          </div>

          {isLoading ? (
            <div className="py-12 text-center text-gray-600 dark:text-muted-foreground">
              <div className="border-brand-green mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-t-transparent" />
              <p>Generating receive address...</p>
            </div>
          ) : errorMessage || !receiveAddress ? (
            <div className="w-full max-w-sm rounded-xl border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/40 p-6 text-center">
              <AlertTriangle className="mx-auto mb-3 h-8 w-8 text-red-600 dark:text-red-400" />
              <p className="mb-1 text-base font-semibold text-red-900 dark:text-red-100">
                Unable to generate an address
              </p>
              <p className="mb-4 text-sm text-red-800 dark:text-red-200">
                {errorMessage ?? 'The wallet did not return an address.'}
              </p>
              <p className="mb-4 text-xs text-red-700 dark:text-red-300">
                This often happens while the wallet is scanning blocks for your transactions.
                Please try again once that step finishes.
              </p>
              <button
                type="button"
                onClick={() => void fetchReceiveAddress()}
                className="rounded-md bg-red-600 px-4 py-1.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500/40"
              >
                Retry
              </button>
            </div>
          ) : (
            <>
              <QrcodeCanvas
                value={receiveAddress}
                size={200}
                padding={10}
                margin={10}
                bgColor="#ffffff"
                bgRounded
                level="M"
                variant="fluid"
                divider
              />

              {/* Address Display */}
              <div className="mt-4 rounded-xl border border-gray-200 dark:border-border bg-white dark:bg-card p-4 shadow-sm">
                <div className="mb-2 text-sm text-gray-600 dark:text-muted-foreground">Your Receive Address</div>
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="break-all font-mono text-sm text-gray-900 dark:text-foreground">
                      {receiveAddress}
                    </div>
                  </div>
                  <button
                    onClick={() => copyToClipboard(receiveAddress)}
                    className="flex-shrink-0 rounded-lg p-2 transition-colors hover:bg-gray-100 dark:hover:bg-muted"
                  >
                    {copiedAddress === receiveAddress ? (
                      <CheckCircle2 className="text-brand-green h-5 w-5" />
                    ) : (
                      <Copy className="h-5 w-5 text-gray-600 dark:text-muted-foreground" />
                    )}
                  </button>
                </div>
              </div>

              {/* Instructions */}
              <div className="border-brand-summer-sky/30 bg-brand-summer-sky/10 mt-4 rounded-lg border p-4">
                <div className="text-sm">
                  <p className="mb-2 font-medium text-gray-900 dark:text-foreground">How to receive Pearl Tokens:</p>
                  <ul className="space-y-1 text-gray-700 dark:text-foreground/90">
                    <li>• Share this address with the sender</li>
                    <li>• Or let them scan the QR code</li>
                    <li>• Transactions will appear in your Activity</li>
                    <li>• Wait for confirmations before considering funds received</li>
                  </ul>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
