import {useState} from 'react';
import {ArrowLeft, PenLine, Copy, AlertCircle, CheckCircle2} from 'lucide-react';
import {useNavigate} from 'react-router-dom';
import {getErrorMessage} from '@/lib/utils';

type Mode = 'sign' | 'verify';

const inputClasses =
  'focus:border-brand-green focus:ring-brand-green/20 w-full rounded-lg border border-gray-300 bg-white px-4 py-3 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2';

export default function SignMessage() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<Mode>('sign');
  const [address, setAddress] = useState('');
  const [message, setMessage] = useState('');
  const [signature, setSignature] = useState('');
  const [isBusy, setIsBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const switchMode = (next: Mode) => {
    setMode(next);
    setSignature('');
    setError(null);
    setSuccess(null);
  };

  // A signature shown next to edited inputs would no longer match them.
  const edit = (set: (v: string) => void) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    set(e.target.value);
    if (mode === 'sign') setSignature('');
    setSuccess(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsBusy(true);
    setError(null);
    setSuccess(null);

    try {
      if (mode === 'sign') {
        // The message is signed byte for byte, so only the address is trimmed.
        setSignature(await window.appBridge.wallet.signMessage(address.trim(), message));
        setSuccess('Message signed. Share the address, message and signature together.');
      } else {
        const valid = await window.appBridge.wallet.verifyMessage(address.trim(), signature.trim(), message);
        if (valid) {
          setSuccess('Signature is valid for this address and message.');
        } else {
          setError('Signature is NOT valid for this address and message.');
        }
      }
    } catch (err) {
      const errorMessage = getErrorMessage(err, 'Failed to process message');
      // signmessage reports a locked wallet as "address manager is locked".
      if (errorMessage.includes('is locked')) {
        setError('Wallet is locked. Redirecting to unlock screen...');
        setTimeout(() => navigate('/unlock'), 3000);
      } else {
        setError(errorMessage);
      }
    } finally {
      setIsBusy(false);
    }
  };

  const copySignature = async () => {
    await navigator.clipboard.writeText(signature);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const canSubmit = !isBusy && address.trim() && message && (mode === 'sign' || signature.trim());

  return (
    <div className="flex h-full w-full flex-col bg-transparent">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center gap-4 border-b border-gray-200 bg-white/80 p-6 shadow-sm backdrop-blur-sm">
        <button onClick={() => navigate('/wallet')} className="rounded-lg p-2 transition-colors hover:bg-gray-100">
          <ArrowLeft className="h-5 w-5 text-gray-700" />
        </button>
        <h1 className="text-xl font-semibold text-gray-900">Sign Message</h1>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-8 py-12">
        <div className="mx-auto flex max-w-md flex-col items-center">
          <div className="mb-8 text-center">
            <div className="mb-6 inline-flex h-16 w-16 items-center justify-center rounded-full bg-black">
              <PenLine className="h-8 w-8 text-white" />
            </div>
            <h2 className="mb-2 text-2xl font-bold text-gray-900">
              {mode === 'sign' ? 'Prove Address Ownership' : 'Verify a Signature'}
            </h2>
            <p className="text-gray-600">
              {mode === 'sign'
                ? 'Sign a message with one of your addresses (BIP-322)'
                : 'Check that a message was signed by an address'}
            </p>
          </div>

          {/* Mode toggle */}
          <div className="mb-6 grid w-full grid-cols-2 rounded-lg border border-gray-200 bg-white p-1">
            {(['sign', 'verify'] as const).map(m => (
              <button
                key={m}
                type="button"
                onClick={() => switchMode(m)}
                className={`rounded-md py-2 text-sm font-medium transition-colors ${
                  mode === m ? 'bg-black text-white' : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                {m === 'sign' ? 'Sign' : 'Verify'}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="w-full space-y-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">Address</label>
              <input
                value={address}
                onChange={edit(setAddress)}
                placeholder={mode === 'sign' ? 'One of your wallet addresses' : 'Signer address'}
                className={`${inputClasses} font-mono text-sm`}
                disabled={isBusy}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">Message</label>
              <textarea
                value={message}
                onChange={edit(setMessage)}
                rows={4}
                placeholder="Message to sign"
                className={inputClasses}
                disabled={isBusy}
              />
            </div>

            {(mode === 'verify' || signature) && (
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Signature (base64)</label>
                <div className="relative">
                  <textarea
                    value={signature}
                    onChange={e => setSignature(e.target.value)}
                    readOnly={mode === 'sign'}
                    rows={3}
                    className={`${inputClasses} break-all pr-12 font-mono text-sm`}
                    disabled={isBusy}
                  />
                  {mode === 'sign' && (
                    <button
                      type="button"
                      onClick={copySignature}
                      className="absolute right-3 top-3 p-1 text-gray-500 transition-colors hover:text-gray-700"
                    >
                      {copied ? <CheckCircle2 className="h-5 w-5 text-green-600" /> : <Copy className="h-5 w-5" />}
                    </button>
                  )}
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 p-4">
                <AlertCircle className="h-5 w-5 flex-shrink-0 text-red-600" />
                <span className="text-red-700">{error}</span>
              </div>
            )}

            {success && (
              <div className="flex items-center gap-3 rounded-lg border border-green-200 bg-green-50 p-4">
                <CheckCircle2 className="h-5 w-5 flex-shrink-0 text-green-600" />
                <span className="text-green-700">{success}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={!canSubmit}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-black py-4 font-semibold text-white transition-colors hover:bg-gray-800 disabled:bg-gray-300 disabled:text-gray-500"
            >
              {isBusy ? (
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              ) : (
                <PenLine className="h-5 w-5" />
              )}
              {mode === 'sign' ? 'Sign Message' : 'Verify Signature'}
            </button>
          </form>

          <div className="mt-6 rounded-lg border bg-amber-50 p-4">
            <div className="text-sm">
              <p className="mb-2 font-medium text-amber-800">Security Notice:</p>
              <ul className="space-y-1 text-amber-700">
                <li>• Signing a message never moves funds and never reveals your seed</li>
                <li>• Never paste your seed or private key into a third-party signing tool</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
