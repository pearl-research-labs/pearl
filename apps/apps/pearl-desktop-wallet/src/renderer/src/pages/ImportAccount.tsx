import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ArrowLeft, KeyRound, Lock, Eye, EyeOff, AlertTriangle, Loader2 } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { useWalletStore } from '../store/walletStore';
import { getErrorMessage } from '../lib/utils';
import { parseSeedInput } from '../lib/seed-input';
import PasswordStrength from './create-wallet/PasswordStrength';
import { setActiveWalletName, setSessionWalletSeed } from '../services/wallet-seed';

type ImportStep = 'seed-input' | 'passphrase-setup';

export default function ImportAccount() {
  const [step, setStep] = useState<ImportStep>('seed-input');
  const [seed, setSeed] = useState('');
  const [normalizedSeed, setNormalizedSeed] = useState('');
  const [walletName, setWalletName] = useState('');
  const [passphrase, setPassphrase] = useState('');
  const [confirmPassphrase, setConfirmPassphrase] = useState('');
  const [showPassphrase, setShowPassphrase] = useState(false);
  const [showConfirmPassphrase, setShowConfirmPassphrase] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [error, setError] = useState('');
  const [existingWallets, setExistingWallets] = useState<string[]>([]);
  const [isCheckingWalletName, setIsCheckingWalletName] = useState(false);
  const [walletNameError, setWalletNameError] = useState<string | null>(null);
  const navigate = useNavigate();
  const { clearWalletData } = useWalletStore();

  useEffect(() => {
    const fetchExistingWallets = async () => {
      try {
        const result = await window.appBridge.manager.getExistingWallets();
        setExistingWallets(result.walletNames);
      } catch (error) {
        console.error('Failed to fetch existing wallets:', error);
      }
    };

    fetchExistingWallets();
  }, []);

  useEffect(() => {
    const validateWalletName = async () => {
      if (!walletName.trim()) {
        setWalletNameError(null);
        return;
      }

      setIsCheckingWalletName(true);
      setWalletNameError(null);

      const timeoutId = setTimeout(() => {
        if (existingWallets.map(wallet => wallet.toLowerCase()).includes(walletName.trim().toLowerCase())) {
          setWalletNameError(
            `A wallet named "${walletName.trim()}" already exists. Please choose a different name.`
          );
        } else {
          setWalletNameError(null);
        }
        setIsCheckingWalletName(false);
      }, 300);

      return () => clearTimeout(timeoutId);
    };

    validateWalletName();
  }, [walletName, existingWallets]);

  const handleSeedSubmit = async () => {
    if (!walletName.trim()) {
      setError('Please enter a wallet name');
      return;
    }

    if (walletNameError) {
      setError(walletNameError);
      return;
    }

    if (isCheckingWalletName) {
      setError('Please wait while we check the wallet name...');
      return;
    }

    const parsed = await parseSeedInput(seed);
    if (parsed.kind === 'invalid') {
      setError(parsed.reason);
      return;
    }

    setNormalizedSeed(parsed.normalized);
    setError('');
    setStep('passphrase-setup');
  };

  const handlePassphraseSubmit = () => {
    if (!passphrase.trim()) {
      setError('Please enter a passphrase');
      return;
    }

    if (passphrase !== confirmPassphrase) {
      setError('Passphrases do not match');
      return;
    }

    setError('');
    setIsImporting(true);
    performWalletImport();
  };

  const performWalletImport = async () => {
    clearWalletData();

    try {
      await window.appBridge.manager.import({
        name: walletName,
        seed: normalizedSeed || seed.trim(),
        password: passphrase,
      });

      setSessionWalletSeed(walletName, normalizedSeed || seed.trim());
      setActiveWalletName(walletName);

      try {
        const result = await window.appBridge.wallet.unlockWallet(passphrase, 3600);
        if (result.seed) {
          setSessionWalletSeed(walletName, result.seed);
        }
      } catch (unlockError) {
        console.warn('⚠️ Could not unlock wallet after import:', unlockError);
      }

      navigate('/wallet');
    } catch (error) {
      console.error('Import error:', error);
      setError(getErrorMessage(error, 'An unexpected error occurred'));
      setIsImporting(false);
      setStep('passphrase-setup');
    }
  };

  const renderStep = () => {
    switch (step) {
      case 'seed-input':
        return (
          <div className="mx-auto flex w-full max-w-md flex-shrink-0 flex-col items-center py-4 sm:py-8">
            {/* Add top spacing for small screens */}
            <div className="h-4 flex-shrink-0 sm:h-8"></div>
            <div className="bg-brand-green mb-6 rounded-2xl p-4">
              <KeyRound className="h-10 w-10" />
            </div>

            <h2 className="mb-2 text-center text-xl font-bold text-gray-900 sm:text-2xl">
              Enter Your recovery phrase
            </h2>
            <p className="mb-6 text-center text-sm text-gray-600 sm:mb-8 sm:text-base">
              Enter your 12- or 24-word BIP39 mnemonic (15/18/21 also supported), or a hex seed,
              to restore your wallet
            </p>

            <div className="w-full space-y-4">
              <div>
                <Label htmlFor="walletName" className="text-gray-900">
                  Wallet Name
                </Label>
                <div className="relative">
                  <Input
                    id="walletName"
                    placeholder="My Pearl Wallet"
                    value={walletName}
                    onChange={e => setWalletName(e.target.value)}
                    className={`focus:border-brand-green focus:ring-brand-green/20 mt-2 border-gray-300 bg-white text-gray-900 placeholder-gray-400 shadow-sm focus:ring-2 ${walletNameError
                      ? 'border-red-500 focus:border-red-500 focus:ring-red-500/20'
                      : ''
                      }`}
                  />
                  {isCheckingWalletName && (
                    <div className="absolute right-3 top-1/2 -translate-y-1/2">
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-500/30 border-t-blue-500" />
                    </div>
                  )}
                </div>

                {/* Wallet Name Error */}
                {walletNameError && (
                  <div className="mt-2 flex items-start gap-2 rounded-lg border border-red-500/30 p-3">
                    <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
                    <span className="text-sm text-red-500">{walletNameError}</span>
                  </div>
                )}
              </div>

              <div>
                <Label htmlFor="seed" className="text-gray-900">
                  Recovery Phrase (12 or 24 words) or Hex Seed
                </Label>
                <textarea
                  id="seed"
                  placeholder="Enter your recovery phrase here..."
                  value={seed}
                  onChange={e => setSeed(e.target.value)}
                  className="focus:border-brand-green focus:ring-brand-green/20 mt-2 h-32 w-full resize-none rounded-lg border border-gray-300 bg-white p-3 text-gray-900 placeholder-gray-400 shadow-sm focus:outline-none focus:ring-2"
                  rows={4}
                />
              </div>

              {error && <div className="text-center text-sm text-red-400">{error}</div>}

              <Button
                onClick={handleSeedSubmit}
                className="h-12 w-full"
                disabled={
                  !seed.trim() || !walletName.trim() || !!walletNameError || isCheckingWalletName
                }
              >
                {isCheckingWalletName ? 'Checking wallet name...' : 'Continue'}
              </Button>
            </div>

            {/* Add bottom spacing */}
            <div className="h-4 flex-shrink-0 sm:h-8"></div>
          </div>
        );

      case 'passphrase-setup':
        return (
          <div className="mx-auto flex w-full max-w-md flex-shrink-0 flex-col items-center py-4 sm:py-8">
            {/* Add top spacing for small screens */}
            <div className="h-4 flex-shrink-0 sm:h-8"></div>
            <div className="mb-6 rounded-2xl bg-amber-600 p-4">
              <Lock className="h-10 w-10 text-white" />
            </div>

            <h2 className="mb-2 text-center text-xl font-bold sm:text-2xl">
              Set Wallet Passphrase
            </h2>
            <p className="mb-6 text-center text-sm text-gray-600 sm:mb-8 sm:text-base">
              Create a secure passphrase to encrypt and protect your wallet
            </p>

            <div className="w-full space-y-4">
              <div>
                <Label htmlFor="passphrase" className="text-gray-900">
                  Wallet Passphrase
                </Label>
                <div className="relative">
                  <Input
                    id="passphrase"
                    type={showPassphrase ? 'text' : 'password'}
                    placeholder="Enter a secure passphrase..."
                    value={passphrase}
                    onChange={e => setPassphrase(e.target.value)}
                    className="focus:border-brand-green focus:ring-brand-green/20 mt-2 border-gray-300 bg-white pr-12 text-gray-900 placeholder-gray-400 shadow-sm focus:ring-2"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassphrase(!showPassphrase)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-500 transition-colors hover:text-gray-700"
                  >
                    {showPassphrase ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                  </button>
                </div>
                {passphrase && (
                  <div className="mt-3">
                    <PasswordStrength password={passphrase} />
                  </div>
                )}
              </div>

              <div>
                <Label htmlFor="confirmPassphrase" className="text-gray-900">
                  Confirm Passphrase
                </Label>
                <div className="relative">
                  <Input
                    id="confirmPassphrase"
                    type={showConfirmPassphrase ? 'text' : 'password'}
                    placeholder="Confirm your passphrase..."
                    value={confirmPassphrase}
                    onChange={e => setConfirmPassphrase(e.target.value)}
                    className="focus:border-brand-green focus:ring-brand-green/20 mt-2 border-gray-300 bg-white pr-12 text-gray-900 placeholder-gray-400 shadow-sm focus:ring-2"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassphrase(!showConfirmPassphrase)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-500 transition-colors hover:text-gray-700"
                  >
                    {showConfirmPassphrase ? (
                      <EyeOff className="h-5 w-5" />
                    ) : (
                      <Eye className="h-5 w-5" />
                    )}
                  </button>
                </div>
              </div>

              {error && <div className="text-center text-sm text-red-400">{error}</div>}

              <div className="flex gap-3">
                <Button
                  onClick={() => setStep('seed-input')}
                  variant="outline"
                  className="flex-1 border-gray-300 bg-white text-gray-700 shadow-sm hover:border-gray-400 hover:bg-gray-50 hover:text-gray-900"
                >
                  Back
                </Button>
                <Button
                  onClick={handlePassphraseSubmit}
                  className="flex-1"
                  disabled={isImporting || !passphrase.trim() || !confirmPassphrase.trim()}
                >
                  {isImporting ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Importing...
                    </>
                  ) : (
                    'Import Wallet'
                  )}
                </Button>
              </div>
            </div>

            <div className="mt-6 rounded-lg border border-amber-300 bg-amber-50 p-4 shadow-sm">
              <p className="text-center text-sm text-amber-800">
                <strong>Important:</strong> Remember this passphrase! You'll need it to unlock your
                wallet and make transactions.
              </p>
            </div>

            {/* Add bottom spacing */}
            <div className="h-4 flex-shrink-0 sm:h-8"></div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-transparent text-gray-900">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center gap-4 px-4 py-4 sm:px-8 sm:py-6">
        <Button
          variant="ghost"
          size="sm"
          asChild
          className="text-gray-700 hover:bg-gray-100 hover:text-gray-900"
        >
          <Link to="/">
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
        <h1 className="text-xl font-semibold">
          {step === 'seed-input' ? 'Import Wallet' : 'Set Passphrase'}
        </h1>
      </div>

      {/* Main Content with Scroll */}
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-4 pb-4 sm:px-8 sm:pb-8">
        {renderStep()}
      </div>
    </div>
  );
}
