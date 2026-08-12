import { useState, useEffect } from 'react';
import { Lock, Eye, EyeOff, AlertCircle, ChevronDown, CheckCircle2 } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useWalletStore } from '../store/walletStore';
import { getErrorMessage } from '../lib/utils';
import { NetworkSelector } from '../components/NetworkSelector';
import { SettingsButton } from '../components/SettingsButton';
import { UpgradeCta } from '../components/UpgradeCta';
import { Button } from '@pearl/ui';
import {
  clearSessionWalletSeed,
  setActiveWalletName,
  setSessionWalletSeed,
} from '../services/wallet-seed';

export default function WalletUnlock() {
  const navigate = useNavigate();
  const location = useLocation();
  const { clearWalletData } = useWalletStore();
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isUnlocking, setIsUnlocking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [walletName, setWalletName] = useState<string>('');
  const [isWaitingForService, setIsWaitingForService] = useState(false);
  const [availableWallets, setAvailableWallets] = useState<string[]>([]);
  const [selectedWallet, setSelectedWallet] = useState<string>('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await window.appBridge.manager.getExistingWallets();

        if (result.walletNames) {
          setAvailableWallets(result.walletNames);

          let defaultWallet =
            location.state?.defaultWallet || result.defaultWallet || result.walletNames[0];

          setSelectedWallet(defaultWallet);
          setWalletName(defaultWallet);
          if (defaultWallet) {
            setActiveWalletName(defaultWallet);
          }
        }
      } catch (error) {
        console.error('[WalletUnlock] Error fetching wallet data:', error);
      }
    };

    fetchData();
  }, [location.state]);

  const handleWalletSelect = async (walletName: string) => {
    setSelectedWallet(walletName);
    setWalletName(walletName);
    clearSessionWalletSeed();
    setActiveWalletName(walletName);
    setIsDropdownOpen(false);
    setError(null);
  };

  const attemptUnlock = async (attempt: number = 1): Promise<void> => {
    try {
      if (attempt === 1) {
        clearWalletData();
        clearSessionWalletSeed();
        setIsWaitingForService(true);
        try {
          await window.appBridge.manager.selectWallet(walletName);
          setActiveWalletName(walletName);
        } catch (error) {
          throw new Error(getErrorMessage(error));
        }
        setIsWaitingForService(false);
      }

      try {
        const result = await window.appBridge.wallet.unlockWallet(password, 3600);
        if (result.seed) {
          setSessionWalletSeed(walletName, result.seed);
        }

        setIsUnlocking(false);
        setIsWaitingForService(false);

        navigate('/wallet');
        return;
      } catch (error) {
        console.error('❌ Failed to unlock wallet:', error);
        const errorMessage = getErrorMessage(error);

        if (
          attempt === 1 &&
          (errorMessage.includes('not running') || errorMessage.includes('not initialized'))
        ) {
          console.log('🔄 Wallet service not ready, waiting 2 seconds and retrying...');
          setIsWaitingForService(true);

          setTimeout(async () => {
            await attemptUnlock(2);
          }, 2000);
          return;
        }

        // Show user-friendly error message
        setIsUnlocking(false);
        setIsWaitingForService(false);

        if (errorMessage.includes('passphrase') || errorMessage.includes('incorrect')) {
          setError(errorMessage);
        } else if (
          errorMessage.includes('not running') ||
          errorMessage.includes('not initialized')
        ) {
          setError('Wallet service is not ready. Please try again in a moment.');
        } else {
          setError(errorMessage);
        }
      }
    } catch (err) {
      console.error('❌ Exception during wallet unlock:', err);
      setIsUnlocking(false);
      setIsWaitingForService(false);

      const errorMessage = getErrorMessage(err);

      // Check if it's a peer connection error
      if (errorMessage.includes('Please make sure your internet connection is') ||
        errorMessage.includes('Cannot resolve hostname') ||
        errorMessage.includes('Connection timeout')) {
        setError(errorMessage);
      } else if (errorMessage.includes('Failed to set current wallet')) {
        setError(errorMessage);
      } else {
        setError('An error occurred while unlocking the wallet');
      }
    }
  };

  const handleUnlock = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!password.trim()) {
      setError('Please enter your wallet password');
      return;
    }

    setIsUnlocking(true);
    setError(null);
    setIsWaitingForService(false);

    await attemptUnlock(1);
  };

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden bg-transparent">
      {/* Network Selector & Settings - Top Right */}
      <div className="absolute right-8 top-8 z-10 flex items-center gap-3">
        <NetworkSelector />
        <SettingsButton />
      </div>

      {/* Content */}
      <div className="flex min-h-0 flex-1 flex-col items-center overflow-y-auto px-4 py-8 sm:px-8 sm:py-12">
        <div className="min-h-fit w-full max-w-md flex-shrink-0 space-y-6 sm:space-y-8">
          {/* Add top spacing to ensure lock icon is always visible */}
          <div className="h-4 flex-shrink-0 sm:h-8"></div>
          {/* Lock Icon */}
          <div className="text-center">
            <div className="mb-4 inline-flex h-16 w-16 items-center justify-center rounded-full bg-gray-200 sm:mb-6">
              <Lock className="h-8 w-8 text-gray-600" />
            </div>

            {/* Wallet Selector */}
            {availableWallets.length > 1 ? (
              <div className="mb-4">
                <div className="relative">
                  <button
                    onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                    className="mx-auto flex items-center gap-2 text-gray-900 transition-colors hover:text-gray-700"
                  >
                    <span className="text-3xl font-bold">Unlock {selectedWallet}</span>
                    <ChevronDown className="h-6 w-6" />
                  </button>

                  {/* Dropdown */}
                  {isDropdownOpen && (
                    <div className="absolute left-1/2 top-full z-10 mt-2 min-w-[200px] -translate-x-1/2 transform rounded-lg border border-gray-300 bg-white shadow-lg">
                      <div className="py-2">
                        {availableWallets.map(wallet => (
                          <button
                            key={wallet}
                            onClick={() => handleWalletSelect(wallet)}
                            className="flex w-full items-center justify-between px-4 py-2 text-left transition-colors hover:bg-gray-100 focus:outline-none"
                          >
                            <span className="text-gray-900">{wallet}</span>
                            {selectedWallet === wallet && (
                              <CheckCircle2 className="h-4 w-4 text-green-500" />
                            )}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <h1 className="mb-2 text-3xl font-bold text-gray-900">
                {walletName ? `Unlock ${walletName}` : 'Wallet Locked'}
              </h1>
            )}

            <p className="text-gray-600">
              Enter your wallet password to unlock and access your funds
            </p>

            <div className="mt-4 flex justify-center">
              <UpgradeCta />
            </div>
          </div>

          {/* Unlock Form */}
          <form onSubmit={handleUnlock} className="space-y-4 sm:space-y-6">
            {/* Password Input */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">Wallet Password</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Enter your wallet password"
                  className="focus:border-brand-green focus:ring-brand-green/20 w-full rounded-lg border border-gray-300 bg-white px-4 py-3 pr-12 text-gray-900 placeholder-gray-400 shadow-sm focus:outline-none focus:ring-2"
                  disabled={isUnlocking}
                  autoFocus
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-500 transition-colors hover:text-gray-700"
                  disabled={isUnlocking}
                >
                  {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="flex items-center gap-3 rounded-lg border border-red-300 bg-red-50 p-4">
                <AlertCircle className="h-5 w-5 flex-shrink-0 text-red-600" />
                <span className="text-red-700">{error}</span>
              </div>
            )}

            {/* Unlock Button */}
            <Button
              type="submit"
              variant="default"
              disabled={isUnlocking || !password.trim()}
              className="flex h-12 w-full items-center justify-center gap-2 rounded-xl py-3 font-semibold disabled:bg-gray-300 disabled:text-gray-500"
            >
              {isUnlocking ? (
                <>
                  <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                  {isWaitingForService ? 'Starting Wallet...' : 'Unlocking...'}
                </>
              ) : (
                <>
                  <Lock className="h-5 w-5" />
                  Unlock Wallet
                </>
              )}
            </Button>
          </form>

          {/* Help Text */}
          <div className="text-center">
            <p className="text-sm text-gray-600">
              This is the same password you use to create transactions and access your wallet.
            </p>
          </div>

          {/* Divider */}
          <div className="flex items-center gap-4">
            <div className="h-px flex-1 bg-gray-300"></div>
            <span className="text-sm text-gray-500">or</span>
            <div className="h-px flex-1 bg-gray-300"></div>
          </div>

          {/* Create New Wallet Button */}
          <button
            onClick={() => {
              clearWalletData();
              navigate('/?skipCheck=true');
            }}
            className="flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-green-600 py-3 font-semibold text-white transition-colors hover:bg-green-700 sm:py-4"
            disabled={isUnlocking}
          >
            Create New Wallet
          </button>

          {/* Warning */}
          <div className="rounded-lg border border-amber-300 bg-amber-50 p-4">
            <div className="text-sm">
              <p className="mb-1 font-medium text-amber-800">Note:</p>
              <p className="text-amber-700">
                Creating a new wallet will not affect your existing wallet. You can always return to
                unlock it later.
              </p>
            </div>
          </div>

          {/* Add bottom spacing to ensure content doesn't get cut off */}
          <div className="h-4 flex-shrink-0 sm:h-8"></div>
        </div>
      </div>
    </div>
  );
}
