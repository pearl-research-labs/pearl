import { Button } from '@/components/ui/button';
import { ShieldCheck, ArrowRightLeft, BarChart3 } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { NetworkSelector } from '../components/NetworkSelector';
import { SettingsButton } from '../components/SettingsButton';
import { Logo, LogoSmall } from '@pearl/ui';

const features = [
  {
    icon: ShieldCheck,
    title: 'Secure Storage',
    description: 'Your keys never leave your device',
  },
  {
    icon: ArrowRightLeft,
    title: 'Easy Transactions',
    description: 'Send and receive Pearl with ease',
  },
  {
    icon: BarChart3,
    title: 'Track Portfolio',
    description: 'Monitor your assets in real-time',
  },
];

export default function WelcomePage() {
  const navigate = useNavigate();
  const [isCheckingWallet, setIsCheckingWallet] = useState(true);
  const [existingWalletName, setExistingWalletName] = useState<string | null>(null);

  useEffect(() => {
    // Check if user came from unlock screen (bypass auto-redirect)
    const urlParams = new URLSearchParams(window.location.hash.split('?')[1]);
    const skipCheck = urlParams.get('skipCheck');

    if (skipCheck === 'true') {
      setIsCheckingWallet(false);
    } else {
      checkForExistingWallet();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- intentionally run once on mount
  }, []);

  const checkForExistingWallet = async () => {
    try {
      console.log('[WelcomePage] Checking for existing wallets...');
      const result = await window.appBridge.manager.getExistingWallets();

      if (result.walletNames.length > 0) {
        const defaultWallet = result.defaultWallet ?? result.walletNames[0] ?? 'Unknown';
        setExistingWalletName(defaultWallet);

        // Pass wallet information to unlock screen
        navigate('/unlock', {
          state: {
            walletNames: result.walletNames,
            walletCount: result.walletNames.length,
            defaultWallet,
          },
        });
      } else {
        console.log('[WelcomePage] No existing wallet found, showing welcome screen');
      }
    } catch (error) {
      console.error('[WelcomePage] Error checking existing wallet:', error);
    } finally {
      setIsCheckingWallet(false);
    }
  };

  if (isCheckingWallet) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center bg-transparent text-gray-900">
        <div className="bg-brand-green mb-6 rounded-2xl p-4">
          <Logo className="h-10 w-10 text-white" />
        </div>
        <h1 className="mb-4 text-2xl font-bold">Pearl Wallet</h1>
        <div className="border-brand-green mb-4 h-8 w-8 animate-spin rounded-full border-2 border-t-transparent" />
        <p className="text-gray-600">
          {existingWalletName ? 'Starting wallet service...' : 'Checking for existing wallet...'}
        </p>
        {existingWalletName && (
          <p className="text-brand-green mt-2 text-sm">Found wallet: {existingWalletName}</p>
        )}
      </div>
    );
  }

  return (
    <div className="relative flex h-full w-full flex-col items-center justify-center bg-transparent p-8 text-center text-gray-900">
      {/* Network Selector & Settings - Top Right */}
      <div className="absolute right-8 top-8 flex items-center gap-3">
        <NetworkSelector />
        <SettingsButton />
      </div>

      <div className="bg-brand-green mb-4 rounded-2xl p-4">
        <LogoSmall className="h-12 w-auto" />
      </div>
      <h1 className="text-4xl font-bold">Pearl Wallet</h1>
      <p className="mt-1 text-gray-600">The official Pearl (PRL) Open Source Desktop Wallet</p>

      <div className="my-10 w-full max-w-xs space-y-6 text-left">
        {features.map(feature => (
          <div key={feature.title} className="flex items-center gap-4">
            <div className="bg-brand-light-green/20 rounded-full p-2">
              <feature.icon className="text-brand-green h-6 w-6" />
            </div>
            <div>
              <h3 className="font-semibold">{feature.title}</h3>
              <p className="text-sm text-gray-600">{feature.description}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="w-full max-w-xs space-y-3">
        <Button
          asChild
          size="lg"
          className="w-full"
        >
          <Link to="/onboarding/create">Create A New Wallet</Link>
        </Button>
        <Button
          asChild
          variant="outline"
          size="lg"
          className="w-full hover:bg-gray-100 hover:text-gray-900"
        >
          <Link to="/import-account">Restore Wallet From A Recovery Phrase</Link>
        </Button>
      </div>
    </div>
  );
}
