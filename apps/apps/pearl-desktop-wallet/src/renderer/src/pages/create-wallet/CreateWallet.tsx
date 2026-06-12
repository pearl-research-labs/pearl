import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

// Step components
import Header from './Header';
import Progress from './Progress';
import WalletSetupStep from './WalletSetupStep';
import SeedDisplay from './SeedDisplay';
import SeedVerification from './SeedVerification';
import { useWalletStore } from '../../store/walletStore';
import { setActiveWalletName, setSessionWalletSeed } from '../../services/wallet-seed';

type CreateWalletStep = 'wallet-setup' | 'seed-display' | 'seed-verification' | 'complete';

export default function CreateWallet() {
  const [step, setStep] = useState<CreateWalletStep>('wallet-setup');
  const [generatedSeed, setGeneratedSeed] = useState<string>('');
  const navigate = useNavigate();
  const { clearWalletData } = useWalletStore();

  async function handleWalletCreated(walletName: string, password: string) {
    try {
      // Clear wallet data immediately to prevent showing old wallet data during creation
      clearWalletData();
      const result = await window.appBridge.manager.create({
        name: walletName,
        password: password,
      });

      setSessionWalletSeed(walletName, result.seed);
      setActiveWalletName(walletName);
      setGeneratedSeed(result.seed);
      setStep('seed-display');
    } catch (error) {
      console.error('Failed to create wallet:', error);
      alert(`Failed to create wallet. Please try again.\n${error}`);
    }
  }

  const handleSeedConfirmed = () => {
    // Move to verification step
    setStep('seed-verification');
  };

  const handleVerificationSuccess = () => {
    // Redirect to wallet dashboard after successful verification
    navigate('/wallet');
  };

  const handleBackToSeed = () => {
    // Allow user to go back and review their seed
    setStep('seed-display');
  };

  const renderStep = () => {
    switch (step) {
      case 'wallet-setup':
        return <WalletSetupStep onCreate={handleWalletCreated} onBack={() => navigate('/')} />;
      case 'seed-display':
        return <SeedDisplay seed={generatedSeed} onConfirm={handleSeedConfirmed} />;
      case 'seed-verification':
        return (
          <SeedVerification
            seed={generatedSeed}
            onSuccess={handleVerificationSuccess}
            onBack={handleBackToSeed}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-transparent text-gray-900">
      {/* Header - Fixed */}
      <Header
        title="Create Wallet"
        onBack={step === 'seed-verification' ? handleBackToSeed : undefined}
      />

      {/* Progress indicator - Fixed */}
      <Progress current={step} />

      {/* Step content - Scrollable */}
      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4 sm:px-8 sm:pb-8">
        <div className="py -4 flex min-h-full flex-col items-center sm:py-8">
          {/* Add top spacing to ensure content stays below header */}
          <div className="h-4 flex-shrink-0 sm:h-8"></div>

          <div className="flex w-full flex-1 items-center justify-center">{renderStep()}</div>

          {/* Add bottom spacing */}
          <div className="h-4 flex-shrink-0 sm:h-8"></div>
        </div>
      </div>
    </div>
  );
}
