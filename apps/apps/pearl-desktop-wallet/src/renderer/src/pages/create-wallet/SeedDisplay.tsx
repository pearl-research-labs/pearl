import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { AlertTriangle, CheckCircle, Copy, Key } from 'lucide-react';
import { useState } from 'react';

type SeedDisplayProps = {
  seed: string;
  onConfirm: () => void;
};

export default function SeedDisplay({ seed, onConfirm }: SeedDisplayProps) {
  const [isCopied, setIsCopied] = useState(false);

  const handleCopySeed = async () => {
    try {
      await navigator.clipboard.writeText(seed);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy seed to clipboard:', error);
      const textArea = document.createElement('textarea');
      textArea.value = seed;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-md flex-col items-center">
      <div className="mb-4 rounded-2xl bg-amber-600 p-3 sm:mb-6 sm:p-4">
        <Key className="h-8 w-8 text-white sm:h-10 sm:w-10" />
      </div>

      <h2 className="mb-2 text-center text-xl font-bold text-gray-900 dark:text-foreground sm:text-2xl">
        Your Wallet Seed
      </h2>
      <p className="mb-4 text-center text-sm text-gray-600 dark:text-muted-foreground sm:mb-6 sm:text-base">
        This is your wallet's seed phrase. Keep it safe and secure!
      </p>

      <Card className="mb-4 w-full border-gray-300 dark:border-border bg-white dark:bg-card shadow-sm sm:mb-6">
        <CardContent className="p-4 sm:p-6">
          <div className="mb-4 rounded-lg border border-gray-200 dark:border-border bg-gray-50 dark:bg-muted/40 p-4">
            <p className="text-brand-green break-words font-mono text-sm leading-relaxed">{seed}</p>
          </div>

          <Button
            onClick={handleCopySeed}
            variant="outline"
            className={`w-full transition-all duration-200 ${isCopied
              ? 'border-brand-green bg-brand-light-green/10 '
              : 'border-gray-300 dark:border-border bg-gray-100 dark:bg-muted'
              }`}
          >
            {isCopied ? (
              <>
                <CheckCircle className="mr-2 h-4 w-4" />
                Copied!
              </>
            ) : (
              <>
                <Copy className="mr-2 h-4 w-4" />
                Copy Seed
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      <div className="mb-4 rounded-lg border border-amber-300 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 p-3 shadow-sm sm:mb-6 sm:p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-600 dark:text-amber-400" />
          <div className="text-sm">
            <p className="mb-1 font-medium text-amber-900 dark:text-amber-100">Important:</p>
            <p className="text-amber-700 dark:text-amber-300">
              Store this seed securely. Anyone with access to it can control your wallet and funds.
            </p>
          </div>
        </div>
      </div>

      <Button
        onClick={onConfirm}
        className="h-12 w-full text-base shadow-sm"
      >
        I've Secured My Seed
      </Button>
    </div>
  );
}
