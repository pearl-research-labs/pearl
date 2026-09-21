import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { AlertTriangle } from 'lucide-react';

type WalletNameFieldProps = {
  onChange: (v: string) => void;
  isChecking: boolean;
  error: string | null;
  onBlur?: () => void;
};

export default function WalletNameField({
  onChange,
  isChecking,
  error,
  onBlur,
}: WalletNameFieldProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor="walletName" className="text-gray-900 dark:text-foreground">
        Wallet Name
      </Label>
      <div className="relative">
        <Input
          id="walletName"
          type="text"
          onChange={e => onChange(e.target.value)}
          onBlur={onBlur}
          placeholder="Insert wallet name"
          className={`border-gray-300 dark:border-border bg-white dark:bg-card text-gray-900 dark:text-foreground shadow-sm ${error ? 'border-red-500 focus:border-red-500' : 'focus:border-brand-green focus:ring-brand-green/20 focus:ring-2'}`}
        />
        {isChecking && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-500/30 border-t-blue-500" />
          </div>
        )}
      </div>
      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/40 p-3 shadow-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-600 dark:text-red-400" />
          <span className="text-sm text-red-700 dark:text-red-300">{error}</span>
        </div>
      )}
    </div>
  );
}
