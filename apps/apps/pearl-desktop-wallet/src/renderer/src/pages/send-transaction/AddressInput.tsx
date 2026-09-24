import { AlertCircle } from 'lucide-react';

type AddressInputProps = {
  address: string;
  onChange: (value: string) => void;
  error?: string | null;
  onBlur?: () => void;
};

export default function AddressInput({ address, onChange, error, onBlur }: AddressInputProps) {
  return (
    <div className="space-y-2">
      <label className="text-sm text-neutral-400">Recipient Address</label>
      <input
        type="text"
        value={address}
        onChange={e => onChange(e.target.value)}
        onBlur={onBlur}
        placeholder="Insert a recpipient address"
        className="focus:border-brand-green focus:ring-brand-green/20 w-full rounded-lg border border-gray-300 dark:border-border bg-white dark:bg-card px-4 py-3 font-mono text-sm text-gray-900 dark:text-foreground placeholder-gray-400 dark:placeholder-muted-foreground/70 shadow-sm focus:outline-none focus:ring-2"
      />
      {error && (
        <div className="mt-1 flex items-center gap-1 rounded border border-red-700/30 px-2 py-1 text-xs text-red-400">
          <AlertCircle className="h-3 w-3 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
