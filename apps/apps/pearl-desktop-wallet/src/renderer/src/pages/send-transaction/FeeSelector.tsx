import { formatPearlAmount } from '../../lib/crypto';
import { Zap, Clock, Timer } from 'lucide-react';

type FeeSelectorProps = {
  feeLevel: 'fast' | 'medium' | 'slow';
  isLoadingFees: boolean;
  currentFee: number | string;
  onSelect: (level: 'fast' | 'medium' | 'slow') => void;
};

export default function FeeSelector({
  feeLevel,
  isLoadingFees,
  currentFee,
  onSelect,
}: FeeSelectorProps) {
  return (
    <div className="space-y-3">
      <label className="text-sm text-gray-600 dark:text-muted-foreground">Transaction Fee</label>
      <div className="rounded-lg border border-gray-300 dark:border-border bg-white dark:bg-card p-4 shadow-sm">
        <div className="mb-4 grid grid-cols-3 gap-2">
          <button
            type="button"
            onClick={() => onSelect('fast')}
            className={`rounded-lg border p-3 transition-all ${feeLevel === 'fast'
              ? 'border-green-500 bg-green-500/10'
              : 'border-gray-300 dark:border-border bg-gray-50 dark:bg-muted/40 hover:border-gray-400 dark:hover:border-muted-foreground/60'
              }`}
          >
            <div className="flex flex-col items-center gap-1">
              <Zap
                className={`h-4 w-4 ${feeLevel === 'fast' ? 'text-green-400' : 'text-neutral-400'}`}
              />
              <span
                className={`text-xs font-medium ${feeLevel === 'fast' ? 'text-green-400' : 'text-neutral-300'}`}
              >
                Fast
              </span>
              <span className="text-xs text-neutral-500">~1 block</span>
            </div>
          </button>

          <button
            type="button"
            onClick={() => onSelect('medium')}
            className={`rounded-lg border p-3 transition-all ${feeLevel === 'medium'
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-gray-300 dark:border-border bg-gray-50 dark:bg-muted/40 hover:border-gray-400 dark:hover:border-muted-foreground/60'
              }`}
          >
            <div className="flex flex-col items-center gap-1">
              <Clock
                className={`h-4 w-4 ${feeLevel === 'medium' ? 'text-blue-400' : 'text-neutral-400'}`}
              />
              <span
                className={`text-xs font-medium ${feeLevel === 'medium' ? 'text-blue-400' : 'text-neutral-300'}`}
              >
                Medium
              </span>
              <span className="text-xs text-neutral-500">~10 blocks</span>
            </div>
          </button>

          <button
            type="button"
            onClick={() => onSelect('slow')}
            className={`rounded-lg border p-3 transition-all ${feeLevel === 'slow'
              ? 'border-orange-500 bg-orange-500/10'
              : 'border-gray-300 dark:border-border bg-gray-50 dark:bg-muted/40 hover:border-gray-400 dark:hover:border-muted-foreground/60'
              }`}
          >
            <div className="flex flex-col items-center gap-1">
              <Timer
                className={`h-4 w-4 ${feeLevel === 'slow' ? 'text-orange-400' : 'text-neutral-400'}`}
              />
              <span
                className={`text-xs font-medium ${feeLevel === 'slow' ? 'text-orange-400' : 'text-neutral-300'}`}
              >
                Slow
              </span>
              <span className="text-xs text-neutral-500">~25 blocks</span>
            </div>
          </button>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600 dark:text-muted-foreground">Transaction Fee / kb:</span>
          <span className="font-medium text-gray-900 dark:text-foreground">
            {isLoadingFees ? (
              <div className="border-t-brand-green h-4 w-4 animate-spin rounded-full border-2 border-gray-300 dark:border-border" />
            ) : (
              `${formatPearlAmount(currentFee)} PRL`
            )}
          </span>
        </div>
      </div>
    </div>
  );
}
