import { ArrowUpCircle } from 'lucide-react';
import { openReleasePage, useUpdateStatus } from '../hooks/useUpdateStatus';

// Small CTA rendered on the unlock screen when a non-breaking update is available.
// Major updates are handled by the persistent MajorUpgradeBanner instead, so we suppress
// this CTA in that case to avoid duplicate prompts.
export function UpgradeCta() {
  const status = useUpdateStatus();

  if (!status) return null;
  if (status.severity !== 'patch' && status.severity !== 'minor') return null;

  return (
    <button
      type="button"
      onClick={() => void openReleasePage()}
      className="mx-auto flex items-center gap-2 rounded-full border border-green-300 dark:border-green-800 bg-green-50 dark:bg-green-950/40 px-4 py-1.5 text-sm font-medium text-green-800 dark:text-green-200 transition-colors hover:bg-green-100 dark:hover:bg-green-900/60 focus:outline-none focus:ring-2 focus:ring-green-500/40"
      title={
        status.latestVersion
          ? `Pearl Wallet ${status.latestVersion} is available`
          : 'A new Pearl Wallet version is available'
      }
    >
      <ArrowUpCircle className="h-4 w-4" />
      <span>Upgrade version{status.latestVersion ? ` to ${status.latestVersion}` : ''}</span>
    </button>
  );
}
