import { AlertTriangle } from 'lucide-react';
import { openReleasePage, useUpdateStatus } from '../hooks/useUpdateStatus';

// Persistent, non-dismissable banner shown across the app when the running version is
// behind by a major release. We intentionally do not offer a "dismiss" action because a
// major bump can affect wallet functionality and we want the user to upgrade.
export function MajorUpgradeBanner() {
  const status = useUpdateStatus();

  if (!status || status.severity !== 'major') return null;

  const versionLabel = status.latestVersion ? ` (v${status.latestVersion})` : '';

  return (
    <div
      role="alert"
      className="flex items-center justify-between gap-4 border-b border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/40 px-4 py-2 text-red-900 dark:text-red-100 shadow-sm"
    >
      <div className="flex min-w-0 items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-red-600 dark:text-red-400" />
        <div className="min-w-0">
          <p className="text-sm font-semibold">
            A major Pearl Wallet update is available{versionLabel}
          </p>
          <p className="truncate text-xs text-red-800 dark:text-red-200">
            This release may change wallet functionality. You are strongly urged to upgrade to
            keep your wallet working correctly.
          </p>
        </div>
      </div>
      <button
        type="button"
        onClick={() => void openReleasePage()}
        className="flex-shrink-0 rounded-md bg-red-600 px-3 py-1.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500/40"
      >
        Upgrade now
      </button>
    </div>
  );
}
