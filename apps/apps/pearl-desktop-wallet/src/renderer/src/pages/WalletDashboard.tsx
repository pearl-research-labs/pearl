import { useState, type ReactNode } from 'react';
import { Copy, CheckCircle2, ArrowUpRight, ArrowDownLeft, Lock, Key, Loader2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useWalletStore } from '../store/walletStore';
import { formatPearlAmount } from '../lib/crypto';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../components/ui/tooltip';

export default function WalletDashboard() {
  const navigate = useNavigate();
  const {
    walletName,
    balance,
    activitiesPreview,
    headerHeight,
    filterHeaderHeight,
    blockHeight,
    bestPeerHeight,
    syncPhase,
    isBlockchainSynced,
  } = useWalletStore();
  const { clearWalletData } = useWalletStore();
  const [copiedAddress, setCopiedAddress] = useState<string | null>(null);
  const [walletAddress] = useState<string>('');

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedAddress(text);
      setTimeout(() => setCopiedAddress(null), 2000);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  };

  const handleLockWallet = async () => {
    clearWalletData();

    try {
      // During birthday recovery the Go wallet holds a bolt write txn for the
      // duration of the current 2000-block batch, which makes the polite
      // `walletlock` RPC hang for up to a minute. Fall back to a force-stop
      // (SIGKILL) so the UI stays responsive. bbolt commits are atomic at txn
      // boundaries, so a mid-batch kill just replays the batch on next open.
      if (syncPhase === 'blocks') {
        await window.appBridge.wallet.forceLockWallet();
      } else {
        await window.appBridge.wallet.lockWallet();
      }
      console.log('✅ Wallet locked successfully');
    } catch (err) {
      console.error('❌ Exception during wallet lock:', err);
    } finally {
      // Always navigate — the Go process is either dead or actively dying.
      // Leaving the user on a blank dashboard would be worse than showing the
      // unlock screen while cleanup finishes in the background.
      navigate('/unlock');
    }
  };

  const formatTimeAgo = (timestamp: number): string => {
    const now = Date.now();
    const diff = now - timestamp;

    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (minutes < 60) {
      return `${minutes}m ago`;
    } else if (hours < 24) {
      return `${hours}h ago`;
    } else {
      return `${days}d ago`;
    }
  };

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-transparent">
      {/* Main Content Area */}
      <div className="flex min-h-0 flex-1 flex-col items-center overflow-y-auto px-4 py-6 sm:px-6 sm:py-8">
        <div className="w-full max-w-md flex-shrink-0 space-y-6 sm:space-y-8">
          {/* Add top spacing for small screens */}
          <div className="h-4 flex-shrink-0 sm:h-8"></div>

          {/* Wallet Name */}
          <div className="text-center">
            <h1 className="text-2xl font-medium text-gray-900">{walletName}</h1>
          </div>

          {/* Address Display - only show if wallet name is default */}
          {walletAddress && walletName === 'Pearl Wallet' && (
            <div className="flex items-center justify-center gap-3 text-gray-600">
              <span className="font-mono text-base" title={`Full address: ${walletAddress}`}>
                {walletAddress.slice(0, 6)}...{walletAddress.slice(-4)}
              </span>
              <button
                onClick={() => copyToClipboard(walletAddress)}
                className="p-1.5 transition-colors hover:text-gray-900"
              >
                {copiedAddress === walletAddress ? (
                  <CheckCircle2 className="text-brand-green h-4 w-4" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </button>
            </div>
          )}

          {/* Balance Section */}
          <div className="space-y-2 text-center">
            <div className="text-base text-gray-600">Total Balance</div>
            {typeof balance === 'number' ? (
              <div className="text-4xl font-bold text-gray-900 sm:text-5xl text-nowrap">
                {formatPearlAmount(balance)} PRL
              </div>
            ) : (
              <Loader2 className="mx-auto h-6 w-6 animate-spin text-gray-900 sm:h-8 sm:w-8" />
            )}

            {!isBlockchainSynced && bestPeerHeight > 0 && (
              <div className="mt-3 space-y-3">
                <SyncStage
                  label="Downloading block headers"
                  current={Math.min(Math.max(headerHeight, filterHeaderHeight), bestPeerHeight)}
                  total={bestPeerHeight}
                  active={syncPhase === 'headers' || syncPhase === 'filters'}
                  done={syncPhase === 'blocks'}
                />
                <SyncStage
                  label="Scanning blocks for your transactions"
                  current={Math.min(blockHeight, headerHeight || bestPeerHeight)}
                  total={headerHeight || bestPeerHeight}
                  active={syncPhase === 'blocks'}
                  done={false}
                  note={
                    syncPhase === 'blocks'
                      ? 'This step scans each relevant block. The wallet is temporarily unavailable until it finishes.'
                      : 'Starts after headers finish downloading'
                  }
                />
                <p className="text-xs text-amber-600">
                  Balance may not reflect all transactions until sync is complete
                </p>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          {/*
            Wallet RPCs that hit the bolt write path (getnewaddress, sendmany, change
            password) block on the recovery write txn held only during block scanning and
            time out after 10s. Header and filter-header downloads do not touch the wallet
            DB, so RPCs stay responsive in those phases. Disable write actions only while
            `syncPhase === 'blocks'`.
          */}
          {(() => {
            const actionsBlocked = syncPhase === 'blocks';
            const blockedTooltip =
              'The wallet is scanning blocks for your transactions and can\u2019t send, receive, or change the password right now. This should finish in less than a minute.';
            return (
              <TooltipProvider delayDuration={200}>
                <div className="grid w-full grid-cols-2 gap-3 sm:gap-4">
                  <ActionTile
                    onClick={() => navigate('/send')}
                    icon={<ArrowUpRight className="h-4 w-4 text-white sm:h-5 sm:w-5" />}
                    label="Send"
                    disabled={actionsBlocked}
                    disabledTooltip={blockedTooltip}
                  />

                  <ActionTile
                    onClick={() => navigate('/receive')}
                    icon={<ArrowDownLeft className="h-4 w-4 text-white sm:h-5 sm:w-5" />}
                    label="Receive"
                    disabled={actionsBlocked}
                    disabledTooltip={blockedTooltip}
                  />

                  <ActionTile
                    onClick={() => navigate('/change-password')}
                    icon={<Key className="h-4 w-4 text-white sm:h-5 sm:w-5" />}
                    label="Password"
                    disabled={actionsBlocked}
                    disabledTooltip={blockedTooltip}
                  />

                  <ActionTile
                    onClick={handleLockWallet}
                    icon={<Lock className="h-4 w-4 text-white sm:h-5 sm:w-5" />}
                    label="Lock"
                  />
                </div>
              </TooltipProvider>
            );
          })()}

          {/* Activity Section */}
          <div className="w-full">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-lg font-medium text-gray-900">Activity</h3>
              <button
                onClick={() => navigate('/activity')}
                className="text-brand-green hover:text-brand-green/80 text-sm transition-colors"
              >
                View All
              </button>
            </div>

            {/* Real Activity Data */}
            <div className="space-y-3">
              {activitiesPreview.length === 0 ? (
                <div className="rounded-lg border border-gray-200 bg-white py-8 text-center shadow-sm">
                  <p className="text-sm text-gray-500">No recent activity</p>
                </div>
              ) : (
                activitiesPreview.map(activity => (
                  <div
                    key={`${activity.type}_${activity.txid}_${activity.amount}`}
                    className="flex items-center justify-between rounded-lg border border-gray-200 bg-white p-4 shadow-sm transition-shadow hover:shadow-md"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gray-100">
                        {activity.type === 'received' ? (
                          <ArrowDownLeft className="text-green-700 h-4 w-4" />
                        ) : (
                          <ArrowUpRight className="h-4 w-4 text-red-500" />
                        )}
                      </div>
                      <div>
                        <div className="text-sm font-medium text-gray-900">
                          {activity.type === 'received' ? 'Received' : 'Sent'}
                        </div>
                        <div className="text-xs text-gray-500">{formatTimeAgo(activity.time)}</div>
                      </div>
                    </div>
                    <div
                      className={`text-sm font-medium ${activity.type === 'received' ? 'text-green-700' : 'text-red-500'
                        }`}
                    >
                      {activity.type === 'received' ? '+' : '-'}
                      {activity.amount} PRL
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Add bottom spacing to ensure content doesn't get cut off */}
          <div className="h-4 flex-shrink-0 sm:h-8"></div>
        </div>
      </div>
    </div>
  );
}

interface ActionTileProps {
  onClick: () => void;
  icon: ReactNode;
  label: string;
  disabled?: boolean;
  disabledTooltip?: string;
}

function ActionTile({ onClick, icon, label, disabled = false, disabledTooltip }: ActionTileProps) {
  const baseClasses =
    'flex w-full flex-col items-center gap-2 rounded-lg border p-4 shadow-sm transition-all sm:gap-3 sm:p-5';
  const enabledClasses =
    'cursor-pointer border-gray-200 bg-white hover:border-gray-300 hover:shadow-md';
  const disabledClasses = 'cursor-not-allowed border-gray-200 bg-gray-100 opacity-60';
  const iconBg = disabled ? 'bg-gray-400' : 'bg-black';

  const button = (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      className={`${baseClasses} ${disabled ? disabledClasses : enabledClasses}`}
    >
      <div className={`flex h-8 w-8 items-center justify-center rounded-full ${iconBg} sm:h-10 sm:w-10`}>
        {icon}
      </div>
      <span className="text-sm font-medium text-gray-900 sm:text-base">{label}</span>
    </button>
  );

  if (!disabled || !disabledTooltip) {
    return button;
  }

  // Disabled <button>s swallow pointer events in most browsers, so Radix never sees the
  // hover. Wrap the trigger in a span (with tabIndex=0 for keyboard focus) so the tooltip
  // still surfaces the reason the tile is inert.
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={0} className="block cursor-not-allowed">
          {button}
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs text-center">
        {disabledTooltip}
      </TooltipContent>
    </Tooltip>
  );
}

interface SyncStageProps {
  label: string;
  current: number;
  total: number;
  active: boolean;
  done: boolean;
  note?: string;
}

function SyncStage({ label, current, total, active, done, note }: SyncStageProps) {
  // Pending stages show an empty bar and a "— / total" counter. Reporting the raw
  // `current` value while pending is misleading: `block_height` is already non-zero during
  // the filter-header phase because the address manager tracks block-level rollforward
  // separately from the full-block recovery we're actually waiting on.
  const pending = !active && !done;
  const pct = done ? 100 : pending ? 0 : total > 0 ? Math.min(100, (current / total) * 100) : 0;
  const barColor = done ? 'bg-green-500' : active ? 'bg-amber-500' : 'bg-gray-300';
  const labelColor = active ? 'font-medium text-amber-700' : done ? 'text-green-700' : 'text-gray-500';
  const displayCurrent = pending ? '—' : current.toLocaleString();
  const displayTotal = total > 0 ? total.toLocaleString() : '—';
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className={labelColor}>{label}</span>
        <span className="tabular-nums text-gray-500">
          {displayCurrent} / {displayTotal}
        </span>
      </div>
      <div className="mx-auto h-1.5 w-full overflow-hidden rounded-full bg-gray-200">
        <div
          className={`h-full rounded-full ${barColor} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {note && <p className="text-left text-[11px] text-gray-500">{note}</p>}
    </div>
  );
}
