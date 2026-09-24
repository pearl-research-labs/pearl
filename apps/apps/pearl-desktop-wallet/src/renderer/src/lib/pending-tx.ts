import type { Transaction } from '../../../types/transaction';
import { formatTimeAgo } from './utils';

// The daemon reports this when it announced a transaction but no peer asked
// for it. On a first send nothing left this machine; on a rebroadcast it is
// ambiguous, because peers that already hold the transaction stay silent.
export function isNotRelayedError(message: string): boolean {
  return message.includes('not relayed');
}

// A self-transfer's receive row carries the same relay fields as its send
// row; only the send row speaks for the broadcast.
export function pendingStatusLabel(tx: Transaction): string {
  if (tx.type === 'received' || tx.relayed === undefined) return 'Pending';
  if (!tx.relayed) return 'Pending, not announced since start';
  if (tx.lastRelayTime) return `Pending, relayed ${formatTimeAgo(tx.lastRelayTime)}`;
  return 'Pending, relayed';
}
