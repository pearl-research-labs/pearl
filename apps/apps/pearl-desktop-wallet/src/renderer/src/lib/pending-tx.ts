import type { Transaction } from '../../../types/transaction';
import { formatTimeAgo } from './utils';

// The daemon reports this when it announced a transaction but no peer asked
// for it. On a first send nothing left this machine; on a rebroadcast it is
// ambiguous, because peers that already hold the transaction stay silent.
export function isNotRelayedError(message: string): boolean {
  return message.includes('not relayed');
}

export const REBROADCAST_NOT_RELAYED_MESSAGE =
  'No peer requested it: either every connected peer already has it, or none will take it.';

export const REMOVE_WARNING =
  'Its inputs become spendable again, and any pending transaction spending from it is removed too. ' +
  'The network is not consulted: if a peer already holds this transaction it may still confirm, ' +
  'and spending the freed inputs again is a double-spend attempt.';

// Status line for an unconfirmed transaction. Relay fields are absent on a
// full-node daemon, where there is nothing to say beyond "pending".
export function pendingStatusLabel(tx: Transaction, nowMs = Date.now()): string {
  // Relay copy is for sends this wallet announced. Incoming 0-conf is just pending.
  if (tx.type === 'received' || tx.relayed === undefined) return 'Pending';
  if (!tx.relayed) return 'Pending, not announced since start';
  if (tx.lastRelayTime) return `Pending, relayed ${formatTimeAgo(tx.lastRelayTime, nowMs)}`;
  return 'Pending, relayed';
}
