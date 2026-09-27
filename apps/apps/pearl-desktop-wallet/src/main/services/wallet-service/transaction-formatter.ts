import {Transaction} from '../../../types/transaction';

export type RawTransaction = {
  txid: string;
  category: string;
  amount: number;
  fee?: number;
  confirmations?: number;
  time: number;
  address?: string;
  account?: string;
  blockhash?: string;
  trusted?: boolean;
  generated?: boolean;
};

export function formatTransaction(tx: RawTransaction): Transaction {
  const isReceived = tx.category === 'receive' || tx.category === 'generate';

  return {
    txid: tx.txid,
    type: isReceived ? 'received' : 'sent',
    amount: Math.abs(tx.amount),
    fee: tx.fee ? Math.abs(tx.fee) : 0,
    confirmations: tx.confirmations || 0,
    time: tx.time * 1000,
    address: tx.address || '',
    account: tx.account || '',
    blockhash: tx.blockhash || '',
    trusted: tx.trusted || false,
    generated: tx.generated || false,
  };
}

export function formatAndSortTransactions(transactions: any[]): Transaction[] {
  const formatted = transactions.map((tx: any) => formatTransaction(tx));
  return formatted.sort((a: Transaction, b: Transaction) => b.time - a.time);
}
