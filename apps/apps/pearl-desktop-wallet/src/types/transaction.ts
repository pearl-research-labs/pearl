export interface Transaction {
  txid: string;
  type: 'received' | 'sent';
  amount: number;
  fee: number;
  confirmations: number;
  time: number;
  address: string;
  account: string;
  blockhash: string;
  trusted: boolean;
  generated: boolean;
}
