export interface TransactionDetail {
  txid: string;
  amount: number;
  fee: number;
  confirmations: number;
  time: number;
  timereceived: number;
  blockhash: string;
  blockindex: number;
  blocktime: number;
  details: unknown[];
  hex: string;
  walletconflicts: string[];
}
