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
  // Present only for a pending send on an SPV daemon. false means no peer
  // requested it since the daemon started, not that the network lacks it.
  relayed?: boolean;
  // Milliseconds; set only when relayed is true.
  lastRelayTime?: number;
}
