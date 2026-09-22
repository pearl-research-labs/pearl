/**
 * 后端 API 类型定义。
 * 与 pearl-mobile-server（node pearl-api）响应保持一致，金额单位均为 Grain。
 */

export interface ApiUtxo {
  txid: string;
  vout: number;
  /** Grain（字符串承载，避免 JSON 精度问题） */
  amount: string;
  scriptPubKey: string; // hex
  confirmations: number;
}

export interface ApiTxItem {
  txid: string;
  /** 相对当前地址集净流入（Grain，可为负） */
  amount: string;
  fee?: string;
  blockHeight: number;
  timestamp: number; // unix 秒
  confirmations: number;
  direction: 'in' | 'out' | 'self';
}

export interface BalanceResponse {
  /** 已确认余额（Grain 字符串） */
  confirmed: string;
  /** 未确认变动（Grain 字符串，可为负） */
  unconfirmed: string;
}

export interface UtxosResponse {
  utxos: ApiUtxo[];
}

export interface HistoryResponse {
  txs: ApiTxItem[];
  /** 游标：用于分页，空字符串表示没有更多 */
  nextCursor: string;
}

export interface BroadcastResponse {
  txid: string;
}

export interface FeeEstimateResponse {
  /** Grain/vbyte */
  feeRate: number;
}

export interface StatusResponse {
  blockHeight: number;
  network: string;
  serverTime: number;
}
