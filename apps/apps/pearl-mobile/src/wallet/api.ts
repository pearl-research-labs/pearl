/**
 * 数据服务客户端（pearl-api）。
 * 金额字段为 Grain 字符串，由 @pearl/pearl-mobile-core 的 api-types 约束。
 */

import type {
  BalanceResponse,
  UtxosResponse,
  HistoryResponse,
  BroadcastResponse,
  FeeEstimateResponse,
  StatusResponse,
  ApiTxItem,
} from '@pearl/pearl-mobile-core';
import {loadApiBase} from './storage';

const DEFAULT_MAINNET = 'https://api.pearlnetwork.net';
const DEFAULT_TESTNET = 'https://testnet-api.pearlnetwork.net';

export class ApiClient {
  constructor(
    private baseUrl: string,
    private token?: string
  ) {}

  static defaultFor(network: 'mainnet' | 'testnet' | 'regtest'): string {
    return network === 'mainnet' ? DEFAULT_MAINNET : DEFAULT_TESTNET;
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15_000);
    try {
      const res = await fetch(this.baseUrl + path, {
        ...init,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...(this.token ? {Authorization: `Bearer ${this.token}`} : {}),
          ...(init?.headers ?? {}),
        },
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = (body as {error?: string}).error ?? `HTTP ${res.status}`;
        throw new Error(msg);
      }
      return body as T;
    } finally {
      clearTimeout(timer);
    }
  }

  status(): Promise<StatusResponse> {
    return this.request('/v1/status');
  }

  balance(address: string): Promise<BalanceResponse> {
    return this.request(`/v1/balance?address=${encodeURIComponent(address)}`);
  }

  utxos(address: string): Promise<UtxosResponse> {
    return this.request(`/v1/utxos?address=${encodeURIComponent(address)}`);
  }

  async history(address: string, skip = 0, count = 25): Promise<HistoryResponse> {
    return this.request(
      `/v1/history?address=${encodeURIComponent(address)}&skip=${skip}&count=${count}`
    );
  }

  /** 地址是否有任何历史（gap-limit 地址发现用），首页即可判断 */
  async historyCount(address: string): Promise<number> {
    const page = await this.history(address, 0, 1);
    return page.txs.length > 0 || page.nextCursor !== '' ? 1 : 0;
  }

  feeEstimate(): Promise<FeeEstimateResponse> {
    return this.request('/v1/fee-estimate');
  }

  broadcast(txHex: string): Promise<BroadcastResponse> {
    return this.request('/v1/broadcast', {
      method: 'POST',
      body: JSON.stringify({txHex}),
    });
  }
}

/** 合并去重交易历史（多地址归并的别名，保持 UI 依赖清晰） */
export type {ApiTxItem};

/** 构造客户端：优先使用设置页保存的自定义节点 */
export async function getApiClient(network: 'mainnet' | 'testnet' | 'regtest'): Promise<ApiClient> {
  const custom = await loadApiBase();
  const base = custom && custom.length > 0 ? custom : ApiClient.defaultFor(network);
  return new ApiClient(base);
}
