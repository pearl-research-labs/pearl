/**
 * 本地持久化（AsyncStorage）：
 * 只保存非敏感数据 —— 网络选择、地址簿、交易历史缓存、公开 API 配置。
 * 助记词永远不进这里（见 secure.ts）。
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import type {NetworkName, ApiTxItem} from '@pearl/pearl-mobile-core';

const KEY_NETWORK = 'pearl.network';
const KEY_ADDRESSES = 'pearl.addresses';
const KEY_HISTORY_PREFIX = 'pearl.history.';
const KEY_API_BASE = 'pearl.apiBase';

export interface StoredAddress {
  index: number;
  chain: 0 | 1;
  address: string;
  scriptHex: string;
  used: boolean;
}

export interface PersistedState {
  network: NetworkName;
  addresses: StoredAddress[];
}

export async function loadNetwork(): Promise<NetworkName | null> {
  const v = await AsyncStorage.getItem(KEY_NETWORK);
  if (v === 'mainnet' || v === 'testnet' || v === 'regtest') return v;
  return null;
}

export async function saveNetwork(network: NetworkName): Promise<void> {
  await AsyncStorage.setItem(KEY_NETWORK, network);
}

export async function loadAddresses(network: NetworkName): Promise<PersistedState | null> {
  const raw = await AsyncStorage.getItem(`${KEY_ADDRESSES}.${network}`);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as PersistedState;
  } catch {
    return null;
  }
}

export async function saveAddresses(state: PersistedState): Promise<void> {
  await AsyncStorage.setItem(`${KEY_ADDRESSES}.${state.network}`, JSON.stringify(state));
}

export async function loadHistoryCache(network: NetworkName): Promise<ApiTxItem[]> {
  const raw = await AsyncStorage.getItem(KEY_HISTORY_PREFIX + network);
  if (!raw) return [];
  try {
    return JSON.parse(raw) as ApiTxItem[];
  } catch {
    return [];
  }
}

export async function saveHistoryCache(network: NetworkName, txs: ApiTxItem[]): Promise<void> {
  // 只缓存最近 200 条
  await AsyncStorage.setItem(KEY_HISTORY_PREFIX + network, JSON.stringify(txs.slice(0, 200)));
}

export async function loadApiBase(): Promise<string | null> {
  return AsyncStorage.getItem(KEY_API_BASE);
}

export async function saveApiBase(url: string): Promise<void> {
  await AsyncStorage.setItem(KEY_API_BASE, url);
}

/** 清空全部本地数据（删除钱包时调用） */
export async function wipeAll(): Promise<void> {
  const keys = await AsyncStorage.getAllKeys();
  const mine = keys.filter(k => k.startsWith('pearl.'));
  await AsyncStorage.multiRemove(mine);
}
