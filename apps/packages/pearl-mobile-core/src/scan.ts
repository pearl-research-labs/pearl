/**
 * 地址发现与多地址数据聚合。
 *
 * 移动钱包无本地链数据，需要向数据服务逐个地址探测：
 * 从 index 0 开始递增，连续 gapLimit 个地址无任何历史即停止（BIP44 gap limit）。
 * 找零固定使用 chain=1/index=0 的单一地址。
 */

import {NetworkName} from './constants';
import {deriveTaprootKey} from './crypto';
import {encodeAddress} from './address';
import {ApiTxItem} from './api-types';
import {bytesToHex} from './utils';

export interface AddressEntry {
  /** BIP86 派生 index */
  index: number;
  /** 0=收款 1=找零 */
  chain: 0 | 1;
  address: string;
  /** P2TR 锁定脚本 hex */
  scriptHex: string;
  /** 是否已产生过链上历史 */
  used: boolean;
}

export interface Querier {
  /** 返回某地址的历史交易条数（0 表示未使用） */
  historyCount(address: string): Promise<number>;
}

export interface DiscoverOptions {
  /** 连续多少个未使用地址后停止，默认 5 */
  gapLimit?: number;
  /** 探测上限，默认 100 */
  maxIndex?: number;
  /** 已知的最大使用 index（恢复钱包时跳过已知段） */
  knownNextIndex?: number;
  onProgress?: (index: number) => void;
}

export interface DiscoverResult {
  entries: AddressEntry[];
  /** 下一个未使用的收款 index */
  nextIndex: number;
}

export function entryFor(
  mnemonic: string,
  network: NetworkName,
  index: number,
  chain: 0 | 1,
  used: boolean
): AddressEntry {
  const key = deriveTaprootKey(mnemonic, network, index, chain);
  return {
    index,
    chain,
    address: encodeAddress(key.outputPubKeyX, network),
    scriptHex: bytesToHex(p2trScript(key.outputPubKeyX)),
    used,
  };
}

function p2trScript(outputX: Uint8Array): Uint8Array {
  const out = new Uint8Array(34);
  out[0] = 0x51;
  out[1] = 0x20;
  out.set(outputX, 2);
  return out;
}

/** 钱包固定找零地址（chain=1, index=0） */
export function changeEntry(mnemonic: string, network: NetworkName): AddressEntry {
  return entryFor(mnemonic, network, 0, 1, true);
}

/** gap-limit 地址发现 */
export async function discoverAddresses(
  mnemonic: string,
  network: NetworkName,
  querier: Querier,
  options: DiscoverOptions = {}
): Promise<DiscoverResult> {
  const gapLimit = options.gapLimit ?? 5;
  const maxIndex = options.maxIndex ?? 100;
  const start = options.knownNextIndex ?? 0;

  const entries: AddressEntry[] = [];
  let gap = 0;
  let nextIndex = 0;

  for (let i = start; i < maxIndex; i++) {
    options.onProgress?.(i);
    const probe = entryFor(mnemonic, network, i, 0, false);
    const count = await querier.historyCount(probe.address);
    if (count > 0) {
      entries.push({...probe, used: true});
      gap = 0;
      nextIndex = i + 1;
    } else {
      gap++;
      if (gap >= gapLimit) break;
    }
  }
  // 保证首个收款地址始终在列表中（新钱包也能立即收款）
  if (nextIndex === 0 && entries.length === 0) {
    entries.push(entryFor(mnemonic, network, 0, 0, false));
  }
  return {entries, nextIndex: Math.max(nextIndex, entries.length > 0 ? nextIndex : 1)};
}

/**
 * 轮换持久化探测（恢复语义 B 的增量补充）。
 *
 * 收款页规则：每次「换新地址」都把新地址持久化，因此恢复端只需
 * 从最大已使用 index 之后继续探测，直到出现 additionalRotation 个
 * 连续未使用地址，就能恰好找回用户曾经展示过的全部地址——
 * 与浏览器的 gap-limit 无关，不会漏掉轮换出去的资金。
 */
export async function discoverRotatedAddresses(
  mnemonic: string,
  network: NetworkName,
  querier: Querier,
  usedMax: number,
  additionalRotation = 3,
  maxIndex = 200,
  onProgress?: (index: number) => void,
): Promise<AddressEntry[]> {
  const entries: AddressEntry[] = [];
  let unusedStreak = 0;
  let i = usedMax + 1;
  while (unusedStreak < additionalRotation && i < maxIndex) {
    onProgress?.(i);
    const probe = entryFor(mnemonic, network, i, 0, false);
    const count = await querier.historyCount(probe.address);
    entries.push({...probe, used: count > 0});
    unusedStreak = count > 0 ? 0 : unusedStreak + 1;
    i++;
  }
  return entries;
}

/** 合并多个地址的历史：按 txid 去重（净流入求和）、时间倒序 */
export function mergeHistories(perAddress: ApiTxItem[][]): ApiTxItem[] {
  const byTxid = new Map<string, ApiTxItem>();
  for (const list of perAddress) {
    for (const tx of list) {
      const existing = byTxid.get(tx.txid);
      if (!existing) {
        byTxid.set(tx.txid, {...tx});
      } else {
        const merged = BigInt(existing.amount) + BigInt(tx.amount);
        existing.amount = merged.toString();
        existing.confirmations = Math.max(existing.confirmations, tx.confirmations);
        existing.timestamp = Math.max(existing.timestamp, tx.timestamp);
        // 方向按合并后的净额重新判定
        existing.direction =
          existing.direction === 'self' || tx.direction === 'self'
            ? 'self'
            : merged >= 0n
              ? 'in'
              : 'out';
      }
    }
  }
  return [...byTxid.values()].sort((a, b) => {
    if (a.confirmations === 0 && b.confirmations !== 0) return -1;
    if (a.confirmations !== 0 && b.confirmations === 0) return 1;
    return b.timestamp - a.timestamp;
  });
}
