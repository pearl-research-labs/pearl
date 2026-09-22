/**
 * 钱包引擎：地址发现、余额聚合、交易构建与广播。
 * UI 层只与本模块交互，不直接触碰密码学与存储细节。
 */

import {
  NetworkName,
  Utxo,
  ApiTxItem,
  deriveTaprootKey,
  addressToScriptPubKey,
  isValidAddress,
  selectCoins,
  buildAndSign,
  discoverAddresses,
  discoverRotatedAddresses,
  changeEntry,
  entryFor,
  mergeHistories,
  hexToBytes,
  grainsToPrl,
  prlToGrains,
  FALLBACK_FEE_RATE,
} from '@pearl/pearl-mobile-core';
import type {ApiClient} from './api';
import {StoredAddress} from './storage';

export interface WalletSnapshot {
  network: NetworkName;
  /** 已确认余额（Grain） */
  confirmed: bigint;
  /** 未确认净流入（Grain） */
  unconfirmed: bigint;
  /** 所有已发现收款地址 */
  addresses: StoredAddress[];
  /** 当前对外展示的收款地址（最新未使用） */
  receiveAddress: string;
  /** 已使用的最大收款 index（无已用地址时为 -1），供收款页轮换 */
  usedMaxIndex: number;
  /** 找零地址 */
  changeAddress: StoredAddress;
  /** 链高度 */
  blockHeight: number;
  /** 推荐费率 Grain/vbyte */
  feeRate: number;
}

export interface PreparedSend {
  /** 签名后的完整交易 hex */
  txHex: string;
  txid: string;
  feeGrains: bigint;
  amountGrains: bigint;
  toAddress: string;
  vbytes: number;
}

/** 发现/恢复地址集（gap-limit） */
export async function discoverWalletAddresses(
  mnemonic: string,
  network: NetworkName,
  api: ApiClient,
  onProgress?: (index: number) => void
): Promise<StoredAddress[]> {
  const result = await discoverAddresses(mnemonic, network, api, {onProgress});
  return result.entries.map(e => ({
    index: e.index,
    chain: e.chain,
    address: e.address,
    scriptHex: e.scriptHex,
    used: e.used,
  }));
}

/** 拉取当前快照：各地址余额聚合 + 链状态 + 费率（轮换持久化地址发现） */
export async function refreshSnapshot(
  mnemonic: string,
  network: NetworkName,
  api: ApiClient,
  addresses: StoredAddress[]
): Promise<WalletSnapshot> {
  const change = changeEntry(mnemonic, network);

  // 已用最大 index 之外，再保证额外 3 个未使用地址被监视，
  // 与收款页轮换规则一致，恢复/日常刷新都不会漏掉曾展示的地址
  const usedMax = addresses.filter(a => a.used).reduce((m, a) => Math.max(m, a.index), -1);
  const rotated = await discoverRotatedAddresses(mnemonic, network, api, usedMax);

  const all = [...addresses];
  for (const r of rotated) {
    if (!all.some(a => a.address === r.address)) all.push(r);
  }

  // 收款展示地址：最大已使用 index + 1；新钱包（无已用地址）用 index 0
  const receiveIndex = usedMax >= 0 ? usedMax + 1 : 0;
  const receiveEntry =
    all.find(a => a.index === receiveIndex && a.chain === 0) ??
    entryFor(mnemonic, network, receiveIndex, 0, false);

  // 余额聚合（找零地址一并计入）
  const [status, fee, perAddressBalance] = await Promise.all([
    api.status().catch(() => ({blockHeight: 0, network, serverTime: 0})),
    api.feeEstimate().catch(() => ({feeRate: FALLBACK_FEE_RATE})),
    Promise.all(
      [...all, change].map(a =>
        api.balance(a.address).catch(() => ({confirmed: '0', unconfirmed: '0'}))
      )
    ),
  ]);

  let confirmed = 0n;
  let unconfirmed = 0n;
  for (const b of perAddressBalance) {
    confirmed += BigInt(b.confirmed);
    unconfirmed += BigInt(b.unconfirmed);
  }

  return {
    network,
    confirmed,
    unconfirmed,
    addresses: all,
    receiveAddress: receiveEntry.address,
    usedMaxIndex: usedMax,
    changeAddress: {
      index: change.index,
      chain: change.chain,
      address: change.address,
      scriptHex: change.scriptHex,
      used: true,
    },
    blockHeight: status.blockHeight,
    feeRate: Math.max(1, Math.ceil(fee.feeRate)),
  };
}

/** 拉取并合并多地址交易历史 */
export async function loadHistory(
  api: ApiClient,
  addresses: StoredAddress[],
  extra?: StoredAddress
): Promise<ApiTxItem[]> {
  const targets = extra ? [...addresses, extra] : addresses;
  const pages = await Promise.all(
    targets.map(a => api.history(a.address, 0, 100).catch(() => ({txs: [], nextCursor: ''})))
  );
  return mergeHistories(pages.map(p => p.txs));
}

/** 收集全部可用 UTXO（含找零地址） */
export async function collectUtxos(
  api: ApiClient,
  addresses: StoredAddress[],
  change: StoredAddress
): Promise<{utxo: Utxo; entry: StoredAddress}[]> {
  const all = [...addresses, change];
  const seen = new Set<string>();
  const result: {utxo: Utxo; entry: StoredAddress}[] = [];
  for (const entry of all) {
    if (seen.has(entry.address)) continue;
    seen.add(entry.address);
    const {utxos} = await api.utxos(entry.address).catch(() => ({utxos: []}));
    for (const u of utxos) {
      // 只花已确认 UTXO，避免依赖未确认找零的链式交易
      if (u.confirmations < 1) continue;
      result.push({
        utxo: {
          txid: u.txid,
          vout: u.vout,
          amountGrains: BigInt(u.amount),
          scriptPubKey: hexToBytes(u.scriptPubKey),
        },
        entry,
      });
    }
  }
  return result;
}

/** 构建并签名一笔转账（返回可广播的交易）。找零固定回到 chain=1/index=0 地址。 */
export async function prepareSend(params: {
  mnemonic: string;
  network: NetworkName;
  api: ApiClient;
  fromAddresses: StoredAddress[];
  toAddress: string;
  amountPrl: string;
  feeRate: number;
}): Promise<PreparedSend> {
  const {mnemonic, network, api, fromAddresses, toAddress, amountPrl, feeRate} = params;

  if (!isValidAddress(toAddress, network)) {
    throw new Error('收款地址无效或不属于当前网络');
  }
  const amountGrains = prlToGrains(amountPrl);
  if (amountGrains <= 0n) throw new Error('金额必须大于 0');

  const change = changeEntry(mnemonic, network);
  const changeStored: StoredAddress = {
    index: change.index,
    chain: change.chain,
    address: change.address,
    scriptHex: change.scriptHex,
    used: true,
  };

  const owned = await collectUtxos(api, fromAddresses, changeStored);
  if (owned.length === 0) throw new Error('没有可用余额');

  const selection = selectCoins({
    utxos: owned.map(o => o.utxo),
    recipients: [
      {
        scriptPubKey: addressToScriptPubKey(toAddress, network),
        amountGrains,
      },
    ],
    feeRate,
    changeScript: hexToBytes(change.scriptHex),
  });

  // 为每个输入找到对应地址并派生签名密钥
  const entryByScript = new Map<string, StoredAddress>();
  for (const o of owned) {
    entryByScript.set(BufferKeyOf(o.utxo.scriptPubKey), o.entry);
  }
  const signers = selection.inputs.map(u => {
    const entry = entryByScript.get(BufferKeyOf(u.scriptPubKey));
    if (!entry) throw new Error('找不到输入对应的地址');
    return deriveTaprootKey(mnemonic, network, entry.index, entry.chain);
  });

  const built = buildAndSign(selection.inputs, selection.outputs, signers);
  return {
    txHex: built.txHex,
    txid: built.txid,
    feeGrains: selection.feeGrains,
    amountGrains,
    toAddress,
    vbytes: built.vbytes,
  };
}

function BufferKeyOf(bytes: Uint8Array): string {
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return s;
}

/** 手续费展示助手 */
export function formatGrains(grains: bigint): string {
  return `${grainsToPrl(grains)} PRL`;
}

export {grainsToPrl, prlToGrains};
