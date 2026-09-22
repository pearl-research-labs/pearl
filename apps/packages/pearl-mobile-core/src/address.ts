/**
 * Pearl 地址编码/解码。
 *
 * Pearl 仅启用 Taproot（witness v1）地址：bech32m 编码，
 * 主网前缀 prl1p...，测试网 tprl1p...，回归网 rprl1p...。
 * （与 apps/packages/pearl-address-validation 的行为保持一致）
 */

import {bech32m} from '@scure/base';
import {NETWORKS, NetworkName} from './constants';

export interface DecodedAddress {
  /** 所属网络（依据 HRP 判断） */
  network: NetworkName;
  /** 32 字节输出公钥 x 坐标 */
  program: Uint8Array;
}

/** 输出公钥 x 坐标 → Pearl 地址 */
export function encodeAddress(outputPubKeyX: Uint8Array, network: NetworkName): string {
  if (outputPubKeyX.length !== 32) throw new Error('Taproot 公钥必须为 32 字节');
  const hrp = NETWORKS[network].hrp;
  const words = bech32m.toWords(outputPubKeyX);
  // witness version 1 → 首字 1
  return bech32m.encode(hrp, [1, ...words]);
}

/** 解析 Pearl 地址，校验网络与版本 */
export function decodeAddress(address: string, expectNetwork?: NetworkName): DecodedAddress {
  const trimmed = address.trim().toLowerCase();
  let decoded;
  try {
    decoded = bech32m.decode(trimmed as `${string}1${string}`);
  } catch {
    throw new Error('地址不是合法的 bech32m 编码');
  }

  const hrp = decoded.prefix;
  const entry = (Object.keys(NETWORKS) as NetworkName[]).find(n => NETWORKS[n].hrp === hrp);
  if (!entry) throw new Error(`未知地址前缀: ${hrp}`);
  if (expectNetwork && entry !== expectNetwork) {
    throw new Error(`地址属于 ${entry}，当前钱包网络为 ${expectNetwork}`);
  }

  const words = decoded.words;
  if (words.length === 0 || words[0] !== 1) {
    throw new Error('仅支持 Taproot（witness v1）地址');
  }
  const program = bech32m.fromWords(words.slice(1));
  if (program.length !== 32) {
    throw new Error(`Taproot 程序长度必须为 32 字节，实际 ${program.length}`);
  }
  return {network: entry, program};
}

/** 地址 → P2TR 输出脚本：OP_1 OP_PUSHBYTES_32 <pubkey x> */
export function addressToScriptPubKey(address: string, expectNetwork?: NetworkName): Uint8Array {
  const {program} = decodeAddress(address, expectNetwork);
  const out = new Uint8Array(34);
  out[0] = 0x51; // OP_1
  out[1] = 0x20; // OP_PUSHBYTES_32
  out.set(program, 2);
  return out;
}

/** 仅做格式校验（不抛异常），供 UI 输入框实时提示 */
export function isValidAddress(address: string, expectNetwork?: NetworkName): boolean {
  try {
    decodeAddress(address, expectNetwork);
    return true;
  } catch {
    return false;
  }
}
