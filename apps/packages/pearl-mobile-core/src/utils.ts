/**
 * 二进制与哈希工具。
 * RN 环境没有 Node Buffer / crypto，所有实现基于 @noble 与 @scure 纯 JS 库。
 */

import {sha256} from '@noble/hashes/sha2.js';
import {hmac} from '@noble/hashes/hmac.js';
import {ripemd160 as ripemd160Hash} from '@noble/hashes/legacy.js';

/** UTF-8 编码 */
export function utf8ToBytes(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

/** 十六进制 → 字节 */
export function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2 !== 0) throw new Error('hex 长度必须为偶数');
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    const byte = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
    if (Number.isNaN(byte)) throw new Error(`非法 hex 字符: ${hex}`);
    out[i] = byte;
  }
  return out;
}

/** 字节 → 十六进制 */
export function bytesToHex(bytes: Uint8Array): string {
  let out = '';
  for (let i = 0; i < bytes.length; i++) {
    out += bytes[i].toString(16).padStart(2, '0');
  }
  return out;
}

/** 拼接多个字节数组 */
export function concatBytes(...arrays: Uint8Array[]): Uint8Array {
  const total = arrays.reduce((n, a) => n + a.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const a of arrays) {
    out.set(a, offset);
    offset += a.length;
  }
  return out;
}

/** 字节数组相等性比较 */
export function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/** 反转字节序（txid 显示序 ↔ 内部小端序） */
export function reverseBytes(bytes: Uint8Array): Uint8Array {
  const out = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) {
    out[i] = bytes[bytes.length - 1 - i];
  }
  return out;
}

/** 双 SHA-256（比特币/Pearl 标准哈希） */
export function sha256d(data: Uint8Array): Uint8Array {
  return sha256(sha256(data));
}

/** HASH160 = RIPEMD160(SHA256(x)) */
export function hash160(data: Uint8Array): Uint8Array {
  return ripemd160Hash(sha256(data));
}

/**  tagged hash：SHA256(SHA256(tag) || SHA256(tag) || data)，BIP340/341/342 使用 */
export function taggedHash(tag: string, data: Uint8Array): Uint8Array {
  const tagHash = sha256(utf8ToBytes(tag));
  return sha256(concatBytes(tagHash, tagHash, data));
}

/** HMAC-SHA512（BIP32 派生用） */
export function hmacSha512(key: Uint8Array, data: Uint8Array): Uint8Array {
  return hmac(sha512FromNoble(), key, data);
}

// @noble/hashes 的 sha512 与 hmac 需要显式关联
import {sha512 as sha512Impl} from '@noble/hashes/sha2.js';
function sha512FromNoble(): typeof sha512Impl {
  return sha512Impl;
}

/** u32 小端序列化 */
export function uint32LE(n: number): Uint8Array {
  const out = new Uint8Array(4);
  new DataView(out.buffer).setUint32(0, n >>> 0, true);
  return out;
}

/** u32 大端序列化 */
export function uint32BE(n: number): Uint8Array {
  const out = new Uint8Array(4);
  new DataView(out.buffer).setUint32(0, n >>> 0, false);
  return out;
}

/** u64（grain 金额）小端序列化，输入按 BigInt 处理避免精度丢失 */
export function uint64LE(n: bigint): Uint8Array {
  const out = new Uint8Array(8);
  const view = new DataView(out.buffer);
  view.setBigUint64(0, n & 0xffffffffffffffffn, true);
  return out;
}

/** CompactSize（varint）编码 */
export function varint(n: number): Uint8Array {
  if (n < 0xfd) return Uint8Array.of(n);
  if (n <= 0xffff) {
    const out = new Uint8Array(3);
    out[0] = 0xfd;
    new DataView(out.buffer).setUint16(1, n, true);
    return out;
  }
  if (n <= 0xffffffff) {
    const out = new Uint8Array(5);
    out[0] = 0xfe;
    new DataView(out.buffer).setUint32(1, n >>> 0, true);
    return out;
  }
  const out = new Uint8Array(9);
  out[0] = 0xff;
  new DataView(out.buffer).setBigUint64(1, BigInt(n), true);
  return out;
}

/** CompactSize 解码，返回 [值, 消耗字节数] */
export function readVarint(data: Uint8Array, offset: number): [number, number] {
  const first = data[offset];
  if (first < 0xfd) return [first, 1];
  const view = new DataView(data.buffer, data.byteOffset);
  if (first === 0xfd) return [view.getUint16(offset + 1, true), 3];
  if (first === 0xfe) return [view.getUint32(offset + 1, true), 5];
  return [Number(view.getBigUint64(offset + 1, true)), 9];
}

/** Grain（bigint）→ PRL 字符串，保留 8 位小数并去除尾部零 */
export function grainsToPrl(grains: bigint): string {
  const negative = grains < 0n;
  const abs = negative ? -grains : grains;
  const whole = abs / 100_000_000n;
  const frac = abs % 100_000_000n;
  let fracStr = frac.toString().padStart(8, '0').replace(/0+$/, '');
  const body = fracStr.length > 0 ? `${whole}.${fracStr}` : whole.toString();
  return (negative ? '-' : '') + body;
}

/** PRL 字符串 → Grain（bigint），最多 8 位小数 */
export function prlToGrains(prl: string): bigint {
  const trimmed = prl.trim();
  if (!/^-?\d+(\.\d{1,8})?$/.test(trimmed)) {
    throw new Error(`非法 PRL 金额: ${prl}`);
  }
  const negative = trimmed.startsWith('-');
  const [wholePart, fracPart = ''] = trimmed.replace('-', '').split('.');
  const grains = BigInt(wholePart) * 100_000_000n + BigInt(fracPart.padEnd(8, '0') || '0');
  return negative ? -grains : grains;
}
