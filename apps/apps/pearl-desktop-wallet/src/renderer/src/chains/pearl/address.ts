import { secp256k1 } from "@noble/curves/secp256k1.js";
import { sha256 } from "@noble/hashes/sha2.js";
import type { PearlNetworkParams } from "./network";

const TAPROOT_VERSION = 1;
const TAPROOT_PROGRAM_LEN = 32;
const CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
const CHARSET_REV = new Map<string, number>(Array.from(CHARSET, (c, i) => [c, i]));

function taggedHash(tag: string, data: Uint8Array): Uint8Array {
  const tagHash = sha256(new TextEncoder().encode(tag));
  const concat = new Uint8Array(tagHash.length * 2 + data.length);
  concat.set(tagHash, 0);
  concat.set(tagHash, tagHash.length);
  concat.set(data, tagHash.length * 2);
  return sha256(concat);
}

function bytesToBigInt(b: Uint8Array): bigint {
  let v = 0n;
  for (const byte of b) v = (v << 8n) | BigInt(byte);
  return v;
}

function bigIntToBytes(v: bigint, length: number): Uint8Array {
  const out = new Uint8Array(length);
  let cur = v;
  for (let i = length - 1; i >= 0; i--) {
    out[i] = Number(cur & 0xffn);
    cur >>= 8n;
  }
  return out;
}

function polymod(values: number[]): number {
  const GENERATORS = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];
  let chk = 1;
  for (const value of values) {
    const top = chk >>> 25;
    chk = ((chk & 0x1ffffff) << 5) ^ value;
    for (let i = 0; i < GENERATORS.length; i++) {
      if ((top >>> i) & 1) chk ^= GENERATORS[i]!;
    }
  }
  return chk;
}

function hrpExpand(hrp: string): number[] {
  const out: number[] = [];
  for (let i = 0; i < hrp.length; i++) out.push(hrp.charCodeAt(i) >>> 5);
  out.push(0);
  for (let i = 0; i < hrp.length; i++) out.push(hrp.charCodeAt(i) & 31);
  return out;
}

function createChecksum(hrp: string, data: number[]): number[] {
  const values = [...hrpExpand(hrp), ...data, 0, 0, 0, 0, 0, 0];
  const mod = polymod(values) ^ 0x2bc830a3;
  const out = new Array<number>(6);
  for (let p = 0; p < 6; p++) out[p] = (mod >>> (5 * (5 - p))) & 31;
  return out;
}

function verifyChecksum(hrp: string, data: number[]): boolean {
  return polymod([...hrpExpand(hrp), ...data]) === 0x2bc830a3;
}

function convertBits(data: Uint8Array, fromBits: number, toBits: number, pad: boolean): number[] {
  let acc = 0;
  let bits = 0;
  const ret: number[] = [];
  const maxv = (1 << toBits) - 1;
  for (const value of data) {
    if (value < 0 || value >>> fromBits !== 0) throw new Error("E_INVALID_ADDRESS");
    acc = (acc << fromBits) | value;
    bits += fromBits;
    while (bits >= toBits) {
      bits -= toBits;
      ret.push((acc >>> bits) & maxv);
    }
  }
  if (pad) {
    if (bits > 0) ret.push((acc << (toBits - bits)) & maxv);
  } else if (bits >= fromBits || ((acc << (toBits - bits)) & maxv) !== 0) {
    throw new Error("E_INVALID_ADDRESS");
  }
  return ret;
}

function bech32mEncode(hrp: string, version: number, program: Uint8Array): string {
  const data = [version, ...convertBits(program, 8, 5, true)];
  const checksum = createChecksum(hrp, data);
  return `${hrp}1${[...data, ...checksum].map((v) => CHARSET[v]).join("")}`;
}

function bech32mDecode(address: string): { hrp: string; version: number; program: Uint8Array } {
  const lower = address.toLowerCase();
  if (address !== lower && address !== address.toUpperCase()) throw new Error("E_INVALID_ADDRESS");
  const normalized = lower;
  const pos = normalized.lastIndexOf("1");
  if (pos <= 0 || pos + 7 > normalized.length) throw new Error("E_INVALID_ADDRESS");
  const hrp = normalized.slice(0, pos);
  const data = normalized.slice(pos + 1);
  const words: number[] = [];
  for (const ch of data) {
    const v = CHARSET_REV.get(ch);
    if (v === undefined) throw new Error("E_INVALID_ADDRESS");
    words.push(v);
  }
  if (!verifyChecksum(hrp, words)) throw new Error("E_INVALID_ADDRESS");
  const payload = words.slice(0, -6);
  const version = payload[0];
  const program = Uint8Array.from(convertBits(Uint8Array.from(payload.slice(1)), 5, 8, false));
  return { hrp, version, program };
}

export function bip86Tweak(xOnlyInternal: Uint8Array): Uint8Array {
  if (xOnlyInternal.length !== 32) throw new Error("internal pubkey must be 32 bytes");
  const tweak = taggedHash("TapTweak", xOnlyInternal);
  const internalPoint = secp256k1.Point.fromBytes(Uint8Array.from([0x02, ...xOnlyInternal]));
  const tweakedPoint = internalPoint.add(secp256k1.Point.BASE.multiply(bytesToBigInt(tweak)));
  const affine = tweakedPoint.toAffine();
  return bigIntToBytes(affine.x, 32);
}

export function encodeTaprootAddress(outputKey: Uint8Array, params: PearlNetworkParams): string {
  if (outputKey.length !== TAPROOT_PROGRAM_LEN) throw new Error("output key must be 32 bytes");
  return bech32mEncode(params.hrp, TAPROOT_VERSION, outputKey);
}

export function decodeTaprootAddress(address: string, params: PearlNetworkParams): Uint8Array {
  const decoded = bech32mDecode(address);
  if (decoded.hrp !== params.hrp) throw new Error(`E_INVALID_ADDRESS: expected HRP "${params.hrp}"`);
  if (decoded.version !== TAPROOT_VERSION) throw new Error(`E_INVALID_ADDRESS: unsupported witness version ${decoded.version}`);
  if (decoded.program.length !== TAPROOT_PROGRAM_LEN) throw new Error("E_INVALID_ADDRESS: program length");
  return decoded.program;
}

export function isValidPearlAddress(address: string, params: PearlNetworkParams): boolean {
  try {
    decodeTaprootAddress(address, params);
    return true;
  } catch {
    return false;
  }
}

export function pearlAddressFromInternalKey(xOnlyInternal: Uint8Array, params: PearlNetworkParams): string {
  return encodeTaprootAddress(bip86Tweak(xOnlyInternal), params);
}

export function pearlAddressFromCompressedPubkey(compressed: Uint8Array, params: PearlNetworkParams): string {
  if (compressed.length !== 33) throw new Error("compressed pubkey must be 33 bytes");
  return pearlAddressFromInternalKey(compressed.slice(1), params);
}
