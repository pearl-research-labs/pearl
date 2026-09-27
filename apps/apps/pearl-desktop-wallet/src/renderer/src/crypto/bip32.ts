import { hmac } from "@noble/hashes/hmac.js";
import { sha512 } from "@noble/hashes/sha2.js";
import { concatBytes } from "@noble/hashes/utils.js";
import { secp256k1, schnorr } from "@noble/curves/secp256k1.js";
import { mnemonicToSeedSync, validateMnemonic } from "bip39";

const HARDENED_OFFSET = 0x80000000;
const SEED_KEY = new TextEncoder().encode("Bitcoin seed");
const CURVE_N = secp256k1.Point.CURVE().n;

function uint32ToBytes(index: number): Uint8Array {
  const out = new Uint8Array(4);
  const view = new DataView(out.buffer);
  view.setUint32(0, index >>> 0, false);
  return out;
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function bytesToBigInt(bytes: Uint8Array): bigint {
  let out = 0n;
  for (const byte of bytes) {
    out = (out << 8n) + BigInt(byte);
  }
  return out;
}

function bigIntToBytes(value: bigint, length: number): Uint8Array {
  const out = new Uint8Array(length);
  let n = value;
  for (let i = length - 1; i >= 0; i--) {
    out[i] = Number(n & 0xffn);
    n >>= 8n;
  }
  if (n !== 0n) {
    throw new Error("Integer too large");
  }
  return out;
}

function seedBytesFromString(seed: string): Uint8Array {
  const trimmed = seed.trim();
  if (/^(?:[0-9a-fA-F]{2})+$/.test(trimmed)) {
    const out = new Uint8Array(trimmed.length / 2);
    for (let i = 0; i < trimmed.length; i += 2) {
      out[i / 2] = Number.parseInt(trimmed.slice(i, i + 2), 16);
    }
    return out;
  }
  if (!validateMnemonic(trimmed)) {
    throw new Error("Invalid wallet seed");
  }
  return mnemonicToSeedSync(trimmed);
}

function deriveHardenedChild(parentKey: Uint8Array, parentChainCode: Uint8Array, index: number): {
  privateKey: Uint8Array;
  chainCode: Uint8Array;
} {
  if (!Number.isInteger(index) || index < 0 || index >= HARDENED_OFFSET) {
    throw new Error("Invalid hardened index");
  }

  const data = concatBytes(new Uint8Array([0x00]), parentKey, uint32ToBytes(index | HARDENED_OFFSET));
  const digest = hmac(sha512, parentChainCode, data);
  const il = digest.slice(0, 32);
  const ir = digest.slice(32);
  const ilNum = bytesToBigInt(il);
  if (ilNum === 0n || ilNum >= CURVE_N) {
    throw new Error("Failed to derive child key");
  }

  const parentNum = bytesToBigInt(parentKey);
  const childNum = (ilNum + parentNum) % CURVE_N;
  if (childNum === 0n) {
    throw new Error("Failed to derive child key");
  }

  return {
    privateKey: bigIntToBytes(childNum, 32),
    chainCode: ir,
  };
}

function deriveUnhardenedChild(parentKey: Uint8Array, parentChainCode: Uint8Array, index: number): {
  privateKey: Uint8Array;
  chainCode: Uint8Array;
} {
  if (!Number.isInteger(index) || index < 0 || index >= HARDENED_OFFSET) {
    throw new Error("Invalid child index");
  }

  const parentPubKey = secp256k1.getPublicKey(parentKey, true);
  const data = concatBytes(parentPubKey, uint32ToBytes(index));
  const digest = hmac(sha512, parentChainCode, data);
  const il = digest.slice(0, 32);
  const ir = digest.slice(32);
  const ilNum = bytesToBigInt(il);
  if (ilNum === 0n || ilNum >= CURVE_N) {
    throw new Error("Failed to derive child key");
  }

  const parentNum = bytesToBigInt(parentKey);
  const childNum = (ilNum + parentNum) % CURVE_N;
  if (childNum === 0n) {
    throw new Error("Failed to derive child key");
  }

  return {
    privateKey: bigIntToBytes(childNum, 32),
    chainCode: ir,
  };
}

function masterNodeFromSeed(seed: Uint8Array): { privateKey: Uint8Array; chainCode: Uint8Array } {
  const digest = hmac(sha512, SEED_KEY, seed);
  const il = digest.slice(0, 32);
  const ir = digest.slice(32);
  const ilNum = bytesToBigInt(il);
  if (ilNum === 0n || ilNum >= CURVE_N) {
    throw new Error("Failed to derive master key");
  }
  return {
    privateKey: il,
    chainCode: ir,
  };
}

export function deriveBip32Path(seed: string, path: string): Uint8Array {
  const segments = path.split("/");
  if (segments[0] !== "m") {
    throw new Error("Invalid derivation path");
  }

  let node = masterNodeFromSeed(seedBytesFromString(seed));
  for (const segment of segments.slice(1)) {
    const hardened = segment.endsWith("'");
    const indexText = hardened ? segment.slice(0, -1) : segment;
    const index = Number(indexText);
    if (!Number.isInteger(index) || index < 0) {
      throw new Error("Invalid derivation path");
    }
    node = hardened
      ? deriveHardenedChild(node.privateKey, node.chainCode, index)
      : deriveUnhardenedChild(node.privateKey, node.chainCode, index);
  }

  return node.privateKey;
}

export function deriveSchnorrXOnlyPubkey(secretKey: Uint8Array): string {
  return bytesToHex(schnorr.getPublicKey(secretKey));
}
