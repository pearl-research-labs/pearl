/**
 * 交易构建与签名：BIP341 Taproot key-path 花费。
 *
 * - sighash 计算遵循 BIP341（epoch 0x00 + SigMsg + TapSighash tagged hash）
 * - 仅支持 SIGHASH_DEFAULT(0x00) 与 SIGHASH_ALL(0x01)，默认 ALL
 * - 所有输入必须为 P2TR key-path 花费（钱包自身产出的 UTXO）
 */

import {sha256} from '@noble/hashes/sha2.js';
import {TX_VERSION, TX_LOCKTIME} from './constants';
import {
  concatBytes,
  hexToBytes,
  bytesToHex,
  reverseBytes,
  taggedHash,
  uint32LE,
  uint64LE,
  varint,
  readVarint,
  sha256d,
} from './utils';
import {TaprootKeyPair, schnorrSign} from './crypto';

export const SIGHASH_DEFAULT = 0x00;
export const SIGHASH_ALL = 0x01;

/** 待花费的 UTXO */
export interface Utxo {
  /** 交易 ID（显示序，大端 hex） */
  txid: string;
  /** 输出序号 */
  vout: number;
  /** 金额（Grain） */
  amountGrains: bigint;
  /** 锁定脚本（应为 34 字节 P2TR） */
  scriptPubKey: Uint8Array;
}

export interface TxOutput {
  scriptPubKey: Uint8Array;
  amountGrains: bigint;
}

/**
 * 计算第 inputIndex 个输入的 BIP341 key-path sighash。
 * 对应 btcd txscript CalcTaprootSignatureHash 的 SIGHASH_ALL 分支：
 * sha_* 承诺均为单次 SHA256。
 */
export function taprootSighash(
  inputs: Utxo[],
  outputs: TxOutput[],
  inputIndex: number,
  hashType: number = SIGHASH_DEFAULT
): Uint8Array {
  if (inputIndex < 0 || inputIndex >= inputs.length) {
    throw new Error(`inputIndex 越界: ${inputIndex}`);
  }
  if (hashType !== SIGHASH_DEFAULT && hashType !== SIGHASH_ALL) {
    throw new Error(`不支持的 sighash 类型: ${hashType}`);
  }

  const shaPrevouts = sha256(
    concatBytes(...inputs.map(u => concatBytes(reverseBytes(hexToBytes(u.txid)), uint32LE(u.vout))))
  );
  const shaAmounts = sha256(concatBytes(...inputs.map(u => uint64LE(u.amountGrains))));
  const shaScriptPubKeys = sha256(
    concatBytes(...inputs.map(u => concatBytes(varint(u.scriptPubKey.length), u.scriptPubKey)))
  );
  const shaSequences = sha256(concatBytes(...inputs.map(() => uint32LE(0xffffffff))));

  // SIGHASH_DEFAULT 与 SIGHASH_ALL 都承诺全部输出（BIP341）
  const shaOutputs = sha256(
    concatBytes(
      ...outputs.map(o =>
        concatBytes(uint64LE(o.amountGrains), varint(o.scriptPubKey.length), o.scriptPubKey)
      )
    )
  );

  const sigMsg = concatBytes(
    Uint8Array.of(hashType),
    uint32LE(TX_VERSION),
    uint32LE(TX_LOCKTIME),
    shaPrevouts,
    shaAmounts,
    shaScriptPubKeys,
    shaSequences,
    shaOutputs,
    Uint8Array.of(0x00), // spend_type：无 annex、key-path 花费
    uint32LE(inputIndex)
  );
  return taggedHash('TapSighash', concatBytes(Uint8Array.of(0x00), sigMsg));
}

/** 完整序列化已签名交易（含见证） */
export function serializeSigned(
  inputs: Utxo[],
  outputs: TxOutput[],
  witnesses: Uint8Array[][]
): Uint8Array {
  const chunks: Uint8Array[] = [uint32LE(TX_VERSION)];
  chunks.push(Uint8Array.of(0x00, 0x01)); // segwit marker + flag
  chunks.push(varint(inputs.length));
  for (const u of inputs) {
    chunks.push(reverseBytes(hexToBytes(u.txid)));
    chunks.push(uint32LE(u.vout));
    chunks.push(varint(0)); // 空 scriptSig
    chunks.push(uint32LE(0xffffffff));
  }
  chunks.push(varint(outputs.length));
  for (const o of outputs) {
    chunks.push(uint64LE(o.amountGrains));
    chunks.push(varint(o.scriptPubKey.length));
    chunks.push(o.scriptPubKey);
  }
  for (const w of witnesses) {
    chunks.push(varint(w.length));
    for (const item of w) {
      chunks.push(varint(item.length));
      chunks.push(item);
    }
  }
  chunks.push(uint32LE(TX_LOCKTIME));
  return concatBytes(...chunks);
}

/** 对每个输入做 key-path 签名，返回完整交易 hex 与 txid */
export function buildAndSign(
  inputs: Utxo[],
  outputs: TxOutput[],
  signers: TaprootKeyPair[]
): {txHex: string; txid: string; vbytes: number} {
  if (signers.length !== inputs.length) {
    throw new Error(`签名者数量(${signers.length})与输入数量(${inputs.length})不一致`);
  }
  const witnesses: Uint8Array[][] = [];
  for (let i = 0; i < inputs.length; i++) {
    const sighash = taprootSighash(inputs, outputs, i, SIGHASH_DEFAULT);
    const sig = schnorrSign(sighash, signers[i].tweakedPrivateKey);
    witnesses.push([sig]); // 64 字节签名 = SIGHASH_DEFAULT 语义
  }
  const raw = serializeSigned(inputs, outputs, witnesses);
  const txid = bytesToHex(reverseBytes(sha256d(stripWitness(raw))));
  return {txHex: bytesToHex(raw), txid, vbytes: computeVBytes(raw)};
}

/** 去除见证数据（用于计算 txid） */
export function stripWitness(raw: Uint8Array): Uint8Array {
  const isSegwit = raw[4] === 0x00 && raw[5] === 0x01;
  if (!isSegwit) return raw;
  let offset = 6;

  const [inCount, inUsed] = readVarint(raw, offset);
  offset += inUsed;
  const inCountBytes = raw.slice(6, offset);
  const inStart = offset;
  for (let i = 0; i < inCount; i++) {
    offset += 32 + 4;
    const [sl, su] = readVarint(raw, offset);
    offset += su + sl;
    offset += 4;
  }
  const inEnd = offset;

  const [outCount, outUsed] = readVarint(raw, offset);
  offset += outUsed;
  const outCountBytes = raw.slice(inEnd, offset);
  const outStart = offset;
  for (let i = 0; i < outCount; i++) {
    offset += 8;
    const [sl, su] = readVarint(raw, offset);
    offset += su + sl;
  }
  const outEnd = offset;

  return concatBytes(
    raw.slice(0, 4),
    inCountBytes,
    raw.slice(inStart, inEnd),
    outCountBytes,
    raw.slice(outStart, outEnd),
    raw.slice(raw.length - 4)
  );
}

/** 实际 vbytes：weight = 基础区×4 + 见证区×1，再 /4 向上取整 */
export function computeVBytes(raw: Uint8Array): number {
  const stripped = stripWitness(raw);
  const weight = stripped.length * 4 + (raw.length - stripped.length);
  return Math.ceil(weight / 4);
}

/** 签名前估算 vbytes（P2TR 输入输出） */
export function estimateVBytes(inputCount: number, outputCount: number): number {
  const base = 4 + 1 + inputCount * 41 + outputCount * 43 + 4; // marker/flag≈1
  const witness = Math.ceil((inputCount * 66) / 4);
  return base + witness;
}
