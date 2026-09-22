/**
 * 黄金向量对照测试。
 *
 * vectors.json 由仓库 node 权威实现生成（go run ./node/cmd/genmobilevectors），
 * 覆盖：BIP39 种子、BIP86 派生、Taproot tweak、地址编码、BIP341 sighash、
 * Schnorr 签名互验、完整交易序列化。
 */

import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname, join} from 'node:path';

import {
  mnemonicToSeed,
  deriveTaprootKey,
  schnorrVerify,
  encodeAddress,
  addressToScriptPubKey,
  decodeAddress,
  isValidAddress,
  taprootSighash,
  buildAndSign,
  SIGHASH_DEFAULT,
  hexToBytes,
  bytesToHex,
  varint,
  readVarint,
  grainsToPrl,
  prlToGrains,
  selectCoins,
} from '../src/index';
import type {Utxo} from '../src/index';

const here = dirname(fileURLToPath(import.meta.url));
const vectors = JSON.parse(readFileSync(join(here, 'vectors.json'), 'utf8'));

// ---------- 基础工具 ----------

test('varint 编解码往返', () => {
  for (const n of [0, 1, 0xfc, 0xfd, 0xffff, 0x10000, 0xffffffff]) {
    const enc = varint(n);
    const [dec, used] = readVarint(enc, 0);
    assert.equal(dec, n);
    assert.equal(used, enc.length);
  }
  assert.deepEqual([...varint(0xfd)], [0xfd, 0xfd, 0x00]);
});

test('PRL 与 Grain 换算', () => {
  assert.equal(grainsToPrl(150_000_000n), '1.5');
  assert.equal(grainsToPrl(1n), '0.00000001');
  assert.equal(grainsToPrl(100_000_000_000n), '1000');
  assert.equal(grainsToPrl(-250_000_000n), '-2.5');
  assert.equal(prlToGrains('1.5'), 150_000_000n);
  assert.equal(prlToGrains('0.00000001'), 1n);
  assert.equal(prlToGrains('1000'), 100_000_000_000n);
  assert.throws(() => prlToGrains('0.000000001')); // 超过 8 位小数
  assert.throws(() => prlToGrains('abc'));
});

// ---------- BIP39 / BIP86 ----------

test('BIP39 种子与 Go 端一致', () => {
  const seed = mnemonicToSeed(vectors.mnemonic);
  assert.equal(bytesToHex(seed), vectors.seedHex);
});

test('BIP86 主网地址派生', () => {
  for (const v of vectors.addressesMainnet) {
    const key = deriveTaprootKey(vectors.mnemonic, 'mainnet', v.index, v.chain as 0 | 1);
    assert.equal(bytesToHex(key.privateKey), v.privkey, `privkey idx=${v.index}`);
    assert.equal(bytesToHex(key.internalPubKeyX), v.internalX, `internalX idx=${v.index}`);
    assert.equal(bytesToHex(key.outputPubKeyX), v.outputX, `outputX idx=${v.index}`);

    const address = encodeAddress(key.outputPubKeyX, 'mainnet');
    assert.equal(address, v.address, `address idx=${v.index}`);
    assert.ok(address.startsWith('prl1p'));

    const script = addressToScriptPubKey(address, 'mainnet');
    assert.equal(bytesToHex(script), v.script, `script idx=${v.index}`);
  }
});

test('BIP86 测试网地址派生', () => {
  for (const v of vectors.addressesTestnet) {
    const key = deriveTaprootKey(vectors.mnemonic, 'testnet', v.index, v.chain as 0 | 1);
    const address = encodeAddress(key.outputPubKeyX, 'testnet');
    assert.equal(address, v.address);
    assert.ok(address.startsWith('tprl1p'));
    // 网络校验：主网期望下解析测试网地址必须失败
    assert.equal(isValidAddress(address, 'mainnet'), false);
    assert.equal(isValidAddress(address, 'testnet'), true);
  }
});

test('地址解码网络识别', () => {
  const main0 = vectors.addressesMainnet[0].address;
  const decoded = decodeAddress(main0);
  assert.equal(decoded.network, 'mainnet');
  assert.equal(decoded.program.length, 32);
  assert.throws(() => decodeAddress(main0, 'testnet'));
  assert.equal(isValidAddress('prl1pnotanaddress'), false);
  assert.equal(isValidAddress('bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4'), false); // 旧 v0 地址
});

// ---------- BIP341 sighash / 签名 ----------

test('单输入 BIP341 sighash 与节点一致', () => {
  const v = vectors.sighashSingle;
  const inputs: Utxo[] = [
    {
      txid: v.txid,
      vout: v.vout,
      amountGrains: BigInt(v.amount),
      scriptPubKey: hexToBytes(v.script),
    },
  ];
  const key0 = deriveTaprootKey(vectors.mnemonic, 'mainnet', 0, 0);
  const outputs = [
    {scriptPubKey: hexToBytes(v.toScript), amountGrains: BigInt(v.toAmount)},
    {scriptPubKey: hexToBytes(v.script), amountGrains: BigInt(v.amount - v.toAmount - v.fee)},
  ];

  const sighash = taprootSighash(inputs, outputs, 0, SIGHASH_DEFAULT);
  assert.equal(bytesToHex(sighash), v.sighash, 'BIP341 sighash 必须与节点实现一致');

  // 节点生成的 Schnorr 签名必须能通过 BIP340 验签（互操作性）
  assert.ok(
    schnorrVerify(hexToBytes(v.sig), sighash, key0.outputPubKeyX),
    'Go 端签名无法通过 TS 验签'
  );

  // TS 完整构建交易：txid（不含见证）必须与 Go 端一致
  const built = buildAndSign(inputs, outputs, [key0]);
  assert.equal(built.txid, v.txidOut, 'txid 不一致');
  // 从 Go 端完整交易 hex 解析出见证并验签
  const goSig = extractFirstWitnessSig(hexToBytes(v.txHex));
  assert.ok(schnorrVerify(goSig, sighash, key0.outputPubKeyX), 'Go 交易内签名验签失败');
});

test('多输入 BIP341 sighash 与节点一致', () => {
  const v = vectors.sighashMulti;
  const inputs: Utxo[] = v.txids.map((txid: string, i: number) => ({
    txid,
    vout: i,
    amountGrains: BigInt(v.amounts[i]),
    scriptPubKey: hexToBytes(v.scripts[i]),
  }));
  const outputs = [
    {scriptPubKey: hexToBytes(v.toScript), amountGrains: BigInt(v.toAmount)},
    {scriptPubKey: hexToBytes(v.changeScript), amountGrains: BigInt(v.changeAmount)},
  ];

  assert.equal(bytesToHex(taprootSighash(inputs, outputs, 0, SIGHASH_DEFAULT)), v.sighash0);
  assert.equal(bytesToHex(taprootSighash(inputs, outputs, 1, SIGHASH_DEFAULT)), v.sighash1);

  // TS 构建整笔交易并逐输入验签
  const key0 = deriveTaprootKey(vectors.mnemonic, 'mainnet', 0, 0);
  const key1 = deriveTaprootKey(vectors.mnemonic, 'mainnet', 1, 0);
  const built = buildAndSign(inputs, outputs, [key0, key1]);
  const raw = hexToBytes(built.txHex);
  const sigs = extractAllWitnessSigs(raw);
  assert.equal(sigs.length, 2);
  assert.ok(schnorrVerify(sigs[0], hexToBytes(v.sighash0), key0.outputPubKeyX));
  assert.ok(schnorrVerify(sigs[1], hexToBytes(v.sighash1), key1.outputPubKeyX));

  // Go 端交易内嵌签名同样逐输入验签
  const goSigs = extractAllWitnessSigs(hexToBytes(v.txHex));
  assert.equal(goSigs.length, 2);
  assert.ok(schnorrVerify(goSigs[0], hexToBytes(v.sighash0), key0.outputPubKeyX));
  assert.ok(schnorrVerify(goSigs[1], hexToBytes(v.sighash1), key1.outputPubKeyX));
});

// ---------- 币选择 ----------

test('币选择：正常找零 / 粉尘并费 / 余额不足', () => {
  const mk = (txidByte: number, amt: bigint): Utxo => ({
    txid: txidByte.toString(16).padStart(2, '0').repeat(32),
    vout: 0,
    amountGrains: amt,
    scriptPubKey: new Uint8Array(34).fill(0x51),
  });
  const changeScript = new Uint8Array(34).fill(0x51);
  const recipient = {scriptPubKey: new Uint8Array(34).fill(0x33), amountGrains: 100_000n};

  // 找零充足 → 有找零输出
  const r1 = selectCoins({
    utxos: [mk(0xaa, 200_000n)],
    recipients: [recipient],
    feeRate: 2,
    changeScript,
  });
  assert.equal(r1.hasChange, true);
  assert.equal(r1.outputs.length, 2);
  assert.equal(r1.inputs.length, 1);
  assert.ok(r1.feeGrains > 0n && r1.feeGrains < 1_000n);

  // 找零低于粉尘 → 并入手续费，无找零输出
  const r2 = selectCoins({
    utxos: [mk(0xbb, 100_300n)],
    recipients: [recipient],
    feeRate: 2,
    changeScript,
  });
  assert.equal(r2.hasChange, false);
  assert.equal(r2.outputs.length, 1);
  assert.equal(r2.feeGrains, 300n);

  // 余额不足必须抛错
  assert.throws(() =>
    selectCoins({
      utxos: [mk(0xcc, 1_000n)],
      recipients: [recipient],
      feeRate: 2,
      changeScript,
    })
  );
});

// ---------- 见证解析辅助 ----------

function extractAllWitnessSigs(raw: Uint8Array): Uint8Array[] {
  // 跳过版本(4) + marker/flag(2)
  let offset = 6;
  const [inCount, inUsed] = readVarint(raw, offset);
  offset += inUsed;
  for (let i = 0; i < inCount; i++) {
    offset += 32 + 4;
    const [sl, su] = readVarint(raw, offset);
    offset += su + sl + 4;
  }
  const [outCount, outUsed] = readVarint(raw, offset);
  offset += outUsed;
  for (let i = 0; i < outCount; i++) {
    offset += 8;
    const [sl, su] = readVarint(raw, offset);
    offset += su + sl;
  }
  const sigs: Uint8Array[] = [];
  for (let i = 0; i < inCount; i++) {
    const [itemCount, ic] = readVarint(raw, offset);
    offset += ic;
    assert.equal(itemCount, 1, 'key-path 花费见证应只有 1 项');
    const [itemLen, il] = readVarint(raw, offset);
    offset += il;
    sigs.push(raw.slice(offset, offset + itemLen));
    offset += itemLen;
  }
  return sigs;
}

function extractFirstWitnessSig(raw: Uint8Array): Uint8Array {
  return extractAllWitnessSigs(raw)[0];
}
