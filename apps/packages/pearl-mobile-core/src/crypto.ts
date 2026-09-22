/**
 * 密钥体系：BIP39 助记词 → BIP32 种子 → BIP86 路径派生 Taproot 密钥。
 *
 * Pearl 使用比特币原生 Taproot（secp256k1 + BIP340 Schnorr 签名），
 * 派生路径 m/86'/<coinType>'/0'/<chain>/<index>（见 constants.ts）。
 */

import {generateMnemonic, mnemonicToSeedSync, validateMnemonic} from '@scure/bip39';
import {wordlist} from '@scure/bip39/wordlists/english';
import {HDKey} from '@scure/bip32';
import {schnorr, secp256k1} from '@noble/curves/secp256k1';
import {NETWORKS, NetworkName, PURPOSE_BIP86, ACCOUNT_INDEX} from './constants';
import {bytesToHex, hexToBytes} from './utils';

const HARDENED = 0x80000000;

/** 生成 12/24 词助记词 */
export function createMnemonic(words: 12 | 24 = 12): string {
  return generateMnemonic(wordlist, words === 12 ? 128 : 256);
}

/** 校验助记词（词表 + 校验和） */
export function isValidMnemonic(mnemonic: string): boolean {
  return validateMnemonic(mnemonic.trim(), wordlist);
}

/** 助记词 → 64 字节 BIP32 种子 */
export function mnemonicToSeed(mnemonic: string, passphrase = ''): Uint8Array {
  if (!isValidMnemonic(mnemonic)) throw new Error('助记词无效');
  return mnemonicToSeedSync(mnemonic.trim(), passphrase);
}

/** 派生账户级 xpub（m/86'/coinType'/0'），用于备份展示 */
export function accountXpub(mnemonic: string, network: NetworkName): string {
  const root = HDKey.fromMasterSeed(mnemonicToSeed(mnemonic));
  const net = NETWORKS[network];
  const account = root
    .deriveChild(PURPOSE_BIP86 + HARDENED)
    .deriveChild(net.coinType + HARDENED)
    .deriveChild(ACCOUNT_INDEX + HARDENED);
  return account.publicExtendedKey;
}

export interface TaprootKeyPair {
  /** 内部私钥（32 字节，未 tweak） */
  privateKey: Uint8Array;
  /** tweak 后的输出公钥 x 坐标（32 字节，即地址中的公钥） */
  outputPubKeyX: Uint8Array;
  /** tweak 后的输出私钥（签名使用） */
  tweakedPrivateKey: Uint8Array;
  /** 内部公钥 x 坐标（32 字节） */
  internalPubKeyX: Uint8Array;
}

/**
 * 从助记词派生单个 Taproot 密钥对。
 * chain: 0=收款地址，1=找零地址
 */
export function deriveTaprootKey(
  mnemonic: string,
  network: NetworkName,
  index: number,
  chain: 0 | 1 = 0
): TaprootKeyPair {
  const net = NETWORKS[network];
  const root = HDKey.fromMasterSeed(mnemonicToSeed(mnemonic));
  const node = root
    .deriveChild(PURPOSE_BIP86 + HARDENED)
    .deriveChild(net.coinType + HARDENED)
    .deriveChild(ACCOUNT_INDEX + HARDENED)
    .deriveChild(chain)
    .deriveChild(index);

  if (!node.privateKey || !node.publicKey) {
    throw new Error(`派生失败: index=${index} chain=${chain}`);
  }
  return tweakKey(node.privateKey);
}

/** 对原始私钥做 BIP341 key-path tweak，得到可直接签名的密钥对 */
export function tweakKey(privateKey: Uint8Array): TaprootKeyPair {
  // 内部公钥（压缩 33 字节）→ x 坐标；首字节 0x03 表示 y 为奇数
  const internalCompressed = secp256k1.getPublicKey(privateKey, true);
  const internalX = internalCompressed.slice(1);

  // BIP341：Q = P_even + H_tapTweak(P.x)*G，私钥 d' = d_even + tweak mod n
  const n = secp256k1.CURVE.n;
  let d = bytesToBigint(privateKey);
  if (internalCompressed[0] === 0x03) d = n - d; // y 为奇 → 取反（BIP340 even-y 约定）
  const tapTweak = schnorr.utils.taggedHash('TapTweak', internalX);
  const t = bytesToBigint(tapTweak);
  if (t >= n) throw new Error('TapTweak 标量越界');
  const tweakedScalar = (d + t) % n;
  const tweaked = bigintToBytes32(tweakedScalar);
  const outputCompressed = secp256k1.getPublicKey(tweaked, true);

  return {
    privateKey,
    tweakedPrivateKey: tweaked,
    internalPubKeyX: internalX,
    outputPubKeyX: outputCompressed.slice(1),
  };
}

function bytesToBigint(b: Uint8Array): bigint {
  return BigInt('0x' + bytesToHex(b));
}

function bigintToBytes32(x: bigint): Uint8Array {
  return hexToBytes(x.toString(16).padStart(64, '0'));
}

/**
 * 用 tweak 后私钥生成 BIP340 Schnorr 签名。
 * auxRand 默认全零（与 btcd schnorr.Sign 默认行为一致），
 * 保证签名确定可复现，且与节点端实现字节级对齐。
 */
export function schnorrSign(
  msgHash: Uint8Array,
  tweakedPrivateKey: Uint8Array,
  auxRand?: Uint8Array
): Uint8Array {
  return schnorr.sign(msgHash, tweakedPrivateKey, auxRand ?? new Uint8Array(32));
}

/** 校验签名（测试用） */
export function schnorrVerify(
  sig: Uint8Array,
  msgHash: Uint8Array,
  outputPubKeyX: Uint8Array
): boolean {
  try {
    return schnorr.verify(sig, msgHash, outputPubKeyX);
  } catch {
    return false;
  }
}
