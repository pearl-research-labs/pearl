/**
 * Pearl 移动钱包 —— 网络参数定义
 *
 * 数值来自仓库 node/chaincfg/params.go：
 *  - 主网 Bech32 HRP: prl / 测试网 tprl / 回归 rprl
 *  - BIP44 coin type: 主网 808276，测试网络 1
 *  - xpub/xprv 版本字节与比特币一致（0x0488b21e / 0x0488ade4）
 *  - 金额单位：1 PRL = 1e8 Grain（见 node/btcutil/amount.go）
 *
 * Pearl 仅启用 Taproot（witness v1）输出，地址形如 prl1p...
 */

export type NetworkName = 'mainnet' | 'testnet' | 'regtest';

export interface PearlNetwork {
  /** 网络标识 */
  name: NetworkName;
  /** bech32m 地址 HRP */
  hrp: string;
  /** BIP44/BIP86 coin type */
  coinType: number;
  /** xpub 版本字节 */
  xpubVersion: number;
  /** xprv 版本字节 */
  xprvVersion: number;
  /** WIF 前缀（仅兼容用，Pearl 不使用） */
  wifPrefix: number;
}

export const NETWORKS: Record<NetworkName, PearlNetwork> = {
  mainnet: {
    name: 'mainnet',
    hrp: 'prl',
    coinType: 808276,
    xpubVersion: 0x0488b21e,
    xprvVersion: 0x0488ade4,
    wifPrefix: 0x80,
  },
  testnet: {
    name: 'testnet',
    hrp: 'tprl',
    coinType: 1,
    xpubVersion: 0x0488b21e,
    xprvVersion: 0x0488ade4,
    wifPrefix: 0xef,
  },
  regtest: {
    name: 'regtest',
    hrp: 'rprl',
    coinType: 1,
    xpubVersion: 0x0488b21e,
    xprvVersion: 0x0488ade4,
    wifPrefix: 0xef,
  },
};

/** 每个最小单位（Grain）对应的 PRL 换算率：1 PRL = 1e8 Grain */
export const GRAINS_PER_PRL = 100_000_000;

/** 派生路径模板：m/86'/<coinType>'/0'/<chain>/<index> */
export const PURPOSE_BIP86 = 86;
export const ACCOUNT_INDEX = 0;
export const CHAIN_EXTERNAL = 0;
export const CHAIN_CHANGE = 1;

/** 交易构建参数 */
export const TX_VERSION = 2;
export const TX_LOCKTIME = 0;
/** P2TR 输出脚本长度：OP_1 OP_PUSHBYTES_32 <32 字节公钥> */
export const P2TR_OUTPUT_SIZE = 43;
/** P2TR 输入的近似序列化大小（含见证分摊） */
export const P2TR_INPUT_VBYTES = 58;
/** 交易基础开销（版本 + locktime + 输入输出计数） */
export const TX_OVERHEAD_VBYTES = 11;
/** 费率兜底（Grain/字节），节点 fee 估算不可用时使用 */
export const FALLBACK_FEE_RATE = 2;
/** 低于该值的找零直接并入矿工费（Grain） */
export const DUST_LIMIT = 546;

/** 密钥在系统安全存储（Keychain/Keystore）中的服务名前缀 */
export const SECURE_STORE_SERVICE = 'ai.pearlresearch.pearl-mobile';
