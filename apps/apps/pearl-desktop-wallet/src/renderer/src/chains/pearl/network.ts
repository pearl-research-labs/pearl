export type PearlNetwork = "mainnet";

export interface PearlNetworkParams {
  name: PearlNetwork;
  hrp: string;
  decimals: number;
  rpcUrl: string;
  rpcLabel: string;
  explorerUrl: string;
  magic: number;
}

export const PEARL_MAINNET: PearlNetworkParams = {
  name: "mainnet",
  hrp: "prl",
  decimals: 8,
  rpcUrl: "https://rpc.pearlbridge.xyz/",
  rpcLabel: "rpc.pearlbridge.xyz",
  explorerUrl: "https://explorer.pearlresearch.ai",
  magic: 0xd9b4bef9,
};

export const PEARL_RPC_POOL: readonly string[] = [
  "https://rpc.pearlbridge.xyz/",
  "https://pearl-sentry-fsn1-1.pearlbridge.xyz/rpc",
  "https://pearl-sentry-nbg1-1.pearlbridge.xyz/rpc",
  "https://pearl-sentry-hel1-1.pearlbridge.xyz/rpc",
];

export function pearlParams(_net: PearlNetwork = "mainnet"): PearlNetworkParams {
  return PEARL_MAINNET;
}

export function pearlTxExplorerUrl(network: PearlNetwork, txid: string): string {
  return `${PEARL_MAINNET.explorerUrl}/tx/${txid}?network=${network}`;
}
