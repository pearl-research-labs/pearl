export type PearlNetwork = "mainnet";

export interface PearlNetworkParams {
	name: PearlNetwork;
	hrp: string;
	decimals: number;
	rpcUrl: string;
	explorerUrl: string;
	magic: number;
}

export const PEARL_MAINNET: PearlNetworkParams = {
	name: "mainnet",
	hrp: "prl",
	decimals: 8,
	rpcUrl: "https://wallet.alphapool.tech/rpc/pearl",
	explorerUrl: "https://explorer.pearlresearch.ai",
	magic: 0xd9b4bef9,
};

export function pearlParams(_net: PearlNetwork = "mainnet"): PearlNetworkParams {
	return PEARL_MAINNET;
}

export function pearlTxExplorerUrl(network: PearlNetwork, txid: string): string {
	return `${PEARL_MAINNET.explorerUrl}/tx/${txid}?network=${network}`;
}
