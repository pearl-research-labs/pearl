import { getCurrentNetwork, type Network } from '../config/network-config';

const BlockbookBaseUrlMap: Record<Network, string> = {
    testnet: 'http://blockbook.testnet.pearlresearch.ai',
    mainnet: 'http://blockbook.pearlresearch.ai',
};

function getBaseUrl(): string {
    return BlockbookBaseUrlMap[getCurrentNetwork()];
}

export interface BlockbookTxInfo {
    txid: string;
    confirmations: number;
    blockhash: string;
    time: number;
}

function parseUnixTime(value: unknown): number {
    const seconds = typeof value === 'number' && Number.isFinite(value) ? value : 0;
    return seconds > 0 ? Math.trunc(seconds) * 1000 : Date.now();
}

export const BlockbookClient = {
    async estimateFee(numBlocks: number) {
        const response = await fetch(`${getBaseUrl()}/api/v1/estimatefee/${numBlocks}`);
        const data = await response.json();
        return data.result;
    },

    async getTransactionInfo(txid: string): Promise<BlockbookTxInfo | null> {
        const response = await fetch(`${getBaseUrl()}/api/v2/tx/${txid}`);
        if (response.status === 404) {
            return null;
        }
        if (!response.ok) {
            throw new Error(`blockbook http ${response.status}`);
        }

        const data = await response.json() as Record<string, unknown>;
        return {
            txid: String(data.txid ?? txid),
            confirmations: typeof data.confirmations === 'number' ? data.confirmations : 0,
            blockhash: String(data.blockHash ?? data.blockhash ?? ''),
            time: parseUnixTime(data.blockTime ?? data.blocktime ?? data.time),
        };
    },
};
