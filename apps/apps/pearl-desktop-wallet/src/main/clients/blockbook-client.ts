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

function describeError(err: unknown): string {
    if (err instanceof Error) {
        const parts = [err.name, err.message].filter(Boolean);
        const cause = (err as Error & { cause?: unknown }).cause;
        if (cause instanceof Error) {
            parts.push(`cause=${cause.name}: ${cause.message}`);
        } else if (cause && typeof cause === 'object') {
            const c = cause as Record<string, unknown>;
            const extra = [c.code, c.errno, c.syscall, c.address, c.port, c.path]
                .filter((value) => value !== undefined && value !== null && value !== '')
                .map(String);
            if (extra.length > 0) {
                parts.push(`cause=${extra.join(' ')}`);
            } else {
                parts.push(`cause=${JSON.stringify(cause)}`);
            }
        }
        return parts.join(': ');
    }
    return String(err);
}

export const BlockbookClient = {
    async estimateFee(numBlocks: number) {
        let response: Response;
        try {
            response = await fetch(`${getBaseUrl()}/api/v1/estimatefee/${numBlocks}`);
        } catch (err) {
            throw new Error(`blockbook fetch ${getBaseUrl()}/api/v1/estimatefee/${numBlocks} failed: ${describeError(err)}`);
        }
        const data = await response.json();
        return data.result;
    },

    async getTransactionInfo(txid: string): Promise<BlockbookTxInfo | null> {
        let response: Response;
        try {
            response = await fetch(`${getBaseUrl()}/api/v2/tx/${txid}`);
        } catch (err) {
            throw new Error(`blockbook fetch ${getBaseUrl()}/api/v2/tx/${txid} failed: ${describeError(err)}`);
        }
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
