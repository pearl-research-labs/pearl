import { getCurrentNetwork, type Network } from '../config/network-config';

const BlockbookBaseUrlMap: Record<Network, string> = {
    // NOTE: plain-https only. The host 301-redirects http -> https, but the
    // wallet must not send its first request in the clear (fee-estimate
    // traffic is metadata about wallet activity; a MITM on the initial
    // plaintext hop could also serve a forged estimate before the redirect).
    testnet: 'https://blockbook.testnet.pearlresearch.ai',
    mainnet: 'https://blockbook.pearlresearch.ai',
};

function getBaseUrl(): string {
    return BlockbookBaseUrlMap[getCurrentNetwork()];
}

export const BlockbookClient = {
    async estimateFee(numBlocks: number) {
        const response = await fetch(`${getBaseUrl()}/api/v1/estimatefee/${numBlocks}`);
        const data = await response.json();
        return data.result;
    },
};
