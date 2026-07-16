/**
 * Central network configuration for the wallet
 * All network-specific settings should be consumed from here
 */

export type Network = 'mainnet' | 'testnet';

export interface NetworkConfig {
    name: Network;
    displayName: string;
    rpcPort: number;
    walletFlag: string;
    dataSubdir: string;
    addressPrefix: string;
}

const NETWORK_CONFIGS: Record<Network, NetworkConfig> = {
    mainnet: {
        name: 'mainnet',
        displayName: 'Mainnet',
        rpcPort: 8335,
        walletFlag: '',  // No flag for mainnet (default)
        dataSubdir: 'mainnet',
        addressPrefix: 'prl1',
    },
    testnet: {
        name: 'testnet',
        displayName: 'Testnet',
        rpcPort: 8335,
        walletFlag: '--testnet2',
        dataSubdir: 'testnet2',
        addressPrefix: 'tprl1',
    },
};

let currentNetwork: Network = 'mainnet';

export function getCurrentNetwork(): Network {
    return currentNetwork;
}

export function setCurrentNetwork(network: Network): void {
    currentNetwork = network;
}

export function getCurrentNetworkConfig(): NetworkConfig {
    return NETWORK_CONFIGS[currentNetwork];
}

export function getAllNetworks(): Network[] {
    return Object.keys(NETWORK_CONFIGS) as Network[];
}
