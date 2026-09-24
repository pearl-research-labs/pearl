import { useState, useEffect } from 'react';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@pearl/ui/components/select';

interface NetworkInfo {
    currentNetwork: string;
    availableNetworks: string[];
    networkConfig: {
        name: string;
        displayName: string;
        addressPrefix: string;
    };
}

export function NetworkSelector() {
    const [networkInfo, setNetworkInfo] = useState<NetworkInfo | null>(null);
    const [isChanging, setIsChanging] = useState(false);

    useEffect(() => {
        loadNetworkInfo();
    }, []);

    const loadNetworkInfo = async () => {
        try {
            const info = await window.appBridge.manager.getNetworkInfo();
            setNetworkInfo(info);
        } catch (error) {
            console.error('Failed to load network info:', error);
        }
    };

    const handleNetworkChange = async (network: string) => {
        if (isChanging) return;

        setIsChanging(true);
        try {
            // Switch network
            await window.appBridge.manager.setNetwork(network);

            window.location.hash = '#/';
            window.location.reload();
        } catch (error) {
            console.error('Failed to change network:', error);
            setIsChanging(false);
        }
    };

    if (!networkInfo) {
        return null;
    }

    return (
        <div className="flex items-center gap-3">
            <span className="text-sm text-gray-600 dark:text-muted-foreground">Network:</span>
            <Select
                value={networkInfo.currentNetwork}
                onValueChange={handleNetworkChange}
                disabled={isChanging}
            >
                <SelectTrigger className="w-[140px]">
                    <SelectValue />
                </SelectTrigger>
                <SelectContent>
                    {networkInfo.availableNetworks.map((network) => (
                        <SelectItem key={network} value={network}>
                            {network.charAt(0).toUpperCase() + network.slice(1)}
                        </SelectItem>
                    ))}
                </SelectContent>
            </Select>
        </div>
    );
}

