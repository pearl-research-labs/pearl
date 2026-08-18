import { useState, useEffect } from 'react';
import { X, RotateCcw, Save, Globe, Server } from 'lucide-react';
import { Button } from '@pearl/ui/components/button';

interface PeerSettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export function PeerSettingsModal({ isOpen, onClose }: PeerSettingsModalProps) {
    const [peerAddress, setPeerAddress] = useState('');
    const [peerPort, setPeerPort] = useState('');
    const [isCustom, setIsCustom] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [network, setNetwork] = useState('');

    useEffect(() => {
        if (isOpen) {
            loadPeerSettings();
        }
    }, [isOpen]);

    const loadPeerSettings = async () => {
        try {
            const settings = await window.appBridge.manager.getPeerSettings();
            setPeerAddress(settings.customPeerAddress ?? '');
            setPeerPort(settings.customPeerPort ? String(settings.customPeerPort) : '');
            setIsCustom(settings.isCustom);
            setNetwork(settings.network);
        } catch (error) {
            console.error('Failed to load peer settings:', error);
        }
    };

    const handleSave = async () => {
        const address = peerAddress.trim();
        const portStr = peerPort.trim();

        // Both fields empty → clear the custom peer and fall back to DNS seeders
        if (!address && !portStr) {
            setIsSaving(true);
            try {
                await window.appBridge.manager.resetPeerToDefault();
                await loadPeerSettings();
                alert('Custom peer cleared. Peers will be discovered automatically via DNS seeders. Please restart the wallet for changes to take effect.');
                onClose();
            } catch (error) {
                console.error('Failed to clear peer settings:', error);
                alert('Failed to clear settings');
            } finally {
                setIsSaving(false);
            }
            return;
        }

        // If one field is filled, the other must be too
        if (!address || !portStr) {
            alert('Please fill in both peer address and port, or leave both empty to use automatic discovery.');
            return;
        }

        const port = parseInt(portStr, 10);
        if (isNaN(port) || port < 1 || port > 65535) {
            alert('Invalid port number');
            return;
        }

        setIsSaving(true);
        try {
            await window.appBridge.manager.setCustomPeerAddress(address, port);
            await loadPeerSettings();
            alert('Peer settings saved! Please restart the wallet for changes to take effect.');
            onClose();
        } catch (error) {
            console.error('Failed to save peer settings:', error);
            alert('Failed to save settings');
        } finally {
            setIsSaving(false);
        }
    };

    const handleReset = async () => {
        setIsSaving(true);
        try {
            await window.appBridge.manager.resetPeerToDefault();
            await loadPeerSettings();
            alert('Reset to automatic discovery. Please restart the wallet for changes to take effect.');
        } catch (error) {
            console.error('Failed to reset peer settings:', error);
            alert('Failed to reset settings');
        } finally {
            setIsSaving(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
            <div className="w-full max-w-md rounded-lg border border-gray-300 bg-white p-6 shadow-xl">
                {/* Header */}
                <div className="mb-4 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <h2 className="text-xl font-bold text-gray-900">Peer Settings</h2>
                        {network && (
                            <span className={`rounded-full px-2 py-0.5 text-xs font-semibold capitalize ${network === 'mainnet' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
                                {network}
                            </span>
                        )}
                    </div>
                    <button
                        onClick={onClose}
                        className="rounded-full p-1 text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-700"
                    >
                        <X className="h-5 w-5" />
                    </button>
                </div>

                {/* Form */}
                <div className="space-y-4">
                    {/* Default Peer Discovery */}
                    <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
                        <div className="mb-1 flex items-center gap-2">
                            <Globe className="h-4 w-4 text-green-600" />
                            <span className="text-sm font-semibold text-gray-700">Default Peer Discovery</span>
                        </div>
                        <p className="text-xs text-gray-500">
                            By default, the wallet automatically discovers peers via Pearl DNS seeders.
                        </p>
                    </div>

                    {/* Custom Peer (optional) */}
                    <div>
                        <div className="mb-2 flex items-center gap-2">
                            <Server className="h-4 w-4 text-gray-500" />
                            <label className="text-sm font-medium text-gray-700">Custom Peer (optional)</label>
                        </div>
                        <p className="mb-2 text-xs text-gray-500">
                            Add a specific peer (IP address or hostname) to connect to at startup. Leave empty to use automatic discovery.
                        </p>
                        <div className="space-y-2">
                            <input
                                type="text"
                                value={peerAddress}
                                onChange={(e) => setPeerAddress(e.target.value)}
                                placeholder="e.g. 192.168.1.1 or my-node.example.com"
                                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-gray-900 focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
                            />
                            <input
                                type="number"
                                value={peerPort}
                                onChange={(e) => setPeerPort(e.target.value)}
                                placeholder="Port (e.g. 44108)"
                                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-gray-900 focus:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-500/20"
                            />
                        </div>
                    </div>

                    {isCustom && (
                        <div className="rounded-lg bg-blue-50 p-3">
                            <p className="text-sm text-blue-700">
                                A custom peer is currently configured. The wallet will connect to it in addition to DNS-discovered peers.
                            </p>
                        </div>
                    )}

                    {/* Actions */}
                    <div className="flex gap-3">
                        <Button
                            onClick={handleReset}
                            variant="outline"
                            disabled={isSaving || !isCustom}
                            className="flex-1"
                        >
                            <RotateCcw className="mr-2 h-4 w-4" />
                            Use Automatic Discovery
                        </Button>

                        <Button
                            onClick={handleSave}
                            disabled={isSaving}
                            className="flex-1 bg-green-600 hover:bg-green-700"
                        >
                            <Save className="mr-2 h-4 w-4" />
                            {isSaving ? 'Saving...' : 'Save'}
                        </Button>
                    </div>

                    <p className="text-xs text-gray-500">
                        Note: You'll need to restart the wallet after changing peer settings.
                    </p>
                </div>
            </div>
        </div>
    );
}
