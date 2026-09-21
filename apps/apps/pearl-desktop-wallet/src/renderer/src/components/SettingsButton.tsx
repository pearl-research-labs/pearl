import { useState } from 'react';
import { Settings } from 'lucide-react';
import { PeerSettingsModal } from './PeerSettingsModal';

export function SettingsButton() {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsModalOpen(true)}
        className="flex items-center gap-2 rounded-lg border border-gray-300 dark:border-border bg-white dark:bg-card px-3 py-2 text-sm text-gray-700 dark:text-foreground/90 transition-colors hover:bg-gray-50 dark:hover:bg-muted/60"
        title="Settings"
      >
        <Settings className="h-4 w-4" />
        <span>Settings</span>
      </button>

      <PeerSettingsModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </>
  );
}

