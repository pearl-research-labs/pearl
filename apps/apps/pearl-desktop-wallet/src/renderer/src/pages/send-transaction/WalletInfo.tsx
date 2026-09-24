import { formatPearlAmount } from '../../lib/crypto';
import KeyValueRow from './KeyValueRow';

type WalletInfoProps = {
  walletName?: string;
  balance?: number;
};

export default function WalletInfo({ walletName, balance }: WalletInfoProps) {
  return (
    <div className="rounded-xl border border-gray-300 dark:border-border bg-white dark:bg-card p-4 shadow-sm">
      <div className="space-y-1">
        <KeyValueRow label="From Wallet:" rightContent={walletName} />

        <KeyValueRow label="Balance:" rightContent={`${formatPearlAmount(balance ?? 0)} PRL`} />
      </div>
    </div>
  );
}
