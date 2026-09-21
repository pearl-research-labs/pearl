type TransactionPreviewProps = {
  amount: string;
  address: string;
  currentFee: number | string;
};

export default function TransactionPreview({ amount, address, currentFee }: TransactionPreviewProps) {
  return (
    <div className="space-y-2 rounded-lg border border-gray-300 dark:border-border bg-white dark:bg-card p-4 shadow-sm">
      <div className="text-sm text-neutral-400">Transaction Preview</div>
      <div className="flex justify-between">
        <span className="text-neutral-400">Amount:</span>
        <span className="font-medium text-gray-900 dark:text-foreground">{amount} PRL</span>
      </div>
      <div className="flex justify-between">
        <span className="text-neutral-400">To:</span>
        <span className="font-mono text-sm text-gray-900 dark:text-foreground">
          {address.slice(0, 8)}...{address.slice(-6)}
        </span>
      </div>
      <div className="flex justify-between">
        <span className="text-neutral-400">Transaction Fee / kb:</span>
        <span className="text-gray-900 dark:text-foreground">
          {typeof currentFee === 'number' ? currentFee : currentFee} PRL
        </span>
      </div>
    </div>
  );
}
