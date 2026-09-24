import { Button } from '@pearl/ui';
import { ArrowUpRight } from 'lucide-react';

type SendButtonProps = {
  disabled: boolean;
  isLoading: boolean;
  onClick: () => void;
};

export default function SendButton({ disabled, isLoading, onClick }: SendButtonProps) {
  return (
    <Button
      onClick={onClick}
      disabled={disabled}
      className="flex h-12 w-full items-center justify-center gap-2 rounded-xl py-3 font-semibold disabled:bg-gray-300 dark:disabled:bg-muted disabled:text-gray-500 dark:disabled:text-muted-foreground"
    >
      {isLoading ? (
        <>
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
          Sending...
        </>
      ) : (
        <>
          <ArrowUpRight className="h-5 w-5" />
          Send Transaction
        </>
      )}
    </Button>
  );
}
