import { Button } from '@/components/ui/button';
import { ArrowLeft, Loader2 } from 'lucide-react';

type ActionButtonsProps = {
  onBack: () => void;
  isValidating: boolean;
  isCreating: boolean;
  isValid: boolean;
};

export default function ActionButtons({
  onBack,
  isValidating,
  isCreating,
  isValid,
}: ActionButtonsProps) {
  return (
    <div className="flex gap-3">
      <Button
        type="button"
        variant="ghost"
        onClick={onBack}
        className="flex-1 border border-gray-300 dark:border-border bg-white dark:bg-card text-gray-700 dark:text-foreground/90 shadow-sm hover:bg-gray-50 dark:hover:bg-muted/60 hover:text-gray-900 dark:hover:text-foreground"
        disabled={isCreating || isValidating}
      >
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back
      </Button>
      <Button
        type="submit"
        disabled={!isValid || isCreating || isValidating}
        className="flex-1 shadow-sm disabled:bg-gray-300 dark:disabled:bg-muted disabled:text-gray-500 dark:disabled:text-muted-foreground"
      >
        {isCreating ? 'Creating Wallet' : 'Create Wallet'}
        {isCreating && <Loader2 className="ml-2 h-4 w-4 animate-spin" />}
      </Button>
    </div>
  );
}
