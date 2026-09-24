import {AlertCircle} from 'lucide-react';

type ErrorAlertProps = {
  message: string;
};

export default function ErrorAlert({message}: ErrorAlertProps) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/40 p-4 shadow-sm">
      <AlertCircle className="h-5 w-5 flex-shrink-0 text-red-600 dark:text-red-400" />
      <span className="text-red-700 dark:text-red-300">{message}</span>
    </div>
  );
}
