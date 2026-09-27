import {clsx, type ClassValue} from 'clsx';
import {twMerge} from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getErrorMessage(
  error: unknown,
  fallbackMessage = 'An unexpected error occurred'
): string {
  if (!error) return fallbackMessage;

  // Extract base error message
  let errorMessage = error instanceof Error ? error.message : String(error);

  // Remove "Error invoking remote method 'method-name':" prefix from Electron IPC errors
  if (errorMessage.includes('Error invoking remote method')) {
    const match = errorMessage.match(/Error invoking remote method[^:]*:\s*(.+)/);
    if (match && match[1]) {
      errorMessage = match[1].trim();
    }
  }

  // Optionally remove leading "Error:" prefix for cleaner messages
  errorMessage = errorMessage.replace(/^Error:\s*/i, '');

  return errorMessage || fallbackMessage;
}

// The daemon reports this when it announced a transaction but no peer asked for it. On a first send nothing left this
// machine; on a rebroadcast it is ambiguous, because peers that already hold the transaction stay silent.
export function isNotRelayedError(message: string): boolean {
  return message.includes('not relayed');
}
